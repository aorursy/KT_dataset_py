from pandas.plotting import scatter_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from xgboost import XGBClassifier

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, MinMaxScaler, StandardScaler, RobustScaler
import seaborn as sns

#got on my nerves
import warnings
warnings.filterwarnings('ignore')
#Get Data

#HAD TO REWRITE A LOT of the CODE for this special .csv!
#The adult.data.txt from the UCI website is a bit different
url = '../input/adult.csv'
data = pd.read_csv(url)
print(set(data.education), '\n\n',
      set(data.occupation),'\n\n', 
      set(data.workclass))
data.head()
#Understand data with descriptive statistics
print(data.nunique(), data.education.value_counts(), data.dtypes, data.describe(), data.corr(), data.shape, data.isnull().values.any())

#correlation heatmap of dataset

#took this code sample from another kernel on this site
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = "YlGn",
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(data)

sns.set(style="ticks")
sns.pairplot(data, hue="income")
plt.show()
#sns.distplot(data.education.value_counts())
#sns.jointplot(x="education_num", y="age", data=data, kind="kde");
to_count = ['education','workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender']

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20,14))
#fig.tight_layout()
[sns.countplot(y=feature, hue='income', data=data, order=data[feature].value_counts().index , ax=axs.flatten()[idx]) for idx, feature in enumerate(to_count)]
#[axs.flatten()[idx].set_title(feature) for idx, feature in enumerate(to_count)]
plt.plot()
sns.distplot(data[data['income'] == '>50K']['age'], kde_kws={"label": ">$50K"})
sns.distplot(data[data['income'] == '<=50K']['age'], kde_kws={"label": "<=$50K"})
#data['income'] == ' >50K'
#code sample borrowed from other kernel on this challenge
g = sns.jointplot(x = 'age', 
              y = 'hours-per-week',
              data = data, 
              kind = 'hex', 
              cmap= 'hot',
              gridsize=50,
              size=12)

#http://stackoverflow.com/questions/33288830/how-to-plot-regression-line-on-hexbins-with-seaborn
sns.regplot(data.age, data['hours-per-week'], ax=g.ax_joint, scatter=False, color='grey')
plt.show()
#I've a feeling that education and education_num might be the same thing. Let's check
data[['education', 'educational-num']].groupby(['education'], as_index=False).mean().sort_values(by='educational-num', ascending=False)
df = data.copy()
#fixes a type bug and let's us use .strip
df['income'] = df['income'].astype('str')

df.income = df.income.apply(lambda el: el.strip() == ">50K")
y = df.income
df.drop(columns=['income','fnlwgt', 'education'], inplace=True)

#data without income
X = df.copy()
print(X.shape, y.shape)

validation_size = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
#X, y
X_train.head(), len(X_train), len(X_test)
#Label Encode ==> eg Binary, but in same column
LE_train = LabelEncoder()
LE_train.fit(X_train['gender'])

X_train.loc[:, 'gender'] = LE_train.transform(X_train.loc[:, 'gender'])
X_test.loc[:, 'gender'] = LE_train.fit_transform(X_test.loc[:, 'gender'])

print(LE_train.classes_)

##Short form without class inspection
#df['sex'] = LabelEncoder().fit_transform(df['sex'])
#This requires exact spelling + knowledge of "Male"
## or data.sex = data.sex.apply(lambda el: 1 if (el.strip() == "Male") else 0)
#let's make it simple. I'd say having citizenship from the start is worth a lot
X_train['native-country'] = X_train['native-country'].apply(lambda el: 1 if el.strip() == "United-States" else 0)
X_test['native-country'] = X_test['native-country'].apply(lambda el: 1 if el.strip() == "United-States" else 0)

X_train.isnull().values.any(), X_test.isnull().values.any(), len(X_train), X_train['marital-status'].isnull().values.sum()
cols_to_binarize = ['marital-status', 'occupation', 'relationship', 'race', 'workclass']

stored_binarizers = []
for col in cols_to_binarize:
  lb = LabelBinarizer()
  lb_fitted = lb.fit(X_train[col])
  stored_binarizers.append(lb_fitted)

#our binarizer class only knows what's in the training data
for b in stored_binarizers:
  print(b.classes_)

for i, val in enumerate(cols_to_binarize):
  print(X_train[val].head())
print(len(X_train))

def replaceWithBinarized_legit(dataframe, column_names, stored_binarizers):
  newDf = dataframe.copy()
  for idx, column_name in enumerate(column_names):
    if (not column_name in newDf):
      return
    
    lb = stored_binarizers[idx]
    lb_results = lb.transform(newDf[column_name])
    binarized_cols = pd.DataFrame(lb_results, columns=lb.classes_)

    newDf.drop(columns=column_name, inplace=True)
    #THIS BS indexing fucked up merge cost 2 hours of my life. Thanks Obama
    binarized_cols.index = newDf.index
    #
    newDf = pd.concat([newDf, binarized_cols], axis=1)
  return newDf

X_train = replaceWithBinarized_legit(X_train, cols_to_binarize, stored_binarizers)
X_test = replaceWithBinarized_legit(X_test, cols_to_binarize, stored_binarizers)

X_train.head()
#X_train = X_train.dropna()
X_train.isnull().values.any(), X_test.isnull().values.any(), len(X_train), X_test.isnull().values.any()
#let scaler only know about training set
cols_to_scale_standard = ['hours-per-week', 'age', 'educational-num']
standardSc = StandardScaler()
standardSc.fit(X_train[cols_to_scale_standard])

cols_to_scale_robust = ['capital-gain', 'capital-loss']
robustSc = RobustScaler()
robustSc.fit(X_train[cols_to_scale_robust])

def scale_columns_legit(df, column_names, scaler):
  d = df.copy()
  for column_name in column_names:
    if (not column_name in d.columns):
      return
  
  scaled_array = scaler.transform(d[column_names])
  scaled_df = pd.DataFrame(scaled_array, columns=column_names)
  scaled_df.index = d.index
  d.drop(columns=column_names, inplace=True)
  return pd.concat([d, scaled_df], axis=1)


#I think there are outliers in sight (capital is always inequal)
#do log for extreme values

X_train = scale_columns_legit(X_train, cols_to_scale_standard, standardSc)
X_train = scale_columns_legit(X_train, cols_to_scale_robust, robustSc)

X_test = scale_columns_legit(X_test, cols_to_scale_standard, standardSc)
X_test = scale_columns_legit(X_test, cols_to_scale_robust, robustSc)

X_train.head()
len(X_train.columns), len(X_train._get_numeric_data().columns), "Are all columns numeric?"

X_train.columns
#this might be a candidate for imputation ?
X_train = X_train.drop(columns=["?"], axis=1)
X_test = X_test.drop(columns=["?"], axis=1)
X_train.isnull().values.any(), X_test.isnull().values.any(), X_train.head()
X_train.head()
scoring = 'accuracy'
models = []

models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('XGB', XGBClassifier()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('SGD', SGDClassifier()))
models.append(('RF', RandomForestClassifier(n_jobs=-1)))


#SVM CRASHES SYS
#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
  kfold = model_selection.KFold(n_splits=10, random_state=seed)
  cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)
print(X_train.shape, X_test.shape)
gb = GradientBoostingClassifier()
gb.fit(X_train, Y_train)
predictions = gb.predict(X_test)
#print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
