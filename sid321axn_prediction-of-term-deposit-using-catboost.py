import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from pandas import plotting

%matplotlib inline

from time import time

from IPython.display import display # Allows the use of display() for DataFrames

from catboost import CatBoostClassifier

from sklearn.model_selection import cross_val_score

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/bank.csv')
# Now lest see the first 5 samples to get the overview of the dataset 

df.head()
# Now lets see the structure of the data

df.info()
# Lets see the overview of the dataset means average, std, min , max of the data

df.describe(include='all')
# Lets see only categorical variables

df.describe(include='object')
# Checking Missing values or null entries in the dataset

df.isna().sum()
print(df.dtypes)
df.shape
sns.countplot(x='deposit',data=df)
sns.countplot(x='deposit',hue='housing',data=df)
sns.countplot(x='deposit',hue='loan',data=df)
# making boolean series for term deposit subscribed customers of bank

filter1 = df["deposit"]=="yes"

    

# filtering data on basis of both filters 

df_subscribed = df.where(filter1).dropna()



df_subscribed.head()
sns.countplot(x='deposit', hue='education',data=df_subscribed)
sns.countplot(x='deposit',hue='marital',data=df_subscribed)
dataset2=df[['age','balance','duration','campaign','pdays']]



sns.pairplot(dataset2)

plt.show()
plt.rcParams['figure.figsize'] = (20, 8)

sns.countplot(df['age'], palette = 'hsv')

plt.title('Distribution of Age', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (25, 8)

sns.countplot(df_subscribed['age'], palette = 'rainbow')

plt.title('Distribution of Age of Subscribed Customers', fontsize = 25)

plt.show()


sns.distplot(df['balance'], hist=True,kde_kws={"color": "k", "lw": 3, "label": "KDE"}, kde=True,bins=50,hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "g"})

plt.title('Distribution of Balance in Account', fontsize = 20)

plt.show()
sns.distplot(df_subscribed['balance'], hist=True,kde_kws={"color": "k", "lw": 3, "label": "KDE"}, kde=True,bins=50,hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "g"})

plt.title('Distribution of Balance of already subscribed account', fontsize = 20)

plt.show()


sns.distplot(df['duration'], hist=True,kde_kws={"color": "k", "lw": 3, "label": "KDE"}, kde=True,bins=50,hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "g"})

plt.title('Distribution of Duration', fontsize = 20)

plt.show()
sns.distplot(df_subscribed['duration'], hist=True,kde_kws={"color": "k", "lw": 3, "label": "KDE"}, kde=True,bins=50,hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "g"})

plt.title('Distribution of Duration of already subscribed account', fontsize = 20)

plt.show()
labels = ['Normal', 'Default']

size = df['default'].value_counts()

colors = ['lightgreen', 'orange']

explode = [0, 0.1]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')

plt.title('Default Loans Status', fontsize = 20)

plt.axis('off')

plt.legend()

plt.show()
labels = ['No Housing Loan','Housing loan taken' ]

size = df['housing'].value_counts()

colors = ['blue', 'yellow']

explode = [0, 0.1]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')

plt.title('Status of Housing Loan', fontsize = 20)

plt.axis('off')

plt.legend()

plt.show()
labels = ['No Loan Taken','Has Taken Loan']

size = df['loan'].value_counts()

colors = ['green', 'blue']

explode = [0, 0.1]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')

plt.title('Status of Loan customer', fontsize = 20)

plt.axis('off')

plt.legend()

plt.show()
labels = ['No deposit','Deposit in Bank']

size = df['deposit'].value_counts()

colors = ['blue', 'orange']

explode = [0, 0.1]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')

plt.title('Status of Deposit customer', fontsize = 20)

plt.axis('off')

plt.legend()

plt.show()
plt.rcParams['figure.figsize'] = (18, 7)

sns.boxenplot(df['housing'], df['balance'],hue=df['deposit'], palette = 'Blues')

plt.title('Hosuing vs Balance vs Deposit', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (18, 7)

sns.boxenplot(df['loan'], df['balance'],hue=df['deposit'], palette = 'rainbow')

plt.title('loan vs Balance vs Deposit', fontsize = 20)

plt.show()

plt.rcParams['figure.figsize'] = (18, 7)

sns.boxenplot(df['housing'], df['balance'],hue=df['default'])

plt.title('Hosuing vs Balance vs Default', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (18, 7)

sns.boxenplot(df['loan'], df['balance'],hue=df['default'],palette="Set1")

plt.title('Personal Loan vs Balance vs Default', fontsize = 20)

plt.show()
# Sort the dataframe by target

deposit_yes = df.loc[df['deposit'] == 'yes']

deposit_no = df.loc[df['deposit'] == 'no']

fig = plt.figure(figsize=(20,8))

sns.distplot(deposit_yes[['duration']], hist=False, rug=True)

sns.distplot(deposit_no[['duration']], hist=False, rug=True)

plt.title('Duration of Deposit vs Non deposit', fontsize = 20)

fig.legend(labels=['Deposit','Non deposit'])

plt.show()

sns.countplot(df['poutcome'])
import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (18, 8)



plt.subplot(1, 2, 1)

sns.set(style = 'whitegrid')

sns.distplot(df['previous'])

plt.title('Distribution of Previous', fontsize = 20)

plt.xlabel('Range of Previous')

plt.ylabel('Count')





plt.subplot(1, 2, 2)

sns.set(style = 'whitegrid')

sns.distplot(df['campaign'], color = 'red')

plt.title('Distribution of Campaign', fontsize = 20)

plt.xlabel('Range of Campaign')

plt.ylabel('Count')

plt.show()
df["deposit"] = df.deposit.apply(lambda  x:1 if x=="yes" else 0)

df["loan"] = df.loan.apply(lambda  x:1 if x=="yes" else 0)

df["housing"] = df.housing.apply(lambda  x:1 if x=="yes" else 0)

df["default"] = df.default.apply(lambda  x:1 if x=="yes" else 0)
df1=df.drop(['deposit'],axis=1)



plt.figure(figsize=(20,10)) 

sns.heatmap(df1.corr(), annot=True) 
df1.corrwith(df.deposit).plot.bar(

        figsize = (20, 10), title = "Correlation with Deposit", fontsize = 20,

        rot = 45, grid = True)
X = df.drop(['deposit'],axis=1) # Feature 



y=df['deposit'] # Target variable
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size = 0.20, random_state=0)
# convert categorical columns to integers

category_cols = ['job','marital','education','contact','month','poutcome']

for header in category_cols:

    X_train[header] = X_train[header].astype('category').cat.codes

    X_test[header] = X_test[header].astype('category').cat.codes
print(X_train.dtypes)
categorical_features_indices = np.where(X.dtypes != np.int64)[0]
model = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42)
model.fit(X_train,y_train,cat_features=categorical_features_indices,eval_set=(X_test,y_test))
y_predict = model.predict(X_test)

from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

roc=roc_auc_score(y_test, y_predict)

acc = accuracy_score(y_test, y_predict)

prec = precision_score(y_test, y_predict)

rec = recall_score(y_test, y_predict)

f1 = f1_score(y_test, y_predict)



results = pd.DataFrame([['CatBoost', acc,prec,rec, f1,roc]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])

results
from catboost import cv,Pool

cv_data = cv(Pool(X,y,cat_features=categorical_features_indices),model.get_params(),fold_count=10)

print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(

    np.max(cv_data['test-Accuracy-mean']), 

    cv_data['test-Accuracy-std'][cv_data['test-Accuracy-mean'].idxmax(axis=0)],

    cv_data['test-Accuracy-mean'].idxmax(axis=0)

))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm, annot=True)
from sklearn import metrics

plt.figure()



# Add the models to the list that you want to view on the ROC plot

models = [

    {

    'label': 'CATBOOST',

    'model': CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42),        

    }

]



# Below for loop iterates through your models list

for m in models:

    model = m['model'] # select the model

    model.fit(X_train,y_train,cat_features=categorical_features_indices,eval_set=(X_test,y_test)) # train the model

    y_pred=model.predict(X_test) # predict the test data

# Compute False postive rate, and True positive rate

    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:,1])

# Calculate Area under the curve to display on the plot

    auc = metrics.roc_auc_score(y_test,model.predict(X_test))

# Now, plot the computed values

    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))

# Custom settings for the plot 

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('1-Specificity(False Positive Rate)')

plt.ylabel('Sensitivity(True Positive Rate)')

plt.title('Receiver Operating Characteristic')

plt.legend(loc="lower right")

plt.show()
features ='duration'

res =model.get_feature_statistics(X_train, y_train,features, plot=True)
features ='balance'

res =model.get_feature_statistics(X_train, y_train,features, plot=True)
import shap

shap_values = model.get_feature_importance(Pool(X_test, label=y_test,cat_features=categorical_features_indices), 

                                                                     type="ShapValues")

expected_value = shap_values[0,-1]

shap_values = shap_values[:,:-1]



shap.initjs()

shap.force_plot(expected_value, shap_values[3,:], X_test.iloc[3,:])
feature_score = pd.DataFrame(list(zip(X.dtypes.index, model.get_feature_importance(Pool(X, label=y, cat_features=categorical_features_indices)))),

                columns=['Feature','Score'])



feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')
plt.rcParams["figure.figsize"] = (12,7)

ax = feature_score.plot('Feature', 'Score', kind='bar', color='b')

ax.set_title("Catboost Feature Importance Ranking", fontsize = 14)

ax.set_xlabel('')



rects = ax.patches



labels = feature_score['Score'].round(2)



for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 0.35, label, ha='center', va='bottom')



plt.show()