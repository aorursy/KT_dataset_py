# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



sns.set(style="whitegrid")
import warnings



warnings.filterwarnings('ignore')
data = '/kaggle/input/income-classification/income_evaluation.csv'



df = pd.read_csv(data)
# print the shape

print('The shape of the dataset : ', df.shape)
df.head()
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',

             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']



df.columns = col_names



df.columns
df.info()
df.dtypes
df.describe()
df.describe().T
df.describe(include='all')
# check for missing values



df.isnull().sum()
#assert that there are no missing values in the dataframe



assert pd.notnull(df).all().all()
def initial_eda(df):

    if isinstance(df, pd.DataFrame):

        total_na = df.isna().sum().sum()

        print("Dimensions : %d rows, %d columns" % (df.shape[0], df.shape[1]))

        print("Total NA Values : %d " % (total_na))

        print("%38s %10s     %10s %10s" % ("Column Name", "Data Type", "#Distinct", "NA Values"))

        col_name = df.columns

        dtyp = df.dtypes

        uniq = df.nunique()

        na_val = df.isna().sum()

        for i in range(len(df.columns)):

            print("%38s %10s   %10s %10s" % (col_name[i], dtyp[i], uniq[i], na_val[i]))

        

    else:

        print("Expect a DataFrame but got a %15s" % (type(df)))

initial_eda(df)
categorical = [var for var in df.columns if df[var].dtype=='O']



print('There are {} categorical variables\n'.format(len(categorical)))



print('The categorical variables are :\n\n', categorical)
df[categorical].head()
for var in categorical: 

    

    print(df[var].value_counts())
for var in categorical:

    

     print(df[var].value_counts()/np.float(len(df)))
# check for missing values



df['income'].isnull().sum()
# view number of unique values



df['income'].nunique()
# view the unique values



df['income'].unique()
# view the frequency distribution of values



df['income'].value_counts()
# view percentage of frequency distribution of values



df['income'].value_counts()/len(df)
# visualize frequency distribution of income variable



f,ax=plt.subplots(1,2,figsize=(18,8))



ax[0] = df['income'].value_counts().plot.pie(explode=[0,0],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Income Share')





#f, ax = plt.subplots(figsize=(6, 8))

ax[1] = sns.countplot(x="income", data=df, palette="Set1")

ax[1].set_title("Frequency distribution of income variable")



plt.show()
f, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(y="income", data=df, palette="Set1")

ax.set_title("Frequency distribution of income variable")

plt.show()
f, ax = plt.subplots(figsize=(10, 8))

ax = sns.countplot(x="income", hue="sex", data=df, palette="Set1")

ax.set_title("Frequency distribution of income variable wrt sex")

plt.show()
f, ax = plt.subplots(figsize=(10, 8))

ax = sns.countplot(x="income", hue="race", data=df, palette="Set1")

ax.set_title("Frequency distribution of income variable wrt race")

plt.show()
# check number of unique labels 



df.workclass.nunique()
# view the unique labels



df.workclass.unique()
# view frequency distribution of values



df.workclass.value_counts()
# replace '?' values in workclass variable with `NaN`



df['workclass'].replace(' ?', np.NaN, inplace=True)
# again check the frequency distribution of values in workclass variable



df.workclass.value_counts()
f, ax = plt.subplots(figsize=(10, 6))

ax = df.workclass.value_counts().plot(kind="bar", color="green")

ax.set_title("Frequency distribution of workclass variable")

ax.set_xticklabels(df.workclass.value_counts().index, rotation=30)

plt.show()
f, ax = plt.subplots(figsize=(12, 8))

ax = sns.countplot(x="workclass", hue="income", data=df, palette="Set1")

ax.set_title("Frequency distribution of workclass variable wrt income")

ax.legend(loc='upper right')

plt.show()
f, ax = plt.subplots(figsize=(12, 8))

ax = sns.countplot(x="workclass", hue="sex", data=df, palette="Set1")

ax.set_title("Frequency distribution of workclass variable wrt sex")

ax.legend(loc='upper right')

plt.show()
# check number of unique labels



df.occupation.nunique()
# view unique labels



df.occupation.unique()

# view frequency distribution of values



df.occupation.value_counts()
# replace '?' values in occupation variable with `NaN`



df['occupation'].replace(' ?', np.NaN, inplace=True)

# again check the frequency distribution of values



df.occupation.value_counts()
# visualize frequency distribution of `occupation` variable



f, ax = plt.subplots(figsize=(12, 8))

ax = sns.countplot(x="occupation", data=df, palette="Set1")

ax.set_title("Frequency distribution of occupation variable")

ax.set_xticklabels(df.occupation.value_counts().index, rotation=30)

plt.show()
# check number of unique labels



df.native_country.nunique()
# view unique labels 



df.native_country.unique()

# check frequency distribution of values



df.native_country.value_counts()

# replace '?' values in native_country variable with `NaN`



df['native_country'].replace(' ?', np.NaN, inplace=True)
# again check the frequency distribution of values



df.native_country.value_counts()
# visualize frequency distribution of `native_country` variable



f, ax = plt.subplots(figsize=(16, 12))

ax = sns.countplot(x="native_country", data=df, palette="Set1")

ax.set_title("Frequency distribution of native_country variable")

ax.set_xticklabels(df.native_country.value_counts().index, rotation=90)

plt.show()
df[categorical].isnull().sum()
# check for cardinality in categorical variables



for var in categorical:

    

    print(var, ' contains ', len(df[var].unique()), ' labels')
numerical = [var for var in df.columns if df[var].dtype!='O']



print('There are {} numerical variables\n'.format(len(numerical)))



print('The numerical variables are :\n\n', numerical)
df[numerical].head()
df[numerical].isnull().sum()
df['age'].nunique()
f, ax = plt.subplots(figsize=(10,8))

x = df['age']

ax = sns.distplot(x, bins=10, color='blue')

ax.set_title("Distribution of age variable")

plt.show()
f, ax = plt.subplots(figsize=(10,8))

x = df['age']

x = pd.Series(x, name="Age variable")

ax = sns.distplot(x, bins=10, color='blue')

ax.set_title("Distribution of age variable")

plt.show()
f, ax = plt.subplots(figsize=(10,8))

x = df['age']

x = pd.Series(x, name="Age variable")

ax = sns.kdeplot(x, shade=True, color='red')

ax.set_title("Distribution of age variable")

plt.show()
f, ax = plt.subplots(figsize=(10,8))

x = df['age']

ax = sns.boxplot(x)

ax.set_title("Visualize outliers in age variable")

plt.show()
f, ax = plt.subplots(figsize=(10, 8))

ax = sns.boxplot(x="income", y="age", data=df)

ax.set_title("Visualize income wrt age variable")

plt.show()
f, ax = plt.subplots(figsize=(10, 8))

ax = sns.boxplot(x="income", y="age", hue="sex", data=df)

ax.set_title("Visualize income wrt age and sex variable")

ax.legend(loc='upper right')

plt.show()
plt.figure(figsize=(8,6))

ax = sns.catplot(x="income", y="age", col="sex", data=df, kind="box", height=8, aspect=1)

plt.show()
plt.figure(figsize=(12,8))

sns.boxplot(x ='race', y="age", data = df)

plt.title("Visualize age wrt race")

plt.show()
# plot correlation heatmap to find out correlations



df.corr().style.format("{:.4}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
sns.pairplot(df)

plt.show()
sns.pairplot(df, hue="income")

plt.show()
sns.pairplot(df, hue="sex")

plt.show()
X = df.drop(['income'], axis=1)



y = df['income']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# check the shape of X_train and X_test



X_train.shape, X_test.shape
categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']



categorical
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']



numerical
# print percentage of missing values in the categorical variables in training set



X_train[categorical].isnull().mean()
# print categorical variables with missing data



for col in categorical:

    if X_train[col].isnull().mean()>0:

        print(col, (X_train[col].isnull().mean()))
# impute missing categorical variables with most frequent value



for df2 in [X_train, X_test]:

    df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)

    df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)

    df2['native_country'].fillna(X_train['native_country'].mode()[0], inplace=True)    
# check missing values in categorical variables in X_train



X_train[categorical].isnull().sum()
# check missing values in categorical variables in X_test



X_test[categorical].isnull().sum()
# check missing values in X_train



X_train.isnull().sum()
# check missing values in X_test



X_test.isnull().sum()
# preview categorical variables in X_train



X_train[categorical].head()
# import category encoders



import category_encoders as ce
# encode categorical variables with one-hot encoding



encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship', 

                                 'race', 'sex', 'native_country'])



X_train = encoder.fit_transform(X_train)



X_test = encoder.transform(X_test)
X_train.head()
X_train.shape
X_test.head()
X_test.shape
cols = X_train.columns

from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()



X_train = scaler.fit_transform(X_train)



X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
# import Random Forest classifier



from sklearn.ensemble import RandomForestClassifier







# instantiate the classifier 



rfc = RandomForestClassifier(random_state=0)







# fit the model



rfc.fit(X_train, y_train)







# Predict the Test set results



y_pred = rfc.predict(X_test)







# Check accuracy score 



from sklearn.metrics import accuracy_score



print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# instantiate the classifier with n_estimators = 100



rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)







# fit the model to the training set



rfc_100.fit(X_train, y_train)







# Predict on the test set results



y_pred_100 = rfc_100.predict(X_test)







# Check accuracy score 



print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred_100)))
# create the classifier with n_estimators = 100



clf = RandomForestClassifier(n_estimators=100, random_state=0)







# fit the model to the training set



clf.fit(X_train, y_train)

# view the feature scores



feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)



feature_scores
# Creating a seaborn bar plot



f, ax = plt.subplots(figsize=(30, 24))

ax = sns.barplot(x=feature_scores, y=feature_scores.index, data=df)

ax.set_title("Visualize feature scores of the features")

ax.set_yticklabels(feature_scores.index)

ax.set_xlabel("Feature importance score")

ax.set_ylabel("Features")

plt.show()

# drop the least important feature from X_train and X_test



X_train = X_train.drop(['native_country_41'], axis=1)



X_test = X_test.drop(['native_country_41'], axis=1)

# instantiate the classifier with n_estimators = 100



clf = RandomForestClassifier(n_estimators=100, random_state=0)







# fit the model to the training set



clf.fit(X_train, y_train)





# Predict on the test set results



y_pred = clf.predict(X_test)







# Check accuracy score 



print('Model accuracy score with native_country_41 variable removed : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# Print the Confusion Matrix and slice it into four pieces



from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred)



print('Confusion matrix\n\n', cm)



# visualize confusion matrix with seaborn heatmap



cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 

                                 index=['Predict Positive:1', 'Predict Negative:0'])



sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
from sklearn.metrics import classification_report



print(classification_report(y_test, y_pred))