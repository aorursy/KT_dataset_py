# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



df = pd.read_csv('../input/income-classification/income_evaluation.csv')

df.head()
df.shape
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',

             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']



df.columns = col_names



df.columns
df.info()
df.dtypes
df.describe()
df.describe(include='object')
df.describe(include='all')
df.isnull().sum().sort_values(ascending=False)
# Categorical Variables:

categorical_features = [col for col in df.columns

                       if df[col].dtypes =='object']

categorical_features
df[categorical_features].nunique()
df[categorical_features].head()
for col in categorical_features:

    print(df[col].value_counts())
for col in categorical_features:

    print(df[col].value_counts()/len(df))
df['income'].value_counts()
df['income'].value_counts()/len(df)
# Lets Visualize frequency distribution of income variable:

f,ax = plt.subplots(figsize=(15,5))

df['income'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True)
sns.countplot(y='income',data=df)
# Vizualize income Vs sex variable

sns.countplot(x='income',hue='sex',data=df)
sns.countplot(x='income',hue='race',data=df)
df['workclass'].nunique()
df['workclass'].unique()
df['workclass'].value_counts()


df['workclass'].replace(' ?', np.NaN, inplace=True)
df['workclass'].value_counts()
df['workclass'].value_counts().plot.bar(color='orange')

plt.xticks(rotation=30)
# vizualize workclass Vs Income



sns.countplot(x='workclass',hue='income',data=df)

plt.xticks(rotation=90)
# Visualize workclass with Sex variable

sns.countplot(y='workclass',hue='sex',data=df)
# Explore Occupation Variable

df['occupation'].nunique()
df['occupation'].unique()
df['occupation'].value_counts()
# Replace ? with NaN

df['occupation'].replace(' ?',np.NaN,inplace=True)
df['occupation'].value_counts()
# Visualize Occupation Variable



df['occupation'].value_counts().plot.bar()
# Explore native_country

df['native_country'].nunique()
df['native_country'].unique()
df['native_country'].value_counts()
# Replace ? 

df['native_country'].replace(' ?',np.NaN,inplace=True)
plt.figure(figsize=(20,13))

sns.countplot(x='native_country',data=df)

plt.xticks(rotation=90)

df[categorical_features].isnull().sum().sort_values(ascending=False)
for col in categorical_features:

    print(col,' contains ',len(df[col].unique()),' labels.')
numerical_features = [col for col in df.columns

                     if df[col].dtypes !='object']

numerical_features
df[numerical_features].head()
df[numerical_features].isnull().sum()
df['age'].nunique()
df['age'].head()
sns.distplot(df['age'],bins=10,color='orange')

plt.show()
sns.boxplot(df['age'])
# Relation Between age and income variable

sns.boxplot(x='income',y='age',data=df)
df.corr()
sns.heatmap(df.corr())
sns.pairplot(df)

plt.show()
sns.pairplot(df,hue='sex')

plt.show()
### Categorical Variables



df[categorical_features].isnull().sum()
# Filling missing class with mode



df['workclass'].fillna(df['workclass'].mode()[0],inplace=True)

df['occupation'].fillna(df['occupation'].mode()[0],inplace=True)

df['native_country'].fillna(df['native_country'].mode()[0],inplace=True)
df[categorical_features].isnull().sum()
X = df.drop('income',axis=1)

y = df.income
X.shape,y.shape
# Encoding

import category_encoders as ce

from sklearn.compose import ColumnTransformer

categorical=['workclass', 'education', 'marital_status', 'occupation', 'relationship', 

                                 'race', 'sex', 'native_country']

encoder = ce.OneHotEncoder()

transformer = ColumnTransformer([('one_hot',encoder,categorical)],

                               remainder='passthrough')

transformed_X = transformer.fit_transform(X)

transformed_X

pd.DataFrame(transformed_X)
transformed_X.shape
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

scaler_transformed=scaler.fit_transform(transformed_X)

scaler_transformed.shape
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(scaler_transformed,y,test_size=0.3,random_state=0)
rfc = RandomForestClassifier(random_state=0)

rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
### n_estimators=100

rfc_100 = RandomForestClassifier(n_estimators=100,random_state=0)

rfc_100.fit(X_train,y_train)

y_pred_100 = rfc_100.predict(X_test)
accuracy_score(y_test,y_pred_100)
clf = RandomForestClassifier(n_estimators=100,random_state=0)

clf.fit(X_train,y_train)
clf.get_params()
clf.feature_importances_
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

print(cm)
cm_matrix = pd.DataFrame(data=cm,columns=['Actual Positive:1',

                                         'Actual Negative:0'],

                        index=['Predict Positive:1',

                              'Predict Negative:0'])

cm_matrix
sns.heatmap(cm_matrix,annot=True,fmt='d')
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))