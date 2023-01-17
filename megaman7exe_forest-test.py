import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Read the data set
df=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
#Display first 5 rows of data set
df.head()
#Display last 5 rows of dataset
df.tail()
#Check shape of the date
df.shape
#Information about the data
df.info()
#statstical summary of the data
df.describe()
#Finding null or missing values
df.isnull().any()
#Replacing 0 with null value(NaN)
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure',
'SkinThickness','Insulin','BMI']].replace(0,np.NaN)
df.head()
df.isnull().any()
#Now find the total number of missing values
df.isnull().sum()
#Replace null values with median
df['Glucose'].fillna(df['Glucose'].median(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].median(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace=True)
df['Insulin'].fillna(df['Insulin'].median(), inplace=True)
df['BMI'].fillna(df['BMI'].median(), inplace=True)
df.head()
#Histogram
df.hist(figsize=(20,20))
#Checking the skewness of the data
import matplotlib.pyplot as plt
import seaborn as sns

fig,axes=plt.subplots(4,2, figsize=(12,12))
sns.distplot(df['Pregnancies'],ax=axes[0,0])
sns.distplot(df['Glucose'],ax=axes[0,1])
sns.distplot(df['BloodPressure'],ax=axes[1,0])
sns.distplot(df['SkinThickness'],ax=axes[1,1])
sns.distplot(df['Insulin'],ax=axes[2,0])
sns.distplot(df['BMI'],ax=axes[2,1])
sns.distplot(df['DiabetesPedigreeFunction'],ax=axes[3,0])
sns.distplot(df['Age'],ax=axes[3,1])
plt.show()
fig,axes=plt.subplots(4,2, figsize=(16,16))
sns.boxplot(df['Pregnancies'],ax=axes[0,0])
sns.boxplot(df['Glucose'],ax=axes[0,1])
sns.boxplot(df['BloodPressure'],ax=axes[1,0])
sns.boxplot(df['SkinThickness'],ax=axes[1,1])
sns.boxplot(df['Insulin'],ax=axes[2,0])
sns.boxplot(df['BMI'],ax=axes[2,1])
sns.boxplot(df['DiabetesPedigreeFunction'],ax=axes[3,0])
sns.boxplot(df['Age'],ax=axes[3,1])
plt.show()

#Let's check out pairplot for the data
sns.pairplot(df, hue ='Outcome')
#creating correlation matrix
corr=df.corr()
corr
#Heatmap
sns.heatmap(corr,annot=True)
#Reload the data
df=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df
#We split first to avoid any data leakage later
from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.20,random_state=2,stratify=df['Outcome'])
train_X=train.drop(columns=['Outcome'])
test_X=test.drop(columns=['Outcome'])
train_Y=train['Outcome']
test_Y=test['Outcome']

#As explored in section 2. Cleaning, we will replace the 0s in specific columns with median (as they cannot be 0 in real world scenarios)
train_X[['Glucose','BloodPressure','Insulin','BMI','SkinThickness']] = train_X[['Glucose','BloodPressure','Insulin','BMI','SkinThickness']].replace(0,np.NaN)
test_X[['Glucose','BloodPressure','Insulin','BMI','SkinThickness']] = test_X[['Glucose','BloodPressure','Insulin','BMI','SkinThickness']].replace(0,np.NaN)
for C in ['Glucose','BloodPressure','Insulin','BMI','SkinThickness']:
    train_X[C].fillna(df[C].median(), inplace=True)
    test_X[C].fillna(df[C].median(), inplace=True)
from sklearn import preprocessing
#Normalization (standard scaling)
scaler = preprocessing.StandardScaler()
normalized_train_X=scaler.fit_transform(train_X)
normalized_test_X=scaler.transform(test_X)
pd.DataFrame(normalized_train_X)
pd.DataFrame(normalized_test_X)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
#Trial fit
forest_model = RandomForestClassifier(n_estimators=30, random_state=1, n_jobs=-1)
forest_model.fit(normalized_train_X, train_Y)
forest_model.score(normalized_test_X,test_Y)
d = 0
for estimator in forest_model.estimators_:
    d = d + estimator.get_depth()
int(d / len(forest_model.estimators_))  #### The average depth of all the decision trees inside the forest ensemble
#We will prune our random forest by 10% of the existing average depth
d = (d / len(forest_model.estimators_))*0.9