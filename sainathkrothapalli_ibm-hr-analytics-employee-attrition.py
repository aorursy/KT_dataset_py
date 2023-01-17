import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import numpy as np

df=pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

df.head()
df.shape
df['Attrition']=df['Attrition'].map({'Yes':1,'No':0})

df.columns
df.info()
df.describe()
num_features=df.select_dtypes(include=['float64','int64'])

cat_features=df.select_dtypes(include=['O'])
num_features.columns
sns.heatmap(df.isnull(),yticklabels=False, cbar=False)
import matplotlib.gridspec as gridspec

def numerical_analysis(feature,data):

    fig=plt.figure(constrained_layout=True)

    grid=gridspec.GridSpec(ncols=1,nrows=1,figure=fig)

   # ax1=fig.add_subplot(grid[0,0])

    #sns.distplot(data[feature],ax=ax1)

    #ax2=fig.add_subplot(grid[1,:])

    #sns.countplot(x=feature,data=df,hue='Attrition',ax=ax2)

    ax3=fig.add_subplot(grid[0,0])

    sns.boxplot(x='Attrition',y=feature,data=df,ax=ax3)

    
df.hist(figsize = (20,20))

plt.show()
cat_features.columns
sns.countplot(x='Attrition',data=df)
sns.countplot(x='MaritalStatus',hue='Attrition',data=df)
sns.countplot(x='BusinessTravel',hue='Attrition',data=df)
sns.countplot(x='Department',hue='Attrition',data=df)
sns.countplot(x='Gender',hue='Attrition',data=df)
plt.subplots(figsize=(20,5))

sns.countplot(x='JobRole',hue='Attrition',data=df)
#numerical_analysis('Education',df)

sns.countplot(x='Over18',hue='Attrition',data=df)
df['PerformanceRating'].value_counts()

sns.countplot(x='PerformanceRating',hue='Attrition',data=df)
sns.countplot(x='RelationshipSatisfaction',hue='Attrition',data=df)
sns.countplot(x='JobLevel',hue='Attrition',data=df)
for i in num_features:

    numerical_analysis(i,df)
from sklearn.preprocessing import LabelEncoder

label= LabelEncoder()

df["Attrition"]=label.fit_transform(df["Attrition"])

df["BusinessTravel"]=label.fit_transform(df["BusinessTravel"])

df["Department"]=label.fit_transform(df["Department"])

df["EducationField"]=label.fit_transform(df["EducationField"])

df["Gender"]=label.fit_transform(df["Gender"])

df["JobRole"]=label.fit_transform(df["JobRole"])

df["MaritalStatus"]=label.fit_transform(df["MaritalStatus"])

df["OverTime"]=label.fit_transform(df["OverTime"])





df.drop(["EmployeeNumber","Over18","EmployeeCount","StandardHours"],axis=1,inplace=True)
from scipy.stats import zscore

z_score=abs(zscore(df))

print("The shape of dataset before removing outliers",df.shape)

df=df.loc[(z_score<3).all(axis=1)]

print("The shape of dataset after removing outliers",df.shape)
skewness=df.skew()

skewness
Y=df['Attrition']

X=df.drop(['Attrition'],axis=1)

type(X)
from sklearn.tree import DecisionTreeClassifier

dc=DecisionTreeClassifier()

dc.fit(X,Y)
feat_importances = pd.Series(dc.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()