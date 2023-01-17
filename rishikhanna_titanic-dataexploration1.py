# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv") 
train_df.head()
print("Size of dataset", len(train_df))
# Basic stats
train_df.describe()
# At a high level lets look at how numerical values differ between Survivors & Non Survivors
survived = train_df[train_df['Survived'] == 1][['Age','Pclass','SibSp','Parch','Fare']].describe()
notsurvived = train_df[train_df['Survived'] == 0][['Age','Pclass','SibSp','Parch','Fare']].describe()
pd.concat([survived,notsurvived], axis=1, keys=['Survived', 'Not Survived'])
train_df[['Age','Fare','Pclass','Parch','SibSp','Survived']].hist(bins=10, color='steelblue', 
           edgecolor='black', linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)    
plt.tight_layout(rect=(0, 0, 1.5, 1.5))
# Check for null values in columns
train_df.isnull().sum()
# Generic function that plots the distribution of a given column as well as prints out common statistical values
def numericalDistribution(df,col,bins=8):
    sns.distplot(df[col].dropna(), kde=True, bins = bins);
    print(df[col].describe())
    print(col, "Skewness:", df[col].skew(axis = 0, skipna = True))
    print("Missing",col,"for",len(df[df[col].isna()]), "out of", len(df))
numericalDistribution(train_df,'Age')
numericalDistribution(train_df,'SibSp')
numericalDistribution(train_df,'Fare',bins=50)
def categoricalDistribution(df,col):
    sns.countplot(x=col,data=df)
    print("Missing",col,"for",len(train_df[df[col].isna()]), "out of", len(df))
categoricalDistribution(train_df,'Survived')
categoricalDistribution(train_df,'Sex')
categoricalDistribution(train_df,'Pclass')
categoricalDistribution(train_df,'Parch')
categoricalDistribution(train_df,'Embarked')
train_df.head()
# one-hot encoding categorical data retaining the Sex & Embarked columns 
train_df['Gender'] = train_df['Sex']
train_df['EmbarkedFrom'] = train_df['Embarked']
train_df = pd.get_dummies(train_df, columns=['Gender','EmbarkedFrom'])
train_df.head()
# Features to evaluate
features = ['Survived','Pclass','Age','SibSp','Parch','Gender_female','Gender_male','EmbarkedFrom_C','EmbarkedFrom_Q','EmbarkedFrom_S']
# How these features correlate to one another
plt.figure(figsize = (15,10))
sns.heatmap(train_df[features].corr(),annot=True,cmap="YlGnBu")
cp = sns.countplot(x="Survived", hue="Sex", data=train_df, 
                   palette={"male": "#FF9999", "female": "#FFE888"})
print("% of male survivals",
      (len(train_df[(train_df.Sex=='male') & (train_df.Survived==1)])/len(train_df[train_df.Sex=='male']))*100)
print("% of female survivals",
      (len(train_df[(train_df.Sex=='female') & (train_df.Survived==1)])/len(train_df[train_df.Sex=='female']))*100)
fc = sns.factorplot(x="Survived", hue="Sex", col="Pclass", 
                    data=train_df, kind="count",
                    palette={"male": "#FF9999", "female": "#FFE888"})
fc = sns.factorplot(x="Survived", hue="Sex", col="Embarked", 
                    data=train_df, kind="count",
                    palette={"male": "#FF9999", "female": "#FFE888"})
sns.factorplot(x="Survived", hue="Sex", col="Parch", 
                    data=train_df, kind="count",
                    palette={"male": "#FF9999", "female": "#FFE888"})
sns.factorplot(x="Survived", hue="Sex", col="SibSp", 
                    data=train_df, kind="count",
                    palette={"male": "#FF9999", "female": "#FFE888"})
# we can also view relationships\correlations as needed                  
sns.lmplot(x='Age', y='Survived', hue='Sex', 
                palette={"male": "#FF9999", "female": "#FFE888"},
                data=train_df, fit_reg=True, legend=True,
                scatter_kws=dict(edgecolor="k", linewidth=0.5))
#1) Replace Nan Age with Age median will do 2) in a separate sheet.
train_df['Age'].fillna((train_df['Age'].median()), inplace=True)
len(train_df[train_df['Age'].isna()])
# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
import base64
from IPython.display import HTML

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
# Create a csv file that can be uploaded into Model building notebooks that work on this processed CSV rather than the raw input
feature_names = ['Survived','Age', 'Gender_male', 'Gender_female', 'Pclass','SibSp','Parch','Fare', 'EmbarkedFrom_C','EmbarkedFrom_Q','EmbarkedFrom_S']
new_df = train_df[feature_names]


create_download_link(new_df)
train_df[train_df.Cabin.notna()].head(10)
# Create New column to flag if Cabin is available or not
train_df['Cabin_Available'] = np.where(train_df.Cabin.notna(), 1, 0)
train_df.head()
train_df[['Survived','Cabin_Available']].corr()
# Create a csv file that can be uploaded into Model building notebooks that work on this processed CSV rather than the raw input
feature_names = ['Survived','Age', 'Gender_male', 'Gender_female', 'Pclass','SibSp','Parch','Fare', 'Cabin_Available','EmbarkedFrom_C','EmbarkedFrom_Q','EmbarkedFrom_S']
new_df = train_df[feature_names]


create_download_link(new_df)
