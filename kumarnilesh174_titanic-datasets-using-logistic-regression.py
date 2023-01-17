#Import Libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.



import os

print(os.listdir("../input"))

df = pd.read_csv('../input/train_data.csv')
df
df.dropna(inplace=True)
# shape

print(df.shape)
#columns*rows

df.size
df.isnull().sum()
print(df.info())
df.head(5)
df.tail() 
df.sample(5) 
df.describe()
df.isnull().sum()
df.columns
# histograms

df.hist(figsize=(16,47))

plt.figure()
# Using seaborn pairplot to see the bivariate relation between each pair of features

sns.pairplot(df)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=df,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=df,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass_1',data=df,palette='rainbow')
sns.distplot(df['Age'].dropna(),kde=False,color='darkred',bins=30)
df['Age'].hist(bins=30,color='darkred',alpha=0.7)
sns.countplot(x='Title_1',data=df)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived',axis=1), 

                                                    df['Survived'], test_size=0.22, 

                                                    random_state=51)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report,classification_report
print(classification_report(y_test,predictions))