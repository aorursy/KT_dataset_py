# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.describe(), train.columns
train = train.drop(['Name','Cabin','Ticket'],axis=1) # dropping columns that are not going to be used
test = test.drop(['Name','Cabin','Ticket'],axis=1)
train.shape, test.shape # rows, columns
len(train) # no of rows
train.info(), train.isna().sum()
train.describe(include=['O']) # summary of objects
pclass_survived = train[['Pclass','Survived']].groupby('Pclass',as_index=False).mean()
train[['Pclass','Survived']].groupby('Pclass',as_index=False).mean()
plt.bar( pclass_survived['Pclass'],pclass_survived['Survived'], align='center', alpha=0.5)
plt.xlabel('Pclass')
plt.ylabel('Survived')
sex_table = train[['Sex','Survived']].groupby('Sex',as_index=False).mean()
train[['Sex','Survived']].groupby('Sex',as_index=False).mean()
plt.bar( sex_table['Sex'],sex_table['Survived'], align='center', alpha=0.5)
plt.xlabel('Sex')
plt.ylabel('Survived')
sib_table = train[['SibSp','Survived']].groupby('SibSp',as_index=False).mean().sort_values(by='Survived',ascending = False)
train[['SibSp','Survived']].groupby('SibSp',as_index=False).mean().sort_values(by='Survived',ascending = False)
plt.bar( sib_table['SibSp'],sib_table['Survived'], align='center', alpha=0.5)
plt.xlabel('Sibling or Spouse')
plt.ylabel('Survived')
parch_table = train[['Parch','Survived']].groupby('Parch',as_index=False).mean().sort_values(by='Survived',ascending=False)
train[['Parch','Survived']].groupby('Parch',as_index=False).mean().sort_values(by='Survived',ascending=False)

plt.bar( parch_table['Parch'],parch_table['Survived'], align='center', alpha=0.5)
plt.xlabel('Parent-child')
plt.ylabel('Survived')
for i in range(1, 5):
    plt.subplot(2, 2, i)
    if i ==1:
        plt.bar( sex_table['Sex'],sex_table['Survived'], align='center', alpha=0.5)
        plt.xlabel('Sex')
        plt.ylabel('Survived')
    elif i==2:
        plt.bar( pclass_survived['Pclass'],pclass_survived['Survived'], align='center', alpha=0.5)
        plt.xlabel('Pclass')
        
    elif i == 3:
        plt.bar( parch_table['Parch'],parch_table['Survived'], align='center', alpha=0.5)
        plt.xlabel('Parent-child')
        plt.ylabel('Survived')
    elif i == 4:
        plt.bar( sib_table['SibSp'],sib_table['Survived'], align='center', alpha=0.5)
        plt.xlabel('Sibling or Spouse')
        
plt.tight_layout()
        
for df in [train,test]:
    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})
mean_age =train[['Age','Sex']].groupby('Sex', as_index=False).mean()
mean_male = mean_age['Age'].iloc[1] # mean age of men
mean_female = mean_age['Age'].iloc[0] #mean age of women
train.loc[train["Sex"] == "female",'Age'] = train.loc[train["Sex"] == "female",'Age'].fillna(mean_female)
test.loc[train["Sex"] == "female",'Age'] = test.loc[train["Sex"] == "female",'Age'].fillna(mean_female)
train.loc[train["Sex"] == "male",'Age'] = train.loc[train["Sex"] == "male",'Age'].fillna(mean_male)
test.loc[train["Sex"] == "male",'Age'] = test.loc[train["Sex"] == "male",'Age'].fillna(mean_male)
features = ['Pclass','Age','Sex_binary','SibSp','Parch']
target = 'Survived'
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(train[features],train[target])
logreg.coef_
pd.DataFrame(logreg.coef_,columns=['Pclass','Age','Sex_binary','SibSp','Parch'])
predictions = logreg.predict(test[features])
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})

#Visualize the first 5 rows
submission.head()

filename = 'Titanic Prediction.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)










