import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
training_df = pd.read_csv('../input/train.csv')

testing_df=pd.read_csv('../input/test.csv')
training_df.head()
testing_df.head()
sns.heatmap(training_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=training_df)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=training_df)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=training_df)
sns.distplot(training_df['Age'].dropna(),kde=False,bins=30)
training_df['Age'].hist(bins=30,color='darkred',alpha=0.7)
sns.countplot(x='SibSp',data=training_df)
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=training_df)
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age
training_df['Age'] = training_df[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(training_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
training_df.drop('Cabin',axis=1,inplace=True)
training_df.head()
training_df.dropna(inplace=True)
training_df.info()
sex = pd.get_dummies(training_df['Sex'],drop_first=True)

embark = pd.get_dummies(training_df['Embarked'],drop_first=True)
training_df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
training_df = pd.concat([training_df,sex,embark],axis=1)
training_df.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(training_df.drop('Survived',axis=1), 

                                                    training_df['Survived'], test_size=0.30)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))