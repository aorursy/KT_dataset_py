# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_dataframe = pd.read_csv('../input/train.csv')
train_dataframe.columns
train_dataframe.head(2)
from IPython.display import HTML, display
data = [['Column','Variable', 'Will it be helpful for Prediction?','Comments'],
        ['PassengerId','Input','No','Not required as it is Row Index'],
        ['Survived','Target','Yes',''],
        ['Pclass','Input','Yes','Need to analyze'],
        ['Name','Input','May be','Need to analyze'],
        ['Sex','Input','Yes','Need to analyze'],
        ['Age','Input','Yes','Need to analyze'],
        ['SibSp','Input','Yes','Need to analyze'],
        ['Parch','Input','Yes','Need to analyze'],
        ['Ticket','Input','N','has relation with Pclass. It can be dropped'],
        ['Fare','Input','N','has relation with Pclass. It can be dropped'],
        ['Cabin','Input','N','has relation with Pclass. It can be dropped'],
        ['Embarked','Input','N','It does not matter where they boarded. It can be dropped']
       ]
display(HTML(
    '<table><tr>{}</tr></table>'.format(
        '</tr><tr>'.join(
            '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in data)
        )
 ))
train_dataframe = train_dataframe.drop(columns=['PassengerId','Ticket','Fare','Cabin','Embarked'])
train_dataframe['SibSp'].value_counts().plot.bar()
train_dataframe['Name'].str.extract(' ([A-Za-z]+)\.', expand=False).value_counts().plot.bar()
train_dataframe['Title'] = train_dataframe['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train_dataframe.head(2)
train_dataframe = train_dataframe.drop(columns=['Name'])
train_dataframe['Title'] = train_dataframe['Title'].replace(['Ms','Mlle'],'Miss')
train_dataframe['Title'] = train_dataframe['Title'].replace('Mme', 'Mrs')
train_dataframe.iloc[train_dataframe[train_dataframe.Title.isin(['Lady', 'Countess','Capt', 'Col', 'Don',\
                                            'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', \
                                            'Dona'])][train_dataframe.Sex == \
                                                      'female'].index,6] = 'Mam'

train_dataframe.iloc[train_dataframe[train_dataframe.Title.isin(['Lady', 'Countess','Capt', 'Col', 'Don',\
                                            'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', \
                                            'Dona'])][train_dataframe.Sex == \
                                                      'male'].index,6] = 'Sir'
train_dataframe[train_dataframe.Title == 'Sir']
train_dataframe['Title'].value_counts().plot.bar()
title_mapping = {"Mr": 1, "Miss": 4, "Mrs": 5, "Master": 3, "Sir": 2,"Mam" : 6}
train_dataframe['Title'] = train_dataframe['Title'].map(title_mapping)
sex_mapping = {"female": 2, "male": 1}
train_dataframe['Sex'] = train_dataframe['Sex'].map(sex_mapping)
train_dataframe.head(5)
cond11 = train_dataframe.Title == 1 
cond12 = train_dataframe.Title == 2
cond13 = train_dataframe.Title == 3
train_dataframe[cond11 | cond12 | cond13 ]['Survived'].value_counts()
cond14 = train_dataframe.Title == 4 
cond15 = train_dataframe.Title == 5
cond16 = train_dataframe.Title == 6
train_dataframe[cond14 | cond15 | cond16]['Survived'].value_counts()
train_dataframe.isna().sum()
train_dataframe['Age'].plot(kind='hist')
ageMean = train_dataframe.groupby('Title').Age.mean().tolist()
(ageMean)
train_dataframe.iloc[train_dataframe[train_dataframe.Age.isna()].Title.index,6]. \
value_counts().plot.bar()
cond1 = train_dataframe.Title == 1
cond2 = train_dataframe.Title == 2
cond3 = train_dataframe.Title == 3
cond4 = train_dataframe.Title == 4
cond5 = train_dataframe.Title == 5
#for Mam not required as is it not empty
cond11 = train_dataframe.Age.isna()
train_dataframe.iloc[train_dataframe[ cond1 &  cond11]['Age'].index,3] = ageMean[0]
train_dataframe.iloc[train_dataframe[ cond2 &  cond11]['Age'].index,3] = ageMean[1]
train_dataframe.iloc[train_dataframe[ cond3 &  cond11]['Age'].index,3] = ageMean[2]
train_dataframe.iloc[train_dataframe[ cond4 &  cond11]['Age'].index,3] = ageMean[3]
train_dataframe.iloc[train_dataframe[ cond5 &  cond11]['Age'].index,3] = ageMean[4]
train_dataframe.isna().sum()
train_dataframe['Age'].plot(kind='hist')
import seaborn as sns
sns.heatmap(train_dataframe.corr(),annot=True , cmap='YlGnBu')
pd.crosstab(index=train_dataframe["Survived"], 
                           columns=[train_dataframe["Title"] ])
train_dataframe['SexAgedivPclass'] = (train_dataframe['Age'] * train_dataframe['Sex']) / train_dataframe['Pclass']
td = train_dataframe.drop(columns=['Age','Sex', 'Pclass'])
sns.heatmap(td.corr(),annot=True , cmap='YlGnBu')
sns.scatterplot(x=td['SexAgedivPclass'] , y=td['Title'], hue=td['Survived'], data=td)
x_train_val = train_dataframe.drop(columns=['Sex','Age','Pclass','Survived'])
y_train_val = train_dataframe.Survived
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_val.shape, y_val.shape)
from keras.models import Sequential
from keras.layers import Dense

# create model
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )
# Fit the model
model.fit(X_train, y_train, epochs=90, batch_size=10,validation_data = (X_val, y_val))
# evaluate the model
scores = model.evaluate(X_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))