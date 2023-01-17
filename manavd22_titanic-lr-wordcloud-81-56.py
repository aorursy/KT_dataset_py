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
import seaborn as sns
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head() 
train.dtypes
sns.heatmap(train.isnull(), yticklabels=False, cmap="YlGn")
sns.pairplot(train[['Survived','Pclass','Age','Fare']],height=2,hue='Survived')
train['Age']=train.groupby('Pclass')['Age'].transform(lambda x:x.fillna(x.median()))
train['Cabin']=train['Cabin'].fillna('U')
words=''
for val in train.Cabin:
    val =str(val)
    val=val.split()
    words+=' '.join(val)+' '
wordcloud = WordCloud(background_color='White').generate(words)
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis('off')
plt.show()
train['Cabin']=train['Cabin'].astype(str).str[0]
sns.heatmap(train.isnull(), yticklabels=False, cmap="YlGn")
words=''
for val in train.Ticket:
    val =str(val)
    val=val.split()
    words+=' '.join(val)+' '
wordcloud = WordCloud(background_color='White').generate(words)
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis('off')
plt.show()
train=train.drop(['Ticket'],axis=1)
words=''
for val in train.Name:
    val =str(val)
    val=val.split()
    words+=' '.join(val)+' '
wordcloud = WordCloud(background_color='White').generate(words)
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis('off')
plt.show()
train['Name'] = train.Name.str.extract(pat=' ([A-Za-z]+)\.', expand = False)
train['Name'] = train['Name'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')
train=train.drop(['PassengerId'],axis=1)
train.head()
one_hot_train=pd.get_dummies(train)
X_train, X_test, Y_train, Y_test = train_test_split(one_hot_train.drop(['Survived'], axis=1), one_hot_train['Survived'], test_size = 0.2, random_state=2)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression(max_iter=10000)
model.fit(X_train,Y_train)
prediction_lr=model.predict(X_test)
accuracy_score(Y_test,prediction_lr)
test=pd.read_csv('/kaggle/input/titanic/test.csv')
test['Name'] = test.Name.str.extract(pat=' ([A-Za-z]+)\.', expand = False)
test['Name'] = test['Name'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')
test['Age']=test.groupby('Pclass')['Age'].transform(lambda x:x.fillna(x.median()))
test['Cabin']=test['Cabin'].fillna('U')
test['Cabin']=test['Cabin'].astype(str).str[0]
test=test.drop(['Ticket'],axis=1)
test['Fare']=test.groupby('Pclass')['Fare'].transform(lambda x:x.fillna(x.median()))
one_hot_test=pd.get_dummies(test)
one_hot_train.columns.difference(one_hot_test.columns)
one_hot_test['Cabin_T']=0
one_hot_train.columns.difference(one_hot_test.columns)
one_hot_test=one_hot_test.drop(['PassengerId'],axis=1)
prediction_test=model.predict(one_hot_test)
submission = pd.DataFrame({'PassengerID':test['PassengerId'],
                          'Survived': prediction_test})
submission
submission.to_csv('submission.csv',index=False)