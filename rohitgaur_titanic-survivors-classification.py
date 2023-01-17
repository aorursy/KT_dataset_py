#This notebook contains the classification of the people who survived or died in the titanic sank
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
titanic_df = pd.read_csv('../input/train.csv')
titanic_df.info()
titanic_df.head()
#Data Visualization

sns.countplot(titanic_df['Pclass'],hue=titanic_df['Sex'])
#Visualizing the Null entries

plt.figure(figsize=(10,10))

sns.heatmap(titanic_df.isnull(),yticklabels=False,cmap='viridis',cbar=False)
titanic_df.groupby('Pclass').mean()
#We will define a function to fill the missing entries in the age column

def age_set(cols):

    age = cols[0]

    clas = cols[1]

    if pd.isnull(age):

        if clas == 1:

            return 37.0

        elif clas ==2:

            return 28.0

        else:

            return 24.0

    else:

        return age
titanic_df['Age'] = titanic_df[['Age','Pclass']].apply(age_set,axis=1)
#Dropping the Cabin column and the rest of Null-entries

titanic_df.drop('Cabin',axis=1,inplace=True)

titanic_df.dropna(axis=0,inplace=True)
#Now we can see all null-entries are gone

plt.figure(figsize=(10,10))

sns.heatmap(titanic_df.isnull(),yticklabels=False,cmap='viridis',cbar=False)
#Encoding the Sex column and the Embarked column and putting it in the Dataframe

f_df = pd.get_dummies(titanic_df[['Embarked','Sex']],drop_first=True)

titanic_df.drop(['Embarked','Sex'],axis=1,inplace=True)

titanic_df = pd.concat([titanic_df,f_df],axis=1)

titanic_df.head()
X = titanic_df.drop(['PassengerId','Survived','Name','Ticket'],axis=1)

y = titanic_df['Survived']

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)
from sklearn import metrics

from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = {'C':[0.01,0.1,1,10,100,1000],'penalty':['l1','l2']}

grid1 = GridSearchCV(LogisticRegression(),param_grid=param_grid,cv=10,scoring='accuracy')

grid1.fit(X_train,y_train)

print(grid1.best_score_)

print(grid1.best_params_)
from sklearn.ensemble import RandomForestClassifier

param_grid = {'n_estimators':[20,50,100,150],'max_features':[5,6,7],'max_depth':[5,7,9]}

grid2 = GridSearchCV(RandomForestClassifier(n_jobs=-1),param_grid=param_grid,cv=10,scoring='accuracy')

grid2.fit(X_train,y_train)

print(grid2.best_score_)

print(grid2.best_params_)
from sklearn.ensemble import GradientBoostingClassifier

param_grid = {'learning_rate':[0.01,0.1,1],'n_estimators':[50,100,120],'max_depth':[5,7,9]}

grid3 = GridSearchCV(GradientBoostingClassifier(),param_grid=param_grid,cv=10,scoring='accuracy')

grid3.fit(X_train,y_train)

print(grid3.best_score_)

print(grid3.best_params_)
#Fitting the 3 models with full train data

model1 = LogisticRegression(C=10, penalty='l1').fit(X,y)

model2 = RandomForestClassifier(max_depth=7, max_features=7, n_estimators=150).fit(X,y)

model3 = GradientBoostingClassifier(learning_rate=0.1, max_depth=5, n_estimators=50).fit(X,y)

pred1 = model1.predict(X)

pred2 = model2.predict(X)

pred3 = model3.predict(X)
#taking the average of the models and selecting the best ones

p_avg = []

for i in range(len(y)):

    avg = (pred1[i]+pred2[i]+pred3[i])/3

    if avg >= 0.6:

        p_avg.append(1)

    else:

        p_avg.append(0)

       
#In Sample Accuracy

print(metrics.accuracy_score(y,p_avg))


#Processing the test data

test_df = pd.read_csv('../input/test.csv')
#Checking the head of the test set

test_df.head()
#Visualizing Null rows in the data

plt.figure(figsize=(10,10))

sns.heatmap(test_df.isnull(),yticklabels=False,cmap='viridis',cbar=False)
#Filling the Age column by average values

test_df['Age'] = test_df[['Age','Pclass']].apply(age_set,axis=1)
#Visualizing the dataset by grouping by the Pclass column

test_df.groupby(by='Pclass').mean()
def set_fare(cols):

    pclass = cols[0]

    fare = cols[1]

    if pd.isnull(fare):

        if pclass == 1:

            return 1098.22

        elif pclass ==2:

            return 1117.94

        else:

            return 1094.17

    else:

        return fare
#Filling the empty Fare rows and dropping the Cabin column

test_df['Fare'] = test_df[['Pclass','Fare']].apply(set_fare,axis=1)

test_df.drop('Cabin',axis=1,inplace=True)

test_df.dropna(axis=0,inplace=True)
#Encoding the Sex and Embarked columns

f1_df = pd.get_dummies(test_df[['Embarked','Sex']],drop_first=True)

test_df.drop(['Embarked','Sex'],axis=1,inplace=True)

test_df = pd.concat([test_df,f1_df],axis=1)

test_df.head()
#Final Test dataframe for making predictions

test_df1 = test_df.drop(['PassengerId','Name','Ticket'],axis=1)

test_df1.head()
predf1 = model1.predict(test_df1)

predf2 = model2.predict(test_df1)

predf3 = model3.predict(test_df1)
#taking the average of the three models to get a better result

p_f = []

for i in range(len(test_df1)):

    avg = (predf1[i]+predf2[i]+predf3[i])/3

    if avg >= 0.6:

        p_f.append(1)

    else:

        p_f.append(0)
#Making a final df of predictions and saving our final df as a csv file

final_df = pd.DataFrame(test_df['PassengerId'],columns=['PassengerId'])

final_df['Survived'] = p_f

final_df.head()

final_df.to_csv('titanic_survivals.csv',index=False)
#The original results of the test set

ori_df = pd.read_csv('../input/gendermodel.csv')

y_ori = ori_df['Survived']
#Evaluating our final model on test data

print(metrics.accuracy_score(y_ori,p_f))

print('\n')

print(metrics.classification_report(y_ori,p_f))