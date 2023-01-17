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

import matplotlib.pyplot as plt 



df_test = pd.read_csv('../input/titanic/test.csv')

df = pd.read_csv('../input/titanic/train.csv')



df.head()
df.info()
df.describe()
df.hist(bins = 50 , figsize = (15 , 10 ))

plt.show()
df['Age'].fillna(df.Age.mean() , inplace = True )

df_test['Age'].fillna(df_test.Age.mean() , inplace = True )
df['Age'] = df['Age'].astype(int)

df_test['Age'] = df_test['Age'].astype(int)

df.info()
df.drop('Cabin'  , axis = 1 , inplace = True)

df_test.drop('Cabin'  , axis = 1 , inplace = True)
df.info()
# show how much different float data we have 

len(df.Fare.unique())
# you can see here that we reduce our values alot and this will help our model to preict more accurate

df.Fare = df.Fare.astype(int)



# there is null value in fare so we will fill it before convert from float to integer

df_test.Fare.fillna(df_test.Fare.mean(),inplace = True)

df_test.Fare = df_test.Fare.astype(int)

len(df.Fare.unique())
# we can see that there is somehow outlires in the fare attribute so we try to clean it more

df.Fare.hist(bins = 100)
df.info()
# these are the two rows that have missing values

df[df.Embarked.isna()]
# we can see also that those two data only who have fare of 80

df[df.Fare == 80 ]
# i choose to delete them because i do not trust these data and it could make me some noise

df.drop(df.loc[df['Fare']==80].index, inplace=True)
df.info()
df.Embarked.unique()
df['Sex'].replace ({'male' :0 , 'female' : 1 } , inplace = True)

df.Embarked = df.Embarked.astype('category')

df.Embarked = df.Embarked.cat.codes



df_test['Sex'].replace ({'male' :0 , 'female' : 1 } , inplace = True)

df_test.Embarked = df_test.Embarked.astype('category')

df_test.Embarked = df_test.Embarked.cat.codes

df.info()
df.drop(columns = ['Name' , 'Ticket'], axis = 1 , inplace = True)

df_test.drop(columns = ['Name' , 'Ticket'], axis = 1 , inplace = True)
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression



X = df.drop('Survived' , axis = 1 )

y = df['Survived']



# split the data to train and test model

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)



dec = DecisionTreeClassifier() 

ran = RandomForestClassifier(n_estimators=100)

knn = KNeighborsClassifier()

#svm = SVC(random_state=1)

naive = GaussianNB()

logReg = LogisticRegression()



models = {"Decision tree" : dec,

          "Random forest" : ran,

          "KNN" : knn,

          #"SVM" : svm,

          "Naive bayes" : naive,

          "Logistics regression": logReg}

scores= { }



for key, value in models.items():    

    model = value

    model.fit(X, y)

    scores[key] = model.score(X, y)

    

scores_frame = pd.DataFrame(scores, index=["Accuracy Score"]).T

scores_frame.sort_values(by=["Accuracy Score"], axis=0 ,ascending=False, inplace=True)

scores_frame
df.info()
d = { 'PassengerId': df_test.PassengerId, 'Survived': ran.predict (df_test)}

d
submission_frame = pd.DataFrame(d)

submission_frame
submission_frame.to_csv ('submission_frame.csv',index=False)