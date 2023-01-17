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
titanic_df = pd.read_csv('../input/train.csv')
from sklearn.preprocessing import LabelEncoder



#####################

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
titanic_dff = pd.read_csv('../input/test.csv')


y_train = titanic_df['Survived'].values

X_train = titanic_df.drop( ['Survived', 'PassengerId','Name', 'Ticket'], axis=1 )

X_test = titanic_dff.drop( ['PassengerId','Name', 'Ticket'], axis=1 )



frames = [X_train, X_test]

result = pd.concat(frames)





#titanic_df.loc[ titanic_df['Age'].isnull() , 'Age'] = -9999



#titanic_dff.loc[ titanic_dff['Fare'].isnull() , 'Fare'] = (titanic_dff.loc[titanic_dff['Pclass']==3, 'Fare']).copy().median()

#titanic_dff.loc[ titanic_dff['Fare'].isnull()& titanic_dff['Pclass']==3, 'Fare'] = (titanic_dff.loc[titanic_dff['Pclass']==3, 'Fare']).copy().median()







result.loc[ result['Age'].isnull() , 'Age'] =result.loc[result['Pclass']==1,]['Age'].median()

result.loc[ result['Fare'].isnull() , 'Fare'] =result['Fare'].median()



#titanic_dff.loc[ titanic_dff['Age'].isnull() , 'Age'] = 37













#titanic_dff.loc[ titanic_dff['Age'].isnull() , 'Age'] = titanic_dff['Age'].median()







result.loc[result['Embarked'].isnull(), 'Embarked'] = '-9999'





result.loc[result['Cabin'].isnull(), 'Cabin'] = '-9999'





from sklearn.preprocessing import LabelEncoder



for col in ['Sex', 'Embarked', 'Cabin']:

    lb = LabelEncoder()

    lb.fit( result[col] )

    result[col] = lb.transform( result[col] )

    





titanic_df_train = result[:891].copy()

titanic_df_test  = result[891:].copy()

#print(titanic_df_train.shape, titanic_df_test.shape)





from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

param_dc = { "max_depth" : range(1,20),

            "min_samples_leaf" : range(1,20)}

vc = StratifiedKFold(n_splits=5, shuffle=True, random_state=230)





gs = GridSearchCV( estimator=dt, param_grid=param_dc, cv=vc )



gs.fit(titanic_df_train,y_train)



xxx=gs.predict(titanic_df_test)
submission=pd.DataFrame({

    "PassengerId": titanic_dff["PassengerId"],

    "Survived": xxx

})

submission.to_csv('titanic.csv',index=False)