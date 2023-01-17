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
train_df=pd.read_csv('../input/train.csv')

test_df=pd.read_csv('../input/test.csv')

train_df.head()
train_df['SX'],S=train_df['Sex'].factorize()

test_df['SX']=test_df['Sex'].map(pd.Series(data=range(S.size),index=S))

train_df['SX'].value_counts().plot.bar()
import re

train_df['Title']=train_df['Name'].map(lambda x: re.search('.*, ([^\.]*).*',x).group(1))

train_df['Title_num'],T=train_df['Title'].factorize()

T

train_df['Title'].value_counts().plot.bar()
TITLES=pd.Series(data=range(T.size),index=T)

TITLES
test_df['Title']=test_df['Name'].map(lambda x: re.search('.*, ([^\.]*).*',x).group(1))

test_df['Title_num']=test_df['Title'].map(TITLES)

test_df['Title_num'].isnull().any()

train_df.groupby('SX')['Title_num'].value_counts().plot.bar()
grouped=train_df.groupby('SX')['Title_num'].value_counts().reset_index(name='count')

#Freq_Title=grouped.loc(grouped.groupby('SX'))

Freq_title=grouped.loc[grouped.groupby('SX')['count'].idxmax()][['SX','Title_num']]

Freq_title
Freq_title.set_index('SX',inplace=True)

Freq_title
Freq_title_Series=Freq_title.T.squeeze()

#test_df['SX'].map(Freq_title)

#test_df['Title_num'].fillna(test_df['SX'].map(Freq_title))
test_df['SX'].map(Freq_title_Series)
test_df['Title_num'].fillna(test_df['SX'].map(Freq_title_Series),inplace=True)

test_df['Title_num'].isnull().any()
familysize=lambda x: 1+x['Parch']+x['SibSp']

train_df['FamilySize']=familysize(train_df)

test_df['FamilySize']=familysize(test_df)
train_df['FamilySize'].describe()
train_df['Age'].isnull().any()
#fill up missing age values

Median_age=train_df.groupby('Title_num')['Age'].median()

Median_age
Median=train_df.groupby('Title_num')['Age'].transform('median')

Median
train_df['Title_num'].map(Median_age)
train_df['Age'].fillna(train_df['Title_num'].map(Median_age),inplace=True)

test_df['Age'].fillna(test_df['Title_num'].map(Median_age),inplace=True)
train_df['Age'].isnull().any()

test_df['Age'].isnull().any()
features=['Age','SX','Pclass','Title_num','Fare','FamilySize']
train_df[features].isnull().any()
test_df[features].isnull().any()
Median_Fare=train_df.groupby('Pclass')['Fare'].median()
Median_Fare
test_df['Fare'].fillna(test_df['Pclass'].map(Median_Fare),inplace=True)
test_df[features].isnull().any()
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()



X_train=scaler.fit_transform(train_df[features])

y_train=train_df['Survived']
scaler.scale_
scaler.mean_
X_train[0:5]
#y_train=train_df['Survived']

X_test=scaler.transform(test_df[features])

X_test[0:5]
type(X_train)
from sklearn import svm,model_selection



classifier=svm.SVC(kernel='rbf')

C=np.exp2(np.arange(1,15,2))

g=np.exp2(np.arange(0,-15,-2))

grid_search=model_selection.GridSearchCV(classifier,{'C':C,'gamma':g},cv=5,refit=False)

grid_search.fit(X_train,y_train)

grid_search.best_params_
Cm=np.log2(grid_search.best_params_['C'])

gm=np.log2(grid_search.best_params_['gamma'])

C=np.exp2(np.arange(Cm-1,Cm+1,0.1))

g=np.exp2(np.arange(gm+1,gm-1,-0.1))

final_model=model_selection.GridSearchCV(classifier,{'C':C,'gamma':g},cv=5)

final_model.fit(X_train,y_train)
final_model.best_params_
final_model.best_score_
final_model.predict(X_train[0:10])
y_train.head(10)
from sklearn.externals import joblib

joblib.dump(final_model,'Titanic_SVM.model')
#final_model=joblib.load('Titanic_SVM.model')

prediction=final_model.predict(X_test)

submission=pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':prediction})

submission.to_csv('submission.csv',index=False)
print(check_output(["ls", "../working"]).decode("utf8"))