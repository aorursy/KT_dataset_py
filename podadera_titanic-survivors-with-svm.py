# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_df=pd.read_csv('../input/titanic/train.csv')
test_df=pd.read_csv('../input/titanic/test.csv')
#train_df.head(20)
test_df.head(20)
sex_df=train_df.loc[:,'Sex']
sex_df=pd.get_dummies(sex_df)
#sex_df.head()

sex_df_test=test_df.loc[:,'Sex']
sex_df_test=pd.get_dummies(sex_df_test)
sex_df_test.head()
embarked_df=train_df.loc[:,'Embarked']
embarked_df=pd.get_dummies(embarked_df)
#embarked_df.head()

embarked_df_test=test_df.loc[:,'Embarked']
embarked_df_test=pd.get_dummies(embarked_df_test)
embarked_df_test.head()
train_df2=train_df.loc[:,['Pclass','Age','SibSp','Parch']]
train_df2=train_df2.join(sex_df)
train_df2=train_df2.join(embarked_df)
#train_df2.head()

test_df2=test_df.loc[:,['Pclass','Age','SibSp','Parch']]
test_df2=test_df2.join(sex_df_test)
test_df2=test_df2.join(embarked_df_test)
test_df2.head()
from sklearn.impute import SimpleImputer
train_df3 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(train_df2)
test_df3 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(test_df2)
from sklearn import preprocessing
train_df4 = preprocessing.scale(train_df3)
test_df4 = preprocessing.scale(test_df3)
#train_df4
#test_df4
objective=train_df.loc[:,'Survived']
objective.head()
from sklearn.svm import SVC
titanic=SVC(class_weight='balanced')
titanic
titanic.fit(train_df4,objective)
predict=titanic.predict(train_df4)
from sklearn.metrics import classification_report as clf
print(clf(objective,predict))
predict_test=titanic.predict(test_df4)
result=test_df.loc[:,"PassengerId"]
result_df=pd.DataFrame(result)
result_df["Survived"]=predict_test
result_df.to_csv('my_submission3.csv', index=False)
print("Your submission was successfully saved!")