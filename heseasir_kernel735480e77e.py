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
import sklearn
sklearn.__version__
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
full = train.append(test,ignore_index = True)
# full.describe()
full.info()

full['Age'] = full['Age'].fillna( full['Age'].mean() )
full['Fare'] = full['Fare'].fillna( full['Fare'].mean() )
full['Embarked'].value_counts()
full['Embarked'] = full['Embarked'].fillna('S')
full['Cabin'].value_counts().sort_values(ascending = False)
full['Cabin'] = full['Cabin'].fillna('U')
full.info()
pclass_df = pd.DataFrame()
pclass_df = pd.get_dummies(full['Pclass'],prefix = 'pclass')
pclass_df.head()
full = pd.concat([full,pclass_df],axis = 1)
full.info()
full.drop('Pclass',axis = 1, inplace = True)
full['Embarked'].unique()
embarked_df = pd.DataFrame()
embarked_df = pd.get_dummies(full['Embarked'],prefix = 'embarked')
full = pd.concat([full,embarked_df],axis = 1)
full.drop('Embarked',axis = 1,inplace = True)
full.info()
full['Cabin'].unique()
cabin_df = pd.DataFrame()
cabin_series = full['Cabin'].map(lambda c : c[0])
cabin_df = pd.get_dummies(cabin_series,prefix = 'cabin')
full = pd.concat([full,cabin_df],axis = 1)
full.drop('Cabin',axis = 1,inplace = True)
full.info()
# full['Sex'].unique()
full['Sex'] = full['Sex'].map(lambda sex : 1 if sex == 'male' else 0)
full['Sex'].unique()
full.info()
full['Name'].unique()
def process_name(name):
    str1 = name.split(',')
    return str1[1].split('.')[0].strip()

full['Name'].map(process_name).value_counts().sort_values(ascending = False)
name_map = { "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty",
                    "Major" :     "Major"}
name_df = pd.DataFrame()
name_series = full['Name'].map(process_name)
name_series = name_series.map(name_map)
name_df = pd.get_dummies(name_series,prefix = 'name')
full = pd.concat([full,name_df],axis = 1)
full.drop('Name',axis = 1, inplace = True)
full.info()
full['Ticket'].unique()
def ticket_map(ticket):
    return float(ticket.split(' ')[-1].strip())

ticket_mean = full[full['Ticket'] != 'LINE']['Ticket'].map(ticket_map).mean()
ticket_mean
full.replace('LINE',str(ticket_mean),inplace = True)

full['Ticket'] = full['Ticket'].map(ticket_map)

full.info()
full.info()
full['Fam'] = full['SibSp'] + full['Parch'] + 1
full['Fam'].unique()
def person_map(fam):
    if fam <= 3:
        return 1
    elif fam > 3 and fam < 6:
        return 2
    elif fam >= 6:
        return 3
    
person_df = pd.DataFrame()
person_series = full['Fam'].map(person_map)

person_df = pd.get_dummies(person_series,prefix = 'person')
person_df
full = pd.concat([full,person_df],axis = 1)
full.drop('SibSp',inplace = True,axis = 1)
full.drop('Parch',inplace = True,axis = 1)
full.info()
corr_df = full.corr()
corr_df.head()
corr_df['Survived'].sort_values(ascending = False)
select_df = pd.DataFrame()
select_df = pd.concat([name_df,full['Sex'],pclass_df,full['Fare']],axis = 1)
select_df.info()
source_row = 890

source_x = select_df.loc[0:source_row,:]
source_y = full.loc[0:source_row,'Survived']

predict_x = select_df.loc[source_row + 1 :,:]
from sklearn.model_selection import train_test_split 

train_x,test_x,train_y,test_y = train_test_split(source_x,source_y,train_size = .8)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(train_x,train_y)


# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False)

score = model.score(test_x,test_y)

predict_y = model.predict(predict_x)
predict_y
predict_y=predict_y.astype(int)
predict_y.shape[0]


#乘客id
passenger_id = full.loc[source_row + 1 :,'PassengerId']

passenger_id.shape[0]

#数据框：乘客id，预测生存情况的值
pred_df = pd.DataFrame( 
    { 'PassengerId': passenger_id , 
     'Survived': predict_y } )


#预测数据集总人数
pred_df.shape[0]


#预测的生存人数
survivor_df=pred_df.loc[pred_df.Survived==1]
survivor_df.shape[0]


#预测生存率
survive_rate=survivor_df.shape[0]/pred_df.shape[0]
print('生存率',survive_rate)