import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data = train_data.drop(columns=['Name','Cabin'])

train_data['family_member'] = train_data['SibSp'] + train_data['Parch']

train_data = train_data.drop(columns=['SibSp', 'Parch'])
test_data = test_data.drop(columns=['Name','Cabin'])

test_data['family_member'] = test_data['SibSp'] + test_data['Parch']

test_data = test_data.drop(columns=['SibSp', 'Parch'])
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        

        if Pclass == 1 :

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
train_data['Age'] = train_data[['Age', 'Pclass']].apply(impute_age, axis=1)

test_data['Age'] = test_data[['Age', 'Pclass']].apply(impute_age, axis=1)
train_data.fillna(0,inplace=True)

test_data.fillna(0,inplace=True)
X = train_data.drop(columns=['Survived'])

Y = train_data['Survived']







#choose the features we want to train, just forget the float data

cate_features_index = np.where(X.dtypes != float)[0]

from catboost import CatBoostClassifier, cv, Pool
from sklearn.model_selection import train_test_split
#make the x for train and test (also called validation data) 

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,train_size=0.90,random_state=42)
clf =CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42)
#now just to make the model to fit the data

clf.fit(xtrain,ytrain,cat_features=cate_features_index,eval_set=(xtest,ytest), early_stopping_rounds=50)
test_id = test_data.PassengerId
test_data.isnull().sum()
prediction = clf.predict(test_data)
len(test_id),len(prediction)
df_sub = pd.DataFrame()

df_sub['PassengerId'] = test_id

df_sub['Survived'] = prediction.astype(np.int)



df_sub.to_csv('gender_submission.csv', index=False)