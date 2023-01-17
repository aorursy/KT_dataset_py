import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.ensemble import VotingRegressor, AdaBoostRegressor, ExtraTreesRegressor,GradientBoostingRegressor, RandomForestRegressor

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

input_train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
input_train_df.head()
input_train_df.isnull().sum()
input_train_df.shape
input_train_df.Embarked.value_counts()
input_train_df['age'] = input_train_df['Age'].fillna(input_train_df['Age'].mean())

input_train_df.isnull().sum()
## Title Extraction



first_name = input_train_df.Name[0]

print(first_name)
first_name.split(', ')[1].split('.')[0]
def title(name):

    if name.split(', ')[1].split('.')[0] not in ['Mr','Miss','Mrs','Master']:

        return 'Other'

    else:

        return name.split(', ')[1].split('.')[0]
def MakeDateModelReady(df):

    start_df = df.drop(columns=['PassengerId', 'Cabin','Ticket']) ## Cabin is missing almost all the data and PassengerID is not useful

    

    dummy_sex_df = pd.get_dummies(df.Sex)

    

    start_df = start_df.drop(columns=['Sex'])

    

    dummy_embarked_df = pd.get_dummies(df.Embarked)

    

    start_df = start_df.drop(columns=['Embarked'])

    

    dummy_title_df = pd.get_dummies(df.Name.apply(lambda x: title(x)))

    

    dummy_first_letter_last_name = pd.get_dummies(df.Name.apply(lambda x: x[0].lower()))

    

    start_df['name_len'] = df.Name.apply(lambda x: len(x))

    

    start_df = start_df.drop(columns=['Name'])

    

    Fare_mean = start_df.Fare.mean()

    

    start_df.Fare.fillna(Fare_mean)

    

    start_df['scaled_fare'] = df.Fare.apply(lambda x: (x - df.Fare.mean())/df.Fare.std())

    

    start_df = start_df.drop(columns=['Fare'])

    

    start_df['group_including_person'] = df.Parch + df.SibSp + 1

    

    start_df = start_df.drop(columns=['Parch','SibSp'])

    

    Age_mean = start_df.Age.mean()

    

    start_df['age'] = start_df['Age'].fillna(Age_mean)

    

    start_df = start_df.drop(columns=['Age'])

    

    end_df = pd.concat([start_df, dummy_embarked_df, dummy_title_df, dummy_sex_df, dummy_first_letter_last_name], axis=1)

    

    return end_df
X = MakeDateModelReady(input_train_df).drop(columns=['Survived'])

y = input_train_df['Survived']



print(X.shape)

print(y.shape)
X.isnull().sum()
train_x, test_x, train_y, test_y = train_test_split(X,y)
adar = AdaBoostRegressor(n_estimators=25, learning_rate=0.001, loss='linear')

extr = ExtraTreesRegressor(n_estimators=100, max_depth=25, min_samples_split=5)

rfr = RandomForestRegressor(n_estimators=100, max_depth=25, min_samples_split=5)

gbr = GradientBoostingRegressor(n_estimators=25, learning_rate=0.001)



vote_reg = VotingRegressor([('adar',adar), ('extr', extr), ('rfr', rfr), ('gbr',gbr)] )
adar.fit(train_x,train_y)
extr.fit(train_x,train_y)
rfr.fit(train_x,train_y)
gbr.fit(train_x,train_y)
vote_reg.fit(train_x,train_y)
from sklearn.metrics import confusion_matrix
adar_pred = adar.predict(test_x)

adar_pred_whole = [1 if x > .5 else 0 for x in adar_pred]

confusion_matrix(test_y, adar_pred_whole)
extr_pred = extr.predict(test_x)

extr_pred_whole = [1 if x > .5 else 0 for x in extr_pred]

confusion_matrix(test_y, extr_pred_whole)
rfr_pred = rfr.predict(test_x)

rfr_pred_whole = [1 if x > .5 else 0 for x in rfr_pred]

confusion_matrix(test_y, rfr_pred_whole)
gbr_pred = gbr.predict(test_x)

gbr_pred_whole = [1 if x > .5 else 0 for x in gbr_pred]

confusion_matrix(test_y, gbr_pred_whole)
vote_reg_pred = vote_reg.predict(test_x)

vote_reg_pred_whole = [1 if x > .5 else 0 for x in vote_reg_pred]

confusion_matrix(test_y, vote_reg_pred_whole)
from sklearn.linear_model import ElasticNetCV, BayesianRidge, LassoCV, LogisticRegression
encv = ElasticNetCV()

encv.fit(train_x,train_y)

encv_pred = encv.predict(test_x)

encv_pred_whole = [1 if x > .5 else 0 for x in encv_pred]

confusion_matrix(test_y, encv_pred_whole)
brr = BayesianRidge()

brr.fit(train_x,train_y)

brr_pred = encv.predict(test_x)

brr_pred_whole = [1 if x > .5 else 0 for x in brr_pred]

confusion_matrix(test_y, brr_pred_whole)
lcvr = LassoCV()

lcvr.fit(train_x,train_y)

lcvr_pred = lcvr.predict(test_x)

lcvr_pred_whole = [1 if x > .5 else 0 for x in lcvr_pred]

confusion_matrix(test_y, lcvr_pred_whole)
test_df=pd.read_csv('/kaggle/input/titanic/test.csv')

test_df.columns
pid = test_df.PassengerId
test_df_for_model=MakeDateModelReady(test_df)

test_df_for_model['scaled_fare'] = test_df_for_model['scaled_fare'].fillna(0)

test_df_for_model['u'] = test_df_for_model.apply(lambda x: 0, axis =1)

test_df_for_model['y'] = test_df_for_model.apply(lambda x: 0, axis =1)
print(test_df_for_model.columns)

print(X.columns)

print(test_df_for_model.shape)

print(X.shape)
preds = vote_reg.predict(test_df_for_model)

preds_whole = [1 if x > .5 else 0 for x in preds]
sub_df = pd.DataFrame.from_dict({'PassengerId':pid, 'Survived':preds_whole})
sub_df.Survived.value_counts()
sub_df.to_csv('sub.csv',index=False)