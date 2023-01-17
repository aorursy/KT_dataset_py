#  This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname,

                           filename))



# Any results you write to the current directory are saved as output.
dataset= pd.read_csv('/kaggle/input/titanic/train.csv')
pd.set_option("display.max_rows", 1000)
dataset.head(10)
dataset.info()
dataset.describe()
dataset[pd.isnull(dataset.Age)]

dataset[pd.isnull(dataset.Cabin)]

dataset[pd.isnull(dataset.Ticket)]

dataset[(dataset.Survived == 0)]

dataset[pd.isnull(dataset.Embarked)]
dataset.Cabin.fillna('Unknown',inplace=True)

dataset.Embarked.fillna('Unknown',inplace=True)
name = dataset.Name

name_titles = name.str.split(',').str[1].str.strip().str.split(' ').str[0]

print(name_titles.unique())

print(name_titles)

dataset['name_titles']= name_titles
dataset.groupby(['name_titles']).count()
# the,ms,Mlle - miss

# capt,col,dr,Jonkheer.,major,Rev.,Sir.,don - mr

# master

#dona,lady,Mme  - mrs

dataset['name_titles'].replace(['the','Ms.','Mlle.','Capt.','Col.','Dr.','Jonkheer.','Major.','Rev.','Sir.','Don.','Dona.','Lady.','Mme.'],['Miss.','Miss.','Miss.','Mr.','Mr.','Mr.','Mr.','Mr.','Mr.','Mr.','Mr.','Mrs.','Mrs.','Mrs.'],inplace=True)

print(dataset['name_titles'].unique())
master_mean = dataset.groupby(['name_titles']).Age.mean()[0]

miss_mean = dataset.groupby(['name_titles']).Age.mean()[1]

mr_mean = dataset.groupby(['name_titles']).Age.mean()[2]

mrs_mean = dataset.groupby(['name_titles']).Age.mean()[3]

print(master_mean,miss_mean,mr_mean,mrs_mean)



dataset['Age'].fillna(dataset.groupby(['name_titles'])['Age'].transform('mean'),inplace=True)
fare_3_mean = dataset.groupby(['Pclass']).Fare.mean()[3]

print(fare_3_mean)
dataset[name_titles=='Miss.']
import seaborn as sns

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import accuracy_score

import lightgbm as lgb
cat_dataset = pd.get_dummies(dataset,columns = ['Sex','Cabin','Embarked' ,'name_titles'])
def get_data_splits(dataframe, valid_fraction=0.1):

    """ Splits a dataframe into train, validation, and test sets. First, orders by 

        the column 'click_time'. Set the size of the validation and test sets with

        the valid_fraction keyword argument.

    """

    valid_rows = int(len(dataframe) * valid_fraction)

    train = dataframe[:-valid_rows * 2]

    # valid size == test size, last two sections of the data

    valid = dataframe[-valid_rows * 2:-valid_rows]

    test = dataframe[-valid_rows:]

    

    for each in [train, valid, test]:

        print(f"Outcome fraction = {each.Survived.mean():.4f}")

    

    return train, valid, test



def train_model(train, valid, test=None, feature_cols=None):

    if feature_cols is None:

        feature_cols = train.columns.drop(['Name'])

    dtrain = lgb.Dataset(train[feature_cols], label=train['Survived'])

    dvalid = lgb.Dataset(valid[feature_cols], label=valid['Survived'])

    

    param = {'num_leaves': 64, 'objective': 'binary', 

             'metric': 'auc', 'seed': 7}

    num_round = 1000

    print("Training model!")

    clf = XGBClassifier(n_estimators=150, max_depth=6, objective='binary:logistic',seed=1,nthread=-1, learning_rate=0.3,

                        early_stopping_rounds=20,verbose_eval=False)

    bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], 

                    early_stopping_rounds=20, verbose_eval=False)

    clf.fit(train[feature_cols],train['Survived'])

    valid_pred = clf.predict(valid[feature_cols])

   

#     for i in range(0,len(valid_pred)):

#         if valid_pred[i]>=.5:       # setting threshold to .5

#            valid_pred[i]=1

#         else:  

#            valid_pred[i]=0

    valid_score = metrics.accuracy_score(valid['Survived'], valid_pred)

    print(f"Validation AUC score: {valid_score}")

    

    if test is not None: 

        test_pred = clf.predict(test[feature_cols])

#         for i in range(0,len(valid_pred)):

#             if valid_pred[i]>=.5:       # setting threshold to .5

#                test_pred[i]=1

#             else:  

#                test_pred[i]=0

        test_score = metrics.accuracy_score(test['Survived'], test_pred)

        return clf, valid_score, test_score

    else:

        return clf, valid_score

    

def submit_predictions(dataset, model,feature_cols):

    sub_pred = model.predict(dataset[feature_cols])

    y_submission = pd.concat([dataset[['PassengerId']],pd.DataFrame(sub_pred,columns=['Survived'])], axis=1, sort=False)

    return y_submission
columns = ['Pclass','Sex_female','Sex_male', 'Age', 'SibSp','Parch', 'Fare','Cabin_F33','Cabin_F38','Cabin_F4','Cabin_G6','Cabin_T','Cabin_Unknown','Embarked_C','Embarked_Q','Embarked_S','Embarked_Unknown','name_titles_Master.','name_titles_Miss.','name_titles_Mr.','name_titles_Mrs.',]

train, valid, test = get_data_splits(cat_dataset)

clf, valid_score,test= train_model(train = train,valid = valid,test=test,feature_cols = columns)
sub_dataset= pd.read_csv('/kaggle/input/titanic/test.csv')

sub_dataset.head(1000)
sub_dataset.Cabin.fillna('Unknown',inplace=True)

sub_dataset.Embarked.fillna('Unknown',inplace=True)

name = sub_dataset.Name

name_titles = name.str.split(',').str[1].str.strip().str.split(' ').str[0]

print(name_titles.unique())

print(name_titles)

sub_dataset['name_titles']= name_titles
sub_dataset['name_titles'].replace(to_replace = ['the','Ms.','Mlle.','Capt.','Col.','Dr.','Jonkheer.','Major.','Rev.','Sir.','Don.','Dona.','Lady.','Mme.'],value = ['Miss.','Miss.','Miss.','Mr.','Mr.','Mr.','Mr.','Mr.','Mr.','Mr.','Mr.','Mrs.','Mrs.','Mrs.'],inplace=True)

print(sub_dataset['name_titles'].unique())
sub_dataset.info()
sub_dataset['Fare'] = sub_dataset['Fare'].replace(np.nan,fare_3_mean)
sub_dataset[name_titles=='Mr.'] = sub_dataset[name_titles=='Mr.'].replace(np.nan,mr_mean)

sub_dataset[name_titles=='Mrs.'] = sub_dataset[name_titles=='Mrs.'].replace(np.nan,mrs_mean)

sub_dataset[name_titles=='Master.'] = sub_dataset[name_titles=='Master.'].replace(np.nan,master_mean)

sub_dataset[name_titles=='Miss.'] = sub_dataset[name_titles=='Miss.'].replace(np.nan,miss_mean)

sub_dataset.at[88,'Age'] = miss_mean

cat_sub_dataset = pd.get_dummies(sub_dataset,columns = ['Sex','Cabin','Embarked' ,'name_titles'])

cat_sub_dataset['Cabin_T'] = 0

cat_sub_dataset['Cabin_F38'] = 0

cat_sub_dataset['Embarked_Unknown'] = 0
sub_y = submit_predictions(cat_sub_dataset,model = clf,feature_cols = columns)

# print(sub_y)

sub_y.to_csv('./sub_1.csv',index=False)
cat_sub_dataset['PassengerId']