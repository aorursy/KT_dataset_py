import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline

from xgboost import XGBRegressor

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Datasources

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.head(20)
test_id = test_data.PassengerId
#Prediction Target

#Single column on train data that contains the prediction

train_y = train_data.Survived
#cols_with_missing_values = [col for col in train_data.columns

#                                   if train_data[col].isnull().any()]

cols_with_missing_values=['Age','Cabin']

train_X = train_data.drop(['PassengerId','Survived']+cols_with_missing_values, axis=1)

test_X = test_data.drop(['PassengerId']+cols_with_missing_values, axis=1)
# Categorical values. 

# Choosing only those columns for on hot encodding where the categorical value for any attribute is not more than 10

low_cardinality_cols = [cname for cname in train_X.columns

                                       if train_X[cname].nunique()< 10 and

                                       train_X[cname].dtype=="object"]

numeric_cols = [cname for cname in train_X.columns

                               if train_X[cname].dtype in ['int64', 'float64']]



useful_cols = low_cardinality_cols + numeric_cols

train_X = train_X[useful_cols]

test_X = test_X[useful_cols]
def pairplot(X,y):

    X['y'] = y

    sns.pairplot(X,hue='y')

    X=X.drop(['y'],axis=1)

    return 
#pairplot(train_X,train_y)

#train_X=train_X.drop(['y'],axis=1)
def data_transform(df):

    df['FamilySize'] = df['SibSp']+df['Parch']

    df['HijoUnico'] = (4-df['Pclass'])/(df['Parch']+1)

    df = pd.get_dummies(df)

    return df



train_X = data_transform(train_X)

test_X = data_transform(test_X)
my_pipeline=make_pipeline(SimpleImputer(),XGBRegressor())

my_pipeline.fit(train_X, train_y)



#Get Predictions

predictions = np.around(my_pipeline.predict(test_X),0).astype(np.int64)
#Submit predictions

my_submission = pd.DataFrame({'PassengerId': test_id, 'Survived': predictions})

my_submission.describe()

my_submission.head(10)
my_submission.to_csv('submission.csv', index=False)
my_submission['producto']=my_submission['PassengerId']*my_submission['Survived']

my_submission['producto'].values.sum()