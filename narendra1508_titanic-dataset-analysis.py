# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Import Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Comment this if the data visualisations doesn't work on your side
%matplotlib inline

# This is to supress the warning messages (if any) generated in our code
import warnings
warnings.filterwarnings('ignore')

# We are using whitegrid style for our seaborn plots. This is like the most basic one
sns.set_style(style = 'whitegrid')

# below comments are part of kaggle. uncomment the code in kaggle.
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os
# print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
pd.options.display.max_rows = 15
pd.options.display.float_format = '{:,.2f}'.format
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.sample(5)
train.describe()
test.describe()
train.drop(['PassengerId'],axis=1,inplace=True)
train.drop(['Name'],axis=1,inplace=True)
train.drop(['Ticket'],axis=1,inplace=True)

test.drop(['PassengerId'],axis=1,inplace=True)
test.drop(['Name'],axis=1,inplace=True)
test.drop(['Ticket'],axis=1,inplace=True)

train.sample(5)
train_data_stats = pd.DataFrame(columns=['column_name','values','values_count_incna',
                                   'values_count_nona','miss_num','miss_pct'])
tmp = pd.DataFrame()

for c in train.columns:
    tmp['column_name'] = [c]
    tmp['values'] = [train[c].unique()]
    tmp['values_count_incna'] = len(train[c].unique())
    tmp['values_count_nona'] = (train[c].nunique())
    tmp['miss_num'] = train[c].isnull().sum()
    tmp['miss_pct'] = (train[c].isnull().sum() / len(train)) *100
    train_data_stats = train_data_stats.append(tmp)
train_data_stats.sort_values(by = 'values_count_incna',ascending=False,inplace=True)
train_data_stats
test_data_stats = pd.DataFrame(columns=['column_name','values','values_count_incna',
                                   'values_count_nona','miss_num','miss_pct'])
tmp = pd.DataFrame()

for c in test.columns:
    tmp['column_name'] = [c]
    tmp['values'] = [test[c].unique()]
    tmp['values_count_incna'] = len(test[c].unique())
    tmp['values_count_nona'] = (test[c].nunique())
    tmp['miss_num'] = test[c].isnull().sum()
    tmp['miss_pct'] = (test[c].isnull().sum() / len(test)) *100
    test_data_stats = test_data_stats.append(tmp)
test_data_stats.sort_values(by = 'values_count_incna',ascending=False,inplace=True)
test_data_stats
train_data_stats.sort_values(by = 'miss_pct',ascending=False)
# set the index to Column Names
train_data_stats.set_index('column_name',inplace = True)
train_data_stats
train.hist(column = 'Age')
missing_values = train.isnull().sum().sort_values(ascending = False)
missing_values = missing_values[missing_values > 0]
percent = (missing_values/len(train)*100)
pd.concat([missing_values,percent],axis=1,keys=['missing_values','percent'])
test_missing_values = test.isnull().sum().sort_values(ascending = False)
test_missing_values = test_missing_values[test_missing_values > 0]
pct = (test_missing_values / len(test) *100)
pd.concat([test_missing_values,pct],axis=1,keys=['missing_values','percent'])
train.Embarked.value_counts(dropna=False)
#since 77% of data is missing in cabin feature. we can drop the column . But let us fill the missing data with 'N'
# train.drop(['Cabin'],axis=1,inplace=True)
train.Cabin.fillna('N',inplace=True)
# missing values in Age can be filled with mean.
train.Age.fillna(train.Age.mean(),inplace=True)
# since embarked has only 0.22% missing data. we can replace the missing value by mode
train.Embarked.fillna(train.Embarked.mode()[0],inplace=True)
train.isnull().sum()
test.Cabin.fillna('N',inplace = True)
test.Age.fillna(test.Age.mean(), inplace = True)
test.Fare.fillna(test.Fare.mean(), inplace = True)
test.isnull().sum()
train['Cabin'] = [x[0] for x in train.Cabin]
train.head()
test['Cabin'] = [x[0] for x in test.Cabin]
test.head()
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
Label_Encoder_temp = LabelEncoder()
train.loc[:,'Cabin'] = Label_Encoder_temp.fit_transform(train.loc[:,'Cabin'])
test.Cabin = Label_Encoder_temp.fit_transform(test.Cabin)
train.Embarked = Label_Encoder_temp.fit_transform(train.Embarked)
embarked_one_hot_encoder = OneHotEncoder()
e = embarked_one_hot_encoder.fit_transform(train.Embarked.values.reshape(-1,1)).toarray()
e
test.Embarked = Label_Encoder_temp.fit_transform(test.Embarked)
e_test = embarked_one_hot_encoder.fit_transform(test.Embarked.values.reshape(-1,1)).toarray()
e_test
dfOneHot = pd.DataFrame(e, columns = ["Embarked"+str(int(i)) for i in range(e.shape[1])])
train_x = pd.concat([train, dfOneHot.iloc[:,:-1]], axis=1)
train_x.drop(columns=['Embarked'],inplace=True)
train_x.head()
dfOneHot_test = pd.DataFrame(e_test, columns = ['Embarked'+str(int(i)) for i in range(e_test.shape[1])])
test_x = pd.concat([test, dfOneHot_test.iloc[:,:-1]], axis=1)
test_x.drop(columns=['Embarked'],inplace = True)
test_x.head()
train_x.Sex = Label_Encoder_temp.fit_transform(train_x.Sex)
train_x.sample(5)
test_x.loc[:,'Sex'] = Label_Encoder_temp.fit_transform(test_x.loc[:,'Sex'])
test_x.sample(5)
train_y = train_x.loc[:,'Survived']
train_x.drop(columns=['Survived'],inplace = True)
train_x.sample(5)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.fit_transform(test_x)
train_x
test_x
from sklearn import linear_model
logistic_regressor = linear_model.LogisticRegression()
logistic_regressor.fit(train_x,train_y)
y_pred = logistic_regressor.predict(test_x)
y_pred
submission = pd.read_csv("../input/test.csv")
submission = submission['PassengerId']
test_Survived = pd.Series(y_pred, name="Survived")
submission = pd.concat([submission, test_Survived], axis=1)
submission.to_csv('titanic.csv', index=False)