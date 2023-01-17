# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_rows',None)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv('/kaggle/input/titanic/train.csv' , index_col = 'PassengerId')

test = pd.read_csv('/kaggle/input/titanic/test.csv' , index_col = 'PassengerId')



whole_data = pd.concat([train, test] , axis = 0, sort=True)

# Any results you write to the current directory are saved as output.
whole_data.isnull().sum()

# Drop the features

data_processing= train.drop(['Cabin','Ticket'],axis=1)



# Replace the NaN data from the column of Embarked as most frequent data 'S'

data_processing.Embarked.fillna('S',inplace=True)



# Processing the test data also

test_processing = test.drop(['Cabin','Ticket'], axis = 'columns')

test_processing.Embarked = test.Embarked.fillna('S')

test_processing.Fare.fillna(test_processing.Fare.mean(), inplace =True)



test_processing.isnull().sum()
# Processing the feature Name

list =[]

for i in range(len(train)):

    list.append(train.Name.iloc[i].split(',')[1].split()[0])

honor = pd.DataFrame(data = list, columns=['honor'], index = train.index)

honor.honor = np.where(honor.honor.isin(['Mr.','Miss.','Mrs.','Master.','Dr.']),

                      honor.honor, 'Rare.')

data_processing.Name = honor.honor



# Processing the feature Sex

data_processing.Sex = np.where(data_processing.Sex=='female',1,0)



# Processing the test data also

list_test = []

for i in range(len(test)):

    list_test.append(test.Name.iloc[i].split(',')[1].split()[0])

honor_test = pd.DataFrame(data = list_test , columns = ['honor'] , index = test.index)

honor_test.honor = np.where(honor_test.honor.isin(['Mr.','Miss.','Mrs.','Master.','Dr.']),

                           honor_test.honor, 'Rare.')



test_processing.Name = honor_test.honor



# Processing the feture sex of the test data

test_processing.Sex = np.where(test_processing.Sex=='female',1,0)



data_processing.head()
# Encoding the columns of Name, Embarked into numerical data using One_hot_encoder.

from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = OneHotEncoder(handle_unknown='ignore')



name_enc = pd.DataFrame(one_hot_encoder.fit_transform(data_processing[['Name']]).toarray(),

                        index = data_processing.index,

                        columns = one_hot_encoder.get_feature_names(['']))

embarked_enc = pd.DataFrame(one_hot_encoder.fit_transform(data_processing[['Embarked']]).toarray(),

                           index = data_processing.index,

                           columns = one_hot_encoder.get_feature_names(['Embarke']))



data_processed = pd.concat([data_processing,name_enc,embarked_enc], axis = 1).drop(['Name','Embarked'], axis = 'columns' )



# Encoding the test data also

name_enc_test = pd.DataFrame(one_hot_encoder.fit_transform(test_processing[['Name']]).toarray(),

                        index = test_processing.index,

                        columns = one_hot_encoder.get_feature_names(['']))

embarked_enc_test = pd.DataFrame(one_hot_encoder.fit_transform(test_processing[['Embarked']]).toarray(),

                           index = test_processing.index,

                           columns = one_hot_encoder.get_feature_names(['Embarke']))



test_processed = pd.concat([test_processing,name_enc_test,embarked_enc_test], axis = 1).drop(['Name','Embarked'], axis = 'columns' )
plt.figure(figsize=(14,10))

sns.heatmap(data_processed.corr(),annot=True,center=0,cmap= 'coolwarm')
# Plotting the histogram of Fare of Survived and Un-Survived.

sns.set(style='darkgrid')

plt.figure(figsize=(10,5))

plt.xlim(-10,150)

sns.distplot(train.groupby('Survived')[['Fare']].get_group(0), bins=300, color='blue', label = 'UnSurvived')

sns.distplot(train.groupby('Survived')[['Fare']].get_group(1), bins=300, color='red', label='Survived')

plt.legend(); plt.xlabel('Fare'); plt.ylabel('Survived'); plt.title('Fare-Survived',fontsize=22); plt.show()

# Analyze Age,Pclass and the others.

plt.figure(figsize=(10,5))

sns.distplot(train.groupby('Survived')[['Age']].get_group(0),

            label = 'UnSurvivied', bins=30)

sns.distplot(train.groupby('Survived')[['Age']].get_group(1),

            label = 'Survivied', bins=30)

sns.kdeplot(data = train.Age)

plt.title('Age - Survived',fontsize = 13) ;plt.xlabel('Age',fontsize = 13)

plt.ylabel('Survived');plt.legend()

man = train.groupby('Sex').get_group('male')

woman = train.groupby('Sex').get_group('female')



f, axes = plt.subplots(1,2 ,figsize=(16,6), sharex = True)

sns.despine(left=True)



sns.distplot(man.groupby('Survived')['Age'].get_group(1),color = 'red', label = 'Survived', ax = axes[0], bins=20)

sns.distplot(man.groupby('Survived')['Age'].get_group(0), label = 'UnSurvived', ax=axes[0], bins=20)

axes[0].set_title('Man - Age'); axes[0].legend(); axes[0].set_ylabel('Ratio')



sns.distplot(woman.groupby('Survived')['Age'].get_group(1),color = 'red', label = 'Survived', ax = axes[1], bins=20)

sns.distplot(woman.groupby('Survived')['Age'].get_group(0), label = 'UnSurvived', ax=axes[1], bins=20)

axes[1].set_title('Woman - Age'); plt.legend()
from sklearn.model_selection import train_test_split as tts

from sklearn.ensemble import RandomForestRegressor





def Age_processing(data):

    man_data = data.groupby('Sex').get_group(0)

    woman_data = data.groupby('Sex').get_group(1)

    

    man_predicted = Age_predict(man_data)

    woman_predicted = Age_predict(woman_data)

    

    man_predicted_cat= pd.cut(x = man_predicted['Age'], bins=[0,18,30,50,man_predicted.Age.max()],

                              labels = [0,1,2,3])

    woman_predicted_cat = pd.cut(x = woman_predicted['Age'], bins=[0,7,22,40,45,woman_predicted.Age.max()],

                                labels = [4,5,6,7,8])

    

    man_predicted.Age = man_predicted_cat

    woman_predicted.Age = woman_predicted_cat

    

    Age_predicted = pd.concat([woman_predicted, man_predicted] , axis = 0)

    

    return Age_predicted.sort_index()



def Age_predict(data):

    Age_data = data[data.Age.isnull()==False]

    Missing_data = data[data.Age.isnull()==True].drop(['Age'], axis = 'columns')

    

    x_data = Age_data.drop(['Age'], axis='columns')

    y_data = Age_data.Age

    

    

    rfr = RandomForestRegressor(n_estimators=1000 , random_state = 0)

    rfr.fit(x_data, y_data)

    Missing_data['Age'] = rfr.predict(Missing_data)

    

    data_predict = pd.concat([Missing_data, Age_data], axis = 'index', sort=True)

    

    return data_predict

# Deal with missing data from train data and also test data.

data_processed_ = Age_processing(data_processed)

test_processed_ = Age_processing(test_processed)



data_Age_enc = pd.DataFrame(one_hot_encoder.fit_transform(data_processed_[['Age']]).toarray(),

                           index = data_processed_.index)

train_data = pd.concat([data_processed_,data_Age_enc], axis=1).drop(['Age'], axis=1)



test_Age_enc = pd.DataFrame(one_hot_encoder.fit_transform(test_processed_[['Age']]).toarray(),

                           index = test_processed_.index)

test_data = pd.concat([test_processed_, test_Age_enc], axis=1).drop(['Age'], axis='columns')
plt.figure(figsize=(9,7))

sns.heatmap(train_data[['Survived',0,1,2,3,4,5,6,7,8]].corr(),

            annot=True,center=0,cmap='BrBG_r')

drop_feature = ['Survived','SibSp','Parch']

from sklearn.model_selection import train_test_split as tts

x_train, x_test, y_train, y_test = tts(train_data.drop(drop_feature, 

                                                       axis = 'columns'),

                                       train_data.Survived, random_state=0, test_size=0.2)



from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(x_train,y_train)
from sklearn.model_selection import cross_validate

cross_validate(LR, x_test, y_test, cv=5)['test_score'].mean()
my_model = LogisticRegression()



x_train_data = train_data.drop(['Survived','SibSp','Parch'], axis = 'columns')

y_train_data = train_data.Survived

my_model.fit(x_train_data, y_train_data)

prediction = my_model.predict(test_data.drop(['SibSp','Parch'], axis = 'columns'))

my_submission = pd.DataFrame(prediction, index=test_data.index, columns=['Survived'])

my_submission.to_csv('submission.csv')
