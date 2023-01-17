# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
gender_submissions_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
gender_submissions_data
train_data
test_data
train_data['Ticket'].value_counts()
train_data['Cabin'].value_counts()
train_data['Pclass'].value_counts()
print('Percentage of null values per column')
for col in train_data.columns:
    print(col, ' ', train_data[col].isnull().sum() / train_data.shape[0])
train_data_drop = train_data.drop(['Cabin'], axis = 1)
train_data_drop.shape
age_sex_id = pd.DataFrame(train_data[['Survived', 'Pclass', 'PassengerId', 'Age', 'Sex']])
fem = age_sex_id[age_sex_id.Sex != 'female']
mal = age_sex_id[age_sex_id.Sex != 'male']
fem_null = fem.Age.isnull().sum()
mal_null = mal.Age.isnull().sum()
print('fem null vals count: ', fem_null, ', mal null vals count: ', mal_null)
total_fem_age = age_sex_id[age_sex_id.Sex == 'female'].sum().Age
total_mal_age = age_sex_id[age_sex_id.Sex == 'male'].sum().Age
total_fem_rows = age_sex_id[age_sex_id.Sex == 'female'].shape[0]
total_mal_rows = age_sex_id[age_sex_id.Sex == 'male'].shape[0]
print('total fem age: ', total_fem_age, ', total rows: ', total_fem_rows)
print('total mal age: ', total_mal_age, ', total rows: ', total_mal_rows)
avg_fem_age = total_fem_age / (total_fem_rows - fem_null)
avg_mal_age = total_mal_age / (total_mal_rows - mal_null)
print('avg fem age: ', avg_fem_age, ', avg mal age: ',avg_mal_age)
print('Average age for both genders')
(total_fem_age + total_mal_age) / (train_data.shape[0] - (fem_null + mal_null))
fem_surv = age_sex_id[age_sex_id.Sex == 'female']
fem_surv = fem_surv[fem_surv.Age > -1]
fem_surv.isnull().sum(), fem_surv.shape
mal_surv = age_sex_id[age_sex_id.Sex == 'male']
mal_surv = mal_surv[mal_surv.Age > -1]
mal_surv.isnull().sum(), mal_surv.shape
femc1, femc2, femc3 = fem_surv[fem_surv.Pclass == 1], fem_surv[fem_surv.Pclass == 2], fem_surv[fem_surv.Pclass == 3]
malc1, malc2, malc3 = mal_surv[mal_surv.Pclass == 1], mal_surv[mal_surv.Pclass == 2], mal_surv[mal_surv.Pclass == 3]
femc1_avg, femc2_avg, femc3_avg = (femc1.Age.sum() / femc1.shape[0]), (femc2.Age.sum() / femc2.shape[0]), (femc3.Age.sum() / femc3.shape[0])
malc1_avg, malc2_avg, malc3_avg = (malc1.Age.sum() / malc1.shape[0]), (malc2.Age.sum() / malc2.shape[0]), (malc3.Age.sum() / malc3.shape[0])
print('fem avg age, class 1: ', femc1_avg, ', class 2: ', femc2_avg, ', class 3: ', femc3_avg, '\n')
print('mal avg age, class 1: ', malc1_avg, ', class 2: ', malc2_avg, ', class 3: ', malc3_avg)
fem_avg = (femc1_avg + femc2_avg + femc3_avg) / 3
mal_avg = (malc1_avg + malc2_avg + malc3_avg) / 3
age_avg = (fem_avg + mal_avg) / 2
fem_avg, mal_avg, age_avg
null_age_rows = train_data_drop.loc[train_data_drop.Age.isnull()]
null_age_rows
null_age_rows['Survived'].value_counts()
null_age_rows.Age.sum()
train_data_avg_ages = train_data_drop
train_data_avg_ages.loc[(train_data_avg_ages.Age.isnull()) & (train_data_avg_ages.Sex == 'female') & (train_data_avg_ages.Pclass == 1), 'Age'] = femc1_avg
train_data_avg_ages.loc[(train_data_avg_ages.Age.isnull()) & (train_data_avg_ages.Sex == 'female') & (train_data_avg_ages.Pclass == 2), 'Age'] = femc2_avg
train_data_avg_ages.loc[(train_data_avg_ages.Age.isnull()) & (train_data_avg_ages.Sex == 'female') & (train_data_avg_ages.Pclass == 3), 'Age'] = femc3_avg
train_data_avg_ages.loc[(train_data_avg_ages.Age.isnull()) & (train_data_avg_ages.Sex == 'male') & (train_data_avg_ages.Pclass == 1), 'Age'] = malc1_avg
train_data_avg_ages.loc[(train_data_avg_ages.Age.isnull()) & (train_data_avg_ages.Sex == 'male') & (train_data_avg_ages.Pclass == 2), 'Age'] = malc2_avg
train_data_avg_ages.loc[(train_data_avg_ages.Age.isnull()) & (train_data_avg_ages.Sex == 'male') & (train_data_avg_ages.Pclass == 3), 'Age'] = malc3_avg
train_data_avg_ages
train_data_avg_ages.isnull().sum()
train_data_avg_ages['Embarked'].value_counts()
train_data_avg_ages['fam_members'] = train_data_avg_ages['SibSp'] + train_data_avg_ages['Parch']
train_data_avg_ages['Embarked'].fillna('S', inplace = True)
train_data_avg_ages
from sklearn.linear_model import LogisticRegression

log_x = train_data_avg_ages.drop(['Name', 'PassengerId', 'Survived', 'Ticket', 'Parch', 'SibSp'], axis = 1)
log_x['Embarked'] = log_x['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
log_x['Sex'] = log_x['Sex'].map({'female': 0, 'male': 1})
log_y = train_data_avg_ages['Survived']

log_reg = LogisticRegression().fit(log_x, log_y)
#log_predict = log_reg.predict(log_x)
#log_predict_proba = log_reg.predict_proba(log_predict, log_y)
log_reg.score(log_x, log_y)
log_reg.coef_
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler

svm_x_scaled = MinMaxScaler().fit_transform(log_x.values)
svm_y = log_y.values

svm_model = svm.SVC().fit(svm_x_scaled, svm_y)
svm_predict = svm_model.predict(svm_x_scaled)
svm_predict_score = svm_model.score(svm_x_scaled, svm_y)
svm_predict_score
from tensorflow.keras import layers, Sequential

hidden = 200
nn = Sequential()
nn.add(layers.Dense(hidden, activation = 'relu', input_shape = (10,)))
nn.add(layers.Dense(hidden, activation = 'relu'))
nn.add(layers.Dense(hidden, activation = 'relu'))
nn.add(layers.Dense(hidden, activation = 'relu'))
nn.add(layers.BatchNormalization())
nn.add(layers.Dense(hidden, activation = 'relu'))
nn.add(layers.Dense(hidden, activation = 'relu'))
nn.add(layers.Dense(hidden, activation = 'relu'))
nn.add(layers.BatchNormalization())
nn.add(layers.Dense(2, activation = 'softmax'))
    
nn.compile(optimizer = 'nadam', loss = 'binary_crossentropy', metrics = ['accuracy'])
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

nn_x = log_x
#nn_x['Embarked'].map({0: 'S', 1: 'C', 2: 'Q'})
sex_dummies = pd.get_dummies(nn_x.Sex)
sex_dummies.columns = ['female', 'male']
class_dummies = pd.get_dummies(nn_x.Pclass)
class_dummies.columns = ['first_cl', 'second_cl', 'third_cl']
embark_dummies = pd.get_dummies(nn_x.Embarked)
embark_dummies.columns = ['S', 'C', 'Q']
nn_x = nn_x.drop(['Pclass', 'Sex', 'Embarked'], axis = 1)
nn_x_dummied = pd.concat([nn_x, sex_dummies, class_dummies, embark_dummies], axis = 1)

nn_x_scaled = MinMaxScaler().fit_transform(nn_x_dummied.drop(['male'], axis = 1).values)

callbacks = [EarlyStopping(patience = 0)]
nn_y = to_categorical(svm_y)
nn_x_dummied
nn_fitted = nn.fit(nn_x_scaled, nn_y, epochs = 200, validation_split = 0.15, callbacks = callbacks)
nn_fitted
x_test_data = test_data.drop(['Cabin'], axis = 1)
x_test_data.isnull().sum()
x_test_data.loc[(x_test_data.Age.isnull()) & (x_test_data.Sex == 'female') & (x_test_data.Pclass == 1), 'Age'] = femc1_avg
x_test_data.loc[(x_test_data.Age.isnull()) & (x_test_data.Sex == 'female') & (x_test_data.Pclass == 2), 'Age'] = femc2_avg
x_test_data.loc[(x_test_data.Age.isnull()) & (x_test_data.Sex == 'female') & (x_test_data.Pclass == 3), 'Age'] = femc3_avg
x_test_data.loc[(x_test_data.Age.isnull()) & (x_test_data.Sex == 'male') & (x_test_data.Pclass == 1), 'Age'] = malc1_avg
x_test_data.loc[(x_test_data.Age.isnull()) & (x_test_data.Sex == 'male') & (x_test_data.Pclass == 2), 'Age'] = malc2_avg
x_test_data.loc[(x_test_data.Age.isnull()) & (x_test_data.Sex == 'male') & (x_test_data.Pclass == 3), 'Age'] = malc3_avg
x_test_data
x_test_data.loc[x_test_data.Fare.isnull()]
avg_fare = nn_x_dummied.loc[(nn_x_dummied.third_cl == 1)].Fare.sum() / nn_x_dummied.shape[0]
x_test_data.loc[x_test_data.Fare.isnull(), 'Fare'] = avg_fare
x_test_data.loc[152, :]
x_test_data.isnull().sum()
x_test_data['Sex'] = x_test_data['Sex'].map({'female': 0, 'male': 1})
x_test_data['fam_members'] = x_test_data['SibSp'] + x_test_data['Parch']
x_test_data = x_test_data.drop(['SibSp', 'Parch'], axis = 1)
nn_x_dummied.columns
class_dummies_ = pd.get_dummies(x_test_data.Pclass)
#class_dummies.columns = ['first_cl', 'second_cl', 'third_cl']
embark_dummies_ = pd.get_dummies(x_test_data.Embarked)
#embark_dummies.columns = ['S', 'C', 'Q']
nn_x_test = x_test_data.drop(['PassengerId', 'Ticket', 'Pclass', 'Embarked', 'Name'], axis = 1)
nn_x_test_dummied = pd.concat([nn_x_test.drop(['Sex'], axis = 1), nn_x_test['Sex'], class_dummies_, embark_dummies_], axis = 1)

nn_x_test_scaled = MinMaxScaler().fit_transform(nn_x_test_dummied.values)
nn_x_test_dummied
nn_prediction = nn.predict(nn_x_test_scaled)
nn_prediction
survived_list = list()
for passenger in nn_prediction:
    if passenger[0] >= passenger[1]:
        survived_list.append(0)
    else:
        survived_list.append(1)
survived_list
nn_pred_df = pd.DataFrame(test_data.PassengerId)
nn_pred_df['Survived'] = survived_list
nn_pred_df
nn_pred_df.to_csv('/kaggle/working/submission_.csv', index = False)

