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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import ensemble

from sklearn import datasets

from sklearn.utils import shuffle

from sklearn.metrics import mean_squared_error

import string

import warnings

warnings.filterwarnings(action='ignore')
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor

from sklearn.svm import SVR

from lightgbm import LGBMRegressor

from sklearn.linear_model import ElasticNet, Lasso
def y_predict_mean(y_predicts):

    y_predicts_sum = 0

    y_predicts_cnt = 0

    for a in y_predicts:

        y_predicts_sum = y_predicts_sum + a

    for a in y_predicts:

        y_predicts_cnt = y_predicts_cnt + 1

    y_predict_mean = y_predicts_sum/y_predicts_cnt

    return y_predict_mean
def fighter_into_data(fighter_name):

    fighter_info = []

    fighter_infos = data[data['fighter']==fighter_name].ix[0]

    fighter_infos = fighter_infos.drop(columns='fighter')

    for row_data in fighter_infos:

        if row_data != fighter_name:

            fighter_info.append(row_data)

    return fighter_info

fighter_name = 'Henry Cejudo'
data = pd.read_csv('../input/ufcdata/data.csv')
data['R_winrate'] = (data['R_wins']/(data['R_wins'] + data['R_draw'] + data['R_losses'])*100)

data['B_winrate'] = (data['B_wins']/(data['B_wins'] + data['B_draw'] + data['B_losses'])*100)

data = data.dropna()
R_list = []

B_list = []
for columns in data:

    if columns.startswith('R_'):

        R_list.append(columns)

    elif columns.startswith('B_'):

        B_list.append(columns)

R_data = data[R_list]

B_data = data[B_list]
common_list = ['Referee','date', 'location', 'weight_class', 'no_of_rounds']

R_data[common_list] = data[common_list].copy()

B_data[common_list] = data[common_list].copy()
for columns in R_data:

    if columns.startswith('R_'):

        R_data.rename(columns={columns : columns[2:]}, inplace=True)

for columns in B_data:

    if columns.startswith('B_'):

        B_data.rename(columns={columns : columns[2:]}, inplace=True)



R_data['corner'] = 'Red'

B_data['corner'] = 'Blue'
data = pd.concat([R_data, B_data])
data.sort_values(by='fighter',ascending=True)

data = data.set_index('fighter')
data = data.drop(columns='draw')
data.to_csv("../input/newdata.csv")

data = pd.read_csv("../input/newdata.csv")
plt.rcParams['figure.figsize'] = (15,5)

plt.rcParams["font.family"]='NanumGothic'

df_aux = data[['fighter','age']].copy()

df_aux.drop_duplicates(subset='fighter', keep='first', inplace=True)

sns.distplot(df_aux['age'], bins=10)

plt.title('Age Distribution by Fighters')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
print(df_aux.groupby('age').max().tail(1))
print(df_aux.groupby('age').min().head(1))
data = data[data['weight_class']!='Catch Weight']
data['weight_class'][data['weight_class']=='Flyweight'] = 10

data['weight_class'][data['weight_class']=='Bantamweight'] = 11

data['weight_class'][data['weight_class']=='Featherweight'] = 12

data['weight_class'][data['weight_class']=='Lightweight'] = 13

data['weight_class'][data['weight_class']=='Welterweight'] = 14

data['weight_class'][data['weight_class']=='Middleweight'] = 15

data['weight_class'][data['weight_class']=='Light Heavyweight'] = 16

data['weight_class'][data['weight_class']=='Heavyweight'] = 17
data['weight_class'][data['weight_class']=="Women's Strawweight"] = 20

data['weight_class'][data['weight_class']=="Women's Flyweight"] = 21

data['weight_class'][data['weight_class']=="Women's Bantamweight"] = 22

data['weight_class'][data['weight_class']=="Women's Featherweight"] = 23
data['Stance'][data['Stance']=='Orthodox'] = 1

data['Stance'][data['Stance']=='Southpaw'] = 2

data['Stance'][data['Stance']=='Switch'] = 3

data = data[data['Stance']!='Open Stance']
data['weight_class'].astype('int64')

data['Stance'].astype('int64')

print(data.weight_class.drop_duplicates())

print(data.weight_class.dtype)
data.sort_values(by='fighter',ascending=True)

data = data.set_index('fighter')

data.to_csv("../input/newdata2.csv")

data = pd.read_csv("../input/newdata2.csv")
from sklearn.model_selection import train_test_split

target = data['winrate'].copy()

data = data.drop(columns=['winrate','date','location','Referee','corner'])

fighter_data = data.drop(columns='fighter')
print(f'The dataset have {data.shape[0]} features now.')

print(f'The dataset have {fighter_data.shape[0]} features now.')

print(f'The target dataset have {target.shape[0]} features now.')
x_train, x_test, y_train, y_test = train_test_split(fighter_data, target,test_size=0.3, random_state=42)
print(f'Input Train Shape {x_train.shape}')

print(f'Output Train Shape {y_train.shape}')

print(f'Input Test Shape {x_test.shape}')

print(f'Output Test Shape {y_test.shape}')
print('x_train null count : ',x_train.isnull().sum().sum())

print('x_test null count : ',x_test.isnull().sum().sum())
print(data.dtypes)

print(data.fighter.drop_duplicates())
print(fighter_into_data(fighter_name))
models = []

y_predicts = []
model = RandomForestRegressor()

model.fit(x_train,y_train)

print("train data : ",model.score(x_train,y_train))

print("test data : ",model.score(x_test,y_test))

models.append(model)
y_predict = model.predict([fighter_into_data(fighter_name)])

print(y_predict)

y_predicts.append(y_predict[0])
model = AdaBoostRegressor()

model.fit(x_train,y_train)

print("train data : ",model.score(x_train,y_train))

print("test data : ",model.score(x_test,y_test))

models.append(model)
y_predict = model.predict([fighter_into_data(fighter_name)])

print(y_predict)

y_predicts.append(y_predict[0])
model = GradientBoostingRegressor()

model.fit(x_train,y_train)

print("train data : ",model.score(x_train,y_train))

print("test data : ",model.score(x_test,y_test))

models.append(model)
y_predict = model.predict([fighter_into_data(fighter_name)])

print(y_predict)

y_predicts.append(y_predict[0])
model = KNeighborsRegressor()

model.fit(x_train,y_train)

print("train data : ",model.score(x_train,y_train))

print("test data : ",model.score(x_test,y_test))

models.append(model)
y_predict = model.predict([fighter_into_data(fighter_name)])

print(y_predict)

y_predicts.append(y_predict[0])
model = RadiusNeighborsRegressor()

model.fit(x_train,y_train)

print("train data : ",model.score(x_train,y_train))

# print("test data : ",model.score(x_test,y_test)) ## float64 error

models.append(model)
y_predict = model.predict([fighter_into_data(fighter_name)])

print(y_predict)

y_predicts.append(y_predict[0])
model = DecisionTreeRegressor()

model.fit(x_train,y_train)

print("train data : ",model.score(x_train,y_train))

print("test data : ",model.score(x_test,y_test))

models.append(model)
y_predict = model.predict([fighter_into_data(fighter_name)])

print(y_predict)

y_predicts.append(y_predict[0])
model = ExtraTreeRegressor()

model.fit(x_train,y_train)

print("train data : ",model.score(x_train,y_train))

print("test data : ",model.score(x_test,y_test))

models.append(model)
y_predict = model.predict([fighter_into_data(fighter_name)])

print(y_predict)

y_predicts.append(y_predict[0])
model = SVR()

model.fit(x_train,y_train)

print("train data : ",model.score(x_train,y_train))

print("test data : ",model.score(x_test,y_test))

models.append(model)
y_predict = model.predict([fighter_into_data(fighter_name)])

print(y_predict)

y_predicts.append(y_predict[0])
model = LGBMRegressor()

model.fit(x_train,y_train)

print("train data : ",model.score(x_train,y_train))

print("test data : ",model.score(x_test,y_test))

models.append(model)
y_predict = model.predict([fighter_into_data(fighter_name)])

print(y_predict)

y_predicts.append(y_predict[0])
model = ElasticNet()

model.fit(x_train,y_train)

print("train data : ",model.score(x_train,y_train))

print("test data : ",model.score(x_test,y_test))

models.append(model)
y_predict = model.predict([fighter_into_data(fighter_name)])

print(y_predict)

y_predicts.append(y_predict[0])
model = Lasso()

model.fit(x_train,y_train)

print("train data : ",model.score(x_train,y_train))

print("test data : ",model.score(x_test,y_test))

models.append(model)
y_predict = model.predict([fighter_into_data(fighter_name)])

print(y_predict)

y_predicts.append(y_predict[0])
y_predicts
y_predict_mean(y_predicts)
models