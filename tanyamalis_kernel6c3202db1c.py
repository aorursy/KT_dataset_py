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
import pandas as pd

from collections import defaultdict

import seaborn as sns

import matplotlib.pyplot as plt

my_filepath = "../input/steamt/steam.csv"

steam = pd.read_csv(my_filepath)
steam_action = steam[steam.genres == 'Action']

steam_action
print(steam_action.groupby(['developer']).size())
plt.figure(figsize=(17,10))

plt.title("Кол-во скачиваний и цены на игру")

sns.barplot(x=steam_action['price'], y=steam_action['owners'])

plt.show()

steam_action.to_csv("output.csv", index=True)
new_filepath = '../input/nedvijimost/Nedvijimost.csv'

model = pd.read_csv(new_filepath)

model.describe()
import pandas as pd

model.isnull().sum()

model = model.fillna(model.mean())



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

s = (model.dtypes == 'object')

object_cols = list(s[s].index)

print("Кат. атрибуты:")

print(object_cols)



label_model = model.copy()

label_encoder = LabelEncoder()

for col in object_cols:

    label_model[col] = label_encoder.fit_transform(model[col])

label_model
features_1 = ['Район','Тип планировки','Количество комнат','Общая площадь (м2)']

X = label_model[features_1]

y = label_model['Стоимость (т.руб.)']

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

rf_model = RandomForestRegressor(random_state = 1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

val_X.reset_index(inplace = True)



output_1 = pd.DataFrame({'Район':val_X['Район'],'Тип планировки':val_X['Тип планировки'],'Количество комнат':val_X['Количество комнат'],'Общая площадь (м2)':val_X['Общая площадь (м2)'],'Стоимость (т.руб.)':rf_val_predictions})

print(rf_val_mae)

print(output_1)
features_2 = ['Район','Тип планировки','Первый/Последний этаж','Количество комнат','Общая площадь (м2)']

X = label_model[features_2]

y = label_model['Стоимость (т.руб.)']

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

rf_model = RandomForestRegressor(random_state = 1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

val_X.reset_index(inplace = True)



output_2 = pd.DataFrame({'Район':val_X['Район'],'Тип планировки':val_X['Тип планировки'],'Первый/Последний этаж':val_X['Первый/Последний этаж'],'Количество комнат':val_X['Количество комнат'],'Общая площадь (м2)':val_X['Общая площадь (м2)'],'Стоимость (т.руб.)':rf_val_predictions})

print(rf_val_mae)

print(output_2)
features_3 = ['Район','Состояние','Тип планировки','Первый/Последний этаж','Количество комнат','Общая площадь (м2)']

X = label_model[features_3]

y = label_model['Стоимость (т.руб.)']

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

rf_model = RandomForestRegressor(random_state = 1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

val_X.reset_index(inplace = True)



output_3 = pd.DataFrame({'Район':val_X['Район'],'Состояние':val_X['Состояние'],'Тип планировки':val_X['Тип планировки'],'Первый/Последний этаж':val_X['Первый/Последний этаж'],'Количество комнат':val_X['Количество комнат'],'Общая площадь (м2)':val_X['Общая площадь (м2)'],'Стоимость (т.руб.)':rf_val_predictions})

print(rf_val_mae)

print(output_3)
output_3.to_csv('best_model.csv', index = False)