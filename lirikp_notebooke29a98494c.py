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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style


import json

import datetime # для работы со времнем


from sklearn.metrics import mean_squared_error as mse, r2_score as r2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

%matplotlib inline

TEST_DATASET_FILE = '../input/ds-prise-predict/test.csv'
TRAIN_DATASET_FILE = '../input/ds-prise-predict/train.csv'

test_all_data = pd.read_csv(TEST_DATASET_FILE)
train_data = pd.read_csv(TRAIN_DATASET_FILE)

preds_final = pd.DataFrame()
preds_final['Id'] = test_all_data['Id'].copy()

def evaluate_preds(true_values, pred_values, save=False):
    """Оценка качества модели и график preds vs true"""
    
#    print("R2:\t" + str(round(r2(true_values, pred_values), 3)) + "\n" +
#          "RMSE:\t" + str(round(np.sqrt(mse(true_values, pred_values)), 3)) + "\n" +
#          "MSE:\t" + str(round(mse(true_values, pred_values), 3))
#         )
    
    plt.figure(figsize=(8,8))
    
    sns.scatterplot(x=pred_values, y=true_values.Price)
    plt.plot([0, 500000], [0, 500000], linestyle='--', color='black')  # диагональ, где true_values = pred_values
    
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.title('True vs Predicted values')
    
    if save == True:
        plt.savefig(REPORTS_FILE_PATH + 'report.png')
    plt.show()
    
def prepare_usage(data):
    # Пробуем заполнить отсутствующие данные по жилой площади 

    # Согласно СНиП 31-01-2003 5.7 В квартирах, указанных ... кухни или кухни-ниши площадью не менее 5
    # Согласно действующим СНиПам, минимальная площадь комнаты должна быть не менее 8 квадратных метров (п. 5.3 СП 5.413330.2011)
    data.loc[(data['KitchenSquare'] < 5), ['KitchenSquare']] = 5
    data.loc[(data['KitchenSquare'] > 50), ['KitchenSquare']] = 50
    data.loc[(data['Square'] < 15), ['Square']] = 15
    data.loc[data.LifeSquare.isnull(), ['LifeSquare']] = (
                data['Square'] - data['KitchenSquare'])
    data.loc[(data['LifeSquare'] > 700), ['LifeSquare']] = data['Square']
    data.loc[(data['LifeSquare'] > data['Square']), ['LifeSquare']] = data['Square']
    data.loc[(data['LifeSquare'] < 8) & (
                data['LifeSquare'] < (data['Square'] - data['KitchenSquare'])), [
                       'LifeSquare']] = (data['Square'] - data['KitchenSquare'])

    # Правим неверно вбитые даты постройки
    data.loc[data['HouseYear'] == 4968, 'HouseYear'] = 1968
    data.loc[data['HouseYear'] == 20052011, 'HouseYear'] = 2005
    # Разбираемся с этажами
    data.loc[data.HouseFloor <= 0, ['HouseFloor']] = 1
    data.loc[(data.Floor == 0) | (data.Floor > data.HouseFloor), ['Floor']] = data[
        'HouseFloor']
    # Переводим экологию на цифровые рельсы
    data.loc[data['Ecology_2'] == 'A', 'Ecology_2'] = 1
    data.loc[data['Ecology_2'] == 'B', 'Ecology_2'] = 2
    data.loc[data['Ecology_3'] == 'A', 'Ecology_3'] = 1
    data.loc[data['Ecology_3'] == 'B', 'Ecology_3'] = 2
    # Переводим магазины на цифровые рельсы
    data.loc[data['Shops_2'] == 'A', 'Shops_2'] = 1
    data.loc[data['Shops_2'] == 'B', 'Shops_2'] = 2

    return data
def plot_f_importances(importances, X):
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(20, 6))
    plt.title("Feature importances", fontsize=16)
    plt.bar(range(X.shape[1]), importances[indices] / importances.sum(),
            color="darkblue", align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90, fontsize=14)
    plt.xlim([-1, X.shape[1]])

    plt.tight_layout()
    # plt.savefig('fe.jpg')
    plt.show()



test_all_data.head()
#Удаляем колонку с ненужными данными
train_data = train_data.drop(columns=['Id'])
train_data = train_data.drop(columns=['Healthcare_1'])
test_all_data = test_all_data.drop(columns=['Id'])
test_all_data = test_all_data.drop(columns=['Healthcare_1'])


train_data.head(7)
train_data.dtypes
train_data.count()
LIST_LIB_FILE = './list_lib_version.txt'
with open(LIST_LIB_FILE, 'w') as lib_listfile:
    pipe = os.popen('pip freeze')
    str = pipe.read()
    lib_listfile.write(str)
    

train_data = prepare_usage(train_data)

train_data.head(5)



features = ['DistrictId', 'Rooms', 'Square', 'LifeSquare', 'KitchenSquare', 'Floor', 'HouseFloor', 'HouseYear',
            'Ecology_1', 'Ecology_2', 'Ecology_3', 'Social_1', 'Social_2', 'Social_3', 'Helthcare_2', 'Shops_1',
            'Shops_2']
target = train_data[['Price']]


X = pd.DataFrame(train_data, columns=features)

y = pd.DataFrame(target, columns=['Price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)

rfr = RandomForestRegressor()

rfr.fit(X_train, y_train.values[:, 0])
RandomForestRegressor(n_estimators=1500, max_depth=14, random_state=666)

y_pred = rfr.predict(X_test)
check_test = pd.DataFrame({'y_test': y_test['Price'], 'y_pred': y_pred.flatten()}, columns=['y_test', 'y_pred'])

print(r2(y_test, y_pred))
evaluate_preds(y_test, y_pred)
plot_f_importances(importances = rfr.feature_importances_, X=X)
test_all_data.head(5)
test_all_data = prepare_usage(test_all_data)

test_all_data.head(5)
y_pred_final = rfr.predict(test_all_data)
# test DATA Learn


preds_final['Price'] = y_pred_final
preds_final.to_csv('predictions.csv', index=False)

preds_final.head(10)