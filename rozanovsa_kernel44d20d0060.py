import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings; warnings.simplefilter('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Загружаем модуль для разделения датасета:
from sklearn.model_selection import train_test_split

# Загружаем соответствующие методы нормализации:
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Загружаем метод создания полиномаиальных признаков:
from sklearn.preprocessing import PolynomialFeatures

# Загружаем модули для работы с датой и временем
from datetime import datetime
from datetime import timedelta

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression


#загружаем датасет(это пробный датасет части потребителей по 2018 году )
data = pd.read_excel('/kaggle/input/DataSet_Cut_2018.xlsx') 
#размер данных
data.shape
data.sample(5)
data.info()
data.isna().sum().sort_values()
for col in data.columns: 
    if (len(data[col].unique()) < 13): #условие не информативности - количество уникальниых значений меньше 12
        if(col != 'holiday'):
            data = data.drop(col, axis = 'columns')
data['dayofweek'] = data['date'].dt.dayofweek
data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month
data['season'] = data['month'].apply(lambda x: 1 if x > 9 else 1 if x < 5 else 2)
label_encoder = LabelEncoder()
data['num_holiday'] = pd.Series(label_encoder.fit_transform(data['holiday']))
data.shape
#data.iloc[:,:-16].hist(bins = 25)
#data1.drop(['holiday','temp','date','month','dayofweek','day'], inplace = True, axis = 1) # удалим не информативные столбцы
correlation = data.corr()
plt.rcParams['figure.figsize'] = (25,15)
sns.heatmap(correlation, annot = True, cmap = 'coolwarm')
def get_boxplot(col):
    fig, ax = plt.subplots(figsize = (20, 14))
    sns.boxplot(y= col, data = data, x = 'dayofweek', hue = 'season')# используем доп признак - отопительный сезон
    plt.xticks(rotation=45)
    ax.set_title('Boxplot for ' + col)
    plt.show()
for col in ['ООО ЛУКОЙЛ - Нижегороднефтеоргсинтез, ИНН: 5250043567','ОАО Теплоэнерго г.Н.Новгорода, ИНН: 5257087027']:
    get_boxplot(col)
def get_boxplot(col):
    fig, ax = plt.subplots(figsize = (20, 14))
    sns.boxplot(y= col, data = data, x = 'dayofweek')# используем доп признак - день недели
    plt.xticks(rotation=45)
    ax.set_title('Boxplot for ' + col)
    plt.show()
for col in ['ООО ЛУКОЙЛ - Нижегороднефтеоргсинтез, ИНН: 5250043567','ОАО Теплоэнерго г.Н.Новгорода, ИНН: 5257087027']:
    get_boxplot(col)
#удаляем нечисловые перменные и признак с температурой, так как с ним наблюдается высокая корреляция 
data['temper'] = data['temp']
data.drop(['holiday','temp','date'], inplace = True, axis = 1)
#создадим массив для зависимой переменной
Y = data.iloc[:,:-5].values
#создадим массив для переменных-признаков]
X = data.iloc[:,23:].values


#разобьем выборку на тестовую и обучаемую(70/30)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3)
# обучим модель
from sklearn.linear_model import LinearRegression
myModel = LinearRegression() #Обозначаем, что наша модель - линейная регрессия
myModel.fit(X_train,Y_train) #обучаем модель на обучающих данных
#предскажем значения расходов газа на тестовой выборке
y_pred = myModel.predict(X_test)
y_pred
#оценим ошибку предсказанной величины
from sklearn import metrics  # подгружаем метрики
print('MSE:', metrics.mean_squared_error(Y_test, y_pred),'MAE:',metrics.mean_absolute_error(Y_test, y_pred))
#вычислим коэффициент детерминации
R_2 = metrics.r2_score(Y_test, y_pred)
print(R_2)