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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
import os

from sklearn.metrics import mean_squared_error as mse
from sklearn.neural_network import MLPClassifier
from pickle import load
with open("/kaggle/input/faces.p", "rb") as f: # выгружаем данные из txt файла faces.p
    data_faces = load(f)
    
with open("/kaggle/input/features.p", "rb") as f: # выгружаем данные из txt файла features.p
    data_features = load(f)

# создаем общую дату, обьединяя content двух дат, делаем проверку на имя, если далее в проекте будут не нормализированные данные 
data = [ 
{"name":face['name'],
"is_bearing":face['is_bearing'],
"content":{**face['content'],**feat['content']}} for face,feat in zip(data_faces,data_features) if face['name']==feat['name']]

content = [content.get("content") for content in data] # создаем массив массивов ключевых данных
is_bearing = [is_bearing.get('is_bearing') for is_bearing in data] # массив определений подшипников (является соответственным с массивом content)

def parameters(content): # определяем все возможные параметры подшипников и загружаем их в массив
    parameters_mass = []
    
    for dictionary in content:
        
        keys = dictionary.keys()
        
        for key in keys:         
            if key in parameters_mass:
                pass
            else:
                parameters_mass.append(key)
                
        
    parameters_mass.sort()
    
    return parameters_mass
        
parameters_mass = parameters(content)

def y_filling(data):  
    
    y_train = []
    y_test = []
    
    for i in data: # наполняем тренировачный массив игриков
        if i:
            y_train.append(1)
        else:
            y_train.append(0)
            
    y_test = y_train[:10] + y_train[-10:] # наполняем тестировачный массив игриков. так как количество данных не так много, 
                                          # тестовые игрики копируются из тренировачных, а не вырезаются
    
    return y_train,y_test

def x_filling(data,parameters_mass):
    
    train = []
    x_train = []
    x_test = []
    
    for dictionary in data: # наполняем тренировачный массив иксов, проходя по всем возможным параметрам из массива параметров
        for param in parameters_mass:
            train.append(dictionary.get(param,0)) 

        x_train.append(train)

        train = []
            
    x_test = x_train[:10] + x_train[-10:] # наполняем тестировачный массив иксов. так как количество данных не так много, 
                                          # тестовые иксы копируются из тренировачных, а не вырезаются
    return x_train,x_test
        



x_train,x_test = x_filling(content,parameters_mass) # создаем тренировачные и тестовые иксы
y_train,y_test = y_filling(is_bearing) # создаем тренировачные и тестовые игрики

mass_of_mass=[]
iterations = 50

for _ in range(iterations): 
    model = MLPClassifier(hidden_layer_sizes=(5,),max_iter=10000) # создание модели
    model.fit(x_train, y_train)        # тренировка модели
    mass_of_mass.append(model.predict(x_test))    # предсказание подшипника, загружаемое в массив

y_pred = list(map(lambda x: sum(x)/iterations, zip(*mass_of_mass))) # итоговые предсказанные игрики

print(y_test)
print(y_pred)
print(mse(y_test,y_pred)) # средне квадратичная ошибка
answer=[]

for i in y_pred:
    if i > 0.5:
        answer.append('Подшипник есть')
    else:
        answer.append('Подшипника нет')
        
print(answer)
    