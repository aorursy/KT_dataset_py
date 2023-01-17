import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

new_filepath = "../input/nedvijimost/Nedvijimost.csv"

NedvijData = pd.read_csv(new_filepath)

NedvijData.head()

NedvijData.isnull().sum()

NedvijData = NedvijData.fillna(NedvijData.mean())

NedvijData.describe()
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error
features = ['Район', 'Тип планировки', 'Количество комнат',  'Общая площадь (м2)', 'Жилая площадь (м2)', 'Площадь кухни (м2)', 'Состояние']

X = NedvijData[features]

y = NedvijData['Стоимость (т.руб.)']

X_train, X_valid, y_train, y_valid = train_test_split(X, y,random_state=1)
def get_mae(max_leaf_nodes, X_train, X_valid, y_train, y_valid):

    categorical_transformer.fit(X_train, y_train)

    preds_val = categorical_transformer.predict(X_valid)

    mae = mean_absolute_error(y_valid, preds_val)

    return(mae)
#Цикл, который пытается использовать значения для max_leaf_nodes из набора возможных значений.

#Вызов функции get_may для каждого значения max_leaf_nodes. 

#Храните выходные данные таким образом, чтобы ,была возможность выбрать значение max_leaf_nodes, которое дает наиболее точную модель данных.
#OneHotEncoder

#handle_unknown {'error', 'ignore'}, default = 'error'

#Вызывать ли ошибку или игнорировать, если во время преобразования присутствует неизвестная категориальная особенность (по умолчанию это error). Если для этого параметра задано 

#значение «игнорировать», и во время преобразования обнаруживается неизвестная категория, то получающиеся столбцы с горячим кодированием для этой функции будут иметь все нули. 

#В обратном преобразовании неизвестная категория будет обозначена как None.
#SimpleImputer

#Если “среднее", то замените пропущенные значения, используя среднее значение вдоль каждого столбца. 

#Может использоваться только с числовыми данными.
mae_dict = {}

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 350,450,500,550,600,650,700,750,800,850,900,950,1000,1200,1400,1600,1800,2000]

# print(candidate_max_leaf_nodes)

for i in range(0, len(candidate_max_leaf_nodes)):

    #кодирование категоральных признаков

    categorical_transformer = Pipeline(steps=[

    ('onehot', OneHotEncoder(handle_unknown='ignore')),

    ('imputer', SimpleImputer(strategy='mean')),

    ('model', DecisionTreeRegressor(max_leaf_nodes= candidate_max_leaf_nodes[i] , random_state=1))])

    #функция по поиску ошибки обучения модели

    mae = get_mae(candidate_max_leaf_nodes[i],X_train, X_valid, y_train, y_valid )

    mae_dict[candidate_max_leaf_nodes[i]] = mae    

mae_dict
#дерево с минимальной ошибкой

best_tree_size = sorted(  [(v,k) for k,v in mae_dict.items()]  )[0][1]

best_tree_size
best_model =  Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore')),

    ('model', DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1))

])



best_model.fit(X_train, y_train)

preds = best_model.predict(X_valid)

mean_absolute_error(y_valid, preds)