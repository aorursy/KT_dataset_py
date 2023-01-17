
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
my_filepath = "../input/nedvijimost/Nedvijimost.csv"
df = pd.read_csv(my_filepath)
df.head()
df_info = df.describe()
for column in df.columns:
    if df[column].count() < len(df):
        df[column] = df[column].fillna(df_info[column]['mean'])
df.describe()
from sklearn.preprocessing import LabelEncoder

s = (df.dtypes == 'object')
cat_cols = list(s[s].index)

label_encoder = LabelEncoder()
for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])
from sklearn.model_selection import train_test_split
cols = ['Район', 'Тип планировки', 'Количество комнат', 'Первый/Последний этаж', 'Общая площадь (м2)', 'Жилая площадь (м2)',
       'Площадь кухни (м2)', 'Наличие агенства', 'Состояние']
X = df[cols]
y = df['Стоимость (т.руб.)']
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=0)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import sys


best_mae = sys.float_info.max
best_model = None
best_leaf_count = 0

for i in range(10, 100, 10):
    regressor = DecisionTreeRegressor(max_leaf_nodes=i, random_state=1)
    regressor.fit(X_train, y_train)
    mae = mean_absolute_error(regressor.predict(X_valid), y_valid)
    if mae <= best_mae:
        best_model = regressor
        best_leaf_count = i

print(best_mae)
print(best_leaf_count)