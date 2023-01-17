import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
data = pd.read_csv("../input/heart-disease-uci/heart.csv")
data.head()
y = data.target.values
x = data.drop(['target'], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
param_grid = {'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 21),
    'min_samples_split': range(1, 51),
    'min_samples_leaf': range(1, 51)}
rt = RandomForestClassifier(n_estimators = 100, random_state = 42)
rt_search = RandomizedSearchCV(rt, param_grid, cv = 5, n_jobs = -1)
rt_search.fit(x_train, y_train)
rt_search.best_params_
rt_search.best_score_
(rt_search.predict(x_train) == y_train).sum()/y_train.shape[0]
age = int(input("Сколько вам лет? "))
sex = int(input("Ваш пол (1 - мужской, 0 - женский): "))
cp = int(input("Тип боли в груди (0-3): "))
trestbps = int(input("Артериальное давление в состоянии покоя (в мм рт. ст.): "))
chol = int(input("Уровень хорестерола (в мл/дл): "))
fbs = int(input("Уровень сахара в крови натощак = 120 мг/дл (1 = истина; 0 = ложь): "))
restecg = int(input("Результат ЭКГ в покое (0-2): "))
thalach = int(input("Максимальная достигнутая частота сердечных сокращений: "))
exang = int(input("Стенокардия, вызванная физическими нагрузками (1 - да; 0 - нет): "))
oldpeak = float(input("Депрессия ST вызванная физической нагрузкой относительно покоя: "))
slope = int(input("Наклон пика упражнения ST сегмента (1-3): "))
ca = int(input("Количество крупных сосудов (0-3), окрашенных флуороскопией: "))
thal = int(input("Thal (3 - нормальный; 6 - исправленный дефект; 7 - обратимый дефект): "))

result = rt_search.predict(pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]))[0]

if result == 1:
    print("\nУ вас болезнь сердца")
else:
    print("\nУ вас нет болезни сердца")
from sklearn import tree
import matplotlib.pyplot as plt

%matplotlib inline

tree.plot_tree(rt_search.best_estimator_.estimators_[0], filled = True, rounded=True, feature_names=x.columns, class_names=['Disease', 'No disease'])

plt.figure(dpi = 300)

plt.show()
