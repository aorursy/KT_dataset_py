import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
df = pd.read_csv('../input/telecom_churn.csv')
df
#ситуация такая, State, International plan, Voice mail plan - object, Churn - категория.
#Преобразить их проще всего (и короче) через LabelEncoder
non_numeric_features = ['State', 'International plan', 'Voice mail plan', 'Churn']

for feature in non_numeric_features:        
     df[feature] = LabelEncoder().fit_transform(df[feature])
#в итоге все преобразовалось в числовой тип, проблем больше не будет
#df
#можно посмотреть, как коррелируют столбики друг с другом
#производительность некоторых алгоритмов может ухудшиться, если две или более переменных тесно связаны между собой, что называется мультиколлинеарностью. 
import seaborn as sns
fig, axs = plt.subplots(nrows=2, figsize=(50, 50))
sns.heatmap((df).corr(), ax=axs[0], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 14})

for i in range(2):    
    axs[i].tick_params(axis='x', labelsize=14)
    axs[i].tick_params(axis='y', labelsize=14)
    
axs[0].set_title('Корреляции', size=15)
#большое спасибо теме отсюда https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on/43104383#43104383

def remove_collinear_features(x, threshold):  
    # Не убираем корреляции между фичами и целью, потому что они наоборот помогают!
    y = x['Churn']
    x = x.drop(columns = ['Churn'])
    
    # Считаем матрицы корреляций
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            
            if val >= threshold:
                drop_cols.append(col.values[0])

    # Убрать одну из коррелирующих колонок в паре
    drops = set(drop_cols)
    x = x.drop(columns = drops)
    
    # Вернуть результат обратно в датасет
    x['Churn'] = y
               
    return x
#сделаем копию датасета, чтобы не испортить основной
features = df.copy()
features = remove_collinear_features(features, 0.8);
#я проделала все то же самое с копией
non_numeric_features = ['State', 'International plan', 'Voice mail plan', 'Churn']

for feature in non_numeric_features:        
     features[feature] = LabelEncoder().fit_transform(features[feature])
#можно еще раз построить карту и глянуть результат, он еще не идеален, но на данное решение хватит (не идеален, тк есть высокие корреляции еще)
import seaborn as sns
fig, axs = plt.subplots(nrows=2, figsize=(50, 50))
sns.heatmap((features).corr(), ax=axs[0], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 14})

for i in range(2):    
    axs[i].tick_params(axis='x', labelsize=14)
    axs[i].tick_params(axis='y', labelsize=14)
    
axs[0].set_title('Корреляции', size=15)
features
#нужно поделить данные, можно просто сделать это с помощью loc, но я напихала колонки для наглядности тебе

X = features[{'State','Account length','Area code','International plan','Voice mail plan','Number vmail messages','Total day minutes','Total day calls','Total eve minutes','Total eve calls','Total night minutes','Total night calls','Total intl minutes','Total intl calls','Customer service calls'}]
y = features.Churn
#Нормализация данных
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
#попробую обучить модель на RandomForest`е
#впихну сюда GridSearch (это вопрос поиска по сетке)
#parameters = {
#    'n_estimators'      : [300, 400, 500],
#    'max_depth'         : [3, 4, 5, 6, 7],
#    'max_features': ['auto', 'sqrt', 'log2'],
#    'criterion' :['gini', 'entropy']
#}

#rf_model = RandomForestClassifier(random_state=42)

#clf = GridSearchCV(rf_model, parameters, cv=5)
#model = clf.fit(X_train, y_train)
#я не уверена, что грид выдал нормальные параметры, потому пихну кросс-валиадцию с 5 блоками
#я попробовала параметры, который выдал Грид, но оказалось, что можно и лучше, так что я поменяла их немного
#вообще, если играть с параметрами, возможно, можно и добиться еще выше результатов только за счет них

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

scores = []
leaderboard_model = RandomForestClassifier(criterion='gini',
                                           n_estimators=700,
                                           max_depth=18,
                                           min_samples_split=4,
                                           min_samples_leaf=1,
                                           max_features='auto',
                                           oob_score=True,
                                           random_state=42,
                                           n_jobs=-1,
                                           verbose=1) 
cv = KFold(n_splits=5)
for train_index, test_index in cv.split(X):

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    leaderboard_model.fit(X_train, y_train)
    scores.append(leaderboard_model.score(X_test, y_test))
#точность вышла 0,945
print(np.mean(scores))
#тут я предсказываю на тестовом датасете, используя обученную модель
y_test_pred = leaderboard_model.predict(X_test)
#для наглядности можно построить датасет из разряда до\после, у меня тут что-то с ID-шниками, вообще их выкинуть можно нахрен, но я этим не занималась, тк тебе сейчас это не существенно
df2 = pd.DataFrame({'Churn': y_test, 'Predicted Churn': y_test_pred})
df2.head(20)