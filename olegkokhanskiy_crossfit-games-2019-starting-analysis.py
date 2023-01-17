import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
import seaborn as sns
nRowsRead = 1000 # specify 'None' if want to read whole file
# 2019_opens_scores.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1_g_athletes = pd.read_csv('../input/crossfit-games/2019_games_athletes.csv', delimiter=',', nrows = nRowsRead)
df2_g_scores = pd.read_csv('../input/crossfit-games/2019_games_scores.csv', delimiter=',', nrows = nRowsRead)
df3_o_athletes = pd.read_csv('../input/crossfit-games/2019_opens_athletes.csv', delimiter=',', nrows = nRowsRead)
df4_o_scores = pd.read_csv('../input/crossfit-games/2019_opens_scores.csv', delimiter=',', nrows = nRowsRead)
df1_g_athletes.dataframeName = '2019_games_athletes.csv'
df3_o_athletes.dataframeName = '2019_opens_athletes.csv'
df2_g_scores.dataframeName = '2019_games_scores.csv'
df4_o_scores.dataframeName = '2019_opens_scores.csv'
nRow, nCol = df1_g_athletes.shape
nRow2, nCol2 = df3_o_athletes.shape
nRow3, nCol3 = df2_g_scores.shape
nRow4, nCol4 = df4_o_scores.shape
print(f'2019_games_athletes.csv have {nRow} rows and {nCol} columns')
print(f'2019_opens_athletes.csv have {nRow2} rows and {nCol2} columns')
print(f'2019_games_scores.csv have {nRow3} rows and {nCol3} columns')
print(f'2019_opens_scores.csv have {nRow4} rows and {nCol4} columns')
df1_g_athletes.dtypes
#df1_g_athletes.describe()

fig, ax = plt.subplots(figsize=(9,5))
sns.heatmap(df1_g_athletes.isnull(), cbar=False, cmap="YlGnBu_r")
plt.show()
# Обрабатываем данные таблици df1_g_athletes

# Убираем в колонке overallrank ненужные символы и преобразуем в тип данных int64.
df1_g_athletes['overallrank'] = df1_g_athletes['overallrank'].replace(['134T', '138T', '143T','92T','102T','117T' ], [134, 138, 143, 92, 102, 117])
df1_g_athletes.overallrank = df1_g_athletes.overallrank.astype(np.int64)
# округляем вес спортсменов и преобразуем в тип данных int64.
#df1_g_athletes['weight'] = df1_g_athletes['weight'].round()
#df1_g_athletes.weight = df1_g_athletes.weight.astype(np.int64)
# преобразуем данные affiliateid в тип данных int64, Nan заменим на 0.
df1_g_athletes['affiliateid'] = df1_g_athletes['affiliateid'].fillna(0).astype(np.int64)
df1_g_athletes['affiliateid'] = df1_g_athletes['affiliateid'].astype(np.int64)
# преобразуем данные overallscore в тип данных int64.Nan заменим на 0.
df1_g_athletes['overallscore'] = df1_g_athletes['overallscore'].fillna(0).astype(np.int64)
df1_g_athletes['overallscore'] = df1_g_athletes['overallscore'].astype(np.int64)
# преобразуем данные height в тип данных int64, заменим нереалистичные показатели на 0.
#df1_g_athletes['height'] = df1_g_athletes['height'] * 100
#df1_g_athletes['height'] = df1_g_athletes['height'].astype(np.int64)
df1_g_athletes['height'] = df1_g_athletes['height'].replace([0.15, 0.03, 0.18], [0, 0, 0])
# преобразуем все значения NaN столбца affiliatename в значение NoInfo
df1_g_athletes['affiliatename'] = df1_g_athletes['affiliatename'].fillna('NoInfo').astype(np.object)
# преобразуем все значения NaN столбца competitorname в значение NoInfoCName
df1_g_athletes['competitorname'] = df1_g_athletes['competitorname'].fillna('NoInfoCName').astype(np.object)
# преобразуем все значения NaN столбца overallscore в значение NoInfoOScore
#df1_g_athletes['overallscore'] = df1_g_athletes['overallscore'].fillna(0).astype(np.int64)
# преобразуем все значения alt столбца bibid в значение 0
df1_g_athletes['bibid'] = df1_g_athletes['bibid'].replace(['alt'], [0])
df1_g_athletes.bibid = df1_g_athletes.bibid.astype(np.int64)
# преобразуем все значения NaN столбца countryoforigincode в значение NoInfoCCode
df1_g_athletes['countryoforigincode'] = df1_g_athletes['countryoforigincode'].fillna('NoInfoCCode').astype(np.object)
df1_g_athletes_MWDivision = df1_g_athletes
df1_g_athletes.head()
#df=df1_g_athletes.groupby('division').overallscore.agg(['mean', 'median'])
#df=df1_g_athletes.groupby('division').TopTenDivision.value_counts(['Scale', 'Not scale'])
df=df1_g_athletes.groupby('division').competitorid.count()
df.plot(kind='bar',stacked = True, figsize=(15, 6), fontsize=20)
plt.xlabel("Дивизион", fontsize=20)
plt.ylabel("Количество атлетов", fontsize=20)
plt.title("Количество атлетов в дивизионе", fontsize=20)

#df=df1_g_athletes.groupby('division').overallscore.agg(['mean', 'median'])
df01 = (df1_g_athletes.division == 'Men') | (df1_g_athletes.division == 'Women')
df1=df1_g_athletes[df01].groupby('countryoforiginname').division.value_counts()
df1.nlargest(10).plot(kind='bar',stacked = True, figsize=(15, 6), fontsize=25)
plt.xlabel("Страна", fontsize=20)
plt.ylabel("Количество атлетов", fontsize=20)
plt.title("Количество представителей по странам (Топ 10 стран)", fontsize=20)
df02 = ((df1_g_athletes.division == 'Men') | (df1_g_athletes.division == 'Women')) & (df1_g_athletes.overallrank < 4)
df2=df1_g_athletes[df02].groupby('countryoforiginname').division.count()
df2.nlargest(5).plot(kind='bar',stacked = True, figsize=(15, 6), fontsize=25)
plt.xlabel("Страна", fontsize=20)
plt.ylabel("Количество атлетов с рангом 3 и выше", fontsize=20)
plt.title("Количество атлетов занявших призовые места (Топ 5 стран)", fontsize=20)
df1_g_athletes[df02]
for df in [df1_g_athletes] :

    df['competitorname_Size'] = df.competitorname.apply(lambda x : len(x)) 
    df['NameLen']=np.nan
    for i in range(20,0,-1):
        df.loc[ df['competitorname_Size'] <= i*5, 'NameLen'] = i
# overallrank - выделим топ 10 спортсменов в ранге по таблице (В топ 10 - 1, не в топ 10 - 0)
df1_g_athletes_MWDivision['overallrank'] = df1_g_athletes_MWDivision['overallrank'].apply((lambda x: x<11))
groupby_NameLen_overallrank = df1_g_athletes.groupby(['NameLen'])['overallrank'].count().to_frame()
groupby_NameLen_overallrank
plt.subplots(figsize=(10,6))
sns.barplot(x='NameLen' , y='overallrank' , data = df1_g_athletes)
plt.ylabel("overallrank Rate")
plt.title("overallrank as function of NameLen")
plt.show()
cm_surv = ["darkgrey" , "lightgreen"]
fig, ax = plt.subplots(figsize=(9,7))
sns.violinplot(x="NameLen", y="division", data=df1_g_athletes, hue='overallrank', split=True, 
               orient="h", bw=0.2 , palette=cm_surv, ax=ax)
plt.show()
g = sns.factorplot(x="NameLen", y="overallrank", col="gender", data=df1_g_athletes, kind="bar", size=5, aspect=1.2)
df1_g_athletes_MWDivision['overallrank'] = df1_g_athletes_MWDivision['overallrank'].replace([1, 0], ['Топ 10', 'Не топ 10'])
df1_g_athletes_MWDivision
# Найдём категориальные признаки
Categorical_cols = list(set(df1_g_athletes_MWDivision.columns) - set(df1_g_athletes_MWDivision._get_numeric_data().columns))
Categorical_cols
df1_g_athletes_MWDivision[Categorical_cols]
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000
encoded_categorical_columns = pd.DataFrame(onehot_encoder.fit_transform(df1_g_athletes_MWDivision[Categorical_cols]))
encoded_categorical_columns.head()
# Подключаем класс для предобработки данных
from sklearn import preprocessing

# Напишем функцию, которая принимает на вход DataFrame, кодирует числовыми значениями категориальные признаки
# и возвращает обновленный DataFrame и сами кодировщики.
def number_encode_features(init_df):
    result = init_df.copy() # копируем нашу исходную таблицу
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object: # np.object -- строковый тип / если тип столбца - строка, то нужно его закодировать
            encoders[column] = preprocessing.LabelEncoder() # для колонки column создаем кодировщик
            result[column] = encoders[column].fit_transform(result[column]) # применяем кодировщик к столбцу и перезаписываем столбец
    return result, encoders

encoded_data, encoders = number_encode_features(df1_g_athletes_MWDivision) # Теперь encoded data содержит закодированные кат. признаки 
encoded_data.head() 

fig = plt.figure(figsize=(19,8))
cols = 6
rows = np.ceil(float(encoded_data.shape[1]) / cols)
for i, column in enumerate(encoded_data.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    encoded_data[column].hist(axes=ax)
    plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)


df1_g_athletes_MWDivision
df2_g_scores
pd.merge(df1_g_athletes_MWDivision, df2_g_scores, on='competitorid', how='right')
   #      left_on=['competitorname', 'competitorid'],
   #      right_on=['scoredisplay', 'competitorid'])
plt.subplots(figsize=(15,15))
encoded_data, encoders = number_encode_features(df1_g_athletes_MWDivision)
sns.heatmap(encoded_data.corr(), square=True)
plt.show()
df_prc = df1_g_athletes_MWDivision.copy()
df_prc['overallrank'] = df1_g_athletes_MWDivision['overallrank'].apply((lambda x: x=='Топ 3')) # Будем предсказывать 1(True), если спортсмен попал в топ 3 призеров, 0(False) иначе
df_prc.head()

# числовые признаки
df1_g_athletes_MWDivision._get_numeric_data().columns
X = np.array(df_prc[df1_g_athletes_MWDivision._get_numeric_data().columns])
# y = np.array(df_prc['salary'], dtype='int')
y = encoders['overallrank'].transform(df1_g_athletes_MWDivision['overallrank']) # применяем наши кодировщики к категориальным фичам

# Функция отрисовки графиков

def grid_plot(x, y, x_label, title, y_label='roc_auc'):
    plt.figure(figsize=(12, 6))
    plt.grid(True)
    plt.plot(x, y, 'go-')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
# Будем использовать модель k ближайших соседей
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
# В sklearn есть специальный модуль для работы с кросс-валидацией
from sklearn.model_selection import GridSearchCV

# Зададим сетку - среди каких значений выбирать наилучший параметр.
knn_grid = {'n_neighbors': np.array(np.linspace(2, 50, 4), dtype='int')} # перебираем по параметру <<n_neighbors>>, по сетке заданной np.linspace(2, 50, 4)

# Создаем объект кросс-валидации
gs = GridSearchCV(knn, knn_grid, cv=5)

# Обучаем его
gs.fit(X, y)
# Строим график зависимости качества от числа соседей
# замечание: результаты обучения хранятся в атрибуте cv_results_ объекта gs

grid_plot(knn_grid['n_neighbors'], gs.cv_results_['mean_test_score'], 'n_neighbors', 'KNeighborsClassifier')
knn_grid = {'n_neighbors': np.array(np.linspace(32, 38, 11), dtype='int')}
gs = GridSearchCV(knn, knn_grid, cv=10)
gs.fit(X, y)

# best_params_ содержит в себе лучшие подобранные параметры, best_score_ лучшее качество
gs.best_params_, gs.best_score_
grid_plot(knn_grid['n_neighbors'], gs.cv_results_['mean_test_score'], 'n_neighbors', 'KNeighborsClassifier')
from sklearn.preprocessing import scale
X_scaled = scale(np.array(X, dtype='float'), with_std=True, with_mean=False)
X_scaled
grid = {'n_neighbors': np.array(np.linspace(1, 50, 10), dtype='int')}
gs = GridSearchCV(knn, grid, cv=5, n_jobs=5)
gs.fit(X_scaled, y)
print(gs.best_params_, gs.best_score_)
grid_plot(grid['n_neighbors'], gs.cv_results_['mean_test_score'], 'n_neighbors', 'KNeighborsClassifier')
grid = {'n_neighbors': np.array(np.linspace(35, 40, 34), dtype='int')}
gs = GridSearchCV(knn, grid, cv=10, n_jobs=5)
gs.fit(X_scaled, y)
print(gs.best_params_, gs.best_score_)
grid_plot(grid['n_neighbors'], gs.cv_results_['mean_test_score'], 'n_neighbors', 'KNeighborsClassifier')
from sklearn.model_selection import train_test_split

X_train, X_test, y_tain, y_test = train_test_split(X,y, test_size=0.33, random_state=42)
from sklearn import dummy

knn = KNeighborsClassifier(n_neighbors=32) 
clf_knn = knn.fit(X_train, y_tain)
clf_mp = dummy.DummyClassifier("most_frequent").fit(X_train, y_tain)
y_knn = clf_knn.predict(X_test)
y_mp = clf_mp.predict(X_test)
y_knn
y_mp
y_test
from sklearn import metrics

print ('knn =', metrics.accuracy_score(y_test, y_knn), 'mp =', metrics.accuracy_score(y_test, y_mp))
import matplotlib

fig = plt.figure(figsize=(10,8))
nn_mtx = metrics.confusion_matrix(y_test, y_knn)

font = {'family' : 'Calibri', 'weight' : 'bold', 'size'   :22}
matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
sns.heatmap(nn_mtx, annot=True, fmt="d", 
            xticklabels=encoders["overallrank"].classes_, 
            yticklabels=encoders["overallrank"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")
mp_mtx = metrics.confusion_matrix(y_test, y_mp)

font = {'family' : 'Calibri', 'weight' : 'bold', 'size'   :22}
matplotlib.rc('font', **font)
fig = plt.figure(figsize=(10,8))
sns.heatmap(mp_mtx, annot=True, fmt="d", 
            xticklabels=encoders["overallrank"].classes_, 
            yticklabels=encoders["overallrank"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")
print ('knn =', metrics.precision_score(y_test, y_knn), 'mp =', metrics.precision_score(y_test, y_mp))
print ('knn =', metrics.recall_score(y_test, y_knn), 'mp =', metrics.recall_score(y_test, y_mp))
df_Regression_g_athletes = pd.read_csv('../input/crossfit-games/2019_games_athletes.csv', delimiter=',', nrows = nRowsRead)
df_Regression_g_athletes.dataframeName = '2019_games_athletes.csv'
nRow, nCol = df_Regression_g_athletes.shape
print(f'2019_games_athletes.csv have {nRow} rows and {nCol} columns')
# Обрабатываем данные таблици df1_g_athletes

# Убираем в колонке overallrank ненужные символы и преобразуем в тип данных int64.
df_Regression_g_athletes['overallrank'] = df_Regression_g_athletes['overallrank'].replace(['134T', '138T', '143T','92T','102T','117T' ], [134, 138, 143, 92, 102, 117])
df_Regression_g_athletes.overallrank = df_Regression_g_athletes.overallrank.astype(np.int64)
# преобразуем данные affiliateid в тип данных int64, Nan заменим на 0.
df_Regression_g_athletes['affiliateid'] = df_Regression_g_athletes['affiliateid'].fillna(0).astype(np.int64)
df_Regression_g_athletes['affiliateid'] = df_Regression_g_athletes['affiliateid'].astype(np.int64)
# преобразуем данные overallscore в тип данных int64.Nan заменим на 0.
df_Regression_g_athletes['overallscore'] = df_Regression_g_athletes['overallscore'].fillna(0).astype(np.int64)
df_Regression_g_athletes['overallscore'] = df_Regression_g_athletes['overallscore'].astype(np.int64)
# преобразуем данные height в тип данных int64, заменим нереалистичные показатели на 0.
df_Regression_g_athletes['height'] = df_Regression_g_athletes['height'].replace([0.15, 0.03, 0.18], [0, 0, 0])
# преобразуем все значения NaN столбца affiliatename в значение NoInfo
df_Regression_g_athletes['affiliatename'] = df_Regression_g_athletes['affiliatename'].fillna('NoInfo').astype(np.object)
# преобразуем все значения NaN столбца competitorname в значение NoInfoCName
df_Regression_g_athletes['competitorname'] = df_Regression_g_athletes['competitorname'].fillna('NoInfoCName').astype(np.object)
# преобразуем все значения alt столбца bibid в значение 0
df_Regression_g_athletes['bibid'] = df_Regression_g_athletes['bibid'].replace(['alt'], [0])
df_Regression_g_athletes.bibid = df_Regression_g_athletes.bibid.astype(np.int64)
# преобразуем все значения NaN столбца countryoforigincode в значение NoInfoCCode
df_Regression_g_athletes['countryoforigincode'] = df_Regression_g_athletes['countryoforigincode'].fillna('NoInfoCCode').astype(np.object)
df_Regression_g_athletes_train_ml = encoded_data.copy()
df_Regression_g_athletes_test_ml = encoded_data.copy()
del df_Regression_g_athletes_test_ml['overallrank']
competitor_id = df_Regression_g_athletes['competitorid']
df_Regression_g_athletes_train_ml.info()
df_Regression_g_athletes_test_ml.info()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# for df_train_ml
scaler.fit(df_Regression_g_athletes_train_ml.drop(['overallrank'],axis=1))
scaled_features = scaler.transform(df_Regression_g_athletes_train_ml.drop(['overallrank'],axis=1))
df_train_ml_sc = pd.DataFrame(scaled_features) # columns=df_train_ml.columns[1::])

# for df_test_ml
df_Regression_g_athletes_test_ml.fillna(df_Regression_g_athletes_test_ml.mean(), inplace=True)
#scaler.fit(df_test_ml)
scaled_features = scaler.transform(df_Regression_g_athletes_test_ml)
df_test_ml_sc = pd.DataFrame(scaled_features) # , columns=df_test_ml.columns)
df_train_ml_sc.head()
df_test_ml_sc.head()
X = df_Regression_g_athletes_train_ml.drop('overallrank', axis=1)
y = df_Regression_g_athletes_train_ml['overallrank']
X_test = df_Regression_g_athletes_test_ml

X_sc = df_train_ml_sc
y_sc = df_Regression_g_athletes_train_ml['overallrank']
X_test_sc = df_test_ml_sc
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

param_grid = {'min_samples_split': [4,7,10,12]}
dtree_grid = GridSearchCV(dtree, param_grid, cv=10, refit=True, verbose=1)
dtree_grid.fit(X_sc,y_sc)

print(dtree_grid.best_score_)
print(dtree_grid.best_params_)
print(dtree_grid.best_estimator_)
def get_best_score(model):
    
    print(model.best_score_)    
    print(model.best_params_)
    print(model.best_estimator_)
    
    return model.best_score_


def plot_feature_importances(model, columns):
    nr_f = 10
    imp = pd.Series(data = model.best_estimator_.feature_importances_, 
                    index=columns).sort_values(ascending=False)
    plt.figure(figsize=(7,5))
    plt.title("Feature importance")
    ax = sns.barplot(y=imp.index[:nr_f], x=imp.values[:nr_f], orient='h')
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

param_grid = {'max_depth': [3, 5, 6, 7, 8], 'max_features': [6,7,8,9,10],  
              'min_samples_split': [5, 6, 7, 8]}

rf_grid = GridSearchCV(rfc, param_grid, cv=10, refit=True, verbose=1)
rf_grid.fit(X_sc,y_sc)
sc_rf = get_best_score(rf_grid)
plot_feature_importances(rf_grid, X.columns)
