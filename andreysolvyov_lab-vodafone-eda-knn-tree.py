import warnings

warnings.filterwarnings('ignore')



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





import matplotlib.pyplot as plt

import seaborn as sns

sns.set();

# Any results you write to the current directory are saved as output.
import pandas as pd

vodafone_subset_5 = pd.read_csv("../input/vodafone-subset-5.csv")

Data = vodafone_subset_5[['Oblast_post_HOME', 'AVG_ARPU', 'SCORING', 'car', 'how_long_same_model', 'gas_stations_sms']]

Data.rename(columns={'Oblast_post_HOME': 'Область проживания', 

                     'AVG_ARPU': 'Стоимость услуг',

                     'SCORING': 'Уровень дохода',

                     'car': 'Наличие машины',

                     'how_long_same_model': 'Срок использования мобильного',

                     'gas_stations_sms': 'Колво сообщений от заправок'}, inplace=True)

Data['Target'] = vodafone_subset_5['target']

#Data
Data.dtypes
Data['Уровень дохода'].value_counts()
temp = Data['Область проживания'].value_counts().reset_index().reset_index()[['level_0', 'index']]

temp.columns = ['Временный индекс','Область проживания']

Map = {temp['Область проживания'][i]:i for i in range(len(temp['Область проживания']))}

Inverse_Map = {I:J for J,I in Map.items()}

Data['Индекс области'] = Data['Область проживания'].map(Map)

Data['Индекс уровня дохода'] = Data['Уровень дохода'].map({'HIGH':6,

                                                           'HIGH_MEDIUM':5,

                                                           'MEDIUM':4,

                                                           'LOW':3,

                                                           'VERY LOW':2,

                                                           '0':1})

Data['Стоимость услуг'] = round(Data['Стоимость услуг'])

Data['Стоимость услуг'] = Data['Стоимость услуг'].astype(int)

Data['Индекс уровня дохода'] = Data['Индекс уровня дохода'].astype(int)

Data = Data.loc[Data['Индекс уровня дохода'] != 6]

Data['Срок использования мобильного'] = Data['Срок использования мобильного'].astype(int)

Data['Колво сообщений от заправок'] = Data['Колво сообщений от заправок'].astype(int)

#Data
WData = Data[['Стоимость услуг', 'Наличие машины', 'Срок использования мобильного', 

              'Колво сообщений от заправок', 'Индекс области', 'Индекс уровня дохода', 'Target']]

WData.columns = ['Cost of services','Availability of car', 'Term of use of mobile',

              'Number of messages from gas stations', 'Region ID', 'Income level index', 'Target']
WData['Region ID'].value_counts().plot(kind='bar');
#sns.pairplot(WData);
WData.head(5)
sns.heatmap(WData.corr(method = 'spearman'), annot = True);
count_conti = WData[['Cost of services', 'Term of use of mobile']]

count_discr = WData[['Number of messages from gas stations']]

rang = WData[['Income level index', 'Target']]

categorial = WData[['Availability of car', 'Region ID']]

count_con_dis = count_conti.merge(count_discr, left_index=True, right_index=True)

count_con_rang = count_conti.merge(rang, left_index=True, right_index=True)

count_con_cat = count_conti.merge(categorial, left_index=True, right_index=True)

count_dis_rang = count_discr.merge(rang, left_index=True, right_index=True)

count_dis_cat = count_discr.merge(categorial, left_index=True, right_index=True)

rang_cat = rang.merge(categorial, left_index=True, right_index=True)
sns.heatmap(count_conti.corr(method = 'spearman'), annot = True);
sns.heatmap(rang.corr(method = 'spearman'), annot = True);
sns.catplot(y="Region ID", x="Availability of car",data=categorial);
sns.heatmap(count_con_dis.corr(method = 'spearman'), annot = True);
sns.heatmap(count_con_rang.corr(method = 'spearman'), annot = True);
ax = sns.boxplot(x="Region ID", y="Term of use of mobile", data=count_con_cat)

sns.heatmap(count_dis_rang.corr(method = 'spearman'), annot = True);
sns.catplot(y="Number of messages from gas stations", x="Availability of car",data=count_dis_cat);
sns.catplot(y="Number of messages from gas stations", x="Region ID",data=count_dis_cat);
sns.catplot(y="Income level index", x="Availability of car",data=rang_cat);
from scipy.stats import pearsonr, spearmanr, kendalltau

r = pearsonr(WData['Target'], WData['Term of use of mobile'])

print('Pearson correlation:', r[0], 'p-value:', r[1])
r = pearsonr(WData['Target'], WData['Income level index'])

print('Pearson correlation:', r[0], 'p-value:', r[1])
r = pearsonr(WData['Target'], WData['Cost of services'])

print('Pearson correlation:', r[0], 'p-value:', r[1])
r = pearsonr(WData['Term of use of mobile'], WData['Income level index'])

print('Pearson correlation:', r[0], 'p-value:', r[1])
r = pearsonr(WData['Term of use of mobile'], WData['Cost of services'])

print('Pearson correlation:', r[0], 'p-value:', r[1])
r = pearsonr(WData['Income level index'], WData['Cost of services'])

print('Pearson correlation:', r[0], 'p-value:', r[1])
WData[['calls_count_in_weekdays', 

       'calls_duration_in_weekdays',

       'calls_count_out_weekdays',

       'calls_duration_out_weekdays', 

       'calls_count_in_weekends', 

       'calls_duration_in_weekends', 

       'calls_count_out_weekends', 

       'calls_duration_out_weekends', 

       'DATA_VOLUME_WEEKDAYS', 

       'ROUM', 

       'DATA_VOLUME_WEEKENDS', 

       'ecommerce_score', 

       'banks_sms_count', 

       'phone_value']] = vodafone_subset_5[['calls_count_in_weekdays', 

                                 'calls_duration_in_weekdays', 

                                 'calls_count_out_weekdays',

                                 'calls_duration_out_weekdays',

                                 'calls_count_in_weekends',

                                 'calls_duration_in_weekends',

                                 'calls_count_out_weekends',

                                 'calls_duration_out_weekends',

                                 'DATA_VOLUME_WEEKDAYS',

                                 'ROUM',

                                 'DATA_VOLUME_WEEKENDS',

                                 'ecommerce_score',

                                 'banks_sms_count',

                                 'phone_value']]
df = WData.drop('Target', axis = 1)
df.shape
WData['Target'].shape
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



X_train, X_valid, y_train, y_valid = train_test_split(df, 

                                                      WData['Target'], 

                                                      test_size=0.25, 

                                                      random_state=1803)

#scaler почему-то уменьшает кол-во данных, поэтому в комментарии

#scaler = StandardScaler()

#scaler.fit(X_train)



#X_train = scaler.transform(X_train)



#scaler = StandardScaler()

#scaler.fit(X_valid)



#X_train = scaler.transform(X_valid)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
X_train.shape
y_train.shape
knn.fit(X_train, y_train)
y_pred = knn.predict(X_valid)

y_pred
knn.score(X_valid, y_valid)
from sklearn.metrics import accuracy_score

accuracy_score(y_valid, y_pred)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

knn = KNeighborsClassifier(n_neighbors=1)

scores = cross_val_score(knn, df, WData['Target'], cv=kf, scoring = 'accuracy')

print(scores)

mean_score = scores.mean()

print(mean_score)
import matplotlib

lams = []

score = []

for lam in range(1, 5):#range(1, 51):

    knn = KNeighborsClassifier(n_neighbors=lam, p=1)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_valid)

    lams.append(lam)

    score.append(accuracy_score(y_valid, y_pred))

matplotlib.pyplot.plot(lams, score);
K = score.index(max(score)) + 1

print(K)
max(score)
lams = []

score = []

rang = np.linspace(1, 10, num=2)#num=200

for lam in rang:

    knn = KNeighborsClassifier(n_neighbors=K, p=lam)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_valid)

    lams.append(lam)

    score.append(accuracy_score(y_valid, y_pred))

matplotlib.pyplot.plot(lams, score)
P = lams[score.index(max(score))]

P
max(score)
knn = KNeighborsClassifier(n_neighbors=K, p=P)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_valid)

accuracy_score(y_valid, y_pred)

knn = KNeighborsClassifier(n_neighbors=K, p=P)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_valid)

knn.score(X_valid, y_valid)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

knn = KNeighborsClassifier(n_neighbors=K, p=P)

scores = cross_val_score(knn, df, WData['Target'], cv=kf, scoring='accuracy')

mean_score = scores.mean()

print(mean_score)
[X_train.shape, X_valid.shape, y_train.shape, y_valid.shape]
from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(max_depth=2, random_state=2019)

tree.fit(X_train, y_train)
from sklearn.tree import export_graphviz



#export_graphviz(tree, out_file='tree.dot')

#print(open('tree.dot').read())
y_pred = tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
from sklearn.model_selection import GridSearchCV



tree_params = {'max_depth': np.arange(2, 11),

               'min_samples_leaf': np.arange(2, 11)}



tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

tree_grid.fit(X_train, y_train)
tree_params_d = {'max_depth': np.arange(1, 11)}



tree_grid_d = GridSearchCV(tree, tree_params_d, cv=5, scoring='accuracy')

tree_grid_d.fit(X_train, y_train)
tree_params_s = {'min_samples_leaf': np.arange(1, 11)}



tree_grid_s = GridSearchCV(tree, tree_params_s, cv=5, scoring='accuracy') 

tree_grid_s.fit(X_train, y_train)
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True) # 2 графика рядом с одинаковым масштабом по оси Оу



ax[0].plot(tree_params_d['max_depth'], tree_grid_d.cv_results_['mean_test_score']) # accuracy vs max_depth

ax[0].set_xlabel('max_depth')

ax[0].set_ylabel('Mean accuracy on test set')



ax[1].plot(tree_params_s['min_samples_leaf'], tree_grid_s.cv_results_['mean_test_score']) # accuracy vs min_samples_leaf

ax[1].set_xlabel('min_samples_leaf')

ax[1].set_ylabel('Mean accuracy on test set');
best_tree = tree_grid.best_estimator_

y_pred = best_tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
#export_graphviz(best_tree, out_file='best_tree.dot')

#print(open('best_tree.dot').read()) 
X_train.columns[[14, 16, 19]]
vodafone_subset_5 = pd.read_csv("../input/vodafone-subset-5.csv")
Data = vodafone_subset_5.drop('target', axis = 1)

#.select_dtypes(exclude=['object'])

Target = vodafone_subset_5[['target']]
#90%

indexes = []

for col in Data.columns:

    if max(Data[col].value_counts()) >= 9000:

        indexes.append(col)

len(indexes)
WData = Data.drop(indexes, axis = 1)
len(WData.columns)
objects = []

types = []

for j in WData.columns:

    types.append(WData[[j]].dtypes[0])

    if WData[[j]].dtypes[0] == 'object':

        objects.append(j)

print(set(types))

print(objects)
objects = objects[:-2]

objects
for col in objects:

    temp = WData[col].value_counts().reset_index().reset_index()[['level_0', 'index']]

    temp.columns = ['Временный индекс',col]

    Map = {temp[col][i]:i for i in range(len(temp[col]))}

    WData[col + '_INDEX_'] = WData[col].map(Map)
WData['SCORING_INDEX_'] = WData['SCORING'].map({'HIGH':6,

                                               'HIGH_MEDIUM':5,

                                               'MEDIUM':4,

                                               'LOW':3,

                                               'VERY LOW':2,

                                               '0':1})
data = WData.drop(objects + ['user_hash', 'SCORING'], axis = 1)
objects = []

types = []

for j in data.columns:

    types.append(data[[j]].dtypes[0])

    if WData[[j]].dtypes[0] == 'object':

        objects.append(j)

print(set(types))

print(objects)
for i in range(1, len(data.columns)):

    if len(data.columns)%i==0:

        print(i)
dfcolumns = [data.columns[:19]]

for i in range(20, len(data.columns)-17, 19):

    dfcolumns.append(data.columns[i:i+19])

dfcolumns.append(data.columns[len(data.columns)-19:])

for i in dfcolumns:

    print(len(i), end = ' ')
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree.export import export_text

i = -1

score_blok = {}

for blok_col in dfcolumns:

    i = i + 1

    df = data[blok_col].copy().dropna()

    X_train, X_valid, y_train, y_valid = train_test_split(df, 

                                                          Target[:len(df)], 

                                                          test_size=0.25, 

                                                          random_state=2303)

    tree = DecisionTreeClassifier(max_depth=2, random_state=2019)

    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_valid)

    tree_params = {'max_depth': np.arange(2, 11),

                   'min_samples_leaf': np.arange(2, 11)}

    tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

    tree_grid.fit(X_train, y_train)

    best_tree = tree_grid.best_estimator_

    y_pred = best_tree.predict(X_valid)

    r = export_text(best_tree, feature_names=list(df.columns))

    score_blok.update([(i, [accuracy_score(y_valid, y_pred), r[r.index('|---')+5:r.index('<=') - 1]])])

d = list(score_blok.items())

d.sort(key=lambda i: i[1])

d = d[::-1]

score_blok = dict(d)
score_blok
tops = []

for i in range(len(score_blok)):

    tops.append(list(score_blok.items())[i][1][1])

tops
df = data[tops].copy().dropna()

X_train, X_valid, y_train, y_valid = train_test_split(df, 

                                                      Target[:len(df)], 

                                                      test_size=0.25, 

                                                      random_state=2303)

tree = DecisionTreeClassifier(max_depth=2, random_state=2019)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_valid)

tree_params = {'max_depth': np.arange(2, 11),

               'min_samples_leaf': np.arange(2, 11)}

tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

tree_grid.fit(X_train, y_train)

best_tree = tree_grid.best_estimator_

y_pred = best_tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
df = data.dropna()

X_train, X_valid, y_train, y_valid = train_test_split(df, 

                                                      Target[:len(df)], 

                                                      test_size=0.25, 

                                                      random_state=2303)

tree = DecisionTreeClassifier(max_depth=2, random_state=2019)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_valid)

tree_params = {'max_depth': np.arange(2, 11),

               'min_samples_leaf': np.arange(2, 11)}

tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

tree_grid.fit(X_train, y_train)

best_tree = tree_grid.best_estimator_

y_pred = best_tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
dfcolumns = [data.columns[:17]]

for i in range(18, len(data.columns)-16, 17):

    dfcolumns.append(data.columns[i:i+17])

dfcolumns.append(data.columns[len(data.columns)-17:])

for i in dfcolumns:

    print(len(i), end = ' ')
i = -1

score_blok = {}

for blok_col in dfcolumns:

    i = i + 1

    df = data[blok_col].copy().dropna()

    X_train, X_valid, y_train, y_valid = train_test_split(df, 

                                                          Target[:len(df)], 

                                                          test_size=0.25, 

                                                          random_state=2303)

    tree = DecisionTreeClassifier(max_depth=2, random_state=2019)

    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_valid)

    tree_params = {'max_depth': np.arange(2, 11),

                   'min_samples_leaf': np.arange(2, 11)}

    tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

    tree_grid.fit(X_train, y_train)

    best_tree = tree_grid.best_estimator_

    y_pred = best_tree.predict(X_valid)

    r = export_text(best_tree, feature_names=list(df.columns))

    score_blok.update([(i, [accuracy_score(y_valid, y_pred), r[r.index('|---')+5:r.index('<=') - 1]])])

d = list(score_blok.items())

d.sort(key=lambda i: i[1])

d = d[::-1]

score_blok = dict(d)
score_blok
df = data[data.columns[20:39]].copy().dropna()

X_train, X_valid, y_train, y_valid = train_test_split(df, 

                                                      Target[:len(df)], 

                                                      test_size=0.25, 

                                                      random_state=2303)

tree = DecisionTreeClassifier(max_depth=2, random_state=2019)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_valid)

tree_params = {'max_depth': np.arange(2, 11),

               'min_samples_leaf': np.arange(2, 11)}

tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

tree_grid.fit(X_train, y_train)

best_tree = tree_grid.best_estimator_

y_pred = best_tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
best_tree
#export_graphviz(best_tree, out_file='best_tree.dot', feature_names=list(df.columns))

#print(open('best_tree.dot').read()) 