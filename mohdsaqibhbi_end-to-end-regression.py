import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df = pd.read_csv('../input/data-concrete-strength/Concrete_Data_Yeh.csv')

df.head(3)
df.info()
df.describe()
df.hist(bins=50, figsize=(20, 15))

plt.show()
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
df.describe()
test_df.describe()
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

df.csMPa.hist(bins=50, ax=ax[0], );test_df.csMPa.hist(bins=50, ax=ax[1])

ax[0].title.set_text('Complete data csMPa');ax[1].title.set_text('Test data csMPa')
temp_com_df = pd.DataFrame()

temp_com_df['Complete_csMPa_cat5'] = pd.cut(df.csMPa, bins=[0, 20, 40, 60, 80, 100],

                                            labels=['A', 'B', 'C', 'D', 'E'])

temp_com_df['Complete_csMPa_cat4'] = pd.cut(df.csMPa, bins=[0, 20, 40, 60, 100], labels=['A', 'B', 'C', 'D'])



temp_test_df = pd.DataFrame()

temp_test_df['Test_csMPa_cat5'] = pd.cut(test_df.csMPa, bins=[0, 20, 40, 60, 80, 100],

                                            labels=['A', 'B', 'C', 'D', 'E'])

temp_test_df['Test_csMPa_cat4'] = pd.cut(test_df.csMPa, bins=[0, 20, 40, 60, 100], labels=['A', 'B', 'C', 'D'])
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

temp_com_df.Complete_csMPa_cat4.hist(ax=ax)

temp_test_df.Test_csMPa_cat4.hist(ax=ax)

ax.title.set_text('Complete & Test csMPa with 4 categories')
print(temp_com_df.Complete_csMPa_cat4.value_counts()/len(temp_com_df))

print(temp_test_df.Test_csMPa_cat4.value_counts()/len(temp_test_df))
train_df.head(2)
train_df.info()
train_df.describe()
train_df.hist(bins=50, figsize=(20, 15))

plt.show()
plt.figure(figsize=(10,10))

ax = sns.heatmap(train_df.corr(),  vmin=-1, vmax=1, center=0,square=True, annot=True, cmap="YlGnBu")

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=30,

    fontsize=10,

    horizontalalignment='right'

)

ax.set_yticklabels(

    ax.get_yticklabels(),

    rotation=30,

    fontsize=10,

    horizontalalignment='right'

)

ax.set_ylim(9, 0);
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

train_df.plot(kind='scatter', x='cement', y='csMPa', ax=ax[0, 0])

train_df.plot(kind='scatter', x='water', y='csMPa', ax=ax[0, 1])

train_df.plot(kind='scatter', x='superplasticizer', y='csMPa', ax=ax[1, 0])

train_df.plot(kind='scatter', x='age', y='csMPa', ax=ax[1, 1])
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

train_df.plot(kind='scatter', x='cement', y='csMPa', ax=ax[0, 0], alpha=0.1)

train_df.plot(kind='scatter', x='water', y='csMPa', ax=ax[0, 1], alpha=0.1)

train_df.plot(kind='scatter', x='superplasticizer', y='csMPa', ax=ax[1, 0], alpha=0.1)

train_df.plot(kind='scatter', x='age', y='csMPa', ax=ax[1, 1], alpha=0.1)
train_dff = train_df.copy()
train_dff.loc[train_dff.flyash == 0, 'flyash_cat'] = '0'

train_dff.loc[(train_dff.flyash  > 0) & (train_dff.flyash  <= 75), 'flyash_cat'] = '0-75'

train_dff.loc[(train_dff.flyash  > 75) & (train_dff.flyash  <= 100), 'flyash_cat'] = '76-100'

train_dff.loc[(train_dff.flyash  > 100) & (train_dff.flyash  <= 125), 'flyash_cat'] = '101-125'

train_dff.loc[(train_dff.flyash  > 125) & (train_dff.flyash  <= 150), 'flyash_cat'] = '126-150'

train_dff.loc[(train_dff.flyash  > 150) & (train_dff.flyash  <= 175), 'flyash_cat'] = '151-175'

train_dff.loc[(train_dff.flyash  > 175), 'flyash_cat'] = '>175'
train_dff.loc[train_dff.slag == 0, 'slag_cat'] = '0'

train_dff.loc[(train_dff.slag  > 0) & (train_dff.slag  <= 50), 'slag_cat'] = '0-50'

train_dff.loc[(train_dff.slag  > 50) & (train_dff.slag  <= 100), 'slag_cat'] = '51-100'

train_dff.loc[(train_dff.slag  > 100) & (train_dff.slag  <= 150), 'slag_cat'] = '101-150'

train_dff.loc[(train_dff.slag  > 150) & (train_dff.slag  <= 200), 'slag_cat'] = '151-200'

train_dff.loc[(train_dff.slag  > 200) & (train_dff.slag  <= 250), 'slag_cat'] = '201-250'

train_dff.loc[(train_dff.slag  > 250) & (train_dff.slag  <= 300), 'slag_cat'] = '251-300'

train_dff.loc[(train_dff.slag  > 300) & (train_dff.slag  <= 350), 'slag_cat'] = '301-350'

train_dff.loc[(train_dff.slag  > 350), 'slag_cat'] = '>350'
train_dff.loc[train_dff.superplasticizer == 0, 'superplasticizer_cat'] = '0'

train_dff.loc[(train_dff.superplasticizer  > 0) & (train_dff.superplasticizer  <= 5), 

             'superplasticizer_cat'] = '0-5'

train_dff.loc[(train_dff.superplasticizer  > 5) & (train_dff.superplasticizer  <= 10), 

             'superplasticizer_cat'] = '6-10'

train_dff.loc[(train_dff.superplasticizer  > 10) & (train_dff.superplasticizer  <= 15), 

             'superplasticizer_cat'] = '11-15'

train_dff.loc[(train_dff.superplasticizer  > 15) & (train_dff.superplasticizer  <= 20), 

             'superplasticizer_cat'] = '16-20'

train_dff.loc[train_dff.superplasticizer  > 20, 'superplasticizer_cat'] = '>20'
sns.catplot('age', 'csMPa', data=train_dff, kind='point')
ax = sns.catplot('flyash_cat', 'csMPa', data=train_dff, kind='point')

ax.set_xticklabels(

    rotation=30,

    fontsize=10,

    horizontalalignment='right'

)
ax = sns.catplot('slag_cat', 'csMPa', data=train_dff, kind='point')

ax.set_xticklabels(

    rotation=30,

    fontsize=10,

    horizontalalignment='right'

)
sns.catplot('superplasticizer_cat', 'csMPa', data=train_dff, kind='point')
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

import eli5

from eli5.sklearn import PermutationImportance



import warnings

warnings.filterwarnings('ignore')
train_df.head(3)
train_df.info()
train_df.describe()
def data_preprocessing(df, label=None):

    return((df.drop([label], axis=1), df[label]) if label else df.drop([label], axis=1))
def get_rmse(scores):

    return np.sqrt(-scores)
x_train, y_train = data_preprocessing(train_df, 'csMPa')

lr = LinearRegression(n_jobs=-1)

rmse = get_rmse(cross_val_score(lr, x_train, y_train, cv=5, scoring='neg_mean_squared_error'))

r2 = cross_val_score(lr, x_train, y_train, cv=5, scoring='r2')

print(rmse.mean(), r2.mean())
x_train, y_train = data_preprocessing(train_df, 'csMPa')

dt = DecisionTreeRegressor(random_state=0)

rmse = get_rmse(cross_val_score(dt, x_train, y_train, cv=5, scoring='neg_mean_squared_error'))

r2 = cross_val_score(dt, x_train, y_train, cv=5, scoring='r2')

print(rmse.mean(), r2.mean())
x_train, y_train = data_preprocessing(train_df, 'csMPa')

rf = RandomForestRegressor(random_state=0)

rmse = get_rmse(cross_val_score(rf, x_train, y_train, cv=5, scoring='neg_mean_squared_error'))

r2 = cross_val_score(rf, x_train, y_train, cv=5, scoring='r2')

print(rmse.mean(), r2.mean())

rf.fit(x_train, y_train);
grf = RandomForestRegressor(random_state=0)

grid_values = [{'n_estimators': [5, 10, 30, 50, 100, 150, 200, 300, 400, 500], 

               'max_features': ['sqrt', 'auto', 2, 3, 5, 8],

              'bootstrap': [False, True]}

              ]

gs = GridSearchCV(grf, param_grid = grid_values, cv = 5, n_jobs=-1, scoring='neg_mean_squared_error')

gs.fit(x_train, y_train)

print(gs.best_params_)
x_train, y_train = data_preprocessing(train_df, 'csMPa')

rfv1 = RandomForestRegressor(bootstrap=False, n_estimators=200, max_features=3, random_state=0)

rmse = get_rmse(cross_val_score(rfv1, x_train, y_train, cv=5, scoring='neg_mean_squared_error'))

r2 = cross_val_score(rfv1, x_train, y_train, cv=5, scoring='r2')

print(rmse.mean(), r2.mean())

rfv1.fit(x_train, y_train);
test_df.head(2)
x_test, y_test = data_preprocessing(test_df, 'csMPa')
y_pred = rf.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print(rmse, r2)
y_pred = rfv1.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print(rmse, r2)
whole_df = pd.read_csv('../input/data-concrete-strength/Concrete_Data_Yeh.csv')
x_train, y_train = data_preprocessing(whole_df, 'csMPa')

rf_full = RandomForestRegressor(bootstrap=False, n_estimators=200, max_features=3, random_state=0)

rmse = get_rmse(cross_val_score(rf_full, x_train, y_train, cv=5, scoring='neg_mean_squared_error'))

r2 = cross_val_score(rf_full, x_train, y_train, cv=5, scoring='r2')

print(rmse.mean(), r2.mean())
shuffled_whole_df = whole_df.sample(frac=1)

x_train, y_train = data_preprocessing(shuffled_whole_df, 'csMPa')

shuffle_rf_full = RandomForestRegressor(bootstrap=False, n_estimators=200, max_features=3, random_state=0)

rmse = get_rmse(cross_val_score(shuffle_rf_full, x_train, y_train, cv=5, scoring='neg_mean_squared_error'))

r2 = cross_val_score(shuffle_rf_full, x_train, y_train, cv=5, scoring='r2')

print(rmse.mean(), r2.mean())

shuffle_rf_full.fit(x_train, y_train);
x_train, y_train = data_preprocessing(train_df, 'csMPa')

x_test, y_test = data_preprocessing(test_df, 'csMPa')

whole_x_train, whole_y_train = data_preprocessing(shuffled_whole_df, 'csMPa')

plot_dict ={'Epoch': [], 'RMSE': [], 'R2': [], 'Data': []}

for i in [150, 200, 250, 300, 400, 500, 800, 1000]:

    

    plot_dict['Epoch'].append(i)

    

    rfv1 = RandomForestRegressor(bootstrap=False, n_estimators=i, max_features=3, random_state=0)

    rmse = get_rmse(cross_val_score(rfv1, x_train, y_train, cv=5, scoring='neg_mean_squared_error'))

    r2 = cross_val_score(rfv1, x_train, y_train, cv=5, scoring='r2')

    

    plot_dict['Data'].append('Train');plot_dict['RMSE'].append(rmse.mean());plot_dict['R2'].append(r2.mean())

    

    rfv1.fit(x_train, y_train);

    y_pred = rfv1.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2 = r2_score(y_test, y_pred)

    

    plot_dict['Epoch'].append(i)

    plot_dict['Data'].append('Test');plot_dict['RMSE'].append(rmse.mean());plot_dict['R2'].append(r2.mean())

    

    rmse = get_rmse(cross_val_score(rfv1,  whole_x_train, whole_y_train, cv=5, scoring='neg_mean_squared_error'))

    

    r2 = cross_val_score(rfv1,  whole_x_train, whole_y_train, cv=5, scoring='r2')

    

    plot_dict['Epoch'].append(i)

    plot_dict['Data'].append('Whole');plot_dict['RMSE'].append(rmse.mean());plot_dict['R2'].append(r2.mean())



plot_df = pd.DataFrame.from_dict(plot_dict)

plot_df.head(2)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.lineplot(x='Epoch',y='RMSE', hue='Data', data=plot_df,ax=ax[0])

sns.lineplot(x='Epoch',y='R2', hue='Data', data=plot_df,ax=ax[1])
perm = PermutationImportance(rfv1, random_state=0).fit(x_test, y_test)

eli5.show_weights(perm, feature_names = x_test.columns.tolist())
perm = PermutationImportance(shuffle_rf_full, random_state=0).fit(x_test, y_test)

eli5.show_weights(perm, feature_names = x_test.columns.tolist())
x_train, y_train = data_preprocessing(shuffled_whole_df[['age', 'cement', 'water', 'superplasticizer', 'csMPa']], 

                                      'csMPa')

feat4_rf_full = RandomForestRegressor(bootstrap=False, n_estimators=200, max_features=3, random_state=0)

rmse = get_rmse(cross_val_score(feat4_rf_full, x_train, y_train, cv=5, scoring='neg_mean_squared_error'))

r2 = cross_val_score(feat4_rf_full, x_train, y_train, cv=5, scoring='r2')

print(rmse.mean(), r2.mean())
x_train, y_train = data_preprocessing(shuffled_whole_df[['age', 'cement', 'water', 'superplasticizer', 

                                                         'slag', 'csMPa']], 'csMPa')

feat5_rf_full = RandomForestRegressor(bootstrap=False, n_estimators=200, max_features=3, random_state=0)

rmse = get_rmse(cross_val_score(feat5_rf_full, x_train, y_train, cv=5, scoring='neg_mean_squared_error'))

r2 = cross_val_score(feat5_rf_full, x_train, y_train, cv=5, scoring='r2')

print(rmse.mean(), r2.mean())