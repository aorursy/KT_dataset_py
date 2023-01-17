import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



# classifiers

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
import warnings



warnings.filterwarnings('ignore')

pd.set_option('Display.max_columns', None)

sns.set_style('darkgrid')

%matplotlib inline

sns.set()
data = pd.read_csv('train_data.csv', na_values='?').drop(0, axis = 0).reset_index(drop = True)
data.columns=['Id', 'age', 'workclass', 'final_weight', 'education', 'education_num', 'marital_status',

              'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',

              'native_country', 'income']
data.head(3)
data.describe()
# 3 colunas apresentam valores nulos

# diversas colunas numéricas estão apresentadas como strings

data.replace('?', np.nan, inplace=True)

data.info()
def work_missing_values(data):

    '''

    Return new data with no missing values for this problem

    '''

    

    aux = data.copy()

    # select index of rows that workclass is nan

    aux_index = aux[aux['workclass'].isna()].index

    

    # fill nan with 'unknown'

    aux['workclass'].loc[aux_index] = 'unknown'

    aux['occupation'].loc[aux_index] = 'unknown'

    

    # complete missing of native_country and occupation with most frequent

    cols = ['native_country', 'occupation']

    for col in cols:

        top = aux[col].value_counts().index[0]

        aux[col] = aux[col].fillna(top)

    aux.reset_index(drop = True)

    

    return aux
data = work_missing_values(data)
%%time

for column in data.columns:

    try:

        data[columns] = pd.to_numeric(data[columns])

    except:

        None
# simple way to get numerical columns (not memory responsible)

num_cols = list(data.describe().columns)
%%time

sns.pairplot(data[num_cols+['income']], vars=num_cols, 

             hue='income', palette='bwr');
fig, ax = plt.subplots(figsize=(20,8))

for index, col in enumerate(num_cols[1:]):

    #print(data[col].nunique())

    g1 = data[data.income=='>50K'].groupby([col], as_index=False).agg({'income':'count'})

    g2 = data[data.income=='<=50K'].groupby([col], as_index=False).agg({'income':'count'})

    z = g1.merge(g2, on=col, how='outer', suffixes=('_high', '_low')).fillna(0)

    z['income_high'] = LabelEncoder().fit_transform(z.income_high)

    z['income_low'] = LabelEncoder().fit_transform(z.income_low)

    plt.subplot(2,3,index+1)

    plt.scatter(x=z[col], y=[1 for i in range(z.shape[0])], s=10*z.income_high, color='blue')

    plt.scatter(x=z[col], y=[0 for i in range(z.shape[0])], s=10*z.income_low, color='red')

    plt.title(col)
categ_cols = ['workclass', 'education', 'marital_status', 'occupation',

              'relationship','race', 'sex', 'native_country']
def bar_plots(data, categ_cols):

    fig, ax = plt.subplots(figsize=(20, 3*len(categ_cols)/2))

    for i, col in enumerate(categ_cols):

        axi = plt.subplot(len(categ_cols)/2, 2, i+1)

        h = data[data.income=='>50K'][col].value_counts().reset_index()

        l = data[data.income=='<=50K'][col].value_counts().reset_index()

        aux = l.merge(h, on='index', how='left')

        # normalized data

        aux.iloc[:, 1:] = aux.iloc[:, 1:].div(aux.iloc[:, 1:].sum(axis=1), axis=0)

        aux.plot(kind='bar', x='index', y=col+'_x', ax=axi)

        aux.plot(kind='bar', x='index', y=col+'_y', color='firebrick', alpha=0.6, ax=axi)

        plt.xticks(rotation=80);

        plt.title(col)

        plt.legend(['<=50K', ' >50K'])

        plt.xlabel('')

    plt.subplots_adjust(hspace = 1)
bar_plots(data, categ_cols[:4])
bar_plots(data, categ_cols[4:])
def box_plot(data, num_cols, var_x='income', orientation = 'v',

             rotate_x_label = False):

    

    fig, ax = plt.subplots(figsize=(20, 3*len(num_cols)/2))

    for i, col in enumerate(num_cols[1:]):

        axi = plt.subplot(len(num_cols)/2, 2, i+1)



        df = pd.concat([data[col], data[var_x]], axis=1)

        sns.boxplot(x=var_x, y=col, data=df, ax=axi, notch = True, 

                    palette = 'Wistia', orient = orientation)

        plt.title('{} distribution analysis'.format(col))

        if rotate_x_label:

            ax.set_xticklabels(data[var_x].unique(), rotation=90)

    plt.subplots_adjust(hspace = 0.7)
box_plot(data, num_cols)
col_names=['Id', 'age', 'workclass', 'final_weight', 'education', 'education_num', 'marital_status',

           'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',

           'native_country', 'income']



test = pd.read_csv('test_data.csv', names = col_names[:-1]).drop(0, axis = 0).reset_index(drop = True)
train = pd.read_csv('train_data.csv', names = col_names).drop(0, axis = 0).reset_index(drop = True)



train.replace('?', 'unknown', inplace=True)

train['native_country'] = train.native_country.apply(lambda x: 1 if x=='United-States' else 0)

train['marital_status'] = train.marital_status.apply(lambda x: 1 if x in ['Married-civ-spouse' or 'Married-AF-spouse'] else 0)

train['workclass'] = train.workclass.apply(lambda x: 2 if x=='Self-emp-inc' else (0 if x in ['Without-pay', 'Never-worked'] else 1))

train['occupation'] = train.occupation.apply(lambda x: 2 if x in ['Exec-managerial', 'Prof-specialty'] else (0 if x =='Priv-house-serv' else 1))

train['relationship'] = train.relationship.apply(lambda x: 1 if x in ['Husband', 'Wife'] else 0)

categ_features = ['race', 'sex', 'native_country']

for feature in categ_features:

    train[feature] = LabelEncoder().fit_transform(train[feature])

X_train, y_train = train[col_names[:-1]], train['income']

X_train.drop(columns=['education', 'Id'], inplace=True)

for col in X_train.columns:

    try:

        X_train[col] = pd.to_numeric(X_train[col])

    except:

        None

X_train = StandardScaler().fit_transform(X_train)
test = pd.read_csv('test_data.csv', names = col_names).drop(0, axis = 0).reset_index(drop = True)



test.replace('?', 'unknown', inplace=True)

test['native_country'] = test.native_country.apply(lambda x: 1 if x=='United-States' else 0)

test['marital_status'] = test.marital_status.apply(lambda x: 1 if x in ['Married-civ-spouse' or 'Married-AF-spouse'] else 0)

test['workclass'] = test.workclass.apply(lambda x: 2 if x=='Self-emp-inc' else (0 if x in ['Without-pay', 'Never-worked'] else 1))

test['occupation'] = test.occupation.apply(lambda x: 2 if x in ['Exec-managerial', 'Prof-specialty'] else (0 if x =='Priv-house-serv' else 1))

test['relationship'] = test.relationship.apply(lambda x: 1 if x in ['Husband', 'Wife'] else 0)

categ_features = ['race', 'sex', 'native_country']

for feature in categ_features:

    test[feature] = LabelEncoder().fit_transform(test[feature])

X_test, y_test = test[col_names[:-1]], test['income']

X_test.drop(columns=['education', 'Id'], inplace=True)

for col in X_test.columns:

    try:

        X_test[col] = pd.to_numeric(X_test[col])

    except:

        None

X_test = StandardScaler().fit_transform(X_test)
def explanation_fn(estimator, instance):

    '''

    fixed function for lime explanation for estimator and given example instance

    '''

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, training_labels=y_train, 

                                                   feature_names=X_train.columns, categorical_features = [1,3,4,5,6,12], 

                                                   class_names = ['<=50K', '>50K'])



    exp = explainer.explain_instance(X_test[instance], estimator.predict_proba, num_features=6, top_labels=None)

    exp.show_in_notebook(show_table=True, show_all=False)
def outputPrediction(ids, predictions):

    data = pd.DataFrame({'Id': ids, 'income': predictions})

    return data
%%time

time_train = [2000]



# train



LogClf = LogisticRegression(solver = 'lbfgs', C = 1.0, penalty = 'l2', warm_start =  True)



LogCV = cross_val_score(LogClf, X_train, y_train, cv = 10)



LogClf.fit(X_train, y_train)



cv_accuracy = [LogCV.mean()]

cv_std = [LogCV.std()]



cv_values = {}

cv_values['Lin'] = LogCV

print('Logistic Regression CV accuracy: {0:1.4f} +-{1:2.5f}\n'.format(LogCV.mean(), LogCV.std()))
%%time

time_train.append(30500)



# train



KNNClf = KNeighborsClassifier(n_neighbors = 19, p = 1, weights = 'uniform')



KNNCV = cross_val_score(KNNClf, X_train, y_train, cv = 10)



KNNClf.fit(X_train, y_train)



cv_accuracy.append(KNNCV.mean())

cv_std.append(KNNCV.std())

cv_values['KNN'] = KNNCV

print('K-Nearest Neighboors CV accuracy: {0:1.4f} +-{1:2.5f}\n'.format(KNNCV.mean(), KNNCV.std()))
%%time

time_train.append(170000)



# train



RFClf = RandomForestClassifier(n_estimators = 750, max_depth = 12)



RFCV = cross_val_score(RFClf, X_train, y_train, cv = 10)



RFClf.fit(X_train, y_train)



cv_accuracy.append(RFCV.mean())

cv_std.append(RFCV.std())

cv_values['RF'] = RFCV

print('Random Forest CV accuracy: {0:1.4f} +-{1:2.5f}\n'.format(RFCV.mean(), RFCV.std()))
%%time

time_train.append(60000)



# train

XGBClf = XGBClassifier(max_depth = 4, n_estimators = 250)



XGBCV = cross_val_score(XGBClf, X_train, y_train, cv = 10)



XGBClf.fit(X_train, y_train)



cv_accuracy.append(XGBCV.mean())

cv_std.append(XGBCV.std())

cv_values['XGB'] = XGBCV

print('XGBoost CV accuracy: {0:1.4f} +-{1:2.5f}\n'.format(XGBCV.mean(), XGBCV.std()))
y_pred = XGBClf.predict(X_test)
test['income'] = y_pred

test['income'] = test['income'].replace({0:'<=50K', 1:'>50k'})

test[['Id', 'income']].to_csv('sample_submission_Adult.csv', index=False)
data = pd.read_csv('../Extra/train.csv', na_values='?').reset_index(drop = True)
data.head()
data.describe()
data.replace('?', np.nan).info()
def plotMap(data, sizes = None, colors = None, cmap = 'Blues', alpha = 0.7, title = 'Mapa'):

    '''

    plot on cartesian plan, coordinatedes according to lat long, with circle sizes em color scale

    '''

    v_sizes, v_colors = None, None

    if sizes is not None:

        scaler = MinMaxScaler()

        v_sizes = scaler.fit_transform(data[sizes].values.reshape(-1,1))*100

        v_sizes = v_sizes.reshape(-1)

        

    if colors is not None:

        v_colors = data[colors]

        

    with plt.style.context('seaborn-whitegrid'):

        data.plot.scatter('longitude', 'latitude', s = v_sizes, figsize = (11,7), c = v_colors, cmap = cmap, alpha = alpha)

        plt.title(title)
plotMap(data, sizes = 'median_income', colors = 'median_house_value', title = 'Income and house value visualization map')
plotMap(data, sizes = 'population', colors = 'median_age', title = 'Population and ages visualization map', alpha = 0.7)
plt.figure(figsize=(14,7))

sns.distplot(data['median_income'], color = 'Red', bins = 20);
plt.figure(figsize=(14,7))

sns.distplot(data['total_rooms'], color = 'Red', bins = 200);
plt.figure(figsize=(14,7))

sns.distplot(data['median_house_value'], color = 'Red', bins = 20);
plt.figure(figsize=(14,7))

sns.boxplot(x = data['median_age'], y = data['median_house_value'], palette = 'Reds')
from scipy import stats
#Retirando outliers da base

data_clean = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]



#Reindexando para ajustar termos faltantes

data_clean = data_clean.assign(index = list(range(0, data_clean.iloc[:,0].size)))

data_clean = data_clean.set_index('index')
selected_columns = ['longitude', 'median_income', 'median_age', 'population']

target = 'median_house_value'



selected_base = pd.concat([data_clean[selected_columns], data_clean[target]], axis = 1)
scaler = {}

for col in selected_columns:

    scaler[col] = StandardScaler()

    selected_base[col] = scaler[col].fit_transform(selected_base[col].values.reshape(-1,1))

    

scaler[target] = MinMaxScaler()

selected_base[target] = scaler[target].fit_transform(selected_base[target].values.reshape(-1,1))
X, y = selected_base[selected_columns].values, selected_base[target].values
# metrica para avaliar os regressores



from sklearn.metrics import mean_squared_log_error, make_scorer



msle = make_scorer(mean_squared_log_error)
%%time

time_train = [40]



# train



LinReg = LinearRegression()



LinCV = cross_val_score(LinReg, X, y.reshape(-1), cv = 10, scoring = msle)



LinReg.fit(X, y)



cv_accuracy = [LinCV.mean()]

cv_std = [LinCV.std()]

cv_values = {}

cv_values['Lin'] = LinCV

print('Linear Regression CV msle: {0:1.4f} +-{1:2.5f}\n'.format(LinCV.mean(), LinCV.std()))
%%time

time_train.append(310)



# train



KNNReg = KNeighborsRegressor(n_neighbors=30)



KNNCV = cross_val_score(KNNReg, X, y, cv = 10, scoring = msle)



KNNReg.fit(X, y)



cv_accuracy.append(KNNCV.mean())

cv_std.append(KNNCV.std())

cv_values['KNN'] = KNNCV

print('KNN Regression CV msle: {0:1.4f} +-{1:2.5f}\n'.format(KNNCV.mean(), KNNCV.std()))
%%time

time_train.append(18700)



# train



RFReg = RandomForestRegressor(n_estimators = 50, max_depth = 14)



RFCV = cross_val_score(RFReg, X, y.reshape(-1), cv = 10, scoring = msle)



RFReg.fit(X, y.reshape(-1))



cv_accuracy.append(RFCV.mean())

cv_std.append(RFCV.std())

cv_values['RF'] = RFCV

print('RF Regression CV msle: {0:1.4f} +-{1:2.5f}\n'.format(RFCV.mean(), RFCV.std()))