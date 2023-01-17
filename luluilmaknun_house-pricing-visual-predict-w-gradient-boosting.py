import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')



sns.set_style('darkgrid')
test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')

train = train.set_index('Id')

test = test.set_index('Id')

feature_y = 'SalePrice'
train.head()
train.describe()
# divides data by types

features = train.columns.values

float_data = []

int_data = []

object_data = []

year_data = []



for feature in features:

    dtype = train[feature].dtypes

    if feature == 'SalePrice':

        continue

    elif 'Yr' in feature or 'Year' in feature:

        year_data.append(feature)

    elif dtype == 'int64':

        int_data.append(feature)

    elif dtype == 'float64':

        float_data.append(feature)

    else:

        object_data.append(feature)
float_data
int_data
year_data
object_data
for feature in float_data:

    sns.jointplot(x=feature, y=feature_y, data=train, kind="reg", height=7)
# visualization for int_data, will learn how to grid in seaborn T,T

for feature in int_data:

    sns.jointplot(x=feature, y=feature_y, data=train, kind="reg", height=7)
len_data = len(object_data)

fig, ax = plt.subplots(len_data//3+1, 3, figsize=(20,70))

for num in range(len_data):

    sns.barplot(ax=ax[num//3,num%3],x=object_data[num], y=feature_y, data=train)
len_data = len(year_data)

fig, ax = plt.subplots(len_data//3+1, 3, figsize=(20,7))

for num in range(len_data):

    sns.lineplot(ax=ax[num//3,num%3],x=year_data[num], y=feature_y, data=train)
train.info()
# na_value in columns

na_val = train.isna().sum().reset_index(name="total_na").sort_values(by="total_na", ascending=False)

fig,ax = plt.subplots(1, figsize=(8,20))

sns.barplot(ax=ax,x="total_na", y="index", data=na_val)
## list of feature maybe deleteable

to_be_removed = ['PoolQC'

                , 'MiscFeature'

                , 'Fence'

                , 'Alley'

                , 'MSSubClass'

                , 'LowQualFinSF'

                , 'MoSold'

                , 'YrSold'

                , 'OverallCond'

                , 'BsmtFinSF2'

#                 , 'KitchenAbvGr'

#                 , 'EnclosedPorch'

                , 'MiscVal'

#                 , 'LotArea'

                , 'BsmtUnfSF'

                , 'BsmtHalfBath'

#                 , 'FireplaceQu'.

#                  ,'BedroomAbvGr'

                

                ]

train = train.drop(to_be_removed, 1)

test = test.drop(to_be_removed, 1)
object_data.remove('PoolQC')

object_data.remove('MiscFeature')

object_data.remove('Fence')

object_data.remove('Alley')

int_data.remove('MSSubClass')

int_data.remove('LowQualFinSF')

int_data.remove('MoSold')

year_data.remove('YrSold')

int_data.remove('OverallCond')

int_data.remove('BsmtFinSF2')

# int_data.remove('KitchenAbvGr')

# int_data.remove('EnclosedPorch')

int_data.remove('MiscVal')

# int_data.remove('LotArea')

int_data.remove('BsmtUnfSF')

int_data.remove('BsmtHalfBath')

# object_data.remove('FireplaceQu')

# int_data.remove('BedroomAbvGr')
## outlier handling

from scipy import stats



def remove_outlier(df, data):

    for feature in (data):

        col = pd.DataFrame(df[feature].dropna())

        col_in = col[(np.abs(stats.zscore(col)) < 3).all(axis=1)]

        col_mean = int(col_in.mean())

        col_out = col[(np.abs(stats.zscore(col)) >= 3).all(axis=1)]

        df.loc[list(col_out.index.values), feature] = col_mean



remove_outlier(train, int_data)

remove_outlier(train, float_data)

remove_outlier(test, int_data)

remove_outlier(test, float_data)
## remove column with low variance



for feature in int_data:

    var_test = test[feature].dropna().var()

    var = train[feature].dropna().var()

    if  var <= 1:

        train = train.drop(feature, 1)

        test = test.drop(feature, 1)

        int_data.remove(feature)
## fill na, maybe will increase



def fill_na(df):

    null_columns = df.columns[df.isnull().any()]

    null_rows = df[df.isnull().any(axis=1)][null_columns]

    

    garage_yr_med = df['GarageYrBlt'].dropna().median()

    df['GarageYrBlt'].fillna(garage_yr_med, inplace=True)

    

    masvnr_mean = df['MasVnrArea'].dropna().mean()

    #masvnr_mode = train['MasVnrArea'].dropna().mode()

    df['MasVnrArea'].fillna(masvnr_mean, inplace=True)

    

    lot_mean = df['LotFrontage'].dropna().mean()

    df['LotFrontage'] = pd.DataFrame(df['LotFrontage']).fillna(lot_mean)

    

    elec_mode = df['Electrical'].dropna().mode()

    df['Electrical'] = pd.DataFrame(df['Electrical']).fillna(elec_mode[0])

    

fill_na(train)    

fill_na(test)
def get_req(df, col):

    data = df[col]

    uniq_data = list(data.unique())

    category = {}

    

    for num in range(len(uniq_data)):

        category[uniq_data[num]] = num

    

    return category



def change_cate(df, col):

    req = get_req(df, col)

    df[col].replace(req, inplace=True)

    return df
for obj in object_data:

    train = change_cate(train, obj)

    test = change_cate(test, obj)
corr = train.corr()

fig,ax = plt.subplots(1, figsize=(15,15))

sns.heatmap(ax=ax,data=corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True)
# tired

train.fillna(0, inplace=True)    # best safe "how to fill na" if tired

test.fillna(0, inplace=True)

x = train.drop('SalePrice', 1)

y = train['SalePrice']
train
test
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import robust_scale, StandardScaler, MinMaxScaler



from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn import ensemble

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPRegressor, MLPClassifier



from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score
# x = robust_scale(x)

# scaler = StandardScaler()

# scaler = MinMaxScaler()

# scaler.fit(x)

# scaler.fit(test)

# # test = robust_scale(test)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



# print(x_train.shape)

# print(x_test.shape)
# # models = {'Lr': LinearRegression(),

# #         'Lgr' : LogisticRegression(),

# #         'Dtc' : DecisionTreeClassifier(),

# #         'Rfc' :RandomForestClassifier(),

# #         'Gnb' : GaussianNB(),

# #         'Lsvc' :LinearSVC(),

# #         'KNN' : KNeighborsClassifier(),

# #         'SCV' : SVC()}



# # model = MLPRegressor(hidden_layer_sizes=(4,),

# #                                        activation='relu',

# #                                        solver='adam',

# #                                        learning_rate='adaptive',

# #                                        max_iter=100,

# #                                        learning_rate_init=0.01,

# #                                        alpha=0.02)



# params = {'n_estimators': 575, 'max_depth': 5, 'min_samples_split': 3,

#           'learning_rate': 0.035, 'loss': 'ls', 'max_features' : 30}

# model = ensemble.GradientBoostingRegressor(**params) 



# # model = MLPClassifier(hidden_layer_sizes=(100,),

# #                                        activation='relu',

# #                                        solver='adam',

# #                                        learning_rate='adaptive',

# #                                        max_iter=100,

# #                                        learning_rate_init=0.03,

# #                                        alpha=0.05)

# fitted = model.fit(x_train, y_train)

# y_pred = fitted.predict(x_test)

# print(mean_squared_error(y_pred,y_test))
best_model = None

best_squared = 10000000000000



for i in range(50):

    scaler = StandardScaler()

    scaler.fit(x)

    scaler.fit(test)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    params = {'n_estimators': 575, 'max_depth': 5, 'min_samples_split': 3,

          'learning_rate': 0.35, 'loss': 'ls', 'max_features' : 30}

    model = ensemble.GradientBoostingRegressor(**params)

    fitted = model.fit(x_train, y_train)

    y_pred = fitted.predict(x_test)

    score = mean_squared_error(y_pred, y_test)

    

    if score < best_squared:

        best_model = fitted

        best_squared = score

    

    print(i, score)

    

print('best', best_squared)
predictions = fitted.predict(test)
predictions