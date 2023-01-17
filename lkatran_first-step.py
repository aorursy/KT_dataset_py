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
data = pd.read_csv('/kaggle/input/house-prices/data.csv')
data.columns
data.head(2)
data.shape
data.isna().sum()/(data.shape[0]/100)
data.drop(data[data.target.isna()].index, inplace=True)
data.shape
import re
def make_target(x):
    x = re.sub('[^0-9]', '', x)
    x = int(x)
    return x
data.target = data.target.apply(make_target)
data.target.describe()
def make_features_float(x):
    if x == -1.0: return x
    x = re.sub('1 1/2', '1.5', x)
    x = re.sub('[^0-9,\.]', '', x)
    x = re.sub(',', '.', x)
    try:
        x = float(x)
    except:
        x = -1.0
    return x
def make_status(x):
    if x.startswith('Coming soon'): x = 'coming soon'
    x = x.lower()
    x = re.sub('[^a-z]', ' ', x)
    x = re.sub(r'\b\w{,2}\b', '', x)
    x = re.sub(r'\s+', ' ', x)
    return x

def make_propertyType(x):
    x = x.lower()
    x = re.sub('[^a-z]', ' ', x)
    x = 1 if x.startswith('single family') else 0
    return x

def make_fireplace(x):
    if x == -1: return x
    x = x.lower()
    x = re.sub('yes', '1', x)
    x = re.sub('no', '0', x)
    if 'fireplace' in x: 
        x = '1'
    try:
        x = int(x)
    except:
        x = 0
    return x
cat_features = ['status', 'state']
columns_to_drop = ['street', 'mls-id', 'MlsId', 'schools', 'homeFacts', 'city', 'zipcode']
data['status'] = data['status'].fillna('')
data['status'] = data['status'].apply(make_status)
data['city'] = data['city'].fillna('other')
top_city = data['city'].value_counts()[:200].index
data['city'] = data['city'].apply(lambda r: r if r in top_city else 'other')
data['propertyType'] = data['propertyType'].fillna('')
data['propertyType'] = data['propertyType'].apply(make_propertyType)
data['fireplace'] = data['fireplace'].fillna(-1)
data['fireplace'] = data['fireplace'].apply(make_fireplace)
data['private pool'] = data['private pool'].fillna('No')
data['private pool'] = data['private pool'].map({'Yes':1, 'No':0})
data['PrivatePool'] = data['PrivatePool'].fillna('No')
data['PrivatePool'] = data['PrivatePool'].map({'Yes':1, 'No':0, 'yes':1})
data['PrivatePool'] = data['private pool'] | data['PrivatePool']
data.drop(['private pool'], axis=1, inplace=True)
data.baths = data.baths.fillna(-1.0)
data.baths = data.baths.apply(make_features_float)
data.sqft = data.sqft.fillna(-1.0)
data.sqft = data.sqft.apply(make_features_float)
data.beds = data.beds.fillna(-1)
data.beds = data.beds.apply(make_features_float)
data.stories = data.stories.fillna(-1)
data.stories = data.stories.apply(make_features_float)
data['homeFacts'] = data['homeFacts'].apply(eval)
def make_homeFacts(x):
    x = x.get('atAGlanceFacts', -1)
    if x == -1: return -1
    x = x[0]
    if x.get('factLabel') == 'Year built':
        x = x.get('factValue')
    try:
        x = int(x)
    except:
        x = -1
    return x
data['year_built'] = data['homeFacts'].apply(make_homeFacts)
data.drop(columns_to_drop, axis=1, inplace=True)
data.describe(include='all')
data = pd.concat([data, pd.get_dummies(data['status'])], axis=1)

data.drop('status', axis=1, inplace=True)
# data = pd.concat([data, pd.get_dummies(data['zipcode'])], axis=1)
# data.drop('zipcode', axis=1, inplace=True)
data = pd.concat([data, pd.get_dummies(data['state'])], axis=1)
data.drop('state', axis=1, inplace=True)
x_train, x_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=0)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
from sklearn.linear_model import LinearRegression, BayesianRidge
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html?highlight=sgdregressor
from sklearn.linear_model import SGDRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
from sklearn.tree import DecisionTreeRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html?highlight=kneighborsregressor
from sklearn.neighbors import KNeighborsRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html?highlight=svr#sklearn.svm.SVR
from sklearn.svm import SVR
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=randomforestregressor#sklearn.ensemble.RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html?highlight=randomforestregressor
from sklearn.ensemble import ExtraTreesRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html?highlight=randomforestregressor
from sklearn.ensemble import GradientBoostingRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html#sklearn.ensemble.StackingRegressor
from sklearn.ensemble import StackingRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html#sklearn.ensemble.VotingRegressor
from sklearn.ensemble import VotingRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html#sklearn.ensemble.HistGradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html#sklearn.ensemble.BaggingRegressor
from sklearn.ensemble import BaggingRegressor
import sklearn.metrics
'''про ошибки подробнее тут https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics'''
def my_evaluate(clf, x_train, y_train, x_test, y_test, mode='all'):
    own_predict = clf.predict(x_train)
    '''мы знаем, что дома не могут стоить отрицательную цену, поэтому преобразуем предсказание'''
    own_predict = np.where(own_predict < 1, 1, own_predict)
    '''найдем значение эквивалентное 1% в денежном выражении'''
    one_percent = (y_train.max() - y_train.min())/100
    
    predict = clf.predict(x_test)
    predict = np.where(predict < 1, 1, predict)
    if mode=='all':
        print(f'''\t\t\t\town evaluate \t\t\tevaluate on test
Explained variance score:\t{metrics.explained_variance_score(y_test, predict)}\t\t{metrics.explained_variance_score(y_test, predict)}\n
Max error:\t\t\t{metrics.max_error(y_train, own_predict)/one_percent}\t\t{metrics.max_error(y_test, predict)/one_percent}\n
Mean absolute error:\t\t{metrics.mean_absolute_error(y_train, own_predict)/one_percent}\t\t{metrics.mean_absolute_error(y_test, predict)/one_percent}\n
Mean squared error:\t\t{metrics.mean_squared_error(y_train, own_predict)/(one_percent**2)}\t\t{metrics.mean_squared_error(y_test, predict)/(one_percent**2)}\n
Mean squared log error:\t\t{metrics.mean_squared_log_error(y_train, own_predict)}\t\t{metrics.mean_squared_log_error(y_test, predict)}\n
Median absolute error:\t\t{metrics.median_absolute_error(y_train, own_predict)/one_percent}\t\t{metrics.median_absolute_error(y_test, predict)/one_percent}\n
R^2 score:\t\t\t{metrics.r2_score(y_train, own_predict)}\t\t{metrics.r2_score(y_test, predict)}''')
    else:
        print(f'''\t\t\t\town evaluate \t\t\tevaluate on test
Mean squared log error:\t\t{metrics.mean_squared_log_error(y_train, own_predict)}\t\t{metrics.mean_squared_log_error(y_test, predict)}\n
''')
my_lr = LinearRegression()
my_lr.fit(x_train, y_train)
my_evaluate(my_lr, x_train, y_train, x_test, y_test)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
pipe_standart = Pipeline([('scaler', StandardScaler()), ('lin_reg', LinearRegression())])
pipe_standart.fit(x_train, y_train)
my_evaluate(pipe_standart, x_train, y_train, x_test, y_test)
pipe_robust = Pipeline([('scaler', RobustScaler()), ('lin_reg', LinearRegression())])
pipe_robust.fit(x_train, y_train)
my_evaluate(pipe_robust, x_train, y_train, x_test, y_test)
'''функция для отбора К лучших признаков по их статистической близости к целевой переменной'''
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression # числовые признаки и числовой выходной признак
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
'''обучаем функцию выбора на основе критерия кси-квадрат и извлекаем лучшие 10 признаков'''
bestfeatures = SelectKBest(score_func=f_regression, k=50)
fit = bestfeatures.fit(x_train,y_train)
'''создаем набор данных признаков с их весом и выбираем 10 лучших'''
featureScores =  pd.DataFrame({'Features':x_train.columns, 'Score': fit.scores_})
print(featureScores.nlargest(50,'Score'))
'''тут стоит поиграть с количеством признаков и пронаблюдать результаты'''
x_train_6 = x_train[featureScores.nlargest(6,'Score').Features.values]
x_test_6 = x_test[featureScores.nlargest(6,'Score').Features.values]
my_lr_6 = LinearRegression()
my_lr_6.fit(x_train_6, y_train)
my_evaluate(my_lr, x_train, y_train, x_test, y_test, 1)
my_evaluate(my_lr_6, x_train_6, y_train, x_test_6, y_test, 1)
embeded_lr_selector = SelectFromModel(LinearRegression(), max_features=6)
embeded_lr_selector.fit(x_train, y_train)

embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = x_train.loc[:,embeded_lr_support].columns.tolist()
print(featureScores.nlargest(6,'Score'))
print(str(embeded_lr_feature), 'selected features')
my_lr_10 = LinearRegression()
my_lr_10.fit(x_train[embeded_lr_feature], y_train)
my_evaluate(my_lr, x_train, y_train, x_test, y_test, 1)
my_evaluate(my_lr_10, x_train[embeded_lr_feature], y_train, x_test[embeded_lr_feature], y_test, 1)
embeded_lr_feature.extend(featureScores.nlargest(6,'Score').Features.values)
my_lr_12 = LinearRegression()
my_lr_12.fit(x_train[embeded_lr_feature], y_train)
my_evaluate(my_lr, x_train, y_train, x_test, y_test, 1)
my_evaluate(my_lr_12, x_train[embeded_lr_feature], y_train, x_test[embeded_lr_feature], y_test, 1)
import seaborn as sns

corrmat = data.corr()
top_corr_features = corrmat.index
# plt.figure(figsize=(20,20))
#plot heat map
# g=sns.heatmap(x_train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
data[top_corr_features].corr().loc['target', data[top_corr_features].corr().loc['target', :]>0.05]
dtr = DecisionTreeRegressor()
dtr.fit(x_train, y_train)
dtr_6 = DecisionTreeRegressor()
dtr_6.fit(x_train_6, y_train)
my_evaluate(dtr, x_train, y_train, x_test, y_test)
my_evaluate(dtr_6, x_train_6, y_train, x_test_6, y_test)





