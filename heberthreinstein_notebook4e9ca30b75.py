import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.style as style
import matplotlib.gridspec as gridspec
import seaborn as sns
from mpl_toolkits import mplot3d
from scipy import stats
%matplotlib inline

# preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
import pandas_profiling as pp

# models
from sklearn.linear_model import LinearRegression,LogisticRegression, SGDRegressor, RidgeCV
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection
from sklearn.model_selection import cross_val_predict as cvp
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import pearsonr

import xgboost as xgb
import lightgbm as lgb

# model tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval

import warnings
warnings.filterwarnings("ignore")


from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import pearsonr

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
housedf = pd.read_csv('/kaggle/input/housedata/data.csv')
housedf.head()
contagem = housedf['city'].value_counts()
cidades = contagem.index
plt.figure(figsize=(20,20))

for n, i in enumerate(contagem):
    plt.barh(n,i)
    plt.text(i+20,n,str(cidades[n]+' '+ str(i) ))

plt.title("Histograma Número Casas")
plt.xlabel("cidade")
plt.ylabel("contagem")
plt.show()
plt.pie(housedf['floors'].value_counts(), labels=(1, 2, 3, 4, 5, 6))
plt.show()
figure(figsize=(20, 10))
plt.scatter(x = housedf['price'],
            y = housedf['city'])
plt.title("Preço x Cidade")
plt.xlabel("Preço")
plt.ylabel("Cidade")
plt.show()
plt.barh(housedf['bathrooms'], housedf['price'])
plt.title("Banheiros x Casas")
plt.xlabel("Casas")
plt.ylabel("Banheiros")
plt.show()
variaveis = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_above','sqft_basement','yr_built','yr_renovated']

for i in variaveis:
    sns.lmplot(x = i, y ='price', data = housedf)
features = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'street', 'city', 'statezip', 'country']
numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
features = housedf.columns.values.tolist()
for col in features:
    if housedf[col].dtype in numerics: continue
    categorical_columns.append(col)
# Encoding categorical features
for col in categorical_columns:
    if col in housedf.columns:
        le = LabelEncoder()
        le.fit(list(housedf[col].astype(str).values))
        housedf[col] = le.transform(list(housedf[col].astype(str).values))

housedf['price'] = (housedf['price']).astype(int)
housedf['floors'] = (housedf['floors']).astype(int)
housedf['bedrooms'] = (housedf['bedrooms']).astype(int)

housedf.sample(5)
val = housedf['price']
train = housedf.drop(['price'], axis=1)
#feature = feature.drop(['city','street','country','statezip'],axis=1)
val.head()
train.head()
Xtrain, Xval, Ztrain, Zval = train_test_split(train, val, test_size=0.1, random_state=2)
linreg = LinearRegression()
linreg.fit(Xtrain, Ztrain)
print('Coefiente Linear do treinamento')
linreg.score(Xval, Zval)
Xval[10:13]
linreg.predict(Xval[10:13])
Zval[10:13]