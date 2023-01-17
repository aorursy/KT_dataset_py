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
df = pd.read_csv("../input/insurance/insurance.csv")

df
f_na = [feature for feature in df.columns if df[feature].isna().sum()>0]

f_na
f_num = [feature for feature in df.columns if df[feature].dtype!='object']

f_num
# Discrete Numerical Features

f_dis = [feature for feature in f_num if len(df[feature].unique())<25]

print(f_dis)

# Continuous Numerical Features

f_cont = [feature for feature in f_num if feature not in f_dis]

print(f_cont)
import matplotlib.pyplot as plt

for feature in f_cont:

    plt.hist(df[feature], bins=50)

    plt.title(feature)

    plt.show()
for feature in f_cont:

    data = df.copy()

    data[feature] = np.log(data[feature])

    plt.hist(data[feature], bins=50)

    plt.title(feature)

    plt.show()
import statsmodels.api as sm

import scipy.stats as stats

import pylab 

fig, axs = plt.subplots(len(f_cont), 2, figsize=(15,15))



for i in range(len(f_cont)):

    data = df.copy()

    data[f_cont[i]] = np.sqrt(data[f_cont[i]])

    stats.probplot(df[f_cont[i]], dist="norm", plot=axs[i,0])

    axs[i,0].set_title(f_cont[i])

    stats.probplot(data[f_cont[i]], dist="norm", plot=axs[i,1])

    axs[i,1].set_title(f_cont[i])

    

    #plt.title(feature)

plt.show()
f_cont[0]
df[f_dis]['children'].unique()
f_cat = [feature for feature in df.columns if df[feature].dtype=='O']

f_cat
for i in f_cat:

    print(df[i].unique(),"\n",df[i].value_counts())
import seaborn as sns

sns.set(style="whitegrid")

for i in f_num:

    ax = sns.boxplot(x=df[i])

    plt.show()
df.head()
df[f_num]
from scipy import stats 

IQR=[]

for i in f_num:

    IQR.append(stats.iqr(df[i], interpolation = 'midpoint'))

IQR
limits = dict()

j=0

for i in f_num:

    Q1 = np.percentile(df[i], 25, interpolation = 'midpoint')  

    Q3 = np.percentile(df[i], 75, interpolation = 'midpoint')  

    #print(Q1, Q3)

    limits[i] = [Q1-(1.5*IQR[j]), Q3+(1.5*IQR[j])]

    j+=1



limits
df[df['bmi']>=47].index
outliers = dict()

for i in f_num:

    outliers[i]=list()

    for x in df[i]:

        if(x<limits[i][0] or x>limits[i][1]):

            outliers[i].append(x)

   

outliers
df
"""



# One Hot Encoding for 'sex' and 'smoker' columns

df_new = df.copy()

pd.get_dummies(df_new, columns=['sex','smoker'], prefix=['sex', 'smoker'])





# Binary Encoding for 'sex' and 'smoker' columns

import category_encoders as ce

df_new = df.copy()

encoder = ce.BinaryEncoder(cols=['sex','smoker'])

df_new = encoder.fit_transform(df_new)

print(df_new.head())



"""
#df_new = df.drop(df[df['bmi']>=47].index, axis=0).reset_index(drop=True)



df_new = df.copy()








"""

# Backward Difference Encoding for 'region' column

import category_encoders as ce

encoder = ce.BackwardDifferenceEncoder(cols=['region'])

df_new = encoder.fit_transform(df_new)

print(df_new.head())

"""



# One Hot Encoding for 'region' column

import category_encoders as ce

import pandas as pd



#Create object for one-hot encoding

encoder=ce.OneHotEncoder(cols=['region'],handle_unknown='return_nan',return_df=True,use_cat_names=True)



#Original Data

print(df_new.head())

#Fit and transform Data

df_new = encoder.fit_transform(df_new)

print(df_new.head())





# Label Encoding for 'sex' and 'smoker' columns

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df_new['sex'] = le.fit_transform(df_new['sex'])

df_new['smoker'] = le.fit_transform(df_new['smoker'])

print(list(le.classes_))

df_new.head()
#df_new.drop(['intercept'], axis=1, inplace=True)
df_new.head()
df_new.shape
df_new['age'] = np.sqrt(df_new['age'])

df_new
"""

## StandardScaler

from sklearn.preprocessing import StandardScaler

data = df_new.copy()

scaler = StandardScaler()

data[data.drop(['charges'],axis=1).columns] = scaler.fit_transform(data.drop(['charges'],axis=1))

data = pd.DataFrame(data, columns=df_new.columns)

#print(scaler.mean_)

#print(scaler.transform([[2, 2]]))



## MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

data = df_new.copy()

scaler = MinMaxScaler()

data[data.drop(['charges'],axis=1).columns] = scaler.fit_transform(data.drop(['charges'],axis=1))

data = pd.DataFrame(data, columns=df_new.columns)



## MaxAbsScaler

from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()

data = df_new.copy()

data[data.drop(['charges'],axis=1).columns] = scaler.fit_transform(data.drop(['charges'],axis=1))

data = pd.DataFrame(data, columns=df_new.columns)



## RobustScaler

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

data = df_new.copy()

data[data.drop(['charges'],axis=1).columns] = scaler.fit_transform(data.drop(['charges'],axis=1))

data = pd.DataFrame(data, columns=df_new.columns)



## QuantileTransformer

from sklearn.preprocessing import QuantileTransformer

scaler = QuantileTransformer()

data = df_new.copy()

data[data.drop(['charges'],axis=1).columns] = scaler.fit_transform(data.drop(['charges'],axis=1))

data = pd.DataFrame(data, columns=df_new.columns)



## PowerTransformer using yeo-johnson

from sklearn.preprocessing import PowerTransformer

scaler = PowerTransformer(method='yeo-johnson')

data = df_new.copy()

data[data.drop(['charges'],axis=1).columns] = scaler.fit_transform(data.drop(['charges'],axis=1))

data = pd.DataFrame(data, columns=df_new.columns)



## PowerTransformer using box-cox

from sklearn.preprocessing import PowerTransformer

scaler = PowerTransformer(method='box-cox')

data = df_new.copy()

data[data.drop(['charges'],axis=1).columns] = scaler.fit_transform(data.drop(['charges'],axis=1))

data = pd.DataFrame(data, columns=df_new.columns)

"""
data = df_new.copy()
plt.figure(figsize=(12,10))

sns.heatmap(data.corr(), annot=True)
X, y = data.iloc[:,:-1], data.iloc[:,-1]

X
y


"""

from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X, y)



from sklearn.feature_selection import RFE

selector = RFE(reg, n_features_to_select=5, step=1)

selector = selector.fit(X, y)

print(selector.support_)

print(selector.ranking_)



f_selected = X.columns[selector.support_]

f_selected



"""
# Feature Importance with Extra Trees Classifier

from pandas import read_csv

from sklearn.ensemble import ExtraTreesRegressor



# feature extraction

model = ExtraTreesRegressor(n_estimators=10)

model.fit(X, y)

print(model.feature_importances_)

top_features = np.array(model.feature_importances_)



indices = (-top_features).argsort()[:3]

print(indices)

f_selected = X.columns[indices]

print(X.columns)

print(f_selected)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=128)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
X_train = X_train[f_selected]

X_test = X_test[f_selected]

print(X_train.shape)

print(X_test.shape)
X_train
X_test
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score



reg = LinearRegression()

score = cross_val_score(reg, X_train[f_selected], y_train, cv=10)

print(score)

print(score.mean())
from sklearn.linear_model import LinearRegression, Ridge

ridge_reg = Ridge(alpha=0.1)

#ridge_reg.fit(X_train, y_train)

score = cross_val_score(ridge_reg, X_train[f_selected], y_train, cv=10)

print(score)

print(score.mean())
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)

score = cross_val_score(lasso_reg, X_train[f_selected], y_train, cv=10)

print(score)

print(score.mean())
from sklearn.svm import SVR

import numpy as np

svr_reg = SVR(C=1.0, epsilon=0.2)

score = cross_val_score(svr_reg, X_train[f_selected], y_train, cv=10)

print(score)

print(score.mean())
X_train[f_selected].shape
from sklearn.linear_model import ElasticNet



elasticnet_reg = ElasticNet(l1_ratio=0.8, random_state=0)

score = cross_val_score(elasticnet_reg, X_train[f_selected], y_train, cv=10)

print(score)

print(score.mean())
from sklearn.ensemble import RandomForestRegressor



randomfor_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

score = cross_val_score(randomfor_reg, X_train[f_selected], y_train, cv=10)

print(score)

print(score.mean())
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeRegressor



decisiontree_reg = DecisionTreeRegressor(max_depth=4, random_state=0)

score = cross_val_score(decisiontree_reg, X_train[f_selected], y_train, cv=10)

print(score)

print(score.mean())
from sklearn.ensemble import AdaBoostRegressor

adaboost_reg = AdaBoostRegressor(n_estimators=100, random_state=0)

score = cross_val_score(adaboost_reg, X_train[f_selected], y_train, cv=10)

print(score)

print(score.mean())
from sklearn.ensemble import GradientBoostingRegressor



gradientboost_reg = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=0)

score = cross_val_score(gradientboost_reg, X_train[f_selected], y_train, cv=10)

print(score)

print(score.mean())
from sklearn.ensemble import BaggingRegressor



bagging_reg = BaggingRegressor(n_estimators=100, max_features=3, max_samples=50, random_state=0)

score = cross_val_score(bagging_reg, X_train[f_selected], y_train, cv=10)

print(score)

print(score.mean())
import xgboost as xgb



xgb_reg = xgb.XGBRegressor(objective ='reg:squarederror', max_depth=4, learning_rate = 0.1, alpha = 0.1, n_estimators = 200)

score = cross_val_score(xgb_reg, X_train[f_selected], y_train, cv=10)

print(score)

print(score.mean())
from keras.callbacks import ModelCheckpoint

from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error 

from matplotlib import pyplot as plt

import seaborn as sb

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import warnings 

warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore', category=DeprecationWarning)
NN_model = Sequential()

weight_init = 'random_normal'

# The Input Layer :

NN_model.add(Dense(256, kernel_initializer=weight_init,input_dim = X_train[f_selected].shape[1], activation='relu'))



# The Hidden Layers :

NN_model.add(Dense(256, kernel_initializer=weight_init,activation='relu'))

NN_model.add(Dense(256, kernel_initializer=weight_init,activation='relu'))



# The Output Layer :

NN_model.add(Dense(1, kernel_initializer=weight_init,activation='linear'))



# Compile the network :

NN_model.compile(loss='mean_absolute_error', optimizer='Nadam', metrics=["mean_absolute_error"])

NN_model.summary()
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 

checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')

callbacks_list = [checkpoint]
#!rm ./*
NN_model.fit(X_train[f_selected], y_train, epochs=500, batch_size=64, validation_split = 0.2, callbacks=callbacks_list)


from sklearn.metrics import mean_absolute_error

y_pred = NN_model.predict(X_train[f_selected])

y_test_pred = NN_model.predict(X_test[f_selected])

mae_train = mean_absolute_error(y_train, y_pred)

mae_test = mean_absolute_error(y_test, y_test_pred)

print("MAE for train :",mae_train)

print("MAE for test :",mae_test)



from sklearn.metrics import r2_score

print("R2 for train :",r2_score(y_train, y_pred))

print("R2 for test :",r2_score(y_test, y_test_pred))
# from sklearn.ensemble import GradientBoostingRegressor



gradientboost_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, max_features='sqrt', random_state=0)

gradientboost_reg.fit(X_train[f_selected], y_train)



from sklearn.metrics import mean_absolute_error

y_pred = gradientboost_reg.predict(X_train[f_selected])

y_test_pred = gradientboost_reg.predict(X_test[f_selected])

mae_train = mean_absolute_error(y_train, y_pred)

mae_test = mean_absolute_error(y_test, y_test_pred)

print("MAE for train :",mae_train)

print("MAE for test :",mae_test)



from sklearn.metrics import r2_score

print("R2 for train :",r2_score(y_train, y_pred))

print("R2 for test :",r2_score(y_test, y_test_pred))

#print(score.mean())
"""

# Load wights file of the best model :

wights_file = './Weights-365--1753.49939.hdf5' # choose the best checkpoint 

NN_model.load_weights(wights_file) # load it

NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

"""
"""

gradientboost_reg = GradientBoostingRegressor(random_state=0)

from skopt import BayesSearchCV

from skopt.space import Real, Categorical, Integer



search_space = {

        "max_depth": Integer(2, 10),

        "max_features": Categorical(['auto', 'sqrt','log2']), 

        "min_samples_leaf": Integer(2, 10),

        "min_samples_split": Integer(2, 10),

        "n_estimators": Integer(50, 300)

    }



def on_step(optim_result):

    score = forest_bayes_search.best_score_

    print("best score: %s" % score)

    if score >= 0.98:

        print('Interrupting!')

        return True



forest_bayes_search = BayesSearchCV(gradientboost_reg, search_space, n_iter=32, scoring="r2_score", n_jobs=-1, cv=5)



forest_bayes_search.fit(X[f_selected], y, callback=on_step) # callback=on_step will print score after each iteration



# Just like in Scikit-Learn we can view the best parameters:

forest_bayes_search.best_params_

# And the best estimator:

forest_bayes_search.best_estimator_

# And the best score:

forest_bayes_search.best_score_



"""
from sklearn.model_selection import RandomizedSearchCV



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 50, stop = 300)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt', 'log2']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(2, 10)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [int(x) for x in np.linspace(2, 10)]

# Minimum number of samples required at each leaf node

min_samples_leaf = [int(x) for x in np.linspace(2, 10)]

loss = ['ls', 'lad', 'huber', 'quantile']

learning_rate = [0.01, 0.03, 0.1, 0.3]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'loss' : loss,

               'learning_rate' : learning_rate,

               }
# Use the random grid to search for best hyperparameters

# First create the base model to tune

gradientboost_reg = GradientBoostingRegressor(random_state=0)

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

gb_random = RandomizedSearchCV(estimator = gradientboost_reg, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)

# Fit the random search model

gb_random.fit(X_train[f_selected], y_train)
print(gb_random.best_params_)

print(gb_random.best_estimator_)
best_random = gb_random.best_estimator_



#gradientboost_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, max_features='sqrt', random_state=0)

best_random.fit(X_train[f_selected], y_train)



from sklearn.metrics import mean_absolute_error

y_pred = best_random.predict(X_train[f_selected])

y_test_pred = best_random.predict(X_test[f_selected])

mae_train = mean_absolute_error(y_train, y_pred)

mae_test = mean_absolute_error(y_test, y_test_pred)

print("MAE for train :",mae_train)

print("MAE for test :",mae_test)



from sklearn.metrics import r2_score

print("R2 for train :",r2_score(y_train, y_pred))

print("R2 for test :",r2_score(y_test, y_test_pred))

#print(score.mean())
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score



def black_box_function(n_estimators, max_depth, min_samples_split, min_samples_leaf):

    

    n_estimators = int(n_estimators*250 + 50)

    max_depth = int(max_depth*8 + 2)

    min_samples_split = int(min_samples_split*8 + 2)

    min_samples_leaf = int(min_samples_leaf*8 + 2)

    

    gradientboost_reg = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,\

                                                  min_samples_split=min_samples_split,\

                                                  min_samples_leaf=min_samples_leaf, random_state=0)

    score = cross_val_score(gradientboost_reg, X_train[f_selected], y_train, cv=10)

    r2 = score.mean()

    n = len(X_train[f_selected])

    p = X_train[f_selected].shape[1]

    

    adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)

    

    return adj_r2

    

# Number of trees in random forest

n_estimators = (0,1)

# Number of features to consider at every split

#max_features = ['auto', 'sqrt', 'log2']

# Maximum number of levels in tree

max_depth = (0, 1)

# Minimum number of samples required to split a node

min_samples_split = (0, 1)

# Minimum number of samples required at each leaf node

min_samples_leaf = (0, 1)

#loss = ('ls', 'lad', 'huber', 'quantile')



# Create the random grid

pbounds = {'n_estimators': n_estimators,

           'max_depth': max_depth,

           'min_samples_split': min_samples_split,

           'min_samples_leaf': min_samples_leaf,

           }



from bayes_opt import BayesianOptimization



optimizer = BayesianOptimization(

    f=black_box_function,

    pbounds=pbounds,

    random_state=0,

)



optimizer.maximize(

    init_points=40,

    n_iter=50,

)



print(optimizer.max)



def black_box_function_test(n_estimators, max_depth, min_samples_split, min_samples_leaf):

    

    n_estimators = int(n_estimators*250 + 50)

    max_depth = int(max_depth*8 + 2)

    min_samples_split = int(min_samples_split*8 + 2)

    min_samples_leaf = int(min_samples_leaf*8 + 2)

    #print(n_estimators, max_depth, min_samples_split, min_samples_leaf)

    

    gradientboost_reg = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,\

                                                  min_samples_split=min_samples_split,\

                                                  min_samples_leaf=min_samples_leaf, random_state=0)

    #print(gradientboost_reg)

    gradientboost_reg.fit(X_train[f_selected], y_train)



    #from sklearn.metrics import mean_absolute_error

    y_pred = gradientboost_reg.predict(X_train[f_selected])

    y_test_pred = gradientboost_reg.predict(X_test[f_selected])

    #mae_train = mean_absolute_error(y_train, y_pred)

    #mae_test = mean_absolute_error(y_test, y_test_pred)

    #print("MAE for train :",mae_train)

    #print("MAE for test :",mae_test)



    from sklearn.metrics import r2_score

    #print("R2 for train :",r2_score(y_train, y_pred))

    #print("R2 for test :",r2_score(y_test, y_test_pred))

    r2_train = r2_score(y_train, y_pred)

    n = len(X_train[f_selected])

    p = X_train[f_selected].shape[1]

    adj_r2_train = 1-(1-r2_train)*(n-1)/(n-p-1)

    print("Train adjusted r2 score = ",adj_r2_train)

    

    #print(gradientboost_reg)

    r2_test = r2_score(y_test, y_test_pred)

    n = len(X_test[f_selected])

    p = X_test[f_selected].shape[1]

    adj_r2_test = 1-(1-r2_test)*(n-1)/(n-p-1)



    print("Test adjusted r2 score = ",adj_r2_test)

    

    

params = optimizer.max['params']

n_estimators = params['n_estimators']

max_depth = params['max_depth']

min_samples_split = params['min_samples_split']

min_samples_leaf = params['min_samples_leaf'

                         ]

print(n_estimators, max_depth, min_samples_split, min_samples_leaf)



black_box_function_test(n_estimators, max_depth, min_samples_split, min_samples_leaf)
from sklearn.ensemble import RandomForestRegressor



randomfor_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

score = cross_val_score(randomfor_reg, X_train[f_selected], y_train, cv=10)

print(score)

print(score.mean())



from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeRegressor



decisiontree_reg = DecisionTreeRegressor(max_depth=4, random_state=0)

score = cross_val_score(decisiontree_reg, X_train[f_selected], y_train, cv=10)

print(score)

print(score.mean())





from sklearn.ensemble import AdaBoostRegressor

adaboost_reg = AdaBoostRegressor(n_estimators=100, random_state=0)

score = cross_val_score(adaboost_reg, X_train[f_selected], y_train, cv=10)

print(score)

print(score.mean())





bagging_reg = BaggingRegressor(n_estimators=100, max_features=3, max_samples=50, random_state=0)

score = cross_val_score(bagging_reg, X_train[f_selected], y_train, cv=10)

print(score)

print(score.mean())



import xgboost as xgb



xgb_reg = xgb.XGBRegressor(objective ='reg:squarederror', max_depth=4, learning_rate = 0.1, alpha = 0.1, n_estimators = 200)

score = cross_val_score(xgb_reg, X_train[f_selected], y_train, cv=10)

print(score)

print(score.mean())
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score 



def randomfor_reg_bayesian_opt_function(n_estimators, max_depth, min_samples_split, min_samples_leaf):

    

    n_estimators = int(n_estimators*250 + 50)

    max_depth = int(max_depth*8 + 2)

    min_samples_split = int(min_samples_split*8 + 2)

    min_samples_leaf = int(min_samples_leaf*8 + 2)

    



    randomfor_reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,\

                                                  min_samples_split=min_samples_split,\

                                                  min_samples_leaf=min_samples_leaf, random_state=0)

    score = cross_val_score(randomfor_reg, X_train[f_selected], y_train, cv=10)

    r2 = score.mean()

    n = len(X_train[f_selected])

    p = X_train[f_selected].shape[1]

    

    adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)

    

    return adj_r2



# Number of trees in random forest

n_estimators = (0,1)

# Number of features to consider at every split

#max_features = ['auto', 'sqrt', 'log2']

# Maximum number of levels in tree

max_depth = (0, 1)

# Minimum number of samples required to split a node

min_samples_split = (0, 1)

# Minimum number of samples required at each leaf node

min_samples_leaf = (0, 1)

#loss = ('ls', 'lad', 'huber', 'quantile')



# Create the random grid

pbounds = {'n_estimators': n_estimators,

           'max_depth': max_depth,

           'min_samples_split': min_samples_split,

           'min_samples_leaf': min_samples_leaf,

           }



from bayes_opt import BayesianOptimization



optimizer = BayesianOptimization(

    f=randomfor_reg_bayesian_opt_function,

    pbounds=pbounds,

    random_state=0,

)





optimizer.maximize(

    init_points=40,

    n_iter=50

)





params = optimizer.max['params']

n_estimators = params['n_estimators']

max_depth = params['max_depth']

min_samples_split = params['min_samples_split']

min_samples_leaf = params['min_samples_leaf']

print(n_estimators, max_depth, min_samples_split, min_samples_leaf)



def randomfor_reg_bayesian_opt_function_test(n_estimators, max_depth, min_samples_split, min_samples_leaf):

    

    n_estimators = int(n_estimators*250 + 50)

    max_depth = int(max_depth*8 + 2)

    min_samples_split = int(min_samples_split*8 + 2)

    min_samples_leaf = int(min_samples_leaf*8 + 2)

    #print(n_estimators, max_depth, min_samples_split, min_samples_leaf)

    randomfor_reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,\

                                                  min_samples_split=min_samples_split,\

                                                  min_samples_leaf=min_samples_leaf, random_state=0)



    

    randomfor_reg.fit(X_train[f_selected], y_train)



    y_pred = randomfor_reg.predict(X_train[f_selected])

    y_test_pred = randomfor_reg.predict(X_test[f_selected])



    from sklearn.metrics import r2_score

    

    r2_train = r2_score(y_train, y_pred)

    n = len(X_train[f_selected])

    p = X_train[f_selected].shape[1]

    adj_r2_train = 1-(1-r2_train)*(n-1)/(n-p-1)

    print("Train adjusted r2 score = ",adj_r2_train)

    

    r2_test = r2_score(y_test, y_test_pred)

    n = len(X_test[f_selected])

    p = X_test[f_selected].shape[1]

    adj_r2_test = 1-(1-r2_test)*(n-1)/(n-p-1)



    print("Test adjusted r2 score = ",adj_r2_test)

    

    

randomfor_reg_bayesian_opt_function_test(n_estimators, max_depth, min_samples_split, min_samples_leaf)
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score



def decisiontree_reg_bayesian_opt_function(criterion, max_depth, min_samples_split, min_samples_leaf):

    

    #n_estimators = int(n_estimators*250 + 50)

    max_depth = int(max_depth*8 + 2)

    min_samples_split = int(min_samples_split*8 + 2)

    min_samples_leaf = int(min_samples_leaf*8 + 2)

    criteria = ["mse", "friedman_mse", "mae"]

    criterion = criteria[int(criterion)-1]

    



    decisiontree_reg = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth,\

                                                  min_samples_split=min_samples_split,\

                                                  min_samples_leaf=min_samples_leaf, random_state=0)

    score = cross_val_score(decisiontree_reg, X_train[f_selected], y_train, cv=10)

    r2 = score.mean()

    n = len(X_train[f_selected])

    p = X_train[f_selected].shape[1]

    

    adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)

    

    return adj_r2





criterion = (1,3.99)

# Number of trees in random forest

#n_estimators = (0,1)

# Number of features to consider at every split

#max_features = ['auto', 'sqrt', 'log2']

# Maximum number of levels in tree

max_depth = (0, 1)

# Minimum number of samples required to split a node

min_samples_split = (0, 1)

# Minimum number of samples required at each leaf node

min_samples_leaf = (0, 1)

#loss = ('ls', 'lad', 'huber', 'quantile')



# Create the random grid

pbounds = {'criterion': criterion,

           'max_depth': max_depth,

           'min_samples_split': min_samples_split,

           'min_samples_leaf': min_samples_leaf,

           }



from bayes_opt import BayesianOptimization



optimizer = BayesianOptimization(

    f=decisiontree_reg_bayesian_opt_function,

    pbounds=pbounds,

    random_state=0,

)





optimizer.maximize(

    init_points=40,

    n_iter=50

)





params = optimizer.max['params']

criterion = params['criterion']

max_depth = params['max_depth']

min_samples_split = params['min_samples_split']

min_samples_leaf = params['min_samples_leaf']

print(criterion, max_depth, min_samples_split, min_samples_leaf)



def decisiontree_reg_bayesian_opt_function_test(criterion, max_depth, min_samples_split, min_samples_leaf):

    

    #n_estimators = int(n_estimators*250 + 50)

    max_depth = int(max_depth*8 + 2)

    min_samples_split = int(min_samples_split*8 + 2)

    min_samples_leaf = int(min_samples_leaf*8 + 2)

    criteria = ["mse", "friedman_mse", "mae"]

    criterion = criteria[int(criterion)]

    #print(n_estimators, max_depth, min_samples_split, min_samples_leaf)

    decisiontree_reg = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth,\

                                                  min_samples_split=min_samples_split,\

                                                  min_samples_leaf=min_samples_leaf, random_state=0)



    

    decisiontree_reg.fit(X_train[f_selected], y_train)



    y_pred = decisiontree_reg.predict(X_train[f_selected])

    y_test_pred = decisiontree_reg.predict(X_test[f_selected])



    from sklearn.metrics import r2_score

    

    r2_train = r2_score(y_train, y_pred)

    n = len(X_train[f_selected])

    p = X_train[f_selected].shape[1]

    adj_r2_train = 1-(1-r2_train)*(n-1)/(n-p-1)

    print("Train adjusted r2 score = ",adj_r2_train)

    

    r2_test = r2_score(y_test, y_test_pred)

    n = len(X_test[f_selected])

    p = X_test[f_selected].shape[1]

    adj_r2_test = 1-(1-r2_test)*(n-1)/(n-p-1)



    print("Test adjusted r2 score = ",adj_r2_test)

    

    

decisiontree_reg_bayesian_opt_function_test(criterion, max_depth, min_samples_split, min_samples_leaf)