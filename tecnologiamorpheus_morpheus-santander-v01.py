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
# 1) Import all Library that will be used



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap



import statsmodels.formula.api as smf



from scipy import stats



from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification



from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn import linear_model, svm, gaussian_process

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score



from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

import lightgbm as lgb



from sklearn import preprocessing

from sklearn import utils



import statsmodels.formula.api as smf

import warnings

warnings.filterwarnings("ignore")



# 1) GradientBoostingRegressor Model

from sklearn.ensemble import GradientBoostingRegressor

import statsmodels.formula.api as smf

from sklearn.preprocessing import scale

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier



parameters = {

    "loss":["deviance"],

    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],

    "min_samples_split": np.linspace(0.1, 0.5, 12),

    "min_samples_leaf": np.linspace(0.1, 0.5, 12),

    "max_depth":[3,5,8],

    "max_features":["log2","sqrt"],

    "criterion": ["friedman_mse",  "mae"],

    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],

    "n_estimators":[10]

    }

GBR = GradientBoostingRegressor()

#GBR = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)



# 2) Logistic Regression Model

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()



# 3) Aplly Random Forest Model

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=100)



# 4) Aplly XGBOOST Model

from xgboost import XGBClassifier

XGB = XGBClassifier()



# 5) Aplly KNeighbors Model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()



# 6) Aplly SVC Model

SVC = SVC(probability=True)



# 7) Aplly Decision Tree Model

DTC = DecisionTreeClassifier()



# 8) Aplly GaussianNB Model

GNB = GaussianNB()



# 9) Aplly Neural Model

NN = MLPClassifier(hidden_layer_sizes=(100,100,50))



# 10) Aplly lasso

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))



# 11) Apply Elastic Net

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))



# 12) Apply Kernel Ridge

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)



# 13) Apply LGBMRegressor

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



# 14) Apply LGBMRegressor

from sklearn.linear_model import LinearRegression

LR2 = LinearRegression()



#15) Linear Regression with Tensor Flow

import tensorflow as tf
# 1) Data treatment and cleaning



df_train_original = pd.read_csv('/kaggle/input/santander-customer-satisfaction/train.csv')

df_test_original = pd.read_csv('/kaggle/input/santander-customer-satisfaction/test.csv')
df_train = df_train_original

df_test = df_test_original



print ('df_train.shape: ', df_train.shape)

print ('df_test.shape: ', df_test.shape)
print ('df_train.columns: ', df_train.columns)

print ('df_test.columns: ', df_test.columns)
all_data = pd.concat((df_train.loc[:,'ID':'var38'],

                      df_test.loc[:,'ID':'var38']))
# Get_Dummies para transformar categoricos em Numéricos 

all_data = pd.get_dummies(all_data)



# Substitui os campos nulos pelas médias da coluna em questão

all_data = all_data.fillna(all_data.mean())



#Cria Matriz X_train utilizando a Matriz com todos os dados all_data: do inicio da matriz (:) até o fim  da matriz df_train.shape[0]

X_train = all_data[:df_train.shape[0]]



#Cria Matriz X_test utilizando a Matriz com todos os dados all_data: a partir do último registro matriz df_train.shape[0], ou seja, todos os registros que não estiverem em df_train

X_test = all_data[df_train.shape[0]:]

# Cria o y, ou seja, o que será previsto, apenas com o campo "Survived"

y = df_train.TARGET



print ('X_train.shape: ', X_train.shape)

print ('X_test.shape: ', X_test.shape)
#

# ! ! ! ! ! ! ! FUNÇÃO PRINCIPAL!!! Todos Modelos

#

def RunModel (ModelName, Model, Df_Test_Original, X_train, X_test, y):



    print ('# # # # Prediction for Model:  ', ModelName, '# # # #')

    print ('Shape X_train: ', X_train.shape)

    print ('Shape X_test: ', X_test.shape)

    print ('Shape y: ', y.shape)

    print ('Model: ', Model)

    

    print ('.FIT Model: ', Model)



    Model.fit(X_train, y)



    print ('PREDICT TRAIN: ', Model)



    yhat_Train = Model.predict(X_train)

    

    print ('PREDICT TEST: ', Model)



    yhat_test = Model.predict(X_test)



    # Verify Accuracy and other metrics:

    RunAcc(ModelName, yhat_Train, y)



    

    print ('# # # # Tamanho do Df_Test_Original:', Df_Test_Original.shape)

    print ('# # # # Prediction:', yhat_test.shape, yhat_test)



    Filename = 'Output_Santander_' + ModelName + '.csv'

    

    df_Output= pd.DataFrame()

    df_Output['ID'] = Df_Test_Original['ID']

    df_Output['TARGET'] = yhat_test

    df_Output.to_csv(Filename, index = False)

    

    return yhat_test
#

# ! ! ! ! ! ! ! Função específica para Runlightgbm

#

def Runlightgbm  (ModelName, Model, Df_Test_Original, X_train, X_test, y):



    print ('# # # # Prediction for Model:  ', ModelName, '# # # #')

    print ('Shape X_train: ', X_train.shape)

    print ('Shape X_test: ', X_test.shape)

    print ('Shape y: ', y.shape)

    print ('Model: ', Model)



    import lightgbm as lgb

    d_train = lgb.Dataset(X_train, label=y)

    

    params ={

                'task': 'train',

                'boosting': 'goss',

                'objective': 'regression',

                'metric': 'rmse',

                'learning_rate': 0.01,

                'subsample': 0.9855232997390695,

                'max_depth': 7,

                'top_rate': 0.9064148448434349,

                'num_leaves': 63,

                'min_child_weight': 41.9612869171337,

                'other_rate': 0.0721768246018207,

                'reg_alpha': 9.677537745007898,

                'colsample_bytree': 0.5665320670155495,

                'min_split_gain': 9.820197773625843,

                'reg_lambda': 8.2532317400459,

                'min_data_in_leaf': 21,

                'verbose': -1,

                'seed':int(2),

                'bagging_seed':int(2),

                'drop_seed':int(2)

                }

    

    print ('.FIT Model (Nesse caso, lgb.train: ', Model)



    clf = lgb.train(params, d_train)



    print ('PREDICT TEST: ', Model)



    yhat_Train = clf.predict(X_train)

    

    yhat_test = clf.predict(X_test)

    

    # Verify Accuracy and other metrics:

    RunAcc(ModelName, yhat_Train, y)

    

    print ('# # # # Tamanho do Df_Test_Original:', Df_Test_Original.shape)

    print ('# # # # Prediction:', yhat_test.shape, yhat_test)



    Filename = 'Output_' + ModelName + '.csv'

    

    df_Output= pd.DataFrame()

    df_Output['ID'] = Df_Test_Original['ID']

    df_Output['TARGET'] = yhat_test

    df_Output.to_csv(Filename, index = False)

    

    return yhat_test
from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score



# Calculate Accuracy:



def RunAcc (ModelName, yhat_Train, y):



    # Micro and None = Accuracy

    #micro - Calculate metrics globally by counting the total true positives, false negatives and false positives.

    #macro - Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account'

    #weighted - Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.'



    

    # Convert to int because does not accept continuous

    print ('Accuracy - Antes da Transformação: ')

    print ('yhat_Train: ', yhat_Train)

    print ('yhat_test: ', y)

    

    yhat_Train_int = yhat_Train * 10

    yhat_Train_int = yhat_Train_int.astype(int)

    

    yhat_test_int = y

    yhat_test_int = yhat_test_int.astype(int)

    

    yhat_Train = yhat_Train_int

    yhat_test = yhat_test_int

    

    print ('Accuracy - Depois da Transformação: ')

    print ('yhat_Train: ', yhat_Train)

    print ('yhat_test: ', yhat_test)

    

    Accuracy = accuracy_score(yhat_Train, yhat_test)



    # - Recall ! ! !

    Recall_Macro = recall_score(yhat_Train, yhat_test, average='macro')

    Recall_weighted = recall_score(yhat_Train, yhat_test, average='weighted')

    Recall_Micro = recall_score(yhat_Train, yhat_test, average='micro')

    Recall_None = recall_score(yhat_Train, yhat_test, average=None)



    # - Precision ! ! !

    Preci_Macro = precision_score(yhat_Train, yhat_test, average='macro')

    Preci_weighted = precision_score(yhat_Train, yhat_test, average='weighted')

    Preci_Micro = precision_score(yhat_Train, yhat_test, average='micro')

    Preci_None = precision_score(yhat_Train, yhat_test, average=None)



    print ('# # # # # # # # # # # # # # # # # # # # #')

    print ('# # #MODEL: ', ModelName)

    print ('# # #Accuracy: ', Accuracy)

    print ('# # # # # # # # # # # # # # # # # # # # #')

    print ('Recall_Macro: ', Recall_Macro)

    print ('Recall_Micro: ', Recall_Micro)

    print ('Recall_None: ', Recall_None)

    print ('Recall_weighted: ', Recall_weighted)

    print ('# # # # # # # # # # # # # # # # # # # # #')

    print ('Preci_Macro: ', Preci_Macro)

    print ('Preci_Micro: ', Preci_Micro)

    print ('Preci_None: ', Preci_None)

    print ('Preci_weighted: ', Preci_weighted)

    print ('# # # # # # # # # # # # # # # # # # # # #')
# 0) Run Runlightgbm: This model has exclusive parameters so a single function was created for it:

Sel_Model = lgb

NameM = 'lgb'

Runlightgbm(NameM, Sel_Model, df_test_original, X_train, X_test, y)
# 1) Gradiente Boost Model:

Sel_Model = GBR

NameM = 'GBR'

MGBR = RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)
# 2) Linear Regression:

Sel_Model = LR

NameM = 'LinearRegress'

RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)
# 3) Random Forest:

Sel_Model = RF 

NameM = 'RanFor'

RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)
# 4) xgboost Model (Megazord):

Sel_Model = XGB

NameM = 'XgBoost'

RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)
# 5) KNeighbors Model:

Sel_Model = knn

NameM = 'KNeighbors'

#RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)
# 6) SVC Model:

Sel_Model = SVC

NameM = 'SVC'

#RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)
# 7) Decision Tree Model:

Sel_Model = DTC

NameM = 'DecisionTree'

RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)
# 8) Gaussian:

Sel_Model = GNB

NameM = 'Gaussian'

RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)
# 9) Neural Model:

Sel_Model = NN

NameM = 'NeuralModel'

RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)
# 10) Lasso:

Sel_Model = lasso

NameM = 'lasso'

MLASSO = RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)
# 11) Elastic Net:

Sel_Model = ENet

NameM = 'ElasticNet'

MENET = RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)
# 12) Kernel Ridge:

Sel_Model = KRR

NameM = 'NeuralModel'

#RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)
# 13) LGBMRegressor:

Sel_Model = model_lgb

NameM = 'LGBMRegressor'

#LGBM  = RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)
# 14) Linear Regression:

Sel_Model = LR2

NameM = 'LinearRegression'

#LINEAR2 = RunModel(NameM, Sel_Model, df_test_original, X_train, X_test, y)
#15) Ensemble with best values

#ensemble = MENET * 0.40 + LGBM * 0.40 + LINEAR2 * 0.20



#ModelName = 'Ensemble'

#Filename = 'Output_' + ModelName + '.csv'

    

#df_Output= pd.DataFrame()

#df_Output['ID_code'] = df_test_original['ID_code']

#df_Output['target'] = ensemble

#df_Output.to_csv(Filename, index = False)