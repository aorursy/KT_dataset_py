#

### Packages importation

#



# Packages to manipulate our data

import pandas as pd

import numpy as np



# Packages to plot our data 

import matplotlib

import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline



# Packages to use de cross-validation in order to see the precision of our models

from sklearn.model_selection import cross_val_score

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split



# Packages to use the gridsearch function in order to parametrize your model

from sklearn.model_selection import GridSearchCV



# Packages which contain models to predict our final grade

from sklearn.linear_model import ElasticNet

from sklearn import svm

from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor

import xgboost as xgb



#

### Dataset importation

#



df = pd.read_csv("../input/student-mat.csv",sep=",")
df = pd.read_csv("../input/student-mat.csv",sep=",")
df.head()
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"grade":df["G3"]})

prices.hist()
df = pd.get_dummies(df)

df.head()
df_X = df.drop(['G1','G2','G3'],1)

df_Y = df[['G3']]
# In this we will initialize all the parameter we want to test.

# We know that alpha and l1_ratio will not have a great value



param_grid = [

  {'alpha': [0.1,0.2,0.3,0.4,0.5,1,10], 'l1_ratio': [0.0001,0.001,0.01,0.1,1,10]  }

 ]



# This function will test all the parameter initialize above and check the best parameter by a cross-validation.

best_param = GridSearchCV(ElasticNet(), param_grid , cv=5).fit(df_X,df_Y).best_params_



# We initialize our model with our best parameters 

lr_1 = ElasticNet( alpha = best_param['alpha'] , l1_ratio = best_param['l1_ratio'] )



# We will see the performance of our model with a cross-validation of 5.

scores = cross_val_score(lr_1, df_X, df_Y, cv=10)





print('Average of our R square : {0}\nVariance of our R square : {1} '.format(scores.mean(),scores.var()))
dict_corr = dict()



df_rmv = df.drop(['G1','G2','G3'],1)



# I write this function to put in dict_corr features with more than 0.9 correlation with each other

for i in df_rmv.columns :

    if (df_rmv.corr()[i].abs().sort_values(ascending=False).drop(i)[0] > 0.9) :

        dict_corr[i] = df_rmv.corr()[i].abs().sort_values(ascending=False).drop(i).index[0]

        

suppr = list()



# When two features have more than 0.9 corr this function delete just one of them

for i in dict_corr :

    if not(i in suppr) and not(dict_corr[i] in suppr) :

        suppr.append(i)

        

df_test = df.drop(suppr,1)
# Now we will see if this improve our R square



scores = cross_val_score(ElasticNet( alpha = best_param['alpha'] , l1_ratio = best_param['l1_ratio'] ),df_test.drop(['G1','G2','G3'],1), df[['G3']], cv=10)



print('Average of our R square : {0}\nVariance of our R square : {1} '.format(scores.mean(),scores.var()))
# We calcul the std in advance for the target

std_target_1 = df['G3'].std()



# We calcul the cov and take only for 'G3'

contrib_1 = df.cov()['G3'].drop(['G1','G2','G3'])



# We will compute the contribution for all features

for i in contrib_1.index :

    std_i = df[i].std()

    if std_i != 0 :

        contrib_1[i] = (contrib_1[i]/(std_target_1*std_i ))

    else :

        contrib_1[i] = 0

        

# Now we take the absolute value and sort or vector

contrib_1 = contrib_1.abs().sort_values(ascending=False)



# We will create a dictionnary for all the number of variables and the performance associated.

performance = {}



for i in range(1,len(contrib_1)) :

    df_X = df[contrib_1.index[list(range(0,i))]]

    performance[i] = cross_val_score(ElasticNet( alpha = best_param['alpha'] , l1_ratio = best_param['l1_ratio'] ), df_X, df_Y, cv=5).mean()

    

# Here we will take the model with the best performance 

var = contrib_1.index[list(range(0,max(performance, key=performance.get)))]
# We select our new set of data

df_test = df[var]



# Now we will see if this improve our R square



scores = cross_val_score(ElasticNet( alpha = best_param['alpha'] , l1_ratio = best_param['l1_ratio'] ),df_test, df_Y , cv=5)



print('Average of our R square : {0}\nVariance of our R square : {1} '.format(scores.mean(),scores.var()))
df_X = df.drop(['G3'],1)

df_Y = df[['G3']]
param_grid = [

  {'alpha': [0.01,0.05,0.1,0.15,0.2,1,10], 'l1_ratio': [0.0001,0.001,0.01,0.1,0.5,1,3,5,10]  }

 ]



# This function will test all the parameter initialize above and check the best parameter by a cross-validation.

best_param = GridSearchCV(ElasticNet(), param_grid , cv=5).fit(df_X,df_Y).best_params_
# We calcul the std in advance for the target

std_target_1 = df_Y.std()



# We calcul the cov and take only for 'G3'

contrib_1 = df.cov()['G3'].drop(['G3'])



# We will compute the contribution for all features

for i in contrib_1.index :

    std_i = df[i].std()

    if std_i != 0 :

        contrib_1[i] = (contrib_1[i]/(std_target_1*std_i ))

    else :

        contrib_1[i] = 0

        

# Now we take the absolute value and sort or vector

contrib_1 = contrib_1.abs().sort_values(ascending=False)



# We will create a dictionnary for all the number of variables and the performance associated.

performance = {}



for i in range(1,len(contrib_1)) :

    df_X = df[contrib_1.index[list(range(0,i))]]

    performance[i] = cross_val_score(ElasticNet( alpha = best_param['alpha'] , l1_ratio = best_param['l1_ratio'] ), df_X, df_Y, cv=5).mean()

    

# Here we will take the model with the best performance 

var = contrib_1.index[list(range(0,max(performance, key=performance.get)))]



# We select our new set of data

df_X = df[var]



# Now we will see if this improve our R square



scores = cross_val_score(ElasticNet( alpha = best_param['alpha'] , l1_ratio = best_param['l1_ratio'] ),df_X, df_Y , cv=5)



print('Average of our R square : {0}\nVariance of our R square : {1} '.format(scores.mean(),scores.var()))
param_grid = [

  {'C': [29,29.5,30,30.5,31,31.5,32], 'gamma' : [0.005,0.007,0.008,0.009,0.01,0.015] }

]



best_param = GridSearchCV(svm.SVR(kernel='rbf',epsilon=0.0001), param_grid , cv=5).fit(df_X,np.ravel(df_Y)).best_params_





scores = cross_val_score(svm.SVR(C=best_param['C'],epsilon=0.002,gamma=best_param['gamma'],kernel='rbf'), df_X, np.ravel(df_Y), cv=5)

print('Average of our R square : {0}\nVariance of our R square : {1} '.format(scores.mean(),scores.var()))
NN_1 = MLPRegressor(alpha=41,hidden_layer_sizes=(1000,1000,1000))



scores = cross_val_score(NN_1, df_X, np.ravel(df_Y), cv=5)

print('Average of our R square : {0}\nVariance of our R square : {1} '.format(scores.mean(),scores.var()))
df_X = df.drop(['G3'],1)

df_Y =  df[['G3']]



param_grid = [ {'n_estimators': [30,35,40,45,50,100], 'max_depth' : [4,5,6,7,8,9,10] }]

best_param = GridSearchCV(RandomForestRegressor(), param_grid , cv=5).fit(df_X,np.ravel(df_Y)).best_params_



scores = cross_val_score(RandomForestRegressor(max_depth =best_param['max_depth'],n_estimators=best_param['n_estimators']), df_X, np.ravel(df_Y), cv=5)



print('Average of our R square : {0}\nVariance of our R square : {1} '.format(scores.mean(),scores.var()))
nombre = 1

params = {"objective": "reg:linear", "booster":"gblinear"}

df = pd.get_dummies(df)

    

    

for i in range(0,nombre) :

    cut = df[int((len(df)/nombre)*i):int((len(df)/nombre)*(1+i))]

    train, test = train_test_split(cut,test_size=0.1)

                

    train_X = train.drop(['G3'],1)

    train_Y = train[['G3']]

    test_X = test.drop(['G3'],1)

    test_Y = test[['G3']]              



    T_train_xgb = xgb.DMatrix(train_X, train_Y)

    T_test_xgb = xgb.DMatrix(test_X, test_Y)

    

    gbm = xgb.train(dtrain=T_train_xgb,params=params)



    Y_pred = gbm.predict(T_test_xgb)

    print(r2_score(test_Y,Y_pred))