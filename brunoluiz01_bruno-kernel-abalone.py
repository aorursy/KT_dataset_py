import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.svm import SVC

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import SGDRegressor

from sklearn.svm import LinearSVC

from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor

from sklearn.tree import DecisionTreeRegressor



from sklearn.preprocessing import StandardScaler







from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import RFE





import seaborn as sns



df_train = pd.read_csv('../input/abalone-train.csv')

df_test = pd.read_csv('../input/abalone-test.csv')

df_train.head(5)
def converte(x):

    if x == 'M':

        return 1

    elif x=='F':

        return 0

    elif x=='I':

        return 2

    else:

        return np.nan
def str_to_int(df):



    binary_variables = ['sex']

    for c in df.columns:

        if c in binary_variables:

            df[c]=list(map(converte,df[c]))

    return df
df_train = str_to_int(df_train)

df_test = str_to_int(df_test)

X_train = df_train.drop(columns=['id','rings'])

Y_train = df_train[['rings']]
param_list = [{'max_depth':[2,3,4,5,6,7],'max_leaf_nodes':[2,3,4,5,6,7,8,9,10]},

              {},

              {'normalize':[True,False],'n_jobs':[1,2,3,4,5]},

              {'tol':[0.1,0.01,0.0001,0.00001,0.000001]},

              {'kernel':['linear','rbf','poly'],'degree':[3,4,5,6,7],'tol':[0.1,0.01,0.0001,0.00001,0.000001]},

              {'hidden_layer_sizes':[(15,10),(15,10,5),(10,10),(10,5)],'activation':['identity','relu','tanh','logistic'],

               'solver':['adam','sgd'],'tol':[0.0001,0.00001,0.000001],

              'alpha':np.logspace(-5,3,5)

              }

             ]
model_dict={'MLP Regressor':[MLPRegressor(),param_list[5]]}
X_train = StandardScaler().fit_transform(X_train)

X_test = StandardScaler().fit_transform(df_test)
model_list=[]

score_list=[]

param_list=[]

best_estimator_list=[]

features_list=[]





for m in model_dict:

        grid = GridSearchCV(model_dict[m][0],model_dict[m][1], cv=10, scoring='r2')

        grid.fit(X_train,Y_train)

        model_list.append(m)

        score_list.append(grid.best_score_)

        param_list.append(grid.best_params_)

        best_estimator_list.append(grid.best_estimator_)



results_dict = {'Modelo':model_list,'Score':score_list,'Estimator':best_estimator_list,

                'Parametros':param_list}        

final_results = pd.DataFrame(results_dict)
final_results=final_results.sort_values('Score',ascending=False)

final_results
best_model = final_results.iloc[0]

best_model[0]
best_model['Estimator']
best_model[2]
model = best_model['Estimator']

model.fit(X_train,Y_train['rings'])
X_test = pd.DataFrame(StandardScaler().fit_transform(df_test.drop(columns=['id'])))

df_test['rings']=model.predict(X_test)

df_test.head()
df_final = df_test[['id','rings']]

df_final.to_csv('arquivo_final_abalone.csv',index=False)