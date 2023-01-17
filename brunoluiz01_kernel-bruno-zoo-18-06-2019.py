import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier



from sklearn.model_selection import GridSearchCV



df_train = pd.read_csv('zoo-train.csv')

df_test = pd.read_csv('zoo-test.csv')

df_train2 = pd.read_csv('zoo-train2.csv')

df_train.head(5)
def converte(x):

    if x == 'y':

        return 1

    elif x=='n':

        return 0

    else:

        return np.nan
def str_to_bool(df):



    binary_variables = ['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','tail','domestic','catsize']

    for c in df.columns:

        if c in binary_variables:

            df[c]=list(map(converte,df[c]))

    return df
#DF com todas as variáveis binárias somente com treino.csv

df_train_temp = df_train.copy()

df_test_temp = df_test.copy()

df_train_temp['type']='Train'

df_test_temp['type']='Test'

df_test_temp['class_type']=-1

df_all_binary=pd.concat([df_train_temp,df_test_temp],sort=False)



df_all_binary[['0_legs','2_legs','4_legs','5_legs','6_legs','8_legs']]=pd.get_dummies(df_all_binary['legs'])



df_all_binary = str_to_bool(df_all_binary)



df_all_binary_train=df_all_binary[df_all_binary['type']=='Train']

df_all_binary_test=df_all_binary[df_all_binary['type']=='Test']



X_abt = df_all_binary_train[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed',

                  'backbone','breathes','venomous','fins','tail','domestic','catsize',

                 '0_legs','2_legs','4_legs','5_legs','6_legs','8_legs']]

Y_abt = df_all_binary_train['class_type']



df_all_binary_test = df_all_binary_test.drop(columns=['type','class_type','legs'])

df_all_binary_test.head(5)
# DF com todas as variáveis binárias com treino.csv + treino2.csv

df_train_temp = df_train.copy()

df_train2_temp = df_train2.copy()

df_test2_temp = df_test.copy()

df_train_temp['type']='Train'

df_train2_temp['type']='Train'

df_test2_temp['type']='Test'

df_test2_temp['class_type']=-1

df_all_binary2=pd.concat([df_train_temp,df_train2_temp],sort=False)

df_all_binary2=pd.concat([df_all_binary2,df_test2_temp],sort=False)



df_all_binary2[['0_legs','2_legs','4_legs','5_legs','6_legs','8_legs']]=pd.get_dummies(df_all_binary2['legs'])



df_all_binary2 = str_to_bool(df_all_binary2)



df_all_binary_train2=df_all_binary2[df_all_binary2['type']=='Train']

df_all_binary_test2=df_all_binary2[df_all_binary2['type']=='Test']



X_abt2 = df_all_binary_train2[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed',

                  'backbone','breathes','venomous','fins','tail','domestic','catsize',

                 '0_legs','2_legs','4_legs','5_legs','6_legs','8_legs']]

Y_abt2 = df_all_binary_train2['class_type']



df_all_binary_test2 = df_all_binary_test2.drop(columns=['class_type','type','legs'])
df_all_binary_test2.head(5)
df_train = str_to_bool(df_train)

df_test = str_to_bool(df_test)



X_train = df_train[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed',

                  'backbone','breathes','venomous','fins','legs','tail','domestic','catsize']]

Y_train = df_train['class_type']
df_train2 = str_to_bool(df_train2)

df_train_plus = pd.concat([df_train,df_train2]) 



X_train_plus = df_train_plus[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed',

                  'backbone','breathes','venomous','fins','legs','tail','domestic','catsize']]

Y_train_plus = df_train_plus['class_type']
df_train_plus.head(5)
databases = {0:['DF com apenas o arquivo zoo-train.csv',X_train,Y_train,df_test],

             1:['DF com os arquivos zoo-train.csv mais zoo-train2.csv',X_train_plus,Y_train_plus,df_test],

             2:['DF com todas a variável leg transformada para bool',X_abt,Y_abt,df_all_binary_test],

             3:['DF com todas a variável leg transformada para bool com os arquivos train.csv + train2.csv',X_abt2,Y_abt2,df_all_binary_test2]

            }
classes = {1:'Mammal',2:'Bird',3:'Reptle',4:'Fish',5:'Amphibian',6:'Bug',7:'Invertebrate','-1':'N/A'}
df_train['class_name']=[classes[x] for x in df_train['class_type']]
Eixo_X = df_train[df_train['feathers']==1]['class_name'].value_counts().index

Eixo_Y = df_train[df_train['feathers']==1]['class_name'].value_counts().values

plt.bar(Eixo_X,Eixo_Y)
Eixo_X = df_train[df_train['tail']==1]['class_name'].value_counts().index

Eixo_Y = df_train[df_train['tail']==1]['class_name'].value_counts().values

plt.bar(Eixo_X,Eixo_Y)
Eixo_X = df_train[df_train['hair']==1]['class_name'].value_counts().index

Eixo_Y = df_train[df_train['hair']==1]['class_name'].value_counts().values

plt.bar(Eixo_X,Eixo_Y)
Eixo_X = df_train[df_train['milk']==1]['class_name'].value_counts().index

Eixo_Y = df_train[df_train['milk']==1]['class_name'].value_counts().values

plt.bar(Eixo_X,Eixo_Y)
Eixo_X = df_train[df_train['domestic']==1]['class_name'].value_counts().index

Eixo_Y = df_train[df_train['domestic']==1]['class_name'].value_counts().values

plt.bar(Eixo_X,Eixo_Y)
Eixo_X = df_train[df_train['predator']==1]['class_name'].value_counts().index

Eixo_Y = df_train[df_train['predator']==1]['class_name'].value_counts().values

plt.bar(Eixo_X,Eixo_Y)
param_list = [{'activation':['relu']},

             {'max_depth':[2,3,4,5,6,7],'random_state':[0,1,2,3,4,5]},

              {'max_depth':[2,3,4,5,6,7],'max_leaf_nodes':[2,3,4,5,6,7,8,9,10]},

              {'kernel':['rbf','linear','sigmoid','poly'],'tol':[0.1,0.001,0.0001,0.00001],'probability':[True,False],

              'degree':[0,1,2,3,4,5,6,7,8,9],'random_state':[0,1,2,3,4]},

              

              {'n_neighbors':[3,5,7,9,11],'metric':['jaccard','matching','dice','kulsinski','rogerstanimoto','russellrao','sokalmichener','sokalsneath'],

                'weights':['uniform','distance'],'algorithm':['ball_tree','auto']

              }

             ]
model_dict={'MLP Classifier':[MLPClassifier(),param_list[0]],

            'Decision Tree': [DecisionTreeClassifier(),param_list[1]],

            'Ramdom Forest': [RandomForestClassifier(),param_list[2]],

            'SCV': [SVC(),param_list[3]],

            'KNN':[KNeighborsClassifier(),param_list[4]]

                       }
model_list=[]

score_list=[]

param_list=[]

best_estimator_list=[]

database_name_list=[]

database_number_list=[]



for m in model_dict:

    for k,d in databases.items():

        grid = GridSearchCV(model_dict[m][0],model_dict[m][1], cv=10, scoring='accuracy')

        grid.fit(d[1],d[2])

        model_list.append(m)

        score_list.append(grid.best_score_)

        param_list.append(grid.best_params_)

        best_estimator_list.append(grid.best_estimator_)

        database_name_list.append(d[0])

        database_number_list.append(k)



results_dict = {'Modelo':model_list,'Base de Dados':database_name_list,'Numero da Base':database_number_list,'Score':score_list,

                'Estimator':best_estimator_list,'Parametros':param_list}        

final_results = pd.DataFrame(results_dict)
final_results=final_results.sort_values('Score',ascending=False)

final_results
best_model = final_results.iloc[0]

best_model[0]
best_model[1]
best_model[3]
model = best_model['Estimator']

model.fit(databases[best_model['Numero da Base']][1],databases[best_model['Numero da Base']][2])
databases[best_model['Numero da Base']][3]['class_type'] = model.predict(databases[best_model['Numero da Base']][3].drop(columns='animal_name'))
df_final = databases[best_model['Numero da Base']][3][['animal_name','class_type']]

df_final.to_csv('arquivo_final.csv',index=False)