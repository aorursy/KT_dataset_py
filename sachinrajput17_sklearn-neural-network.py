import numpy as np

import pandas as pd

import random

import itertools

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.datasets import load_digits, load_boston

from sklearn.neural_network import MLPRegressor, MLPClassifier

from sklearn.model_selection import train_test_split ,GridSearchCV

from sklearn.metrics import confusion_matrix,r2_score

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline

#Digits Dataset

digits = load_digits()

x_digits = digits.data

y_digits = digits.target

print("x_digits Datasets size",x_digits.shape,"\n y_digits Datasets size",y_digits.shape)
# Display Images

fig, axes = plt.subplots(2,5,figsize=(10,10),subplot_kw = {'xticks':[],'yticks':[]})

for i,ax in enumerate(axes.flat):

    ax.imshow(digits.images[i], cmap='gray', interpolation='nearest') 

    ax.text(0.5,-0.2,str(digits.target[i]),transform = ax.transAxes)

x_train,x_test,y_train,y_test = train_test_split(x_digits,y_digits,test_size=0.2,random_state=42,stratify= y_digits)

print("Train size: ",x_train.shape,y_train.shape,"Test Size: ",x_test.shape,y_test.shape )
mlp_clf = MLPClassifier(random_state=42)

mlp_clf.fit(x_train,y_train)
y_preds = mlp_clf.predict(x_test)

print(y_test[:20])

print(y_preds[:20])
print("Train Accuracy: ", mlp_clf.score(x_train,y_train))

print("Test Accuracy: ", mlp_clf.score(x_test,y_test))

print("Loss : ", mlp_clf.loss_)
con_max= confusion_matrix(y_test,y_preds)

plt.figure(figsize = (9,9))

sns.heatmap(con_max,annot= True,square= True,cbar= False,cmap='YlOrBr')

plt.xlabel("Predicted Value")

plt.ylabel('Actual_value')

plt.show()

print("Number of Iterations: ",mlp_clf.n_iter_)

print("Output Layer Activation Function :", mlp_clf.out_activation_)
%%time

params = {'activation': ['relu', 'tanh', 'logistic', 'identity','softmax'],

          'hidden_layer_sizes': [(100,), (50,100,), (50,75,100,)],

          'solver': ['adam', 'sgd', 'lbfgs'],

          'learning_rate' : ['constant', 'adaptive', 'invscaling']

         }



mlp_clf_grid = GridSearchCV(MLPClassifier(random_state=42), param_grid=params, n_jobs=-1, cv=5, verbose=5)

mlp_clf_grid.fit(x_train,y_train)
print('Train Accuracy : ',mlp_clf_grid.best_estimator_.score(x_train,y_train))

print('Test Accuracy : ',mlp_clf_grid.best_estimator_.score(x_test, y_test))

print('Grid Search Best Accuracy  :',mlp_clf_grid.best_score_)

print('Best Parameters : ',mlp_clf_grid.best_params_)

print('Best Estimators: ',mlp_clf_grid.best_estimator_)
y_preds = mlp_clf_grid.best_estimator_.predict(x_test)

con_max= confusion_matrix(y_test,y_preds)

plt.figure(figsize = (9,9))

sns.heatmap(con_max,annot= True,square= True,cbar= False,cmap='Pastel1')

plt.xlabel("Predicted Value")

plt.ylabel('Actual_value')

plt.show()
clf_model = MLPClassifier(activation = 'logistic', hidden_layer_sizes= (100,), learning_rate = 'constant', solver = 'adam')

clf_model.fit(x_train,y_train)

y_preds = clf_model.predict(x_test)

print("Loss: ",clf_model.loss_)

print(" Score is ",clf_model.score(x_test,y_test))
from sklearn.datasets import load_boston

boston = load_boston()

x_boston = boston.data

y_boston = boston.target

print("Dataset Sizes ",x_boston.shape,y_boston.shape)
# Spliting dataset into train and test dataset

x_train,x_test,y_train,y_test = train_test_split(x_boston,y_boston,test_size = 0.25, random_state = 42)
# import the regressor

from sklearn.neural_network import MLPRegressor

reg = MLPRegressor(random_state  = 42)

reg.fit(x_train,y_train)
y_preds = reg.predict(x_test)



print(y_preds[:5])

print(y_test[:5])



print("Train Score",reg.score(x_train,y_train))

print("Test Score" , reg.score(x_test,y_test))

print("Loss:",reg.loss_)
print("Number of Coefficents :", len(reg.coefs_))

[weights.shape for weights in reg.coefs_]

print("Number of intecepts :",len(reg.intercepts_))

[intercepts.shape for intercepts in reg.intercepts_]
print("Number of iterations estimators run: ", reg.n_iter_)

print("name of output layer activation function: ", reg.out_activation_)
%%time

reg= MLPRegressor(random_state = 42)

params= {'activation': ['relu','identity','tanh','logistic'],

        'hidden_layer_sizes': [50,100,150] + list(itertools.permutations([50,100,150],2)) + list(itertools.permutations([50,100,150],3)),

         'solver' : ['lbfgs','adam'],

         'learning_rate': ['constant','adaptive','invscaling']

        }



reg_grid = GridSearchCV(reg,param_grid = params,n_jobs= -1,verbose = 10,cv=5)

reg_grid.fit(x_train,y_train)



print("Train score: ", reg_grid.score(x_train,y_train))

print("Test score: ", reg_grid.score(x_test,y_test))

print("Best R2 Score by grid search: ",reg_grid.best_score_)

print("Best Parameters: ", reg_grid.best_params_)

print("Best Estimators: ",reg_grid.best_estimator_)
reg_model = MLPRegressor(activation = 'relu', hidden_layer_sizes = (150, 50, 100), learning_rate= 'constant', solver= 'adam',random_state = 42)

reg_model.fit(x_train,y_train)

y_preds = reg_model.predict(x_test)

print("Loss: ",reg_model.loss_)

print("R2 Score is ",r2_score(y_test,y_preds))