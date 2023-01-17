import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline

#Stats and other tools
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import classification_report,confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
#from sklearn.preprocessing import StandardScaler, PolynomialFeatures
#import scipy.stats as stats
#from scipy.stats import uniform
#from scipy.stats import randint as sp_randint
#from sklearn.pipeline import Pipeline


#Models we will test and try
from sklearn.neural_network import MLPClassifier

#from keras import models
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
#from keras import regularizers
from keras import backend as K

cv_k_global = 5 #the amount of f_folds to be used in all CV

datasets=['1','2']
def my_DNN_classifier(in_layer,optimizer='adam',neurons=64,dropout=0.1,activation='relu',activation_final='sigmoid',shape='one'):
    K.clear_session()
    model=None
    model = Sequential()
    model.add(Dense(neurons, input_dim=in_layer, kernel_initializer='normal',activation=activation))
    model.add(Dropout(dropout))
    if(shape=='none'):
        None
    if(shape=='one'):
        model.add(Dense(neurons, kernel_initializer='normal',activation=activation))
        model.add(Dropout(dropout))
    if(shape=='two'):
        model.add(Dense(neurons, kernel_initializer='normal',activation=activation))
        model.add(Dropout(dropout))
        model.add(Dense(neurons, kernel_initializer='normal',activation=activation))
        model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer='normal', activation=activation_final))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=optimizer)
    return model
for dataset in ['deck_pred_2']:#datasets
    print('Grid-Searching for Data-Set '+dataset)
    data=pd.read_csv('../input/data_train_'+dataset+'.csv',index_col='PassengerId').drop(['Missing_Embark','Missing_Deck'],axis=1)
    X=data.drop('Survived',axis=1).as_matrix()
    Y=data['Survived'].as_matrix()
    K.clear_session()
    classifier = None
    classifier = KerasClassifier(build_fn=my_DNN_classifier,verbose=0,batch_size=1024,in_layer=X.shape[1])
    param_grid ={
        'epochs': [1000],
        'optimizer': ['adam','adagrad','rmsprop','adadelta','nadam'],
        'dropout' : np.linspace(0,1,11),
        'neurons' : [16,32],
        'activation' : ['relu','sigmoid','tanh'],
        #'activation_final' : ['relu','sigmoid','tanh'],
        'shape':['none','one','two']
    }
    gscv=GridSearchCV(estimator=classifier,param_grid=param_grid,scoring='accuracy',cv=cv_k_global,verbose=1,n_jobs=-1)
    gscv.fit(X,Y)
    results_df=pd.DataFrame(gscv.cv_results_)
    results_df.to_csv('./CV_DNN_'+dataset+'.csv')
    del classifier
    K.clear_session()
for dataset in datasets:
    print('Grid-Searching for Data-Set '+dataset)
    data=pd.read_csv('../input/data_train_'+dataset+'.csv',index_col='PassengerId').drop(['Missing_Embark','Missing_Deck'],axis=1)
    X=data.drop('Survived',axis=1).as_matrix()
    Y=data['Survived'].as_matrix()
    MLP_class = MLPClassifier(max_iter=1000)
    param_grid ={
        'hidden_layer_sizes': [(8*x,)*y for x in range(1,11) for y in range(1,5)],
        'activation':['logistic', 'tanh', 'relu'],
        'solver':['lbfgs','adam'],
        'alpha':[10**x for x in range(-4,0)],
    }
    gscv=GridSearchCV(MLP_class,param_grid,scoring='accuracy',cv=cv_k_global,verbose=1,n_jobs=-1)
    gscv.fit(X,Y)
    results_df=pd.DataFrame(gscv.cv_results_)
    results_df.to_csv('./CV_MLP_'+dataset+'.csv')
