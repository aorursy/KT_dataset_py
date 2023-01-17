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
import warnings

warnings.filterwarnings('ignore')
import pandas as pd

df=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

df.head()
import numpy as np

df['Glucose']=np.where(df['Glucose']==0,df['Glucose'].median(),df['Glucose'])

df.head(10)
#### Independent And Dependent features

X=df.drop('Outcome',axis=1)

y=df['Outcome']
pd.DataFrame(X,columns=df.columns[:-1])
#### Train Test Split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
from sklearn.ensemble import RandomForestClassifier

rf_classifier=RandomForestClassifier(n_estimators=10).fit(X_train,y_train)

prediction=rf_classifier.predict(X_test)
y.value_counts()
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

print(confusion_matrix(y_test,prediction))

print(accuracy_score(y_test,prediction))

print(classification_report(y_test,prediction))
### Manual Hyperparameter Tuning

model=RandomForestClassifier(n_estimators=300,criterion='entropy',

                             max_features='sqrt',min_samples_leaf=10,random_state=100).fit(X_train,y_train)

predictions=model.predict(X_test)

print(confusion_matrix(y_test,predictions))

print(accuracy_score(y_test,predictions))

print(classification_report(y_test,predictions))
import numpy as np

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt','log2']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 1000,10)]

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10,14]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4,6,8]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features':max_features,

               'max_depth':max_depth,

               'min_samples_split':min_samples_split,

               'min_samples_leaf':min_samples_leaf,

               'criterion':['gini','entropy']}
print(random_grid)
rf=RandomForestClassifier()

rf_randomcv=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=100,cv=3,verbose=2,

                               random_state=100,n_jobs=-1)

### fit the randomized model

rf_randomcv.fit(X_train,y_train)
rf_randomcv.best_params_
rf_randomcv
best_random_grid=rf_randomcv.best_estimator_
from sklearn.metrics import accuracy_score

y_pred=best_random_grid.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print("Accuracy Score:{}" .format(accuracy_score(y_test,y_pred)))

print("Classification report: {}".format(classification_report(y_test,y_pred)))


rf_randomcv.best_params_
from sklearn.model_selection import GridSearchCV

param_grid={

    'criterion':[rf_randomcv.best_params_['criterion']],

    'max_depth':[rf_randomcv.best_params_['max_depth']],

    'max_features':[rf_randomcv.best_params_['max_features']],

    'min_samples_leaf': [rf_randomcv.best_params_['min_samples_leaf'], 

                         rf_randomcv.best_params_['min_samples_leaf']+2, 

                         rf_randomcv.best_params_['min_samples_leaf'] + 4],

    'min_samples_split':[rf_randomcv.best_params_['min_samples_split']-2,

                        rf_randomcv.best_params_['min_samples_split']-1,

                        rf_randomcv.best_params_['min_samples_split'],

                        rf_randomcv.best_params_['min_samples_split']+1,

                        rf_randomcv.best_params_['min_samples_split']+2] ,

    'n_estimators': [rf_randomcv.best_params_['n_estimators'] - 200, rf_randomcv.best_params_['n_estimators'] - 100, 

                     rf_randomcv.best_params_['n_estimators'], 

                     rf_randomcv.best_params_['n_estimators'] + 100, rf_randomcv.best_params_['n_estimators'] + 200]

    

}

param_grid
#### Fit the grid_search to the data

rf=RandomForestClassifier()

grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=10,n_jobs=-1,verbose=2)

grid_search.fit(X_train,y_train)
grid_search.best_params_
best_grid_search = grid_search.best_estimator_

best_grid_search
y_pred=best_grid_search.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print("Accuracy Score {}".format(accuracy_score(y_test,y_pred)))

print("Classification report: {}".format(classification_report(y_test,y_pred)))
from hyperopt import hp,fmin,tpe,Trials,STATUS_OK
space = {'criterion': hp.choice('criterion', ['entropy', 'gini']),

         'max_depth':hp.quniform('max_depth',10,1200,10),

         'max_features':hp.choice('max_features',['auto','sqrt','log',None]),

         'min_samples_leaf':hp.uniform('min_samples_leaf',0,0.5),

         'min_samples_split':hp.uniform('min_samples_split',0,1),

         'n_estimators':hp.choice('n_estimators',[10, 50, 300, 750, 1200,1300,1500])

        

        }
space
def objective(space):

    model = RandomForestClassifier(criterion = space['criterion'], max_depth = space['max_depth'],

                                 max_features = space['max_features'],

                                 min_samples_leaf = space['min_samples_leaf'],

                                 min_samples_split = space['min_samples_split'],

                                 n_estimators = space['n_estimators'], 

                                 )

    

    accuracy = cross_val_score(model, X_train, y_train, cv = 5).mean()



    # We aim to maximize accuracy, therefore we return it as a negative value

    return {'loss': -accuracy, 'status': STATUS_OK }

    
from sklearn.model_selection import cross_val_score

trials = Trials()

best = fmin(fn= objective,

            space= space,

            algo= tpe.suggest,

            max_evals = 80,

            trials= trials)

best
crit={0:'entropy',1:'gini'}

feat={0: 'auto', 1: 'sqrt', 2: 'log2', 3: None}

est={0: 10, 1: 50, 2: 300, 3: 750, 4: 1200,5:1300,6:1500}



print(crit[best['criterion']])

print(feat[best['max_features']])

print(est[best['n_estimators']])
print(best['min_samples_split'])

print(best['min_samples_leaf'])
trainedforest=RandomForestClassifier(criterion=crit[best['criterion']],max_depth=best['max_depth'],

                                    max_features=feat[best['max_features']],

                                    min_samples_leaf=best['min_samples_leaf'],

                                    min_samples_split=best['min_samples_split'],

                                    n_estimators=est[best['n_estimators']]).fit(X_train,y_train)

predforest=trainedforest.predict(X_test)

print(confusion_matrix(y_test,predforest))

print("Accuracuy Score: {}" .format(accuracy_score(y_test,predforest)))

print(classification_report(y_test,predforest))

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt','log2']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 1000,10)]

#min number of samples required to split a node

min_samples_split=[2,5,10,14]

#min number of samples required at each leaf node

min_samples_leaf=[1,2,4,6,8]

# Create the random grid

param={'n_estimators':n_estimators,

            'max_features':max_features,

            'max_depth':max_depth,

            'min_samples_split':min_samples_split,

            'min_samples_leaf':min_samples_leaf,

            'criterion':['gini','entropy']

      }

print(param)
from tpot import TPOTClassifier





tpot_classifier=TPOTClassifier(generations=5,population_size=24,offspring_size=12,

                              verbosity=2,early_stop=12,

                              config_dict={'sklearn.ensemble.RandomForestClassifier': param},

                              cv=4,scoring='accuracy')

tpot_classifier.fit(X_train,y_train)
accuracy=tpot_classifier.score(X_test,y_test)

print(accuracy)
import optuna

import sklearn.svm

def objective(trial):

    

    classifier=trial.suggest_categorical('classifier',['RandomForest','SVC'])

    

    if classifier=='RandomForest':

        n_estimators=trial.suggest_int('n_estimators', 200, 2000,10)

        max_depth=int(trial.suggest_float('max_depth', 10, 100, log=True))

        

        clf=sklearn.ensemble.RandomForestClassifier(

            n_estimators=n_estimators,max_depth=max_depth)

    else:

        c = trial.suggest_float('svc_c', 1e-10, 1e10, log=True)

        

        clf = sklearn.svm.SVC(C=c, gamma='auto')

    

    return sklearn.model_selection.cross_val_score(

           clf,X_train,y_train,n_jobs=-1,cv=3).mean()
study=optuna.create_study(direction='maximize')

study.optimize(objective,n_trials=100)





trial=study.best_trial



print("Accuracy: {}".format(trial.value))

print("Best Hyperparameters: {}".format(trial.params))
trial
study.best_params
rf=RandomForestClassifier(n_estimators=330,max_depth=30)

rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))
optuna.visualization.plot_optimization_history(study)       ## In this graph the best value and Objective value has been displayed
optuna.visualization.plot_slice(study)                       ## the graph showing the accuracies of model which is coming 

                                                             ##to be greater than 70%

                                         ## Darker the colour  better is ths accuracy
optuna.visualization.plot_contour(study,params=['n_estimators','max_depth'])   

                                                                             ## In this Plot we are plotting the Accuracy surface

                                                                             ## involved in Random Forest.