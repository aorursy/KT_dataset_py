# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import log_loss



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename)) 



# Any results you write to the current directory are saved as output.
training_set = pd.read_csv('/kaggle/input/create-training-set-with-four-factor-stats/training_set.csv')

record = pd.read_csv('/kaggle/input/create-training-set-with-four-factor-stats/record.csv')

submission = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')
training_set.describe()
import sklearn.base



def model_test(model, model_params):

    model_name = type(model).__name__

    print(model_name)

    model_results[model_name + ' Accuracy'] = 0

    model_results[model_name + ' LogLoss'] = 0    



    cvresults = pd.DataFrame()

    gs = GridSearchCV(model, model_params, scoring='neg_log_loss', n_jobs=-1, verbose = 3, cv = 3)

    x_train = training_set[x_vars]

    y_train = training_set['Result']



    cv = gs.fit(x_train, y_train)

    params = cv.best_params_

    results = cv.cv_results_

    cvresults = cvresults.append(results, ignore_index=True)



    clf = model.set_params(**params)

           

    for season in seasons:

        print('Processing ', season)

        x_train = training_set[training_set['Season'] != season][x_vars]        

        x_test = training_set[training_set['Season'] == season][x_vars]

        y_train = training_set[training_set['Season'] != season]['Result']

        y_test = training_set[training_set['Season'] == season]['Result']

        

        model = clf.fit(x_train, y_train)

        y_proba = model.predict_proba(x_test)[:,1]

        

        

        model_results.loc[model_results['Season']==season,model_name + ' Accuracy'] = model.score(x_test,y_test)

        model_results.loc[model_results['Season']==season,model_name + ' LogLoss'] = log_loss(y_test, y_proba)

        

    return model_results, cvresults
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



x_vars = ['deltaSeed','deltaRPI','deltaPace','deltaAdjORtng','deltaAdjDRtng','deltaOeFG','deltaTOP','deltaOR%','deltaFTR']



rfc = RandomForestClassifier()

rfcparams = {'n_estimators' : [100,200,300],

             'criterion' : ['gini','entropy'],

             'min_samples_split' : [2,4,],

             'min_samples_leaf' : [2,4],

             'random_state' : [0],}



mlp = MLPClassifier()

mlpparams = {'solver' : ['lbfgs'],

             'max_iter': [1000, 2500],

             'alpha' : 10.0 ** -np.arange(1, 5),

             'hidden_layer_sizes' : [10, 25, 50],

             'random_state' : [0],}



knn = KNeighborsClassifier()

knnparams = {'weights' : ['uniform','distance']}



svc = SVC()

svcparams = {'probability' : [True],

             'random_state' : [0]}



lr = LogisticRegression()

lrparams = {'C':[0.5,1.0,2.0,3.0],

            'random_state' : [0],

            'max_iter' : [100,200,500]}



eclf = VotingClassifier(estimators = [('rfc',rfc),('mlp',mlp),('svc',svc),('lr',lr)], voting='soft')



models = {rfc : rfcparams,

          mlp : mlpparams,

          knn : knnparams,

          svc : svcparams,

          lr : lrparams,

          }
model_results = pd.DataFrame()

seasons = list(training_set['Season'].unique())

model_results['Season'] = seasons

for model in models:

    model_results, cvresults = model_test(model, model_params = models[model])
model_results.describe()


params = {'rfc__n_estimators' : [100,200,300], 

          'rfc__criterion' : ['gini'],

          'rfc__min_samples_split' : [2,4,],

          'rfc__min_samples_leaf' : [2,4],

          'mlp__solver' : ['lbfgs'],

          'mlp__max_iter': [1000, 2500],

          'mlp__alpha' : 10.0 ** -np.arange(1, 5),

          'mlp__hidden_layer_sizes' : [10, 25, 50],

          'mlp__random_state' : [0],

          'mlp__verbose' : [False],

          'lr__C':[0.5,1.0,2.0,3.0],

          'lr__random_state' : [0],

          'lr__max_iter' : [100,200,500],

          'svc__probability':[True]}



eclf = VotingClassifier(estimators = [('rfc',rfc),('mlp',mlp),('svc',svc),('lr',lr)], voting='soft')

        

clf = GridSearchCV(estimator=eclf, param_grid=params, cv=3)



train_x, test_x, train_y, test_y = train_test_split(training_set[x_vars], training_set['Result'], test_size = 0.3)
clf = clf.fit(train_x, train_y)
cvresults