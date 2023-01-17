import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

from sklearn.model_selection import train_test_split        
df = pd.read_csv("../input/heart-disease-uci/heart.csv")

df.head()
y = df.target.values

x_data = df.drop(['target'], axis = 1)



# Normalize

#x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
#!pip install -U scikit-learn



import sklearn

print(sklearn.__version__)
!curl -OL https://github.com/AxeldeRomblay/mlbox/tarball/3.0-dev
!apt-get -y remove swig



!apt-get -y install swig3.0 build-essential -y



!ln -s /usr/bin/swig3.0 /usr/bin/swig

!apt-get -y install build-essential



#!pip install --upgrade setuptools

#!pip install auto-sklearn

#!pip install --no-cache-dir -v pyrfr





#try:

 #   import autosklearn.classification

#except:

 #   pass
!pip install git+https://github.com/automl/auto-sklearn
#!pip uninstall -y scikit-learn



#import sklearn

#print(sklearn.__version__)
#!pip install scikit-learn



#import sklearn

#print(sklearn.__version__)
# from sklearn import model_selection, metrics

#import sklearn

#import autosklearn

#import autosklearn.classification



#%timeit



#X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x_data, y, random_state=1)



#automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=3600,

#per_run_time_limit=300,resampling_strategy='cv', resampling_strategy_arguments={'folds': 5},

#include_preprocessors=["no_preprocessing"],ensemble_size=2)



# Do not construct ensembles in parallel to avoid using more than one

# core at a time. The ensemble will be constructed after auto-sklearn

# finished fitting all machine learning models.



#automl.fit(X_train, y_train)



# This call to fit_ensemble uses all models trained in the previous call

# to fit to build an ensemble which can be used with automl.predict()



#automl.fit_ensemble(y_train, ensemble_size=50)



#print(automl.show_models())



#predictions = automl.predict(X_test)



#print(automl.sprint_statistics())



#print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))    
!pip install mlbox
#import warnings

#warnings.filterwarnings("ignore")



#from mlbox.preprocessing.reader import Reader

#from mlbox.preprocessing.drift_thresholder import Drift_thresholder

#from mlbox.optimisation.optimiser import Optimiser 

#from mlbox.prediction.predictor import Predictor
paths = ["../input/titanic/train.csv", "../input/titanic/test.csv"] 



target_name = "Survived"
#rd = Reader(sep=",")

#df = rd.train_test_split(paths, target_name)
#df["train"].head()
#dft = Drift_thresholder()

#df = dft.fit_transform(df)
#opt = Optimiser()



# Then we can run it using the default model configuration set as default (LightGBM) without any autoML or complex grid search.



# This should be the first baseline



#warnings.filterwarnings('ignore', category=DeprecationWarning)

#score = opt.evaluate(None, df)
space = {

        'ne__numerical_strategy':{"search":"choice","space":[0, "mean"]},

        'ce__strategy':{"search":"choice", "space":["label_encoding", "random_projection", "entity_embedding"]}, 

        'fs__threshold':{"search":"uniform", "space":[0.001, 0.2]}, 

        'est__strategy':{"search":"choice", "space":["RandomForest", "ExtraTrees", "LightGBM"]},

        'est__max_depth':{"search":"choice", "space":[8, 9, 10, 11, 12, 13]}

        }
#opt = Optimiser(scoring="accuracy",n_folds=5)
#opt.evaluate(params, df)



#best=opt.optimise(space, df, 40)
import pandas as pd



#prd = Predictor()

#prd.fit_predict(best, df)
import h2o

from h2o.automl import H2OAutoML
h2o.init()

#df = h2o.import_file("../input/heart-disease-uci/heart.csv")

df = h2o.import_file("../input/titanic/train.csv")
train, test = df.split_frame([0.7], seed=42)
train.head(2)
y = "Survived"



ignore = ["Survived", "PassengerId", "Name"] 



x = list(set(train.names) - set(ignore))
splits = df.split_frame(ratios=[0.7], seed=1)



train = splits[0]



test = splits[1]
y = "Survived" 



x = df.columns 



x.remove(y) 



x.remove("PassengerId")



x.remove("Name")
#H2OAutoML(nfolds=5, max_runtime_secs=3600, max_models=None, stopping_metric='AUTO', stopping_tolerance=None, stopping_rounds=3, seed=None, project_name=None)



aml = H2OAutoML(max_models=25, max_runtime_secs_per_model=30, seed=42)



%time aml.train(x=x, y=y, training_frame=train)
aml = H2OAutoML(max_runtime_secs=120, seed=1)



aml.train(x=x,y=y, training_frame=train)
lb = aml.leaderboard



# lb.head(rows=lb.nrows)



lb.head()
from h2o.automl import get_leaderboard



lb2 = get_leaderboard(aml, extra_columns='ALL')



lb2.head(rows=lb2.nrows)
# Get model ids for all models in the AutoML Leaderboard



model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])



# Get the "All Models" Stacked Ensemble model



se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_BestOfFamily" in mid][0])



# Get the Stacked Ensemble metalearner model



metalearner = h2o.get_model(se.metalearner()['name'])

metalearner.coef()
metalearner.std_coef_plot()
aml.leader.model_performance(test_data=test)
#%matplotlib inline

#aml.leader.model_performance(test_data=test).plot()
# Lastly, let's make some predictions on our test set.





pred = aml.predict(test)



pred.head()
h2o.save_model(aml.leader, path="./output")
df = pd.read_csv("../input/titanic/train.csv") 

df.head()
df = df.fillna(-999)



df_class = df['Survived'].values
from sklearn.model_selection import train_test_split



training_indices, validation_indices = training_indices, testing_indices = train_test_split(df.index,

                                                                                            stratify = df_class,

                                                                                            train_size=0.75, test_size=0.25)
training_indices.size, validation_indices.size
#df.info()

df.drop(['Name', 'PassengerId', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
#from tpot import TPOTClassifier

#from tpot import TPOTRegressor



#tpot = TPOTClassifier(generations=5, verbosity=2)

#tpot.fit(df.drop('Survived', axis=1).loc[training_indices].values, df.loc[training_indices,'Survived'].values)
#tpot.score(df.drop('Survived',axis=1).loc[validation_indices].values,  df.loc[validation_indices, 'Survived'].values)
#tpot.export('pipeline.py')
!pip install autokeras



!pip install git+https://github.com/keras-team/keras-tuner.git@1.0.2rc1
import pandas as pd

import numpy as np

import autokeras as ak



x_data = pd.read_csv("../input/titanic/train.csv")



print(type(x_data))



y = x_data.pop('Survived')



print(type(y))
y_train = pd.DataFrame(y)



print(type(y_train)) 



# You can also use numpy.ndarray for x_train and y_train.



x_train = x_data.to_numpy().astype(np.unicode)



y_train = y.to_numpy()



print(type(x_train)) 



print(type(y_train)) 
# Preparing testing data.



x_test = pd.read_csv("../input/titanic/test.csv")
import sklearn



from sklearn import model_selection, metrics

%timeit



x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_data, y, random_state=1)
clf = ak.StructuredDataClassifier(overwrite=True , max_trials=3)
# Feed the structured data classifier with training data.



clf.fit(x_train, y_train, epochs=10)
# Predict with the best model.



predicted_y = clf.predict(x_test)



# Evaluate the best model with testing data.



print(clf.evaluate(x_test, y_test))
clf.export_model()