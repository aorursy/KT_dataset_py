import pandas as pd

data = pd.read_csv('/kaggle/input/titanic/train.csv')
data.head()
!pip install pycaret
from pycaret.classification import *

clf1 = setup(data, target = 'Survived', ignore_features = ['Ticket', 'Name', 'PassengerId'], silent = True, session_id = 786) 



#silent is True to perform unattended run when kernel is executed.
compare_models()
catboost = create_model('catboost')
tuned_catboost = tune_model('catboost', optimize = 'AUC', n_iter = 100)
test = pd.read_csv('/kaggle/input/titanic/test.csv')

test.head()
predictions = predict_model(tuned_catboost, data = test)
predictions.head()