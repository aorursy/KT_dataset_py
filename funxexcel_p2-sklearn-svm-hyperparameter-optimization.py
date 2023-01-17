import pandas as pd
from sklearn.svm import SVC
data = pd.read_csv('/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')
data.head()
#Get Target data 
y = data['default.payment.next.month']

#Load X Variables into a Pandas Dataframe with columns 
X = data.drop(['ID','default.payment.next.month'], axis = 1)
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001], 
              'kernel': ['rbf']} 
SVM_Model = SVC(gamma='auto')
#Setup Random Grid Search
from sklearn.model_selection import RandomizedSearchCV 
rf_Grid = RandomizedSearchCV (estimator = SVM_Model, param_distributions = param_grid, cv = 3, verbose=2, n_jobs = 4)
rf_Grid.fit(X,y)
rf_Grid.best_params_
rf_Grid.best_estimator_
print (f'Accuracy - : {rf_Grid.score(X,y):.3f}')