import pandas as pd
from sklearn.svm import SVC
data = pd.read_csv('/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')
#Get Target data 
y = data['default.payment.next.month']

#Load X Variables into a Pandas Dataframe with columns 
X = data.drop(['ID','default.payment.next.month'], axis = 1)
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']} 
SVM_Model = SVC(gamma='auto')
from sklearn.model_selection import RandomizedSearchCV 
random_Grid = RandomizedSearchCV (estimator = SVM_Model, param_distributions = param_grid, cv = 3, verbose=2, n_jobs = 4)
from sklearn.model_selection import GridSearchCV 
Grid_Search = GridSearchCV (estimator = SVM_Model, param_grid = param_grid, cv = 3, verbose=2, n_jobs = 4)
random_Grid.fit(X,y)
Grid_Search.fit(X,y)
print (f'Accuracy - : {random_Grid.score(X,y):.3f}')
print (f'Accuracy - : {Grid_Search.score(X,y):.3f}')