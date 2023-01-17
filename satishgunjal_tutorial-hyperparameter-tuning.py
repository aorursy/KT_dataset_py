import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn import svm
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

# Remove rows with missing target values
train_data.dropna(axis=0, subset=['Survived'], inplace=True)
y = train_data.Survived # Target variable             
train_data.drop(['Survived'], axis=1, inplace=True) # Removing target variable from training data

train_data.drop(['Age'], axis=1, inplace=True) # Remove columns with null values

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()

print("Shape of input data: {} and shape of target variable: {}".format(X.shape, y.shape))

# X.head() 
pd.concat([X,y], axis=1).head()# Show first 5 training examples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)
print('X_train dimension= ', X_train.shape)
print('X_test dimension= ', X_test.shape)
print('y_train dimension= ', y_train.shape)
print('y_train dimension= ', y_test.shape)
clf= svm.SVC()
clf.fit(X_train, y_train)
print('Model score using default parameters is = ', clf.score(X_test, y_test))
# Let create parameter grid for GridSearchCV
parameters = {  'C':[0.01, 1, 5],
                'kernel':('linear', 'rbf'),
                'gamma' :('scale', 'auto')
             }
gsc = GridSearchCV(estimator = svm.SVC(), param_grid= parameters,cv= 5,verbose =1)

# Fitting the model for grid search. It will first find the best parameter combination using cross validation. 
# Once it has the best combination, it runs fit again on all data passed to fit (without cross-validation), 
# to built a single new model using the best parameter setting.
gsc.fit(X_train, y_train) 
print(f'Best hyperparameters: {gsc.best_params_}') 
print(f'Best score: {gsc.best_score_}')
print('Detailed GridSearchCV result is as below')
gsc_result = pd.DataFrame(gsc.cv_results_).sort_values('mean_test_score',ascending= False)
gsc_result[['param_C','param_kernel','param_gamma','mean_test_score']]
# n_iter=5 > Number of parameter settings that are sampled. 
# So instaed of 12 it will randomly search for only 5 combinations for each fold
rsc = RandomizedSearchCV(estimator = svm.SVC(), param_distributions= parameters,cv=5,n_iter = 5,verbose =1)
rsc.fit(X_train, y_train)
print(f'Best hyperparameters: {rsc.best_params_}') 
print(f'Best score: {rsc.best_score_}')
print('Detailed RandomizedSearchCV result is as below')
rsc_result = pd.DataFrame(rsc.cv_results_).sort_values('mean_test_score',ascending= False)
rsc_result[['param_C','param_kernel','param_gamma','mean_test_score']]