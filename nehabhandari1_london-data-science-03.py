import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics#Import scikit-learn metrics module for accuracy calculation

from sklearn.model_selection import train_test_split,cross_val_score

import seaborn as sns

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO  

from IPython.display import Image 

from sklearn import tree

from sklearn import metrics

from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier 

from sklearn.linear_model import LogisticRegression 

from sklearn.preprocessing import StandardScaler

import pandas as pd

test = pd.read_csv("../input/data-science-london-scikit-learn/test.csv",header=None)

train = pd.read_csv("../input/data-science-london-scikit-learn/train.csv",header=None)

trainLabels = pd.read_csv("../input/data-science-london-scikit-learn/trainLabels.csv",header=None)
train.head()
trainLabels.columns = ['Target']

pd.crosstab(index=trainLabels['Target'].astype('category'),  # Make a crosstab

                              columns="count")   



train.iloc[:,0:10].describe()


Full_Data = pd.concat([train,trainLabels],axis=1)

Full_Data

Mean_Sum = Full_Data.groupby('Target').agg('mean')

Mean_Sum["Type"] = "Mean"



Sum_Sum = Full_Data.groupby('Target').agg('sum')

Sum_Sum["Type"] = "Sum"



Sum_By_Target = pd.concat([Mean_Sum,Sum_Sum])

Sum_By_Target



Full_Data[Full_Data['Target'] == 0].describe()
Full_Data[Full_Data['Target'] == 1].describe()
##trying to combine predictor(x) and traget(y), as both are store in differnt varible and combing both will give entire 

##training data which will also include target variable.



X,y = train,np.ravel(trainLabels)



##spliting training data into train set and test set. train set has 70% of data while test set has 30% of data. 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaled_logistic_pipe = Pipeline(steps = [('sc', StandardScaler()),('classifier', LogisticRegression())])



#scaled_logistic_param_grid = { "classifier__penalty": ['l2','l1'], "classifier__C": np.logspace(0, 4, 10)}



C = np.logspace(-4, 4, 50)

# Create a list of options for the regularization penalty

penalty = ['l2']

# Create a dictionary of all the parameter options 

# Note has you can access the parameters of steps of a pipeline by using '__’

scaled_logistic_param_grid = dict(classifier__C=C,

                  classifier__penalty=penalty)

np.random.seed(1)



scaled_logistic_grid_search = GridSearchCV(scaled_logistic_pipe, scaled_logistic_param_grid, cv=10)



scaled_logistic_grid_search.fit(X_train, y_train)



scaled_logistic_model = scaled_logistic_grid_search.best_estimator_



 



print('Cross Validation Score:', scaled_logistic_grid_search.best_score_)



print('Best Hyperparameters:  ', scaled_logistic_grid_search.best_params_)



print('Training Accuracy:     ', scaled_logistic_model.score(X_train, y_train))
unscaled_knn_pipe = Pipeline(steps = [('classifier', KNeighborsClassifier())])



unscaled_knn_param_grid = {'classifier__n_neighbors': range(1,10),'classifier__p': [1,2,3]}



np.random.seed(1)



unscaled_knn_grid_search = GridSearchCV(unscaled_knn_pipe, unscaled_knn_param_grid, cv=10, refit='True')



unscaled_knn_grid_search.fit(X_train, y_train)



unscaled_knn_model = unscaled_knn_grid_search.best_estimator_



 



print('Cross Validation Score:', unscaled_knn_grid_search.best_score_)



print('Best Hyperparameters:  ', unscaled_knn_grid_search.best_params_)



print('Training Accuracy:     ', unscaled_knn_model.score(X_train, y_train))
scaled_knn_pipe = Pipeline(steps = [('sc', StandardScaler()),('classifier', KNeighborsClassifier())])



scaled_knn_param_grid = {'classifier__n_neighbors': range(1,10),'classifier__p': [1,2,3]}



np.random.seed(1)



scaled_knn_grid_search = GridSearchCV(scaled_knn_pipe, scaled_knn_param_grid, cv=10, refit='True')



scaled_knn_grid_search.fit(X_train, y_train)



scaled_knn_model = scaled_knn_grid_search.best_estimator_



 



print('Cross Validation Score:', scaled_knn_grid_search.best_score_)



print('Best Hyperparameters:  ', scaled_knn_grid_search.best_params_)



print('Training Accuracy:     ', scaled_knn_model.score(X_train, y_train))
unscaled_tree_pipe = Pipeline(steps = [('decisiontree', DecisionTreeClassifier())])



#Create lists of parameter for Decision Tree Classifier

criterion = ['gini', 'entropy']

max_depth = [1,2,3,4,5,6,7,8,9,10,11,12]

    

unscaled_tree_param_grid = dict(decisiontree__criterion=criterion,decisiontree__max_depth=max_depth)



np.random.seed(1)



unscaled_tree_grid_search = GridSearchCV(unscaled_tree_pipe, unscaled_tree_param_grid, cv=10)



unscaled_tree_grid_search.fit(X_train, y_train)



unscaled_tree_model = unscaled_tree_grid_search.best_estimator_



 



print('Cross Validation Score:', unscaled_tree_grid_search.best_score_)



print('Best Hyperparameters:  ', unscaled_tree_grid_search.best_params_)



print('Training Accuracy:     ', unscaled_tree_model.score(X_train, y_train))
unscaled_rf_pipe = Pipeline([("classifier", RandomForestClassifier())])

unscaled_rf_param_grid = {

                 "classifier__n_estimators": [10, 100, 1000],

                 "classifier__max_depth":[5,8,15,25,30,None],

                 "classifier__min_samples_leaf":[1,2,5,10,15,100],

                 "classifier__max_leaf_nodes": [2, 5,10]}

unscaled_rf_grid_search = GridSearchCV(unscaled_rf_pipe, unscaled_rf_param_grid, cv=5, verbose=0,n_jobs=-1)



unscaled_rf_grid_search.fit(X_train, y_train)



unscaled_rf_model = unscaled_tree_grid_search.best_estimator_



 



print('Cross Validation Score:', unscaled_rf_grid_search.best_score_)



print('Best Hyperparameters:  ', unscaled_rf_grid_search.best_params_)



print('Training Accuracy:     ', unscaled_rf_model.score(X_train, y_train))
final_model = KNeighborsClassifier(n_neighbors = 6)

final_model.fit(train, trainLabels)

y_pred_knn=final_model.predict(X_test)

print("Training final: ", final_model.score(train, trainLabels))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_knn))
pred_test = final_model.predict(test)

pred_test[:5]

pred_test.shape
submission = pd.DataFrame(pred_test)

submission.columns = ['Solution']

submission['Id'] = np.arange(1,submission.shape[0]+1)

submission = submission[['Id', 'Solution']]

submission.head()
filename = 'London_Example.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)