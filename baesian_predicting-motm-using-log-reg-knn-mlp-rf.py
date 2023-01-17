import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.model_selection import GridSearchCV as gs
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.externals import joblib

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
def model_output(grid_model):
    """Outputs the best mean cross validation score, the test set accuracy and the parameters of a gridsearch"""
    print('Best mean CV accuracy: ', grid_model.best_score_)
    print('Holdout test set accuracy: ', accuracy_score(grid_model.best_estimator_.predict(xtest), ytest))
    print('Best parameters: ', grid_model.best_params_)
fifa = pd.read_csv('../input/FIFA 2018 Statistics.csv')
fifa.head()
heat_map = sns.heatmap(fifa.isnull(), yticklabels = False,
            cbar = False,
            cmap = 'viridis')
fifa = fifa.drop(['1st Goal', 'Own goal Time', 'Date', 'Team', 'Opponent', 'Round'], axis = 1)
fifa['Own goals'] = fifa['Own goals'].fillna(0).astype(int)
fifa['PSO'] = pd.get_dummies(fifa.PSO).Yes
fifa['Man of the Match'] = pd.get_dummies(fifa['Man of the Match']).Yes
game_group = [n for n in range(len(fifa)//2)]
game_group = np.repeat(game_group, 2)

fifa['winner'] = fifa.groupby(game_group)['Goal Scored'].transform(lambda x: x == max(x))
fifa['winner'] = fifa['winner'].map({True: 1, False: 0})
fifa_x = fifa.drop('Man of the Match', axis = 1)
fifa_y = fifa['Man of the Match']
f = plt.figure(figsize = (20, 15))

## Add a density plot for each of the continuous predictors
for i in range(0, 15):
    f.add_subplot(4, 4, i + 1)
    fifa.iloc[:, i].groupby(fifa_y).plot(kind = 'kde', title = fifa.columns[i])
scaler = StandardScaler()
fifa_x = scaler.fit_transform(fifa_x)
fifa_x = pd.DataFrame(fifa_x)
xtrain, xtest, ytrain, ytest = train_test_split(fifa_x, 
                                                fifa_y, 
                                                random_state = 42, 
                                                test_size = .33,
                                                stratify = fifa_y)
log_reg = LogisticRegression()
log_reg.fit(xtrain, ytrain)
preds = log_reg.predict(xtest)
print(confusion_matrix(preds, ytest))
print('Test accuracy: ', accuracy_score(preds, ytest))
errors_knn = pd.DataFrame(columns = ['k_value', 'train_acc', 'test_acc'])

for n in range(1, len(xtrain)):
    knn_clf = knn(n_neighbors = n)
    knn_clf.fit(X = xtrain, y = ytrain)
    
    preds = knn_clf.predict(xtrain)
    errors_knn.loc[n, 'train_acc'] = accuracy_score(preds, ytrain)
    
    preds = knn_clf.predict(xtest)
    errors_knn.loc[n, 'test_acc'] = accuracy_score(preds, ytest)
errors_knn.plot(title = 'K Value Selection')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()
preds = knn_clf.predict(xtest)
print(confusion_matrix(preds, ytest))
print('Test accuracy: ', errors_knn.test_acc.max())
mlp_clf = mlp()

grid = {'hidden_layer_sizes': [(10, 10), (20, 10), (30, 10),
                               (10, 20), (20, 20), (30, 20),
                               (10, 30), (20, 30), (30, 30)], 
        'max_iter': [1000, 2000], 
        'learning_rate_init': [1e-10, 1e-5, 1e-3, 1e-2, 1e-1],
        'random_state': [420]}
grid_mlp = gs(mlp_clf, grid, cv = 10)

grid_mlp.fit(xtrain, ytrain)
model_output(grid_mlp)
rand_for = rf()

grid = {'n_estimators': [5, 10, 15, 20, 30, 50, 100, 200, 500],
        'max_depth' : [None, 2, 3, 5, 10, 20],
        'criterion': ['gini', 'entropy'],
        'random_state' : [69]}

grid_rand = gs(rand_for, grid, cv = 10)

grid_rand.fit(xtrain, ytrain)
model_output(grid_rand)
feat_imp_rf = pd.DataFrame({'Feature' : fifa.drop('Man of the Match', axis = 1).columns,
                            'Importance' : grid_rand.best_estimator_.feature_importances_})

feat_imp_rf.set_index('Feature', inplace = True)
feat_imp_rf.sort_values('Importance', inplace = True)

feat_imp_rf.plot(kind = 'barh', legend = None, title = 'Feature Importance')
plt.show()
joblib.dump(grid_rand, 'fifa_rf.pkl')