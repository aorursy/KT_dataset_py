import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



#test_id will be used later, so save it

test_id = test['id']



train.drop(['id'], axis=1, inplace=True)

test.drop(['id'], axis=1, inplace=True)
sns.countplot(x='type', data=train, palette='Set3')
sns.countplot(x='color', data=train, palette='Set3')
pd.crosstab(train.type,train.color)
bytreatment = train.groupby(['color','type'])

bytreatment['bone_length'].agg([np.median, np.mean, np.std, len])
# per color

some_color = ['green','white']

some_type = ['Ghost']

sns.set()

sns.pairplot(train.loc[train['color'].isin(some_color) & train['type'].isin(some_type)], hue='color', palette='Set1') 
sns.set()

sns.pairplot(train, hue="type", palette='Set1', diag_kind="kde")
# correlation matrix for margin features

corr = train.corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(8, 6))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap)
# Create some new feature to make classes more distinct



train['hair_soul'] = train['hair_length'] * train['has_soul']

train['hair_bone'] = train['hair_length'] * train['bone_length']

train['soul_bone'] = train['has_soul'] * train['bone_length']



#train['hair_soul_bone'] = train['hair_length'] * train['has_soul'] * train['bone_length']



test['hair_soul'] = test['hair_length'] * test['has_soul']

test['hair_bone'] = test['hair_length'] * test['bone_length']

test['soul_bone'] = test['has_soul'] * test['bone_length']
features = ["bone_length", "rotting_flesh", "hair_length", "has_soul", "hair_soul", "hair_bone", "soul_bone"]



X_Train = pd.concat([train[features], pd.get_dummies(train["color"])], axis=1)



y_Train = train["type"]



X_Test = pd.concat([test[features], pd.get_dummies(test["color"])], axis=1)
from sklearn.preprocessing import LabelEncoder



le_y = LabelEncoder()

y_Train = le_y.fit_transform(y_Train)



print (X_Train.shape)

print (y_Train.shape)
#Splitting data for validation

X_Train1, X_Test1, y_Train1, y_Test1 = train_test_split(X_Train, y_Train, test_size=0.20, random_state=36)



print (X_Train1.shape)

print (y_Train1.shape)


from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(n_estimators = 20,

                                criterion = 'entropy',

                                max_features = 'auto')

parameter_grid = {

                  'max_depth' : [None, 5, 20, 100],

                  'min_samples_split' : [2, 5, 7],

                  'min_weight_fraction_leaf' : [0.0, 0.1],

                  'max_leaf_nodes' : [20, 30],

                 }



grid_search_rf = GridSearchCV(forest, param_grid=parameter_grid, cv=StratifiedKFold(3))

grid_search_rf.fit(X_Train1, y_Train1)



print('Best score: {}'.format(grid_search_rf.best_score_))

print('Best parameters: {}'.format(grid_search_rf.best_params_))
# make predection on randon forest

y_test_pred = grid_search_rf.predict(X_Test1)

y_test_pred



from sklearn.metrics import accuracy_score

accuracy_score(y_Test1, y_test_pred)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()



parameter_grid = {'solver' : ['newton-cg', 'lbfgs'],

                  'multi_class' : ['multinomial'],

                  'C' : [0.005, 0.01, 1, 10],

                  'tol': [0.0001, 0.001, 0.005, 0.01]

                 }



grid_search_logit = GridSearchCV(logreg, param_grid=parameter_grid, cv=StratifiedKFold(3))

grid_search_logit.fit(X_Train1, y_Train1)



print('Best score: {}'.format(grid_search_logit.best_score_))

print('Best parameters: {}'.format(grid_search_logit.best_params_))
# make predection on logit forest

y_test_pred = grid_search_logit.predict(X_Test1)

y_test_pred



from sklearn.metrics import accuracy_score

accuracy_score(y_Test1, y_test_pred)
from sklearn import svm



svr = svm.SVC()



parameter_grid = {'kernel':('linear', 'rbf', 'poly'), 'C':[0.005, 0.01, 1, 10, 100, 1000], 'degree':[2, 3]}



grid_search_svc = GridSearchCV(svr, param_grid=parameter_grid, cv=StratifiedKFold(3))

grid_search_svc.fit(X_Train1, y_Train1)



print('Best score: {}'.format(grid_search_svc.best_score_))

print('Best parameters: {}'.format(grid_search_svc.best_params_))
# make predection on logit forest

y_test_pred = grid_search_svc.predict(X_Test1)

y_test_pred



from sklearn.metrics import accuracy_score

accuracy_score(y_Test1, y_test_pred)
from sklearn import neural_network



import warnings

warnings.filterwarnings("ignore")

                        

nnet = neural_network.MLPClassifier()



parameter_grid = {'activation':['logistic','relu'], 'hidden_layer_sizes':[10,15,25], 'alpha':[1e-4, 1e-3, 1e-2, 1e-1], 'solver':['lbfgs','sgd'], 'learning_rate':['constant', 'adaptive']}



grid_search_nnet = GridSearchCV(nnet, param_grid=parameter_grid, cv=StratifiedKFold(3))

grid_search_nnet.fit(X_Train1, y_Train1)



print('Best score: {}'.format(grid_search_nnet.best_score_))

print('Best parameters: {}'.format(grid_search_nnet.best_params_))
from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()



parameter_grid = {'average':[True],'loss':['log','perceptron','squared_loss'], 'penalty':[None,'l2','l1'], 'alpha':[0.001,0.01, 0.1, 1], 'l1_ratio':[0.15, 0.1, 0.3], 'learning_rate':['optimal'], 'random_state':[0]}

              

grid_search_sgd = GridSearchCV(sgd, param_grid=parameter_grid, cv=StratifiedKFold(3))

grid_search_sgd.fit(X_Train1, y_Train1)



print('Best score: {}'.format(grid_search_sgd.best_score_))

print('Best parameters: {}'.format(grid_search_sgd.best_params_))
from sklearn import ensemble



from sklearn.dummy import DummyClassifier

from sklearn.linear_model import Perceptron

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC



bag = ensemble.BaggingClassifier()



parameter_grid = {'base_estimator':[None,DummyClassifier(),Perceptron(),DecisionTreeClassifier(),KNeighborsClassifier(),SVC()], "max_samples": [0.5, 1.0], "max_features": [1, 2, 4],"bootstrap": [True, False],"bootstrap_features": [True, False],'random_state':[0]}



grid_search_bag = GridSearchCV(bag, param_grid=parameter_grid, scoring='log_loss', refit='True', cv=StratifiedKFold(3))

grid_search_bag.fit(X_Train1, y_Train1)



print('Best score: {}'.format(grid_search_bag.best_score_))

print('Best parameters: {}'.format(grid_search_bag.best_params_))
import tensorflow as tf

from tensorflow.contrib import learn

x=tf.contrib.learn.infer_real_valued_columns_from_input(X_Train1)

tf_clf_dnn = learn.DNNClassifier(hidden_units=[16], n_classes=3, feature_columns=x, activation_fn=tf.sigmoid)

tf_clf_dnn.fit(X_Train1, y_Train1, max_steps=5000)



from sklearn.metrics import accuracy_score as acc_s



print(acc_s(y_Test1,tf_clf_dnn.predict(X_Test1)))
clf = tf.contrib.learn.LinearClassifier(

        feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(X_Train),

        n_classes=3,

        #optimizer=tf.train.FtrlOptimizer(

        #    learning_rate=0.1,

        #    l2_regularization_strength=0.001,

        optimizer=tf.train.AdagradOptimizer(

            learning_rate=0.5,

        ))



clf.fit(X_Train, y_Train, steps=500)



y_pred = clf.predict(X_Test)



pred = le_y.inverse_transform(y_pred)
submission = pd.DataFrame({'id':test_id,

                           'type':pred})
submission.to_csv('submission.csv', index=False)