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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
from scipy.stats import reciprocal
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import StackingClassifier

def data_process(data):
    data = data.drop("Cabin", 1)
    data = data.drop("Embarked", 1)
    data = data.drop("Ticket",1)
    data = data.drop("Name", 1)
    data = data.drop("PassengerId", 1)
    data["Sex"] = LabelEncoder().fit_transform(data["Sex"])
    
    numerical_attr = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    
    for attr in numerical_attr:
        data[attr].fillna(round(data[attr].mean(), 0), inplace=True)
    return data
train_data = data_process(train_data)

sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)

training_set = []
validation_set = []

for train_index, test_index in sss1.split(train_data, train_data["Survived"]):
    training_set = train_data.loc[train_index]
    validation_set = train_data.loc[test_index]

training_features = training_set.drop("Survived", 1).to_numpy()
training_labels = training_set.drop(["Age", "Pclass", "SibSp", "Parch", "Fare", "Sex"], 1).to_numpy().ravel()

validation_features = validation_set.drop("Survived", 1).to_numpy()
validation_labels = validation_set.drop(["Age", "Pclass", "SibSp", "Parch", "Fare", "Sex"], 1).to_numpy().ravel()
# Logistic Regression Modeling 
logistic_reg_baseline = LogisticRegression(solver = 'liblinear', random_state = 1)
logistic_reg_baseline.fit(training_features, training_labels)
log_reg_baseline_train_score = logistic_reg_baseline.score(training_features, training_labels)
print("Training Score for Baseline Model:", log_reg_baseline_train_score)

log_reg_baseline_validation_score = logistic_reg_baseline.score(validation_features, validation_labels)
print("Validation Score for Baseline Model:", log_reg_baseline_validation_score)

# Using GridSearchCV to fine-tune the Logistic Regression model
max_iter_logistic_reg = [i*10 for i in range(10, 31)]
c_logistic_reg = [i/100 for i in range (1,20)]
parameters_logistic_reg = {'C' : c_logistic_reg, 'max_iter' : max_iter_logistic_reg, 'penalty' : ['l1', 'l2']}

logistic_reg_best = GridSearchCV(logistic_reg_baseline, parameters_logistic_reg)
logistic_reg_best.fit(training_features, training_labels)
logistic_reg_finetune = logistic_reg_best.best_estimator_
logistic_reg_finetune.fit(training_features, training_labels)
log_reg_finetune_train_score = logistic_reg_finetune.score(training_features, training_labels)
print("Training Score for Finetuned Model:", log_reg_finetune_train_score)

log_reg_finetune_validation_score = logistic_reg_finetune.score(validation_features, validation_labels)
print("Validation Score for Finetuned Model:", log_reg_finetune_validation_score)
# Random Forest Modeling 
randomforest_baseline = RandomForestClassifier(random_state = 1)
randomforest_baseline.fit(training_features, training_labels)
randomforest_baseline_train_score = randomforest_baseline.score(training_features, training_labels)
print("Training Score for Baseline Model:", randomforest_baseline_train_score)

randomforest_baseline_validation_score = randomforest_baseline.score(validation_features, validation_labels)
print("Validation Score for Baseline Model:", randomforest_baseline_validation_score)

# Using GridSearchCV to fine-tune the Random Forest model
n_estimators_randomforest = [i*10 for i in range(10,21)]
max_features = ['sqrt', 'log2']
max_depth_randomforest = range(2, 6)
parameters_randomforest = {'n_estimators' : n_estimators_randomforest, 'max_features' : max_features, 'max_depth' : max_depth_randomforest}

randomforest_best = GridSearchCV(randomforest_baseline, parameters_randomforest)
randomforest_best.fit(training_features, training_labels)
randomforest_finetune = randomforest_best.best_estimator_

randomforest_finetune.fit(training_features, training_labels)
randomforest_finetune_train_score = randomforest_finetune.score(training_features, training_labels)
print("Training Score for Finetuned Model:", randomforest_finetune_train_score)

randomforest_finetune_validation_score = randomforest_finetune.score(validation_features, validation_labels)
print("Validation Score for Finetuned Model:", randomforest_finetune_validation_score)
# Gradient Boosting Classifier Modeling 
gradboosting_baseline = GradientBoostingClassifier(random_state = 1)
gradboosting_baseline.fit(training_features, training_labels)
gradboosting_baseline_train_score = gradboosting_baseline.score(training_features, training_labels)
print("Training Score for Baseline Model:", gradboosting_baseline_train_score)

gradboosting_baseline_validation_score = gradboosting_baseline.score(validation_features, validation_labels)
print("Validation Score for Baseline Model:", gradboosting_baseline_validation_score)

# # Using GridSearchCV to fine-tune the Gradient Boosting model
learning_rate_gradboosting = [i/100 for i in range(5, 21)]
n_estimators_gradboosting = [i*10 for i in range(10,21)]
max_depth_gradboosting = range(3,6)
parameters_gradboosting = {'learning_rate' : learning_rate_gradboosting, 'n_estimators' : n_estimators_gradboosting, 'max_depth' : max_depth_gradboosting}

gradboosting_best = GridSearchCV(gradboosting_baseline, parameters_gradboosting)
gradboosting_best.fit(training_features, training_labels)
gradboosting_finetune = gradboosting_best.best_estimator_
gradboosting_finetune.fit(training_features, training_labels)
gradboosting_finetune_train_score = gradboosting_finetune.score(training_features, training_labels)
print("Training Score for Finetuned Model:", gradboosting_finetune_train_score)

gradboosting_finetune_validation_score = gradboosting_finetune.score(validation_features, validation_labels)
print("Validation Score for Finetuned Model:", gradboosting_finetune_validation_score)
# SVM Classifier Modeling 
svm_baseline = SVC(random_state = 1)
svm_baseline.fit(training_features, training_labels)
svm_baseline_train_score = svm_baseline.score(training_features, training_labels)
print("Training Score for Baseline Model:", svm_baseline_train_score)

svm_baseline_validation_score = svm_baseline.score(validation_features, validation_labels)
print("Validation Score for Baseline Model:", svm_baseline_validation_score)

# Using GridSearchCV to fine-tune the SVC model
kernel_svm = ['poly', 'rbf', 'sigmoid']
degree_svm = range(3,10)
parameters_svm = {'kernel' : kernel_svm, 'degree' : degree_svm}
svm_best = GridSearchCV(svm_baseline, parameters_svm)
svm_best.fit(training_features, training_labels)
svm_finetune = svm_best.best_estimator_
svm_finetune.fit(training_features, training_labels)
svm_finetune_train_score = svm_finetune.score(training_features, training_labels)
print("Training Score for Finetuned Model:", svm_finetune_train_score)

svm_finetune_validation_score = svm_finetune.score(validation_features, validation_labels)
print("Validation Score for Finetuned Model:", svm_finetune_validation_score)
# AdaBoostClassifier Modeling
adaboost_baseline = AdaBoostClassifier(random_state = 1)
adaboost_baseline.fit(training_features, training_labels)
adaboost_baseline_train_score = adaboost_baseline.score(training_features, training_labels)
print("Training Score for Baseline Model :", adaboost_baseline_train_score)

adaboost_baseline_validation_score = adaboost_baseline.score(validation_features, validation_labels)
print("Validation Score for Baseline Model :", adaboost_baseline_validation_score)

# Using GridSearchCV to fine-tune the AdaBoost model
learning_rate_adaboost = [i/10 for i in range(1, 11)]
n_estimators_adaboost = range(50, 101)
parameters_adaboost = {'n_estimators' : n_estimators_adaboost, 'learning_rate' : learning_rate_adaboost}

adaboost_best = GridSearchCV(adaboost_baseline, parameters_adaboost)
adaboost_best.fit(training_features, training_labels)
adaboost_finetune = adaboost_best.best_estimator_
adaboost_finetune.fit(training_features, training_labels)
adaboost_finetune_train_score = adaboost_finetune.score(training_features, training_labels)
print("Training Score for Finetuned Model:", adaboost_finetune_train_score)

adaboost_finetune_validation_score = adaboost_finetune.score(validation_features, validation_labels)
print("Validation Score for Finetuned Model:", adaboost_finetune_validation_score)
extratree_baseline = ExtraTreeClassifier(random_state = 1)
extratree_baseline.fit(training_features, training_labels)
extratree_baseline_train_score = extratree_baseline.score(training_features, training_labels)
print("Training Score for Baseline Model :", extratree_baseline_train_score)

extratree_baseline_validation_score = extratree_baseline.score(validation_features, validation_labels)
print("Validation Score for the Baseline Model :", extratree_baseline_validation_score)

# Using GridSearchCV to fine-tune the ExtraTreeClassifier model
criterion_extratree = ['gini', 'entropy']
splitter_extratree = ['random']
max_depth_extratree = [None, 3, 5, 7, 11]
parameters_extratree = {'criterion' : criterion_extratree, 'splitter' : splitter_extratree, 'max_depth' : max_depth_extratree}
extratree_best = GridSearchCV(extratree_baseline, parameters_extratree)
extratree_best.fit(training_features, training_labels)
extratree_finetune = extratree_best.best_estimator_
extratree_finetune.fit(training_features, training_labels)
extratree_finetune_train_score = extratree_finetune.score(training_features, training_labels)
print("Training Score for Finetuned Model:", extratree_finetune_train_score)

extratree_finetune_validation_score = extratree_finetune.score(validation_features, validation_labels)
print("Validation Score for Finetuned Model:", extratree_finetune_validation_score)
# Voting Classifier attempt
voting_clf = VotingClassifier(estimators = [('rf', randomforest_finetune), ('gb', gradboosting_finetune),('lg', logistic_reg_finetune), ('ab', adaboost_finetune), ('xtree', extratree_finetune)], voting = 'hard')
voting_clf.fit(training_features, training_labels)
voting_clf_training_score = voting_clf.score(training_features, training_labels)
print("Training Score for the Voting Classifier :", voting_clf_training_score)
voting_clf_validation_score = voting_clf.score(validation_features, validation_labels)
print("Validation Score for the Voting Classifier :", voting_clf_validation_score)
# Basic Stacking Classifier attempt
estimators_stacking = [('rf', randomforest_finetune), ('gb', gradboosting_finetune),('lg', logistic_reg_finetune), ('ab', adaboost_finetune), ('xtree', extratree_finetune)]
final_estimator = [LogisticRegression(random_state = 1)]
parameters_stacking = {'final_estimator' : final_estimator}
stacking_clf = StackingClassifier(estimators_stacking, LogisticRegression(random_state = 1))
stacking_best = GridSearchCV(stacking_clf, parameters_stacking)
stacking_best.fit(training_features, training_labels)
stacking_finetune = stacking_best.best_estimator_

stacking_finetune.fit(training_features, training_labels)
stacking_finetune_train_score = stacking_finetune.score(training_features, training_labels)
print("Training Score for Finetuned Model:", stacking_finetune_train_score)

stacking_finetune_validation_score = stacking_finetune.score(validation_features, validation_labels)
print("Validation Score for Finetuned Model:", stacking_finetune_validation_score)
# Multilayer Stacking Classifier attempt
level1_train_set = []
level2_train_set = []

sss2 = StratifiedShuffleSplit(n_splits = 1, test_size = 0.5, random_state=0)

for index1, index2 in sss2.split(training_set, training_set["Survived"]):
    level1_train_set = train_data.loc[index1]
    level2_train_set = train_data.loc[index2]

level1_training_features = level1_train_set.drop("Survived", 1).to_numpy()
level1_training_labels = level1_train_set.drop(["Age", "Pclass", "SibSp", "Parch", "Fare", "Sex"], 1).to_numpy().ravel()

level2_training_features = level2_train_set.drop("Survived", 1).to_numpy()
level2_training_labels = level2_train_set.drop(["Age", "Pclass", "SibSp", "Parch", "Fare", "Sex"], 1).to_numpy().ravel()

level1_estimator1 = StackingClassifier(estimators_stacking, LogisticRegression(random_state = 1))
level1_estimator2 = StackingClassifier(estimators_stacking, AdaBoostClassifier(random_state = 1))
level1_estimator3 = StackingClassifier(estimators_stacking, RandomForestClassifier(random_state = 1))
level1_estimator4 = StackingClassifier(estimators_stacking, DecisionTreeClassifier(random_state = 1))

level1_estimator1.fit(level1_training_features, level1_training_labels)
level1_estimator2.fit(level1_training_features, level1_training_labels)
level1_estimator3.fit(level1_training_features, level1_training_labels)
level1_estimator4.fit(level1_training_features, level1_training_labels)

level2_estimator = StackingClassifier([('l1',level1_estimator1), ('l2', level1_estimator2), ('l3', level1_estimator3), ('l4', level1_estimator4)], LogisticRegression(random_state = 1))
level2_estimator.fit(level2_training_features, level2_training_labels)
level2_estimator_train_score = level2_estimator.score(level2_training_features, level2_training_labels)
print("Training Score for Finetuned Model:", level2_estimator_train_score)

level2_estimator_validation_score = level2_estimator.score(validation_features, validation_labels)
print("Validation Score for Finetuned Model:", level2_estimator_validation_score)
import tensorflow as tf
from tensorflow import keras
basic_dnn_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [6]),
    keras.layers.Dense(300, activation = keras.activations.relu
, kernel_initializer = 'he_normal'),
    keras.layers.Dense(200, activation = keras.activations.relu
, kernel_initializer = 'he_normal'),
    keras.layers.Dense(100, activation = keras.activations.relu
, kernel_initializer = 'he_normal'),
    keras.layers.Dense(50, activation = keras.activations.relu
, kernel_initializer = 'he_normal'),
    keras.layers.Dense(1, activation = 'sigmoid')
])
basic_dnn_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
basic_dnn_model.fit(training_features, training_labels, epochs = 100, verbose = 0, batch_size = 32)

basic_dnn_model_training_score = basic_dnn_model.evaluate(training_features, training_labels)
print("Basic Deep Neural Network Training Score :", basic_dnn_model_training_score[1])

basic_dnn_model_validation_score = basic_dnn_model.evaluate(validation_features, validation_labels)
print("Basic Deep Neural Network Validation Score :", basic_dnn_model_validation_score[1])
tf.random.set_seed(2)
def build_model1(activation_function = keras.activations.relu, n_hidden = 1, n_neurons = 100, learning_rate = 1e-3, input_shape = [6]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape = input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation = activation_function))
    model.add(keras.layers.Dense(1, activation = tf.keras.activations.sigmoid))
    optimizer = keras.optimizers.SGD(learning_rate = learning_rate)
    model.compile(loss="binary_crossentropy", optimizer = optimizer, metrics = ['accuracy'])
    return model

keras_classifier1 = keras.wrappers.scikit_learn.KerasClassifier(build_model1, verbose = 0)
keras_classifier1.fit(training_features, training_labels, epochs = 100, callbacks=[keras.callbacks.EarlyStopping(patience=10)], verbose = 0)
kerass_model1_training_score = keras_classifier1.score(training_features, training_labels)
print("Training Score for the First Keras Classifier :", kerass_model1_training_score)
keras_model1_validation_score = keras_classifier1.score(validation_features, validation_labels)
print("Validation Score for the First Keras Classifier :", keras_model1_validation_score)

activation_function_keras_classifier1 = [keras.activations.relu, keras.activations.selu, keras.activations.elu]
n_hidden_keras_classifier1 = [1, 2, 3, 5, 7]
n_neurons_keras_classifier1 = [10, 20, 40, 60, 80, 120, 160]
learning_rate_keras_classifier1 = [1e-4, 1e-3, 1e-2, 2e-4, 2e-3, 2e-2, 3e-4, 3e-3, 3e-2]
parameters_keras_classifier1 = {'activation_function' : activation_function_keras_classifier1, 'n_hidden' : n_hidden_keras_classifier1, 'n_neurons' : n_neurons_keras_classifier1,'learning_rate' : learning_rate_keras_classifier1}

rnd_search_keras_classifier1 = RandomizedSearchCV(keras_classifier1, parameters_keras_classifier1, n_iter = 10, cv = 3, verbose = 0)
rnd_search_keras_classifier1.fit(training_features, training_labels, epochs = 100, callbacks=[keras.callbacks.EarlyStopping(patience=10)], verbose = 0)
print("Best Parameters for the First Keras Classifier", rnd_search_keras_classifier1.best_params_)
print("Best Cross Validation Score for the First Keras Classifier :", rnd_search_keras_classifier1.best_score_)

rnd_search_keras_classifier1_best = rnd_search_keras_classifier1.best_estimator_
rnd_search_keras_classifier1_best.fit(training_features, training_labels)
rnd_search_keras_classifier1_best_validation_score = rnd_search_keras_classifier1_best.score(validation_features, validation_labels)
print("Best Parameter First Keras Classifier Validation Score :", rnd_search_keras_classifier1_best_validation_score)
# Furthering our attempt using a Neural Network with Batch Normalization
def build_model2(input_shape = 6, activation_function = keras.activations.elu,  optimization_function = keras.optimizers.Adam(), weight_initializer = 'he_normal'):
    def input_less_build_model2():
        keras_model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape = [input_shape]),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(400, activation = activation_function, kernel_initializer = weight_initializer),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(200, activation = activation_function, kernel_initializer = weight_initializer),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(100, activation = activation_function, kernel_initializer = weight_initializer),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(50, activation = activation_function, kernel_initializer = weight_initializer),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(1, activation = keras.activations.sigmoid)
        ])
        keras_model.compile(loss = 'binary_crossentropy', optimizer = optimization_function, metrics = ['accuracy'])
        return keras_model
    return input_less_build_model2
# Experimenting with varying optimizers, activation functions, weight initializers, batch size, and learning schedules
tf.random.set_seed(2)
keras_model1 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(),verbose = 0)

keras_model1_cross_val_score = cross_val_score(keras_model1, training_features, training_labels, verbose = 0)
print("Base Keras Model Cross Validation Score :", keras_model1_cross_val_score.mean())

keras_model1.fit(training_features, training_labels, batch_size = 32, epochs = 100, verbose = 0)

keras_model1_training_score = keras_model1.score(training_features, training_labels)
print("Base Keras Model Training Score :", keras_model1_training_score)

keras_model1_validation_score = keras_model1.score(validation_features, validation_labels)
print("Base Keras Model Validation Score :", keras_model1_validation_score)
keras_model1.model.save("keras_model1")
# Working with SELU activation and LeCun Normal Initialization
tf.random.set_seed(2)
keras_model2 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(activation_function = keras.activations.selu, weight_initializer = keras.initializers.lecun_normal()), verbose = 0)

keras_model2_cross_val_score = cross_val_score(keras_model2, training_features, training_labels, verbose = 0)
print("Keras Model with SELU activation and LeCun Normal Initialization Cross Validation Score :", keras_model2_cross_val_score.mean())

keras_model2.fit(training_features, training_labels, batch_size = 32, epochs = 100, verbose = 0)

keras_model2_training_score = keras_model2.score(training_features, training_labels)
print("Keras Model with SELU activation and LeCun Normal Initialization Training Score :", keras_model2_training_score)

keras_model2_validation_score = keras_model2.score(validation_features, validation_labels)
print("Keras Model with SELU activation and LeCun Normal Initialization Validation Score :", keras_model2_validation_score)
keras_model2.model.save("keras_model2")
tf.random.set_seed(2)
momentum_optimizer = keras.optimizers.SGD(lr = 0.001, momentum = 0.9)
keras_model3 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(optimization_function = momentum_optimizer),verbose = 0)

keras_model3_cross_val_score = cross_val_score(keras_model3, training_features, training_labels, verbose = 0)
print("Keras Model with Momentum Optimization Cross Validation Score :", keras_model3_cross_val_score.mean())

keras_model3.fit(training_features, training_labels, batch_size = 32, epochs = 100, verbose = 0)

keras_model3_training_score = keras_model3.score(training_features, training_labels)
print("Keras Model with Momentum Optimization Training Score :", keras_model3_training_score)

keras_model3_validation_score = keras_model3.score(validation_features, validation_labels)
print("Keras Model with Momentum Optimization Validation Score :", keras_model3_validation_score)
keras_model3.model.save("keras_model3")
tf.random.set_seed(2)
nesterov_optimizer = keras.optimizers.SGD(lr = 0.001, momentum = 0.9, nesterov = True)
keras_model4 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(optimization_function = nesterov_optimizer), verbose = 0)

keras_model4_cross_val_score = cross_val_score(keras_model4, training_features, training_labels, verbose = 0)
print("Keras Model with Nesterov Accelerated Gradient Optimization Cross Validation Score :", keras_model4_cross_val_score.mean())

keras_model4.fit(training_features, training_labels, batch_size = 32, epochs = 100, verbose = 0)

keras_model4_training_score = keras_model4.score(training_features, training_labels)
print("Keras Model with Nesterov Accelerated Gradient Optimization Training Score :", keras_model4_training_score)

keras_model4_validation_score = keras_model4.score(validation_features, validation_labels)
print("Keras Model with Nesterov Accelerated Gradient Optimization Validation Score :", keras_model4_validation_score)
keras_model4.model.save("keras_model4")
tf.random.set_seed(2)
keras_model5 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(optimization_function = keras.optimizers.Adagrad()), verbose = 0)

keras_model5_cross_val_score = cross_val_score(keras_model5, training_features, training_labels, verbose = 0)
print("Keras Model with AdaGrad Optimization Cross Validation Score :", keras_model5_cross_val_score.mean())

keras_model5.fit(training_features, training_labels, batch_size = 32, epochs = 100, verbose = 0)

keras_model5_training_score = keras_model5.score(training_features, training_labels)
print("Keras Model with AdaGrad Optimization Training Score :", keras_model5_training_score)

keras_model5_validation_score = keras_model5.score(validation_features, validation_labels)
print("Keras Model with AdaGrad Optimization Validation Score :", keras_model5_validation_score)
keras_model5.model.save("keras_model5")
tf.random.set_seed(2)
rmsprop_optimizer = keras.optimizers.RMSprop(lr = 0.001, rho = 0.9)
keras_model6 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(optimization_function = rmsprop_optimizer), verbose = 0)

keras_model6_cross_val_score = cross_val_score(keras_model6, training_features, training_labels, verbose = 0)
print("Keras Model with RMSProp Optimization Cross Validation Score :", keras_model6_cross_val_score.mean())

keras_model6.fit(training_features, training_labels, batch_size = 32, epochs = 100, verbose = 0)

keras_model6_training_score = keras_model6.score(training_features, training_labels)
print("Keras Model with RMSProp Optimization Training Score :", keras_model6_training_score)

keras_model6_validation_score = keras_model6.score(validation_features, validation_labels)
print("Keras Model with RMSProp Optimization Validation Score :", keras_model6_validation_score)
keras_model6.model.save("keras_model6")
tf.random.set_seed(2)
adam_nadam_optimizer = keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
keras_model7 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(optimization_function = adam_nadam_optimizer), verbose = 0)

keras_model7_cross_val_score = cross_val_score(keras_model7, training_features, training_labels, verbose = 0)
print("Keras Model with Adam and Nadam Optimization Cross Validation Score :", keras_model7_cross_val_score.mean())

keras_model7.fit(training_features, training_labels, batch_size = 32, epochs = 100, verbose = 0)

keras_model7_training_score = keras_model7.score(training_features, training_labels)
print("Keras Model with Adam and Nadam Optimization Training Score :", keras_model7_training_score)

keras_model7_validation_score = keras_model7.score(validation_features, validation_labels)
print("Keras Model with Adam and Nadam Optimization Validation Score :", keras_model7_validation_score)
keras_model7.model.save("keras_model7")
# Custom Stacking for Deep Neural Networks
tf.random.set_seed(2)
class single_layer_stacking_dnn:
    def __init__(self, level0_estimators = None, final_estimator = RandomForestClassifier(random_state = 1)):
        self.level0_estimators = level0_estimators
        self.final_estimator = final_estimator
    
    def fit(self, train_X, train_y):
        level0_train_set = []
        level1_train_set = []

        sss3 = StratifiedShuffleSplit(n_splits = 1, test_size = 0.1, random_state = 2)

        for index1, index2 in sss3.split(training_set, training_set["Survived"]):
            level0_train_set = train_data.loc[index1]
            level1_train_set = train_data.loc[index2]
        level0_training_features = level0_train_set.drop("Survived", 1).to_numpy()
        level0_training_labels = level0_train_set.drop(["Age", "Pclass", "SibSp", "Parch", "Fare", "Sex"], 1).to_numpy().ravel()
        
        level1_training_features = level1_train_set.drop("Survived", 1).to_numpy()
        level1_training_labels = level1_train_set.drop(["Age", "Pclass", "SibSp", "Parch", "Fare", "Sex"], 1).to_numpy().ravel()
        
        for level0_estimator in self.level0_estimators:
            level0_estimator.fit(level0_training_features, level0_training_labels, batch_size = 32, epochs = 100, verbose = 0)
        
        level0_outputs = [level0_estimator.predict(level1_training_features) for level0_estimator in self.level0_estimators]
        level0_outputs_merge = np.concatenate(level0_outputs, axis = 1)
        
        self.final_estimator.fit(level0_outputs_merge, level1_training_labels, batch_size = 32, epochs = 100, verbose = 0)
        
    def predict(self, test_X):
        level0_outputs = [level0_estimator.predict(test_X) for level0_estimator in self.level0_estimators]
        level0_outputs_merge = np.concatenate(level0_outputs, axis = 1)
        
        final_predictions = self.final_estimator.predict(level0_outputs_merge).reshape(-1, 1)
        return final_predictions
    
    def score(self, validation_X, validation_y):
        level0_outputs = [level0_estimator.predict(validation_X) for level0_estimator in self.level0_estimators]
        level0_outputs_merge = np.concatenate(level0_outputs, axis = 1)
        return self.final_estimator.score(level0_outputs_merge, validation_y)
test_model1 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(),verbose = 0)
test_model2 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(activation_function = keras.activations.selu, weight_initializer = keras.initializers.lecun_normal()), verbose = 0)
test_model3 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(optimization_function = momentum_optimizer),verbose = 0)
test_model4 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(optimization_function = nesterov_optimizer), verbose = 0)
test_model5 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(optimization_function = keras.optimizers.Adagrad()), verbose = 0)
test_model6 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(optimization_function = rmsprop_optimizer), verbose = 0)
test_model7 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(optimization_function = adam_nadam_optimizer), verbose = 0)
# Using Custom Stacking with a Deep Neural Network as the final estimator
tf.random.set_seed(3)
final_estimator_model = keras.wrappers.scikit_learn.KerasClassifier(build_model2(input_shape = 5), verbose = 0)
stacking_dnn = single_layer_stacking_dnn([test_model3, test_model4, test_model5, test_model6, test_model7], final_estimator_model)
stacking_dnn.fit(training_features, training_labels)

stacking_dnn_training_score = stacking_dnn.score(training_features, training_labels)
print("Stacking Deep Neural Networks Training Score :", stacking_dnn_training_score)

stacking_dnn_validation_score = stacking_dnn.score(validation_features, validation_labels)
print("Stacking Deep Neural Networks Validation Score :", stacking_dnn_validation_score)
tf.random.set_seed(3)
final_test_model1 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(),verbose = 0)
final_test_model2 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(activation_function = keras.activations.selu, weight_initializer = keras.initializers.lecun_normal()), verbose = 0)
final_test_model3 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(optimization_function = momentum_optimizer),verbose = 0)
final_test_model4 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(optimization_function = nesterov_optimizer), verbose = 0)
final_test_model5 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(optimization_function = keras.optimizers.Adagrad()), verbose = 0)
final_test_model6 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(optimization_function = rmsprop_optimizer), verbose = 0)
final_test_model7 = keras.wrappers.scikit_learn.KerasClassifier(build_model2(optimization_function = adam_nadam_optimizer), verbose = 0)
final_stacking_dnn_estimator = keras.wrappers.scikit_learn.KerasClassifier(build_model2(input_shape = 5), verbose = 0)
final_stacking_dnn = single_layer_stacking_dnn([final_test_model3, final_test_model4, final_test_model5, final_test_model6, final_test_model7], final_stacking_dnn_estimator)

full_train_data = train_data
full_train_features = train_data.drop("Survived", 1).to_numpy()
full_train_labels = training_set.drop(["Age", "Pclass", "SibSp", "Parch", "Fare", "Sex"], 1).to_numpy().ravel()
final_stacking_dnn.fit(full_train_features, full_train_labels)

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
processed_test_data = data_process(test_data)
test_features = processed_test_data.to_numpy()
predictions = final_stacking_dnn.predict(test_features)
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions.ravel()
    })
submission.to_csv('titanic_submission1.csv')

