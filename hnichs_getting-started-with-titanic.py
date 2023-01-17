# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
from sklearn import datasets, metrics, preprocessing, model_selection

from sklearn.preprocessing import MinMaxScaler, LabelBinarizer,StandardScaler, LabelEncoder, OneHotEncoder 

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV

from sklearn.metrics import classification_report,confusion_matrix, mean_squared_error, matthews_corrcoef

from sklearn.datasets import make_classification, load_digits

from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression, LassoCV, LassoLarsCV, LassoLarsIC

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.impute import SimpleImputer



from xgboost import XGBClassifier





import dask_ml.model_selection as dcv



import matplotlib

import matplotlib.pyplot as plt



import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from keras.constraints import maxnorm

from keras.optimizers import SGD

from keras.utils import np_utils

from keras import regularizers

from keras.regularizers import l1

reg = l1(0.001)

seed = 7 #enables consistency in entropy for all models





#Setting gridsearch parameters for each model



logreg = LogisticRegression(penalty='l1', solver='liblinear',multi_class='auto',random_state=seed)

LR_par= {'penalty':['l1'], 'C': [0.5, 1, 5, 10], 'max_iter':[500, 1000, 5000]}



rfc =RandomForestClassifier(n_estimators=100,random_state=seed)

param_grid = {"max_depth": [5],

             "max_features": ["auto", "sqrt"],

              "min_samples_split": [50],

              "min_samples_leaf": [50],

              "bootstrap": [False, True],

              "criterion": ["entropy","gini"]}



gbm = GradientBoostingClassifier(random_state=seed)

param = {"loss":["deviance"],

    "learning_rate": [0.001, 0.01],

    "min_samples_split": [50],

    "min_samples_leaf": [50],

    "max_depth":[5],

    "max_features":["auto"],

    "criterion": ["friedman_mse"],

    "n_estimators":[1000]

    }





XGB = XGBClassifier(num_class=4)

xgb_parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'objective':['multi:softprob'],

              'learning_rate': [0.01, 0.001], #so called `eta` value

              'max_depth': [5],

              'min_child_weight': [11],

              'silent': [1],

              'subsample': [1.0],

              'colsample_bytree': [0.6],

              'n_estimators': [1000],

              'missing':[-999],

              'seed': [7]}

                   



mlp = MLPClassifier(random_state=seed)

parameter_space = {'hidden_layer_sizes': [(50,)],

     'activation': ['relu'],

     'solver': ['adam'],

     'max_iter': [10000],

     'alpha': [0.001, 0.01],

     'learning_rate': ['constant']}



svm = SVC(gamma="scale", probability=True,random_state=seed)

tuned_parameters = {'kernel':('linear', 'rbf'), 'C':(0.1, 0.25, 0.5, 0.75, 1.0)}







def baseline_model(learn_rate=0.01):

    model = Sequential()

    model.add(Dense(100, input_dim=X.shape[1], activation='relu', activity_regularizer=l1(0.0001)))

    model.add(Dropout(0.5))

    model.add(Dense(75, activation='relu',activity_regularizer=l1(0.0001))) 

    model.add(Dense(50, activation='relu',activity_regularizer=l1(0.0001))) 

    model.add(Dense(25, activation='relu',activity_regularizer=l1(0.0001))) 

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

    return model



keras = KerasClassifier(build_fn=baseline_model,batch_size=32, epochs=100, verbose=0)



learn_rate = [0.001, 0.01]

kerasparams = dict(learn_rate=learn_rate)



#creating folds of the training data:



inner_cv = KFold(n_splits=5, shuffle=True, random_state=seed)

outer_cv = KFold(n_splits=5, shuffle=True, random_state=seed)



models = []

models.append(('LR', dcv.GridSearchCV(logreg, LR_par, cv=inner_cv, iid=False, n_jobs=-1)))

models.append(('SVM', dcv.GridSearchCV(svm, tuned_parameters, cv=inner_cv, iid=False, n_jobs=-1)))

models.append(('RF', dcv.GridSearchCV(rfc, param_grid, cv=inner_cv,iid=False, n_jobs=-1)))

models.append(('GBM', dcv.GridSearchCV(gbm, param, cv=inner_cv,iid=False, n_jobs=-1)))

models.append(('XGB',dcv.GridSearchCV(XGB, xgb_parameters,  cv=inner_cv, iid=False, n_jobs=-1)))

models.append(('MLP', dcv.GridSearchCV(mlp, parameter_space, cv=inner_cv,iid=False, n_jobs=-1)))

models.append(('Keras', GridSearchCV(estimator=keras, param_grid=kerasparams, cv=inner_cv,iid=False, n_jobs=-1)))



results = []

names = []

scoring = 'accuracy'





y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

for name, model in models:

    nested_cv_results = model_selection.cross_val_score(model, X, y, cv=outer_cv, scoring=scoring)

    results.append(nested_cv_results)

    names.append(name)

    msg = "Nested CV Accuracy %s: %f (+/- %f )" % (name, nested_cv_results.mean()*100, nested_cv_results.std()*100)

    print(msg)

    





fig = plt.figure()

fig.suptitle('Nested Cross-Validation Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()



print(results)
SVM = dcv.GridSearchCV(svm, tuned_parameters,  cv=inner_cv, iid=False, n_jobs=-1)

SVM.fit(X,y)

predictions = SVM.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")