import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
cancer = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

cancer.head()
cancer.shape
cancer.info()
cancer['Unnamed: 32'].isnull().sum() #delete this column as all null values
#drop id and Unnamed: 32 columns

cancer.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

cancer.head()
cancer.dtypes #will need to encode diagnosis column values
cancer['diagnosis'].value_counts()
sns.countplot(cancer['diagnosis'], label = 'count of diagnoses')
#encode diagnosis column values

from sklearn.preprocessing import LabelEncoder

labelencoder_Y = LabelEncoder()

cancer.iloc[:,0] = labelencoder_Y.fit_transform(cancer.iloc[:,0].values)

cancer.head()
#lets look at correlation

sns.pairplot(cancer, hue='diagnosis')
cancer.corr()
plt.figure(figsize=(20,20)) #make heatmap biger

sns.heatmap(cancer.corr(), annot=True, fmt='.0%')
#split data into X (features) and Y (labels)

X = cancer.drop(['diagnosis'], axis=1)

Y = cancer['diagnosis']
#train test split

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape
#Feature scaling

#from sklearn.preprocessing import StandardScaler

#sc = StandardScaler()

#X_train = sc.fit_transform(X_train)

#X_test = sc.fit_transform(X_test)
from sklearn.model_selection import GridSearchCV

import joblib

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

warnings.filterwarnings('ignore', category=DeprecationWarning)
def print_results(results):

    print('BEST PARAMS: {}\n'.format(results.best_params_))



    means = results.cv_results_['mean_test_score']

    stds = results.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, results.cv_results_['params']):

        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 0) 

parameters = {

    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]

}



cv = GridSearchCV(lr, parameters, cv=5)

cv.fit(X_train, Y_train)



print_results(cv)
cv.best_estimator_
joblib.dump(cv.best_estimator_, '../../../LR_model.pkl')
from sklearn.svm import SVC

svc = SVC()

parameters = {

    'kernel': ['linear', 'rbf'],

    'C': [0.1, 1, 10]

}



cv = GridSearchCV(svc, parameters, cv=5)

cv.fit(X_train, Y_train)



print_results(cv)
cv.best_estimator_
joblib.dump(cv.best_estimator_, '../../../SVM_model.pkl')
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state = 0)

parameters = {

    'n_estimators': [5, 50, 250],

    'max_depth': [2, 4, 8, 16, 32, None]

}



cv = GridSearchCV(rf, parameters, cv=5)

cv.fit(X_train, Y_train)



print_results(cv)
joblib.dump(cv.best_estimator_, '../../../RF_model.pkl')
from sklearn.neural_network import MLPRegressor, MLPClassifier



mlp = MLPClassifier()

parameters = {

    'hidden_layer_sizes': [(10,), (50,), (100,)],

    'activation': ['relu', 'tanh', 'logistic'],

    'learning_rate': ['constant', 'invscaling', 'adaptive']

}



cv = GridSearchCV(mlp, parameters, cv=5)

cv.fit(X_train, Y_train)



print_results(cv)
cv.best_estimator_
joblib.dump(cv.best_estimator_, '../../../MLP_model.pkl')
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor



gb = GradientBoostingClassifier()

parameters = {

    'n_estimators': [5, 50, 250, 500],

    'max_depth': [1, 3, 5, 7, 9],

    'learning_rate': [0.01, 0.1, 1, 10, 100]

}



cv = GridSearchCV(gb, parameters, cv=5)

cv.fit(X_train, Y_train)



print_results(cv)
cv.best_estimator_
joblib.dump(cv.best_estimator_, '../../../GB_model.pkl')
#from sklearn.naive_bayes import GaussianNB

#gnb = GaussianNB()

#gnb.fit(X_train, Y_train)



#from sklearn.tree import DecisionTreeClassifier

#dtc = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

#dtc.fit(X_train, Y_train)



#from sklearn.neighbors import KNeighborsClassifier

#knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

#knn.fit(X_train, Y_train)
models = {}



for mdl in ['LR', 'SVM', 'MLP', 'RF', 'GB']:

    models[mdl] = joblib.load('../../../{}_model.pkl'.format(mdl))
from sklearn.metrics import accuracy_score, precision_score, recall_score

from time import time



def evaluate_model(name, model, features, labels):

    start = time()

    end = time()

    pred = model.predict(features)

    accuracy = round(accuracy_score(labels, pred), 3)

    precision = round(precision_score(labels, pred), 3)

    recall = round(recall_score(labels, pred), 3)

    print('{} -- Accuracy: {} / Precision: {} / Recall: {} / Latency: {}ms'.format(name,

                                                                                   accuracy,

                                                                                   precision,

                                                                                   recall,

                                                                                   round((end - start)*1000, 1)))
for name, mdl in models.items():

    evaluate_model(name, mdl, X_train, Y_train)
evaluate_model('svc', models['SVM'], X_test, Y_test)
#check whether RF and GB are overfitting

evaluate_model('rf', models['RF'], X_test, Y_test)
evaluate_model('gb', models['GB'], X_test, Y_test)