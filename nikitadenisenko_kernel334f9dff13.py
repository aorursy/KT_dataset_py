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
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")



df['log_amount'] = df['Amount'].apply(np.log1p)



y = df['Class']

X = df.drop(['Class', 'Time', 'Amount'], axis=1)
y.value_counts()/len(y)*100
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(X_train)



X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
X_train_scaled.shape
from sklearn.preprocessing import PolynomialFeatures



poly = PolynomialFeatures(2, include_bias=False)



X_train_poly = poly.fit_transform(X_train_scaled)

X_test_poly = poly.transform(X_test_scaled)
X_test_poly.shape
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report



classifier = SGDClassifier(loss='log', random_state=1)

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import plot_precision_recall_curve



from sklearn.metrics import auc
#Функция отображающая веса модели.



def round_coef(coef):

    print('Classifier coef: \n', list(map(lambda x : round(x, 2), coef[0])))

    

#Функция отображающая веса модели, метрики качества precision, recall, f1, AUC PR

#и Precision-recall curve.

def classifier_results(model, data, label):

       

    round_coef(model.coef_)

    

    print('\n Classification report: \n' + classification_report(label, model.predict(data)))

    

    precision, recall, _ = precision_recall_curve(label, model.predict_proba(data)[:,1])

    print('\n AUC precision-recall: ' + str(auc(recall, precision)))

    

    disp = plot_precision_recall_curve(model, data, label)

    disp.ax_.set_title('Precision Recall-Curve')

    
classifier.fit(X_train_scaled, y_train)



classifier_results(classifier, X_test_scaled, y_test)
classifier.fit(X_train_poly, y_train)



classifier_results(classifier, X_test_poly, y_test)
classifier_balanc = SGDClassifier(class_weight='balanced', loss='log', random_state=1)
classifier_balanc.fit(X_train_scaled, y_train)



classifier_results(classifier_balanc, X_test_scaled, y_test)
classifier_balanc.fit(X_train_poly, y_train)



classifier_results(classifier_balanc, X_test_poly, y_test)
from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler



rus = RandomUnderSampler()

ros = RandomOverSampler()
X_rus, y_rus = rus.fit_sample(X_train_scaled, y_train)

X_ros, y_ros = ros.fit_sample(X_train_poly, y_train)
X_rus.shape, X_ros.shape
classifier.fit(X_rus, y_rus)



classifier_results(classifier, X_test_scaled, y_test)
classifier.fit(X_ros, y_ros)



classifier_results(classifier, X_test_poly, y_test)
classifier_results(classifier, X_ros, y_ros)
from imblearn.under_sampling import ClusterCentroids



cc = ClusterCentroids()
X_train_cc, y_train_cc = cc.fit_sample(X_train_scaled, y_train)
X_train_cc.shape
classifier.fit(X_train_cc, y_train_cc)



classifier_results(classifier, X_test_scaled, y_test)
from sklearn.model_selection import GridSearchCV



parameters_grid = {

    'alpha': [0.001, 0.01, 1, 5, 10],

    'loss' : ['log'],

    'penalty' : ['l1', 'l2']}



grid_cv = GridSearchCV(classifier, parameters_grid, scoring = 'recall', cv = 3)
grid_cv.fit(X_train_cc, y_train_cc)
grid_cv.best_score_, grid_cv.best_params_
classifier_results(grid_cv.best_estimator_, X_test_scaled, y_test)
from imblearn.over_sampling import SMOTE



smote = SMOTE()
X_train_sm, y_train_sm = smote.fit_sample(X_train_scaled, y_train)
classifier.fit(X_train_sm, y_train_sm)



classifier_results(classifier, X_test_scaled, y_test)
grid_cv = GridSearchCV(classifier, parameters_grid, scoring = 'f1', cv = 5)
grid_cv.fit(X_train_sm, y_train_sm)
grid_cv.best_score_, grid_cv.best_params_
classifier_results(grid_cv.best_estimator_, X_test_scaled, y_test)
X_train_sm_poly = poly.transform(X_train_sm)

classifier.fit(X_train_sm_poly, y_train_sm)
print(classification_report(y_train_sm, classifier.predict(X_train_sm_poly)))

print(classification_report(y_test, classifier.predict(X_test_poly)))



round_coef(classifier.coef_)
plot_precision_recall_curve(classifier, X_test_poly, y_test)