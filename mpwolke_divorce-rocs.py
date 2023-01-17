# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/divorce-prediction/divorce_data.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'divorce_data.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')

df.head()
# checking dataset



print ("Rows     : " ,df.shape[0])

print ("Columns  : " ,df.shape[1])

print ("\nFeatures : \n" ,df.columns.tolist())

print ("\nMissing values :  ", df.isnull().sum().values.sum())

print ("\nUnique values :  \n",df.nunique())
corr = df.corr()

corr.style.background_gradient(cmap = 'coolwarm')
import statsmodels.formula.api as smf



from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.preprocessing import LabelEncoder, StandardScaler 



from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier



from warnings import filterwarnings

filterwarnings('ignore')
y = df["Divorce"]

X = df.drop(["Divorce"], axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
cart = DecisionTreeClassifier(max_depth = 12)
cart_model = cart.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
print('Decision Tree Model')



print('Accuracy Score: {}\n\nConfusion Matrix:\n {}\n\nAUC Score: {}'

      .format(accuracy_score(y_test,y_pred), confusion_matrix(y_test,y_pred), roc_auc_score(y_test,y_pred)))
pd.DataFrame(data = cart_model.feature_importances_*100,

                   columns = ["Importances"],

                   index = X_train.columns).sort_values("Importances", ascending = False)[:20].plot(kind = "barh", color = "r")



plt.xlabel("Feature Importances (%)")
# We can use the functions to apply the models and roc curves to save space.

def model(algorithm, X_train, X_test, y_train, y_test):

    alg = algorithm

    alg_model = alg.fit(X_train, y_train)

    global y_prob, y_pred

    y_prob = alg.predict_proba(X_test)[:,1]

    y_pred = alg_model.predict(X_test)



    print('Accuracy Score: {}\n\nConfusion Matrix:\n {}'

      .format(accuracy_score(y_test,y_pred), confusion_matrix(y_test,y_pred)))

    



def ROC(y_test, y_prob):

    

    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_prob)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    

    plt.figure(figsize = (10,10))

    plt.title('Receiver Operating Characteristic')

    plt.plot(false_positive_rate, true_positive_rate, color = 'red', label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1], linestyle = '--')

    plt.axis('tight')

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')
print('Model: Logistic Regression\n')

model(LogisticRegression(solver = "liblinear"), X_train, X_test, y_train, y_test)
LogR = LogisticRegression(solver = "liblinear")

cv_scores = cross_val_score(LogR, X, y, cv = 8, scoring = 'accuracy')

print('Mean Score of CV: ', cv_scores.mean())
ROC(y_test, y_prob)
print('Model: Gaussian Naive Bayes\n')

model(GaussianNB(), X_train, X_test, y_train, y_test)
NB = GaussianNB()

cv_scores = cross_val_score(NB, X, y, cv = 8, scoring = 'accuracy')

print('Mean Score of CV: ', cv_scores.mean())
ROC(y_test, y_prob)
#I excluded probability in the function for SVC, also I could not use other kernel methods because it takes really long and I don't think SVC as a good model for this dateset. 

print('Model: SVC\n')



def model1(algorithm, X_train, X_test, y_train, y_test):

    alg = algorithm

    alg_model = alg.fit(X_train, y_train)

    global y_pred

    y_pred = alg_model.predict(X_test)

    

    print('Accuracy Score: {}\n\nConfusion Matrix:\n {}'

      .format(accuracy_score(y_test,y_pred), confusion_matrix(y_test,y_pred)))

    

model1(SVC(kernel = 'linear'), X_train, X_test, y_train, y_test)
print('Model: Decision Tree\n')

model(DecisionTreeClassifier(max_depth = 12), X_train, X_test, y_train, y_test)
DTC = DecisionTreeClassifier(max_depth = 12)

cv_scores = cross_val_score(DTC, X, y, cv = 8, scoring = 'accuracy')

print('Mean Score of CV: ', cv_scores.mean())
ROC(y_test, y_prob)
print('Model: Random Forest\n')

model(RandomForestClassifier(), X_train, X_test, y_train, y_test)
RFC = RandomForestClassifier()

cv_scores = cross_val_score(RFC, X, y, cv = 8, scoring = 'accuracy')

print('Mean Score of CV: ', cv_scores.mean())
ROC(y_test, y_prob)
rf_parameters = {"max_depth": [10,13],

                 "n_estimators": [10,100,500],

                 "min_samples_split": [2,5]}
rf_model = RandomForestClassifier()
rf_cv_model = GridSearchCV(rf_model,

                           rf_parameters,

                           cv = 10,

                           n_jobs = -1,

                           verbose = 2)



rf_cv_model.fit(X_train, y_train)
print('Best parameters: ' + str(rf_cv_model.best_params_))
rf_tuned = RandomForestClassifier(max_depth = 13,

                                  min_samples_split = 2,

                                  n_estimators = 500)



print('Model: Random Forest Tuned\n')

model(rf_tuned, X_train, X_test, y_train, y_test)
print('Model: XGBoost\n')

model(XGBClassifier(), X_train, X_test, y_train, y_test)
XGB = XGBClassifier()

cv_scores = cross_val_score(XGB, X, y, cv = 8, scoring = 'accuracy')

print('Mean Score of CV: ', cv_scores.mean())
ROC(y_test, y_prob)
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
print('Model: Neural Network\n')

model(MLPClassifier(), X_train_scaled, X_test_scaled, y_train, y_test)
ROC(y_test, y_prob)
mlpc_parameters = {"alpha": [1, 0.1, 0.01, 0.001],

                   "hidden_layer_sizes": [(50,50,50),

                                          (100,100)],

                   "solver": ["adam", "sgd"],

                   "activation": ["logistic", "relu"]}
mlpc = MLPClassifier()

mlpc_cv_model = GridSearchCV(mlpc, mlpc_parameters,

                             cv = 10,

                             n_jobs = -1,

                             verbose = 2)



mlpc_cv_model.fit(X_train_scaled, y_train)
print('Best parameters: ' + str(mlpc_cv_model.best_params_))
mlpc_tuned = MLPClassifier(activation = 'relu',

                           alpha = 0.1,

                           hidden_layer_sizes = (100,100),

                           solver = 'adam')
print('Model: Neural Network Tuned\n')

model(mlpc_tuned, X_train_scaled, X_test_scaled, y_train, y_test)
ROC(y_test, y_prob)
randomf = RandomForestClassifier()

rf_model1 = randomf.fit(X_train, y_train)



pd.DataFrame(data = rf_model1.feature_importances_*100,

                   columns = ["Importances"],

                   index = X_train.columns).sort_values("Importances", ascending = False)[:15].plot(kind = "barh", color = "r")



plt.xlabel("Feature Importances (%)")
table = pd.DataFrame({"Model": ["Decision Tree (reservation status included)", "Logistic Regression",

                                "Naive Bayes", "Support Vector", "Decision Tree", "Random Forest",

                                "Random Forest Tuned", "XGBoost", "Neural Network", "Neural Network Tuned"],

                     "Accuracy Scores": ["0.88", "0.98", "0.98", "1.00", "0.846",

                                         "1.00", "0.851", "0.98", "0.98", "0.98"],

                     "ROC | Auc": ["0.88", "1.00", "0.98", "0.88",

                                   "0.92", "0.98", "0", "0.99",

                                   "1.00", "1.00"]})





table["Model"] = table["Model"].astype("category")

table["Accuracy Scores"] = table["Accuracy Scores"].astype("float32")

table["ROC | Auc"] = table["ROC | Auc"].astype("float32")



pd.pivot_table(table, index = ["Model"]).sort_values(by = 'Accuracy Scores', ascending=False)