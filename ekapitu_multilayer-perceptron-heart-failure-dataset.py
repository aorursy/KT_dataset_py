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
dataset = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
dataset.describe()
# Splitting predictors and target
x = dataset.drop(columns=["DEATH_EVENT"])
y = dataset["DEATH_EVENT"]
# Scaling input
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x)

x = scaler.fit_transform(x)

# Splitting into train/test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print("x_train Shape : ", x_train.shape)
print("x_test Shape  : ", x_test.shape)
print("y_train Shape : ", y_train.shape)
print("y_test Shape  : ", y_test.shape)
# Model building
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

nn_model = MLPClassifier(random_state=0)
nn_model.fit(x_train, y_train)

y_pred = nn_model.predict(x_test)

print("Accuracy score  : {:.4f}".format(accuracy_score(y_pred, y_test)))
print("Precision score : {:.4f}".format(precision_score(y_pred, y_test)))
print("Recall score    : {:.4f}".format(recall_score(y_pred, y_test)))
print("F1 score        : {:.4f}".format(f1_score(y_pred, y_test)))
print("AUC ROC score   : {:.4f}".format(roc_auc_score(y_pred, y_test)))
print("\n", classification_report(y_pred, y_test))
from sklearn.model_selection import GridSearchCV

activation_fn = ["identity", "relu", "logistic", "tanh"] # activation function
solver = ["lbfgs", "adam", "sgd"] # optimizer
alpha = [0.0001, 0.05] # Ridge regression's alpha
learning_rate = list(['constant','adaptive'])
hidden_layer_sizes = list([(50,50,50), (50,100,50), (100,)]) # different sizes of hidden layers


param_grid = dict(
    activation = activation_fn,
    solver = solver,
    alpha = alpha,
    learning_rate = learning_rate,
    hidden_layer_sizes = hidden_layer_sizes
)

mlp = MLPClassifier(max_iter=100)
clf = GridSearchCV(mlp, param_grid=param_grid, 
                   cv=10, 
                   scoring='roc_auc',
                   n_jobs=-1, verbose=2
                  )
clf.fit(x_train, y_train)
clf.best_params_
mlpc_final = MLPClassifier(activation = 'tanh',
                           alpha = 0.05,
                           hidden_layer_sizes = (50, 50, 50),
                           learning_rate = 'adaptive',
                           solver = 'adam',
                           random_state = 0
                          )
mlpc_final.fit(x_train, y_train)
y_pred = mlpc_final.predict(x_test)

print("Accuracy score  : {:.4f}".format(accuracy_score(y_pred, y_test)))
print("Precision score : {:.4f}".format(precision_score(y_pred, y_test)))
print("Recall score    : {:.4f}".format(recall_score(y_pred, y_test)))
print("F1 score        : {:.4f}".format(f1_score(y_pred, y_test)))
print("AUC ROC score   : {:.4f}".format(roc_auc_score(y_pred, y_test)))
print("\n", classification_report(y_pred, y_test))
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred, y_test)

sns.heatmap((cm/np.sum(cm) * 100),
            annot = True,
            fmt = ".2f",
            cmap = "Greens"
           )
from sklearn.utils import resample

all_accuracy_scores = []

for i in range(0, 101): # repeat bootstrap sampling 100 times
    x_boot = resample(dataset, replace=True)
    oob = dataset[~dataset.apply(tuple,1).isin(x_boot.apply(tuple,1))]
    
    mlpc_boot = MLPClassifier(activation = 'tanh',
                              alpha = 0.05,
                              hidden_layer_sizes = (50, 50, 50),
                              learning_rate = 'adaptive',
                              solver = 'adam',
                              random_state = 0
                             )
    mlpc_boot.fit(x_boot.drop(columns=["DEATH_EVENT"]), x_boot["DEATH_EVENT"])
    boot_pred = mlpc_boot.predict(oob.drop(columns=["DEATH_EVENT"]))
    
    all_accuracy_scores.append(accuracy_score(boot_pred, oob["DEATH_EVENT"]))

print("Mean accuracy score  : {:.4f}".format(np.mean(all_accuracy_scores)))
from imblearn.over_sampling import SMOTE

sms = SMOTE(random_state=0)

x_res, y_res = sms.fit_sample(x, y)
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.2, random_state=42)

print("x_train Shape : ", x_train.shape)
print("x_test Shape  : ", x_test.shape)
print("y_train Shape : ", y_train.shape)
print("y_test Shape  : ", y_test.shape)
mlpc_model = MLPClassifier(activation="relu",
                            alpha=0.05,
                            hidden_layer_sizes= (100,),
                            learning_rate= 'adaptive',
                            solver= 'adam',
                            random_state=42)
mlpc_model.fit(x_train, y_train)
y_pred = mlpc_model.predict(x_test)

print('Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
print('SVC f1-score  : {:.4f}'.format(f1_score(y_pred, y_test)))
print('SVC precision : {:.4f}'.format(precision_score(y_pred, y_test)))
print('SVC recall    : {:.4f}'.format(recall_score(y_pred, y_test)))
print("\n",classification_report(y_pred, y_test))
cm = confusion_matrix(y_pred, y_test)

sns.heatmap((cm/np.sum(cm) * 100),
            annot = True,
            fmt = ".2f",
            cmap = "Oranges"
           )