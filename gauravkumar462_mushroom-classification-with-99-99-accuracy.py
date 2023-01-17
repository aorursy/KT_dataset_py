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
import seaborn as sns

import matplotlib.pyplot as pt

data = pd.read_csv("/kaggle/input/mushrooms/mushrooms.csv")
data.head()
data.info()
data.isnull().sum()
print("Number of Unique values in each columns :- ")

for col in data.columns:

    print(col ,": ", len(data[col].unique()))
for col in data.columns:

    print(col, ": ", data[col].value_counts())
data.shape
encoded = pd.get_dummies(data, columns = data.columns[1:], drop_first = True)
encoded.head()
encoded['class'] = encoded['class'].map({'p':1, 'e':0})
sns.heatmap(encoded.corr())
def count_plot(col_name):

    color = np.array(['#808080',

    '#000000', 

    '#FF0000',

    '#800000', 

    '#FFFF00',

    '#808000', 

    '#00FF00', 

    '#008000',

    '#00FFFF', 

    '#008080',

    '#0000FF',

    '#000080', 

    '#FF00FF', 

    '#800080'])

    length_col_counts = len(data[col_name].value_counts())

    rand_index = np.random.randint(0, len(color), length_col_counts)

    col_counts = data[col_name].value_counts()

    x_label = col_counts.index

    pt.title(str(col_name)+" Size counts")

    pt.xlabel("Types of "+str(col_name))

    pt.ylabel("Counts")

    pt.bar(x_label, col_counts, color = color[rand_index])

    pt.show()

    return
for col in data.columns:

    count_plot(col)
def compare_plot(x, hue = 'class'):

    ax = sns.countplot(data = data, x = x, hue = hue)

    return ax

    

for col in data.columns[1:]:

    ax = compare_plot(col)

    pt.show(ax)
x = encoded.iloc[:, 1:].values

y = encoded.iloc[:, 0].values
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x, y, 

                                                     test_size = 0.33, 

                                                     random_state = 0)
def score(model, y_pred):

    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

    cm = confusion_matrix(y_valid, y_pred)

    ac = accuracy_score(y_valid, y_pred)

    prec = precision_score(y_valid, y_pred)

    rec = recall_score(y_valid, y_pred)

    f1 = f1_score(y_valid, y_pred)

    

    return cm, ac, prec, rec, f1
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(C = 1.0, 

                                solver = 'lbfgs', 

                                random_state=0, penalty = 'l2')

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_valid)
score(classifier, y_pred)
from sklearn.ensemble import RandomForestClassifier

classifier_forest = RandomForestClassifier(n_estimators = 100, 

                                          criterion = 'gini' , 

                                          n_jobs = -1, 

                                          random_state = 0)
classifier_forest.fit(x_train, y_train)
y_pred2 = classifier_forest.predict(x_valid)
score(classifier_forest, y_pred2)
import pickle

file_name = 'model_IDEAL.sav'

saved_model = pickle.dump(classifier_forest, open(file_name, 'wb'))
probs = classifier_forest.predict_proba(x_valid)

probs = probs[:, 1]

ns_probs = [0 for _ in range(len(y_valid))]



from sklearn.metrics import roc_curve

ns_fpr, ns_tpr, _ = roc_curve(y_valid, ns_probs)

lr_fpr, lr_tpr, _ = roc_curve(y_valid, probs)

pt.plot(ns_fpr, ns_tpr, linestyle = '--', c = 'blue', label = 'mid_mark')

pt.plot(lr_fpr, lr_tpr, c = 'orange', label = 'random_forest')

pt.xlabel("False Positive Rate")

pt.ylabel("True Positive Rate")

pt.title('Roc_Auc Curve')

pt.legend()

pt.show()
log_probs = classifier.predict_proba(x_valid)

log_probs = log_probs[:, 1]



ns_probs2 = [0 for _ in range(len(x_valid))]



ns_fpr_log, ns_tpr_log, _ = roc_curve(y_valid, ns_probs2)

lr_fpr_log, lr_tpr_log, _ = roc_curve(y_valid, log_probs)
pt.plot(ns_fpr_log, ns_tpr_log, linestyle = '--', c = 'violet', label = 'mid_mark')

pt.plot(lr_fpr_log, lr_tpr_log, c = 'indigo', label = 'logistic')

pt.xlabel("False Positive Rate")

pt.ylabel("True Positive Rate")

pt.title('Roc_Auc Curve for Logistic Regression')

pt.legend()

pt.show()