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
# Ignore warning



import warnings

warnings.filterwarnings('ignore')
FILEPATH = '/kaggle/input/creditcardfraud/creditcard.csv'
df = pd.read_csv(FILEPATH)
df.info()
df.head()
df.isnull().sum()
import missingno as miss



miss.matrix(df)
import seaborn as sns



sns.heatmap(df.corr())
X = df.iloc[:, :-1]

y = df.iloc[:, -1]
# Show X Columns



X.columns
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 56, stratify=y)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
import matplotlib.pyplot as plt



def show_confusion_matrix(_model_cm, title = None):

    

    f, ax = plt.subplots(figsize = (5, 5))

    

    sns.heatmap(_model_cm, annot = True, linewidth = 0, linecolor = 'red', fmt = 'g', ax = ax, cmap = 'Greens')

    

    # cmap colors:

    # YlGnBu, Blues, BuPu, Greens

    

    plt.title(title + ' Confusion Matrix')

    plt.xlabel('y Predict')

    plt.ylabel('y test')

    

    plt.show()
def predict_with_model(model):

    

    model = model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    

    return y_pred, accuracy
def show_metrics(model_cm):



    total = sum(sum(model_cm))

    

    accuracy = (model_cm[0, 0] + model_cm[1, 1]) / total

    accuracy = float("{:.2f}".format(accuracy))



    sensitivity = model_cm[0, 0] / (model_cm[0, 0] + model_cm[0, 1])

    sensitivity = float("{:.2f}".format(sensitivity))



    specificity = model_cm[1, 1]/(model_cm[1, 0] + model_cm[1, 1])

    specificity = float("{:.2f}".format(specificity))

    

    print(f'accuracy : {accuracy}, sensitivity : {sensitivity}, specificity : {specificity}')
best_model_accuracy = 0

best_model = None



models = [

#     MLPClassifier(),

    RandomForestClassifier(),

    KNeighborsClassifier(),

    LogisticRegression(solver = "liblinear"),

    DecisionTreeClassifier(),

    GaussianNB()

]



results = pd.DataFrame(columns = ['Accuracy'])



for model in models:

    

    model_name = model.__class__.__name__



    y_pred, accuracy = predict_with_model(model)

    

    print("-" * 30)

    print(model_name + ": " )

    

    current_model_cm = confusion_matrix(y_test, y_pred)

    show_metrics(current_model_cm)

    

    results.loc[model_name] = accuracy

    

    if(accuracy > best_model_accuracy):

        best_model_accuracy = accuracy

        best_model = model_name

    

    print("Accuracy: {:.2%}".format(accuracy))

    

    show_confusion_matrix(current_model_cm, model_name)
print("Best Model : {}".format(best_model))

print("Best Model Accuracy : {:.2%}".format(best_model_accuracy))
results
error_rate_list = []

range_min = 1

range_max = 10



best_k_value = 0

best_k_error_rate = 100



for i in range(range_min, range_max):

    

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(X_train, y_train)

    

    pred_i = knn.predict(X_test)

    current_error_rate = np.mean(pred_i != y_test)

    error_rate_list.append(current_error_rate)

    

    if(best_k_error_rate > current_error_rate):

        best_k_error_rate = current_error_rate

        best_k_value = i
current_error_rate, best_k_value
plt.figure(figsize = (16, 6))

plt.plot(range(range_min,range_max), error_rate_list, color='green', linestyle = 'dotted', marker = 'o',

         markerfacecolor = 'green', markersize = 10)

plt.title('Error Rate vs K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors = i)

y_pred, accuracy = predict_with_model(knn)



current_model_cm = confusion_matrix(y_test, y_pred)

show_metrics(current_model_cm)



results.loc[model_name] = accuracy
results
!pip install python-vivid
from vivid.utils import timer, get_logger

logger = get_logger(__name__)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
def calculate_score(y_true, y_pred):

    

    _functions = [

        accuracy_score,

        f1_score,

        precision_score,

        recall_score

    ]

    

    score_map = {}

    

    for func in _functions:

        score_map[func.__name__] = func(y_true, y_pred)

        

    return score_map

with timer(logger, prefix = '\ntime taken for svc : '):

    

    svc_model = SVC()

    

    svc_model.fit(X_train, y_train)

    y_pred = svc_model.predict(X_test)

    

    score = calculate_score(y_test, y_pred)

    

    logger.info(score)