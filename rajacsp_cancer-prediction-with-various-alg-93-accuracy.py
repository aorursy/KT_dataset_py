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
import missingno as miss
FILEPATH = '/kaggle/input/breast-cancer-wisconsin-data/data.csv'
df = pd.read_csv(FILEPATH)
df.describe()
df.info()
df.sample(3)
df.shape
# show 50-54 row

df[50:55]
df.isnull().any().any()
df.isnull().any()
df.isnull().any().any().sum()
miss.matrix(df)
miss.dendrogram(df)
miss.bar(df)
df = df.drop(columns = ['Unnamed: 32'])
diag_se = df['diagnosis'].value_counts()
diag_se
import seaborn as sns



sns.barplot(diag_se.index, diag_se.values)
sns.heatmap(df.corr(), square = False, mask = False)
from sklearn.model_selection import train_test_split
y = df['diagnosis']

X = df.drop(['diagnosis'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 23)
def show_split_data(X_train, X_test, y_train, y_test):

    

    print(f'X train shape : {X_train.shape}')

    print(f'Y train shape : {y_train.shape}')

    print(f'X test shape  : {X_train.shape}')

    print(f'Y test shape  : {y_train.shape}')
show_split_data(X_train, X_test, y_train, y_test)
# Confusion matrix

from sklearn.metrics import confusion_matrix
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
def get_metrics(model_cm):

    

    total = sum(sum(model_cm))

    

    accuracy = (model_cm[0, 0] + model_cm[1, 1]) / total

    accuracy = float("{:.2f}".format(accuracy))



    sensitivity = model_cm[0, 0] / (model_cm[0, 0] + model_cm[0, 1])

    sensitivity = float("{:.2f}".format(sensitivity))



    specificity = model_cm[1, 1]/(model_cm[1, 0] + model_cm[1, 1])

    specificity = float("{:.2f}".format(specificity))

    

    return accuracy, sensitivity, specificity
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB
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

    MLPClassifier(),

    RandomForestClassifier(),

    KNeighborsClassifier(),

    LogisticRegression(solver = "liblinear"),

    DecisionTreeClassifier(),

    GaussianNB()

]



for model in models:

    

    model_name = model.__class__.__name__



    y_pred, accuracy = predict_with_model(model)

    

    print("-" * 30)

    print(model_name + ": " )

    

    current_model_cm = confusion_matrix(y_test, y_pred)

    show_metrics(current_model_cm)

    

    if(accuracy > best_model_accuracy):

        best_model_accuracy = accuracy

        best_model = model_name

    

    print("Accuracy: {:.2%}".format(accuracy))

    

    show_confusion_matrix(current_model_cm, model_name)
print("Best Model : {}".format(best_model))

print("Best Model Accuracy : {:.2%}".format(best_model_accuracy))