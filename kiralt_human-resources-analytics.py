# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import time

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import sklearn.metrics as metrics

from sklearn.covariance import EmpiricalCovariance

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/HR_comma_sep.csv')

df.dropna

df.head(2)
def a(d):

    if d=='low':

        return 1

    elif d=='medium':

        return 2

    elif d=='high':

        return 3

    

def b(c):

    if c=='sales':

        return 1

    elif c=='accounting':

        return 2

    elif c=='hr':

        return 3

    elif c=='technical':

        return 4

    elif c=='support':

        return 5

    elif c=='management':

        return 6

    elif c=='IT':

        return 7

    elif c=='product_mng':

        return 8

    elif c=='marketing':

        return 9

    elif c=='RandD':

        return 10

    

df['salary']=df['salary'].apply(a)

df['sales']=df['sales'].apply(b)
train, test = train_test_split(df, test_size=0.5, random_state=23)

features_train = train.drop('left',axis=1).values

labels_train = train['left'].values

features_test = test.drop('left',axis=1).values

labels_test = test['left'].values

class_labels = ["Left", "Didn't Leave"]
def show_confusion_matrix(cnf_matrix, class_labels):

    plt.matshow(cnf_matrix,cmap=plt.cm.YlGn,alpha=0.7)

    ax = plt.gca()

    ax.set_xlabel('Predicted Label', fontsize=16)

    ax.set_xticks(range(0,len(class_labels)))

    ax.set_xticklabels(class_labels,rotation=45)

    ax.set_ylabel('Actual Label', fontsize=16, rotation=90)

    ax.set_yticks(range(0,len(class_labels)))

    ax.set_yticklabels(class_labels)

    ax.xaxis.set_label_position('top')

    ax.xaxis.tick_top()



    for row in range(len(cnf_matrix)):

        for col in range(len(cnf_matrix[row])):

            ax.text(col, row, cnf_matrix[row][col], va='center', ha='center', fontsize=16)
rfmodel = RandomForestClassifier(random_state=32)

start = time.clock()

rfmodel.fit(features_train,labels_train)

stop = time.clock()



y_pred_rf = rfmodel.predict(features_test)

cnf_matrix_rf = confusion_matrix(labels_test, y_pred_rf)

acc_score = metrics.accuracy_score(labels_test, y_pred_rf)



print("Accuracy Score: {}".format(acc_score))

print("Time: {} seconds".format(stop-start))

show_confusion_matrix(cnf_matrix_rf, class_labels)
rfmodel.get_params()