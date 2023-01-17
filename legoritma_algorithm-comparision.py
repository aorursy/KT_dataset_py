# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neural_network import MLPClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv("../input/data.csv",header=0)

data.drop("Unnamed: 32",axis=1,inplace=True)

data.drop("id",axis=1,inplace=True)



data.info()

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split 

train, test = train_test_split(data, test_size = 0.3)



prediction_var = list(data.columns[1:31])

outcome_var = "diagnosis"

print(prediction_var)



train_X = train[prediction_var]# taking the training data input 

train_y=train.diagnosis# This is output of our training data

# same we have to do for test

test_X= test[prediction_var] # taking test data inputs

test_y =test.diagnosis   #output value of test dat
from sklearn.model_selection  import KFold

from sklearn import metrics



def model_test(model):

    model.fit(train_X,train_y)

    predictions = model.predict(test_X)

    accuracy = metrics.accuracy_score(predictions,test_y)

    print("Test Accuracy : %s" % "{0:.3%}".format(accuracy))

    

def classification_model(model,data,prediction_input,output):

    model_test(model)

    model.fit(data[prediction_input],data[output])

    predictions = model.predict(data[prediction_input])

    accuracy = metrics.accuracy_score(predictions,data[output])

    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

    

    kf = KFold(n_splits=5)

    error = []

    for train, test in kf.split(data):

        train_X = (data[prediction_input].iloc[train,:])

        train_y = data[output].iloc[train]

        model.fit(train_X, train_y)

        test_X=data[prediction_input].iloc[test,:]

        test_y=data[output].iloc[test]

        error.append(model.score(test_X,test_y))

        print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
from sklearn import svm # for Support Vector Machine

from sklearn import metrics # for the check the error and accuracy of the model



model = svm.SVC(kernel='rbf')

classification_model(model,data,prediction_var,outcome_var)
model = svm.SVC(kernel='linear')

classification_model(model,data,prediction_var,outcome_var)
model = svm.SVC(kernel='sigmoid')

classification_model(model,data,prediction_var,outcome_var)


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

classification_model(model,data,prediction_var,outcome_var)
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

classification_model(model,data,prediction_var,outcome_var)
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score

y_labels = np.where(data[outcome_var]== "M", 1, 0)



kmean = KMeans(n_clusters=2, random_state=0)

kmean.fit(data[prediction_var],data[outcome_var])

labels = kmean.predict(data[prediction_var])

print(accuracy_score(labels, y_labels))



kmean = KMeans(n_clusters=3, random_state=0)

kmean.fit(data[prediction_var],data[outcome_var])

labels = kmean.predict(data[prediction_var])

print(accuracy_score(labels, y_labels))



kmean = KMeans(n_clusters=4, random_state=0)

kmean.fit(data[prediction_var],data[outcome_var])

labels = kmean.predict(data[prediction_var])

print(accuracy_score(labels, y_labels))