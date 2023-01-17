import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

 

    ###########  METRICS AND MODEL SELECTION 

from pandas.plotting import scatter_matrix  ###forscatter plots

from sklearn.linear_model import LogisticRegression

from sklearn import model_selection



from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
Irisdataset = pd.read_csv("../input/Iris.csv")

Irisdataset = Irisdataset.drop(['Id'], axis=1)

Irisdataset.head(10)
print("Shape of data is",Irisdataset.shape,"\n")



print(Irisdataset.groupby('Species').size())
print("Statistical Description SepalLength\n",Irisdataset['SepalLengthCm'].describe())

print("Statistical Description SepalWidth\n",Irisdataset['SepalWidthCm'].describe())

print("Statistical Description PetalLength\n",Irisdataset['PetalLengthCm'].describe())

print("Statistical Description PetalWidth\n",Irisdataset['PetalWidthCm'].describe())
Irisdataset.plot(kind = 'box',subplots = True , layout=(2,2),sharex=False,sharey=False)

plt.show()
Irisdataset.hist()

plt.show()
scatter_matrix(Irisdataset)

plt.show()
array =  Irisdataset.values    ### keep all the values into the data 

#Irisdataset.values

#array[:,0:4] # assign these values

#array[:,4]

X= array[:,0:4]

Y = array[:,4]

validation_size = 0.20

seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,random_state=seed)


models = []

models.append(('LR', LogisticRegression()))


# Test options and evaluation metric

seed = 7

scoring = 'accuracy'



results = []

names = []



# evaluate each model in turn

for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 

    print(msg)