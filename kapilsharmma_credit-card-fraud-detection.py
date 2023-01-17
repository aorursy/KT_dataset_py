#Importing the Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sys 

import scipy

import seaborn as sns

import sklearn

print('Python version',format(sys.version))

print('Numpy version',format(np.__version__))

print('pandas version',format(pd.__version__))

#print('matplotlib version',format(plt1.__version__))

print('scipy version',format(scipy.__version__))

print('seaborn version',format(sns.__version__))

print('sklearn version',format(sklearn.__version__))
#Importing the essential packeges

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 
#Importing the dataset

dataset=pd.read_csv("../input/creditcardfraud/creditcard.csv")

print(dataset.columns)
print(dataset.shape)
print(dataset.describe)
datset=dataset.sample(frac=0.001,random_state=1)

print(dataset.shape)
#plot a histogram of the dataset

dataset.hist(figsize=(20,20))

plt.xlabel("x-axis")

plt.show()
#Now we are going to detect no. of fraud transactions in our dataset

fraud=dataset[dataset['Class']==1]

valid=dataset[dataset['Class']==0]

outlier_fraction=len(fraud)/float(len(valid))

print(outlier_fraction)

print("fraud cases():",format(len(fraud)))

print("valid cases():",format(len(valid)))
#Building the Co-relation Matrix

coremat=dataset.corr()

figu=plt.figure(figsize=(12,9))



sns.heatmap(coremat,vmax=.8,square=True)

plt.show()
#Now we are getting all the column from the dataframe

columns=dataset.columns.tolist()



#Eliminating the data from the columns which id not required

columns=[c for c in columns if c not in["Class"]]



#storing the variable we are going to predict

target="Class"

X=dataset[columns]

Y=dataset[target]



#print the shape of target and column

print(X.shape)

print(Y.shape)

from sklearn.metrics import classification_report,accuracy_score

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor



#defining a random state

state=1



#define the outlier detection method

classifiers={

    "Isolation forest":IsolationForest(max_samples=len(X),contamination=outlier_fraction,random_state=state),

    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20,contamination=outlier_fraction)

}

#Fitting the model

n_outliers=len(fraud)

for i,(clf_name,clf) in enumerate(classifiers.items()):

    #fit the data and tag Outliers

    if clf_name=="Local Outlier Factor":

        y_pred=clf.fit_predict(X)

        scores_pred=clf.negative_outlier_factor_

    else:    

        clf.fit(X)

        scores_pred=clf.decision_function(X)

        y_pred=clf.predict(X)

    

  #reshape the predictin values to 0 for valid, 1 for fraud

y_pred[y_pred==1]=0

y_pred[y_pred==-1]=1

n_errors=(y_pred!=Y).sum()



#Run Classificatin Matrices

print('{}:{}',format(clf_name,str(n_errors)))

print(accuracy_score(Y,y_pred))

print(classification_report(Y,y_pred))