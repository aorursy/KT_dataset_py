import pandas as pd

import numpy as numpy

from sklearn.neural_network import MLPClassifier
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
data.info()
#Get Target data 

y = data['Outcome']



#Load X Variables into a Pandas Dataframe with columns 

X = data.drop(['Outcome'], axis = 1)
X.head()
#Check size of data

X.shape
X.isnull().sum()

#We do not have any missing values
nnModel = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter = 1000)
nnModel.fit(X,y)
print (f'Accuracy - : {nnModel.score(X,y):.3f}')