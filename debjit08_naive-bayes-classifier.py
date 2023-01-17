# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from sklearn.naive_bayes import CategoricalNB

# Import LabelEncoder

from sklearn import preprocessing
# Imporitng the File

nbflu=pd.read_csv('/kaggle/input/naivebayes.csv')

# Checking the shape of the data frame

print(nbflu.shape)

# Printig top few rows

print(nbflu.head(5))
# Collecting the features and target individually

x1= nbflu.iloc[:,0]

x2= nbflu.iloc[:,1]

x3= nbflu.iloc[:,2]

x4= nbflu.iloc[:,3]

y=nbflu.iloc[:,4]

list(nbflu.index[:4])
#creating labelEncoder

le = preprocessing.LabelEncoder()

x1= le.fit_transform(x1)

x2= le.fit_transform(x2)

x3= le.fit_transform(x3)

x4= le.fit_transform(x4)

y=le.fit_transform(y)



X = pd.DataFrame(list(zip(x1,x2,x3,x4)))

X
#Create a Gaussian Classifier

model = CategoricalNB()



# Train the model using the training sets

model.fit(X,y)



#Predict Output

#['Y','N','Mild','Y']

predicted = model.predict([[1,0,0,1]]) 

print("Predicted Value:",model.predict([[1,0,0,1]]))

print(model.predict_proba([[1,0,0,1]]))
# Looking at the model parameters

print(model.get_params())

# Checking the likelyhood Table

print(model.category_count_[2])