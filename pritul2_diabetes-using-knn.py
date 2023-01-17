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
os.mkdir("mydata")
import shutil  

shutil.copytree("../input/pima-indians-diabetes-database","mydatas" )
#IMPORTS#

import pandas as pd

import numpy as np

import sklearn.model_selection

import math

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score
dataframe = pd.read_csv("mydatas/diabetes.csv")

dataframe.head()
#Data cleaning#

dataframe['Glucose'] = dataframe['Glucose'].replace(0,np.NAN) #replacing 0 with NAN.#

dataframe['BloodPressure'] = dataframe['BloodPressure'].replace(0,np.NAN) #replacing 0 with NAN. #

dataframe['SkinThickness'] = dataframe['SkinThickness'].replace(0,np.NAN) #replacing 0 with NAN. #

dataframe['BMI'] = dataframe['BMI'].replace(0,np.NAN) #replacing 0 with NAN. #

dataframe['Insulin'] = dataframe['BMI'].replace(0,np.NAN) #replacing 0 with NAN. #
# Splitting the dataset into training dataset and testing dataset#

#Determining input x#

#.iloc is used to extract data#

x = dataframe.iloc[:,0:8] #Taking first 7 column as x#

#Corresponding output y#

y = dataframe.iloc[:,8] #Last column is our output#



#Splitting into training and testing#

train_x,test_x,train_y,test_y = sklearn.model_selection.train_test_split(x,y,random_state=0,test_size=0.2) #0.2 indicating we are giving 20% testing dataset#
# Creating KNN classifier#

# We take the value of K = sqrt(len(y))

k = int(math.sqrt(len(test_y)))





# k value should never be even so,#

if k%2==0:

    k=k-1

    

print("value of k is ",k)



classifier = KNeighborsClassifier(k,p=2,metric="euclidean") #p=2 indicating euclidian distance refer docs#
'''

#Standardizing the dataset and dropping the NaN rows#

sc_x = StandardScaler()

train_x = sc_x.fit_transform(train_x)

test_x = sc_x.transform(test_x)

'''

my_imputer = SimpleImputer()

train_x = my_imputer.fit_transform(train_x)

test_x = my_imputer.transform(test_x)

# There is no need for y#



# Training the model#

classifier.fit(train_x,train_y)
y_pred = classifier.predict(test_x)
accuracy_score(test_y,y_pred)