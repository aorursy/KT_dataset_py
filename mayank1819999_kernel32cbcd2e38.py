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
df=pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")



#print(y)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()

ds=df.apply(le.fit_transform)

data = ds.values



y = data[:,0]

x = data[:,1:]



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

xtr, xts, ytr,yts = train_test_split(x, y, test_size = 0.2, random_state = 0)

#print(y)



ds.head()



def prior_prob(y_train,label):

    

    total_examples = y_train.shape[0]#total no. of classes

    class_examples = np.sum(y_train==label)#total no. of rows of that class

    

    return (class_examples)/float(total_examples)
def predict(x_train,y_train,xtest):

    """Xtest is a single testing point, n features"""

    

    classes = np.unique(y_train)

    n_features = x_train.shape[1]

    post_probs = [] # List of prob for all classes and given a single testing point

    #Compute Posterior for each class

    for label in classes:

        

        #Post_c = likelihood*prior

        likelihood = 1.0

        for f in range(n_features):

            cond = cond_prob(x_train,y_train,f,xtest[f],label)

            likelihood *= cond 

            

        prior = prior_prob(y_train,label)

        post = likelihood*prior

        post_probs.append(post)

        

    pred = np.argmax(post_probs)

    return pred
def cond_prob(x_train,y_train,feature_col,feature_val,label):

    

    x_filtered = x_train[y_train==label]

    numerator = np.sum(x_filtered[:,feature_col]==feature_val)

    denominator = np.sum(y_train==label)

    

    return numerator/float(denominator)
def score(x_train,y_train,x_test,y_test):



    pred = []

    for i in range(x_test.shape[0]):

        pred_label = predict(x_train,y_train,x_test[i])

        pred.append(pred_label) # <===Correction

    

    pred = np.array(pred)

    

    accuracy = np.sum(pred==y_test)/y_test.shape[0]

    return accuracy

print(score(xtr,ytr,xts,yts))