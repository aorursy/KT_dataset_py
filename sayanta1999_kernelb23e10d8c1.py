# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



# -*- coding: utf-8 -*-



import pandas as pd

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.ensemble import RandomForestClassifier



def read_data():

    df = pd.read_csv('../input/diabetes.csv')

    x = df.drop('Outcome',axis=1)

    y = df['Outcome']

    x,y = shuffle(x,y,random_state = 2)

    return x,y



def model_rfc(x_train,y_train):

    num_trees = 30

    max_features = 3

    rfc = RandomForestClassifier(n_estimators = num_trees,max_features = max_features, random_state = 5,n_jobs = 1)

    rfc.fit(x_train,y_train)

    return rfc





def main():

    x,y = read_data()

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=2)

    

    rfc = model_rfc(x_train,y_train)

    predictions = rfc.predict(x_test)

    print("random forest results : ",accuracy_score(predictions,y_test))

    

    print("Confusion Matrix")

    print(confusion_matrix(predictions,y_test))

    

    plt.plot(predictions,y_test)

    plt.xlabel('Predicted Result')

    plt.ylabel('Actual Result')

    plt.show()

    

if __name__ == '__main__':

    main()