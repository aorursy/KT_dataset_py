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
import pandas as pd

data = pd.read_csv("../input/adult_numerical_binned.csv")
data



#If you want to fill every column with its own most frequent value 





data = data.apply(lambda x:x.fillna(x.value_counts().index[0]))



data.to_csv('adult_numerical_binned1.csv',index=False)

#X = data.iloc[:, [0, 11]].values

X = data.iloc[:, :-1].values

Y = data.iloc[:, 12].values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
def Classfier():

    model.fit(X_train, y_train)

    y_pred=model.predict(X_test)

    #Import scikit-learn metrics module for accuracy calculation

    from sklearn import metrics

    # Model Accuracy, how often is the classifier correct?

    print('Accuracy:',metrics.accuracy_score(y_test, y_pred))

    print('Confusion Matrix:\n', metrics.confusion_matrix(y_test, y_pred))

    print('Report:\n', metrics.classification_report(y_test, y_pred)) 
# Naive Bayes

from sklearn.naive_bayes import GaussianNB as GNB



model = GNB()

print('Naive Bayes') 

Classfier()
# k-nearest neighbor

from sklearn.neighbors import KNeighborsClassifier as KNN



model = KNN()

print('k-nearest neighbor') 

Classfier()
#  DecisionTree

from sklearn.tree import DecisionTreeClassifier as DT



model = DT()

print('DecisionTree') 

Classfier()