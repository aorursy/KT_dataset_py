# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
## Code zum einlesen der train.arff
def read_data(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() # get rid of newline at the end of each input line
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data

train = read_data("../input/kiwhs-comp-1-complete/train.arff")

test = pd.read_csv("../input/kiwhs-comp-1-complete/test.csv")
# Test Dataframe
d = {'# Id( String)': [0,1,2,3,4,5], '# Category (String)':[-1,1,1,1,-1,-1]} 
dataframe = pd.DataFrame(data=train, columns=['X','Y', 'Kategorie'])
dataframe.index +=1
dataframe.head()

#testframe = pd.DataFrame(test)
#testframe
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#iris = datasets.load_iris()
#knn = KNeighborsClassifier()
#knn.fit(iris['data'], iris['target'])
#prediction = knn.predict(train)
#print(prediction)

from sklearn.model_selection import train_test_split

X = dataframe[["X", "Y"]].values
Y = dataframe["Kategorie"].values

#Aufspiltung der Daten
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, random_state=0, test_size = 0.2)


from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#iris = datasets.load_iris()
#knn = KNeighborsClassifier()
#knn.fit(iris['data'], iris['target'])
#prediction = knn.predict(train)
#print(prediction)

#stand_scaler = StandardScaler()
#stand_scaler.fit(X_Train)
#Ergebnis ausplottern
color = {-1:'blue',1:'red'}
plt.scatter(X[:,0],X[:,1],c=dataframe["Kategorie"].apply(lambda x: color[x]))
plt.xlabel("x")
plt.ylabel("y")
plt.show()
#iris = datasets.load_iris()
knn = KNeighborsClassifier(14)
knn.fit(X_Train,Y_Train)
#score berechnen
print(knn.score(X_Train,Y_Train))

predictions = knn.predict(test[["X", "Y"]].values)
#CSV datei
submissions = pd.DataFrame({"Id (String)": list(range(0,len(predictions))),
                         "Category (String)": predictions.astype(int)})
submissions.index +=1
submissions.to_csv("submissions.csv", index=False, header=True)