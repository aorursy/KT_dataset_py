

from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import numpy as np

np.random.seed(0)
#Import scikit-learn dataset library

from sklearn import datasets



#Load dataset

iris = datasets.load_iris()
#we can see the species names as 0,1 and 2



print(iris.target)
data=pd.DataFrame({

    'sepal length':iris.data[:,0],

    'sepal width':iris.data[:,1],

    'petal length':iris.data[:,2],

    'petal width':iris.data[:,3],

    'species':iris.target})

    
data.head()
#CHECKING THE UNIQUE VALUES



data.nunique()

#CHECKING THE NAN VALUES



data.isnull().sum()
X=data.drop('species',axis=1)

y=data['species']
#SPLITTING THE DATA FOR TEST AND TRAIN



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train,y_train)
feautres=pd.Series(random_forest.feature_importances_,index=iris.feature_names)

feautres
import seaborn as sns

sns.set(style='whitegrid')

sns.barplot(y=random_forest.feature_importances_,x=iris.feature_names,)
#PREDICTING THE MODEL



predictions=random_forest.predict(X_test)
from sklearn.metrics import classification_report

classification_report(y_test,predictions)
#GENERATING CONFUSION MATRIX FOR ANALYSING THE PERFORMANCE OF THE MODEL



from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,predictions)

#FOR FINDING THE ACCURACY OF THE MODEL



from sklearn.metrics import accuracy_score

accuracy_score(y_test,predictions)