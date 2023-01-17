# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from IPython.display import Image

url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'

Image(url,width=800, height=800)
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'

Image(url,width=800, height=800)
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'

Image(url,width=800, height=800)
import seaborn as sns

iris=sns.load_dataset("iris")

iris.head()
iris.info()

#The data is not so long, just 150 row with 5 columns only one of which is a string, which is our target variable
iris.describe(include="all")
sns.pairplot(iris,hue="species")
import matplotlib.pyplot as plt

%matplotlib inline
setosa=iris[iris["species"]=="setosa"]

plt.figure(figsize=(15,10))

sns.kdeplot(setosa["sepal_width"],setosa["sepal_length"],cmap="plasma",shade=True,shade_lowest=False)
versicolor=iris[iris["species"]=="versicolor"]

plt.figure(figsize=(15,10))

sns.kdeplot(versicolor["sepal_width"],versicolor["sepal_length"],cmap="BuPu",shade=True,shade_lowest=False)
virginica=iris[iris["species"]=="virginica"]

plt.figure(figsize=(15,10))

sns.kdeplot(virginica["sepal_width"],virginica["sepal_length"],cmap="YlGnBu",shade=True,shade_lowest=False)
X=iris.drop("species",axis=1)

y=iris["species"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
from sklearn.svm import SVC

model=SVC()

model.fit(X_train,y_train)
predictions=model.predict(X_test)

predictions
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))

#here we get the classification report to learn how accurate our model is
print(confusion_matrix(y_test,predictions))
from sklearn.model_selection import GridSearchCV
param_grid={"C":[1,10,100,100],"gamma":[1,0.1,0.01,0.001,0.0001]} 

#here we select values for grid search to try

grid=GridSearchCV(SVC(),param_grid,verbose=2)
grid.fit(X_train,y_train)

# we apply it to our training data to see the best C and gamma values

#grid.fit() will find the best combination of C and gamma values for our model
grid.best_params_
grid.best_estimator_
grid_predictions=grid.predict(X_test) 

#Now we predict with this readjustment
print(classification_report(y_test,grid_predictions))
print(confusion_matrix(y_test,grid_predictions))