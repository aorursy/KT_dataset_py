# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)
# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)
# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data=pd.read_csv('../input/iris-dataset/iris.csv')
data.head()
data.info()
sns.set()
sns.pairplot(data, hue='species', palette='husl')
subset=data[data['species']=='Iris-setosa']
subset2=data[data['species']=='Iris-versicolor']
subset3=data[data['species']=='Iris-virginica']
sns.kdeplot(data=subset[['sepal_length','sepal_width']],cmap="plasma", shade=True, shade_lowest=False)
plt.title('Iris-setosa')
plt.xlabel('sepal_length in cm')
plt.ylabel('Sepal Width in cm')
sns.kdeplot(data=subset2[['sepal_length','sepal_width']],cmap="plasma", shade=True, shade_lowest=False)
plt.title('Iris-versicolor')
plt.xlabel('sepal_length in cm')
plt.ylabel('Sepal Width in cm')
sns.kdeplot(data=subset3[['sepal_length','sepal_width']],cmap="plasma", shade=True, shade_lowest=False)
plt.title('Iris-virginica')
plt.xlabel('sepal_length in cm')
plt.ylabel('Sepal Width in cm')
array = data.values
array
x = array[:,0:4]
y = array[:,4]
from sklearn.model_selection import train_test_split
x, x_test, y, y_test = train_test_split(x,y, test_size=0.30, random_state=101)
from sklearn.svm import SVC
model = SVC()
model.fit(x,y)
predictions = model.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001]} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(x,y)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(x_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))