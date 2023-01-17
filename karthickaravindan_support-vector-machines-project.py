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
#This code doest not work in kaggle 
# import seaborn as sns
# iris=sns.load_dataset('iris')
#follow this
import pandas as pd
iris=pd.read_csv("../input/Iris.csv")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
%matplotlib inline
sns.pairplot(data=iris,hue='Species')
setosa = iris[iris['Species']=='Iris-setosa']
sns.kdeplot( setosa['SepalWidthCm'], setosa['SepalLengthCm'],
                 cmap="plasma", shade=True, shade_lowest=False)
from sklearn.model_selection import train_test_split
X = iris.drop('Species',axis=1)
y = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train,y_train)
predict=svc_model.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predict))
print(classification_report(y_test,predict))
from sklearn.grid_search import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))