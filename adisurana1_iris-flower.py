import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
iris=pd.read_csv("../input/Iris.csv")
iris.head()
iris=iris.drop(columns='Id')
sns.set_style('darkgrid')
sns.pairplot(data=iris,hue='Species',diag_kws={'edgecolor':"black"})
setosa = iris[iris['Species']=='Iris-setosa']
sns.set_style('darkgrid')
sns.jointplot(x=setosa['SepalWidthCm'],y= setosa['SepalLengthCm'],
                 cmap="plasma",data=setosa,kind='kde')
from sklearn.model_selection import train_test_split
X = iris.drop('Species',axis=1)
y = iris['Species']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)
from sklearn.svm import SVC
svc_model=SVC()
svc_model.fit(X_train,y_train)
predictions=svc_model.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1, 10, 100],'gamma':[1,0.1,0.01,0.001],'kernel':['rbf']}
grid=GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_
grid_predictions=grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))