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
import seaborn as sns
import pandas as pd
df=pd.read_csv('../input/Iris.csv')
df.head(5)
import matplotlib.pyplot as plt
%matplotlib inline
sns.pairplot(df,hue='Species')
setosa=df[df['Species']=='Iris-setosa']
sns.kdeplot(setosa['SepalLengthCm'],setosa['SepalWidthCm'],cmap='plasma',shade=True,shade_lowest=False)
import cufflinks
cufflinks.go_offline()
df.corr().iplot(kind='heatmap')
from sklearn.model_selection import train_test_split
x=df.drop('Species',axis=1)
y=df['Species']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)
from sklearn.svm import SVC
svm=SVC(C=0.1,gamma='scale')
svm.fit(x_train,y_train)
from sklearn.metrics import confusion_matrix as cm,classification_report as cr, accuracy_score as asc
pr=svm.predict(x_test)
print(asc(y_test,pr))
print(cm(y_test,pr))
print(cr(y_test,pr))
from sklearn.grid_search import GridSearchCV
param_grid={'C':[1,10,100,0.1,0.01],'gamma':[1,10,0.1,0.01,0.001,0.0001]}
grid=GridSearchCV(SVC(),param_grid,verbose=3,refit=True)

grid.fit(x_train,y_train)
grid_pr=grid.predict(x_test)

print(asc(y_test,grid_pr))
print(cm(y_test,grid_pr))
print(cr(y_test,grid_pr))