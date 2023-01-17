import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('../input/voice.csv')
data
x=data.iloc[:, 0:20].values
y=data.iloc[:,20].values
y
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_y=LabelEncoder()
y=le_y.fit_transform(y)
y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2)
x_train=lda.fit_transform(x_train,y_train)  #supervised model, y_train is required
x_test=lda.transform(x_test)
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, kernel='rbf')
x_train=kpca.fit_transform(x_train)
x_test=kpca.transform(x_test)
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100, criterion='entropy',max_depth=10, min_samples_leaf=50, min_samples_split=50, random_state=0)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from matplotlib.colors import ListedColormap
x_set,y_set = x_train,y_train
x1,x2 = np.meshgrid(np.arange(start=x_set[:,0].min()-1, stop=x_set[:,0].max()+1, step=0.01),
                    np.arange(start=x_set[:,1].min()-1, stop=x_set[:,1].max()+1, step=0.01))
#create the boundary
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape), 
             alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0], x_set[y_set==j,1],
               c=ListedColormap(('red','green','blue'))(i), label=j)
plt.title("Random Forest(train)")
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()
from matplotlib.colors import ListedColormap
x_set,y_set = x_test,y_test
x1,x2 = np.meshgrid(np.arange(start=x_set[:,0].min()-1, stop=x_set[:,0].max()+1, step=0.01),
                    np.arange(start=x_set[:,1].min()-1, stop=x_set[:,1].max()+1, step=0.01))
#create the boundary
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape), 
             alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0], x_set[y_set==j,1],
               c=ListedColormap(('red','green'))(i), label=j)
plt.title("Random Forest(test)")
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()
