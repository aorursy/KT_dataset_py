import pandas as pd    
import seaborn as slt  
import numpy as np      
from sklearn import svm  
from sklearn.model_selection import train_test_split  
from mlxtend.plotting import plot_decision_regions    
from matplotlib import pyplot as plt                  
df = pd.read_csv('../input/iris-dataset/iris.data.csv')
df.head()
col = df.columns
col
df.columns=['slen','swid','plen','pwid','class']
df.loc[150]=col
print (df.shape)
df.head()
df.isna().sum()
df[['slen','swid','plen','pwid']]=df[['slen','swid','plen','pwid']].apply(pd.to_numeric)
slt.pairplot(df , hue='class')
plt.figure(figsize=(7,6))
slt.scatterplot(df['slen'],df['swid'],data=df,hue='class',s=60)
plt.figure(figsize=(7,6))
slt.scatterplot(df['plen'],df['pwid'],data=df,hue='class',s=60)
slt.lmplot(x='slen',y='swid',data=df,col='class')
slt.lmplot(x='plen',y='pwid',data=df,col='class')
plt.figure(figsize=(8,5)) 
slt.heatmap(df.corr(),annot=True,vmax=1,cmap='Greens')
m = pd.Series(df['class']).astype('category')
df['class']=m.cat.codes
Y=df['class']
X = df.drop(columns=['class'])
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3)
ytrain.index=np.arange(105)
clf = svm.SVC(gamma='auto')
pre = clf.fit(xtrain,ytrain)
clf.score(xtest,ytest)

plt.figure(figsize=(8,6))
for i in range (0,pca.shape[0]):
    if ytrain[i]==0:
       c1=plt.scatter(pca[i,0],pca[i,1],c='r',marker='+',s=60)
    elif ytrain[i]==1:
       c2=plt.scatter(pca[i,0],pca[i,1],c='b',marker='o',s=50) 
    elif ytrain[i]==2:
       c3=plt.scatter(pca[i,0],pca[i,1],c='g',marker='*',s=60)
plt.legend([c1,c2,c3],['Iris-setosa','Iris-versicolor','Iris-viginica'])
x_min, x_max = pca[:, 0].min() - 1,   pca[:,0].max() + 1
y_min, y_max = pca[:, 1].min() - 1,   pca[:, 1].max() + 1
x1, y1 = np.meshgrid(np.arange(x_min, x_max, .01),   np.arange(y_min, y_max, .01))
m = m.reshape(x1.shape)
plt.contour(x1, y1, m)
plt.title("SVM Classifiers of 3 classes")
plt.show()
