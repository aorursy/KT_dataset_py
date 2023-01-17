import pandas as pd
data = pd.read_csv("../input/iris/Iris.csv")
data
x = data.iloc[:,0:5].values
y = data.iloc[:,5].values
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)
x = pd.DataFrame(x)
x
from sklearn.decomposition import PCA
pca = PCA()
x_train = pca.fit_transform(x)
x_train = pd.DataFrame(x_train)
x_train.head()
variance = pca.explained_variance_ratio_
variance
x_train
x_train['target']=y
x_train.columns = ['PC1','PC2','PC3','PC4','PC5','target']
x_train.head()
import matplotlib.pyplot as plt
fig = plt.figure()
label = fig.add_subplot(1,1,1) 
label.set_xlabel('Principal Component 1') 
label.set_ylabel('Principal Component 2') 
label.set_title('2 component PCA') 
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
 indicesToKeep = x_train['target'] == target
 label.scatter(x_train.loc[indicesToKeep, 'PC1']
 , x_train.loc[indicesToKeep, 'PC2']
 , c = color
 , s = 50)
label.legend(targets)
label.grid()
