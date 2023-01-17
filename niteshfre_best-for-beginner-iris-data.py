import pandas as pd     #load data and data manipulation
import seaborn as slt  #high level inteface visualization
import numpy as np      #for mathematical computational
from sklearn import svm  #svm (Support Vector Machine) a classification machine learning algorithm
from sklearn.model_selection import train_test_split  #splits the dataset into training and testing data
from mlxtend.plotting import plot_decision_regions    #for plotting SVM classes
from matplotlib import pyplot as plt                  #basic for visualization
#reading the csv of iris-data
df = pd.read_csv('../input/iris.data.csv')
df.head()
#it has no columns 
#by default pandas takes my first row as column names
#storing the first rows i.e. Columns in variable
col = df.columns
#setting column names
#slen- sepal length
#swid- sepal width
#plen- petal length
#pwid- petal width

df.columns=['slen','swid','plen','pwid','class']
#adding our first row to the data
df.loc[150]=col
#dimensions of our dataset
print (df.shape)
#few top rows of iris data
df.head()
#checking for number of nan values in any column
df.isna().sum()
#converts string values into integer data-type
df[['slen','swid','plen','pwid']]=df[['slen','swid','plen','pwid']].apply(pd.to_numeric)
#plotting pairplot of whole data
#used to find the relation among the all columns
slt.pairplot(df , hue='class')
#increasing the default figure size of matplot
plt.figure(figsize=(7,6))
#plotting scatter plot
#hue is for categorizing our data
slt.scatterplot(df['slen'],df['swid'],data=df,hue='class',s=60)
plt.figure(figsize=(7,6))
#scatterplot between petal length and petal width
slt.scatterplot(df['plen'],df['pwid'],data=df,hue='class',s=60)
#lmplot of seaborn between sepal length and sepalt width
#linear line is regression line with 95% of confidence interval for regression
slt.lmplot(x='slen',y='swid',data=df,col='class')
#lmplot between petal length and petal width
slt.lmplot(x='plen',y='pwid',data=df,col='class')
plt.figure(figsize=(8,5))
#heatmap of correlation matrix of our data
#it clearly indicates that sepal width is less correlated with other attributes
#sepal length, petal length and petal width are highly correalted with each other 
slt.heatmap(df.corr(),annot=True,vmax=1,cmap='Greens')
#this code is for mapping different categories of class to integer value
#Iris-setosa = 0
#Iris-versicolor = 1
#Iris-virginica =2
m = pd.Series(df['class']).astype('category')
df['class']=m.cat.codes
#dividing data into our dependent and independent attributes
#classes we are going to predict
Y=df['class']
#By using other attributes  
X = df.drop(columns=['class'])
#split the data randomly into train and test set
#where test data is 30% of original data
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3)
#re-indexing the ytrain data to linear numerical values between 0-105
#it is used for plotting SVM classes
ytrain.index=np.arange(105)
#select the classifier i.e. Linear SVC (Support Vector Classifier)
clf = svm.SVC(gamma='auto')
#fit the train data or training to our model
pre = clf.fit(xtrain,ytrain)
#check score of our model on test data
clf.score(xtest,ytest)
#PCA (Principal Component Analysis)
from sklearn.decomposition import PCA
#Linear dimensionality reduction using Singular Value Decomposition 
#of the data to project it to a lower dimensional space.
pc = PCA(n_components=2).fit(xtrain)
#in our case the dimension of train data has 4 columns 
#then, it is going to reduce the dimension to 2 columns 
pca = pc.transform(xtrain)
#classifying another classifier for pca(low dimentional data)
clf2=svm.LinearSVC().fit(pca,ytrain)

plt.figure(figsize=(8,6))
#plotting scatter plot for each different classes
for i in range (0,pca.shape[0]):
    if ytrain[i]==0:
       c1=plt.scatter(pca[i,0],pca[i,1],c='r',marker='+',s=60)   # Iris-setosa
    elif ytrain[i]==1:
       c2=plt.scatter(pca[i,0],pca[i,1],c='b',marker='o',s=50)   # Iris-versicolor 
    elif ytrain[i]==2:
       c3=plt.scatter(pca[i,0],pca[i,1],c='g',marker='*',s=60)   # Iris-virginica

#defining legends of our plot
plt.legend([c1,c2,c3],['Iris-setosa','Iris-versicolor','Iris-viginica'])
 

x_min, x_max = pca[:, 0].min() - 1,   pca[:,0].max() + 1
y_min, y_max = pca[:, 1].min() - 1,   pca[:, 1].max() + 1

#np.meshgrid is to create a rectangular grid out of an array of x values and an array of y values.
x1, y1 = np.meshgrid(np.arange(x_min, x_max, .01),   np.arange(y_min, y_max, .01))
#np.ravel returns contiguous flattened array (1D array)
# ex. np.array((1,2),(3,4)) => [1,2,3,4]
m = clf2.predict(np.c_[x1.ravel(),  y1.ravel()])


m = m.reshape(x1.shape)
#draw contour lines 
plt.contour(x1, y1, m)
plt.title("SVM Classifiers of 3 classes")
plt.show()
