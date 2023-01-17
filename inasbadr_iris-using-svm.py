# import nedded libraries
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
import warnings
warnings.simplefilter('ignore')
%matplotlib inline

#Import or upload dataset¶
filename="//kaggle/input/iriscsv/Iris.csv"
df=pd.read_csv(filename)
#Preview of Data

df.head(10)
#Let's check, If there is any inconsistency in the dataset¶

df.info()
#Data Analysis 
#Let's check some statistical facts¶

df.describe()
df['Species'].value_counts()

df.isna().sum()
df1 = df.drop('Id', axis=1)
g = sns.pairplot(df, hue='Species', markers='+')
plt.show()

# Sepal Length VS Width
fig = df1[df1.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
df1[df1.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
df1[df1.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(12,8)
plt.show()

# Sepal Length VS Width
fig = df1[df1.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
df1[df1.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)
df1[df1.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(12,8)
plt.show()
# we can notice that the petal length will help us more than the sepal leangth
df1.hist(edgecolor='black')
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()
plt.figure(figsize=(10,8)) 
fig, ax = plt.subplots(figsize=(12,9)) 
sns.heatmap(df1.corr(),annot=True,ax=ax)
#sns.heatmap(df1.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())
#plt.show()

#The Sepal Width and Length are not correlated The Petal Width and Length are highly correlated

#We will use all the features for training the algorithm and check the accuracy.

#Then we will use 1 Petal Feature and 1 Sepal Feature to check the accuracy of the algorithm 
#as we are using only 2 features that are not correlated. Thus we can have a variance in the 
#dataset which may help in better accuracy. We will check it later.
label = df['Species']
df.drop(['Species'],inplace=True,axis=1)

df.drop(['Id'] ,inplace =True , axis =1)
df
# we will choose SVM to learn more about how to use it sklearn
from sklearn.svm import SVC #import svm as classifier
from sklearn.model_selection import train_test_split

Xtrain,Xtest,ytrain,ytest=train_test_split(df,label,test_size=0.25)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df
# train model
svm = SVC(class_weight='balanced') # create new svm classifier with default parameters
svm.fit(Xtrain,ytrain)
#now we pass the testing data to the trained algorithm
from sklearn.metrics import confusion_matrix, accuracy_score
predictions = svm.predict(Xtest) # test model against test set
print("Model Acurracy in testing = {}".format(accuracy_score(ytest, predictions))) # print accuracy
import matplotlib.pyplot as plt
#confusion matrix
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(svm, Xtest, ytest)
plt.show()

