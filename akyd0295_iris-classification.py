#importing libraries
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
#importing data set
df=pd.read_csv('../input/Iris.csv',index_col='Id')
df.head()
#splitting data into features and labels
label=df.Species
fea=df.drop('Species',axis=1)
fea.head()
#data vizualization

#scatterplot of seaplelength
plt.scatter(df.SepalLengthCm,df.SepalWidthCm,s=10)
plt.scatter(df.loc[df.Species=='Iris-setosa'].SepalLengthCm,
            df.loc[df.Species=='Iris-setosa'].SepalWidthCm,
            c='red',label='Iris-setosa')
plt.scatter(df.loc[df.Species=='Iris-versicolor'].SepalLengthCm,
            df.loc[df.Species=='Iris-versicolor'].SepalWidthCm,
            c='blue',label='Iris-versicolor')
plt.scatter(df.loc[df.Species=='Iris-virginica'].SepalLengthCm,
            df.loc[df.Species=='Iris-virginica'].SepalWidthCm,
            c='green',label='Iris-virginica')
plt.legend()
plt.ylabel('Sepal Width')
plt.xlabel('Sepal Length')
plt.show()
#scatterplot petallength and petalwidth
plt.scatter(df.loc[df.Species=='Iris-setosa'].PetalLengthCm,
            df.loc[df.Species=='Iris-setosa'].PetalWidthCm,
            c='red',label='Iris-setosa')
plt.scatter(df.loc[df.Species=='Iris-versicolor'].PetalLengthCm,
            df.loc[df.Species=='Iris-versicolor'].PetalWidthCm,
            c='blue',label='Iris-versicolor')
plt.scatter(df.loc[df.Species=='Iris-virginica'].PetalLengthCm,
            df.loc[df.Species=='Iris-virginica'].PetalWidthCm,
            c='green',label='Iris-virginica')
plt.legend()
plt.ylabel('Petal Width')
plt.xlabel('Petal Length')
plt.show()
#splitting data into training data and testing data
x_test,x_train,y_test,y_train=train_test_split(fea,label,test_size=0.2)
#training model
clf=tree.DecisionTreeClassifier()

clf=clf.fit(x_train,y_train)
#testing model
pr=clf.predict(x_test)
accuracy=0
p=pr.tolist()
m=y_test.tolist()
# print('predictions:',p)
# print('test values:',m)
for i in range(len(p)):
    if p[i]==m[i]:
        accuracy=accuracy+1
accuracy=accuracy/len(p)
print('Accuracy:',accuracy)