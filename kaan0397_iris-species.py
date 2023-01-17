import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use("ggplot")
data = pd.read_csv("../input/Iris.csv")
data.head()
data.info()
data.isnull().sum()
data.describe()
#Drop "Id" Column

data.drop(["Id"],axis=1,inplace=True)
data.Species.value_counts(ascending=False) 
#One-Hot-Encoding

data.loc[:,"Species"] = data.Species.replace(["Iris-virginica" , "Iris-versicolor" , "Iris-setosa"] , [1,2,3])
#Correlation Map

plt.figure(figsize = (10,5))

sns.heatmap(data.corr() , annot=True)



plt.tight_layout()

plt.show()
corr_m = data.corr()

corr_m["Species"].sort_values(ascending=False)
plt.figure(figsize = (20,7))



sns.countplot(x = "SepalLengthCm" , hue="Species" , data=data)

plt.legend(["Iris-virginica" , "Iris-versicolor" , "Iris-setosa"] , loc=1 , fontsize="xx-large" )

plt.title("Sepal Length")



plt.show()
plt.figure(figsize = (20,7))



sns.countplot(x = "SepalWidthCm" , hue="Species" , data=data)

plt.legend(["Iris-virginica" , "Iris-versicolor" , "Iris-setosa"] , loc=1 , fontsize="xx-large" )

plt.title("Sepal Width")



plt.show()
plt.figure(figsize = (20,7))



sns.countplot(x = "PetalLengthCm" , hue="Species" , data=data)

plt.legend(["Iris-virginica" , "Iris-versicolor" , "Iris-setosa"] , loc=1 , fontsize="xx-large" )

plt.title("Petal Length")



plt.show()
plt.figure(figsize = (20,7))



sns.countplot(x = "PetalWidthCm" , hue="Species" , data=data)

plt.legend(["Iris-virginica" , "Iris-versicolor" , "Iris-setosa"] , loc=1 , fontsize="xx-large" )

plt.title("Petal Width")



plt.show()
data.plot(kind = "scatter" ,  x = "Species" , y = "SepalLengthCm" , figsize = (10,5) )



plt.title("SpealLengthCm")

plt.show()
data.plot(kind = "scatter" ,  x = "Species" , y = "SepalWidthCm" , figsize = (10,5) )



plt.title("SepalWidthCm")

plt.show()
data.plot(kind = "scatter" ,  x = "Species" , y = "PetalLengthCm" , figsize = (10,5) )



plt.title("PetalLengthCm")

plt.show()
data.plot(kind = "scatter" ,  x = "Species" , y = "PetalWidthCm" , figsize = (10,5) )



plt.title("PetalWidthCm")

plt.show()
sns.pairplot(data , hue="Species")



plt.show()
data.boxplot(by = "Species" , figsize=(12,6))



plt.show()
#Create x and y

y = data["Species"]

data.drop(["Species"] , axis=1 , inplace=True)

x = data
#Normalization

from sklearn.preprocessing import StandardScaler

x = StandardScaler().fit_transform(x)
#train test split

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.25 , random_state=0)
#Decision Tree

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)



#Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=27,random_state=42)

rf.fit(x_train,y_train)



#Knn

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(x_train,y_train)



#Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)



#Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)



#SVM

from sklearn.svm import SVC

svm = SVC(random_state=1)

svm.fit(x_train,y_train)





print("Decision Tree Score ...: {}".format(dt.score(x_test,y_test)))

print("Random Forest Score ...: {}".format(rf.score(x_test,y_test)))

print("Knn Score : {}".format(knn.score(x_test,y_test)))

print("Logistic Regression Score {}".format(lr.score(x_test,y_test)))

print("Naive Bayes Score ...: {}".format(nb.score(x_test,y_test)))

print("SVM Score ...: {}".format(svm.score(x_test,y_test)))