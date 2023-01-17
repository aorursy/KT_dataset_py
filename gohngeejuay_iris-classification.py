# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

%matplotlib inline

from scipy.stats import linregress

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

Iris = pd.read_csv("../input/iris/Iris.csv")
Iris.head()
Iris.describe()
Iris["Species"].value_counts()
Iris[Iris.isna().any(axis=1)]
Iris.info()
X = Iris.iloc[:,1:5]    #Dependent variable

X
y = Iris.iloc[:,5]

y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Fitting Decision Tree Classification to the Training set

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)
from sklearn import tree

tree.plot_tree(classifier,fontsize = 7)
# Predicting the Test set results

y_pred = classifier.predict(X_test)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
import matplotlib.pyplot as plt
Iris["Species"].unique()

Iris_Setosa = Iris.loc[Iris["Species"] == "Iris-setosa"]

Iris_Versicolor = Iris.loc[Iris["Species"] == "Iris-versicolor"]

Iris_Virginica = Iris.loc[Iris["Species"] == "Iris-virginica"]

#Check if the categories of species are well separated by their sepal length/width or petal length/width, to judge suitability of clustering

plt.figure(figsize = (15,7))

seto = plt.scatter(x = Iris_Setosa["SepalLengthCm"],y = Iris_Setosa["SepalWidthCm"],marker = "o",color = "r")

vers = plt.scatter(x = Iris_Versicolor["SepalLengthCm"],y = Iris_Versicolor["SepalWidthCm"],marker = "o",color = "b")

virg = plt.scatter(x = Iris_Virginica["SepalLengthCm"],y = Iris_Virginica["SepalWidthCm"],marker = "o",color = "g")

plt.legend((seto,vers,virg),("Setosa","Versicolor","Virginica"),scatterpoints = 1)

plt.xlabel("SepalLengthCm")

plt.ylabel("SepalWidthCm")

plt.title("Sepal Length vs Sepal Width")







#Difficult to plot legend so use above method

#colours = np.where(Iris["Species"] == "Iris-setosa",'r','-')

#print(colours)

#colours[Iris["Species"] == "Iris-versicolor"] = 'g'

#colours[Iris["Species"] == "Iris-virginica"] = 'b'

#print(colours)

#plt.scatter(x = Iris["SepalLengthCm"],y = Iris["SepalWidthCm"],c = colours)

#plt.xlabel("SepalLengthCm")

#plt.ylabel("SepalWidthCm")

#plt.title("Sepal Length vs Sepal Width")
plt.figure(figsize = (15,7))

seto = plt.scatter(x = Iris_Setosa["PetalLengthCm"],y = Iris_Setosa["PetalWidthCm"],marker = "o",color = "r")

vers = plt.scatter(x = Iris_Versicolor["PetalLengthCm"],y = Iris_Versicolor["PetalWidthCm"],marker = "o",color = "b")

virg = plt.scatter(x = Iris_Virginica["PetalLengthCm"],y = Iris_Virginica["PetalWidthCm"],marker = "o",color = "g")

plt.legend((seto,vers,virg),("Setosa","Versicolor","Virginica"),scatterpoints = 1)

plt.xlabel("PetalLengthCm")

plt.ylabel("PetalWidthCm")

plt.title("Petal Length vs Petal Width")



#Difficult to include legend so use above method to separate each plot by color.

#plt.scatter(x = Iris["PetalLengthCm"],y = Iris["PetalWidthCm"],c = colours) 

#plt.xlabel("PetalLengthCm")

#plt.ylabel("PetalWidthCm")

#plt.title("Petal Length vs Petal Width")

#plt.legend(Iris["Species"].unique(),())
plt.figure(figsize = (15,7))

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3).fit(Iris[['PetalLengthCm','PetalWidthCm']])

# Visualise the output labels

plt.scatter(x=Iris['PetalLengthCm'],y=Iris['PetalWidthCm'], c=kmeans.labels_)



# Visualise the cluster centers (black stars)

plt.plot(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],'k*',markersize=20)

plt.xlabel('Distance_Feature')

plt.ylabel('Speeding_Feature')

plt.show()
print(kmeans.labels_)

print(kmeans.cluster_centers_)
colours = np.where(Iris["Species"] == "Iris-setosa",1,-1)

colours[Iris["Species"] == "Iris-versicolor"] = 2

colours[Iris["Species"] == "Iris-virginica"] = 0



for i in range(len(colours)):

    colours[i] = int(colours[i])

cm = confusion_matrix(kmeans.labels_, colours)

cm
import seaborn as sns

sns.set_palette('hls')
#Reference: https://www.kaggle.com/suneelpatel/learn-ml-from-scratch-with-iris-dataset

IrisNoID = Iris.drop('Id', axis=1)    #drop the ID column

sns.pairplot(IrisNoID, hue='Species', markers='+')

plt.show()
IrisNoID.corr()
#Visualizing the correlation: using heatmap

plt.figure(figsize=(10,8)) 

#sns.heatmap(IrisNoID.corr(),annot=True,cmap = "PuBu")

#sns.heatmap(IrisNoID.corr(),annot=True,cmap = "YlOrBr") #cmap matplotlib colormap name or object, or list of colors, optional



#Diverging color map better for correlation

sns.heatmap(IrisNoID.corr(),annot=True,cmap = "PiYG") #cmap matplotlib colormap name or object, or list of colors, optional

#sns.heatmap(iris1.corr(),annot=True,cmap='cubehelix_r') 



plt.show()

#Visualizing the correlation: using diagonal heatmap.

#Reference: Basic correlation plot.(n.d.).Retrieved from: https://riptutorial.com/seaborn/example/31922/basic-correlation-plot



#Compute correlations

corr = IrisNoID.corr()



# Exclude duplicate correlations by masking uper right values

mask = np.zeros_like(corr, dtype=np.bool)    #np.zeros_like return a matrix with same shape except entries = 0 (or boolean/float/etc)

mask[np.triu_indices_from(mask)] = True      #np.triu_indices_from: Return the indices for the upper-triangle of arr.



# Set background color / chart style

sns.set_style(style = 'white')



# Set up  matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Add diverging colormap

cmap = sns.diverging_palette(10, 250, as_cmap=True)



# Draw correlation plot

sns.heatmap(corr, mask=mask, cmap=cmap, 

        square=True,

        linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
# Importing alll the necessary packages to use the various classification algorithms



from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm

from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

from sklearn import metrics #for checking the model accuracy
logr = LogisticRegression()

logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)

acc_log = metrics.accuracy_score(y_pred,y_test)    #metrics from scikit learn. Can be used instead of manually calculating from confusion matrix

print('The accuracy of the Logistic Regression is', acc_log)
sv = svm.SVC() #select the algorithm

sv.fit(X_train,y_train) # we train the algorithm with the training data and the training output

y_pred = sv.predict(X_test) #now we pass the testing data to the trained algorithm

acc_svm = metrics.accuracy_score(y_pred,y_test)

print('The accuracy of the SVM is:', acc_svm)
knc = KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class

knc.fit(X_train,y_train)

y_pred = knc.predict(X_test)

acc_knn = metrics.accuracy_score(y_pred,y_test)

print('The accuracy of the KNN is', acc_knn)
a_index = list(range(1,11))

a = pd.Series()

x = [1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    kcs = KNeighborsClassifier(n_neighbors=i) 

    kcs.fit(X_train,y_train)

    y_pred = kcs.predict(X_test)

    a=a.append(pd.Series(metrics.accuracy_score(y_pred,y_test)))

plt.plot(a_index, a)

plt.xticks(x)