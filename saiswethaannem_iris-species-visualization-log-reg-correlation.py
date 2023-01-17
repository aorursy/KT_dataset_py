import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#with the help of pandas will will read the CSV file
iris = pd.read_csv("../input/iris/Iris.csv")
#describe function is used for calculatig some statistical data
iris.describe()
# print the top 5 records in the dataset
iris.head(5)
#print all the columns present
iris.columns
#remove the Id column since it is not going to help us in classification of the species.
iris=iris.drop(['Id'], axis=1)
# here we are printing the no of labels present in the dataset
iris['Species'].value_counts()
import warnings
warnings.filterwarnings("ignore")

ax=sns.FacetGrid(iris, hue="Species", size=5) .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") .add_legend()
plt.title('SepalWidthCm and SepalLengthCm')
sns.FacetGrid(iris, hue="Species", size=5) .map(plt.scatter, "PetalLengthCm", "PetalWidthCm") .add_legend()
#here we will observe petallength and petalwidth separately with the help of boxplt
ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")
#visualization of petalwidthcm 
ax = sns.boxplot(x="Species", y="PetalWidthCm", data=iris)
ax = sns.stripplot(x="Species", y="PetalWidthCm", data=iris, jitter=True, edgecolor="gray")
sns.violinplot(x="Species", y="PetalLengthCm", data=iris, size=6)
sns.violinplot(x="Species", y="PetalWidthCm", data=iris, size=6)
sns.FacetGrid(iris, hue="Species", size=6).map(sns.kdeplot, "PetalLengthCm").add_legend()
sns.FacetGrid(iris, hue="Species", size=6).map(sns.kdeplot, "PetalWidthCm").add_legend()
#let us find the correlation between all the variables

plt.figure(figsize=(7,4)) 
sns.heatmap(iris.corr(),annot=True,cmap='cubehelix_r')
plt.show()
import sklearn
from sklearn.model_selection  import train_test_split

from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection  import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
train, test = train_test_split(iris, test_size = 0.3)
#here we will not be taking the petal length 
train_X = train[['SepalLengthCm','SepalWidthCm','PetalWidthCm']]
train_y=train.Species
test_X= test[['SepalLengthCm','SepalWidthCm','PetalWidthCm']]
test_y =test.Species 
model2 = LogisticRegression()
model2.fit(train_X,train_y)
prediction=model2.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_y))
#here we will not be taking the petal length 
train_X = train[['SepalLengthCm','SepalWidthCm','PetalWidthCm','PetalLengthCm']]
train_y=train.Species
test_X= test[['SepalLengthCm','SepalWidthCm','PetalWidthCm','PetalLengthCm']]
test_y =test.Species 
model3 = LogisticRegression()
model3.fit(train_X,train_y)
prediction=model3.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_y))
from prettytable import PrettyTable
tb = PrettyTable()
tb.field_names= ("Model", "Accuracy","Removed_Correlated_Var")
tb.add_row(["Logistic Regression", 0.95,'Yes'])
tb.add_row(["Logistic Regression", 0.97,'No'])

print(tb.get_string(titles = "Iris Species"))
#print(tb)

