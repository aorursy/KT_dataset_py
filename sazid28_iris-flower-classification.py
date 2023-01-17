import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv("../input/Iris.csv",index_col=0)
data.head(5)
data.tail(5)
data.count()
data.columns
data.corr()
data.describe()
data.all()
map_data = {'Iris-versicolor':1,'Iris-setosa':0,'Iris-virginica':2}
data["Species"] = data["Species"].map(map_data)
data.head()

data.tail(5)
data.shape
data.hist(figsize=(12,10))
feature = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
feature.head(5)
target = data[["Species"]]
target.head(5)
target.tail(5)
sns.pairplot(data,x_vars=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],y_vars=['Species'],size=7,aspect=0.7)
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(feature,target,test_size =.4,random_state=1)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
pred = logreg.predict(X_test)
pred
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,Y_test))
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=1)
KNN.fit(X_train,Y_train)
KNNpred = KNN.predict(X_test)
KNNpred
print(accuracy_score(KNNpred,Y_test))
KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_train,Y_train)
KNNpred = KNN.predict(X_test)
KNNpred
print(accuracy_score(KNNpred,Y_test))
k_range = range(1,26)
scores =[]
for k in k_range :
    KNN = KNeighborsClassifier(n_neighbors=k)
    KNN.fit(X_train,Y_train)
    pred = KNN.predict(X_test)
    scores.append(accuracy_score(pred,Y_test))
    
print(pd.DataFrame(scores))
plt.plot(k_range,scores)
plt.xlabel("K for KNN")
plt.ylabel("Testing accuracy")
plt.show()
data.groupby("Species").size()
data.groupby("Species").count()
data.groupby("SepalLengthCm").count()
data.groupby("SepalLengthCm").size()
