import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')
data=pd.read_csv("../input/heart.csv")
data.head()
data.info()
data.describe()
plt.figure(figsize=(15,8))
sns.heatmap(data.corr() , annot = True , cmap="inferno")
plt.show()
sns.countplot(x = "target" , hue="sex" ,  data=data , palette="dark")
plt.legend(["Female", "Male"])
plt.xlabel(" Disease                             Not Disease")
plt.show()
y = data["target"].values
x = data.drop(["target"] , axis=1)
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
rf = RandomForestClassifier(n_estimators=27,random_state=42) #n_estimators : Oluşacak Subsample Tree lerin sayısı
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
