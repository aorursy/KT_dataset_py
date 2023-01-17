import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

iris=pd.read_csv("../input/Iris.csv")
iris.head()
X=iris.iloc[:,1:-1]
y=iris.iloc[:,-1]
X.describe()
iris.boxplot(by="Species",figsize=(12,6))
iris.hist(figsize=(12,12))
pd.plotting.scatter_matrix(iris)
plt.figure(figsize=(12,12))
plt.subplot(221)
iris.groupby("Species")["SepalLengthCm"].plot()
plt.subplot(222)
iris.groupby("Species")["SepalWidthCm"].plot()
plt.subplot(223)
iris.groupby("Species")["PetalLengthCm"].plot()
plt.subplot(224)
iris.groupby("Species")["PetalWidthCm"].plot()
#import all the classifers
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,KFold,cross_val_score

seed=7#to get same splis for all models
x_train,x_validation,y_train,y_validation=train_test_split(X,y,test_size=.3,random_state=seed)
models=[]
models.append(["LR",LogisticRegression(solver="liblinear",multi_class="ovr")])
models.append(["LDA",LinearDiscriminantAnalysis()])
models.append(["KNN",KNeighborsClassifier()])
models.append(["CART",DecisionTreeClassifier()])
models.append(["NB",GaussianNB()])
models.append(["SVM",SVC(gamma="auto")])
#evaluating each models

results=[]
names=[]

scoring="accuracy"#metric score acuracy is udes to evaluate each model
for name,model in models:
    kfold=KFold(n_splits=10,random_state=seed)
    cv_result=cross_val_score(model,X=x_train,y=y_train,scoring=scoring,cv=kfold)
    results.append(cv_result)
    names.append(name)
    print(name," :",cv_result.mean(),"(%f)" %cv_result.std()) 
    
    
clf=LinearDiscriminantAnalysis()
clf.fit(x_train,y_train)
predictions=clf.predict(x_validation)
#importing some metric functions to check the model perfomance
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_validation,predictions)
confusion_matrix(y_validation,predictions)
print(classification_report(y_validation,predictions))
