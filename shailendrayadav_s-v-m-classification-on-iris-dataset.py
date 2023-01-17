# Solving the Iris dataset classification using SVM model and check its accuracy as well.

# the species are differntiated based on their petal and sepal length and widths so it is our criteria as well.

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import svm  # SVM: SUPPORT VECTOR MACHINES

from sklearn.datasets import load_iris

iris=load_iris()

type(iris)
dir(iris) # contents of iris directory
iris.feature_names
iris.data  # dataset which will be used as dataframe
iris.target_names #the different species of iris floweres.
#The dataframe construction of dataframces

df=pd.DataFrame(iris.data,columns =iris.feature_names)

df.head()
iris.target_names
# adding new columns astarget and target names to data frame

df["target"]= iris.target

df.head()
df["flower_type"]=df.target.apply(lambda x:iris.target_names[x])

df.head()
#lets divide the data based on different flowers species

df0=df[df.target==0]#setosa

df1=df[df.target==1]#vercicolor

df2=df[df.target==2]#virginica
# Lets plot it to see the model we can use for it. # model selection criteria

plt.scatter(df0["sepal width (cm)"],df0[ "sepal length (cm)"],color="r")

plt.scatter(df1["sepal width (cm)"],df1[ "sepal length (cm)"],color="b")
plt.scatter(df0["petal width (cm)"],df0[ "petal length (cm)"],color="r")

plt.scatter(df1["petal width (cm)"],df1[ "petal length (cm)"],color="b")

#As the plot shows ,we can draw a line to classify the different varieties

#lelts drop the columns flowers_type and target for an unbiased dataset and train our original iris_data datset.

#FEATURES

X=df.drop(["target","flower_type"],axis="columns")

X.head()
#label

y=df.target
#lets train our dataset using test train and split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
len(X_train)
X_train
len(y_train)
y_train
#support vector machine as classifier

from sklearn.svm import SVC

model=SVC()
model
model.gamma ="auto" # setting as auto to avoid warning
model.fit(X_train,y_train)
#predict

y_predict=model.predict(X_test)
model.score(X_test,y_test) # checking the score.
#lets use confusion matrix  to see model performance

from sklearn.metrics import confusion_matrix

cn=confusion_matrix(y_test,y_predict)

cn
#visualise the same using heat map

import seaborn as sns

sns.heatmap(cn,annot=True)

plt.show()