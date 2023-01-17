import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import svm

from sklearn.datasets import load_iris
#Loading DataSets

iris=load_iris()

X=iris.data

y=iris.target

t_names=iris.target_names

features=iris.feature_names

features.append("Target")

df=pd.DataFrame(np.concatenate((X,y.reshape(150,1)), axis=1),columns=features)

#Renaming columns

df.columns=df.columns.str.strip().str.lower().str.replace(' ','_').str.replace('(','').str.replace(')','')

X=df.iloc[:,:2]

y=df.loc[:,"target"]
#create an instance of SVM

def SVMModel(c,kernel,gamma):

    model=svm.SVC(kernel=kernel,random_state=1,C=c,gamma=gamma).fit(X,y)

    return (model,kernel,c,gamma)
#Create to mesh to plot in

def plot(mdl,krn,c, gamma):

    x_min,x_max=X.iloc[:,0].min()-1,X.iloc[:,0].max()+1

    y_min, y_max=X.iloc[:,1].min()-1,X.iloc[:,1].max()+1

    h=(x_max/x_min)/100

    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

    plt.subplot(1,1,1)

    Z=mdl.predict(np.c_[xx.ravel(),yy.ravel()])

    Z=Z.reshape(yy.shape)

    plt.contourf(xx,yy,Z,alpha=0.8,cmap=plt.cm.Paired)

    plt.scatter(X.iloc[:,0],X.iloc[:,1],c=y)

    plt.xlabel('Sepal Length (cm)')

    plt.ylabel('Sepal Width (cm)')

    plt.xlim(xx.min(),xx.max())

    plt.ylim(yy.min(),yy.max())

    plt.title(f'SVM with Kernel={krn} , value of C={c} and gamma={gamma}')

    plt.show()
#Plotting the prediction of SVM model with linear kernel and gamma as scale

model,kernel,c,g=SVMModel(c=1,kernel="linear",gamma="scale")

plot(model,kernel,c,g)
#plotting the prediction of svm with linear kernel and gamma as auto

model1,kernel1,c,g=SVMModel(c=1,kernel='linear',gamma="auto")

plot(model1,kernel1,c,g)
#plotting the prediction of svm with linear kernel and gamma as auto

model,kernel,c,g=SVMModel(c=10,kernel='linear',gamma="auto")

plot(model,kernel,c,g)
#plotting the prediction of svm with rbf kernel and gamma as auto

model,kernel,c,g=SVMModel(c=1,kernel='rbf',gamma="auto")

plot(model,kernel,c,g)
#plotting the prediction of svm with rbf kernel and gamma as auto

model,kernel,c,g=SVMModel(c=1,kernel='rbf',gamma=1)

plot(model,kernel,c,g)
#plotting the prediction of svm with poly kernel and gamma as auto

model,kernel,c,g=SVMModel(c=1,kernel='poly',gamma=10)

plot(model,kernel,c,g)
#Hyperplane equation for the model1<SVM Model with Linear Kernal and C=1>

#get the separating hyperplane

w=model1.coef_[0]

a=-w[0]/w[1]

xx=np.linspace(1,10)

yy=a*xx-(model1.intercept_[0])/w[1]



#plot the parallels to the separating hyperplane that passes through the supporting vectors

b=model1.support_vectors_[0]

yy_down=a*xx + (b[1] - a*b[0])

b1=model1.support_vectors_[-1]

yy_up=a*xx + (b1[1] -  a*b1[0])



#Plot the lines,points and nearest vector to plane

plt.figure(figsize=(15,5))

plt.plot(xx,yy,'k-')

plt.plot(xx,yy_down,'k--')

plt.plot(xx,yy_up,'k--')

plt.scatter(model1.support_vectors_[:,0], model1.support_vectors_[:,1],s=80, facecolors='none')

plt.scatter(X.iloc[:,0],X.iloc[:,1],c=y, cmap=plt.cm.Paired)

plt.axis('tight')

plt.show()