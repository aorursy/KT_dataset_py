import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm # for Support Vector Machine
from sklearn import metrics # for the check the error and accuracy
import matplotlib as mpl
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

df.shape#569 rows and 33 columns

df.columns
df.drop('Unnamed: 32',axis = 1,inplace = True)

X = df.iloc[:,:]
y = df.iloc[:,1]

X = X.drop(['diagnosis','id'],axis = 1)

X.info()

# VISUALIZATION
mpl.style.use(['ggplot']) 
# for ggplot-like style

sns.pairplot(df.loc[:,'diagnosis':'area_mean'], hue="diagnosis");

y.value_counts().plot(kind ="bar")

y.value_counts().plot(kind ="pie")

new_data_B= df[df.diagnosis !='M']
new_data_M= df[df.diagnosis !='B']
new_data_B.plot(kind = "density",x= 'radius_mean', y = 'concavity_mean')
plt.xlabel("mean radius for benigm")
plt.ylabel("mean concavity for benigm")
new_data_M.plot(kind = "density",x= 'radius_mean', y = 'concavity_mean')
plt.xlabel("mean radius for malignant")
plt.ylabel("mean concavity for malignant")
new_data_B.plot(kind = "scatter",x= 'radius_mean', y = 'area_mean')
plt.xlabel("mean radius for benigm")
plt.ylabel("mean area for benigm")

new_data_M.plot(kind = "scatter",x= 'radius_mean', y = 'area_mean')
plt.xlabel("mean radius for malignant")
plt.ylabel("mean area for malignant")

g = sns.jointplot(x=new_data_M['radius_mean'], y=new_data_M['texture_mean'], data=new_data_M, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$mean radius for malignant$", "$mean texture for malignant$");


g = sns.jointplot(x=new_data_B['radius_mean'], y=new_data_B['texture_mean'], data=new_data_B, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$mean radius$", "$mean texture$");


avgB = {}
for i in range(2,new_data_B.shape[1]):
    m = np.mean(new_data_B.iloc[:,i])
    avgB.update({new_data_B.columns[i]:m})

avgB_df = pd.DataFrame(avgB,index = np.arange(1,31))
avgB_df = avgB_df.transpose()
avgB_df = avgB_df.iloc[:,:1]

avgM = {}
for i in range(2,new_data_M.shape[1]):
    m = np.mean(new_data_M.iloc[:,i])
    avgM.update({new_data_M.columns[i]:m})

avgM_df = pd.DataFrame(avgM,index = np.arange(1,31))
avgM_df = avgM_df.transpose()
avgM_df = avgM_df.iloc[:,:1]


#so now, i have 2 data frames and i want to have a combined barplot

avgB_df['hue']='B'
avgM_df['hue']='M'
res=pd.concat([avgB_df,avgM_df])
res = res.reset_index(level =0)
sns.barplot(x = res.iloc[:,0],y = res.iloc[:,1],data=res,hue='hue')
plt.xticks(rotation=90)
plt.ylabel('average of feature mentioned on X axis')
plt.show()
g = sns.PairGrid(new_data_B.loc[:,'radius_mean':'smoothness_mean'])
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6);

g = sns.PairGrid(new_data_M.loc[:,'radius_mean':'smoothness_mean'])
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6);


data = pd.DataFrame(X)
data_n_2 = (data - data.mean()) / (data.std())  
data = pd.concat([y,data_n_2.iloc[:,10:25]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=45);
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax);
#lets see the amount of benigan and melignant tissues:
#lets use countplot for this.
B,M = y.value_counts()

print(B,M)
#we can see that there are 357 B type and 212 M type cells

#lets split the data now
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size = 0.2,random_state = 1)

model=RandomForestClassifier(n_estimators=100)
model.fit(xtrain,ytrain)# now fit our model for traiing data
prediction=model.predict(xtest)# predict for the test data
print(metrics.accuracy_score(prediction,ytest))


    
model = svm.SVC()
model.fit(xtrain,ytrain)# now fit our model for traiing data
prediction=model.predict(xtest)# predict for the test data

metrics.accuracy_score(prediction,ytest)
print(metrics.accuracy_score(prediction,ytest))
metrics.confusion_matrix(ytest,prediction)


#knn
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(xtrain,ytrain)
ypred = knn.predict(xtest)
print(metrics.accuracy_score(ypred,ytest))
metrics.confusion_matrix(ytest,prediction)




#naive bayes
knn = GaussianNB()
knn.fit(xtrain,ytrain)
ypred = knn.predict(xtest)
print(metrics.accuracy_score(ypred,ytest))
metrics.confusion_matrix(ytest,prediction)


#decision tree

dt = DecisionTreeClassifier()
dt.fit(xtrain,ytrain)
ypred = dt.predict(xtest)
print(metrics.accuracy_score(ypred,ytest))
metrics.confusion_matrix(ytest,prediction)

#decision tree

dt = DecisionTreeClassifier()
dt.fit(xtrain,ytrain)
ypred = dt.predict(xtest)
print('DECISION TREE CLASSIFIER:: ',metrics.accuracy_score(ypred,ytest))

#random forest
model=RandomForestClassifier(n_estimators=100)
model.fit(xtrain,ytrain)# now fit our model for traiing data
prediction=model.predict(xtest)# predict for the test data
print('FORSEST TREE CLASSIFICATION:: ',metrics.accuracy_score(prediction,ytest))


#SVM
model = svm.SVC()
model.fit(xtrain,ytrain)# now fit our model for traiing data
prediction=model.predict(xtest)# predict for the test data
metrics.accuracy_score(prediction,ytest)
print('SUPPORT VECTOR MACHINE:: ',metrics.accuracy_score(prediction,ytest))

#knn
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(xtrain,ytrain)
ypred = knn.predict(xtest)
print('K NEAREST NEIGHBOURS:: ',metrics.accuracy_score(ypred,ytest))


#naive bayes
NB = GaussianNB()
NB.fit(xtrain,ytrain)
ypred = NB.predict(xtest)
print('NAIVE BAYES ALGORITHM:: ',metrics.accuracy_score(ypred,ytest))




#what if i scale the data now::
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
#lets split the data now
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size = 0.2,random_state = 1)

#decision tree
dt = DecisionTreeClassifier()
dt.fit(xtrain,ytrain)
ypred = dt.predict(xtest)
print('DECISION TREE CLASSIFIER:: ',metrics.accuracy_score(ypred,ytest))

#random forest
model=RandomForestClassifier(n_estimators=100)
model.fit(xtrain,ytrain)# now fit our model for traiing data
prediction=model.predict(xtest)# predict for the test data
print('FORSEST TREE CLASSIFICATION:: ',metrics.accuracy_score(prediction,ytest))


#SVM
model = svm.SVC()
model.fit(xtrain,ytrain)# now fit our model for traiing data
prediction=model.predict(xtest)# predict for the test data
metrics.accuracy_score(prediction,ytest)
print('SUPPORT VECTOR MACHINE:: ',metrics.accuracy_score(prediction,ytest))

#knn
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(xtrain,ytrain)
ypred = knn.predict(xtest)
print('K NEAREST NEIGHBOURS:: ',metrics.accuracy_score(ypred,ytest))


#naive bayes
NB = GaussianNB()
NB.fit(xtrain,ytrain)
ypred = NB.predict(xtest)
print('NAIVE BAYES ALGORITHM:: ',metrics.accuracy_score(ypred,ytest))




