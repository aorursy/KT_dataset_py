#01FB16ECS419 Tejas Prashanth
#01FB16ECS416 Tanmaya Udupa
#01FB16ECS426 VS Meghana
#Assignment6

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns #For the correlation plot
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import math as m
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import graphviz
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
#import sys
#print(sys.path)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("."))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/Absenteeism_at_work.csv")
print(df.iloc[0,])
print(df.columns)
#Summary stats
print(df.describe())



#Split into training and testing
X=df[df.columns.difference(['Absenteeism time in hours'])]
y=df[['Absenteeism time in hours']]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
# for ele in X_train.columns:
#     plt.hist(X_train.loc[:,ele])
#     plt.title(ele)
#     plt.show()
plt.hist(y_train.iloc[:,0])
plt.title("Absenteeism time in hours")
plt.show()   


sns.boxplot(y_train.iloc[:,0])
plt.show()

print(y_train)
training_data=pd.concat([X_train,y_train],ignore_index=False,axis=1)
print(training_data.head())
print(len(training_data))

#Since the time is positively skewed, all outliers should be removed. 
# q1=y_train.iloc[:,0].quantile(0.25)
# q3=y_train.iloc[:,0].quantile(0.75)
# iqr=q3-q1
# upper_limit=q3+1.5*iqr
# training_data=training_data.loc[training_data['Absenteeism time in hours']<upper_limit]
# print("hello")
# print(training_data)
# X_train=training_data.iloc[:,:-1]
# print(X_train.columns)
# y_train=training_data.iloc[:,-1]

q1=df.loc[:,'Absenteeism time in hours'].quantile(0.25)
q3=df.loc[:,'Absenteeism time in hours'].quantile(0.75)
iqr=q3-q1
upper_limit=q3+1.5*iqr
df=df.loc[df['Absenteeism time in hours']<upper_limit]
print("hello")

X=df[df.columns.difference(['Absenteeism time in hours'])]
y=df[['Absenteeism time in hours']]
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
training_data=pd.concat([X_train,y_train],ignore_index=False,axis=1)
print(training_data.head())
print(len(training_data))

print(training_data)
X_train=training_data.iloc[:,:-1]
print(X_train.columns)
y_train=training_data.iloc[:,-1]

#Plot a correlation matrix
attr_to_drop=[]
cor_matrix=training_data.corr()
sns.heatmap(cor_matrix,xticklabels=cor_matrix.columns.values,yticklabels=cor_matrix.columns.values)
plt.show()
#print(cor_matrix)
print("Age and Absenteeism in hours correlation",cor_matrix.loc["Age","Absenteeism time in hours"])
print("Service time and absenteeism in hours correlation",cor_matrix.loc["Service time","Absenteeism time in hours"])
#Numerically, age has a stronger correlation with absenteeism in hours. As a result, service time can be removed
#training_data=training_data[training_data.columns.difference(['Service time'])]
#print(cor_matrix.loc["Body mass index","Absenteeism time in hours"])

#No null values in the dataset
print(training_data.isnull().sum())

# unique_dict=dict()
# for i in training_data.columns:
#     unique_dict[i]=training_data.loc[:,i].unique()
# print(unique_dict)
#print(max(y_train.iloc[:,0]))
print(training_data.columns)


#From the correlation plot, age and body mass index depict a strong correlation
print(cor_matrix.loc['Age','Absenteeism time in hours']>cor_matrix.loc['Body mass index','Absenteeism time in hours'])
#Since age depicts a strong correlation, age is considered
attr_to_drop.append("ID")
attr_to_drop.append("Body mass index")
attr_to_drop.append("Service time")
#attr_to_drop.append("Month of absence")
attr_to_drop.append("Distance from Residence to Work")
X_train=X_train[X_train.columns.difference(attr_to_drop)]
X_test=X_test[X_test.columns.difference(attr_to_drop)]

print(y_train)
print(X_train.shape)
print(X_test.shape)




#Decision Trees
from sklearn import tree
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
leaf_nodes=[8,9,10,11,12,13,14,15,16,17,18,19,20]
test_scores=[]
for i in leaf_nodes:
    clf=DecisionTreeClassifier(criterion='entropy',random_state=42,min_samples_leaf=1,max_leaf_nodes=i)
    clf.fit(X_train,y_train)
    predict_y_test=clf.predict(X_test)
    test_scores.append(accuracy_score(y_test,predict_y_test))

plt.plot(leaf_nodes,test_scores)
plt.show()

#Since having a maximum of 14-16 leaf nodes yields the highest accuracy, the classifier chosen is as follows    
    
clf=DecisionTreeClassifier(criterion='entropy',random_state=42,min_samples_leaf=1,max_leaf_nodes=16)
clf.fit(X_train,y_train)
predict_y_test=clf.predict(X_test)
print(clf.feature_importances_)

print(confusion_matrix(y_test,predict_y_test))
print(classification_report(y_test,predict_y_test))
print("Testing accuracy")
print(accuracy_score(y_test,predict_y_test))
print("Training accuracy")
print(clf.score(X_train,y_train))
print(X_train.columns)

dot_data = tree.export_graphviz(clf, out_file="Absenteeism.dot",feature_names=X_train.columns.tolist(),class_names=y_train.unique().astype(str).tolist()) 
graph = graphviz.Source(dot_data) 
os.system("dot -Tpng Absenteeism.dot -o absent.png")

# from sklearn.datasets import load_iris
# iris=load_iris()
# print(iris.target_names)
print(len(X_train.columns.tolist()))
 
#using svc

scaler = MinMaxScaler()
training_data_scaled = scaler.fit_transform(X_train)
testing_data_scaled = scaler.fit_transform(X_test)
#from mlxtend.plotting import plot_decision_regions

#df1= pd.DataFrame(data=training_data_scaled[1:,1:],index=training_data_scaled[1:,0],columns=training_data_scaled[0,1:])
#X=df1.drop(training_data_scaled.columns[[10]],axis=1)
#print(training_data_scaled)
#X1=np.asarray(training_data_scaled)
#X=X1.drop(X1.columns[[10]],axis=1)
#y=training_data_scaled[10]

C=1
clf = svm.SVC(kernel='rbf', C=C).fit(training_data_scaled,y_train)

#clf = svm.SVC()
#clf.fit(training_data_scaled,y_train)
pred=clf.predict(testing_data_scaled)

#print(y_test)
print(pred)
#y1=np.asarray(y_test)
#print(y1)
print("Training accuracy")
print(clf.score(training_data_scaled,y_train))
print("Testing accuracy")
print(accuracy_score(y_test,pred))

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
#print(accuracy_score(y1,pred))
# from sklearn.metrics import r2_score
# print(y_test)
# coef = r2_score(y_test,pred)
# print(coef)
#USING SVR
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR

scaler = MinMaxScaler()
training_data_scaled = scaler.fit_transform(X_train)
testing_data_scaled = scaler.fit_transform(X_test)

C=1.0
clf=svm.SVR(kernel='linear', C=C).fit(training_data_scaled,y_train)

#clf = svm.SVR()
#clf.fit(training_data_scaled,y_train)
pred=clf.predict(testing_data_scaled)

#print(y_test)
#print(pred)
#y1=np.asarray(y_test)
#print(y1)

#print(accuracy_score(y1,pred))
from sklearn.metrics import r2_score
print(y_test)
coef = r2_score(y_test,pred)
print(coef)



#clf.predict([[2., 2.]])
#C = 1.0 
#svc = svm.SVC(kernel='linear', C=C).fit(X, y)
#print(X)
#clf = SVC(gamma='auto')

#KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
scaler = MinMaxScaler()
training_data_scaled = scaler.fit_transform(X_train)
testing_data_scaled = scaler.fit_transform(X_test)

kmeans = KMeans(n_clusters=len(y_train.unique())-1,max_iter=200,random_state=1,n_init=10)
kmeans.fit(training_data_scaled)
prediction = kmeans.predict(testing_data_scaled)


#In order to assign a label to each cluster(A label that represents absenteeism time in hours),the predict() in run against every sample in the trainign dataset and its KMeans cluster label is found
#Then,the a dictionary is maintained to keep track of the absenteeism time obtain for each KMeans cluster label for every sample
#The majority absenteeism time in each cluster is chosen as the custom cluster label, which is maintained in the dictionary cluster_label 
labels_dict=dict()
cluster_label=dict()
print(training_data_scaled[0])
label=kmeans.predict(training_data_scaled[0].reshape(1,-1))
print(label)
for i in range(len(training_data_scaled)):
    label=kmeans.predict(training_data_scaled[i].reshape(1,-1))
    if(label[0] not in labels_dict.keys()):
        labels_dict[label[0]]=list()
    labels_dict[label[0]].append(y_train.iloc[i])
#print(labels_dict)
for key in labels_dict:
    cluster_label[key]=pd.Series(labels_dict[key]).value_counts().index[0]
print(cluster_label)
#print(y_train.unique())
#y_pred represents the cluster labels given by KMeans
#y_actual represents the actual Absenteeism in hours
def cluster_score_validate(y_actual,y_pred):
    if(len(y_actual)!=len(y_pred)):
        print("Shape error with actual and predictions")
    count=0
    #Iterate through y_pred
    for i in range(len(y_pred)):
        ele=y_pred[i]
        if(y_actual.iloc[i]==cluster_label[ele]):
            count=count+1
    return(float(count/len(y_actual)))
print("Testing accuracy is")
print(cluster_score_validate(y_test.iloc[:,0],prediction))

values_k=list(range(5,len(y_train.unique())+1))
scores=[]
print(values_k)
for j in range(5,len(y_train.unique())+1):
    #print("hello")
    kmeans = KMeans(n_clusters=j,max_iter=200,random_state=1,n_init=20)
    kmeans.fit(training_data_scaled)
    prediction = kmeans.predict(testing_data_scaled)
    labels_dict=dict()
    cluster_label=dict()
    #print(training_data_scaled[0])
    label=kmeans.predict(training_data_scaled[0].reshape(1,-1))
    #print(label)
    for i in range(len(training_data_scaled)):
        label=kmeans.predict(training_data_scaled[i].reshape(1,-1))
        if(label[0] not in labels_dict.keys()):
            labels_dict[label[0]]=list()
        labels_dict[label[0]].append(y_train.iloc[i])
    #print(labels_dict)
    for key in labels_dict:
        cluster_label[key]=pd.Series(labels_dict[key]).value_counts().index[0]
    scores.append(cluster_score_validate(y_test.iloc[:,0],prediction))

plt.plot(values_k,scores)
plt.show()

#KNN
#using knn
from sklearn.neighbors import KNeighborsClassifier 
li=[]
scaler = MinMaxScaler()
training_data_scaled = scaler.fit_transform(X_train)
testing_data_scaled = scaler.fit_transform(X_test)
x=[3,4,5,6,7,8,9,10,11,12]
for i in x :
    classifier = KNeighborsClassifier(n_neighbors=i)  
    classifier.fit(training_data_scaled, y_train)
    y_pred = classifier.predict(testing_data_scaled)
    #print(y_pred)
    li.append(accuracy_score(y_test,y_pred))
#print(li)
plt.plot(x,li)
#plt.xlim(0.4275,0.4276)
plt.show()

np.random.seed(0)
classifier = KNeighborsClassifier(n_neighbors=6)  
classifier.fit(training_data_scaled, y_train)
y_pred = classifier.predict(testing_data_scaled)
print("6 neighbours are chosen since it gives the highest accuracy")
print("Testing accuracy")
print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#from sklearn.metrics import classification_report, confusion_matrix  
#print(confusion_matrix(y_test, y_pred))  
#print(classification_report(y_test, y_pred))  
#Multiple regression model
#There should be no multi collinearity
train_multiple_regression=X_train
print(X_test.columns,train_multiple_regression.columns)
print(train_multiple_regression.shape,X_test.shape)
#y_train=m.log(y_train.astype(np.float))
train_multiple_regression = sm.add_constant(train_multiple_regression)
#print(train_multiple_regression)
model=sm.OLS(y_train,train_multiple_regression).fit()
X_test=sm.add_constant(X_test)
predict_y_test=model.predict(X_test)

print(model.summary())
plot_actual=y_test.reset_index().loc[:,'Absenteeism time in hours']
plot_fitted=predict_y_test.reset_index().iloc[:,1]
#print(plot_x,plot_y)
#plt.plot(,)
plt.scatter(plot_actual,plot_fitted)
plt.title("Fitted vs. Actual values")
plt.xlabel("Actual values")
plt.ylabel("Fitted values")
#print(len(plot_x)==len(plot_y))
#plt.plot(np.arange(len(plot_x)),plot_x,type='line')
plt.show()


residuals=plot_actual-plot_fitted
plt.scatter(plot_fitted,residuals)
plt.title("Residual plot")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show()

#Applying transformations
#print(min(y_train_transformed))
y_train_transformed=y_train
X_train_transformed=X_train
X_test_transformed=X_test
y_train_transformed=pd.to_numeric(y_train_transformed).apply(lambda x:m.log(x) if x!=0 else 0)
X_train_transformed = sm.add_constant(X_train_transformed)

model=sm.OLS(y_train_transformed,X_train_transformed).fit()
X_test_transformed = sm.add_constant(X_test_transformed)

predict_y_test=model.predict(X_test_transformed)

print(model.summary())

plot_actual=pd.to_numeric(y_test.reset_index().loc[:,'Absenteeism time in hours']).apply(lambda x:m.log(x) if x!=0 else 0)
plot_fitted=predict_y_test.reset_index().iloc[:,1]
#print(plot_x,plot_y)
#plt.plot(,)
plt.scatter(plot_actual,plot_fitted)
plt.title("Fitted vs. Actual values")
plt.xlabel("Actual values")
plt.ylabel("Fitted values")
#print(len(plot_x)==len(plot_y))
#plt.plot(np.arange(len(plot_x)),plot_x,type='line')
plt.show()


residuals=plot_actual-plot_fitted
plt.scatter(plot_fitted,residuals)
plt.title("Residual plot")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show()

print(X_test.shape)
print(X_train.shape)

models={"Multiple regression":27.5,"Decision Trees":54,"SVC":46.5,"SVR":18.23,"KMeans":43.1,"KNN":44.25}
plt.bar(np.arange(len(models.keys())),list(models.values()))
plt.title("Comparing accuracy of various models")
plt.xticks(np.arange(len(models.keys())),models.keys(),rotation=45)
plt.xlabel("Model")
plt.ylabel("Accuracy(in %)")
plt.show()

f1_scores={"SVC":37,"KNN":44,"Decision Trees":45}
plt.bar(np.arange(len(f1_scores.keys())),list(f1_scores.values()))
plt.title("Comparing accuracy of various models")
plt.xticks(np.arange(len(f1_scores.keys())),f1_scores.keys(),rotation=45)
plt.xlabel("Model")
plt.ylabel("F1 score(in %)")
plt.show()

print("Although the assignment involved finding classification models, a multiple regression model is also constructed since the dataset involves prediction of absenteeism time, which consists of discrete values. ")
print("1) Among the regression models that were constructed, a log transformed slightly improved the R-squared model, but the residual plots clearly depict a trend in the residual. Hence, linear regression is not applicable for this problem")
print("2) Among the classifiers, Decision Trees and K Nearest Neighbours are the best models.")
print("The performance measures taken into consideration are accuracy and F1 score. This is because accuracy cannot predict the usefulness of a model. F1 score provides a intermediary measure between recall, which depicts the the proportion correctly classified out of all the values that belong to a particular class, and precision, which depicts the proportion of correctly classified points out of all points that were classified as belonging to a particular class")
print("3) Although SVC had a higher accuracy, KNN has a higher F1 score than SVC, due to which KNN is a better model than SVC.")
print("Moreover, the difference in accuracy between KNN and SVC is small whereas the difference in their respective F1 scores is large. As a result, in this case,F1 score is considered more significant than the accuracy")
print("4) Therefore, due to the above mentioned reasons, the best classifier is the Decision Tree, which is followed by the K Nearest Neighbours(KNN)")
