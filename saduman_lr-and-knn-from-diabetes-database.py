# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for data visualization
import seaborn as sns #for data visualization

import warnings            
warnings.filterwarnings("ignore") 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#load data from csv file
data=pd.read_csv('../input/diabetes.csv')
#peeak at the data
data.head(10)
#checking for missing values
print('Are there missing values? {}'.format(data.isnull().any().any()))
#missing value control in features
data.isnull().sum()
data.iloc[:,1:8].describe()
#unique class values
data['Outcome'].value_counts()
#visualization
sns.countplot(data['Outcome'])
plt.show()
color_list=['purple' if each==1 else 'aqua' for each in data.loc[:,'Outcome']]
pd.plotting.scatter_matrix(data.loc[:,data.columns!='Outcome'],
                           c=color_list,
                           figsize=(18,18),
                           diagonal='hist',
                           alpha=0.5,
                           s=200,
                           marker='.',
                           edgecolor='black')
plt.show()
y=data.Outcome.values
x_data=data.drop(['Outcome','Age'],axis=1)
x_data.head(15)
#another way for normalization
#from sklearn import preprocessing
#min_max_scaler=preprocessing.MinMaxScaler()
#x=min_max_scaler.fit_transform(x_data)

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train=x_train.T
x_test=x_test.T
y_train=y_train.T
y_test=y_test.T
def initialize_weights_and_bias(dimension):
    w=np.full((dimension,1),0.01)   
    b=0.0
    return w,b
def sigmoid(z):
    y_head=1/(1+np.exp(-z))
    return y_head
def forward_backward_propagation(w,b,x_train,y_train):
    #forward propagation
    z=np.dot(w.T,x_train)+b
    y_head=sigmoid(z)
    loss= -y_train*np.log(y_head)- (1-y_train)*np.log(1-y_head)
    cost=(np.sum(loss))/x_train.shape[1]   #x_train.shape[1] is for scaling
    
    #backward propagation
    #x_train.shape[1] is for scaling
    derivative_weight=(np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] 
    derivative_bias=np.sum(y_head-y_train)/x_train.shape[1]
    gradients={'derivative_weight':derivative_weight,'derivative_bias':derivative_bias}
    
    return cost,gradients
def update(w,b,x_train,y_train,learnig_rate,number_of_iteration):
    cost_list=[]
    cost_list2=[]
    index=[]
    
    #updating(learning) parametres is number_of_iteration times
    for i in range(number_of_iteration):
        #make fordward and backward propagation and find cost and gradients
        cost,gradients=forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        #update
        w=w-learnig_rate*gradients['derivative_weight']
        b=b-learnig_rate*gradients['derivative_bias']
        
        if i%20 == 0:
            cost_list2.append(cost)
            index.append(i)
            print('cost after iteration %i: %f:' %(i,cost))
    
    #we update(learn) parametres weights and bias
    parameters={'weight':w,'bias':b}
    plt.figure(figsize=(10,8))
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel('Number of iteration')
    plt.ylabel('Cost')
    plt.show()
    
    return parameters,gradients,cost_list
def predict(w,b,x_test):
    z=sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction=np.zeros((1,x_test.shape[1]))
    #We're making an estimate based on our condition.
    for i in range(z.shape[1]):
        if z[0,i]<=0.5:
            Y_prediction[0,i]=0
        else:
            Y_prediction[0,i]=1
    return Y_prediction
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,num_iterations):
    #initialize
    dimension=x_train.shape[0] #that is 30
    w,b=initialize_weights_and_bias(dimension)
    #do not change learning rate
    parameters,gradients,cost_list=update(w,b,x_train,y_train,learning_rate,num_iterations)
    
    y_prediction_test=predict(parameters['weight'],parameters['bias'],x_test)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_prediction_test.T)
    print("confusion matrix:\n",cm)
    #print test errors
    print('test accuracy: {} %'.format(100-np.mean(np.abs(y_prediction_test-y_test))*100))
    
logistic_regression(x_train,y_train,x_test,y_test,learning_rate=1,num_iterations=500)

#Now we use sklearn libray 
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train.T,y_train.T)

y_pred=lr.predict(x_test.T)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:\n",cm)

print('test accuracy: {}'.format(lr.score(x_test.T,y_test.T)))
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC  

models=[]
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('K-NN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NaiveBayes',GaussianNB()))
models.append(('SVM',SVC()))
#Let's train models
max_model=0
print("\nAchievements of each model according to ACC (Accuracy) criteria:")
for name,model in models:
    model=model.fit(x_train.T,y_train.T)
    y_pred=model.predict(x_test.T)
    from sklearn import metrics 
    m_accuracy_score=(metrics.accuracy_score(y_test.T,y_pred)*100)
    print("Model --> %s --> ACC: %%%.2f" %(name,m_accuracy_score))
    if m_accuracy_score > max_model:
        max_model=metrics.accuracy_score(y_test,y_pred)*100
        max_model_name=name
print("\nAlgorithm giving the best ACC result:")
print("Model:%s ACC:%%%.2f"%(max_model_name,max_model))
data_knn=pd.read_csv('../input/diabetes.csv')
data_knn.head()
diabet=data[data.Outcome==1]
non_diabet=data[data.Outcome==0]
plt.figure(figsize=(15, 5))
plt.scatter(diabet.Age,diabet.Insulin,color="red",label="Diabet patient")
plt.scatter(non_diabet.Age,non_diabet.Insulin,color="green",label="Non Diabet patient")
plt.xlabel("Age")
plt.ylabel("Insulin")
plt.legend(loc='upper right')
plt.show()
y=data.Outcome.values
x_data=data.drop(['Outcome','Age'],axis=1)
#normalization
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
#knn algorithm
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7) #n_neighbors=k
knn.fit(x_train,y_train)
prediction=knn.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)
print("confusion matrix:\n",cm)
print("{} knn score: {}".format(7,knn.score(x_test,y_test)))
#find k value
score_list=[]
for each in range(1,15):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()