import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from collections import Counter





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv("../input/mushroom-classification/mushrooms.csv")
df.info() 
df.columns = df.columns.str.strip().str.replace('-', '_')
df=df.rename(columns={"class": "m_class"})
df.head() 
df.describe()
for i in df:

    unique=np.unique(df[i])

    print('{}:{}'.format(i,unique))
df = df.drop(['veil_type'], axis=1)
df2=df.copy()
list_df_features=[]

for i in df:

    df_features=i

    list_df_features.append(df_features)

list_df_features
#xtick olarak atamak için

list_unique_real=[]

for i in df:

    unique=np.unique(df[i]).tolist()

    list_unique_real.append(unique)

list_unique_real
print(df['m_class'].value_counts())
#ratio of poisonous and edible mushrooms..

#plt.style.use('bmh')

%matplotlib inline

#

f,ax=plt.subplots(1,2,figsize=(10,4))

df['m_class'].value_counts().plot.pie(explode=[0.1,0],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('m_class')

ax[0].set_ylabel('')

#

sns.countplot('m_class',data=df,ax=ax[1])

ax[1].set_title('m_class')

plt.show()
#for i in df:

#    print(df[i].value_counts(),'\n')
#for i in df:

#    print(df.groupby([i,'m_class'])['m_class'].count())
#Counts of values in each features descending from ascending..

f,ax=plt.subplots(1,6,figsize=(19,4))

for i in range(0,6):

    df[list_df_features[i]].value_counts().plot.bar(color=sns.color_palette("rocket"),ax=ax[i])

    ax[i].set_title(list_df_features[i])
#Counts of poisonous and edible mushrooms in each features.

#plt.style.use('fivethirtyeight')

f,ax=plt.subplots(1,6,figsize=(19,4))

for i in range(0,6):

    sns.countplot(list_df_features[i],hue='m_class',data=df,ax=ax[i])
f,ax=plt.subplots(1,6,figsize=(19,4))

for i in range(0,6):

    df[list_df_features[i+6]].value_counts().plot.bar(color=sns.color_palette("rocket"),ax=ax[i])

    ax[i].set_title(list_df_features[i+6])
f,ax=plt.subplots(1,6,figsize=(19,4))

for i in range(0,6):

    sns.countplot(list_df_features[i+6],hue='m_class',data=df,ax=ax[i])
f,ax=plt.subplots(1,6,figsize=(19,4))

for i in range(0,6):

    df[list_df_features[i+12]].value_counts().plot.bar(color=sns.color_palette("rocket"),ax=ax[i])

    ax[i].set_title(list_df_features[i+12])
f,ax=plt.subplots(1,6,figsize=(19,4))

for i in range(0,6):

    sns.countplot(list_df_features[i+12],hue='m_class',data=df,ax=ax[i])
f,ax=plt.subplots(1,4,figsize=(19,4))

for i in range(0,4):

    df[list_df_features[i+18]].value_counts().plot.bar(color=sns.color_palette("rocket"),ax=ax[i])

    ax[i].set_title(list_df_features[i+18])
f,ax=plt.subplots(1,4,figsize=(19,4))

for i in range(0,4):

    sns.countplot(list_df_features[i+18],hue='m_class',data=df,ax=ax[i])
le=LabelEncoder()

for col in df.columns:

    df[col] = le.fit_transform(df[col])
df.head()
list_unique_encoded=[]

for i in df:

    unique=np.unique(df[i]).tolist()

    list_unique_encoded.append(unique)

list_unique_encoded
for col in df:

    print('{}:{}'.format(col,np.unique(df[col])))

print("\n")

for i in df2:

    print('{}:{}'.format(i,np.unique(df2[i])))
df3=df.copy()
df.describe()
sns.relplot(x="bruises",y="odor",col="m_class",data=df)
sns.relplot(x="spore_print_color",y="odor",col="m_class",data=df)
sns.relplot(x="population",y="odor",col="m_class",data=df)
sns.relplot(x="habitat",y="odor",col="m_class",data=df)
gd_df=pd.get_dummies(df,columns=df.columns)

gd_df
gd_df.loc[3:5, 'm_class_0':'cap_shape_5']
df.corr()
f,ax=plt.subplots(figsize=(15,15))

sns.heatmap(df.corr(), annot=True, lineWidth=.5, fmt='.2f', ax=ax)

plt.show()
color_columns= []

for i in df.columns:

    if 'color' in i:

        color_columns.append(i)

df_color = df[color_columns]

df_color.head()
list_color_features=[]

for i in df_color:

    color_features=i

    list_color_features.append(color_features)

list_color_features
f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(df_color.corr(), annot=True, lineWidth=.5, fmt='.2f',cmap="YlGnBu", ax=ax)

plt.show()
#number of poisonous mushroom in every odor types.

odor_list=list(df['odor'].unique())

for i in odor_list:

    x=df[df['odor']==i]

    print('{}:{}'.format(i,sum(x.m_class)))
#number of edible or poisonous mushroom in every odor types

odor_list_2=list(df['odor'].unique())

for i in odor_list_2:

    x=df[df['odor']==i]

    print('{}:{}'.format(i,x.groupby('m_class').size()))

    print('\n')
#number of mushrooms in every odor types

df.odor.value_counts()
#total poisonous mushroom number

print(len(df[df.m_class == 1]))
#total edible mushroom number

print(len(df[df.m_class == 0]))
df.odor.plot(kind='hist', bins=20, figsize=(8,8), color="gray") 

plt.show()


odor_list=list(df['odor'].unique())

m_class_ratio=[]

for i in odor_list:

    x=df[df['odor']==i]

    m_class_rate=sum(x.m_class)/len(x)

    m_class_ratio.append(m_class_rate)



plt.figure(figsize=(15,10))

sns.barplot(x=odor_list, y=m_class_ratio, palette="Greens_d")

plt.xticks(rotation= 360)

plt.xlabel('Odors')

plt.ylabel('Poison Rate')

plt.title('Poison Rate Given Odors')
df.cap_shape.plot(kind='hist', bins=20, figsize=(8,8), color="gray") 

plt.show()
cap_shape_list=list(df['cap_shape'].unique())

m_class_ratio=[]

for i in cap_shape_list:

    x=df[df['cap_shape']==i]

    m_class_rate=sum(x.m_class)/len(x)

    m_class_ratio.append(m_class_rate)



plt.figure(figsize=(15,10))

sns.barplot(x=cap_shape_list, y=m_class_ratio, palette = sns.cubehelix_palette(len(x)))

plt.xticks(rotation= 360)

plt.xlabel('Cap Shapes')

plt.ylabel('Poison Rate')

plt.title('Poison Rate Given Cap Shapes')
y = df.m_class.values

x_df = df.drop(["m_class"],axis=1)
x = (x_df - np.min(x_df))/(np.max(x_df)-np.min(x_df)).values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

print("logistic regression test accuracy {}".format(lr.score(x_test,y_test)))

y_pred =lr.predict(x_test)

y_true = y_test

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

#visualization

import seaborn as sns

import matplotlib.pyplot as plt



f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



#K fold CV K = 10

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(lr,x,y, cv = 10)

print("average accuracy: ",np.mean(accuracies))

print("average std: ",np.std(accuracies))

#devam
lr2 = LogisticRegression(C=1000.0, penalty="l1", random_state=1)

lr2.fit(x_train,y_train)

print("logistic regression test accuracy with Grid Search Cross Validation {}".format(lr2.score(x_test,y_test)))
y_pred =lr2.predict(x_test)

y_true = y_test

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

#visualization

import seaborn as sns

import matplotlib.pyplot as plt



f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
# %% parameter initialize and sigmoid function

dimension = 22

def initialize_weights_and_bias(dimension):

    

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w,b





w,b = initialize_weights_and_bias(22)



def sigmoid(z):

    

    y_head = 1/(1+ np.exp(-z))

    return y_head

print(sigmoid(0))
def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

    

    # backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    

    return cost,gradients
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    

    # updating(learning) parameters is number_of_iterarion times

    for i in range(number_of_iterarion):

        # make forward and backward propagation and find cost and gradients

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        # lets update

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        

        cost_list2.append(cost)

        index.append(i)

        print ("Cost after iteration %i: %f" %(i, cost))

            

    # we update(learn) parameters weights and bias

    parameters = {"weight": w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list
#%%  # prediction

def predict(w,b,x_test):

    # x_test is a input for forward propagation

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    # initialize

    dimension =  x_train.shape[0] 

    w,b = initialize_weights_and_bias(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)



    # Print test Errors

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(x_train.T, y_train.T, x_test.T, y_test.T,learning_rate = 4, num_iterations = 300)    
p=df[df.m_class== 1]

e=df[df.m_class== 0]
plt.scatter(p.odor,p.cap_shape,color="red",label="poisonous")

plt.scatter(e.odor,e.cap_shape,color="green",label="edible")

plt.xlabel("odors")

plt.ylabel("cap_shapes")

plt.legend()

plt.show()
plt.scatter(p.veil_color,p.gill_attachment,color="red",label="poisonous")

plt.scatter(e.veil_color,e.gill_attachment,color="green",label="edible")

plt.xlabel("veil_color")

plt.ylabel("gill_attachment")

plt.legend()

plt.show()
df["gill_attachment"].value_counts()
df["veil_color"].value_counts()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors =3)

knn.fit(x_train,y_train)

print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))
y_pred =knn.predict(x_test)

y_true = y_test

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

#visualization

import seaborn as sns

import matplotlib.pyplot as plt



f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, random_state=42, shuffle=False)

for trainkf, testkf in kf.split(x):

    #print("%s %s" % (trainkf, testkf))

    print("trainkf:",len(trainkf))

    print("testkf:",len(testkf))
#K fold CV K = 10

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(knn,x,y, cv = 10)

print(accuracies)

print("average accuracy: ",np.mean(accuracies))

print("average std: ",np.std(accuracies))

#devam
# find k value

score_list = []

for each in range(1,50):

    knn3 = KNeighborsClassifier(n_neighbors = each)

    knn3.fit(x_train,y_train)

    score_list.append(knn3.score(x_test,y_test))

    

plt.plot(range(1,50),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(x_train,y_train)

print("svm score: {} ".format(svm.score(x_test,y_test)))
y_pred = svm.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

#visualization

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
#K fold CV K = 10

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(svm,x,y, cv = 10)

print(accuracies)

print("average accuracy: ",np.mean(accuracies))

print("average std: ",np.std(accuracies))

#devam
from sklearn.model_selection import cross_val_score

svm2 = SVC(C=1,gamma=1)

accuracies = cross_val_score(svm2,x,y, cv = 10)

print(accuracies)

print("average accuracy: ",np.mean(accuracies))

print("average std: ",np.std(accuracies))
#Naive bayes 

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("accuracy of naive bayes algo: ",nb.score(x_test,y_test))
y_pred = nb.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

#visualization

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
#K fold CV K = 10

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(nb,x,y, cv = 10)

print(accuracies)

print("average accuracy: ",np.mean(accuracies))

print("average std: ",np.std(accuracies))

#devam
from sklearn.model_selection import GridSearchCV

grid = {'var_smoothing': np.logspace(0,-9, num=100)} 

nb = GaussianNB()

nb_cv = GridSearchCV(nb,grid,cv = 10)

nb_cv.fit(x_train,y_train)

print("tuned hyperparameters: (best parameters): ",nb_cv.best_params_)

print("accuracy: ",nb_cv.best_score_)
nb2 = GaussianNB(var_smoothing=0.0023101297000831605)

nb2.fit(x_train,y_train)

print("score: ", nb2.score(x_test,y_test))
y_pred = nb2.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

#visualization

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
#K fold CV K = 10

from sklearn.model_selection import cross_val_score

nb2 = GaussianNB(var_smoothing=0.0023101297000831605)

accuracies = cross_val_score(nb2,x,y, cv = 10)

print(accuracies)

print("average accuracy: ",np.mean(accuracies))

print("average std: ",np.std(accuracies))

#devam
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

print("score: ", dt.score(x_test,y_test))
y_pred = dt.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

#visualization

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
#K fold CV K = 10

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(dt,x,y, cv = 10)

print(accuracies)

print("average accuracy: ",np.mean(accuracies))

print("average std: ",np.std(accuracies))

#devam
from sklearn.model_selection import GridSearchCV

grid = {'criterion': ['gini', 'entropy'],

             'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],

             'min_samples_split': [2, 3]}

dt = DecisionTreeClassifier()

dt_cv = GridSearchCV(dt,grid,cv = 10)

dt_cv.fit(x_train,y_train)

print("tuned hyperparameters: (best parameters): ",dt_cv.best_params_)

print("accuracy: ",dt_cv.best_score_)

from sklearn.model_selection import cross_val_score

dt2 = DecisionTreeClassifier(criterion='gini', max_depth= 8, min_samples_split= 2)

accuracies = cross_val_score(dt2,x,y, cv = 10)

print(accuracies)

print("average accuracy: ",np.mean(accuracies))

print("average std: ",np.std(accuracies))

#devam
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state = 1)

rf.fit(x_train,y_train)

print("random forest algo result: ",rf.score(x_test,y_test))
y_pred = rf.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

#visualization

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
#K fold CV K = 10

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(rf,x,y, cv = 10)

print(accuracies)

print("average accuracy: ",np.mean(accuracies))

print("average std: ",np.std(accuracies))

#devam
#K fold CV K = 10

from sklearn.model_selection import cross_val_score

rf2 = RandomForestClassifier(bootstrap= True, criterion='gini', n_estimators= 100)

accuracies = cross_val_score(rf2,x,y, cv = 10)

print(accuracies)

print("average accuracy: ",np.mean(accuracies))

print("average std: ",np.std(accuracies))

#devam
from sklearn.decomposition import PCA

pca = PCA(n_components = 2, whiten= True )  # whitten = normalize

pca.fit(df3)



df3_pca = pca.transform(df3)



print("variance ratio: ", pca.explained_variance_ratio_)



print("sum: ",sum(pca.explained_variance_ratio_))