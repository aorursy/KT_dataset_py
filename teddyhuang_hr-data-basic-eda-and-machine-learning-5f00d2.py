# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# import the file

Data = pd.read_csv('../input/HR_comma_sep.csv')
Data.info() # information about data
columns= Data.columns.tolist() # Make a list of name of columns

columns
import seaborn as sns # for Interactive plots

import matplotlib.pyplot as plt # for plots

% matplotlib inline

import itertools
#let count no of of different categories in catgorical variables

#categorical variables are left, promotion_last_5years,sal,salary,work_accident

#more than two or three categories are number_project,time spend_company

categorical=['number_project','time_spend_company','Work_accident','left', 'promotion_last_5years','sales','salary']

fig=plt.subplots(figsize=(10,15))

length=len(categorical)

for i,j in itertools.zip_longest(categorical,range(length)): # itertools.zip_longest for to execute the longest loop

    plt.subplot(np.ceil(length/2),2,j+1)

    plt.subplots_adjust(hspace=.5)

    sns.countplot(x=i,data = Data)

    plt.xticks(rotation=90)

    plt.title("No. of employee")
# no of employee who left the company 

print("The Number of employee who left the company :",len(Data[Data['left']==1]))

print("The Number of employee who didn't left the company",len(Data[Data['left']==0]))

print("The proportion of employee who left",len(Data[Data['left']==1])/len(Data))
####let's Analysis the Categorical and ordinal Variable
# here we will do it only for categorical variable

categorical=['number_project','time_spend_company','Work_accident','promotion_last_5years','sales','salary'] # here I have removed left to see who is leaving cpmpany

fig=plt.subplots(figsize=(12,15))# to define the size of figure

length=len(categorical) # no of categorical and ordinal variable

for i,j in itertools.zip_longest(categorical,range(length)): # itertools.zip_longest for to execute the longest loop

    plt.subplot(np.ceil(length/2),2,j+1) # this is to plot the subplots like as 2,2,1 it means 2x2 matrix and graph at 1 

    plt.subplots_adjust(hspace=.5) # to adjust the distance between subplots

    sns.countplot(x=i,data = Data,hue="left") # To plot the countplot of variable with hue left

    plt.xticks(rotation=90) # to rotate the xticks by 90 such that no xtixks overlap each other

    plt.title("No.of employee who left") # to plot the title of graph
# Lets Calcualte proportion for above same

fig=plt.subplots(figsize=(12,15))

for i,j in itertools.zip_longest(categorical,range(length)):# itertools.zip_longest for to execute the longest loop

    Proportion_of_data = Data.groupby([i])['left'].agg(lambda x: (x==1).sum()).reset_index()# only counting the number who left 

    Proportion_of_data1=Data.groupby([i])['left'].count().reset_index() # Counting the total number 

    Proportion_of_data2 = pd.merge(Proportion_of_data,Proportion_of_data1,on=i) # mergeing two data frames

    # Now we will calculate the % of employee who left category wise

    Proportion_of_data2["Proportion"]=(Proportion_of_data2['left_x']/Proportion_of_data2['left_y'])*100 

    Proportion_of_data2=Proportion_of_data2.sort_values(by="Proportion",ascending=False).reset_index(drop=True)#sorting by percentage

    plt.subplot(np.ceil(length/2),2,j+1)

    plt.subplots_adjust(hspace=.5)

    sns.barplot(x=i,y='Proportion',data=Proportion_of_data2)

    plt.xticks(rotation=90)

    plt.title("percentage of employee who left")

    plt.ylabel('Percentage')
# Let see who is getting Promotions proportion wise.

fig=plt.subplots(figsize=(12,15))

categorical=['number_project','time_spend_company','sales','salary']

length=len(categorical)

for i,j in itertools.zip_longest(categorical,range(length)):# itertools.zip_longest for to execute the longest loop

    Proportion_of_data = Data.groupby([i])['promotion_last_5years'].agg(lambda x: (x==1).sum()).reset_index()# only counting the number who left 

    Proportion_of_data1=Data.groupby([i])['promotion_last_5years'].count().reset_index() # Counting the total number 

    Proportion_of_data2 = pd.merge(Proportion_of_data,Proportion_of_data1,on=i) # mergeing two data frames

    # Now we will calculate the % of employee who  category wise

    Proportion_of_data2["Proportion"]=(Proportion_of_data2['promotion_last_5years_x']/Proportion_of_data2['promotion_last_5years_y'])*100 

    Proportion_of_data2=Proportion_of_data2.sort_values(by="Proportion",ascending=False).reset_index(drop=True)#sorting by percentage

    plt.subplot(np.ceil(length/2),2,j+1)

    plt.subplots_adjust(hspace=.3)

    sns.barplot(x=i,y='Proportion',data=Proportion_of_data2)

    plt.xticks(rotation=90)

    plt.title("Pecentage of employee getting promotion")

    plt.ylabel('Percentage')

# I want to have a look at the which department is accident prone



Proportion_of_data = Data.groupby(["sales"])["Work_accident"].agg(lambda x: (x==1).sum()).reset_index()# only counting the number who left 

Proportion_of_data1=Data.groupby(["sales"])["Work_accident"].count().reset_index() # Counting the total number 

Proportion_of_data2 = pd.merge(Proportion_of_data,Proportion_of_data1,on='sales') # mergeing two data frames

# Now we will calculate the % of employee who  category wise

Proportion_of_data2["Proportion"]=(Proportion_of_data2['Work_accident_x']/Proportion_of_data2['Work_accident_y'])*100 

Proportion_of_data2=Proportion_of_data2.sort_values(by="Proportion",ascending=False).reset_index(drop=True)#sorting by percentage

sns.barplot(x='sales',y='Proportion',data=Proportion_of_data2)

plt.xticks(rotation=90)

plt.title('Department wise accident')

plt.ylabel('Percentage')



continues_variable=['satisfaction_level','last_evaluation','average_montly_hours']

categorical_variable=['promotion_last_5years','sales','salary','left','time_spend_company','number_project']

Data['Impact']=(Data['number_project']/Data["average_montly_hours"])*100
def pointplot(x, y, **kwargs): # making a function to plot point plot

    sns.pointplot(x=x, y=y)      

    x=plt.xticks(rotation=90)
# Start with Satisfaction Level

categorical_variable=['promotion_last_5years','sales','salary','left','time_spend_company','number_project']

f = pd.melt(Data, id_vars=["satisfaction_level"], value_vars=categorical_variable)

g=sns.FacetGrid(f,col='variable',col_wrap=2,sharex=False,sharey=False,size=5)



g.map(pointplot,"value","satisfaction_level")
# Now with last evaluation

f = pd.melt(Data, id_vars=['last_evaluation'], value_vars=categorical_variable)

g=sns.FacetGrid(f,col='variable',col_wrap=2,sharex=False,sharey=False,size=5)



g.map(pointplot,"value",'last_evaluation')
# Now let's explore the variable average_monthly_hours

# With a new type of plot violin plot

Categorical_variable= ['sales','salary','time_spend_company','number_project']

fig=plt.subplots(figsize=(12,15))

length=len(Categorical_variable)

for i,j in itertools.zip_longest(Categorical_variable,range(length)): # itertools.zip_longest for to execute the longest loop

    plt.subplot(np.ceil(length/2),2,j+1) # here j repersent the number of graphs like as subplot 221

    plt.subplots_adjust(hspace=.5) # to get the space between subplots

    sns.violinplot(x=i,y="average_montly_hours", data = Data,hue="left",split=True,scale="count")#here count scales the width of violin

    plt.xticks(rotation=90)
# Now we have to predict who will left the company beofore going ahead lets do a part of feature engineering selecting import



# Let's plot the correlation Matrix

Data.drop('Impact',axis=1,inplace=True)

corr= Data.corr()

plt.figure(figsize=(12,10))

sns.heatmap(corr,annot=True,cbar=True,cmap="coolwarm")

plt.xticks(rotation=90)
from sklearn.preprocessing import LabelEncoder # For change categorical variable into int

from sklearn.metrics import accuracy_score 

le=LabelEncoder()

Data['salary']=le.fit_transform(Data['salary'])

Data['sales']=le.fit_transform(Data['sales'])

# we can select importance features by using Randomforest Classifier

from sklearn.ensemble import RandomForestClassifier 

model= RandomForestClassifier(n_estimators=100)

feature_var = Data.ix[:,Data.columns != "left"]

pred_var = Data.ix[:,Data.columns=='left']

model.fit(feature_var,pred_var.values.ravel())
featimp = pd.Series(model.feature_importances_,index=feature_var.columns).sort_values(ascending=False)

print(featimp)
# Importing Machine learning models library used for classification

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.cross_validation import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier as knn

from sklearn.naive_bayes import GaussianNB as GB

from sklearn.svm import SVC
def Classification_model(model,Data,x,y): # here x is the variable which are used for prediction

    # y is the prediction variable

    train,test = train_test_split(Data,test_size= 0.33)

    train_x = Data.ix[train.index,x] # Data for training only with features

    train_y = Data.ix[train.index,y] # Data for training only with predcition variable

    test_x = Data.ix[test.index,x] # same as for training 

    test_y = Data.ix[test.index,y]

    model.fit(train_x,train_y.values.ravel())

    pred=model.predict(test_x)

    accuracy=accuracy_score(test_y,pred)

    return accuracy
All_features=['satisfaction_level',

'number_project',

'time_spend_company',

'average_montly_hours',

'last_evaluation',

'sales',

'salary',

'Work_accident',       

'promotion_last_5years']

print(All_features)

Important_features = ['satisfaction_level',

'number_project',

'time_spend_company',

'average_montly_hours',

'last_evaluation']

print(Important_features)

Pred_var = ["left"]

print(Pred_var)
# Lets us make a list of models

models=["RandomForestClassifier","Gaussian Naive Bays","KNN","Logistic_Regression","Support_Vector"]

Classification_models = [RandomForestClassifier(n_estimators=100),GB(),knn(n_neighbors=7),LogisticRegression(),SVC()]

Model_Accuracy = []

for model in Classification_models:

    Accuracy=Classification_model(model,Data,All_features,Pred_var)

    Model_Accuracy.append(Accuracy)
Accuracy_with_all_features = pd.DataFrame(

    { "Classification Model" :models,

     "Accuracy with all features":Model_Accuracy

     

    })
Accuracy_with_all_features.sort_values(by="Accuracy with all features",ascending=False).reset_index(drop=True)
# Lets try with Important features

Model_Accuracy = []

for model in Classification_models:

    Accuracy=Classification_model(model,Data,Important_features,Pred_var) # Just instead of all features give only important features

    Model_Accuracy.append(Accuracy)

Accuracy_with_important_features = pd.DataFrame(

    { "Classification Model" :models,

     "Accuracy with Important features":Model_Accuracy

     

    })

Accuracy_with_important_features.sort_values(by="Accuracy with Important features",ascending=False).reset_index(drop=True)
from sklearn.model_selection import cross_val_score # This is used for to caculate the score of cross validation by using Kfold

def Classification_model_CV(model,Data,x,y): # here x is the variable which are used for prediction

    # y is the prediction variable

    data_x = Data.ix[:,x] # Here no need of training and test data because in cross validation it splits data into 

    

    # train and test itself # data_x repersent features

    data_y = Data.ix[:,y] # data for predication

    data_y=data_y.values.ravel()

    scores= cross_val_score(model,data_x,data_y,scoring="accuracy",cv=10)

    print(scores) # print the scores

    print('')

    accuracy=scores.mean()

    return accuracy
models=["RandomForestClassifier","Gaussian Naive Bays","KNN","Logistic_Regression","Support_Vector"]

Classification_models = [RandomForestClassifier(n_estimators=100),GB(),knn(n_neighbors=7),LogisticRegression(),SVC()]

Model_Accuracy = []

for model,z in zip(Classification_models,models):

    print(z) # Print the name of model

    print('')

    Accuracy=Classification_model_CV(model,Data,Important_features,Pred_var)

    

    Model_Accuracy.append(Accuracy)
Accuracy_with_CV = pd.DataFrame(

    { "Classification Model" :models,

     "Accuracy with CV":Model_Accuracy

     

    })

Accuracy_with_CV.sort_values(by="Accuracy with CV",ascending=False).reset_index(drop=True)
from sklearn.model_selection import GridSearchCV 

def Classification_model_GridSearchCV(model,Data,x,y,params):

    

    # here params repersent Parameters

    data_x = Data.ix[:,x]  

    data_y = Data.ix[:,y] 

    data_y=data_y.values.ravel()

    clf = GridSearchCV(model,params,scoring="accuracy",cv=5)

    clf.fit(data_x,data_y)

    print("best score is :")

    print(clf.best_score_)

    print('')

    print("best estimator is :")

    print(clf.best_estimator_)



    return (clf.best_score_)
models=["RandomForestClassifier","Gaussian Naive Bays","KNN","Logistic_Regression","Support_Vector"]

Model_Accuracy=[]

model = RandomForestClassifier()

param_grid = {'n_estimators':(70,80,90,100),'criterion':('gini','entropy'),'max_depth':[25,30]}

Accuracy=Classification_model_GridSearchCV(model,Data,Important_features,Pred_var,param_grid)

Model_Accuracy.append(Accuracy)
model = GB()

param_grid={}

Accuracy=Classification_model_GridSearchCV(model,Data,Important_features,Pred_var,param_grid)

Model_Accuracy.append(Accuracy)
model=knn()

param_grid={'n_neighbors':[5,15],'weights':('uniform','distance'),'p':[1,5]}

Accuracy=Classification_model_GridSearchCV(model,Data,Important_features,Pred_var,param_grid)

Model_Accuracy.append(Accuracy)
model=LogisticRegression()

param_grid={'C': [0.01,0.1,1,10],'penalty':('l1','l2')}

Accuracy=Classification_model_GridSearchCV(model,Data,Important_features,Pred_var,param_grid)

Model_Accuracy.append(Accuracy)
model=SVC()

param_grid={'C': [1,10,20,100],'gamma':[0.1,1,10]} 

Accuracy=Classification_model_GridSearchCV(model,Data,Important_features,Pred_var,param_grid)

Model_Accuracy.append(Accuracy)
Accuracy_with_GridSearchCV = pd.DataFrame(

    { "Classification Model" :models,

     "Accuracy with GridSearchCV":Model_Accuracy

     

    })

Accuracy_with_GridSearchCV.sort_values(by="Accuracy with GridSearchCV",ascending=False).reset_index(drop=True)
Comparison=pd.merge(pd.merge(pd.merge(Accuracy_with_all_features,Accuracy_with_important_features,on='Classification Model'),Accuracy_with_CV,on='Classification Model'),Accuracy_with_GridSearchCV,on='Classification Model')
Comparison1=Comparison.ix[:,["Classification Model","Accuracy with all features","Accuracy with Important features","Accuracy with CV","Accuracy with GridSearchCV"]]
Comparison1