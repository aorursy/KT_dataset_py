### INTRODUCTION ###



# The problem here is not to build a classifier but the real task is to Extract useful information.

# In the Begining some columns may seem useless but when you Visualize the data You will notice

# that there is a lot of Impotant Content Hiding in Plane Site.



# I am Here to Guide you through My Kernel and help you understand how and why I did what I did



#### NOTE: THE RESULT MAY VARY DUE TO THE RANDOMNESS INTRODUCED IN THE CODE BELOW ####



# The code is Divied into 5 Phases:

# 1st Phase is to Import and preprocess Data

# 2nd Phase is to encode Data and dropiing unwanted Colomns

# 3rd Phase is to Fit the data in classifier 

# 4th Phase is to Visualize the Result

# 5th and the final Phase is to Build the Csv file for submition



########## 1St Phase ##########



# First Thing to Due is to Import Some Basic libraries that will help us to process data



# Numpy is one of the most essential Libraries as it provides us with the Numpy array support

import numpy as np



# matplotlib.pyplot and Seaborn is used to Visualize the data

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# pandas is used to import data to our code

import pandas as pd



import warnings

warnings.filterwarnings('ignore')
# Let us first import the dataset to our code

dataset_train = pd.read_csv('../input/train.csv')

dataset_test = pd.read_csv('../input/test.csv')

dataset_gd = dataset_test



# Now we have imported both the test set and the train set

# We will then concactinated the 2 so as to increase the accuracy of the data preprocessing



dataset_train = pd.concat([dataset_test,dataset_train],axis=0)
dataset_train.info()
# From the above Information we can conclude that there are columns with null values

# Hence the 1st things we need to do is to handle these null values

# The columns with nulls values clearly are (Cabin , Survived, Embarked, Age)



# HANDLING THE NULL VALUES



#FARE

# Since there is only one value missing in this Column we would simply replace it by the column's mean

dataset_train['Fare'].fillna(dataset_train.Fare.mean(),inplace=True)



#AGE

# This is the randomness that I was talking about

# What I have done is that i have replaced the missing values between 

# mean + standard diviation and mean - standard diviation



age_mean = dataset_train.Age.mean()

age_std = dataset_train.Age.std()

dataset_train['Age'][dataset_train.Age.isnull()] = np.random.randint(high=age_mean+age_std,low=age_mean-age_std,size=len(dataset_train['Age'][dataset_train.Age.isnull()]))



#EMBARKED

# You have to understand that 'nan' means undefined and hence I have created a new category 

# for the missing values in the embarked column and replaced it with 'n'

dataset_train.Embarked.fillna('n',inplace=True)





#CABIN

# As you can see that most of the values in the column is null hence will not replace these values

# I will drop this column in the future after extarcting some useful information
# Extracting useful Information Form the data provied 



# Extracting Info from Cabin





# Def_Cabin

# Here we create a column which would tell us that whether the cabin is defined for a customer or not

dataset_train['def_Cabin'] = dataset_train.Cabin.notnull().astype(int)



sns.barplot(data=dataset_train, x='def_Cabin', y='Survived')

plt.show()

# From the graph below we can notice a clear Pattern and hence we will keep this Column
# NAME



# Name Length

# I Can't really Explain why but name length has a clear pattern and seems to significantly improve the

# result and hence I am obliged to use it in my Code

dataset_train['Name_Length'] = dataset_train['Name'].apply(lambda x : len(x))

dataset_train['Name_Length'] = ((dataset_train.Name_Length)/15).astype(np.int64)+1

print(dataset_train[['Name_Length','Survived']].groupby(['Name_Length'], as_index = False).mean())

plt.subplots(figsize=(15, 6))

sns.barplot(data=dataset_train,x='Name_Length',y='Survived')



# You may now think that Name is a useless column but Name contains somethings very important,'Titles'

# If You observe closely you will notice that all names have a Title, example : 'MR','Mrs','Cpt',etc



# EXTRACTING TITLE FORM NAME

title = dataset_train.Name.values

import re

for i in range(len(title)):

    r = re.search(', ([A-Za-z ]*)',title[i])

    title[i] = r.group(1)

dataset_train.loc[:,'Name'] = title 

plt.subplots(figsize=(15, 6))

sns.barplot(data=dataset_train,x='Name',y='Survived')

# Hence from the figure below show that it may play an important role in the decision making process
# EXTRACTING INFO FROM TICKETS

# Now Ticket here is unique for each cutomer but there are a few things 

# that are common and contains usefull information like:



# TICKET LENGTH

# Now You may think why ticket_length, You see this may indicated the type of Ticket one has

dataset_train['Ticket_Length'] = dataset_train['Ticket'].apply(lambda x : len(x))

plt.subplots(figsize=(15, 6))

sns.barplot(data=dataset_train,x='Ticket_Length',y='Survived')
# TICKET lETTERS

# This will tell us the category of the ticket a customer has

dataset_train['Ticket_Initials'] = dataset_train['Ticket'].apply(lambda x : str(x)[0]) 

dataset_train['Ticket_Initials'] = dataset_train['Ticket_Initials'].apply(lambda x : re.sub('[0-9]','N',x))

plt.subplots(figsize=(15, 6))

sns.barplot(data=dataset_train,x='Ticket_Initials',y='Survived')

# A Clear Pattern in the Graph below
# COMBINING THE SIBSP AND PARCH INTO FAMILY

# We can Combine 2 Columns from our dataset to reduced the complexicity

family = dataset_train.SibSp + dataset_train.Parch+1

dataset_train['Family'] = family

plt.subplots(figsize=(15, 6))

sns.barplot(data=dataset_train,x='Family',y='Survived')

# Similarly this show a clear pattern
# CREATING THE COLUMN IS_ALONE

# The second piece of information we could extract is whether a person is travelling Alone or not

dataset_train['Is_Alone'] = dataset_train['Family'].apply(lambda x : 1 if x>1 else 0)

plt.subplots(figsize=(15, 6))

sns.barplot(data=dataset_train,x='Is_Alone',y='Survived')
# DIVIDING THE FARE INTO GROUPS

# We Cannot Use Fare in the Current format as the range on the Fare is Quite High and

# hence may dominate the decision. To avoid This We would Divide it into groups

# This may not be the best way to divide but is simple to understand and implement

dataset_train['Fare_Group'] = (dataset_train['Fare']/25).values.astype(np.int64)

plt.subplots(figsize=(15, 6))

sns.barplot(data=dataset_train,x='Fare_Group',y='Survived')
# DIVIDING AGE INTO AGE GROUPS

# Similarly, I will also divide age into groups using the same process

dataset_train['Age_Group'] = ((dataset_train['Age']/15)+1).astype(np.int64)

plt.subplots(figsize=(15, 6))

sns.barplot(data=dataset_train,x='Age_Group',y='Survived')

# It is a clear indication that kids and elderly are likely to survive  


# Now We have completed Phase 1 of the Code that is Preprocesing Infromation

# the 2nd Phase of the code is to encode the data and and dropiing unwanted columns



# Encoding String values to Numbers

from sklearn.preprocessing import LabelEncoder



#SEX

lb_Sex = LabelEncoder()

dataset_train['Sex'] = lb_Sex.fit_transform(dataset_train.Sex)



#EMBARKED

lb_Emb = LabelEncoder()

dataset_train['Embarked'] = lb_Emb.fit_transform(dataset_train.Embarked)



#TITLE

lb_Title = LabelEncoder()

dataset_train['Name'] = lb_Title.fit_transform(dataset_train.Name)



#TICKET_INITIAL

lb_Ticket_init = LabelEncoder()

dataset_train['Ticket_Initials'] = lb_Ticket_init.fit_transform(dataset_train.Ticket_Initials)



# DROPPING THE EXTRA COLUMNS

dataset_train.drop(labels=['SibSp','Parch','Ticket','Fare','Age','PassengerId','Cabin'],axis=1,inplace=True)



dataset_train.head()
# This marks the Completion of the 2nd Phase and the Start of 3rd Phase

# This phase is about preparing the data for our classifiers



#RETREIVING THE TEST AND THE TRAIN SETS

dataset_test = dataset_train[dataset_train.Survived.isnull()]

dataset_train = dataset_train[dataset_train.Survived.notnull()]



dataset_test = dataset_test.drop(['Survived'],axis=1)
#DIVIDING THE DATA INTO Y_TRAIN AND X_TRAIN AND CONVERTING THEM INTO NP ARRAYS

y_train = dataset_train.loc[:,'Survived'].values

x_train =dataset_train.drop(['Survived'],axis=1).values

x_test = dataset_test.values
# Feature Scaling 

from sklearn.preprocessing import MinMaxScaler

sc_x = MinMaxScaler((-1,1))

x_train  = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)
########## Phase 3 ##########

# In this Phase we would simply fitting the data into our classifier and and apply k-Fold Validation



# Confusion Matrix

from sklearn.metrics import confusion_matrix

dict_K = {}

dic = {}



#Kfold Validation

def get_acc(Xtrain,Ytrain,model):

    from sklearn.model_selection import KFold

    acc = []

    k=KFold(n_splits=4)

    for train , test in k.split(Xtrain,y=Ytrain):

        x_train = Xtrain[train,:]

        y_train = Ytrain[train]

        x_test = Xtrain[test,:]

        y_test = Ytrain[test]

        model.fit(x_train,y_train)

        y_pred = model.predict(x_test)

        cm = confusion_matrix(y_true=y_test,y_pred=y_pred)

        acc.append((cm[1,1]+cm[0,0])/((cm[1,0]+cm[0,1]+cm[1,1]+cm[0,0])+1e-5))

    return acc

  
# Fitting Decision Tree Classification to the Training set

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

dict_K['Decision'] = get_acc(x_train,y_train,classifier)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
# Fitting K-NN to the Training set

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=50,algorithm='auto')

dict_K['KNN'] = get_acc(x_train,y_train,classifier)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
# Fitting Kernel SVM to the Training set

from sklearn.svm import SVC

classifier = SVC(kernel='rbf')

dict_K['kernel-SVM'] = get_acc(x_train,y_train,classifier)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(solver='newton-cg')

dict_K['Logistic'] = get_acc(x_train,y_train,classifier)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

dict_K['Naive'] = get_acc(x_train,y_train,classifier)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=25,criterion='entropy')

dict_K['Random_forest'] = get_acc(x_train,y_train,classifier)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
########## Phase 4 ##########

# This is about visualizing the result of the K-Fold Validation process



#VISUALISING THE RESULTS

df_k = pd.DataFrame(dict_K)

s =df_k.plot(figsize=(10,10),linewidth=5.0)

plt.show()
#Calculating the Mean of the k_fold validation

df_k.mean()
########## 5th phase ##########

# This Final Phase is about preparing the csv file for Submition

# Preparing the CSV For Submition

p = dataset_gd.PassengerId

p = pd.concat([p,pd.DataFrame(y_pred.astype(np.int64),columns=['Survived'])],axis=1)

p.to_csv('Tit_pred.csv',index=False)