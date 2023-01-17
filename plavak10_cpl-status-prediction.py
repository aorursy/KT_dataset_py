import numpy as np

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#importing libraries

import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt



from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC





#Getting the data

train_data_path=("../input/cpl-prediction/Train.xlsx")

test_data_path=("../input/cpl-prediction/Test.xlsx")

train_data=pd.read_excel(train_data_path)

test_data=pd.read_excel(test_data_path)

train_data.head()
#As we have obtained the data,let's get more insights of it by performing some basic EDA

train_data.info()
#We can see there are 614 examples

#Number of features = 20(incl the target variable 'CPL_Status')

train_data.describe()
#Missing values info

missing = train_data.isnull().sum().sort_values(ascending=False)

missing_percent = train_data.isnull().sum()/train_data.isnull().count()*100

sorted_missing_percent = (round(missing_percent, 2)).sort_values(ascending=False)

missing_data = pd.concat([missing,sorted_missing_percent],axis=1,keys=['Total Missing', '%Missing'])

missing_data.head(20)
#We see that 7 features have missing values

#As for any missing feature, there isn't a significant percentage(max is 8.14% for Credit History), it's not wise to drop any 'consider worth' column

#Everything except 'first_name','last_name','email','address','AGT_ID','INT_ID','Prev_ID','Loanapp_ID'

#makes sense to contribute to 'CPL_Status'



#converting target variable to numeric form

cpl_status = {"N": 0, "Y": 1}

train_data['CPL_Status'] = train_data['CPL_Status'].map(cpl_status)

train_data['CPL_Status'] = train_data['CPL_Status'].astype(int)
gender=[]

cpl=[]

for i in range(len(train_data)):

    if train_data.Sex[i]=="M" or train_data.Sex[i]=="F":

        gender.append(train_data.Sex[i])

        cpl.append(train_data.CPL_Status[i])

df=pd.DataFrame({"Sex":gender,"CPL_Status":cpl})

male_approve=0

male_disapprove=0

female_approve=0

female_disapprove=0



for i in range(len(df)):

    if df.Sex[i]=="M" and df.CPL_Status[i]==1:

        male_approve+=1

    if df.Sex[i]=="M" and df.CPL_Status[i]==0:

        male_disapprove+=1

    if df.Sex[i]=="F" and df.CPL_Status[i]==1:

        female_approve+=1

    if df.Sex[i]=="F" and df.CPL_Status[i]==0:

        female_disapprove+=1

data = [[male_approve,female_approve],

  [male_disapprove,female_disapprove]]

plt.figure(figsize=(10,5))

X = np.arange(2)

plt.bar(X+0.0,data[0], color = 'red', width = 0.25)

plt.bar(X+0.25,data[1], color = 'blue', width = 0.25)

plt.xticks([0.125,1.125],['Men','Women'],fontsize=13)

plt.legend('YN')

plt.title('Loan approval status',fontsize=15)

plt.show()



#we see that number of women applicants present is significantly less compared to men

#Also, approx. one-third of applications for CPL seem to be rejected(roughly from the figure)
plt.figure(figsize=(8,5))

sns.barplot(x='Marital_Status', y='CPL_Status', data=train_data)

plt.xlabel('Marital Status',fontsize=15)

plt.ylabel('CPL Status',fontsize=15)



#We see that persons who are married tend to have a slightly higher chances of getting loan approved
train_data.Dependents.unique()
axes = sns.factorplot('Dependents','CPL_Status',data=train_data, aspect = 2.5, )

plt.ylabel('CPL Status',fontsize=15)

plt.xlabel('Number of Dependents',fontsize=15)



#We infer that persons with number of dependents = 2 tend to have a higher chance of loan approval. 

#Other dependents cases have a fair enough chance as well
train_data.Qual_var.unique()
plt.figure(figsize=(8,5))

sns.barplot(x='Qual_var',y='CPL_Status',data=train_data)

plt.xlabel('Qualification',fontsize=15)

plt.ylabel('CPL Status',fontsize=15)



#We see that graduates have a slightly more chance of getting their loan approved that their non-graduate counterparts
plt.figure(figsize=(8,5))

sns.barplot(x='Prop_Area',y='CPL_Status',data=train_data)

plt.xlabel('Area',fontsize=15)

plt.ylabel('CPL Status',fontsize=15)



#We see that people living in semi urban regions have a higher tendency to get loan approved
train_data['SE'].value_counts()
plt.figure(figsize=(8,5))

sns.barplot(x='SE',y='CPL_Status',data=train_data)

plt.xlabel('Self Employed',fontsize=15)

plt.ylabel('CPL Status',fontsize=15)



#Chances of getting a loan in case of Self-Employed status seems unbiased
train_data['Credit_His'].value_counts()
plt.figure(figsize=(8,5))

sns.barplot(x='Credit_His',y='CPL_Status',data=train_data)

plt.xlabel('Credit History',fontsize=15)

plt.xticks([0,1],['No','Yes'],fontsize=13)

plt.ylabel('CPL Status',fontsize=15)



#We can see that a person with no credit history stands with a little chance of loan approval

#Persons with credit history have a high probability of getting loan approval
#We add a new column 'Loan_by_income' which is the person's loan amount divided by his/her income



datasets = [train_data, test_data]

for data in datasets:

    data['Loan_by_income'] = data['CPL_Amount']/data['App_Income_1']

    data['Loan_by_income'] = data['Loan_by_income'].astype(float)

    data['Loan_by_income'] = round(data['Loan_by_income'],2)

    
train_data.describe()

#We see that the maximum value of 'Loan_by_income' is 1.13 and minimum is 0.
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = train_data[train_data['Sex']=='F']

men = train_data[train_data['Sex']=='M']

ax = sns.distplot(women[women['CPL_Status']==1].Loan_by_income.dropna(),bins=18,label = 'Approved', ax = axes[0], kde =False)

ax = sns.distplot(women[women['CPL_Status']==0].Loan_by_income.dropna(),bins=30,label = 'Not Approved', ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['CPL_Status']==1].Loan_by_income.dropna(),bins=18,label = 'Approved', ax = axes[1], kde = False)

ax = sns.distplot(men[men['CPL_Status']==0].Loan_by_income.dropna(),bins=40,label = 'Not Approved', ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')





#We infer that most of the data for 'Loan_by_income' lie within 0.2. 

#For males and females both, 'Loan_by_income' with value less than 0.1 tends to have a higher chance of loan approval
train_data['CPL_Term'].value_counts()
plt.figure(figsize=(8,5))

sns.barplot(x='CPL_Term',y='CPL_Status',data=train_data)

plt.xlabel('CPL Term(Days)',fontsize=15)

plt.ylabel('CPL Status',fontsize=15)
train_data_2=train_data.copy()

datasets=[train_data_2]

for data in datasets:

    data.loc[data['App_Income_2']>0,'Income_2'] = 1

    data.loc[data['App_Income_2']==0,'Income_2'] = 0

    data['Income_2']=data['Income_2'].astype(int)
plt.figure(figsize=(8,5))

sns.barplot(x='Income_2',y='CPL_Status',data=train_data_2)

plt.xlabel('Applicant Income 2 ( 0 = No, 1 = Yes)',fontsize=15)

plt.ylabel('CPL Status',fontsize=15)

plt.title("Income 2 and CPL Status")





#We see that applicants having another person's income taken into consideration has a higher chance of getting loan approval
#Now we need to deal with missing values



#Summary of missing values:

#Credit_His = 50

#SE = 32

#Dependents = 15

#CPL_Term = 14

#Sex = 13

#Marital_Status = 3

#CPL_Amount = 2
#We observed that there are 512 entries for 'CPL_Term' with value 360

#So we can fill the missing 14 values with the mode,i.e 360

mode=360

datasets=[train_data,test_data]

for data in datasets:

    data['CPL_Term'] = data['CPL_Term'].fillna(mode)
#For the two missing values of 'CPL_Amount' we can use the mean value to fill them



mean_val = round(train_data.CPL_Amount.mean(),1)

for data in datasets:

    data['CPL_Amount'] = data['CPL_Amount'].fillna(mean_val)

    data['CPL_Amount'] = data['CPL_Amount'].astype(float)
train_data['Marital_Status'].value_counts()
#We also fill the missing values for 'Marital_Status' with the mode,i.e Y

mode="Y"

for data in datasets:

    data['Marital_Status'] = data['Marital_Status'].fillna(mode)

    
names=[]

for i in range(len(train_data)):

    if train_data.Sex[i]!="M" and train_data.Sex[i]!="F":

        names.append(train_data.first_name[i])

names        

        
#For filling the missing values in 'Sex' column, we can refer

#https://www.gpeters.com/names/baby-names.php?

#which is a website to guess gender by names



#The following results were obtained:

#Crosby,Ludovico,Rudie,Conant,Byrom,Wendell and Robby were classified as male

#Sarine,Vina,Clemmie,Francesca were classified as female

#Heall couldn't be classified(so we assume it as a mode value,i.e male)



gender=['M','M','M','M','M','M','F','M','F','M','M','F','F']

j=0

for i in range(len(train_data)):

    if train_data.Sex[i]!="M" and train_data.Sex[i]!="F":

        train_data.Sex[i]=gender[j]

        j+=1
#Repeating the same process for test data

names=[]

for i in range(len(test_data)):

    if test_data.Sex[i]!="M" and test_data.Sex[i]!="F":

        names.append(test_data.first_name[i])

names    
gender=['F','M','F','M','F','F','F','F','M','M','M']

j=0

for i in range(len(test_data)):

    if test_data.Sex[i]!="M" and test_data.Sex[i]!="F":

        test_data.Sex[i]=gender[j]

        j+=1
train_data['SE'].value_counts()
#We see that there are 32 missing values in 'SE' column

#Also,500 have value "N" and 82 have value "Y"

#So, 86% of the non-missing values are "N" and rest 14% are "Y"

#What we can do is, generate random numbers b/w 1 to 100 and each time we get more than 86

#we can fill the missing value with "Y" and if less than 86,

#we can fill the missing value with "N",each time we iterate along the missing values.

#(Basically a weighted random)



import random

for i in range(len(train_data)):

    if train_data.SE[i]!="Y" and train_data.SE[i]!="N":

        if random.randint(1,100) > 86:

            train_data.SE[i] = "Y"

        else:

            train_data.SE[i] = "N"
#Applying same for test data



for i in range(len(test_data)):

    if test_data.SE[i]!="Y" and test_data.SE[i]!="N":

        if random.randint(1,100) > 86:

            test_data.SE[i] = "Y"

        else:

            test_data.SE[i] = "N"
train_data['Dependents'].value_counts()
#Again for applying same weighted random theory,

#we obtain

#0:57%

#1:17%

#2:17%

#3+:9%

for i in range(len(train_data)):

    if train_data.Dependents[i] not in [0,1,2,"3+"]:

        x = random.randint(1,100)

        if x>=1 and x<57:

            train_data.Dependents[i] = 0

        if x>=57 and x<74:

            train_data.Dependents[i] = 1

        if x>74 and x<91:

            train_data.Dependents[i] = 2

        if x>=91 and x<100:

            train_data.Dependents[i] = "3+"    

for i in range(len(test_data)):

    if test_data.Dependents[i] not in [0,1,2,"3+"]:

        x = random.randint(1,100)

        if x>=1 and x<57:

            test_data.Dependents[i] = 0

        if x>=57 and x<74:

            test_data.Dependents[i] = 1

        if x>74 and x<91:

            test_data.Dependents[i] = 2

        if x>=91 and x<100:

            test_data.Dependents[i] = "3+" 
train_data['Credit_His'].value_counts()
#Filling values for 'Credit_His' using weighted random

for i in range(len(train_data)):

    if train_data.Credit_His[i] not in [1.0,0.0]:

        if random.randint(1,100) > 84:

            train_data.Credit_His[i] = "Y"

        else:

            train_data.Credit_His[i] = "N"

                        

for i in range(len(test_data)):

    if test_data.Credit_His[i] not in [0.0,1.0]:

        if random.randint(1,100) > 84:

            test_data.Credit_His[i] = "Y"

        else:

            test_data.Credit_His[i] = "N"  
datasets=[train_data,test_data]

for data in datasets:

    data['Loan_by_income'] = data['CPL_Amount']/data['App_Income_1']

    data['Loan_by_income'] = data['Loan_by_income'].astype(float)

    data['Loan_by_income'] = round(data['Loan_by_income'],2)
train_data.info()
test_data.info()
#As we can see we have successfully filled all missing values in train and test data

#We can remove columns that we think would'nt contribute to 'CPL_Status' prediction

train_data = train_data.drop(['Loanapp_ID','first_name','last_name','email','address','INT_ID','Prev_ID','AGT_ID'],axis=1)

train_data.head()
#We need to convert categorical variables into numeric form



genders = {"M": 0, "F": 1}

datasets = [train_data, test_data]

for data in datasets:

    data['Sex'] = data['Sex'].map(genders)
marriage = {"N": 0, "Y": 1}

for data in datasets:

    data['Marital_Status'] = data['Marital_Status'].map(marriage)
depend = {0:0,1:1,2:2,"3+":3}

for data in datasets:

    data['Dependents'] = data['Dependents'].map(depend)

   

    
qual={"Non Grad":0,"Grad":1}

for data in datasets:

    data['Qual_var'] = data['Qual_var'].map(qual)

    
se={"N":0,"Y":1}

for data in datasets:

    data['SE'] = data['SE'].map(se)

    
area={"Urban":0,"Rural":1,"Semi U":2}

for data in datasets:

    data['Prop_Area'] = data['Prop_Area'].map(area)

    
train_data['CPL_Term'].value_counts()
cpl = {360:1,180:2,480:3,300:4,84:5,240:6,120:7,36:8,60:9,12:10,6:11,350:12}

for data in datasets:

    data['CPL_Term'] = data['CPL_Term'].map(cpl)
credit_history={1:1,0:0,"N":0,"Y":1}

for data in datasets:

    data['Credit_His'] = data['Credit_His'].map(credit_history)

    
train_data['Loan_by_income'].value_counts()
for data in datasets:

    data['Loan_by_income'] = data['Loan_by_income'].astype(float)

    data.loc[(data['Loan_by_income'] >=0.00) & (data['Loan_by_income'] <= 0.01), 'Loan_per_income'] = 0

    data.loc[(data['Loan_by_income'] > 0.01) & (data['Loan_by_income'] <= 0.02), 'Loan_per_income'] = 1

    data.loc[(data['Loan_by_income'] > 0.02) & (data['Loan_by_income'] <= 0.03), 'Loan_per_income'] = 2

    data.loc[(data['Loan_by_income'] > 0.03) & (data['Loan_by_income'] <= 0.04), 'Loan_per_income'] = 3

    data.loc[(data['Loan_by_income'] > 0.04) & (data['Loan_by_income'] <= 0.05), 'Loan_per_income'] = 4

    data.loc[(data['Loan_by_income'] > 0.05) & (data['Loan_by_income'] <= 0.06), 'Loan_per_income'] = 5

    data.loc[(data['Loan_by_income'] > 0.06) & (data['Loan_by_income'] <= 0.09), 'Loan_per_income'] = 6

    data.loc[ data['Loan_by_income'] > 0.09, 'Loan_per_income'] = 7

    data['Loan_per_income'] = data['Loan_per_income'].astype(int)



# let's see how it's distributed train_df['Age'].value_counts()
train_data['Loan_per_income'].value_counts()
train_data=train_data.drop(['Loan_by_income'],axis=1)
train_data.info()

#We have converted values into numeric form 
train_data.describe()
train_data.head()
#We need to bin the incomes and CPL amounts



fig, ax = plt.subplots()

train_data['App_Income_1'].hist(color='red', edgecolor='black',  

                          grid=False)

ax.set_title('Applicant Income 1', fontsize=12)

ax.set_xlabel('Income', fontsize=12)

ax.set_ylabel('Frequency', fontsize=12)
fig, ax = plt.subplots()

train_data['App_Income_2'].hist(color='blue', edgecolor='black',  

                          grid=False)

ax.set_title('Applicant Income 2', fontsize=12)

ax.set_xlabel('Income', fontsize=12)

ax.set_ylabel('Frequency', fontsize=12)
fig, ax = plt.subplots()

train_data['CPL_Amount'].hist(color='green', edgecolor='black',  

                          grid=False)

ax.set_title('Loan Amount', fontsize=12)

ax.set_xlabel('Amount', fontsize=12)

ax.set_ylabel('Frequency', fontsize=12)
quantile_list = [0, .25, .5, .75, 1.]

quantiles = train_data['App_Income_1'].quantile(quantile_list)

quantiles
datasets=[train_data,test_data]

for data in datasets:

    data.loc[(data['App_Income_1'] >=0) & (data['App_Income_1'] <= 180), 'Income_1'] = 0

    data.loc[(data['App_Income_1'] > 180) & (data['App_Income_1'] <= 3452.75), 'Income_1'] = 1

    data.loc[(data['App_Income_1'] > 3452.75) & (data['App_Income_1'] <= 4574.50), 'Income_1'] = 2

    data.loc[(data['App_Income_1'] > 4574.50) & (data['App_Income_1'] <= 6954), 'Income_1'] = 3

    data.loc[ data['App_Income_1'] > 6954, 'Income_1'] = 4

    data['Income_1'] = data['Income_1'].astype(int)
quantile_list = [0, .25, .5, .75, 1.]

quantiles = train_data['App_Income_2'].quantile(quantile_list)

quantiles
for data in datasets:

    data.loc[data['App_Income_2'] == 0, 'Income_2'] = 0

    data.loc[(data['App_Income_2'] > 0) & (data['App_Income_2'] <= 1426), 'Income_2'] = 1

    data.loc[(data['App_Income_2'] > 1426) & (data['App_Income_2'] <= 2756.25), 'Income_2'] = 2

    data.loc[data['App_Income_2'] > 2756.25, 'Income_2'] = 3

    data['Income_2'] = data['Income_2'].astype(int)
quantile_list = [0, .25, .5, .75, 1.]

quantiles = train_data['CPL_Amount'].quantile(quantile_list)

quantiles
for data in datasets:

    data.loc[(data['CPL_Amount'] >=0) & (data['CPL_Amount'] <= 10.8), 'Loan_Amount'] = 0

    data.loc[(data['CPL_Amount'] > 10.8) & (data['CPL_Amount'] <= 120), 'Loan_Amount'] = 1

    data.loc[(data['CPL_Amount'] > 120) & (data['CPL_Amount'] <= 153.6), 'Loan_Amount'] = 2

    data.loc[(data['CPL_Amount'] > 153.6) & (data['CPL_Amount'] <= 200.1), 'Loan_Amount'] = 3

    data.loc[ data['CPL_Amount'] > 200.1, 'Loan_Amount'] = 4

    data['Loan_Amount'] = data['Loan_Amount'].astype(int)
train_data.head()
#Now we can drop 'App_Income_1','App_Income_2' and 'CPL_Amount' from train data

train_data = train_data.drop(["App_Income_1","App_Income_2","CPL_Amount"],axis=1)

train_data.head()
test_data=test_data.drop(["first_name","last_name","email","address","INT_ID","Prev_ID","AGT_ID","Loan_by_income","App_Income_1","App_Income_2","CPL_Amount"],axis=1)
Y_train = train_data["CPL_Status"]

X_train = train_data.drop("CPL_Status",axis=1)

X_test  = test_data.drop("Loanapp_ID", axis=1).copy()
#Logistic Regression

log_reg = LogisticRegression()

log_reg.fit(X_train, Y_train)

Y_pred = log_reg.predict(X_test)

log_reg_accuracy = round(log_reg.score(X_train, Y_train) * 100, 2)
#Support Vector Classifier

svc = LinearSVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

svc_accuracy = round(svc.score(X_train, Y_train) * 100, 2)
#K-nearest Neighbor

knn = KNeighborsClassifier(n_neighbors = 3) 

knn.fit(X_train, Y_train)  

Y_pred = knn.predict(X_test)  

knn_accuracy = round(knn.score(X_train, Y_train) * 100, 2)
#Decision Tree

d_tree = DecisionTreeClassifier()

d_tree.fit(X_train, Y_train)

Y_pred = d_tree.predict(X_test)

d_tree_accuracy = round(d_tree.score(X_train, Y_train) * 100, 2)
#Random Forest

rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train, Y_train)

Y_prediction = rf.predict(X_test)

rf_accuracy = round(rf.score(X_train, Y_train) * 100, 2)
#XGB Classifier

xgb = XGBClassifier()

xgb.fit(X_train, Y_train)

Y_pred = xgb.predict(X_test)

xgb_accuracy = round(xgb.score(X_train, Y_train) * 100, 2)
models = ['Logistic Regression','SVC','KNN', 'Decision Tree','Random Forest','XGB']

accuracy = [log_reg_accuracy,svc_accuracy,knn_accuracy,d_tree_accuracy,rf_accuracy,xgb_accuracy]

table = pd.DataFrame({'Model':models,'Accuracy':accuracy}).sort_values(by='Accuracy',ascending=False)

table = table.set_index('Accuracy')

table.head(6)
#Random Forest and Decision Tree give the highest accuracy

#Let's see the results with K-Fold Cross Validation

from sklearn.model_selection import cross_val_score

random_forest = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(random_forest, X_train, Y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())



#This is more realistic.

#The model shows an average accuracy of 72% with standard deviation of 4%
#Let's try to improve the accuracy 

importance = pd.DataFrame({'Feature':X_train.columns,'Importance':np.round(rf.feature_importances_,3)}).sort_values(by='Importance',ascending=False)

importance = importance.set_index('Feature')

importance.head(15)
#As we can see Credit_His contributes maximum importance towards predicting CPL_Status

importance.plot.bar()