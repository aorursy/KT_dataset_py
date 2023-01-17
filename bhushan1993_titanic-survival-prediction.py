# Import Libraries

%matplotlib inline

import numpy as np                                         ## Numerical Python for data analysis

import pandas as pd                                        ## Python Pandas for data analysis

import seaborn as sns                                      ## Seaborn for data visualization

import matplotlib.pyplot as plt                            ## Matrix plot library for data visualization

from sklearn.model_selection import train_test_split       ## train and test split of data for model training and pred

from sklearn.linear_model import LogisticRegression        ## To perform Logistic Regression

from sklearn.metrics import confusion_matrix               ## Confusion matrix for Model Evaluation

from sklearn.metrics import classification_report          ## To generate the classification report

from sklearn.metrics import accuracy_score                 ## To check accuracy of linear/polynomial model

from sklearn import metrics                                ## To perform metrix related operation    

import warnings                                            ## To avoid Warning

warnings.filterwarnings('ignore')

## Reading Datasets using Pandas



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

## Extracting DataSets right below after using concatination

pd.concat([train.head(5),test.head(5)], keys=["Train_Data","Test_Data"])  ## Concated datasets, means 2nd dataframes display right below to first
pd.concat([train.describe(),test.describe()], keys=["Train_Data","Test_Data"])
## Columns in the Train Data..

train.columns
## Checking of Null Values in the Train Data_Set

train.isnull().sum().to_frame().T 
fig = plt.figure(figsize=(13, 13))                                                                    ## Figure size

fig.subplots_adjust(hspace=0.4, wspace=1)                                                             ## Define Spaces in Subplot



for i in range(1, 11):                                                                                ## Apply for Loop for subplotting

    

    ax = fig.add_subplot(5, 2, i)

    

    if i==1:

        plt.title('Features',fontsize=20)

        sns.barplot(x="Sex", y="Survived",data=train)

    

    elif i==2:

        

        plt.title('Features in Percentage',fontsize=20)

        Females=train["Survived"][train["Sex"] == 'female'].value_counts(normalize=True)[1]*100

        males=train["Survived"][train["Sex"] == 'male'].value_counts(normalize=True)[1]*100

        values1=[males,Females]

        plt.pie(values1, labels=('males','Females'),explode=(0.0,0.1),autopct='%1.1f%%')

    

    elif i==3:

        sns.barplot(x="Pclass", y="Survived",data=train)

    

    elif i==4:

        

        Pclass1=train["Survived"][train["Pclass"] == 1].value_counts(normalize=True)[1]*100

        Pclass2=train["Survived"][train["Pclass"] == 2].value_counts(normalize=True)[1]*100

        Pclass3=train["Survived"][train["Pclass"] == 3].value_counts(normalize=True)[1]*100

        values2=[Pclass1,Pclass2,Pclass3]

        plt.pie(values2, labels=('Pclass1','Pclass2','Pclass3'),explode=(0.1,0.0,0.0),autopct='%1.1f%%')

    

    elif i==5:

        

        sns.barplot(x="Pclass", y="Survived", hue = "Sex", data=train)

    

    elif i==6:

        

        PC1 = train["Survived"][train["Sex"] == 'female'][train['Pclass']==1].value_counts(normalize=True)[1]*100

        PC12 = train["Survived"][train["Sex"] == 'male'][train['Pclass']==1].value_counts(normalize=True)[1]*100

        PC2 = train["Survived"][train["Sex"] == 'female'][train['Pclass']==2].value_counts(normalize=True)[1]*100

        PC22 = train["Survived"][train["Sex"] == 'male'][train['Pclass']==2].value_counts(normalize=True)[1]*100

        PC3 = train["Survived"][train["Sex"] == 'female'][train['Pclass']==3].value_counts(normalize=True)[1]*100

        PC33 = train["Survived"][train["Sex"] == 'male'][train['Pclass']==3].value_counts(normalize=True)[1]*100

        values3=[PC1,PC12,PC2,PC22,PC3,PC33]

        

        plt.pie(values3, labels=('Class1-Female','Class1-Male','Class2-Female','Class2-Male','Class3-Female','Class3-Male'),autopct='%1.1f%%')

    

    elif i ==7:

        

        sns.barplot(x="SibSp", y="Survived",data=train)

    

    elif i == 8:

        

        a = train["Survived"][train['SibSp'] == 0].value_counts(normalize=True)[1]*100

        b = train["Survived"][train['SibSp'] == 1].value_counts(normalize=True)[1]*100

        c = train["Survived"][train['SibSp'] == 2].value_counts(normalize=True)[1]*100

        d = train["Survived"][train['SibSp'] == 3].value_counts(normalize=True)[1]*100

        e = train["Survived"][train['SibSp'] == 4].value_counts(normalize=True)[1]*100

        values4=[a,b,c,d,e]

        plt.pie(values4, labels=('SibSp0','SibSp1','SibSp2','SibSp3','SibSp4'),autopct='%1.1f%%')

    

    elif i == 8:

        

        sns.barplot(x="Parch", y="Survived",data=train)

    

    elif i == 8:

        P0=train["Survived"][train["Parch"] == 0].value_counts(normalize=True)[1]*100

        P1=train["Survived"][train["Parch"] == 1].value_counts(normalize=True)[1]*100

        P2=train["Survived"][train["Parch"] == 2].value_counts(normalize=True)[1]*100

        P3=train["Survived"][train["Parch"] == 3].value_counts(normalize=True)[1]*100

        P5=train["Survived"][train["Parch"] == 5].value_counts(normalize=True)[1]*100

        values5=[P0,P1,P2,P3,P5]

        plt.pie(values5, labels=('Parch0','Parch1','Parch2','Parch3','Parch5'),explode=(0.0,0.0,0.0,0.1,0.0),autopct='%1.1f%%')

    

    elif i==9:

        

        sns.barplot(x="Parch", y="Survived",data=train)

    

    elif i==10:

        

        P0=train["Survived"][train["Parch"] == 0].value_counts(normalize=True)[1]*100

        P1=train["Survived"][train["Parch"] == 1].value_counts(normalize=True)[1]*100

        P2=train["Survived"][train["Parch"] == 2].value_counts(normalize=True)[1]*100

        P3=train["Survived"][train["Parch"] == 3].value_counts(normalize=True)[1]*100

        P5=train["Survived"][train["Parch"] == 5].value_counts(normalize=True)[1]*100

        values6=[P0,P1,P2,P3,P5]

        plt.pie(values6, labels=('Parch0','Parch1','Parch2','Parch3','Parch5'),explode=(0.0,0.0,0.0,0.1,0.0),autopct='%1.1f%%')
## Age of People in Passenger Class and distribution plot of Age

fig = plt.figure(figsize=(13, 8))

fig.subplots_adjust(hspace=0.4, wspace=0.4)

plt.subplot(1,2,1)

sns.boxplot(x='Pclass',y='Age',data=train)

plt.subplot(1,2,2)

sns.distplot(train['Age'].dropna(), kde=False, bins=30, color='Red')
## Assumed value of Ages inplace of Nan..



def Assumed_Age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 45

        if Pclass == 2:

            return 37

        if Pclass == 3:

            return 25

    else:

        return Age

    
train['Age'] = train[['Age','Pclass']].apply(Assumed_Age,axis=1)
## Deal with Nan in Embarked column...

train.isnull().sum().to_frame().T 
train.dropna(subset=['Embarked'],inplace=True)
train.drop(['Name','Ticket','Fare','Cabin'], axis=1, inplace=True)
train.isnull().sum().to_frame().T  
## Check the dtype of all columns

train.info()
Sex_Mapping = {"male":0,"female":1}

train['Sex']=train['Sex'].map(Sex_Mapping)
train['Sex'].head()
train['Sex'].isin(['0']).value_counts()
Embarked_Mapping = {'S':1,'C':2,'Q':3}

train['Embarked']=train['Embarked'].map(Embarked_Mapping)
train.Embarked.head()
train['Age']=train['Age'].astype('int64') ## from now on data type of Age column is int64
train.info()
predictors = train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
## Lgistic_Regression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_val)



#Accuracy

acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)

print("Accuracy:",acc_logreg)
print(classification_report(y_val,y_pred))  ##(y_test,predictions)
print(confusion_matrix(y_val, y_pred))  ##y_test, predictions
test.head()
test.drop(['Name','Ticket','Fare','Cabin'], axis=1, inplace=True) ## Drop down unnecessary columns 
test.info()
Sex_Mapp = {"male":0,"female":1}

test['Sex']=test['Sex'].map(Sex_Mapp)
EM = {'S':1,'C':2,'Q':3}

test['Embarked']=test['Embarked'].map(EM)
test.isnull().sum().to_frame().T
def Age_Assumption(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 45

        if Pclass == 2:

            return 37

        if Pclass == 3:

            return 25

    else:

        return Age
test['Age'] = test[['Age','Pclass']].apply(Age_Assumption, axis=1)
test['Age']=test['Age'].astype('int64')
test.isnull().sum().to_frame().T
test.head()
idss = test['PassengerId']

predictions = logreg.predict(test.drop(['PassengerId'], axis=1))

output = pd.DataFrame({"PassengerId":idss,"Survived":predictions})
output.head(10)
sns.countplot(x="Survived", data=output)
output['Survived'].value_counts()
output.to_csv('submission.csv', index=False)