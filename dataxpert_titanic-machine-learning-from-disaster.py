# Basic / Essential Set of Python Library



import pandas as pd                         # Dataframe / data analysis

import numpy as np                          # Linear Algebra functionality

import seaborn as sns                       # Data visualization

import matplotlib.pyplot as plt             # Data visualization  



sns.set_style('whitegrid')                  # Default sytle for Seaborn / Matplotlib Visualization libraries



%matplotlib inline  



#Machine Learning libararies and other modules will be loaded and displayed within their respective

#sections for ease of understanding and continuity 

titanic = pd.read_csv("../input/train.csv")    # Loading of 'train.csv' data file in 'titanic" dataframe
print("'titanic' dataframe shape:  ", titanic.shape)  # To get a quick count of number of columns and rows in the datasets
titanic.info()   # Concise summary of combined 'titanic' dataset containing column names and data types
titanic.head()          # This provides an overview of first five rows of titanic dataset
titanic.describe(include='all')     # To generate descriptive statitics of combined titanic dataset.
titanic.isnull().sum()          # To get the count of missing data in each column
plt.figure(figsize=(8,5))

sns.set_style('whitegrid')

sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False)   # Visualization of missing data

plt.title('Graphical Representations of Null Values in Titanic Dataframe')
titanic[titanic['Embarked'].isnull()]    # This will give us record where there are missing values in "Embarked" column
titanic[(titanic['Pclass']==1)].groupby('Embarked')['Pclass'].count() # Shows the number of 1st Class passengers by Embarked
plt.figure(figsize=(12,5))

sns.countplot(data=titanic, x='Pclass', hue='Embarked')

plt.title('Number of Passengers by Pclass from Port of Emabarkation')
titanic['Embarked'].fillna('S',inplace=True) # To impute missing values in "Embarked" column with "S"
 # This will give us the number of missing values within "Embarked" column AFTER fillin



titanic[(titanic['PassengerId'] == 62) | (titanic['PassengerId'] == 830)]   
plt.figure(figsize=(10,5))

sns.pointplot(x='Embarked', y='Survived',data=titanic, color='g')  # Overall survival based on port of embark

plt.title('Survival vs. Port of Embarkation')
sns.factorplot('Embarked','Survived', col='Pclass',data=titanic) # Survival based on Embarked vs. Pclass
sns.factorplot('Embarked','Survived', hue='Sex', data=titanic, palette='RdBu_r') # Surivial based on port of Embark and gender

plt.title('Survival vs. Port of Embarkation based on Gender')
plt.figure(figsize=(10,5))

sns.violinplot(data=titanic, x='Embarked', y= 'Age', hue='Sex',palette='RdBu_r',split=True)

plt.title('Port of Embarkation vs. Age & Gender')
plt.figure(figsize=(12,5))

sns.distplot(titanic['Age'].dropna(),bins=50,color='blue',kde=True)

plt.title('Age Distribution')
print ("Mean Age:   ", titanic.Age.dropna().mean(), "years")   # Calculates the mean age

print ("Median Age: ", titanic.Age.dropna().median(), "years") # Calculates the mediam age
fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(15,7))

sns.violinplot(x = 'Pclass', y = 'Age', data=titanic,palette='rainbow', ax=ax1)

ax1.set(title='Age vs. Class of Travel')



sns.boxplot(x = 'Pclass', y = 'Age', data=titanic, hue='Sex',palette='RdBu_r', ax=ax2)  

ax2.set(title='Age vs. Class of Travel - Based on Gender')

#sns.despine()
# Overall average Age per Class

print("Class 1, Overall average age: ",(titanic[titanic['Pclass']==1])['Age'].mean())

print("Class 2, Overall average age: ",(titanic[titanic['Pclass']==2])['Age'].mean())

print("Class 3, Overall average age: ",(titanic[titanic['Pclass']==3])['Age'].mean())
# Average Age per Class based on Gender

print("Class 1, Male average age  : ",(titanic[(titanic['Pclass']==1) & (titanic['Sex']== 'male')])['Age'].mean())

print("Class 1, Female average age: ",(titanic[(titanic['Pclass']==1) & (titanic['Sex']== 'female')])['Age'].mean())

print("Class 2, Male average age  : ",(titanic[(titanic['Pclass']==2) & (titanic['Sex']== 'male')])['Age'].mean())

print("Class 2, Female average age: ",(titanic[(titanic['Pclass']==2) & (titanic['Sex']== 'female')])['Age'].mean())

print("Class 3, Male average age  : ",(titanic[(titanic['Pclass']==3) & (titanic['Sex']== 'male')])['Age'].mean())

print("Class 3, Female average age: ",(titanic[(titanic['Pclass']==3) & (titanic['Sex']== 'female')])['Age'].mean())
titanic['Title'] = titanic.Name.str.extract('([A-Za-z]+)\.',expand=True)
titanic['Title'].value_counts()
pd.crosstab(titanic['Title'],titanic['Sex'],margins=True)
# Replacing titles to reduce overall times to Child, Mr, Mrs, Miss, and Other



titanic['Title'].replace(['Master','Ms','Mlle','Mme','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],\

            ['Child','Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
titanic.Title.value_counts()  # Rechecking whether the changes are made correctly
titanic.groupby('Title')['Age'].mean()   # Mean age based on the engineered feature "Title"
titanic.groupby(['Title','Pclass'])['Age'].mean()   # Mean age based on "Title" and "Pclass"
# Let's round the above mentioned values to the nearest whole number before imputing into missing/null values of Age column

round(titanic.groupby(['Title','Pclass'])['Age'].mean())
sns.factorplot(data=titanic, x='Title', col = 'Pclass',kind='count',hue='Survived')
"""

The following function will be used to populate missing / null values of Age column that are calculated above.



"""



def age_fix(cols):

    

    Age = cols[0]

    Pclass = cols[1]

    Title = cols[2]

    

    if pd.isnull(Age):

        

        if Pclass == 1 and Title == 'Child':

            return 5

        elif Pclass == 2 and Title == 'Child':

            return 2

        elif Pclass == 3 and Title == 'Child':

            return 5

        

        elif Pclass == 1 and Title == 'Miss':

            return 30

        elif Pclass == 2 and Title == 'Miss':

            return 23

        elif Pclass == 3 and Title == 'Miss':

            return 16

        

        elif Pclass == 1 and Title == 'Mr':

            return 42

        elif Pclass == 2 and Title == 'Mr':

            return 33

        elif Pclass == 3 and Title == 'Mr':

            return 29

        

        elif Pclass == 1 and Title == 'Mrs':

            return 41

        elif Pclass == 2 and Title == 'Mrs':

            return 34

        elif Pclass == 3 and Title == 'Mrs':

            return 34

        

        elif Pclass == 1 and Title == 'Other':

            return 51

        elif Pclass == 2 and Title == 'Other':

            return 43

              

        else:

            return Age

    else:

        return Age

    
titanic['Age'] = titanic[['Age','Pclass','Title']].apply(age_fix,axis=1) #The "age_fix" function is applied to "titanic" dataset
titanic.isnull().sum()
sns.factorplot(x='Pclass',y='Survived',col='Title',data=titanic)
titanic['Age'].describe(include=all)
titanic['Age'].plot(kind='hist',bins=30,xlim=(0,75),figsize=(12,4))
titanic['AgeBins'] = 0  # New feature "AgeBins" created and an initial value '0' is assigned to it
titanic['AgeBins']=pd.qcut(titanic['Age'],5)  # Divides data into five equal bins
titanic.groupby('AgeBins')['AgeBins'].count()    # Confirms the values in each bin
pd.crosstab(titanic['AgeBins'],titanic['Survived'],margins=True)
plt.figure(figsize=(10,5))

sns.countplot(x='AgeBins',hue='Survived',data=titanic,palette='rainbow')  # AgeBins vs. Survival

plt.title('Survival vs. Age Bins')
pd.crosstab(titanic['AgeBins'],titanic['Survived'],margins=True)
plt.figure(figsize=(10,5))

sns.countplot(x='AgeBins',hue='Sex',data=titanic,palette='RdBu_r')   # AgeBins vs. Gender (Sex)

plt.title('Age Bins vs. Gender')
plt.figure(figsize=(10,5))

sns.stripplot(data=titanic, x='Pclass', y= 'Age',size=7)        # Age vs. Pclass

plt.title('Age vs. Class of Travel')
titanic['NAge'] = 0  # Create a new feature 'NAge' and assign initial value '0'
titanic.loc[titanic['Age']<=19.00,'NAge']=0

titanic.loc[(titanic['Age']>19.00)&(titanic['Age']<=26.00),'NAge']=1

titanic.loc[(titanic['Age']>26.00)&(titanic['Age']<=30.00),'NAge']=2

titanic.loc[(titanic['Age']>30.00)&(titanic['Age']<=40.00),'NAge']=3

titanic.loc[(titanic['Age']>40.00)&(titanic['Age']<=81.00),'NAge']=4

titanic.groupby('NAge')['NAge'].count()   # Confirm values in 'NAge" feature after imputation
titanic[(titanic['Age'] <= 16) & (titanic['Title'] == 'Mrs')]   # Child brides?
titanic[(titanic['Age'] <= 16) & (titanic['Title'] !='Mrs')]['Age'].count()   # Count of Child Passengers
titanic['Child'] = 0    # Creates a new feature "Child" and assigns initial value '0'
# Assigns value '1' to all Children based on the above-mentioned criteria

titanic.loc[(titanic['Age'] <= 16) & (titanic['Title'] !='Mrs'),'Child'] = 1 
titanic.Child.value_counts()   # Reconfirms that values have been successfully put
pd.crosstab(titanic['Child'],titanic['Survived'],margins=True)    # Survived vs. Child
sns.factorplot('Child','Survived',data=titanic)       # Children surived vs. died

plt.title('Children Survival')
titanic['Deck'] = titanic['Cabin'].astype(str).str[0]  # Extracting first character in "Cabin" to create a new column "Deck"
titanic.head(3)
titanic.Deck.value_counts()  # Gives the count for each value in the "Deck" column
pd.crosstab(titanic['Deck'],titanic['Survived'],margins=True)
titanic['IsCabin'] = 1 # Create a new feature "IsCabin" and assign a default value "1"
titanic.loc[titanic['Cabin'].isnull(),'IsCabin'] = 0  # Populate "IsCabin" with value '0' where "Cabin" is Null/NaN
titanic.loc[titanic['Cabin'].isnull(),'IsCabin'] = 0  # Populate "IsCabin" with value '0' where "Cabin" is Null/NaN
titanic['IsCabin'].value_counts()  # Calculate values in 'IsCabin' feature
sns.factorplot(x='IsCabin',y='Survived',col='Pclass',hue='Sex',data=titanic, palette='RdBu_r')
pd.crosstab(titanic['Pclass'],titanic['Survived'],margins=True,) # Passengers survived vs. died based on Pclass feature
 # Percentage of passengers travelled per class of travel

titanic['Pclass'].value_counts().plot.pie(explode=[0.02,0.02,0.02],autopct='%1.1f%%',figsize=(7,7))

plt.title('Passengers per Class of Travel (%age)')

plt.figure(figsize=(10,5))

sns.countplot(x='Pclass',hue='Survived',data=titanic,palette='rainbow')    # Survived vs. Died per class of travel

plt.title('Survival vs. Class of Travel')
plt.figure(figsize=(8,5))

sns.factorplot(x='Sex',y='Survived',col='Pclass',data=titanic)
pd.crosstab(titanic['SibSp'],titanic['Survived'],margins=True) # Passengers survived vs. died based on SibSp feature
 # Graphical representation of passengers survived vs. die based on Siblings and Spouses  

fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(15,5))

sns.pointplot('SibSp','Survived',hue='Pclass',data=titanic,ax=ax1)

ax1.set(title='Siblings & Spouses Survival based on Class of Travel')

sns.countplot(x='SibSp',data=titanic,hue='Survived',ax=ax2) 

ax2.set(title='Siblings & Spouses Survival')
pd.crosstab(titanic['SibSp'],titanic['Pclass'])
pd.crosstab(titanic['Parch'],titanic['Survived'],margins=True) # Passengers survived vs. died based on Parch feature
 # Graphical representation of passengers survived vs. die based on Siblings and Parch  

fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(15,5))

sns.pointplot('Parch','Survived',hue='Pclass',data=titanic,ax=ax1)

ax1.set(title='Parent/Child Survival - Based on Class of Travel')

sns.countplot(x='Parch',data=titanic,hue='Survived',ax=ax2, palette='rainbow') 

ax2.set(title='Parents & Children Survival')
pd.crosstab(titanic['Parch'],titanic['Pclass'])
#Creating new feature "FamSize" by adding values in "SibSp" and "Parch"



titanic['FamSize'] = titanic['SibSp'] + titanic['Parch'] 
titanic.FamSize.value_counts()
pd.crosstab(titanic['Survived'],titanic['FamSize'],margins=True)  # Survival vs. family size feature
titanic['Alone'] = 0  # Creating a new feature "Alone" with default value = 0
titanic.loc[titanic['FamSize']== 0,'Alone'] = 1  # Populate "Alone" with value '1' where family size is '0'
titanic.Alone.value_counts()
pd.crosstab(titanic['Alone'],titanic['Survived'],margins=True)  # Survival vs. Alone feature
f,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))

sns.pointplot('FamSize','Survived',data=titanic,ax=ax1)

ax1.set(title='Survival Based on Family Size')

sns.pointplot('Alone','Survived',hue='Sex',data=titanic,palette='RdBu_r',ax=ax2)

ax2.set(title='Survival vs. Alone Based on Gender')
pd.crosstab(titanic['Sex'],titanic['Survived'],margins=True)  # Survival vs. gender (Sex)
f, (ax1,ax2) = plt.subplots(1,2,figsize=(15,5))

sns.countplot(x='Sex',data=titanic,hue='Survived',palette='rainbow',ax=ax1)

ax1.set(title='Survival Based on Gender')

sns.pointplot(x='Embarked',y='Survived',hue='Sex',data=titanic, palette='RdBu_r',ax=ax2)

ax2.set(title='Survival Based on Port of Embarkation & Gender')
sns.factorplot('Sex','Survived', col='Pclass',data=titanic,kind='bar',palette='RdBu_r') 
titanic['Fare'].describe(include=all) # Descriptive stats for "Fare"
titanic[titanic['Fare'] >= 300]  # Passengers paid more than 300 
titanic['Fare'].plot(kind='hist',bins=50,xlim=(-1,100),figsize=(12,4))

plt.title('Fare Distrubution')
titanic['FareBins']=pd.qcut(titanic['Fare'],4)  # Divides data into equal bins
titanic.groupby('FareBins')['FareBins'].count()  # Confirms the values in each bin
titanic['NFare'] = 0  # Creates a feature 'NFare' and assign an initial value '0'
titanic.loc[titanic['Fare']<=7.91,'NFare']=0

titanic.loc[(titanic['Fare']>7.91)&(titanic['Fare']<=14.454),'NFare']=1

titanic.loc[(titanic['Fare']>14.454)&(titanic['Fare']<=31),'NFare']=2

titanic.loc[(titanic['Fare']>31)&(titanic['Fare']<=513),'NFare']=3
pd.crosstab(titanic['NFare'],titanic['Survived'],margins=True)  # Survived vs. NFare
plt.figure(figsize=(12,5))

sns.countplot(x='FareBins',hue='Survived',data=titanic,palette='rainbow')

plt.title('Survival Based on Fare Bins')
sns.factorplot('Pclass','NFare',data=titanic, hue='Survived')

plt.title('Survival Based on Fare & Class of Travel')
titanic['Ticket'].value_counts().head(20)   # A quick value count of "Ticket"
titanic['SharedTicket']= 0 # A new feature "FanTicket" created with initial value "0"
ticketV = titanic['Ticket'].value_counts()  #Calculates passengers groups on each tickets and assign it to a variable 'ticketV'

ticketV.head(2)
single = ticketV.loc[ticketV ==1].index.tolist()  # Creates a list of tickets used by individual(single) passemgers

multi  = ticketV.loc[ticketV > 1].index.tolist()  # Creates a list of tickets shared by group of passemgers
print("Number of Individual Tickets: ", len(multi))     # Prints individual tickets count

print("Number of Shared Tickets    : ", len(single))    # Prints shared tickets count
# Compares the ticket number in the "multi" list that was created above with titanic dataset "Ticket" feature and plugin '1'

for ticket in multi:

    titanic.loc[titanic['Ticket'] == ticket, 'SharedTicket'] = 1

    
titanic['SharedTicket'].value_counts() # Checks the values in "SharedTicket" column to confirm the accuracy of imputation
pd.crosstab(titanic['SharedTicket'],titanic['Survived'],margins=True) # Survived vs. SharedTicket
f, (ax1,ax2) = plt.subplots(1,2,figsize=(15,5))

sns.countplot(x='SharedTicket',data=titanic,hue='Survived',palette='rainbow',ax=ax1)

ax1.set(title='Survival Based on Shared Ticket')

sns.pointplot(x='SharedTicket',y='Survived',hue='Sex',data=titanic, palette='RdBu_r',ax=ax2)

ax2.set(title='Survival Based on Shared Ticket & Gender')
pd.crosstab(titanic['SharedTicket'],titanic['Pclass'],margins=True) # Pclass vs. SharedTicket
# Survival based on SharedTicket vs. Pclass taking into account the gender (Sex)

sns.factorplot('SharedTicket','Survived', col='Pclass',hue='Sex',data=titanic, palette='RdBu_r') 
titanic.shape   # Our dataset now contains 23 features, most of which are not required for predictive modeling.
titanic.info()  # Overview of dataset features
titanic.head()   # First five rows of titanic dataset
int1 = titanic.copy()
int1.isnull().sum()
emb  = pd.get_dummies(titanic['Embarked'],drop_first=True) #Creates two Dummy Varable "Q" and "C" and drops the values for "S"   

nsex = pd.get_dummies(titanic['Sex'],drop_first=True)     #Creates Dummy Varable "male" and drops the values for Female

titanic = pd.concat([titanic,emb],axis=1)  # Concatenate titanic dataset with emb

titanic = pd.concat([titanic,nsex],axis=1)  # Concatenate titanic dataset with nsex
titanic.shape
titanic['Title'].replace(['Mr','Mrs','Miss','Child','Other'],[0,1,2,3,4],inplace=True)

# Removes unwanted features

titanic.drop(['PassengerId','Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked',\

              'AgeBins','Deck', 'FareBins', ],inplace=True,axis=1)      
plt.figure(figsize=(15,7))

sns.heatmap(titanic.corr(),cmap='RdYlGn_r',annot=True)

plt.title('Titanic Correlation Chart')
titanic.shape
titanic.head(2)
y = titanic['Survived']                             # Target variable

X = titanic.drop('Survived',inplace=False,axis=1)   # Predictors
X.head(2)         # Header of the Predictor Variable (X)
y.head(2)       # Header of the Target Variable (y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)   # A 25-70 split of Test and Train data

print ("'X_train', train data count : ", X_train.shape)

print ("'X_test', test data count   : ", X_test.shape)
from sklearn.linear_model import LogisticRegression      # Importing of Logistic Regression Library from Scikit-Learn

lr = LogisticRegression()                                # Creation of Logistic Regression Model    

lr.fit(X_train,y_train)                                  # Model Training

lr_pred = lr.predict(X_test)                             # Prediction based on X_test
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report   # Importing of different metrics

print('\nAccuracy Score = ',accuracy_score(y_test,lr_pred))                         # Accuracy Score of the Model

print('\nConfusion Matrix:','\n',confusion_matrix(y_test,lr_pred))                  # Confusion Matrix

print('\nClassification Report:','\n',classification_report(y_test,lr_pred))        # Classification Report 
from sklearn.tree import DecisionTreeClassifier          # Importing of Logistic Regression Library from Scikit-Learn

dt = DecisionTreeClassifier()                            # Creation of Decision Tree Classifier Model   

dt.fit(X_train,y_train)                                  # Model Training

dt_pred = dt.predict(X_test)                             # Prediction based on X_test
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report   # Importing of different metrics

print('\nAccuracy Score = ',accuracy_score(y_test,dt_pred))                         # Accuracy Score of the Model

print('\nConfusion Matrix:','\n',confusion_matrix(y_test,dt_pred))                  # Confusion Matrix

print('\nClassification Report:','\n',classification_report(y_test,dt_pred))        # Classification Report 
from sklearn.ensemble import RandomForestClassifier      # Importing of Logistic Regression Library from Scikit-Learn

rfc = RandomForestClassifier(n_estimators=100)           # Creation of Random Forest Classifier Model  

rfc.fit(X_train,y_train)                                 # Model Training

rfc_pred = rfc.predict(X_test)                           # Prediction based on X_test
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report    # Importing of different metrics

print('\nAccuracy Score = ',accuracy_score(y_test,rfc_pred))                         # Accuracy Score of the Model

print('\nConfusion Matrix:','\n',confusion_matrix(y_test,rfc_pred))                  # Confusion Matrix

print('\nClassification Report:','\n',classification_report(y_test,rfc_pred))        # Classification Report 
from sklearn.svm import SVC                              # Importing of Logistic Regression Library from Scikit-Learn

svr = SVC(kernel='rbf')                                  # Creation of Support Vector Machine (Radial) Model  

svr.fit(X_train,y_train)                                 # Model Training

svr_pred = svr.predict(X_test)                           # Prediction based on X_test
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report    # Importing of different metrics

print('\nAccuracy Score = ',accuracy_score(y_test,svr_pred))                         # Accuracy Score of the Model

print('\nConfusion Matrix:','\n',confusion_matrix(y_test,svr_pred))                  # Confusion Matrix

print('\nClassification Report:','\n',classification_report(y_test,svr_pred))        # Classification Report 
from sklearn.svm import SVC                              # Importing of Logistic Regression Library from Scikit-Learn

sv = SVC(kernel='linear')                                # Creation of Support Vector Machine (Linear) Model  

sv.fit(X_train,y_train)                                  # Model Training

sv_pred = sv.predict(X_test)                             # Prediction based on X_test
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report   # Importing of different metrics

print('\nAccuracy Score = ',accuracy_score(y_test,sv_pred))                         # Accuracy Score of the Model

print('\nConfusion Matrix:','\n',confusion_matrix(y_test,sv_pred))                  # Confusion Matrix

print('\nClassification Report:','\n',classification_report(y_test,sv_pred))        # Classification Report 
from sklearn.neighbors import KNeighborsClassifier      # Importing of Logistic Regression Library from Scikit-Learn

knn = KNeighborsClassifier()                            # Creation of K-Nearest Neighbors Classifier Model 

knn.fit(X_train,y_train)                                # Model Training

knn_pred = knn.predict(X_test)                          # Prediction Based on X_train 
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report    # Importing of different metrics

print('\nAccuracy Score = ',accuracy_score(y_test,knn_pred))                         # Accuracy Score of the Model

print('\nConfusion Matrix:','\n',confusion_matrix(y_test,knn_pred))                  # Confusion Matrix

print('\nClassification Report:','\n',classification_report(y_test,knn_pred))        # Classification Report 
cnt = list(range(1,21))       # Range for the count of K-values

knn_score = []                # List to populate accurracy scores based on the K-value  



for i in cnt:                 # Loop to iterate through K-values, apply KNN Model and append the accuracy score to the list    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    kpred = knn.predict(X_test)

    score = accuracy_score(y_test,kpred)

    knn_score.append(score)



# Plotting the graph to dipict Accuracy Score vs. K-values



plt.figure(figsize=(12,5))

plt.plot(cnt,knn_score,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.title('Accuracy Score vs. K Values')

plt.xlabel('K-Value')

plt.ylabel('Accuracy Score')
# Summarizing accuracy scores into a dictionery and plotting the graph

summary = {'Logistic Reg':accuracy_score(y_test,lr_pred),'DecisionTree':accuracy_score(y_test,dt_pred),\

           'RandomForest':accuracy_score(y_test,rfc_pred),'SVM-Linear':accuracy_score(y_test,sv_pred),\

           'SVM-rbc':accuracy_score(y_test,svr_pred),'KNN':accuracy_score(y_test,knn_pred)}

print(summary)

plt.figure(figsize=(12,5))

sns.pointplot(x=list(summary.keys()),y=list(summary.values()))

plt.title('Accuracy Scores vs. Predictive Models')

plt.xlabel('Predictive Models')

plt.ylabel('Accuracy Scores')
test = pd.read_csv("../input/test.csv")    # Loading of 'test.csv' data file in 'test" dataframe
# Missing data imputation, feature engineering and data cleanup



test.loc[test['Fare'].isnull(),'Fare'] = 35.63

test['Title'] = test.Name.str.extract('([A-Za-z]+)\.',expand=True)  # Extract 'Title' from 'Name"

# Replacing titles to reduce overall times to Child, Mr, Mrs, Miss, and Other

test['Title'].replace(['Dona','Master','Rev','Col','Dr','Ms'],['Miss','Child','Other','Other','Other','Miss'],inplace=True)

test.groupby(['Title','Pclass'])['Age'].mean()   # Mean age based on "Title" and "Pclass"



#Function to populate missing values in test dataset

def test_age_fix(cols):

    

    Age = cols[0]

    Pclass = cols[1]

    Title = cols[2]

    

    if pd.isnull(Age):

        

        if Pclass == 1 and Title == 'Child':

            return 10

        elif Pclass == 2 and Title == 'Child':

            return 5

        elif Pclass == 3 and Title == 'Child':

            return 7

        

        elif Pclass == 1 and Title == 'Miss':

            return 32

        elif Pclass == 2 and Title == 'Miss':

            return 17

        elif Pclass == 3 and Title == 'Miss':

            return 20

        

        elif Pclass == 1 and Title == 'Mr':

            return 41

        elif Pclass == 2 and Title == 'Mr':

            return 32

        elif Pclass == 3 and Title == 'Mr':

            return 27

        

        elif Pclass == 1 and Title == 'Mrs':

            return 46

        elif Pclass == 2 and Title == 'Mrs':

            return 33

        elif Pclass == 3 and Title == 'Mrs':

            return 30

        

        elif Pclass == 1 and Title == 'Other':

            return 51

        elif Pclass == 2 and Title == 'Other':

            return 36

              

        else:

            return Age

    else:

        return Age

    



test['Age'] = test[['Age','Pclass','Title']].apply(test_age_fix,axis=1) #The "test_age_fix" function is applied to "test" dataset



test['NAge'] = 0  # Create a new feature 'NAge' and assign initial value '0'



test.loc[test['Age']<=19.00,'NAge']=0

test.loc[(test['Age']>19.00)&(test['Age']<=26.00),'NAge']=1

test.loc[(test['Age']>26.00)&(test['Age']<=30.00),'NAge']=2

test.loc[(test['Age']>30.00)&(test['Age']<=40.00),'NAge']=3

test.loc[(test['Age']>40.00)&(test['Age']<=81.00),'NAge']=4



test['Child'] = 0    # Creates a new feature "Child" and assigns initial value '0'



# Assigns value '1' to all Children based on the above-mentioned criteria

test.loc[(test['Age'] <= 16) & (test['Title'] !='Mrs'),'Child'] = 1 



test['Deck'] = test['Cabin'].astype(str).str[0]  # Extracting first character in "Cabin" to create a new column "Deck"



test['IsCabin'] = 1 # Create a new feature "IsCabin" and assign a default value "1"



test.loc[test['Cabin'].isnull(),'IsCabin'] = 0  # Populate "IsCabin" with value '0' where "Cabin" is Null/NaN



#Creating new feature "FamSize" by adding values in "SibSp" and "Parch"



test['FamSize'] = test['SibSp'] + test['Parch'] 



test['Alone'] = 0  # Creating a new feature "Alone" with default value = 0



test.loc[test['FamSize']== 0,'Alone'] = 1  # Populate "Alone" with value '1' where family size is '0'



test['FareBins']=pd.qcut(test['Fare'],4)  # Divides data into equal bins



test['NFare'] = 0  # Creates a feature 'NFare' and assign an initial value '0'



# Now, let's assign a value (from 0 to 3) based on the *'FareBins'*



test.loc[test['Fare']<=7.91,'NFare']=0

test.loc[(test['Fare']>7.91)&(test['Fare']<=14.454),'NFare']=1

test.loc[(test['Fare']>14.454)&(test['Fare']<=31),'NFare']=2

test.loc[(test['Fare']>31)&(test['Fare']<=513),'NFare']=3



test['SharedTicket']= 0 # A new feature "FanTicket" created with initial value "0"



ticketV = test['Ticket'].value_counts()  #Calculates passengers groups on each tickets and assign it to a variable 'ticketV'



single = ticketV.loc[ticketV ==1].index.tolist()  # Creates a list of tickets used by individual(single) passemgers

multi  = ticketV.loc[ticketV > 1].index.tolist()  # Creates a list of tickets shared by group of passemgers



# Compares the ticket number in the "multi" list that was created above with test dataset "Ticket" feature and plugin '1'

for ticket in multi:

    test.loc[test['Ticket'] == ticket, 'SharedTicket'] = 1



emb  = pd.get_dummies(test['Embarked'],drop_first=True) #Creates two Dummy Varable "Q" and "C" and drops the values for "S"   

nsex = pd.get_dummies(test['Sex'],drop_first=True)     #Creates Dummy Varable "male" and drops the values for Female



test = pd.concat([test,emb],axis=1)  # Concatenate test dataset with emb

test = pd.concat([test,nsex],axis=1)  # Concatenate test dataset with nsex



test['Title'].replace(['Mr','Mrs','Miss','Child','Other'],[0,1,2,3,4],inplace=True)



test1 = test.copy()

# Removes unwanted features

test.drop(['Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked','Deck', 'FareBins']\

          ,inplace=True,axis=1) 

dsubmit = test.copy()    # Make a copy of test dataset as dsubmit that will be used for further processing

dsubmit.drop('PassengerId',inplace=True,axis=1)
dsubmit.head(2)
# Creating trainX and trainY datasets using full titanic dataset

trainX = titanic.drop('Survived',axis=1)

trainY = titanic['Survived']
trainX.head(2)            # Header of trainX
trainY.head(2)           # Header of trainY
from sklearn.svm import SVC                 # Importing Support Vector Machine library from Scikit-Learn  

model = SVC(kernel='rbf')                   # Model building

model.fit(trainX,trainY)                    # Model training

kpred = model.predict(dsubmit)               # Prediction  
submit = pd.read_csv("../input/gender_submission.csv")

submit.set_index('PassengerId',inplace=True)



submit['Survived'] = kpred

submit['Survived'] = submit['Survived'].apply(int)

submit.to_csv('submit_titanic.csv')