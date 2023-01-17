import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier

import seaborn as sns
%matplotlib inline

##Train Data set Details
#Survived : 0= No ; 1 = Yes
#pClass: Passenger Class: 1, 2 or 3
#Name, Sex, Age
#SibSp: No of siblings/spouses onboard
#Parch: No of parents/children onboard
#Tickets, Fare, Cabin
#Embarked: Port where embarked. C: Cherboug, Q: Queenstown, S: Southampton
#Read the csv files
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
#Create a consolidated data set for better accuracy for Impute
tot_df = train_df.append(test_df)
print(train_df.shape)
print(test_df.shape)
train_df.describe(include = 'all')

#Keeping the Test DF Passenger Id for prediction at later stage.
passenger_id = test_df['PassengerId']
#Drop PassengerId as this doesn't add value. 
train_df.drop(['PassengerId'], axis = 1, inplace = True)
test_df.drop(['PassengerId'], axis = 1, inplace = True)
tot_df.drop(['PassengerId'], axis = 1, inplace = True)

#Since only one Fare is null, replaced it with mean
test_df.Fare.fillna(test_df.Fare.mean(),inplace = True)
print("Count of Nulls by Field Name in Train set")
print(train_df.isnull().sum())
print("Count of Nulls by Field Name in Test set")
print(test_df.isnull().sum())

#Setting Sex as 1 or 0 for Correlation Plot. 
train_df['Sex'] = train_df.Sex.apply(lambda x: 0 if x == "female" else 1)
test_df['Sex'] = test_df.Sex.apply(lambda x: 0 if x == "female" else 1)
tot_df['Sex'] = tot_df.Sex.apply(lambda x: 0 if x == "female" else 1)

#Boxplot to check Fare outlier data by Class
sns.boxplot(x='Pclass',y='Fare', data = train_df)

#Since only one record is an outlier, we consider all other records in the train set
#train_df = train_df[train_df['Fare']<300]
train_df.shape

#For age calculation, we should take the consolidated dataframe
#If we try to separate the Title from name, we might get an indicator of age
for nm in tot_df['Name']:
    tot_df['Title'] = tot_df['Name'].str.extract('([A-Za-z]+)\.',expand = True)

tot_df['Title'].value_counts()

#There are multiple titles with little count of occurrence. We'll map them to existing Titles
#Mlle : French for little Madame
#Dona: honor word for ladies
#Johnkheer: lowest rank within nobility
#Reverend and Dr are assumed to have higher age and hence not mapped to Mr
mapping = { 'Col': 'Mr', 'Major': 'Mr','Mlle':'Miss','Ms': 'Miss','Capt': 'Mr', 'Countess': 'Mrs', 
            'Lady': 'Mrs','Sir': 'Mr', 'Don': 'Mr', 'Dona': 'Mrs', 'Jonkheer': 'Mr', 'Mme': 'Miss'}
tot_df.replace({'Title': mapping}, inplace=True)

#Distribute the titles on train and test dataset
train_df['Title']=tot_df['Title'][:891]
test_df['Title']=tot_df['Title'][891:]
tot_df['Title'].value_counts()

#Grouping the Age by titles, we take the median age for each title. 
titles = ['Mr','Miss','Mrs','Master','Dr','Rev']
for title in titles:
    age_imp = tot_df.groupby('Title')['Age'].median()[titles.index(title)]
    tot_df.loc[(tot_df['Age'].isnull()) & (tot_df['Title'] == title), 'Age'] = age_imp 
tot_df.isnull().sum()

#Setting the Imputed values on Train and Test Data
train_df['Age']=tot_df['Age'][:891]
test_df['Age'] = tot_df['Age'][891:]
train_df.isnull().sum()

#There are two nulls for Embarked Location. 
tot_df.fillna(tot_df['Embarked'].value_counts().index[0])
tot_df.isnull().sum()

#Setting Embarked value for Tot DF. Not required for Test DF as there are no nulls. 
train_df['Embarked'] = tot_df['Embarked'][:891]
train_df.isnull().sum()

#There are lot of nulls in Cabin. Creating HasCabin column for those passengers who have been allotted Cabins. #
train_df['HasCabin'] = train_df.Cabin.apply(lambda x: 0 if pd.isnull(x) else 1)
test_df['HasCabin'] = test_df.Cabin.apply(lambda x: 0 if pd.isnull(x) else 1)

train_df.describe(include='all')

train_df.groupby('Survived').mean()

train_df.groupby('Sex').mean()

#Checking to see the correlation plots on Train set
plt.subplots(figsize = (15,8))
sns.heatmap(train_df.corr(), annot=True,cmap="BrBG")

#Getting a Pairplot to observe behaviour by feature pairs.
pp = sns.pairplot(train_df[[u'Survived', u'Pclass', u'Sex', u'Age',u'SibSp', u'Parch', u'Fare', u'Embarked',
       u'HasCabin', u'Title']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
pp.set(xticklabels=[])


plt.subplots(figsize = (15,8))
sns.set(style="darkgrid")
bp = sns.barplot(x = "Sex", y = "Survived", data=train_df)
plt.title("Survived Passengers", fontsize = 25)
labels = ['Female', 'Male']
plt.ylabel("% of Passengers survived", fontsize = 15)
plt.xlabel("Gender",fontsize = 15)
#plt.xticks(sorted(train_df.Sex.unique()), labels)
for p in bp.patches:
    height = p.get_height()
    bp.text(p.get_x()+p.get_width()/2.,height+0.04, '{:1.2f}'.format(height),ha="center") 

plt.subplots(figsize = (15,8))
sns.set(style="darkgrid")
bp = sns.barplot(x = "Pclass", y = "Survived", data=train_df)
plt.title("Survived Passengers", fontsize = 25)
labels = ['1', '2','3']
plt.ylabel("% of Passengers survived", fontsize = 15)
plt.xlabel("Passenger Class",fontsize = 15)
for p in bp.patches:
    height = p.get_height()
    bp.text(p.get_x()+p.get_width()/2.,height+0.05, '{:1.2f}'.format(height),ha="center") 

plt.subplots(figsize = (15,8))
plt.title("Survived Passengers vs Fare", fontsize = 25)
sns.kdeplot(train_df.loc[(train_df['Survived']==0),'Fare'], color ='r',shade = True, label = "Did not Survive")
sns.kdeplot(train_df.loc[(train_df['Survived']==1),'Fare'], color ='b',shade = True, label = "Survived")

plt.subplots(figsize = (15,8))
plt.title("Survived Passengers vs Age", fontsize = 25)
sns.kdeplot(train_df.loc[(train_df['Survived']==0),'Age'], color ='r',shade = True, label = "Did not Survive")
sns.kdeplot(train_df.loc[(train_df['Survived']==1),'Age'], color ='b',shade = True, label = "Survived")

plt.subplots(figsize = (15,8))
plt.title("Survived Passengers vs Sibling & Spouse", fontsize = 25)
sns.kdeplot(train_df.loc[(train_df['Survived']==0),'SibSp'], color ='r',shade = True, label = "Did not Survive")
sns.kdeplot(train_df.loc[(train_df['Survived']==1),'SibSp'], color ='b',shade = True, label = "Survived")

#Define a function to identify family type based on number of members in family
def family_type(count):
    ftype = ''
    if(count <=1):
        ftype = 'Single'
    elif(count <=4):
        ftype = 'Small'
    elif(count >4):
        ftype = 'Large'
    return ftype
#Count of self, SiblingSpouse and ParentsChildren
train_df['Family_Type']= (train_df['SibSp']+train_df['Parch']+1).map(family_type)
test_df['Family_Type']= (test_df['SibSp']+test_df['Parch']+1).map(family_type)

#Define a function to identify Age Group based on Age
def age_group_fun(age):
    agroup = ''
    if age <=1:
        agroup = 'Infant'
    elif age <=3:
        agroup = 'Toddler'
    elif age <=12:
        agroup = 'Child'
    elif age <=18:
        agroup = 'Teen'
    elif age <=30:
        agroup = 'Young'
    elif age <=45:
        agroup = 'Adult'
    elif age <=65:
        agroup = 'Middle_Age'
    else:
        agroup = 'Old'
    return agroup

#Setting the Age Group by calling the function
train_df['Age_Group']= (train_df['Age']).map(age_group_fun)
test_df['Age_Group']= (test_df['Age']).map(age_group_fun)

plt.subplots(figsize = (15,8))
sns.set(style="darkgrid")
bp = sns.barplot(x = "Family_Type", y = "Fare", data=train_df)
plt.title("Fare vs Family type", fontsize = 25)
plt.ylabel("Fare", fontsize = 15)
plt.xlabel("Family Type",fontsize = 15)
for p in bp.patches:
    height = p.get_height()
    bp.text(p.get_x()+p.get_width()/2.,height+0.05, '{:1.2f}'.format(height),ha="center") 
    
plt.subplots(figsize = (15,8))
sns.set(style="darkgrid")
bp = sns.barplot(x = "Family_Type", y = "Survived", data=train_df)
plt.title("Survived Passengers", fontsize = 25)
plt.ylabel("% of Passengers survived", fontsize = 15)
plt.xlabel("Family Type",fontsize = 15)
for p in bp.patches:
    height = p.get_height()
    bp.text(p.get_x()+p.get_width()/2.,height+0.05, '{:1.2f}'.format(height),ha="center") 
    
train_df['Average_Fare'] = train_df.Fare/(train_df.SibSp+train_df.Parch+1)
test_df['Average_Fare'] = test_df.Fare/(test_df.SibSp+test_df.Parch+1)

#Running the above plot for the AverageFare
plt.subplots(figsize = (15,8))
sns.set(style="darkgrid")
bp = sns.barplot(x = "Family_Type", y = "Average_Fare", data=train_df)
plt.title("Average Fare vs Family type", fontsize = 25)
plt.ylabel("Avg Fare", fontsize = 15)
plt.xlabel("Family Type",fontsize = 15)
for p in bp.patches:
    height = p.get_height()
    bp.text(p.get_x()+p.get_width()/2.,height+0.05, '{:1.2f}'.format(height),ha="center") 
    
    
#checking box plot for Average Fares. Looks like Singles from 1st class paid higher fares.
print("Mean", train_df.Average_Fare.mean())
print("Median", train_df.Average_Fare.median())
print("Mode",train_df.Average_Fare.mode())
plt.subplots(figsize = (15,8))
sns.boxplot(x='Family_Type',y='Average_Fare', data = train_df)

#Define a function to identify Fare Group based on Fare
def fare_group_fun(fare):
    fgroup = ''
    if fare <=8:
        fgroup = 'Low'
    elif fare <=19:
        fgroup = 'Medium'
    elif fare <=50:
        fgroup = 'High'
    else:
        fgroup = 'Exorbitant'
    return fgroup

#Setting the Age Group by calling the function
train_df['Fare_Group']= (train_df['Average_Fare']).map(fare_group_fun)
test_df['Fare_Group']= (test_df['Average_Fare']).map(fare_group_fun)

train_df = pd.get_dummies(train_df, columns=['Pclass','Embarked','Title', 'Family_Type','Age_Group', 'Fare_Group'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Pclass','Embarked','Title', 'Family_Type','Age_Group', 'Fare_Group'], drop_first=True)
train_df.drop(['Name','SibSp','Parch','Cabin','Ticket','Fare'], axis=1, inplace=True)
test_df.drop(['Name','SibSp','Parch','Cabin','Ticket','Fare'], axis=1, inplace=True)

y = train_df['Survived']
train_df_reg = train_df.drop('Survived', 1)

train_df_reg = StandardScaler().fit_transform(train_df_reg)
test_df_reg = StandardScaler().fit_transform(test_df)


#Splitting Train set into Train and Validate
X_train, X_validate, y_train, y_validate = train_test_split(train_df_reg,y,test_size = 0.20, random_state = 1000)

logreg = LogisticRegression(solver='liblinear', penalty='l1')
logreg.fit(X_train,y_train)
predict=logreg.predict(X_validate)
print(accuracy_score(y_validate,predict))
print(confusion_matrix(y_validate,predict))
print(precision_score(y_validate,predict))
print(recall_score(y_validate,predict))

dectree = DecisionTreeClassifier( criterion="entropy",
                                 max_depth=5,
                                class_weight = 'balanced',
                                min_weight_fraction_leaf = 0.009,
                                random_state=2000)
dectree.fit(X_train, y_train)
predict = dectree.predict(X_validate)
print(accuracy_score(y_validate, predict))
print(confusion_matrix(y_validate,predict))
print(precision_score(y_validate,predict))
print(recall_score(y_validate,predict))


randomforest = RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_split=20,max_features=0.2, min_samples_leaf=8,random_state=20)
randomforest.fit(X_train, y_train)
predict = randomforest.predict(X_validate)
print(accuracy_score(y_validate, predict))
print(confusion_matrix(y_validate,predict))
print(precision_score(y_validate,predict))
print(recall_score(y_validate,predict))

xgb=XGBClassifier(max_depth=2, n_estimators=700, learning_rate=0.009,nthread=-1,subsample=1,colsample_bytree=0.8)
xgb.fit(X_train,y_train)
predict=logreg.predict(X_validate)
print(accuracy_score(y_validate,predict))
print(confusion_matrix(y_validate,predict))
print(precision_score(y_validate,predict))
print(recall_score(y_validate,predict))

svc=SVC(probability=True)
svc.fit(X_train,y_train)
predict=logreg.predict(X_validate)
print(accuracy_score(y_validate,predict))
print(confusion_matrix(y_validate,predict))
print(precision_score(y_validate,predict))
print(recall_score(y_validate,predict))

voting_classifier = VotingClassifier(estimators=[
    ('logreg',logreg), 
    ('decision_tree',dectree), 
    ('random_forest', randomforest),
    ('XGB Classifier', xgb)])
voting_classifier.fit(X_train,y_train)
predict = voting_classifier.predict(X_validate)
print(accuracy_score(y_validate,predict))
print(confusion_matrix(y_validate,predict))
print(precision_score(y_validate,predict))
print(recall_score(y_validate,predict))

y_predict=dectree.predict(test_df_reg)
prediction = pd.DataFrame(pd.DataFrame({
        "PassengerId": passenger_id,
        "Survived": y_predict
    }))

prediction.to_csv("Predictions.csv", index = False)




