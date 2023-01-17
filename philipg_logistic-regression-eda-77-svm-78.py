##################Problem statement#######################

#Titanic: Machine Learning from Disaster

#some group of people were more likely to escape in the Titanic sinking.

#we need to find out what sorts of people are likely to survive.

#Lets use simple logistic regression and see if it works.



#I have referred to some other kernels and whatever I have written below are as per my understanding.
##################Approach##############################

#1. understanding the problem statement

#2. understanding the data

#3. Data preparation and missing value imputation

#4. Exploratory data analysis

#5. Feature engineering.

#6. Model building

#7. Training the model to get the correct parameters.

#8.Testing the mode
#import the necessary packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

import sklearn.model_selection

import sklearn.svm as svm

from sklearn.linear_model import LogisticRegression

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn import metrics

import os



############# 1. Understanding the data ##############

#read the files

train_df=pd.read_csv('../input/train.csv')

test_df=pd.read_csv('../input/test.csv')
#check the shape of the data

print('Train data shape',train_df.shape) #

print('Test data shape',test_df.shape)
train_df.head()
train_df.info()
#There are some categorical variables like Pclass,Sex,Age,SibSp,Parch,cabin,embarked.

#Continuous variable in the dataset is fare.

#There are some clumns like PassengerId,Name,Ticket which seems to be not useful.
################## 2.Data preparation and missing value imputation ######################
#lets check for the missing values in the data 

train_df.isnull().sum()
test_df.isnull().sum()
#There are missing values in the data for the age and cabin.

#lets impute them with relevant values
#first impute the field Age with relevant values
train_df['Age'].dropna(inplace=True)

sns.distplot(train_df['Age'])
#It is not a perfect normal distribution. the Age is right skewed.



#we have to use either mean or median to impute

print('The mean age is %.2f' %(train_df['Age'].mean(skipna=True)))

print('The median age is %.2f' %(train_df['Age'].median(skipna=True)))



#Both the values are almost same. Since it is right skewed the mean might give a biased result.

#so lets take the median. 

#To be more specific on the missing value imputation lets use the below approach.
#find the median values specific to Sex and class

print('pclass1:Sex_Male : ',train_df[(train_df.Pclass==1) & (train_df.Sex=='male')]['Age'].median())

print('pclass2:Sex_Male : ',train_df[(train_df.Pclass==2) & (train_df.Sex=='male')]['Age'].median())

print('pclass3:Sex_Male : ',train_df[(train_df.Pclass==3) & (train_df.Sex=='male')]['Age'].median())



print('pclass1:Sex_feMale : ',train_df[(train_df.Pclass==1) & (train_df.Sex=='female')]['Age'].median())

print('pclass2:Sex_feMale : ',train_df[(train_df.Pclass==2) & (train_df.Sex=='female')]['Age'].median())

print('pclass3:Sex_feMale : ',train_df[(train_df.Pclass==3) & (train_df.Sex=='female')]['Age'].median())
#lets impute the values

def impute_age(cols):

    age=cols[0]

    sex=cols[1]

    pclass=cols[2]

    if pd.isnull(age):

        if sex=='male':

            if pclass==1:

                return 40

            elif pclass==2:

                return 30

            elif pclass==3:

                return 25

            else:

                print('pclass not in 1,2,3',pclass)

                return np.nan

        elif sex=='female':

            if pclass==1:

                return 35

            elif pclass==2:

                return 28

            elif pclass==3:

                return 21.5

            else:

                print('pclass not in 1,2,3',pclass)

                return np.nan

    

        else:

            print('Sex not in male or female',sex)

    else:

        return age

    

    
train_data=train_df.copy()

train_data['Age']=train_data[['Age','Sex','Pclass']].apply(impute_age,axis=1)
test_data=test_df.copy()

test_data['Age']=test_data[['Age','Sex','Pclass']].apply(impute_age,axis=1)
# next lets see the missing values in the cabin

print('missing "Cabin" records is %.2f%%' %((train_df['Cabin'].isnull().sum()/train_df.shape[0])*100))
#since majority of the values are missing in this field it is better to remove that field.

train_data.drop('Cabin',axis=1,inplace=True)
test_data.drop('Cabin',axis=1,inplace=True)
#next lets see the missing values in the Embarked

#as seen above only two records have missing values.So lets impute than with the most frequently occuring value

train_df['Embarked'].value_counts()
sns.countplot(data=train_df,x='Embarked')
#embarked as 'S' is the most frequently occuring one.

train_df['Embarked'].value_counts().idxmax()
train_data['Embarked'].fillna(train_df['Embarked'].value_counts().idxmax(),inplace=True)
test_data['Embarked'].fillna(test_df['Embarked'].value_counts().idxmax(),inplace=True)
train_data.isnull().sum()
#we are done with the missing value imputation. There os no other data preparation needed now.
##########Exploratory Data analysis ######################

#this step will help you in the feature engineering.

#EDA will show some hidden patterns in the data.
#lets define some hypothesis and we can confirm that using the exploratory data analysis



#H1; Passengers who are on the premium class is likely escape compared to other lower class passengers.

#H2: Female passengers are likely to survive compared to male.

#H3: Passengers who are kids are likely to escape



#lets check above are true or not and also try to add more features.
#Lets see the Age field

#check the distribution of survived and not by age.



plt.figure(figsize=(30,8))

ax=sns.distplot(train_data["Age"][train_data.Survived == 1])

sns.distplot(train_data['Age'][train_data.Survived==0])

plt.legend(['Survived','Died'])

ax.set(xlabel='Age')

plt.xlim(0,85)

plt.show()
#lets plot a bar chart

plt.figure(figsize=(50,8))

avg_survival_byage = train_data[["Age", "Survived"]].groupby(['Age'], as_index=False).mean()

g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage, color="LightSeaGreen")

plt.show()



#looks like passengers with less than 16 are likely to escape

#H3 is also correct.
#Now look at the passenger class field

# survival by passenger class

sns.barplot('Pclass','Survived',data=train_data)



#passengers in the class 1 are likely to survive more compared to 2 and 3. H1 is true
#lets check the survival by female

sns.barplot('Sex','Survived',data=train_data)



#Female passengers are more likely to survive.H2 is also true

#lets check the field sibsp

sns.barplot('SibSp','Survived',data=train_data)



#Avg survival by age

#plt.figure(figsize=(20,8))

#avg_survival_bysibsp = train_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()

#g=sns.barplot(x='SibSp',y='Survived',data=avg_survival_bysibsp)
#lets check the field parch

sns.barplot('SibSp','Survived',data=train_data)
#individuals travellling alone are more likely to die.


#now lets explore the fare

plt.figure(figsize=(20,8))

ax=sns.kdeplot(train_data['Fare'][train_data.Survived==1],shade=True)

sns.kdeplot(train_data['Fare'][train_data.Survived==0],shade=True)

plt.legend(['Survived','Died'])

ax.set(xlabel='Fare')

plt.show()
#Those people who paid less for the tickets are more likely to die.
#now lets look at the  field embarked

sns.barplot('Embarked','Survived',data=train_data)
#Those people who embarked from C are more likely to escape
#Now we are done with the exploratory data analysis.

#Below are the stories which we can use in feature creation
# story1. passengers with age below <16  are more likely to escape

# story2. passengers travelling in the first class are more likely to escape. 

# story3. female passengers are more likely to escape 

# story4. people who travel alone are less likely to escape

# story5. people who paid less fare are less likely to escape

# story6. people who embarked from C are more likely to escape
#lets ensure we have the variables to support the above stories
# story1. create a feature age<16 to support the story.

train_data['Age<16']=np.where(train_data['Age']<16,1,0)
test_data['Age<16']=np.where(test_data['Age']<16,1,0)
# story2. passenger class is a categorical variable. lets create dummy values.

# story3. sex is a categorical variable.Lets createa a dummy variable.

# story6. Embarked is a categorical variable. Lets create a dummy variable.





train_data=pd.get_dummies(train_data,columns=["Pclass","Embarked",'Sex'])

train_data.head()
test_data=pd.get_dummies(test_data,columns=["Pclass","Embarked",'Sex'])
#story 4: Lets create a field for whether a person is travelling alone or group

train_data['Travel_group']=np.where(train_data['SibSp']+train_data['Parch']>0,1,0)
test_data['Travel_group']=np.where(test_data['SibSp']+test_data['Parch']>0,1,0)
#story 5: lets use the field fair.
#now lets see the structure of the data

train_data.info()
#There are some fields which are not useful.lets remove them

train_data.drop('Name',axis=1,inplace=True)

train_data.drop('SibSp',axis=1,inplace=True)

train_data.drop('PassengerId',axis=1,inplace=True)

train_data.drop('Parch',axis=1,inplace=True)

train_data.drop('Ticket',axis=1,inplace=True)

train_data.drop('Age',axis=1,inplace=True)

train_data.drop('Sex_male',axis=1,inplace=True)

train_data.drop('Pclass_3',axis=1,inplace=True)

train_data.drop('Embarked_Q',axis=1,inplace=True)

#There are some fields which are not useful.lets remove them

test_data.drop('Name',axis=1,inplace=True)

test_data.drop('SibSp',axis=1,inplace=True)

test_data.drop('PassengerId',axis=1,inplace=True)

test_data.drop('Parch',axis=1,inplace=True)

test_data.drop('Ticket',axis=1,inplace=True)

test_data.drop('Age',axis=1,inplace=True)

test_data.drop('Sex_male',axis=1,inplace=True)

test_data.drop('Pclass_3',axis=1,inplace=True)

test_data.drop('Embarked_Q',axis=1,inplace=True)
train_data.head()
#now we have the feature engineered dataset available for modelling.
#Lets divide the training set itself to train and test so that we can test the model output in the 
########### model building #####################

#calculating the logloss score



#lower log loss value means better predictions 

#log loss is -1*the log of the likelihood function 



#what is likelihood 



#assume a model predicts [.8,.4,.1] for three houses. The first two houses were sold and the last one was not sold 

#so the actual outcome could be represented as [1,1,0]



#the first house sold and the model said that was likely 80%. so the likelihood after looking at one prediction is 80%



#The second house sold and the model said that is likely 40%



#probability of multiple independent events is the product of their individual probabilities 



#so we get the combined likelihood by multiplying both probabilities 

#that is .8*.4 =.32



#now go to the third prediction..our model said there is only 10% likely to sell the house. 90%not likely to sell.Observed outcome

#of not selling was 90% likeli according to the model. so multiply .32 with .9 = .288



#step through all the predictions. each time the find the probability associated with the outcome that actually occured and

#multiply that with the previous result 



#each prediction is between 1 and 0. if you continue multiplying then the values will become vary small .

#so instead keep track of the log of likelihood . this range is easy to track. 



#multiply with -1 to maintain a common convention that lower scores are better. 



#The likelihood function answers the question "How likely did the model think the actually observed set of outcomes was." 

#Lets create a logit model.using logit model we are doing log odds. log odds is having linear combination of the variables.

#it is easy to interpret the coefficients similar to multiple linear regression.
cols=['Fare','Age<16','Pclass_1','Pclass_2','Embarked_C','Embarked_S','Sex_female','Travel_group']
train_data_x1=train_data[cols]

train_data_y1=train_data['Survived']

logit_model_1=sm.Logit(train_data_y1,train_data_x1).fit()
print(logit_model_1.summary2())
vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(train_data_x1.values, i) for i in range(train_data_x1.shape[1])]

vif["features"] = train_data_x1.columns

vif
#fare is having the highest vif.

#also the p value of fare is more than .05

#lets remove the field fare and check the model

train_data_x2=train_data[['Age<16','Pclass_1','Pclass_2','Embarked_C','Embarked_S','Sex_female','Travel_group']]

train_data_y2=train_data['Survived']

logit_model_2=sm.Logit(train_data_y2,train_data_x2).fit()

print(logit_model_2.summary2())
vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(train_data_x2.values, i) for i in range(train_data_x2.shape[1])]

vif["features"] = train_data_x2.columns

vif
#Travel group is having the highest vif.

#also the p value of fare is more than .05

#lets remove the field TravelGroup and check the model

train_data_x3=train_data[['Age<16','Pclass_1','Pclass_2','Embarked_C','Embarked_S','Sex_female','Travel_group']]

train_data_y3=train_data['Survived']

logit_model_3=sm.Logit(train_data_y3,train_data_x3).fit()

print(logit_model_3.summary2())
#now we see all the fields are relevant.
logreg=LogisticRegression()

logreg.fit(train_data_x3,train_data_y3)

Predict_y=logreg.predict(train_data_x3)

cnf_matrix=metrics.confusion_matrix(train_data_y3,Predict_y)
class_names=[0,1]

fig,ax=plt.subplots()

tick_marks=np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,cmap="YlGnBu",fmt='g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('confusion matrix',y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')



print('accuracy',metrics.accuracy_score(train_data_y3,Predict_y))

print('precision',metrics.precision_score(train_data_y3,Predict_y))

print('recall',metrics.recall_score(train_data_y3,Predict_y))
test_data_x3=test_data[['Age<16','Pclass_1','Pclass_2','Embarked_C','Embarked_S','Sex_female','Travel_group']]

test_data_y3=logreg.predict(test_data_x3)
test_data_x3['Survived']=test_data_y3

test_data_x3['PassengerID']=test_df['PassengerId']
submission=test_data_x3[['PassengerID','Survived']]
submission.to_csv("submission.csv", index=False)
################### Lets check the performance of the SVM ################################3

#I have used the default kernel 'rbf'.

#gamma used is 2
sup_vec=svm.SVC(gamma=1.8) #default kernel is 'rbf'.

sup_vec.fit(train_data_x3,train_data_y3)
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC



clf = sklearn.linear_model.Ridge(alpha=.5)
prediction_svm=sup_vec.predict(train_data_x3)
svm_cnf_matrix=sklearn.metrics.confusion_matrix(train_data_y3,prediction_svm)
print('accuracy',metrics.accuracy_score(train_data_y3,prediction_svm))
class_names=[0,1]

fig,ax=plt.subplots()

tick_marks=np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(svm_cnf_matrix),annot=True,cmap="YlGnBu",fmt='g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('SVM_confusion matrix',y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')



print('accuracy',metrics.accuracy_score(train_data_y3,prediction_svm))

print('precision',metrics.precision_score(train_data_y3,prediction_svm))

print('recall',metrics.recall_score(train_data_y3,prediction_svm))
test_data_x3=test_data[['Age<16','Pclass_1','Pclass_2','Embarked_C','Embarked_S','Sex_female','Travel_group']]
test_data_y3_svm=sup_vec.predict(test_data_x3)
test_data_x3['Survived']=test_data_y3_svm

test_data_x3['PassengerID']=test_df['PassengerId']
submission=test_data_x3[['PassengerID','Survived']]
submission.to_csv("submission.csv", index=False)
#SVM Understanding ..

#SVM mainly used for classification.Plot each data item as a point in the n-dimensional space.n is the number of features.



#value of each feature being the value of each coordinate. perform classification by finding the hyperplane that separates the 

#two classes very well.



#support vectors are the coordinates of each of the datapoint.



#The separator hyperplane is the one with max distance from the closest support vectors from both category. 



#How to get the the best hyperplane that differentiate the classes?

#Maximize the distance  between nearest datapoint and the hyperplane will help us to find the best hyperplane.This distance is-

#called margin. 



#if we select a hyperplane with low margin then there is high chances of misclassification.



#Note : SVM select the hyperplane that classify the classes accurately before maximizing the margin.



#SVM has a feature to ignore the outliers and then maximize the margin.Hence SVM is robust to outliers 



#SVM has a feature called Kernel trick. These are functions which takes lower dimensional input space and transform that to 

#higher dimensional space. It converts non separable problems to separable problems.This is kernel trick 



#How to tune parameters for svm?. 

#kernel,gamma and C are the importance parameters in a model. 



#Kernel: 

#There are kernel like, “linear”, “rbf”,”poly” and others . 'rbf' and 'poly' are used for non linear classification. 

#default value as rbf.



#gamma: kernel coefficient for 'rbf','poly' and 'sigmoid'.Higher the value of gamma it will try to overfit the data as per the 

#training data and it can cause overfitting. 



#C : penalty parameter C of the error term. Trade off between smooth decision boundary and classifying the training points correctly


