import numpy as np

import pandas as pd

import seaborn as sns

import math

import matplotlib.pyplot as plt

%matplotlib inline
training_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')
training_data.head(10)
# Let's check data types, we are dealing with

training_data.info()
# so over here we can see there are 5 variables with data type as integer, 2 with float, & 5 as object.

# Also Column cabin has more than 77% missing values, so ill drop this column later. Column Embarked has 2 missing values which can be easily imputed

# Challenge is dealing with age column.

# now we will perform a basic check on features



training_data.describe()



# This will show us 1, 2, 3 quartiles, how spread our data is and also we can figure out some basic ouliers through this.

# For Eg: minimum fare is 0, who are those and did they survive?
# we will analyze categorical variables:

sns.countplot(x= 'Survived', data = training_data)



# count of survival is less.
# Now we will check survival rate with respect to gender:

sns.countplot(x= 'Survived',hue='Sex',  data = training_data)

# seeing plot we can say that survival rate of female is high.
# Now we will check survival rate with respect to Pclass:

sns.countplot(x= 'Survived',hue='Pclass',  data = training_data)

# seeing plot we can say that survival rate of passengers in 1st class is high on the other side, count of survival for passengers of class-3 was very low 
# Now we will check survival rate with respect to Embarked

sns.countplot(x= 'Survived',hue='Embarked',  data = training_data)

# seeing plot we can say that maximum people are from S = Southampton, survival rate of people from C = Cherbourg is decent.
# Now we will check Pclass with Embarked

sns.countplot(x= 'Pclass',hue='Embarked',  data = training_data)

# the reason above we saw S = Southampton had low rate of survival because majority of them are in class 3, and survival rate of people from C = Cherbourg is decent because majority fo them belonged to 1st class.
# Now we will check Collinearity among to avoid Multicollinearity issues.

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(training_data.corr(), annot=True, linewidths=0.5, fmt='.2f',ax=ax)

ax.set_ylim(7, 0)



# Eg: so it's common that higher the pclass, higher is the fare. Hence we would drop one of them to avoid redundant data, also passenger_Id will be dropped since not contributing to dependent variable.

# once all our categorical data has been converted we can also check Variance inflation factor, and correlation again.
#Survived by Sex and Age



ploteg = sns.FacetGrid(training_data,size=5, col="Sex", row="Embarked", margin_titles=True, hue = "Survived"

                  )

plotegg = ploteg.map(plt.hist, "Age", edgecolor = 'white').add_legend();

ploteg.fig.suptitle("Survived by Sex and Age", size = 25)

plt.subplots_adjust(top=0.90)
# let's check missing values in both training and test data



training_data.info()

print('*' *100)

test_data.info()

# Let's deal with fare column in test data

test_data[test_data['Fare'].isnull()]
sns.catplot(x="Pclass", y="Fare",hue = "Embarked", kind="swarm", data=test_data);

# we can clearly see that majority of people who are from 'S' and 'Q' are in 2nd and 3rd class, also they are paying less fare.
#So we will group by Pclass, Embarked, and Parch after which we will impute fare with median values of people having same values.

median_far = test_data.groupby(['Pclass','Embarked','Parch']).Fare.median()[3]['S'][0]

test_data['Fare'] = test_data['Fare'].fillna(median_far)
# The above Graph will help us to deal with missing values in Embarked

training_data[training_data['Embarked'].isnull()]
# seeing the above graph we can impute these values by 'C'

training_data['Embarked'] = training_data['Embarked'].fillna('C')
training_data[training_data['Embarked'].isnull()]
# Let's deal With Age column Now

training_data[training_data['Age'].isnull()]

# As of now i'll impute age column with Median based on gender and check performance of the model and later on find a different strategy to deal with it.
median_male_age = training_data.groupby(['Sex']).Age.median()['male']

median_male_age



median_female_age = training_data.groupby(['Sex']).Age.median()['female']

print(median_male_age)

print('*'*100)

print(median_female_age)
copy = training_data.copy()



conditions = [copy['Sex'] == "male", copy['Sex'] == "female"]

values = [29.0, 27.0]



# apply logic where company_type is null

copy['Age'] = np.where(copy['Age'].isnull(),

                              np.select(conditions, values),

                              copy['Age'])
#let's apply above logic on Training_data and Test_data



conditions = [training_data['Sex'] == "male", training_data['Sex'] == "female"]

values = [29.0, 27.0]

# apply logic where company_type is null

training_data['Age'] = np.where(training_data['Age'].isnull(),

                              np.select(conditions, values),

                              training_data['Age'])
training_data.info()

# As we can see we are done with training data and now let's apply same logic on test data
# Test Data, first we need to find median for age column in test data



testing_male = test_data.groupby(['Sex']).Age.median()['male']



testing_female= test_data.groupby(['Sex']).Age.median()['female']

print(testing_male)

print('*'*100)

print(testing_female)
# since Median comes to same i'll apply same age to all missing values

test_data['Age'] = test_data['Age'].fillna(27)
## let's drop collumns which we won't we adding in our model

training_data = training_data.drop('PassengerId', axis=1)

training_data = training_data.drop('Cabin', axis=1)

training_data = training_data.drop('Ticket', axis=1)
# Drop them from test data also

test_data.drop(['PassengerId', 'Cabin'], axis=1, inplace=True)

test_data = test_data.drop('Ticket', axis=1)
train_categorical_features = ['Pclass', 'Sex', 'Embarked']

for feature in train_categorical_features:

    dummies = pd.get_dummies(training_data[feature]).add_prefix(feature + '_')

    training_data = training_data.join(dummies)
training_data.head(5)
test_categorical_features = ['Pclass', 'Sex', 'Embarked']

for feature in test_categorical_features:

    dummies = pd.get_dummies(test_data[feature]).add_prefix(feature + '_')

    test_data = test_data.join(dummies)

    

# Found this particular code from, source: https://www.kaggle.com/reighns/titanic-a-complete-beginner-s-guide
# let's drop useless features.

drop_column = ['Pclass','Name','Sex', 'Embarked']

training_data.drop(drop_column, axis=1, inplace = True)



drop_column = ['Pclass','Name','Sex', 'Embarked']

test_data.drop(drop_column, axis=1, inplace = True)
training_data.info()
test_data.info()
f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(training_data.corr(), annot=True, linewidths=1, fmt='.2f',ax=ax)

ax.set_ylim(13, 0)

drop_column = ['Pclass_3','Sex_male', 'Embarked_S']

training_data.drop(drop_column, axis=1, inplace = True)



drop_column = ['Pclass_3','Sex_male', 'Embarked_S']

test_data.drop(drop_column, axis=1, inplace = True)
f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(training_data.corr(), annot=True, linewidths=1, fmt='.2f',ax=ax)

ax.set_ylim(10, 0)
copy_of_training_data = training_data.copy()



import statsmodels.api as sm

from statsmodels.stats import diagnostic as diag

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

#from sklearn.linear_model import LinearRegression

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import accuracy_score 

from sklearn.metrics import mean_absolute_error, accuracy_score
x1 = sm.tools.add_constant(copy_of_training_data)

series_before = pd.Series([variance_inflation_factor(x1.values, i)for i in range(x1.shape[1])], index = x1.columns)

display(series_before)



# the score for the independent features must be below 5 which shows there is no multicollinearity.

# link for more details, source: https://www.youtube.com/watch?v=8DhvVs59It4&list=PLcFcktZ0wnNkMqnUi8zUAPlO_swt-3GiJ&index=2
original_training_set_without_survived = training_data.drop("Survived", axis=1)

orginal_training_set_with_only_survived = training_data["Survived"]
original_training_set_without_survived.shape
orginal_training_set_with_only_survived.shape
X_train, X_test, y_train, y_test = train_test_split(

    original_training_set_without_survived, orginal_training_set_with_only_survived, train_size=0.8, test_size=0.2, random_state=0)



print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
## We will be using standardscaler to transform the data.

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



## transforming "train_x"

X_train = sc.fit_transform(X_train)

## transforming "test_x"

X_test = sc.transform(X_test)



## transforming "The testset"

test_data = sc.transform(test_data)
## Lets call logistic regression



logreg = LogisticRegression()



logreg.fit(X_train,y_train)



y_pred = logreg.predict(X_test)



print ("So, Our accuracy Score is: {}".format(round(accuracy_score(y_test,y_pred),8)))



# Let's predict survival for our Test_Data

testingonunknowndata = logreg.predict(test_data)
# 1. Confusion matrix

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,logreg.predict(X_test))

col=["Predicted Dead","Predicted Survived"]

cm=pd.DataFrame(cm)

cm.columns=["Predicted Dead","Predicted Survived"]

cm.index=["Actual Dead","Actual Survived"]

cm
# 2. Classification report:

from sklearn.metrics import classification_report

target_names = ['class 0', 'class 1']

classification_report(y_test, y_pred,target_names=target_names)

#The report shows the main classification metrics precision, recall and f1-score on a per-class basis.



# For detailed explanation visit, source: https://muthu.co/understanding-the-classification-report-in-sklearn/
# A) Precision – What percent of your predictions were correct? Precision = TP/(TP + FP)

from sklearn.metrics import precision_score

print("Precision score: {}".format(precision_score(y_test,y_pred)))



# B) Recall – What percent of your predictions were correct? Recall = TP/(TP+FN)

from sklearn.metrics import recall_score

print("Recall score: {}".format(recall_score(y_test,y_pred)))



# C) F1 score – What percent of positive predictions were correct? F1 Score = 2*(Recall * Precision) / (Recall + Precision)

from sklearn.metrics import f1_score

print("F1 Score: {}".format(f1_score(y_test,y_pred)))
# Lets plot Roc-Curve of our model



from sklearn.metrics import roc_curve

y_pred_proba = logreg.decision_function(X_test) # because roc curve needs actual labels and predicted probabilities



FPR, TPR, THR = roc_curve(y_test, y_pred_proba)

#Next is draw roc graph.

plt.figure(figsize =[10,9])

plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)',linewidth= 4)

plt.plot([0,1],[0,1], 'k--', linewidth = 4)

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.05])

plt.xlabel('False Positive Rate', fontsize = 17)

plt.ylabel('True Positive Rate', fontsize = 17)

plt.title('ROC for Logistic Regression (Titanic)', fontsize= 17)

plt.show()



# code Source: https://www.kaggle.com/reighns/titanic-a-complete-beginner-s-guide
# Lets check Roc-Auc(Area under Curve) score

from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_pred_proba)
Test_data1 = pd.read_csv('../input/titanic/test.csv')

Logisticoutput = pd.DataFrame({"PassengerId": Test_data1.PassengerId, "Survived":testingonunknowndata})

Logisticoutput.PassengerId = Logisticoutput.PassengerId.astype(int)

Logisticoutput.Survived = Logisticoutput.Survived.astype(int)



Logisticoutput.to_csv("Logisticoutput.csv", index=False)

print("Your submission was successfully saved!")

Logisticoutput.head(10)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

naivebayespred = gnb.predict(X_test)

print ("So, Our accuracy Score is: {}".format(round(accuracy_score(y_test,naivebayespred),8)))
    # 1. Confusion matrix for logistic regression

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,logreg.predict(X_test))

col=["Predicted Dead","Predicted Survived"]

cm=pd.DataFrame(cm)

cm.columns=["Predicted Dead","Predicted Survived"]

cm.index=["Actual Dead","Actual Survived"]

print(cm)



print('*' *100)



    # 2. Confusion matrix for Naive Bayes

from sklearn.metrics import confusion_matrix

cm1=confusion_matrix(y_test,gnb.predict(X_test))

col1=["Predicted Dead","Predicted Survived"]

cm1=pd.DataFrame(cm1)

cm1.columns=["Predicted Dead","Predicted Survived"]

cm1.index=["Actual Dead","Actual Survived"]

print(cm1)

    # 1. For logistic Regression:



# A) Precision – What percent of your predictions were correct? Precision = TP/(TP + FP)

from sklearn.metrics import precision_score

print("Precision score: {}".format(precision_score(y_test,y_pred)))



# B) Recall – What percent of your predictions were correct? Recall = TP/(TP+FN)

from sklearn.metrics import recall_score

print("Recall score: {}".format(recall_score(y_test,y_pred)))



# C) F1 score – What percent of positive predictions were correct? F1 Score = 2*(Recall * Precision) / (Recall + Precision)

from sklearn.metrics import f1_score

print("F1 Score: {}".format(f1_score(y_test,y_pred)))



print('*#'*50)

    # 2. For Naive Bayes Regression:



# A) Precision – What percent of your predictions were correct? Precision = TP/(TP + FP)

from sklearn.metrics import precision_score

print("Precision score: {}".format(precision_score(y_test,naivebayespred)))



# B) Recall – What percent of your predictions were correct? Recall = TP/(TP+FN)

from sklearn.metrics import recall_score

print("Recall score: {}".format(recall_score(y_test,naivebayespred)))



# C) F1 score – What percent of positive predictions were correct? F1 Score = 2*(Recall * Precision) / (Recall + Precision)

from sklearn.metrics import f1_score

print("F1 Score: {}".format(f1_score(y_test,naivebayespred)))
