#data analysis libraries 

import numpy as np

import pandas as pd



#visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#ignore warnings

import warnings

warnings.filterwarnings('ignore')
#import train and test CSV files

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



#take a look at the training data

train.describe(include="all")
#get a list of the features within the dataset

print(train.columns)
#see a sample of the dataset to get an idea of the variables

train.sample(5)
#see a summary of the training dataset

train.describe(include = "all")
#check for any other unusable values

print(pd.isnull(train).sum())
train.hist(bins=10,figsize=(9,7),grid=False);
g = sns.FacetGrid(train, col="Sex", row="Survived", margin_titles=True)

g.map(plt.hist, "Age",color="purple");
g = sns.FacetGrid(train, hue="Survived", col="Pclass", margin_titles=True,

                  palette={1:"seagreen", 0:"gray"})

g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();
g = sns.FacetGrid(train, hue="Survived", col="Sex", margin_titles=True,

                palette="Set1",hue_kws=dict(marker=["^", "v"]))

g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()

plt.subplots_adjust(top=0.8)

g.fig.suptitle('Survival by Gender , Age and Fare');
train.Embarked.value_counts().plot(kind='bar', alpha=0.55)

plt.title("Passengers per boarding location");
sns.set(font_scale=1)

g = sns.factorplot(x="Sex", y="Survived", col="Pclass",

                    data=train, saturation=.5,

                    kind="bar", ci=None, aspect=.6)

(g.set_axis_labels("", "Survival Rate")

    .set_xticklabels(["Men", "Women"])

    .set_titles("{col_name} {col_var}")

    .set(ylim=(0, 1))

    .despine(left=True))  

plt.subplots_adjust(top=0.8)

g.fig.suptitle('How many Men and Women Survived by Passenger Class');
import matplotlib.pyplot as plt

ax = sns.boxplot(x="Survived", y="Age", 

                data=train)

ax = sns.stripplot(x="Survived", y="Age",

                   data=train, jitter=True,

                   edgecolor="gray")

plt.title("Survival by Age",fontsize=12);
train.Age[train.Pclass == 1].plot(kind='kde')    

train.Age[train.Pclass == 2].plot(kind='kde')

train.Age[train.Pclass == 3].plot(kind='kde')

 # plots an axis lable

plt.xlabel("Age")    

plt.title("Age Distribution within classes")

# sets our legend for our graph.

plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') ;
sns.barplot(x="Sex", y="Survived", hue="Pclass", data=train)

plt.ylabel("Survival Rate")

plt.title("Survival Rates Based on Gender and Class")
sns.stripplot(x="Survived", y="Age", data=train, jitter=True)
sns.factorplot("Pclass", "Survived", hue = "Sex", data = train)

plt.show()
pd.crosstab([train["Sex"], train["Survived"]], train["Pclass"], 

            margins = True).style.background_gradient(cmap = "summer_r")
#draw a bar plot of survival by sex

sns.barplot(x="Sex", y="Survived", data=train)



#print percentages of females vs. males that survive

print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)



print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
#draw a bar plot of survival by Pclass

sns.barplot(x="Pclass", y="Survived", data=train)



#print percentage of people by Pclass that survived

print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)



print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)
#draw a bar plot for SibSp vs. survival

sns.barplot(x="SibSp", y="Survived", data=train)



#I won't be printing individual percent values for all of these.

print("Percentage of SibSp = 0 who survived:", train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100)



print("Percentage of SibSp = 1 who survived:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of SibSp = 2 who survived:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)
#draw a bar plot for Parch vs. survival

sns.barplot(x="Parch", y="Survived", data=train)

plt.show()
#sort the ages into logical categories

train["Age"] = train["Age"].fillna(-0.5)

test["Age"] = test["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)



#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=train)

plt.show()
train["CabinBool"] = (train["Cabin"].notnull().astype('int'))

test["CabinBool"] = (test["Cabin"].notnull().astype('int'))



#calculate percentages of CabinBool vs. survived

print("Percentage of CabinBool = 1 who survived:", train["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of CabinBool = 0 who survived:", train["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)

#draw a bar plot of CabinBool vs. survival

sns.barplot(x="CabinBool", y="Survived", data=train)

plt.show()
test.describe(include="all")
#we'll start off by dropping the Cabin feature since not a lot more useful information can be extracted from it.

train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)
#we can also drop the Ticket feature since it's unlikely to yield any useful information

train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)
#now we need to fill in the missing values in the Embarked feature

print("Number of people embarking in Southampton (S):")

southampton = train[train["Embarked"] == "S"].shape[0]

print(southampton)



print("Number of people embarking in Cherbourg (C):")

cherbourg = train[train["Embarked"] == "C"].shape[0]

print(cherbourg)



print("Number of people embarking in Queenstown (Q):")

queenstown = train[train["Embarked"] == "Q"].shape[0]

print(queenstown)
#replacing the missing values in the Embarked feature with S

train = train.fillna({"Embarked": "S"})
#create a combined group of both datasets

combine = [train, test]



#extract a title for each Name in the train and test datasets

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train['Title'], train['Sex'])
#replace various titles with more common names

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',

    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#map each of the title groups to a numerical value

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train.head()
# fill missing age with mode age group for each title

mr_age = train[train["Title"] == 1]["AgeGroup"].mode() #Young Adult

miss_age = train[train["Title"] == 2]["AgeGroup"].mode() #Student

mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() #Adult

master_age = train[train["Title"] == 4]["AgeGroup"].mode() #Baby

royal_age = train[train["Title"] == 5]["AgeGroup"].mode() #Adult

rare_age = train[train["Title"] == 6]["AgeGroup"].mode() #Adult



age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}



#I tried to get this code to work with using .map(), but couldn't.

#I've put down a less elegant, temporary solution for now.

#train = train.fillna({"Age": train["Title"].map(age_title_mapping)})

#test = test.fillna({"Age": test["Title"].map(age_title_mapping)})



for x in range(len(train["AgeGroup"])):

    if train["AgeGroup"][x] == "Unknown":

        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]

        

for x in range(len(test["AgeGroup"])):

    if test["AgeGroup"][x] == "Unknown":

        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]
#map each Age value to a numerical value

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test['AgeGroup'] = test['AgeGroup'].map(age_mapping)



train.head()



#dropping the Age feature for now, might change

train = train.drop(['Age'], axis = 1)

test = test.drop(['Age'], axis = 1)
#drop the name feature since it contains no more useful information.

train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)
#map each Sex value to a numerical value

sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)



train.head()
#map each Embarked value to a numerical value

embarked_mapping = {"S": 1, "C": 2, "Q": 3}

train['Embarked'] = train['Embarked'].map(embarked_mapping)

test['Embarked'] = test['Embarked'].map(embarked_mapping)



train.head()
#fill in missing Fare value in test set based on mean fare for that Pclass 

for x in range(len(test["Fare"])):

    if pd.isnull(test["Fare"][x]):

        pclass = test["Pclass"][x] #Pclass = 3

        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)

        

#map Fare values into groups of numerical values

train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])

test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])



#drop Fare values

train = train.drop(['Fare'], axis = 1)

test = test.drop(['Fare'], axis = 1)
#check train data

train.head()
#check test data

test.head()
from sklearn.model_selection import train_test_split



X = train.drop(['Survived', 'PassengerId'], axis=1)

Y = train["Survived"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.22, random_state = 0)
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred_NB = gaussian.predict(x_test)

acc_gaussian = round(accuracy_score(y_pred_NB, y_test) * 100, 2)

print(acc_gaussian)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_NB))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred_NB)
# Logistic Regression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred_lg = logreg.predict(x_test)

acc_logreg = round(accuracy_score(y_pred_lg, y_test) * 100, 2)

print(acc_logreg)
print(classification_report(y_test, y_pred_lg))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred_lg)
# Support Vector Machines

from sklearn.svm import SVC



svc = SVC()

svc.fit(x_train, y_train)

y_pred_svm = svc.predict(x_test)

acc_svc = round(accuracy_score(y_pred_svm, y_test) * 100, 2)

print(acc_svc)
print(classification_report(y_test, y_pred_svm))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred_svm)
# Linear SVC

from sklearn.svm import LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

y_pred_svc = linear_svc.predict(x_test)

acc_linear_svc = round(accuracy_score(y_pred_svc, y_test) * 100, 2)

print(acc_linear_svc)
print(classification_report(y_test, y_pred_svc))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred_svc)
# Perceptron

from sklearn.linear_model import Perceptron



perceptron = Perceptron()

perceptron.fit(x_train, y_train)

y_pred_per = perceptron.predict(x_test)

acc_perceptron = round(accuracy_score(y_pred_per, y_test) * 100, 2)

print(acc_perceptron)
print(classification_report(y_test, y_pred_per))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred_per)
#Decision Tree

from sklearn.tree import DecisionTreeClassifier



decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)

y_pred_dt = decisiontree.predict(x_test)

acc_decisiontree = round(accuracy_score(y_pred_dt, y_test) * 100, 2)

print(acc_decisiontree)
print(classification_report(y_test, y_pred_dt))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred_dt)
# Random Forest

from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred_rf = randomforest.predict(x_test)

acc_randomforest = round(accuracy_score(y_pred_rf, y_test) * 100, 2)

print(acc_randomforest)
print(classification_report(y_test, y_pred_rf))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred_rf)
# KNN or k-Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

y_pred_knn = knn.predict(x_test)

acc_knn = round(accuracy_score(y_pred_knn, y_test) * 100, 2)

print(acc_knn)
print(classification_report(y_test, y_pred_knn))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred_knn)
# Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(x_train, y_train)

y_pred_sgd = sgd.predict(x_test)

acc_sgd = round(accuracy_score(y_pred_sgd, y_test) * 100, 2)

print(acc_sgd)
print(classification_report(y_test, y_pred_sgd))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred_sgd)
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred_gbc = gbk.predict(x_test)

acc_gbk = round(accuracy_score(y_pred_gbc, y_test) * 100, 2)

print(acc_gbk)
print(classification_report(y_test, y_pred_gbc))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred_gbc)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],

    'Score': [acc_svc, acc_knn, acc_logreg, 

              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,

              acc_sgd, acc_gbk]})

models.sort_values(by='Score', ascending=False)
#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = gbk.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)
#Visualizing the model's ROC curve (**source for graph code given below the plot)

from sklearn.metrics import roc_curve, auc

logreg.fit(x_train, y_train)





 

# Determine the false positive and true positive rates

FPR, TPR, _ = roc_curve(y_test, logreg.predict_proba(x_test)[:,1])

 

# Calculate the AUC



roc_auc = auc(FPR, TPR)

print ('ROC AUC: %0.3f' % roc_auc )

 

# Plot of a ROC curve

plt.figure(figsize=(10,10))

plt.plot(FPR, TPR, label='ROC curve (area = %0.3f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([-0.05, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve (Test Sample Performance)')

plt.legend(loc="lower right")

plt.show()
#Visualizing the model's ROC curve (**source for graph code given below the plot)

from sklearn.metrics import roc_curve, auc

gaussian.fit(x_train, y_train)





 

# Determine the false positive and true positive rates

FPR, TPR, _ = roc_curve(y_test, gaussian.predict_proba(x_test)[:,1])

 

# Calculate the AUC



roc_auc = auc(FPR, TPR)

print ('ROC AUC: %0.3f' % roc_auc )

 

# Plot of a ROC curve

plt.figure(figsize=(10,10))

plt.plot(FPR, TPR, label='ROC curve (area = %0.3f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([-0.05, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve (Test Sample Performance)')

plt.legend(loc="lower right")

plt.show()
from sklearn.metrics import roc_curve, auc

decisiontree.fit(x_train, y_train)





 

# Determine the false positive and true positive rates

FPR, TPR, _ = roc_curve(y_test, decisiontree.predict_proba(x_test)[:,1])

 

# Calculate the AUC



roc_auc = auc(FPR, TPR)

print ('ROC AUC: %0.3f' % roc_auc )

 

# Plot of a ROC curve

plt.figure(figsize=(10,10))

plt.plot(FPR, TPR, label='ROC curve (area = %0.3f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([-0.05, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve (Test Sample Performance)')

plt.legend(loc="lower right")

plt.show()
from sklearn.metrics import roc_curve, auc

randomforest.fit(x_train, y_train)





 

# Determine the false positive and true positive rates

FPR, TPR, _ = roc_curve(y_test, randomforest.predict_proba(x_test)[:,1])

 

# Calculate the AUC



roc_auc = auc(FPR, TPR)

print ('ROC AUC: %0.3f' % roc_auc )

 

# Plot of a ROC curve

plt.figure(figsize=(10,10))

plt.plot(FPR, TPR, label='ROC curve (area = %0.3f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([-0.05, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve (Test Sample Performance)')

plt.legend(loc="lower right")

plt.show()
from sklearn.metrics import roc_curve, auc

knn.fit(x_train, y_train)





 

# Determine the false positive and true positive rates

FPR, TPR, _ = roc_curve(y_test, knn.predict_proba(x_test)[:,1])

 

# Calculate the AUC



roc_auc = auc(FPR, TPR)

print ('ROC AUC: %0.3f' % roc_auc )

 

# Plot of a ROC curve

plt.figure(figsize=(10,10))

plt.plot(FPR, TPR, label='ROC curve (area = %0.3f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([-0.05, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve (Test Sample Performance)')

plt.legend(loc="lower right")

plt.show()
from sklearn.metrics import roc_curve, auc

gbk.fit(x_train, y_train)





 

# Determine the false positive and true positive rates

FPR, TPR, _ = roc_curve(y_test, gbk.predict_proba(x_test)[:,1])

 

# Calculate the AUC



roc_auc = auc(FPR, TPR)

print ('ROC AUC: %0.3f' % roc_auc )

 

# Plot of a ROC curve

plt.figure(figsize=(10,10))

plt.plot(FPR, TPR, label='ROC curve (area = %0.3f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([-0.05, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve (Test Sample Performance)')

plt.legend(loc="lower right")

plt.show()
from sklearn.metrics import roc_curve, roc_auc_score

# Instantiate the classfiers and make a list

classifiers = [LogisticRegression(), 

               GaussianNB(), 

               KNeighborsClassifier(), 

               DecisionTreeClassifier(),

               RandomForestClassifier(),

              GradientBoostingClassifier()]



# Define a result table as a DataFrame

result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])



# Train the models and record the results

for cls in classifiers:

    model = cls.fit(x_train, y_train)

    y_pred = model.predict_proba(x_test)[::,1]

    

    fpr, tpr, _ = roc_curve(y_test,  y_pred)

    auc = roc_auc_score(y_test, y_pred)

    

    result_table = result_table.append({'classifiers':cls.__class__.__name__,

                                        'fpr':fpr, 

                                        'tpr':tpr, 

                                        'auc':auc}, ignore_index=True)



# Set name of the classifiers as index labels

result_table.set_index('classifiers', inplace=True)
fig = plt.figure(figsize=(8,6))



for i in result_table.index:

    plt.plot(result_table.loc[i]['fpr'], 

             result_table.loc[i]['tpr'], 

             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    

plt.plot([0,1], [0,1], color='orange', linestyle='--')



plt.xticks(np.arange(0.0, 1.1, step=0.1))

plt.xlabel("Flase Positive Rate", fontsize=15)



plt.yticks(np.arange(0.0, 1.1, step=0.1))

plt.ylabel("True Positive Rate", fontsize=15)



plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)

plt.legend(prop={'size':13}, loc='lower right')



plt.show()