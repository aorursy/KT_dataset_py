# Importing the libraries

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

import numpy as np
# Importing the train full_set

train_full_set = pd.read_csv('../input/train.csv')

#Printing the names and datatypes of the columns

print("/n/nInformation about Null/ empty data points in each Column of Training set\n\n")

print(train_full_set.info())
# Importing the test full_set

test_full_set = pd.read_csv('../input/test.csv')

print("\n\nInformation about Null/ empty data points in each Column of Test set\n\n")

print(test_full_set.info())
#Creating dummy variables in the training set and checking intial correlations

full_set_initial = pd.get_dummies(data=train_full_set, columns=['Embarked','Sex','Survived'],drop_first = True)

full_set_initial = full_set_initial.drop(['PassengerId','Name','Ticket','Cabin'], axis = 1)

corr = full_set_initial.corr()

print(corr)

plt.figure(figsize=(20,20))

plt.imshow(corr, cmap='GnBu')

plt.colorbar()

plt.xticks(range(len(corr)), corr.columns, rotation='vertical')

plt.yticks(range(len(corr)), corr.columns);

plt.suptitle('Correlation Matrix', fontsize=15, fontweight='bold')

plt.show()
full_set = pd.concat([train_full_set,test_full_set])

full_set=full_set.reset_index(drop=True)
#Printing the top 5 lines of full_set to check if it was imported correctly

print(full_set.head(5))
#Null Values in each column

print(full_set.isnull().sum())
full_set.drop(['Ticket','Cabin','PassengerId'],axis=1,inplace=True)
#Splitting the 'Name' column into Titles using RegEx

full_set['Title'] = full_set['Name'].str.extract('([A-Z]\w+\.)', expand=True)
#Checking for information on Titles

print(full_set.Title.value_counts())
plt.figure(figsize=(15,8))

sns.set_style("whitegrid")

ax=sns.countplot(x="Title", data=full_set)

ax.set_ylabel("COUNT",size = 20,color="black",alpha=0.5)

ax.set_xlabel("TITLE",size = 20,color="black",alpha=0.5)

ax.set_title("COUNT OF TITLES IN EACH CATEGORY BEFORE COMBINATION",size = 20,color="black",alpha=0.5)
#Combine anything from rare titles under one group called 'Rare.'

full_set.loc[full_set['Title'].isin(['Dona.', 'Lady.', 'Countess.','Capt.', 'Col.', 'Don.',

                'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.']),'Title'] = 'Rare'
#Converting anything from 'Mlle.', 'Ms.', 'Mme.' to 'Miss.'

full_set.loc[full_set['Title'].isin(['Mlle.', 'Ms.', 'Mme.']),'Title'] = 'Miss.'
#Checking to see how many unique titles are there after changes

print(full_set.Title.value_counts())
plt.figure(figsize=(15,8))

sns.set_style("whitegrid")

ax=sns.countplot(x="Title", data=full_set)

ax.set_ylabel("COUNT",size = 20,color="black",alpha=0.5)

ax.set_xlabel("TITLE",size = 20,color="black",alpha=0.5)

ax.set_title("COUNT OF TITLES IN EACH CATEGORY AFTER COMBINATION",size = 20,color="black",alpha=0.5)
#Creating a new variable for Family Size

full_set['FamilyMembers'] = full_set['SibSp'] + full_set['Parch'] + 1
#Checking for Survival based on Family size

family_size_survival = full_set[['FamilyMembers', 'Survived']].groupby(['FamilyMembers'], as_index=False).count().sort_values(by='Survived', ascending=False)
plt.figure(figsize=(15,8))

sns.set_style("whitegrid")

ax = sns.barplot(x="FamilyMembers", y="Survived", data=family_size_survival)

ax.set_title("SURVIVED PASSENGER COUNT BASED ON FAMILY SIZE",size = 20,color="black",alpha=0.5)

ax.set_ylabel("NUMBER SURVIVED",size = 20,color="black",alpha=0.5)

ax.set_xlabel("FAMILY SIZE",size = 20,color="black",alpha=0.5)
#Creating a new category based on family size based on the number of people who survived

''' 1 ---Family Size =1

    2 ---Family Size between 2 and 4(included)

    3 ---Family Size more than 4'''

family_size = []

for row in full_set.FamilyMembers:

    if row in [1]:

        family_size.append(1)

    elif row in [2,3,4]:

        family_size.append(2)

    else :

        family_size.append(3)

full_set['FamilySize'] = family_size
#Null Values in each column

print("\n\n Number of null in each column before imputing:\n")

print(full_set.isnull().sum())
#IMPUTE EMBARKED LOCATION

#Finding out which passengers have missing embarked location

full_set[full_set['Embarked'].isnull()]
#Checking correlation between embarked location and Fare+Class by creating a Box plot segregated by port of Embarkment

plt.figure(figsize=(15,8))

embarked_impute = full_set.dropna(subset=['Embarked'])

sns.set_style("whitegrid")

sns.set(style="ticks")

sns.boxplot(x="Pclass", y="Fare",hue="Embarked",data=embarked_impute,palette="PRGn");

sns.despine(offset=10, trim=True)
#Filling the Nan values in Embarked column for passengers 61 and 829

full_set.Embarked.fillna('S', inplace=True)
#IMPUTE MISSING FARE

#Finding out which passengers have missing fare

full_set[full_set['Fare'].isnull()]
full_set.groupby(['Pclass', 'Embarked']).mean()
full_set.Fare.fillna('14.43', inplace=True)
#Checking the relation between Age and Fare

plt.figure(figsize=(15,8))

plt.scatter(x='Age', y = 'Fare', data = full_set)

plt.xlabel('Age')

plt.ylabel('Fare')

plt.title('Age V/s Fare')

plt.show()
#Relation between Age and Pclass

#BOX PLOTS

plt.figure(figsize=(15,8))

sns.set_style("whitegrid")

ax = sns.boxplot(x="Pclass", y="Age", data=full_set.dropna(subset=['Age']))

ax.set_title("PASSENGER CLASS V/S AGE BOX PLOTS",size = 20,color="black",alpha=0.5)

ax.set_ylabel("AGE",size = 20,color="black",alpha=0.5)

ax.set_xlabel("PASSENGER CLASS",size = 20,color="black",alpha=0.5)
#ONE WAY ANOVA FOR SIGNIFICANCE

#If P > 0.05, we can claim with high confidence that the means of the results of all three experiments are not significantly different.

full_set['Age'].groupby(full_set['Pclass']).describe()

pclass1 = (full_set.dropna(subset=["Age"]))[full_set["Pclass"] == 1]["Age"]

pclass2 = (full_set.dropna(subset=["Age"]))[full_set["Pclass"] == 2]["Age"]

pclass3 = (full_set.dropna(subset=["Age"]))[full_set["Pclass"] == 3]["Age"]

f_val, p_val = stats.f_oneway(pclass1, pclass2, pclass3)

print ("One-way ANOVA P =", p_val)
plt.figure(figsize=(15,8))

sns.set_style("whitegrid")

ax = sns.boxplot(x="Sex", y="Age", data=full_set.dropna(subset=['Age']))

ax.set_title("GENDER V/S AGE BOX PLOTS",size = 20,color="black",alpha=0.5)

ax.set_ylabel("AGE",size = 20,color="black",alpha=0.5)

ax.set_xlabel("GENDER",size = 20,color="black",alpha=0.5)
male = (full_set.dropna(subset=["Age"]))[full_set["Sex"] == "male"]["Age"]

female = (full_set.dropna(subset=["Age"]))[full_set["Sex"] == "female"]["Age"]

z_stat, p_val = stats.ranksums(female, male)

print ("\n\nMWW RankSum P for ages of male and female =", p_val)

plt.figure(figsize=(20,20))

sns.set_style("whitegrid")

ax = sns.boxplot(x="Embarked", y="Age", data=full_set.dropna(subset=['Age']))

ax.set_title("PORT OF EMBARKMENT V/S AGE BOX PLOTS",size = 20,color="black",alpha=0.5)

ax.set_ylabel("AGE",size = 20,color="black",alpha=0.5)

ax.set_xlabel("PORT OF EMBARKMENT",size = 20,color="black",alpha=0.5)
#Compute one-way ANOVA P value

#If P > 0.05, we can claim with high confidence that the means of the results of all three experiments are not significantly different.

port_s = (full_set.dropna(subset=["Age"]))[full_set["Embarked"] == "S"]["Age"]

port_c  = (full_set.dropna(subset=["Age"]))[full_set["Embarked"] == "C"]["Age"]

port_q  = (full_set.dropna(subset=["Age"]))[full_set["Embarked"] == "Q"]["Age"]

f_val, p_val = stats.f_oneway(port_s , port_c , port_q )

print ("One-way ANOVA P =", p_val)
plt.figure(figsize=(20,20))

sns.set_style("whitegrid")

ax = sns.boxplot(x="Title", y="Age", data=full_set.dropna(subset=['Age']))

ax.set_title("TITLE V/S AGE BOX PLOTS",size = 20,color="black",alpha=0.5)

ax.set_ylabel("AGE",size = 20,color="black",alpha=0.5)

ax.set_xlabel("TITLE",size = 20,color="black",alpha=0.5)
title_master = (full_set.dropna(subset=["Age"]))[full_set["Title"] == "Master."]["Age"]

title_miss  = (full_set.dropna(subset=["Age"]))[full_set["Title"] == "Miss."]["Age"]

title_mr  = (full_set.dropna(subset=["Age"]))[full_set["Title"] == "Mr."]["Age"]

title_mrs  = (full_set.dropna(subset=["Age"]))[full_set["Title"] == "Mrs."]["Age"]

title_rare  = (full_set.dropna(subset=["Age"]))[full_set["Title"] == "Rare"]["Age"]

f_val, p_val = stats.f_oneway(title_master,title_miss,title_mr,title_mrs,title_rare)

print ("One-way ANOVA P =", p_val)
#Checking the relation between Age and Family Members

plt.figure(figsize=(15,8))

plt.scatter(x='Age', y = 'FamilyMembers', data = full_set)

plt.xlabel("Age")

plt.ylabel("Family Members")

plt.title("Relation between Age and Number of Family Members",size = 20,color="black",alpha=0.5)

plt.show()
full_set['Age'].corr(full_set['FamilyMembers'])# Very weak correlation
#Filling the NA values in each column with mean in each group

full_set["Age"] = full_set.groupby(['Embarked','Pclass','Title','Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))

age_impute = full_set.groupby(['Embarked','Pclass','Title','Sex'])['Age'].mean()

print(age_impute[1:20])
#Creating a new feature called Person Type

full_set.loc[(full_set['Age'] <16), 'PersonType'] = 'child'

full_set.loc[(full_set['Age'] >=16) & (full_set['Sex'] == 'female'), 'PersonType'] = 'female'

full_set.loc[(full_set['Age'] >=16) & (full_set['Sex'] == 'male'), 'PersonType'] = 'male'
#Final check for null values

print("\n\n Number of null in each column after imputing:\n")

print(full_set.isnull().sum())
'''DATA VISUALIZATION'''

#Initial data visualization
#Graph showing survival of passengers

plt.figure(figsize=(15,8))

sns.set_style("whitegrid")

ax = sns.countplot(x="Survived", data=train_full_set)

ax.set_ylabel("COUNT",size = 20,color="black",alpha=0.5)

ax.set_xlabel("SURVIVED",size = 20,color="black",alpha=0.5)

ax.set_title("COUNT OF SURVIVED AND DIED",size = 20,color="black",alpha=0.5)
#Gender based on class

plt.figure(figsize=(15,8))

ax = sns.countplot(x="Pclass", hue="PersonType", data=full_set)

ax.set_ylabel("COUNT",size = 20,color="black",alpha=0.5)

ax.set_xlabel("Pclass",size = 20,color="black",alpha=0.5)

ax.set_title("COUNT OF PERSON TYPE IN EACH CLASS",size = 20,color="black",alpha=0.5)
#Survival based on Class

plt.figure(figsize=(40,40))

ax = sns.factorplot(x="Pclass", hue="PersonType", col="Survived",data=full_set, kind="count")

#Percentage of survival per class as per person Type

plt.figure(figsize=(20,20))

pd.crosstab([full_set.PersonType,full_set.Survived],full_set.Pclass,margins=True)

sns.factorplot('Pclass','Survived',hue='PersonType',data=full_set)
#Survival Age per person Type

plt.figure(figsize=(20,20))

sns.violinplot(x="PersonType",y="Age", hue="Survived",data= full_set,palette="muted", split=True)
'''CREATING FINAL DATASET'''

#Dropping unwanted columns

full_set = full_set.drop(['Name','SibSp','Parch','FamilyMembers','Sex'], axis=1)
full_set = pd.get_dummies(data=full_set, columns=['Embarked','Title','PersonType'],drop_first = True)
full_set.columns
#Rearranging the columns to put target variable coulmn in the last column of the full_set

full_set= full_set[['Age', 'Fare', 'Pclass','FamilySize', 'Embarked_Q',

       'Embarked_S', 'Title_Miss.', 'Title_Mr.', 'Title_Mrs.', 'Title_Rare',

       'PersonType_female', 'PersonType_male','Survived']]

train_set = full_set.loc[0:890,:]

test_set = full_set.iloc[891:,:-1]

X = train_set.iloc[:,:-1]

y = train_set.iloc[:,-1]
from sklearn.cross_validation import cross_val_score

#LINEAR MODELS:LOGISTIC REGRESSION AND SVM

#Logistic Regression

from sklearn. linear_model import LogisticRegression

logreg_classifier = LogisticRegression()

logreg_kfold_mean_accuracy = cross_val_score(logreg_classifier,X,y, cv=10, scoring='accuracy').mean()

#Creating confusion Matrix for the entire dataset over 10 folds:

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

y_pred = cross_val_predict(logreg_classifier,X,y,cv=10)

conf_mat = confusion_matrix(y,y_pred)

y_actual = pd.Series(y, name='Actual')

y_pred = pd.Series(y_pred, name='Predicted')

confusion_matrix = pd.crosstab(y_actual, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

print("Confusion Matrix for the entire dataset over  10 folds is:")

print(confusion_matrix)

#Calculating Sensitivity and Specificity

cm= confusion_matrix.iloc[:,:].values

true_positive = cm[0,0]

false_positive = cm[0,1]

true_negative = cm[1,1]

false_negative = cm[1,0]

logreg_sensitivity = true_positive/(true_positive+false_negative)

logreg_specificity = true_negative/(true_negative+false_positive)

logreg_mean_sensitivity_specificity = (logreg_sensitivity+logreg_specificity)/2



print("Cross Validated Mean Accuracy over 10 folds for Logistic Regression is " + str(logreg_kfold_mean_accuracy))

print("Cross Validated Sensitivity over 10 folds for Logistic Regression is " + str(logreg_sensitivity))

print("Cross Validated Specificity over 10 folds for Logistic Regression is " + str(logreg_specificity))

print("Cross Validated Mean Sensitivity and Specificity over 10 folds for Logistic Regression is " + str(logreg_mean_sensitivity_specificity))
'''Linear SVM'''

# Fitting SVM to the Training set

from sklearn.svm import SVC

SVM_classifier = SVC(kernel = 'linear', random_state = 0)

SVM_kfold_mean_accuracy = cross_val_score(SVM_classifier,X,y, cv=10, scoring='accuracy').mean()

#Creating confusion Matrix for the entire dataset over 10 folds:

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

y_pred1 = cross_val_predict(SVM_classifier,X,y,cv=10)

conf_mat = confusion_matrix(y,y_pred1)

y_actual = pd.Series(y, name='Actual')

y_pred1 = pd.Series(y_pred1, name='Predicted')

confusion_matrix1 = pd.crosstab(y_actual, y_pred1, rownames=['Actual'], colnames=['Predicted'], margins=True)

print("Confusion Matrix for the entire dataset over  10 folds is:")

print(confusion_matrix1)

#Calculating Sensitivity and Specificity

cm1= confusion_matrix1.iloc[:,:].values

true_positive1 = cm1[0,0]

false_positive1 = cm1[0,1]

true_negative1 = cm1[1,1]

false_negative1 = cm1[1,0]

SVM_sensitivity = true_positive1/(true_positive1+false_negative1)

SVM_specificity = true_negative1/(true_negative1+false_positive1)

SVM_mean_sensitivity_specificity = (SVM_sensitivity+SVM_specificity)/2



print("Cross Validated Mean Accuracy over 10 folds for Linear SVM is " + str(SVM_kfold_mean_accuracy))

print("Cross Validated Sensitivity over 10 folds for Linear SVM is " + str(SVM_sensitivity))

print("Cross Validated Specificity over 10 folds for Linear SVM is " + str(SVM_specificity))

print("Cross Validated Mean Sensitivity and Specificity over 10 folds for Linear SVM is " + str(SVM_mean_sensitivity_specificity))
'''Non-Linear Classifiers'''



'''Naiive Bayes'''

from sklearn.naive_bayes import GaussianNB

NB_classifier = GaussianNB()

NB_kfold_mean_accuracy = cross_val_score(NB_classifier,X,y, cv=10, scoring='accuracy').mean()

#Creating confusion Matrix for the entire dataset over 10 folds:

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

y_pred2 = cross_val_predict(NB_classifier,X,y,cv=10)

conf_mat2 = confusion_matrix(y,y_pred)

y_actual = pd.Series(y, name='Actual')

y_pred2 = pd.Series(y_pred2, name='Predicted')

confusion_matrix2 = pd.crosstab(y_actual, y_pred2, rownames=['Actual'], colnames=['Predicted'], margins=True)

print("Confusion Matrix for the entire dataset over  10 folds is:")

print(confusion_matrix2)

#Calculating Sensitivity and Specificity

cm2= confusion_matrix2.iloc[:,:].values

true_positive2 = cm2[0,0]

false_positive2 = cm2[0,1]

true_negative2 = cm2[1,1]

false_negative2 = cm2[1,0]

NB_sensitivity = true_positive2/(true_positive2+false_negative2)

NB_specificity = true_negative2/(true_negative2+false_positive2)

NB_mean_sensitivity_specificity = (NB_specificity+NB_sensitivity)/2



print("Cross Validated Mean Accuracy over 10 folds for Naive Bayes Classifier is " + str(NB_kfold_mean_accuracy))

print("Cross Validated Sensitivity over 10 folds for Naive Bayes Classifier is " + str(NB_sensitivity))

print("Cross Validated Specificity over 10 folds for Naive Bayes Classifier is " + str(NB_specificity))

print("Cross Validated Mean Sensitivity and Specificity over 10 folds for Naive Bayes Classifier is " + str(NB_mean_sensitivity_specificity))
'''Decision Tree Classifier'''



from sklearn.tree import DecisionTreeClassifier

min_split_range = list(range(2,100))

min_split_scores = []

for m in min_split_range:

    DT_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0,min_samples_split=m)

    scores = cross_val_score(DT_classifier,X,y, cv =10, scoring='accuracy')

    min_split_scores.append(scores.mean())    

#Choosing the min sample split value that gives the highest accuracy by plotting a graph

plt.plot(min_split_range, min_split_scores)

plt.xlabel('Minimum samples split')

plt.ylabel('Cross validated Accuracy')

#DT Classifier using min sample split of 57

DT_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0,min_samples_split=57)

DT_kfold_mean_accuracy = cross_val_score(DT_classifier,X,y, cv=10, scoring='accuracy').mean()

print("Mean Accuracy over 10 folds for Decision Tree Classifier is " +str(DT_kfold_mean_accuracy))

#Creating confusion Matrix for the entire dataset over 10 folds:

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

y_pred3 = cross_val_predict(DT_classifier,X,y,cv=10)

conf_mat3 = confusion_matrix(y,y_pred3)

y_actual = pd.Series(y, name='Actual')

y_pred3 = pd.Series(y_pred3, name='Predicted')

confusion_matrix3 = pd.crosstab(y_actual, y_pred3, rownames=['Actual'], colnames=['Predicted'], margins=True)

print("Confusion Matrix for the entire dataset over  10 folds is:")

print(confusion_matrix3)

#Calculating Sensitivity and Specificity

cm3= confusion_matrix3.iloc[:,:].values

true_positive3 = cm3[0,0]

false_positive3 = cm3[0,1]

true_negative3 = cm3[1,1]

false_negative3 = cm3[1,0]

DT_sensitivity = true_positive3/(true_positive3+false_negative3)

DT_specificity = true_negative3/(true_negative3+false_positive3)

DT_mean_sensitivity_specificity = (DT_sensitivity+DT_specificity)/2





print("Cross Validated Mean Accuracy over 10 folds for Decision Tree Classifier is " + str(DT_kfold_mean_accuracy))

print("Cross Validated Sensitivity over 10 folds for Decision Tree Classifier is " + str(DT_sensitivity))

print("Cross Validated Specificity over 10 folds for Decision Tree Classifier is " + str(DT_specificity))

print("Cross Validated Mean Sensitivity and Specificity over 10 folds for Decision Tree Classifier is " + str(DT_mean_sensitivity_specificity))
'''Random Forest'''



from sklearn.ensemble import RandomForestClassifier

estimators = list(range(10,30))

list_estimators = []

for n in estimators:

    RF_classifier = RandomForestClassifier(n_estimators = n, criterion = 'entropy', random_state = 0)

    scores = cross_val_score(RF_classifier,X,y, cv =10, scoring='accuracy')

    list_estimators.append(scores.mean())    

#Choosing the min sample split value that gives the highest accuracy by plotting a graph

plt.plot(estimators, list_estimators)

plt.xlabel('Number of Estimators')

plt.ylabel('Cross validated Accuracy')

#Choosing estimator as 20

RF_classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0)

RF_kfold_mean_accuracy = cross_val_score(RF_classifier,X,y, cv=10, scoring='accuracy').mean()

print("Mean Accuracy over 10 folds for Random Forest Classifier is " +str(DT_kfold_mean_accuracy))

#Creating confusion Matrix for the entire dataset over 10 folds:

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

y_pred4 = cross_val_predict(DT_classifier,X,y,cv=10)

conf_mat4= confusion_matrix(y,y_pred4)

y_actual4 = pd.Series(y, name='Actual')

y_pred4 = pd.Series(y_pred4, name='Predicted')

confusion_matrix4= pd.crosstab(y_actual4, y_pred4, rownames=['Actual'], colnames=['Predicted'], margins=True)

print("Confusion Matrix for the entire dataset over  10 folds is:")

print(confusion_matrix4)

#Calculating Sensitivity and Specificity

cm4= confusion_matrix4.iloc[:,:].values

true_positive4 = cm4[0,0]

false_positive4 = cm4[0,1]

true_negative4 = cm4[1,1]

false_negative4 = cm4[1,0]

RF_sensitivity = true_positive4/(true_positive4+false_negative4)

RF_specificity = true_negative4/(true_negative4+false_positive4)

RF_mean_sensitivity_specificity = (RF_sensitivity+RF_specificity)/2



print("Cross Validated Mean Accuracy over 10 folds for Random Forest Classifier is " + str(RF_kfold_mean_accuracy))

print("Cross Validated Sensitivity over 10 folds for Random Forest Classifier is " + str(RF_sensitivity))

print("Cross Validated Specificity over 10 folds for Random Forest Classifier is " + str(RF_specificity))

print("Cross Validated Mean Sensitivity and Specificity over 10 folds for Random Forest Classifier is " + str(RF_mean_sensitivity_specificity))
'''KNN'''



#KNN Classifier

#Parameter Tuning

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import cross_val_score

k_range = list(range(1,41))

k_scores = []

for k in k_range:

    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn_classifier,X,y, cv =10, scoring='accuracy')

    k_scores.append(scores.mean())

#Choosing the K value that gives the highest accuracy by plotting a graph

plt.plot(k_range, k_scores)

plt.xlabel('K Values')

plt.ylabel('Cross validated Accuracy')

'''We can settle for a k value of 25 in this case as it gives the highest prediction accuracy'''

#Calculating mean classification accuracy for KNN where k=25

knn_classifier = KNeighborsClassifier(n_neighbors=25)

knn_kfold_mean_accuracy = cross_val_score(knn_classifier,X,y, cv =10, scoring='accuracy').mean()

print("Mean Accuracy over 10 folds for KNN is " + str(knn_kfold_mean_accuracy))

#Creating confusion Matrix for the entire dataset over 10 folds:

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

y_pred5 = cross_val_predict(knn_classifier,X,y,cv=10)

conf_mat5 = confusion_matrix(y,y_pred)

y_actual5 = pd.Series(y, name='Actual')

y_pred5 = pd.Series(y_pred5, name='Predicted')

confusion_matrix5 = pd.crosstab(y_actual5, y_pred5, rownames=['Actual'], colnames=['Predicted'], margins=True)

print("Confusion Matrix for the entire dataset over  10 folds is:")

print(confusion_matrix5)

#Calculating Sensitivity and Specificity

cm5= confusion_matrix5.iloc[:,:].values

true_positive5 = cm5[0,0]

false_positive5 = cm5[0,1]

true_negative5 = cm5[1,1]

false_negative5 = cm5[1,0]

KNN_sensitivity = true_positive5/(true_positive5+false_negative5)

KNN_specificity = true_negative5/(true_negative5+false_positive5)

KNN_mean_sensitivity_specificity = (KNN_sensitivity+KNN_specificity)/2



print("Cross Validated Mean Accuracy over 10 folds for KNN Classifier is " + str(knn_kfold_mean_accuracy))

print("Cross Validated Sensitivity over 10 folds for KNN Classifier is " + str(KNN_sensitivity))

print("Cross Validated Specificity over 10 folds for KNN Classifier is " + str(KNN_specificity))

print("Cross Validated Mean Sensitivity and Specificity over 10 folds for KNN Classifier is " + str(KNN_mean_sensitivity_specificity))
'''Kernel SVM'''



# Fitting SVM to the Training set

from sklearn.svm import SVC

C_range = list(range(1,40))

C_scores = []

for c in C_range:

    kSVC_classifier = SVC(kernel = 'rbf', random_state = 0, C=c)

    scores = cross_val_score(kSVC_classifier,X,y, cv =10, scoring='accuracy')

    C_scores.append(scores.mean())    

#Choosing the C value that gives the highest accuracy by plotting a graph

plt.plot(C_range, C_scores)

plt.xlabel('Regularization parameter, C, of the error term')

plt.ylabel('Cross validated Accuracy')

#Calculations using C=2

kSVC_classifier = SVC(kernel = 'rbf', random_state = 0)

kSVC_kfold_mean_accuracy = cross_val_score(kSVC_classifier,X,y, cv=10, scoring='accuracy').mean()

print("Mean Accuracy over 10 folds for Kernel SVC Classification is " +str(kSVC_kfold_mean_accuracy))

#Creating confusion Matrix for the entire dataset over 10 folds:

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

y_pred6 = cross_val_predict(kSVC_classifier,X,y,cv=10)

conf_mat6 = confusion_matrix(y,y_pred)

y_actual6 = pd.Series(y, name='Actual')

y_pred6 = pd.Series(y_pred6, name='Predicted')

confusion_matrix6 = pd.crosstab(y_actual6, y_pred6, rownames=['Actual'], colnames=['Predicted'], margins=True)

print("Confusion Matrix for the entire dataset over  5 folds is:")

print(confusion_matrix6)

#Calculating Sensitivity and Specificity

cm6= confusion_matrix6.iloc[:,:].values

true_positive6 = cm6[0,0]

false_positive6 = cm6[0,1]

true_negative6 = cm6[1,1]

false_negative6 = cm6[1,0]

kSVC_sensitivity = true_positive6/(true_positive6+false_negative6)

kSVC_specificity = true_negative6/(true_negative6+false_positive6)

kSVC_SVC_mean_sensitivity_specificity = (kSVC_sensitivity+kSVC_specificity)/2



print("Cross Validated Mean Accuracy over 10 folds for Kernel SVM Classifier is " + str(kSVC_kfold_mean_accuracy))

print("Cross Validated Sensitivity over 10 folds for Kernel SVM Classifier is " + str(kSVC_sensitivity))

print("Cross Validated Specificity over 10 folds for Kernel SVM is " + str(kSVC_specificity))

print("Cross Validated Mean Sensitivity and Specificity over 10 folds for Kernel SVM is " + str(kSVC_SVC_mean_sensitivity_specificity))
summary_all_models = pd.DataFrame({

                    'Model': ['Logistic Regression', 'Support Vector Machine(linear)','Naive Bayes',   'Decision Tree', 'Random Forest','KNN','Kernel SVC'],

                    'Mean Accuracy Score': [logreg_kfold_mean_accuracy, SVM_kfold_mean_accuracy, NB_kfold_mean_accuracy, DT_kfold_mean_accuracy, RF_kfold_mean_accuracy,knn_kfold_mean_accuracy,kSVC_kfold_mean_accuracy],

                    'Sensitivity':[logreg_sensitivity,SVM_sensitivity,NB_sensitivity,DT_sensitivity,RF_sensitivity,KNN_sensitivity,kSVC_sensitivity],

                    'Specificity':[logreg_specificity,SVM_specificity,NB_specificity,DT_specificity,RF_specificity,KNN_specificity,kSVC_specificity],

                    'Mean Sensitivity Specificity':[logreg_mean_sensitivity_specificity,SVM_mean_sensitivity_specificity,NB_mean_sensitivity_specificity,DT_mean_sensitivity_specificity,RF_mean_sensitivity_specificity,KNN_mean_sensitivity_specificity,kSVC_SVC_mean_sensitivity_specificity]})



summary_all_models= summary_all_models[[ 'Model','Mean Accuracy Score', 'Mean Sensitivity Specificity',

       'Sensitivity', 'Specificity']]

summary_all_models=summary_all_models.sort_values(by='Mean Sensitivity Specificity', ascending=False)

print(summary_all_models)

plt.figure(figsize=(20,20))

ax=sns.barplot(x="Model", y="Mean Sensitivity Specificity", data=summary_all_models)

ax.set_ylabel("Mean Sensitivity Specificity",size = 30,color="black",alpha=0.5)

ax.set_xlabel("Models",size = 30,color="black",alpha=0.5)

ax.set_title("Summary of All Models",size = 30,color="black",alpha=0.5)

#Predicting the test set results from Kaggle

#Decision Tree Classifier using min sample split of 57

DT_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0,min_samples_split=57)

DT_classifier.fit(X,y)

test_results = DT_classifier.predict(test_set)

print("First 40 predictions for the test set using Decision Tree Classifier:")

print(test_results[1:40])