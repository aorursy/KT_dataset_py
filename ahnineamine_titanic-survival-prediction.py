%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

titanic_training = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')
titanic_training.head()
print(titanic_training.describe())
#titanic_training['Age'].fillna(titanic_training.groupby(['Sex'])['Age'].mean(), inplace=True)
print(titanic_training.isnull().sum())
titanic_training.head()
fig, ax = plt.subplots()
summ = titanic_training['PassengerId'].count()
summ_Survived = titanic_training['Survived'].sum()
titanic_males=titanic_training[titanic_training["Sex"]=='male']
titanic_females=titanic_training[titanic_training["Sex"]=='female']
summ_males = titanic_males['PassengerId'].count()
summ_females = titanic_females['PassengerId'].count()
summ_Survived_males = titanic_males['Survived'].sum()
summ_Survived_females = titanic_females['Survived'].sum()
groupby_sex=titanic_training.groupby(['Sex'])['PassengerId'].count()
groupby_sex_Class=titanic_training.groupby(['Sex','Pclass'])['PassengerId'].count()
groupby_sex_survived=titanic_training.groupby(['Sex'])['Survived'].sum()
groupby_sex_Class_survived=titanic_training.groupby(['Sex','Pclass'])['Survived'].sum()



print(summ)
print(summ_males)
print(summ_females)
print(summ_Survived)
print(summ_Survived_males)
print(summ_Survived_females)
print(groupby_sex)
print(groupby_sex_survived)
print(groupby_sex_Class)
print(groupby_sex_Class_survived)

plt.figure(1)
groupby_sex.plot.pie(labels=['female', 'male'],autopct='%.2f')
plt.figure(2)
groupby_sex_survived.plot.pie(labels=['female', 'male'],autopct='%.2f')


#plt.figure(2)
#count_sex = titanic_training.groupby(['Sex'])['Survived'].mean()
#print(count_sex)
#count_sex.plot(kind='pie')
plt.figure(1)
age_DropNA = titanic_training[titanic_training.Age.notnull()]
plt.ylabel("count passangers")
plt.ylabel("Age")
plt.legend(title="nbr of passangers by age")
plt.hist(age_DropNA.Age)

plt.figure(2)
sum_age_class = titanic_training.groupby(['Pclass'])['Age'].mean()
plt.ylabel("mean Ages")
plt.legend(title="mean age per class")
sum_age_class.plot.bar()
print(sum_age_class)

plt.figure(3)
sum_age_class_survived = titanic_training.groupby(['Pclass','Survived'])['Age'].mean()
print(sum_age_class_survived)
sum_age_class_survived_plot = sns.catplot(x="Pclass", y="Age", hue="Survived", data=titanic_training,
                height=6, kind="bar", palette="muted")


Age_survived = pd.crosstab(age_DropNA.Age,age_DropNA.Survived)
Age_survived = Age_survived.apply(lambda r : r/r.sum(),axis=1)
plt.bar(Age_survived.index,Age_survived[0],label="died")
plt.bar(Age_survived.index,Age_survived[1],bottom=Age_survived[0],label="survived")
plt.ylabel("fraction")
plt.xlabel("Age")
plt.legend(loc='upper right')
plt.show()
Pclass_survived = pd.crosstab(titanic_training.Pclass,titanic_training.Survived)
plt.bar([0,1,2],Pclass_survived[0],label='died')
plt.bar([0,1,2],Pclass_survived[1],bottom=Pclass_survived[0],label='survived')
plt.xticks([0,1,2],["First class","Second class","third class"], rotation = "horizontal")
plt.ylabel("count")
plt.xlabel("")
plt.legend(loc="upper left")
plt.figure(1)
survived_per_sex_class = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=titanic_training,
                height=6, kind="bar", palette="muted")
survived_per_sex_class.despine(left=True)
survived_per_sex_class.set_ylabels("survival probability")

passenger_class_survived_avg = titanic_training.groupby(["Pclass"])['Survived'].mean()
print(passenger_class_survived_avg)
plt.figure(2)
survived_per_class = sns.catplot(x="Pclass", y="Survived", data=titanic_training,
                height=6, kind="bar", palette="muted")
survived_per_class.set_ylabels("survival probability")
plt.hist(titanic_training.Fare,histtype='stepfilled', bins=50)
plt.xlabel('Fare')
plt.ylabel('Count')
plt.figure(1)
sns.set_style('whitegrid')
sns.kdeplot(titanic_training.loc[titanic_training["Survived"] == 1, "Fare"],label="survied")
sns.kdeplot(titanic_training.loc[titanic_training["Survived"] == 0, "Fare"],label="dead")
plt.xlim([100, 800])
plt.ylim([0, 0.002])
plt.figure(2)
plt.scatter(titanic_training.Fare,titanic_training.Survived)
plt.show()
plt.figure(3)
ser=titanic_training.groupby(['Fare'])['Survived'].sum()
df=ser.to_frame()
df.reset_index(inplace=True)
df.columns = ['Fare','Survived']
df.plot(kind='scatter',x='Fare',y='Survived')
plt.show()
plt.figure(4)
ser_mean=titanic_training.groupby(['Fare'])['Survived'].mean()
df_mean=ser_mean.to_frame()
df_mean.reset_index(inplace=True)
df_mean.columns = ['Fare','Survived']
df_mean.plot(kind='scatter',x='Fare',y='Survived')
plt.show()
#print(Fare_survived)
parch_survived = pd.crosstab(titanic_training.Parch,titanic_training.Survived)
plt.subplot(121)
parch_survived = parch_survived.apply(lambda r : r/r.sum(),axis=1)
plt.bar(parch_survived.index,parch_survived[0],label='dead')
plt.bar(parch_survived.index,parch_survived[1],bottom=parch_survived[0],label='survived')
plt.xlabel('parch')
plt.ylabel('fraction')
plt.legend(loc='upper right')

sibsp_survived = pd.crosstab(titanic_training.SibSp,titanic_training.Survived)
plt.subplot(122)
sibsp_survived = sibsp_survived.apply(lambda r : r/r.sum(),axis=1)
plt.bar(sibsp_survived.index,sibsp_survived[0],label='dead')
plt.bar(sibsp_survived.index,sibsp_survived[1],bottom=sibsp_survived[0],label='survived')
plt.xlabel('sibsp')
plt.legend(loc='upper right')
#plt.show()
print(titanic_training[titanic_training.Embarked.isnull()])
print(titanic_training.groupby(['Embarked'])['PassengerId'].count())
embarked_survived = pd.crosstab(titanic_training.Embarked,titanic_training.Survived)
embarked_survived = embarked_survived.apply(lambda r : r/r.sum(),axis=1)
plt.bar([0,1,2],embarked_survived[0],label='dead')
plt.bar([0,1,2],embarked_survived[1],bottom=embarked_survived[0],label='survived')
plt.xticks([0,1,2],["C","Q","S"], rotation = "horizontal")
plt.xlabel('embarked')
plt.ylabel('fraction')
plt.legend(loc='lower right')
Fare_Pclass_Age = titanic_training.loc[:,["Fare","Pclass","Age"]]
Fare_Pclass_Age = Fare_Pclass_Age[Fare_Pclass_Age.Age.notnull()]
X_df = Fare_Pclass_Age.loc[:,["Fare","Pclass"]]
Y_df = Fare_Pclass_Age.loc[:,"Age"]
from sklearn.linear_model import LinearRegression
linR_model = LinearRegression()
Y = linR_model.fit(X_df,Y_df)
Age_Na = titanic_training.loc[:,["Fare","Pclass"]][titanic_training.Age.isnull()]
age_prediction = (Age_Na*(linR_model.coef_).T).sum(axis=1)
age_prediction = age_prediction + linR_model.intercept_
age_LR_prediction = titanic_training.Age
age_LR_prediction = age_LR_prediction.fillna(age_prediction)
print(age_LR_prediction)
titanic_training.Age = age_LR_prediction
print(titanic_training.isnull().sum())
Fare_mean = titanic_test['Fare'].mean()
titanic_test.Fare.fillna(Fare_mean,inplace=True)
print(titanic_test.isnull().sum())
Fare_Pclass_Age_test = titanic_test.loc[:,["Fare","Pclass","Age"]]
Fare_Pclass_Age_test = Fare_Pclass_Age_test[Fare_Pclass_Age_test.Age.notnull()]
X_df_test = Fare_Pclass_Age_test.loc[:,["Fare","Pclass"]]
Y_df_test = Fare_Pclass_Age_test.loc[:,"Age"]
from sklearn.linear_model import LinearRegression
linR_model_test = LinearRegression()
Y_test = linR_model_test.fit(X_df_test,Y_df_test)
Age_Na_test = titanic_test.loc[:,["Fare","Pclass"]][titanic_test.Age.isnull()]
age_prediction_test = (Age_Na_test*(linR_model_test.coef_).T).sum(axis=1)
age_prediction_test = age_prediction_test + linR_model_test.intercept_
age_LR_prediction_test = titanic_test.Age
age_LR_prediction_test = age_LR_prediction_test.fillna(age_prediction_test)
print(age_LR_prediction_test)
titanic_test.Age = age_LR_prediction_test
print(titanic_test.isnull().sum())
Title_list = pd.DataFrame(index=titanic_training.index, columns=['Title'])
Surname_list = pd.DataFrame(index=titanic_training.index, columns=['Surname'])
name_list = list(titanic_training.Name)
NL_1 = [elem.split("\n") for elem in name_list]
ctr=0
for j in NL_1:
    Full_name = j[0]
    Full_name = Full_name.split(",")
    Surname_list.loc[ctr,"Surname"] = Full_name[0]
    Full_name=Full_name.pop(1)
    Full_name=Full_name.split(".")
    Full_name=Full_name.pop(0)
    Full_name=Full_name.replace(" ","")
    Title_list.loc[ctr,"Title"]=str(Full_name)
    ctr= ctr+1
Title_list.Title.value_counts()
c=[titanic_training,titanic_test]
for dataset in c:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(titanic_training['Title'], titanic_training['Sex'])
for dataset in c:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
titanic_training[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in c:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

titanic_training.head()
titanic_training['Parch/SibSp'] = titanic_training.Parch + titanic_training.SibSp
titanic_training.Sex[titanic_training.Sex == "male"] = 1
titanic_training.Sex[titanic_training.Sex == "female"] = 2
titanic_training.Embarked[titanic_training.Embarked == "S"] = 0
titanic_training.Embarked[titanic_training.Embarked == "Q"] = 1
titanic_training.Embarked[titanic_training.Embarked == "C"] = 2
titanic_training.Embarked.fillna("0",inplace=True)
titanic_training.drop(['Ticket', 'Cabin'], axis=1)
print(titanic_training.isnull().sum())
#Applying logistic regression on the training set
feature_cols = ["Pclass","Age","Embarked","Sex","Fare","Parch/SibSp","Title"]
X_train = titanic_training[feature_cols]
y_train = titanic_training.Survived
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_train)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_train, y_pred)
cnf_matrix
print(titanic_test.isnull().sum())
titanic_test['Parch/SibSp'] = titanic_test.Parch + titanic_test.SibSp
titanic_test.Sex[titanic_test.Sex == "male"] = 1
titanic_test.Sex[titanic_test.Sex == "female"] = 2
titanic_test.Embarked[titanic_test.Embarked == "S"] = 0
titanic_test.Embarked[titanic_test.Embarked == "Q"] = 1
titanic_test.Embarked[titanic_test.Embarked == "C"] = 2
titanic_test.drop(['Ticket', 'Cabin'], axis=1)
print(titanic_test.isnull().sum())
#Applying logistic regression on the test set
X_test = titanic_test[feature_cols]
logregression = LogisticRegression()
logregression.fit(X_train,y_train)
y_predict=logregression.predict(X_test)
print(y_predict)
Submission = pd.DataFrame(columns=['PassengerId','Survived'])
Submission.PassengerId = titanic_test.PassengerId
Submission.Survived = y_predict
Submission.head()
Submission.to_csv('Submission.csv')
df = pd.read_csv('Submission.csv',index_col=0)
print(df)
from sklearn import svm
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.SVC(kernel='linear', C=1, gamma=1) 
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(X_train, y_train)
model.score(X_train, y_train)
#Predict Output
y_predicted= model.predict(X_test)
print(y_predicted)
SubmissionV2 = pd.DataFrame(columns=['PassengerId','Survived'])
SubmissionV2.PassengerId = titanic_test.PassengerId
SubmissionV2.Survived = y_predicted
SubmissionV2.head()
SubmissionV2.to_csv('SubmissionV2.csv')
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
clf.score(X_train, y_train)
prediction = clf.predict(X_test)
print(prediction)
SubmissionV3 = pd.DataFrame(columns=['PassengerId','Survived'])
SubmissionV3.PassengerId = titanic_test.PassengerId
SubmissionV3.Survived = prediction
SubmissionV3.head()
SubmissionV3.to_csv('SubmissionV3.csv')
from sklearn.neighbors import KNeighborsClassifier
KNNmodel = KNeighborsClassifier(n_neighbors=3)
KNNmodel.fit(X_train,y_train)
KNNmodel.score(X_train, y_train)
KNN_predicted= KNNmodel.predict(X_test) 
print(KNN_predicted)
from sklearn.ensemble import VotingClassifier
#from sklearn import model_selection
#kfold = model_selection.KFold(n_splits=10, random_state=seed)
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = tree.DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = svm.SVC()
estimators.append(('svm', model3))
model4 = KNeighborsClassifier(n_neighbors=3)
estimators.append(('KNN', model4))
ensemble = VotingClassifier(estimators,voting='hard')
ensemble.fit(X_train,y_train)
ensemble.score(X_train, y_train)
ensemble_predicted= ensemble.predict(X_test)
print(ensemble_predicted)
SubmissionV4 = pd.DataFrame(columns=['PassengerId','Survived'])
SubmissionV4.PassengerId = titanic_test.PassengerId
SubmissionV4.Survived = ensemble_predicted
SubmissionV4.head()
SubmissionV4.to_csv('SubmissionV4.csv')
