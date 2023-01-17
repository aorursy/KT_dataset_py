import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import preprocessing

from sklearn.model_selection import train_test_split
train_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")
# Lets drop these attributes because they are not logically related to survival

train_df = train_df.drop(['PassengerId','Name','Ticket','Embarked'], axis=1)

test_df    = test_df.drop(['Name','Ticket','Embarked'], axis=1)
train_df[train_df['Cabin'].isnull()==False]
#I guess we need to remove Cabin column as well because it has lot of missing values and there seems to be no analogy with cabin and any of the other relatable variables such as Fare, Sibsp and Parch.

train_df[train_df['Cabin'].isnull()==False]['Survived'].size
train_df=train_df.drop(['Cabin'], axis=1)

test_df=test_df.drop(['Cabin'], axis=1)
train_df.isnull().sum()
test_df.isnull().sum()
#Fill in the 1 missing value in Fare with the median

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)
#Now we need to take care only of the age attribute

x = pd.read_csv("../input/train.csv")

y = pd.read_csv("../input/test.csv")

average_age_titanic   = (x["Age"].mean()+ y["Age"].mean())/2

std_age_titanic       = (x["Age"].std() + y["Age"].mean())/2

null_age_titanic = (x["Age"].isnull().sum() + y["Age"].isnull().sum())/2
average_age_titanic
average_age_titanic + std_age_titanic
average_age_titanic   = (train_df["Age"].mean()+ test_df["Age"].mean())/2

std_age_titanic       = (train_df["Age"].std()+ test_df["Age"].std())/2
#Predicting ages on the  data's missing values

#Let's combine both test and training datasets because we need to predict ages for both the datasets and more training data is better.

c=train_df.append(test_df)

#Take only the finite values

c = c[np.isfinite(c['Age'])] 

#I split the combined data into train and test with 20% test data to check the performance of the algorithms on Age prediction

trc,tec= train_test_split(c, test_size = 0.2)

#trc is my training set with 80% values and tec is my test set for 20% values
agestrc=trc['Age']

agestec=tec['Age']

trc=trc.drop(['Age','Survived'],axis=1)

tec=tec.drop(['Age','Survived'],axis=1)
#One Hot Encoding to Convert Categorical Variable 'Sex' to numerical

le = preprocessing.LabelEncoder()

a=le.fit_transform(trc['Sex'])

b=le.fit_transform(tec['Sex'])

trc=trc.drop(['Sex'],axis=1)

tec=tec.drop(['Sex'],axis=1)

trc['Sexconv']= a

tec['Sexconv']= b
#drop passenger id column

trc=trc.drop(['PassengerId'],axis=1)

tec=tec.drop(['PassengerId'],axis=1)
# We will now see which model performs the best on age prediction. The model is trained on the training set [80%] and then the score is computed on the training set to observe how well the model fit and on the test set to observe the model's behavior to new data

# Train a Logistic Regression Model

Y_train = np.asarray(agestrc, dtype=np.uint64) #Assign age values of trc to Y_train 

aa= np.asarray(agestec, dtype=np.uint64) # Assign age values of tec to aa

logreg = LogisticRegression()

logreg.fit(trc,Y_train)

print ("Logistic Regression Model")

print ("Score on trained values")

print (logreg.score(trc,Y_train))

print ("Score on test values")

print (logreg.score(tec,aa))

print ("Overall Score")

print (logreg.score(tec,aa)+logreg.score(trc,Y_train))
#Support Vector Machine Analysis

svc = SVC()

svc.fit(trc, Y_train)

print ("Support Vector Machine Model")

print ("Score on trained values")

print (svc.score(trc,Y_train))

print ("Score on test values")

print (svc.score(tec,aa))

print ("Overall Score")

print (svc.score(tec,aa)+svc.score(trc,Y_train))
# k-Nearest Neighbors Analysis

knn = KNeighborsClassifier(n_neighbors = 3)

x=knn.fit(trc, Y_train)

print ("k-Nearest Neighbors Model")

print ("Score on trained values")

print (knn.score(trc,Y_train))

print ("Score on test values")

print (knn.score(tec,aa))

print ("Overall Score")

print (knn.score(tec,aa)+knn.score(trc,Y_train))
#Gaussian Naive Bayes Analysis

gaussian = GaussianNB()

gaussian.fit(trc, Y_train)

print ("Gaussian Naive Bayes Model")

print ("Score on trained values")

print (gaussian.score(trc,Y_train))

print ("Score on test values")

print (gaussian.score(tec,aa))

print ("Overall Score")

print (gaussian.score(tec,aa)+gaussian.score(trc,Y_train))
# I am going to go ahead with the kNN model as it has the highest overall score
# Plotting all the model scores

mod=["LogReg","SVM","kNN","GNB"]

d=[0.126520847573,0.286807928913,0.319264069264,0.0525860104807]

ff = {'Model': mod, 'OverallScore': d}

df = DataFrame(data=ff)

ax=sns.barplot(x='Model', y='OverallScore',data=df)

ax.set(xlabel='Model', ylabel='OverallScore')

plt.show()
natrain=train_df[train_df['Age'].isnull()] #All rows in train_df with NaNs

trainfin = train_df[np.isfinite(train_df['Age'])]  #All finite age rows in train_df
temp=natrain #temporary data frame

a1=le.fit_transform(temp['Sex']) #One hot Encoding

temp['Sexconv']= a1

temp=temp.drop(['Survived','Age','Sex'],axis=1)
temp
predtra=knn.predict(temp)
# This is to ensure that all the predicted age values lie between mean-std and mean+std in case the model predicts ages other than this range

for i in range(0,len(predtra)):

    if(predtra[i]< (average_age_titanic -std_age_titanic) or predtra[i] > (average_age_titanic -std_age_titanic)):

        predtra[i]= np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = 1)
predtra
# Drop age column which has all Nans

natrain=natrain.drop(['Age'],axis=1)
# Add age column with all predicted ages

natrain['Age']=predtra

natrain=natrain.drop(['Sex'],axis=1)
natrain
# Merge both dataframes which had finite and NaN age values originally

a2=le.fit_transform(trainfin['Sex'])

trainfin['Sexconv']= a2

trainfin=trainfin.drop(['Sex'],axis=1)

ftrain_df= trainfin.append(natrain) # Let ftrain_df be the final train_df

#Avoid the warning
# Repeat the same process for test_df to combat the missing age values

#test_df=test_df.drop(['PassengerId'],axis=1)

natest=test_df[test_df['Age'].isnull()]

testfin = test_df[np.isfinite(test_df['Age'])]
temp2=natest

a3=le.fit_transform(temp2['Sex'])

temp2['Sexconv']= a3

temp2=temp2.drop(['Age','Sex'],axis=1)

#Avoid Warning
temp2=temp2.drop(['PassengerId'],axis=1)
predtest=knn.predict(temp2)
for i in range(0,len(predtest)):

    if(predtest[i]< (average_age_titanic -std_age_titanic) or predtest[i] > (average_age_titanic -std_age_titanic)):

        predtest[i]= np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = 1)
natest['Age']=predtest

natest=natest.drop(['Sex'],axis=1)

a4=le.fit_transform(testfin['Sex'])

testfin['Sexconv']= a4

testfin=testfin.drop(['Sex'],axis=1)

ftest_df= testfin.append(natest)# Let ftest_df be the final test_df

pids=ftest_df['PassengerId']

ftest_df=ftest_df.drop(['PassengerId'],axis=1)
# Now we are ready to train a Random Forest on ftrain_df and predict survival on ftest_df

rlabels=np.asarray(ftrain_df['Survived'], dtype=np.uint64) 

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(ftrain_df.drop(['Survived'],axis=1),rlabels)
random_forest.score(ftrain_df.drop(['Survived'],axis=1),rlabels)
Surv_pred = random_forest.predict(ftest_df)
# Mean score using 10-fold cross validation

from sklearn.cross_validation import cross_val_score

np.mean(cross_val_score(random_forest, ftrain_df.drop(['Survived'],axis=1),rlabels, cv=10))
submission = pd.DataFrame({

        "PassengerId": pids,

        "Survived": Surv_pred

    })

submission.to_csv("titanicsurvival.csv", index=False)