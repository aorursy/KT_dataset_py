import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn.svm import SVC #support vector Machine

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.model_selection import train_test_split #training and testing data split

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix #for confusion matrix

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing



# for plotting

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



# remove rows with missing values from training set

train = train[np.isfinite(train['Age'])]



# pre-processing

def changeFeatures(df, test_data=False):

    # add nFamily column

    df['nFamily'] = df['SibSp'] + df['Parch']

    

    # add initial

    df['Initial']=0

    for i in df:

        df['Initial']=df.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations

        df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col',

                               'Rev','Capt','Sir','Don', 'Dona'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs',

                                            'Other','Other','Other','Mr','Mr','Mr','Other'],inplace=True)

    

    # only fill in Age values if it is testing data

    if test_data==True:

        df.loc[(df.Age.isnull())&(df.Initial=='Mr'),'Age']=33

        df.loc[(df.Age.isnull())&(df.Initial=='Mrs'),'Age']=36

        df.loc[(df.Age.isnull())&(df.Initial=='Master'),'Age']=5

        df.loc[(df.Age.isnull())&(df.Initial=='Miss'),'Age']=22

        df.loc[(df.Age.isnull())&(df.Initial=='Other'),'Age']=46

    

    # rationale: Continuous features are a problem in ML

    # add age_band column (7 bins)

    df['Age_band']=0

    df.loc[df['Age']<=10,'Age_band']=0

    df.loc[(df['Age']>11)&(df['Age']<=20),'Age_band']=1

    df.loc[(df['Age']>21)&(df['Age']<=30),'Age_band']=2

    df.loc[(df['Age']>31)&(df['Age']<=40),'Age_band']=3

    df.loc[(df['Age']>41)&(df['Age']<=50),'Age_band']=4

    df.loc[(df['Age']>51)&(df['Age']<=60),'Age_band']=5

    df.loc[(df['Age']>61)&(df['Age']<=70),'Age_band']=6

    df.loc[df['Age']>70,'Age_band']=7

    

    # add Fare_cat columns 

    df['Fare_cat']=0

    df.loc[(df['Fare']<=7.91),'Fare_cat']=1

    df.loc[(df['Fare']>7.91)&(df['Fare']<=14.454),'Fare_cat']=2

    df.loc[(df['Fare']>14.454)&(df['Fare']<=31),'Fare_cat']=3

    df.loc[(df['Fare']>31)&(df['Fare']<=513),'Fare_cat']=4

    

    # change nominal to categorical 

    df['Sex'] = df['Sex'].replace(['male','female'], [0,1])

    df['Embarked'] = df["Embarked"].replace(['S','C','Q'], [0,1,2])

    df['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)

    

    # drop columns that are not required

    df.drop(['Name','Age','Ticket','Fare','Cabin'],axis=1,inplace=True)

    

changeFeatures(train)

changeFeatures(test, test_data=True)



# replace missing value of "Embarked" in training set

train["Embarked"].fillna(0, inplace=True)



print("Train data: \n", pd.isnull(train).sum())

print("Test data: \n", pd.isnull(test).sum())
train_data,test_data = train_test_split(train,test_size=0.3,random_state=0,stratify=train['Survived'])

predictor_cols = ["Pclass","Sex", "Embarked", "nFamily", "Initial", "Age_band", "Fare_cat"]

target_col = "Survived"

train_X = train_data[predictor_cols]

train_Y = train_data[target_col]

test_X = test_data[predictor_cols]

test_Y = test_data[target_col]



scaler = StandardScaler()

train_X = scaler.fit_transform(train_X)

test_X = scaler.transform(test_X)
svm = SVC(kernel="rbf", C=0.1, gamma=0.1)

# I used ravel() as I was getting a data conversion warning

svm.fit(train_X, train_Y.values.ravel()) 

svm_prediction = svm.predict(test_X)

print("Accuracy for SVM = ", metrics.accuracy_score(svm_prediction, test_Y))
lr = LogisticRegression()

lr.fit(train_X, train_Y.values.ravel())

lr_prediction = lr.predict(test_X)

print("Accuracy for Logistic Regression = ", metrics.accuracy_score(lr_prediction, test_Y))
tree = DecisionTreeClassifier()

tree.fit(train_X, train_Y.values.ravel())

tree_prediction = tree.predict(test_X)

print("Accuracy for Decision Tree = ", metrics.accuracy_score(tree_prediction, test_Y))
knn = KNeighborsClassifier()

knn.fit(train_X, train_Y.values.ravel())

knn_prediction = knn.predict(test_X)

print("Accuracy for kNN = ", metrics.accuracy_score(knn_prediction, test_Y))
nb = GaussianNB()

nb.fit(train_X, train_Y.values.ravel())

nb_prediction = nb.predict(test_X)

print("Accuracy for Gaussian NB = ", metrics.accuracy_score(nb_prediction, test_Y))
rf = RandomForestClassifier(n_estimators=15)

rf.fit(train_X, train_Y.values.ravel())

rf_prediction = rf.predict(test_X)

print("Accuracy for Random Forest = ", metrics.accuracy_score(rf_prediction, test_Y))
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict



# datasets for cross validation

X = train[predictor_cols]

Y = train[target_col]



kfold = KFold(n_splits=10, random_state=22)

mean = []

accuracy =[]

std = []



classifiers = ["Radial SVM", "Logistic Regression",

              "KNN", "Decision Tree", "Naive Bayes",

              "Random Forest"]

models = [SVC(kernel="rbf"), LogisticRegression(),

         KNeighborsClassifier(n_neighbors=9), 

         DecisionTreeClassifier(), GaussianNB(),

         RandomForestClassifier(n_estimators=15)]



for model in models:

    cv_result = cross_val_score(model, X, Y, cv=kfold, scoring="accuracy")

    cv_result = cv_result

    mean.append(cv_result.mean())

    std.append(cv_result.std())

    accuracy.append(cv_result)

    

new_models_df = pd.DataFrame({"Mean": mean, "Standard Deviation": std}, index=classifiers)

new_models_df
plt.subplots(figsize=(12,6))

box = pd.DataFrame(accuracy, index=classifiers)

box.T.boxplot()
f, ax = plt.subplots(2,3,figsize=(12,10))



# svc

y_pred = cross_val_predict(SVC(kernel="rbf"), X, Y, cv=10)

sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[0,0], annot=True, fmt="2.0f")

ax[0,0].set_title("Matrix for SVM")



# logistic regression

y_pred = cross_val_predict(LogisticRegression(), X, Y, cv=10)

sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[0,1], annot=True, fmt="2.0f")

ax[0,1].set_title("Matrix for Logistic Regression")



# knn

y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9), X, Y, cv=10)

sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[0,2], annot=True, fmt="2.0f")

ax[0,2].set_title("Matrix for kNN")



# decision tress

y_pred = cross_val_predict(DecisionTreeClassifier(), X, Y, cv=10)

sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[1,0], annot=True, fmt="2.0f")

ax[1,0].set_title("Matrix for Decision Tree")



# naive bayes

y_pred = cross_val_predict(GaussianNB(), X, Y, cv=10)

sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[1,1], annot=True, fmt="2.0f")

ax[1,1].set_title("Matrix for Naive Bayes")



# random forest

y_pred = cross_val_predict(RandomForestClassifier(), X, Y, cv=10)

sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[1,2], annot=True, fmt="2.0f")

ax[1,2].set_title("Matrix for Random Forest")



plt.subplots_adjust(hspace=0.5, wspace=0.1)

plt.show()
from sklearn.model_selection import GridSearchCV

C = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]

gamma = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

kernel = ["rbf"]

hyper = {"kernel":kernel, "C":C, "gamma":gamma}



gd = GridSearchCV(estimator=SVC(), param_grid=hyper, verbose=True)

gd.fit(X,Y)

print(gd.best_score_)

print(gd.best_estimator_)
final_model = SVC(C=0.25, cache_size=200, class_weight=None, coef0=0.0, 

           decision_function_shape="ovr", degree=3, gamma=0.1,

           kernel="rbf", max_iter=-1, probability=False, random_state=None,

           shrinking=True, tol=0.001, verbose=False)

y_pred = cross_val_predict(final_model, X, Y, cv=10)

sns.heatmap(confusion_matrix(Y, y_pred), annot=True, fmt="0.2f")

plt.show()
final_model.fit(train_X, train_Y)



test2_X = test[predictor_cols]

test2_X = scaler.transform(test2_X)

predictions = final_model.predict(test2_X)

my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

my_submission.to_csv('submission.csv', index=False)