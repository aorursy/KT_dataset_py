#Problem---Evaluate the model

#Solution--- We will first create a pipeline that preprocesses the data then trains the data and then evalutes it using cross-validation:



#importing libraries

from sklearn import datasets

from sklearn import metrics

from sklearn.model_selection import KFold, cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler



#Load digits dataset

digits =datasets.load_digits()



#cretaing features matrix

features = digits.data



#creating target vector

target = digits.target



#creating standardizer

standardizer = StandardScaler()



#create logistic regression object

logit = LogisticRegression()



#creating a pipeline that standardizes, then run logistic regression

pipeline = make_pipeline(standardizer, logit)



#create k-Fold cross-validation

kf = KFold(n_splits =10, shuffle = True, random_state =1)



#conduct k-fold cross validation 

cv_results =  cross_val_score(pipeline, #pipeline

                             features, # feature matrix

                             target, # target vector

                             cv = kf, #cross- validation technique

                             scoring="accuracy", # loss function

                             n_jobs = -1) # use all CPU scores



cv_results.mean()
cv_results
#Importing library

from sklearn.model_selection import train_test_split



#creating training and test sets

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.1, random_state =1)



#Fit Standardizer to training sets

standardizer.fit(features_train)



#applying to both training and test sets

fetaures_train_std = standardizer.transform(features_train)

features_test_std = standardizer.transform(features_test)

#create a pipeline

pipeline = make_pipeline(standardizer, logit)



#the we wun KFCV using that pipeline and scikit does all the work for us



#Do k-kfold cross-validation

cv_results = cross_val_score(pipeline , features, target, cv= kf, scoring ="accuracy", n_jobs =-1)

#Problem---Simple baseline regression model to compare against your model.

#Solution--- We will use scikit-learn's DummyRegressor to create a simple model to use as a baseline.



#importing libraries

from sklearn.datasets import load_boston

from sklearn.dummy import DummyRegressor

from sklearn.model_selection import train_test_split



#loading data

boston = load_boston()



#creating features

features, target = boston.data, boston.target



#Make test and training split

features_train, features_test, target_train, target_test= train_test_split(features, target, random_state =0)



# creating a dummy regressor

dummy = DummyRegressor(strategy = 'mean')



#"Train" dummy regressor

dummy.fit(features_train, target_train)



# getting R-squared score

dummy.score(features_test, target_test)

#To compare , we train our model and evaluate the performance score:



#importing library

from sklearn.linear_model import LinearRegression



# Train simple linear regressioj model

ols = LinearRegression()

ols.fit(features_train, target_train)



# Getting R-squared score

ols.score(features_test, target_test)
#creating summy regressor that predicts 20's for everything

clf = DummyRegressor(strategy = 'constant', constant = 20)

clf.fit(features_train, target_train)



#Evaluating score

clf.score(features_test, target_test)

#Problem--- Create a baseline classifier to compare against your model

#Solution--We will use scikit-learn's DummyClassifier:



#importing libraries

from sklearn.datasets import load_iris

from sklearn.dummy import DummyClassifier

from sklearn.model_selection import train_test_split



#load data

iris = load_iris()



#create target vector and feature matrix

features, target = iris.data, iris.target





#split into training and testsset

features_train, features_test, target_train, target_test =  train_test_split( features, target, random_state =0 )



#creating dummy classifier

dummy =DummyClassifier(strategy = "uniform", random_state=1)



#train model

dummy.fit(features_train, target_train)



#get accuracy score

dummy.score(features_test, target_test)
#By comparing the baseline classifier to our trained classifier, we can see the improvement:



# load library

from sklearn.ensemble import RandomForestClassifier



#create Classifier

classifier = RandomForestClassifier()



#train model

classifier.fit(features_train, target_train)



#get accuracy score

classifier.score(features_test, target_test)



#importing libraries

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_classification



#Generating features matrix and target vector

X, y =make_classification(n_samples =10000,

                         n_features=3,

                         n_informative =3,

                         n_redundant =0,

                         n_classes =2,

                         random_state =1)



#create Logistic Regression

logit =LogisticRegression()



#cross-validate model using accuracy

cross_val_score(logit, X,y,scoring="accuracy")
# Cross-validate model using precision

cross_val_score(logit, X, y, scoring="precision")
# Cross-validate model using recall

cross_val_score(logit, X, y, scoring="recall")
# Cross-validate model using f1

cross_val_score(logit, X, y, scoring="f1")
# Load library

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

# Create training and test split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=1)

# Predict values for training target vector

y_hat = logit.fit(X_train, y_train).predict(X_test)

# Calculate accuracy

accuracy_score(y_test, y_hat)