#Importing all the necessary libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#importing scaling libraries in order to normalize and standardize data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#uploading the test and train data from the kaggle competition
from google.colab import files 
upload1 = files.upload()
upload2 = files.upload()
train = pd.read_csv('bank-train.csv')
train
#My initial thought process was to remove any rows that had null values, but I learned afterwards that this isn't possible
#I change my approach later on in the notebook, but I am leaving this in because this is the original way I tried it
train = train.dropna(how='any',axis=0) 
#Observing the dimensions of the train data
train.shape
#Finding the statistical summary of the features in the dataset 
train.describe
#Retaining and removing the id's of the dataset as they will not be useful in creating the model
ids = train['id']
train = train.drop(['duration','id'],1)
#Here I am observing the distributions of the variables to assess whether standardization is neccesary in the data
train.hist()
#Deteriming the number of unknowns present in each feature of the data set
for col in tester.columns:
  print("# of Unknowns in " + col + ": " + str((tester[col]=='unknown').sum()))
#Doing basic cleaning on the data by removing the 'default' feature and transforming 'housing' and 'loan' features into binary values
#I removed the default feature because there was a significant number of unknown values
train = train.drop(['default'],1)
train["housing"].replace(["yes","no"],[1,0],inplace=True)
train["loan"].replace(["yes","no"],[1,0],inplace=True)
#In this step, I used my initial approach of removing rows that contained unknown values
#As stated previously, I changed this later on but this was my initial though process.
for col in train.columns:
  if(train[col].dtype == np.float64 or train[col].dtype == np.int64):
      continue
  else:
      train = train.drop(train[train[col]=="unknown"].index)
#Dropping 'poutcome' because of the very low variance in the data set
train = train.drop('poutcome',1)
train
#Initially, I thought it would be profitable to replace the day_of_week category into two options, weekday or weekend
#However, upon realizing that every day was a weekday, I removed the feature entirely
train[['day_of_week']] = train[['day_of_week']].replace(dict.fromkeys(['mon','tue','wed','thu','fri'], 'weekday'))
train[['day_of_week']] = train[['day_of_week']].replace(dict.fromkeys(['sat','sun'], 'weekend'))
train.head(50)
#I also decided to remove the 'month' feature because I did not expect it to greatly influence the model
train = train.drop(['day_of_week','month'],1)
#In this step, I start to use one hot encoding with the help of get_dummies() to represent categorical features numerically
#After the dummy dataframe is created, I rejoin it with the original train dataset
dum_job = pd.get_dummies(train['job'])
train = pd.concat([train,dum_job],axis=1)
#Following with the previous process, I created dummy variables and encoded the remaining categorical features
dum_mar = pd.get_dummies(train['marital'])
train = pd.concat([train,dum_mar],axis=1)
dum_edu = pd.get_dummies(train['education'])
train = pd.concat([train,dum_edu],axis=1)
train = train.drop('education',1)
train = train.drop('job',1)
train = train.drop('marital',1)
#In this step, I used a simple function to determine what features are kept at different variance levels using variance thresholding
#My motivation for this was to remove any very low variance features that probably don't influence the model greatly
from sklearn.feature_selection import VarianceThreshold
def variance_threshold_selector(data, threshold=0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]
#Getting rid of duplicate columns in the train data
train = train.loc[:,~train.columns.duplicated()]
#Dropping the 'pdays' features because not many people were contacted prior and thus this feature had very low variance
train = train.drop(['pdays'],1)
train
#Here I start modeling the data with various different approaches
#First I split the train data into a train and test split
X = train.drop('y',1)
y = train['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#Then, I created a random forest model using the sklearn library on the train and test sets
#Using this model, I then predicted the outcome of the X_test data
sel = RandomForestClassifier(n_estimators = 100, bootstrap=True, max_features = 'sqrt')
sel.fit(X_train,y_train)
y_pred=sel.predict(X_test)
#This gave an initial accuracy of around 88.2%
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#I then realized that I could standardize the features using the StandardScaler library in sklearn
#My motivation behind this was to prevent variables with high ranges from influencing the model more than they should, and ensuring that
#each of the metric features in the data had a mean at 0 and a standard deviation of 1
X = train.drop('y',1)
X[['emp.var.rate', 'cons.price.idx','cons.conf.idx','euribor3m','nr.employed']] = StandardScaler().fit_transform(X[['emp.var.rate', 'cons.price.idx','cons.conf.idx','euribor3m','nr.employed']])
#Splitting the data once again
y = train['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#Running a random forest model once again gave a slightly higher score in terms of accuracy
sel = RandomForestClassifier(n_estimators = 100, bootstrap=True, max_features = 'sqrt')
sel.fit(X_train,y_train)
y_pred=sel.predict(X_test)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#Following the random forest model, I decided to try a Logistic Regression model to observe if it is any more accurate
#However, it showed about the same accuracy as the previous attempts
clf = LogisticRegression(fit_intercept=False, max_iter = 4000, solver='lbfgs')
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Here is all of the data preprocessing on the test data condensed into one cell for viewability
#It was at this point that I realized the rows of the data could not be deleted and thus I decided
#to replace the unknown values with 0 when appropriate
tester = pd.read_csv('bank-test.csv')
ids = tester['id']
tester = tester.drop(['duration','id'],1)
tester = tester.drop(['default'],1)
tester["housing"].replace(["yes","no",'unknown'],[1,0,0],inplace=True)
tester["loan"].replace(["yes","no",'unknown'],[1,0,0],inplace=True)
tester["contact"].replace(["cellular","telephone"],[1,0],inplace=True)
tester = tester.drop('poutcome',1)
tester = tester.drop(['day_of_week','month'],1)
dum_job = pd.get_dummies(tester['job'])
tester = pd.concat([tester,dum_job],axis=1)
dum_mar = pd.get_dummies(tester['marital'])
tester = pd.concat([tester,dum_mar],axis=1)
dum_edu = pd.get_dummies(tester['education'])
tester = pd.concat([tester,dum_edu],axis=1)
tester = tester.drop('education',1)
tester = tester.drop('job',1)
tester = tester.drop('marital',1)
tester = tester.loc[:,~tester.columns.duplicated()]
tester = tester.drop(['pdays'],1)
tester.head(100)
#Here is all of the data preprocessing previously done on the train data condensed into one cell for viewability
#It was at this point that I realized the rows of the data could not be deleted and thus I decided
#to replace the unknown values with 0 when appropriate
train = pd.read_csv('bank-train.csv')
train = train.drop(['duration','id'],1)
train = train.drop(['default'],1)
train["housing"].replace(["yes","no",'unknown'],[1,0,0],inplace=True)
train["loan"].replace(["yes","no",'unknown'],[1,0,0],inplace=True)
train["contact"].replace(["cellular","telephone"],[1,0],inplace=True)
train = train.drop('poutcome',1)
train = train.drop(['day_of_week','month'],1)
dum_job = pd.get_dummies(train['job'])
train = pd.concat([train,dum_job],axis=1)
dum_mar = pd.get_dummies(train['marital'])
train = pd.concat([train,dum_mar],axis=1)
dum_edu = pd.get_dummies(train['education'])
train = pd.concat([train,dum_edu],axis=1)
train = train.drop('education',1)
train = train.drop('job',1)
train = train.drop('marital',1)
train = train.loc[:,~train.columns.duplicated()]
train = train.drop(['pdays'],1)
train.head(100)
#Once again, I standardized certain features of the train data that had high ranges and variability
X = train.drop('y',1)
y = train['y']
X[['emp.var.rate', 'cons.price.idx','cons.conf.idx','euribor3m','nr.employed']] = StandardScaler().fit_transform(X[['emp.var.rate', 'cons.price.idx','cons.conf.idx','euribor3m','nr.employed']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#Performing a logistic regression on the standardized data proved to show slightly higher accuracy than before
clf = LogisticRegression(fit_intercept=False, max_iter = 4000, solver='lbfgs')
clf.fit(X, y)
tester[['emp.var.rate', 'cons.price.idx','cons.conf.idx','euribor3m','nr.employed']] = StandardScaler().fit_transform(tester[['emp.var.rate', 'cons.price.idx','cons.conf.idx','euribor3m','nr.employed']])
y_pred=clf.predict(tester)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#I used the results from the logistic regression on the bank-test data to submit into the competition
#First, I converted the predicted results into a series and concatenated this with the id column that was retained from prior steps
ser = pd.Series(y_pred)
results = pd.concat([ids, ser], axis=1)
results.head(50)
#Because the "Predicted" is the required column name of the submission results, I changed that here and created another csv file
results.rename(columns={0:'Predicted'}, 
                 inplace=True)
print(results)
results.to_csv('results.csv', index=False)
#Here are some alternative approaches that I tried following the sumbmission of my results
#First, I tried using a support vector machine model on the train data, but this gave a similar accuracy
from sklearn import svm
clf = svm.SVC()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Then I tried using the xgboost classifier on the train data
#This gave by far the best results off of the bat, and after doing more
#feature selection with recursive feature elimination (as shown below), this 
#model occasionally eclipsed accuracy of above 90%
import xgboost
mod = xgboost.XGBClassifier()
mod.fit(X_train,y_train)
tester[['emp.var.rate', 'cons.price.idx','cons.conf.idx','euribor3m','nr.employed']] = StandardScaler().fit_transform(tester[['emp.var.rate', 'cons.price.idx','cons.conf.idx','euribor3m','nr.employed']])
newtest = tester[['contact','emp.var.rate','cons.conf.idx','euribor3m','nr.employed']]
y_pred = mod.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#This is how I performed recursive feature elimination on the xgboost model to pick the
#5 best features as ranked by this method
#I used the RFE libary from sklearn to achieve this
from sklearn.feature_selection import RFE
rfe = RFE(mod, 5)
fit = rfe.fit(X_train, y_train)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
train.iloc[0]
#This is how I restructured the data to only contain the important features from the recursive feature elimination
X = train.drop('y',1)
y = train['y']
X[['emp.var.rate', 'cons.price.idx','cons.conf.idx','euribor3m','nr.employed']] = StandardScaler().fit_transform(X[['emp.var.rate', 'cons.price.idx','cons.conf.idx','euribor3m','nr.employed']])
newframe = X[['contact','emp.var.rate','cons.conf.idx','euribor3m','nr.employed']]
X_train, X_test, y_train, y_test = train_test_split(newframe, y, test_size=0.1)

