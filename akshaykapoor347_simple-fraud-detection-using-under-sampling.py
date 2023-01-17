import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("../input/creditcard.csv")
data.head()
sns.countplot(data['Class'])
#We can see the imbalance in data
from sklearn.preprocessing import StandardScaler

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Time','Amount'],axis=1)
data.head()
#After standardizing let us try to resolve imbalance by using undersampling as we have a large dataset
X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']
# Number of data points in the minority class
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)
normal_indices = data[data.Class == 0].index
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)
# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

# Under sample dataset
under_sample_data = data.iloc[under_sample_indices,:]

X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))
#This is our undersampled data
from sklearn.model_selection import train_test_split

# Whole dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))

# Undersampled dataset
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample
                                                                                                   ,y_undersample
                                                                                                   ,test_size = 0.3
                                                                                                   ,random_state = 0)
print("")
print("Number transactions train dataset: ", len(X_train_undersample))
print("Number transactions test dataset: ", len(X_test_undersample))
print("Total number of transactions: ", len(X_train_undersample)+len(X_test_undersample))
#Let us now use machine learning and create a model to predict frauds
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# I have created a function to perform k folds cross validation which helps in obtaining a better insight to test the accuracy of the model
# More info at https://www.analyticsvidhya.com/blog/2018/05/improve-model-performance-cross-validation-in-python-r/

def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  predictions = model.predict(data[predictors])
  
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(n_splits=5)
  error = []
  for train, test in kf.split(data[predictors],data[outcome]):
    # Filter the training data
    train_predictors = (data[predictors].iloc[train,:])
    train_target = data[outcome].iloc[train]
    model.fit(train_predictors, train_target)
    
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
    
  model.fit(data[predictors],data[outcome]) 
X_undersample.head()
#Using Logistic regression using the entire dataset
lg_model = LogisticRegression()
lg_model.fit(X_train, y_train)

Y_pred = lg_model.predict(X_test)

predictor_var = list(X_train[1:])
outcome_var='Class'
classification_model(lg_model,data,predictor_var,outcome_var)
print(confusion_matrix(y_test, Y_pred))
print(classification_report(y_test, Y_pred))
#We can see that we have a very low recall
#That means real frauds are not detected properly which is a big problem for any banking company.
lg_model = LogisticRegression()
lg_model.fit(X_train_undersample, y_train_undersample)

Y_pred = lg_model.predict(X_test_undersample)

predictor_var = list(X_train_undersample[1:])
outcome_var='Class'
classification_model(lg_model,under_sample_data,predictor_var,outcome_var)
#We have a good accuracy. let us check recall and precision
print(confusion_matrix(y_test_undersample, Y_pred))
print(classification_report(y_test_undersample, Y_pred))
#Let us now apply this model to the entire dataset
Y_pred = lg_model.predict(X_test)

predictor_var = list(X_train[1:])
outcome_var='Class'
classification_model(lg_model,data,predictor_var,outcome_var)
print(confusion_matrix(y_test, Y_pred))
print(classification_report(y_test, Y_pred))
#We can see we have a recall of 0.93 and hence we can detect frauds with the accuracy of 0.93
#Even though we have a bad precision that means users might get calls regarding suspicisous activity
#but that is fine as long as real frauds are detected.
#Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train_undersample, y_train_undersample)

Y_pred = rf_model.predict(X_test)

#predictor_var = list(X_train[1:])
#outcome_var='Class'
#classification_model(rf_model,data,predictor_var,outcome_var)
print(confusion_matrix(y_test, Y_pred))
print(classification_report(y_test, Y_pred))
#Using SVM
svm_model = SVC()
svm_model.fit(X_train_undersample, y_train_undersample)

Y_pred = svm_model.predict(X_test)

#predictor_var = list(X_train[1:])
#outcome_var='Class'
#classification_model(svm_model,data,predictor_var,outcome_var)
print(confusion_matrix(y_test, Y_pred))
print(classification_report(y_test, Y_pred))
