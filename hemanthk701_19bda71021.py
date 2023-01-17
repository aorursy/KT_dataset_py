# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing all the necessacry Packages
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict,KFold
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
#Reading the train and test datasets
data = pd.read_csv("/kaggle/input/bda-2019-ml-test/Train_Mask.csv")
test = pd.read_csv("/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv")
#This function gives the dimension,data information,columns with missing value
def first(data):
    print("Dimension")
    print(data.shape)
    print("------------------------------------------------------")
    print("Data Info")
    print(data.info())
    print("------------------------------------------------------")
    #to print columns name which have missing value in them
    print("Columns with missing values")
    print(data[data.columns[data.isnull().any()].tolist()].isnull().sum())
    #print("Data Na")
    #print(data.isna().sum())
    print("------------------------------------------------------")
    print("Columns with more than 30% data missing")
    print(data.columns[(data.isnull().sum()/data.shape[0])>.3].tolist())
    print("------------------------------------------------------")
    print("Describe")
    print(data.describe())
    
first(data)
#This function gives the plot of whole dataset
def plot(data):
    #Dividing numerical and categorical data
    data_cat = data.select_dtypes(include=['object'])
    data_num = data.select_dtypes(include=['number'])
    for x in data_cat:
        #Count plot for categorical data
        data.groupby([x,x]).size().unstack().plot(kind='bar',stacked=True)
        plt.title(x)
        plt.show()
    for y in data_num:
        #Histogram for numerical data
        plt.hist(data[y], bins=10)
        plt.title(y)
        plt.show()
plot(data)        
#Dropping the target value from X
X = data.drop(['flag'],axis=1)
#Assigning target variable to Y
y = data['flag']
#Dropping timeindex column as it is only serial number
X = X.drop(['timeindex'],axis=1)
test = test.drop(['timeindex'],axis = 1)
#X.head()
#y.head()
#this function performs logistic regression and gives f1 score and also predicts for test data
def logreg(X_train,X_test,y_train,y_test,newdata):
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    classifier = LogisticRegression(random_state = 0,class_weight = "balanced",max_iter =500)
    #Fit model on X_train and y_train
    classifier.fit(X_train, y_train)
    #Predict on y_test
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test,y_pred )
    print("Confusion Matrix :\n", cm)
    print("Accuracy :" , accuracy_score(y_test,y_pred))
    print("Recall Score :",recall_score(y_test,y_pred))
    print("Precision Score :",precision_score(y_test,y_pred))
    #f1 = f1_score(y_test,y_pred)
    print("F1 Score : ",f1_score(y_test,y_pred))
    #Predict on new data
    a = classifier.predict(newdata)
    #return the answer
    return a 
#Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
result = logreg(X_train,X_test,y_train,y_test,test)
#submit(result1)
#This function is used to write csv in submission.csv format
def submit(result):
    submission = pd.read_csv("/kaggle/input/bda-2019-ml-test/Sample Submission.csv")
    #drop flag in submission
    submission = submission.drop('flag',axis=1)
    #Replace Flag
    submission['flag'] = result
    #Write
    submission.to_csv("submit.csv")

#submit(result)    

#this function is used to strandadize the data
def Scalar(X):
    #Calling function
    sc = StandardScaler()
    #Storing names of the columns
    names = X.columns
    #Strandadize
    scaled_df = sc.fit_transform(X)
    #convert to dataframe
    scaled_df = pd.DataFrame(scaled_df, columns=names)
    #return the data frame
    return scaled_df
X = Scalar(X)
test = Scalar(X)
#X.head()
#test.head()
#This function gives the heat map for the columns specified with data
def correlation(data,columns):
    #get the columns 
    simp = data[columns]
    #do correlation on the columns
    correlation= simp.corr()
    #correlation = pd.DataFrame(data=correlation)
    #Heat map
    sns.heatmap(correlation, annot=True)
 #Different columns names can be given
col = ['timeindex','currentBack']
correlation(data,col)
#This function performs kfold validation
def KFold_validation(splits,X,y):
    #Enter the number of splits to be done
    kf=KFold(n_splits=splits)
    kf.get_n_splits(X)
    #print(kf)
    for train_index,test_index in kf.split(X):
        print("----------------------------------------------------")
        #print("TRAin: ",train_index,"Test: ",test_index)
        #Getting different values for train and test and predicting using logistic regression
        X_train ,X_test = X.iloc[train_index],X.iloc[test_index]
        y_train ,y_test = y.iloc[train_index],y.iloc[test_index]
        logreg(X_train,X_test,y_train,y_test,test)
#Call the function with splits ,X and y
KFold_validation(10,X,y)
#this function is used select the optimal C value which is Inverse of regularization strength 
def c_parameter(X,y,C_array):
    #split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    #Here C_array is the array with possible C values
    C_param_range = C_array
    sepal_acc_table = pd.DataFrame(columns = ['C_parameter','F1 Score'])
    sepal_acc_table['C_parameter'] = C_param_range
    #plt.figure(figsize=(10, 10))
    j = 0
    for i in C_param_range:
        # Apply logistic regression model to training data
        lr = LogisticRegression(penalty = 'l2', C = i,random_state = 0,max_iter = 1000)
        lr.fit(X_train,y_train)
        #Predict using model
        y_pred = lr.predict(X_test)
        # Saving accuracy score in table
        sepal_acc_table.iloc[j,1] = f1_score(y_test,y_pred)
        j += 1
    print(sepal_acc_table)

    
C_array = [0.01,0.1,0.5,1,10,100,1000]
c_parameter(X,y,C_array)
#defining the svm model
svc_classifier = SVC(kernel = 'rbf', random_state = 0,C=1000,gamma=0.3)
#Function to perform SVM algorithm and give F1 score and predict the test
def svcc(X_train,X_test,y_train,y_test,new_data):
    #fit the model to train 
    svc_classifier.fit(X_train, y_train)
    #Predict on test
    y_pred = svc_classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("F1 Score : " ,f1_score(y_test,y_pred))
    #Predict on New data
    a = svc_classifier.predict(new_data)
    return a

#call function with X_train,,X_test,y_train,y_test and test
svcc(X_train,X_test,y_train,y_test,test)
#This function performs grid search for svm to give best hyperparameters
#from sklearn.svm import SVC
def grid(X_train,X_test,y_train,y_test):
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    #cm = confusion_matrix(y_test, y_pred)
    #print(cm)
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
    print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
    from sklearn.model_selection import GridSearchCV
    parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': 
               [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
    grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'f1_score',
                           cv = 10,
                           n_jobs = -1)
    grid_search = grid_search.fit(X_train, y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_
    print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
    print("Best Parameters:", best_parameters)
#grid(X_train,X_test,y_train,y_test)
#This function does the kfold validation for svm 
def KFold_validation(splits,X,y,test):
    kf=KFold(n_splits=splits)
    kf.get_n_splits(X)
    #print(kf)
    for train_index,test_index in kf.split(X):
        print("----------------------------------------------------")
        #print("TRAin: ",train_index,"Test: ",test_index)
        X_train ,X_test = X.iloc[train_index],X.iloc[test_index]
        y_train ,y_test = y.iloc[train_index],y.iloc[test_index]
        #logreg(X_train,X_test,y_train,y_test,test)
        svcc(X_train,X_test,y_train,y_test,test)

KFold_validation(5,X,y,test)
#This function does naive Bayes algorithm
def nav(X_train,X_test,y_train,y_test,newdata):
    nav_classifier = GaussianNB()
    nav_classifier.fit(X_train, y_train)
    y_pred = nav_classifier.predict(X_test)
    print("F1 Score :" , f1_score(y_test,y_pred))
    a = nav_classifier.predict(newdata)
    return a


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
nav(X_train, X_test, y_train, y_test,test)

#This Function performs Random forest
from sklearn.ensemble import RandomForestClassifier
def ran(X_train,X_test,y_train,y_test,newdata):
    ran_classifier = RandomForestClassifier(n_estimators = 40, random_state = 0)
    ran_classifier.fit(X_train, y_train)
    y_pred = ran_classifier.predict(X_test)
    print("F1 Score :",f1_score(y_test,y_pred))
    a = ran_classifier.predict(newdata)
    print(a)
    return a

ran(X_train,X_test,y_train,y_test,test)
#Optimal features using random forest
def features(X,y):
    clf=RandomForestClassifier(n_estimators=100)
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X,y)
    feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)
    print("Important Features are :\n",feature_imp)
    
features(X,y)