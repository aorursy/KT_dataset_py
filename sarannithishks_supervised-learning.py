import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import sklearn.utils as skutils
import math
from sklearn.model_selection import train_test_split
ploan_data = pd.read_csv("/kaggle/input/bank-personal-loan/Bank_Personal_Loan_Modelling.csv")
print ("Rows:",ploan_data.shape[0] ," Columns:",ploan_data.shape[1] )
ploan_data.head(10)
ploan_data.info()
#This clearly shows that all the data types are numerical and there are no objects.

ploan_data.isna().sum()
#The below results show that there are no null data in the dataset
ploan_data.describe().T
#Since there are many categorical columns , we see weird values for mean and median in the data description.
#Columns  "Personal Loan","Securities Account","CD Account","Online","Credit Card" are categorical columns.

#Transforming the data type of categorical columns
ploan_data["Personal Loan"] = ploan_data["Personal Loan"].astype("category")
ploan_data["Securities Account"] = ploan_data["Securities Account"].astype("category")
ploan_data["CD Account"] = ploan_data["CD Account"].astype("category")
ploan_data["Online"] = ploan_data["Online"].astype("category")
ploan_data["CreditCard"] = ploan_data["CreditCard"].astype("category")
ploan_data["ZIP Code"] = ploan_data["ZIP Code"].astype("category")

print(ploan_data.info())
ploan_data.describe().T

#Data distribution of every attribute 
sns.pairplot(ploan_data)



sns.heatmap(ploan_data.corr(), annot = True, cmap= 'coolwarm')
ploan_cleaned = ploan_data.drop(["Experience","ID","ZIP Code"],axis=1)

ploan_cleaned.head()
#Target Column Distribution
#Since the target column is a categorical column, we use the plot of counts.

value_counts = ploan_cleaned["Personal Loan"].value_counts()
print(value_counts)
sns.barplot(ploan_cleaned["Personal Loan"].unique(),value_counts)


# From the barplot and value_counts , we see a heavy imbalance in classes. 
# Class imbalance can cause the model to behave wrongly on the test data since the model would not have seen much of 
# data in the minority class. 

# So we try to resample the data to manage the class imbalance problem
ploan_cleaned_min = ploan_cleaned[ploan_cleaned["Personal Loan"] == 1] 
ploan_cleaned_maj = ploan_cleaned[ploan_cleaned["Personal Loan"] == 0]
ploan_cleaned_min_upsampled = skutils.resample(ploan_cleaned_min,n_samples=4520,random_state=1);

ploan_upsampled = pd.concat([ploan_cleaned_maj,ploan_cleaned_min_upsampled])

ploan_upsampled["Personal Loan"].value_counts()
                                     
# Splitting the data into independent and dependent variables

y = ploan_upsampled["Personal Loan"]
X = ploan_upsampled.drop(["Personal Loan"],axis=1)

#Splitting the data into training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)

print(y_test.shape)
from sklearn.linear_model import LogisticRegression as logisticRegressor
import sklearn.metrics as skmetrics
from  sklearn.neighbors import KNeighborsClassifier as knnClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import StandardScaler
def get_accuracy(algorithm,X_train,y_train):
    
     #Doing 10 fold cross validation to find the average accuracy on the training data
     cross_val_scores = cross_val_score(algorithm,X_train,y_train,cv=10);
     print(cross_val_scores)
     print("Average cross validation accuracy on training data:", cross_val_scores.mean()*100,"%")
     
    
def fit_and_predict(algorithm,hyperparameter, X_train,y_train,X_test,y_test):
     
     if (hyperparameter != None):
      model = GridSearchCV(algorithm,hyperparameter).fit(X_train,y_train)
     else:
      model = algorithm.fit(X_train,y_train)
     y_pred = model.predict(X_test)
     test_score = skmetrics.accuracy_score(y_test,y_pred)
        
        
     print("Accuracy score on test data:",test_score*100,"%")
     return model,y_pred
algorithm = logisticRegressor(max_iter=10000,random_state=1);

hyperparameter = {'solver' : ['newton-cg', 'lbfgs','liblinear', 'sag', 'saga']}

get_accuracy(algorithm,X_train,y_train)
logisticModel, y_pred_log = fit_and_predict(algorithm,hyperparameter, X_train,y_train,X_test,y_test)


print(logisticModel.best_estimator_)

scaler = StandardScaler()
scaledX_train = scaler.fit_transform(X_train)
scaledX_test  = scaler.fit_transform(X_test)


model = logisticRegressor(random_state=1)
model.fit(scaledX_train,y_train)
y_pred = model.predict(scaledX_test)
test_score = skmetrics.accuracy_score(y_test,y_pred)
print("Accuracy score on test data:",test_score*100,"%")
algorithm = knnClassifier();

hyperparameter = {'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
                   'n_neighbors': np.arange(5 , 20 , 2),
                   'metric': ['euclidean','manhattan','minkowski']}

get_accuracy(algorithm,X_train,y_train)

knnModel ,y_pred_knn = fit_and_predict(algorithm,hyperparameter,X_train,y_train,X_test,y_test)

print(knnModel.best_estimator_)

#Gaussian Naive Bayes is chosen here as the features are partially continuous and partially categorical

get_accuracy(algorithm,X_train,y_train)
NBModel,y_pred_NB = fit_and_predict(GaussianNB(),None,X_train,y_train,X_test,y_test)
def confusion_matrix(y_test,y_pred):
    
        data = {'y_Actual':y_test,
                'y_Predicted': y_pred
                }

        cm_df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
   
       
        cm_crosstab = pd.crosstab(cm_df['y_Actual'], cm_df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
       
        sns.heatmap(cm_crosstab, annot=True,fmt='d')
    

confusion_matrix(y_test,y_pred_log)

report = skmetrics.classification_report(y_test,y_pred_log)
print(report)


confusion_matrix(y_test,y_pred_knn)

report = skmetrics.classification_report(y_test,y_pred_knn)
print(report)


confusion_matrix(y_test,y_pred_NB)

report = skmetrics.classification_report(y_test,y_pred_NB)
print(report)

