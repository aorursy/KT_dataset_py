import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
customer_data = pd.read_csv('../input/bank-customer-churn-modeling/Churn_Modelling.csv')

customer_data.head()
type(customer_data) # Data Frame
customer_data.info()

Total_Columns = customer_data.columns.to_list()
type(Total_Columns)
Total_Columns
dataset = customer_data.drop(['RowNumber','CustomerId','Surname'],axis =1)
dataset.info()
dataset['Geography']
dataset['Gender']
dataset =dataset.drop(['Geography','Gender'], axis =1) # dropping geography and gender columns from data frame.
dataset.head()
Geography = pd.get_dummies(customer_data.Geography).iloc[:,1:]
Geography
# Similarly for Gender

# If Gender is male , Male column will be 1
# If Gender is Female, Male column will be 0
Gender = pd.get_dummies(customer_data.Gender).iloc[:,1:]
Gender
# After one hot encoding these 2 coulmns(Geography and Gender). We need to concat it with our 'dataset'
dataset = pd.concat([dataset,Geography,Gender], axis =1)
dataset.info()

X = dataset.drop(['Exited'],axis=1)   # Features
y =dataset['Exited']  # target

# Splitting dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2, random_state =12)
# 80 % data for training the model and 20 % for testing the model
X_train.shape
X_test.shape

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200,random_state =0)
clf.fit(X_train,y_train)
predictor = clf.predict(X_test)
predictor

from sklearn.metrics import accuracy_score,classification_report
print("Accuracy of the model is : {} %".format(accuracy_score(y_test,predictor)*100))
print(classification_report(y_test,predictor))

feat_importances = pd.Series(clf.feature_importances_, index=X.columns)

feat_importances
feat_importances.nlargest(7).plot(kind='barh') # displays top 7 features which impacts the target variable
