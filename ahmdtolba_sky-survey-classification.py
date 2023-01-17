import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

%matplotlib inline
data = pd.read_csv("../input/sloan-digital-sky-survey/Skyserver_SQL2_27_2018 6_51_39 PM.csv")
data.head()
data['class'].unique()
def convert(classmat):
    if classmat == "STAR":
        return 0
    elif classmat == "GALAXY":
        return 1
    else :
        return 2
data['class'] = data['class'].apply(convert)

data.head()
data.corr() 

correlations=[]
correlations.append( (data.columns))
for i in data.columns:
    correlations.append(data['class'].corr(data[i]))
correlations

y = data['class']
X = data.drop(['ra','dec','objid','rerun','camcol','field','specobjid','plate','mjd','fiberid','class'], axis=1)

X
y
sns.countplot(y)

# Import Libraries
from sklearn.preprocessing import StandardScaler
#----------------------------------------------------

#----------------------------------------------------
#Standard Scaler for Data
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X = scaler.fit_transform(X)

#showing data
print('X \n' , X[:10])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,  shuffle=True, random_state=42)


from sklearn.tree import DecisionTreeClassifier
#----------------------------------------------------
#Applying DecisionTreeClassifier Model 

DecisionTreeClassifierModel = DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=33) #criterion can be entropy
DecisionTreeClassifierModel.fit(X_train, y_train)

#Calculating Details
ScoreTrainig = DecisionTreeClassifierModel.score(X_train, y_train)
ScoreTesting =  DecisionTreeClassifierModel.score(X_test, y_test)
print('DecisionTreeClassifierModel Train Score is : ' , ScoreTrainig )
print('DecisionTreeClassifierModel Test Score is : ' ,ScoreTesting)

#Calculating Prediction
y_pred = DecisionTreeClassifierModel.predict(X_test)


#Calculating Confusion Matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)

# drawing confusion matrix
sns.heatmap(CM, center = True)
plt.show()
#Import Libraries
from sklearn.ensemble import RandomForestClassifier
#----------------------------------------------------

#----------------------------------------------------
#Applying RandomForestClassifier Model 

RandomForestClassifierModel = RandomForestClassifier(criterion = 'gini',n_estimators=100,max_depth=2,random_state=33) #criterion can be also : entropy 
RandomForestClassifierModel.fit(X_train, y_train)

#Calculating Details
print('RandomForestClassifierModel Train Score is : ' , RandomForestClassifierModel.score(X_train, y_train))
print('RandomForestClassifierModel Test Score is : ' , RandomForestClassifierModel.score(X_test, y_test))
#print('RandomForestClassifierModel features importances are : ' , RandomForestClassifierModel.feature_importances_)
#print('----------------------------------------------------')

#Calculating Prediction
y_pred = RandomForestClassifierModel.predict(X_test)

#Calculating Confusion Matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)

# drawing confusion matrix
sns.heatmap(CM, center = True)
plt.show()
models = pd.DataFrame({
    'Model': ['DecisionTreeClassifier', 'RandomForestClassifier'],
    'Score': [ DecisionTreeClassifierModel.score(X_test, y_test), RandomForestClassifierModel.score(X_test, y_test)]})
models.sort_values(by='Score', ascending=False)