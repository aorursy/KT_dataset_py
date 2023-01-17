import pandas as pd

import numpy as np



from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
data=pd.read_csv('../input/insurance.csv')
data.head()
#The data consists of 30 rows and 13 columns

data.shape
data.describe()
data.info()
data.isnull().sum()
#Factorizing the dependent variable

ins_factors=pd.factorize(data['Insurance Type'])



data['Insurance Type']=ins_factors[0]



ins_definations=ins_factors[1]



#creating one hot encoders

data=pd.get_dummies(data,drop_first=True)



data.head()
ins_factors[1]
data.dtypes,data.shape

#dividing the data into train test split

from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.2,random_state=10)
train.shape,test.shape
X_train=train.drop('Insurance Type',1)

y_train=train['Insurance Type']


X_test=test.drop('Insurance Type',1)

y_test=test['Insurance Type']
scaler=StandardScaler()
#scaling the independent varaibles so that the model could learn better

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)


# Making RF classifier object using entropy and random state as 10 in RF classifier

classifier=RandomForestClassifier(criterion='entropy',random_state=10)
classifier.fit(X_train,y_train)


#predicting the test data over out trained model

y_pred=classifier.predict(X_test)


#reverse factoring our dependent variable so that the resulst are readable.

ins_reversefactor=dict(zip(range(3),ins_definations))

ins_reversefactor
y_test = np.vectorize(ins_reversefactor.get)(y_test)

y_pred = np.vectorize(ins_reversefactor.get)(y_pred)
print("This is y_test",y_test)

print("This is y_pred",y_pred)
#Making a pandas cross table for visualizing the results

print(pd.crosstab(y_test, y_pred, rownames=['Actual Species'], colnames=['Predicted Species']))
prediction=pd.DataFrame(columns=['Actual','Predicted'])
prediction['Actual']=y_test

prediction['Predicted']=y_pred

prediction
#checking the accuracy of model

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred)*100)
print(classification_report(y_test, y_pred,labels=None, sample_weight=None))