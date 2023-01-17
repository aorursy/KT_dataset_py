# Importing the required libraries

import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Reading the csv file and putting it into 'df' object.

df = pd.read_csv("../input/Barley.data.csv")

df.head()
#collecting indepedent variables in X

X = df.iloc[:,1:332]

X_col = X.columns

X.head()
#collecting depedent variable in Y

Y=df['Predictor']

Y=pd.DataFrame(Y)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

Y_encoded = Y.apply(le.fit_transform)

Y_encoded.head()
#Savitzky-Golay filter with second degree derivative.

from scipy.signal import savgol_filter 



sg=savgol_filter(X,window_length=11, polyorder=3, deriv=2, delta=1.0)
sg_x=pd.DataFrame(sg, columns=X_col)



sg_x.head()
# Splitting the data into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(sg_x, Y_encoded ,

                                                    test_size=0.2,

                                                    random_state=123,stratify=Y)
from sklearn.ensemble import RandomForestClassifier



Rf = RandomForestClassifier(random_state=52)

Rf_fit=Rf.fit(X_train, y_train)


y_pred = Rf_fit.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Test Result:\n")        

print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, y_pred)))

print("Classification Report: \n {}\n".format(classification_report(y_test, y_pred)))

print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test,y_pred))) 
#Reduction of variables using Recursive Feature Elimination(RFE) techineque



from sklearn.feature_selection import RFE



# RFE with 10 features



rfe_10 = RFE(Rf,10)



rfe_10.fit(X_train, y_train)



# selected features

features_bool = np.array(rfe_10.support_)

features = np.array(X_col)

result = features[features_bool]

print('10 selected Features:')

print(result)            
y_pred = rfe_10.predict(X_test)



print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, y_pred)))

print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, y_pred))) 
# RFE with 15 features

rfe_15 = RFE(Rf,15)



# fit with 15 features

rfe_15.fit(X_train, y_train)



# selected features

features_bool = np.array(rfe_15.support_)

features = np.array(X_col)

result = features[features_bool]

print('15 selected Features:')

print(result)        

y_pred = rfe_15.predict(X_test)



print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, y_pred)))

print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, y_pred))) 
# RFE with 17 features



rfe_17 = RFE(Rf,17)



# fit with 17 features

rfe_17.fit(X_train, y_train)



# selected features

features_bool = np.array(rfe_17.support_)

features = np.array(X_col)

result = features[features_bool]

print('17 selected Features:')

print(result)     
y_pred = rfe_17.predict(X_test)



print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, y_pred)))

print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, y_pred))) 
# RFE with 20 features



rfe_20 = RFE(Rf,20)



# fit with 20 features

rfe_20.fit(X_train, y_train)



# selected features

features_bool = np.array(rfe_20.support_)

features = np.array(X_col)

result = features[features_bool]

print('20 selected Features:')

print(result)     
y_pred = rfe_20.predict(X_test)



print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, y_pred)))

print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, y_pred))) 
#collecting variables selected by Rfe 17 in X_imp

X_imp=sg_x[['756','759','783','789','790','842','843','854','858','954','982','983',

             '1001','1002','1038','1059','1064']]

X_col_imp=X_imp.columns

X_imp.head()
from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train1, y_test1 =train_test_split(X_imp, Y_encoded,

                                                    test_size=0.2,

                                                    random_state=123,stratify=Y)

X_train1.head()
from sklearn.ensemble import RandomForestClassifier



Rf = RandomForestClassifier(random_state=52)

Rf_fit=Rf.fit(X_train1, y_train1)
y_pred1 = Rf.predict(X_test1)
print("Test Result:\n")        

print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test1, y_pred1)))

print("Classification Report: \n {}\n".format(classification_report(y_test1, y_pred1)))

print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test1,y_pred1))) 
#sorting features with their importance

feature_importances = pd.DataFrame(Rf.feature_importances_,

                                   index = X_col_imp,

                                    columns=['importance']).sort_values('importance',ascending=False)



print(feature_importances)
features =X_col_imp

importances = Rf.feature_importances_

indices = np.argsort(importances)



plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='gray', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()