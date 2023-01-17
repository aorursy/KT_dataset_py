#important library

!pip install dabl

import pandas as pd

import pandas_profiling as npp

import dabl

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from xgboost import XGBClassifier
#read data

data = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')

df= data.copy()
#EDA

profile = npp.ProfileReport(data)

profile
#label enconding

label_encoder = preprocessing.LabelEncoder() 

data['Gender']= label_encoder.fit_transform(data['Gender']) 

data['Geography']= label_encoder.fit_transform(data['Geography'])
#drop columns

data.drop('Surname',axis=1,inplace=True)
#graphs

df_hello = dabl.clean(data, verbose=1)

types = dabl.detect_types(df_hello)

types
#exicited

dabl.plot(df_hello, target_col="Exited")
ec = dabl.SimpleClassifier(random_state=0).fit(df_hello, target_col="Exited") 
#standard scaling

X = data.drop('Exited', axis =1)

y = data['Exited']

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs')

log_reg.fit(X_train, y_train)
y_predicted = log_reg.predict(X_test)

y_predicted

cm = confusion_matrix(y_test, y_predicted)

cm
import seaborn as sn

import matplotlib.pyplot as plt

plt.figure(figsize = (10,7))

sn.heatmap(cm, annot=True)

plt.xlabel('Predicted')

plt.ylabel('actual')
X = df.iloc[:, 3:13].values

y = df.iloc[:, 13].values
# Encoding categorical data

labelencoder_X_1 = LabelEncoder()

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()

X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1])

X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
classifier = XGBClassifier()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# Making the confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
# Applying K-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)

accuracies.mean()

accuracies.std()