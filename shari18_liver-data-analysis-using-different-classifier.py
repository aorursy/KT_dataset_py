import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
dataset = pd.read_csv('../input/indian_liver_patient.csv')
dataset.head()
dataset.info()
# fill the missing values with mean of the coressponding column
dataset['Albumin_and_Globulin_Ratio'] = dataset.Albumin_and_Globulin_Ratio.fillna(value = dataset['Albumin_and_Globulin_Ratio'].mean())
# let's build a correlation matrix and use seaborn to plot the heatmap of these
# correlation matrix
corr_matrix = dataset.corr()
sns.heatmap(corr_matrix,cmap="YlGnBu")
# from the heatmap, dark shades represent positive correlation and light shades represent negative correlation
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr_matrix,cmap="YlGnBu", annot=True, linewidths=.5, ax = ax)
# classifying data into independent and dependent variable
X = dataset.drop(['Dataset'],axis = 1).values
y = dataset['Dataset'].values
# encoding the categorical data of Gender varaible
labelencoder = LabelEncoder()
X[:,1] = labelencoder.fit_transform(X[:,1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# avoid dummy varaible trap
X = X[:,1:]
# creating test and training set data
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train,y_train)
y_pred = rf_classifier.predict(X_test)
rf_cm = confusion_matrix(y_test, y_pred)
rf_cr = classification_report(y_test, y_pred)
print("Accuracy Score in percentage : \n",round(accuracy_score(y_test,y_pred) * 100,2))
print("Random Forrest Confusion Matrix : \n",rf_cm)
print("Random Forrest Classification Report : \n",rf_cr)
# let's remove the highly correlated params and apply random forrest see whether the precision improves
# classifying data into independent and dependent variable
X_dropped = dataset.drop(['Dataset','Direct_Bilirubin','Aspartate_Aminotransferase','Albumin'],axis = 1).values
y_dropped = dataset['Dataset'].values

# encoding the categorical data of Gender varaible
labelencoder = LabelEncoder()
X_dropped[:,1] = labelencoder.fit_transform(X_dropped[:,1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_dropped = onehotencoder.fit_transform(X_dropped).toarray()

# avoid dummy varaible trap
X_dropped = X_dropped[:,1:]

# creating test and training set data
X_train_dropped,X_test_dropped,y_train_dropped,y_test_dropped = train_test_split(X_dropped, y_dropped, test_size = 0.2, random_state = 0)

# feature scaling
sc = StandardScaler()
X_train_dropped = sc.fit_transform(X_train_dropped)
X_test_dropped = sc.transform(X_test_dropped)
rf_classifier_dropped = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier_dropped.fit(X_train_dropped,y_train_dropped)
y_pred_dropped = rf_classifier_dropped.predict(X_test_dropped)
rf_cm_dropped = confusion_matrix(y_test_dropped, y_pred_dropped)
rf_cr_dropped = classification_report(y_test_dropped, y_pred_dropped)
print("Accuracy Score in percentage : \n",round(accuracy_score(y_test_dropped,y_pred_dropped) * 100,2))
print("Random Forrest Confusion Matrix : \n",rf_cm_dropped)
print("Random Forrest Classification Report : \n",rf_cr_dropped)
# let's create Naive Bayes model and fit the dataset
classifier_NB = GaussianNB()
classifier_NB.fit(X_train,y_train)
y_pred = classifier_NB.predict(X_test)
gnb_cm = confusion_matrix(y_test, y_pred)
gnb_cr = classification_report(y_test, y_pred)
print("Accuracy Score in percentage : \n",round(accuracy_score(y_test,y_pred) * 100,2))
print("Naive Bayes Confusion Matrix : \n",gnb_cm)
print("Naive Bayes Classification Report : \n",gnb_cr)
# applying logistic regression model to training set
# here we are considering dropped varaibles which are correlated
classifier_logistic = LogisticRegression()
classifier_logistic.fit(X_train_dropped,y_train_dropped)
y_pred_dropped = classifier_logistic.predict(X_test_dropped)
log_cm = confusion_matrix(y_test_dropped, y_pred_dropped)
log_cr = classification_report(y_test_dropped, y_pred_dropped)
print("Accuracy Score in percentage : \n",round(accuracy_score(y_test_dropped,y_pred_dropped) * 100,2))
print("Logistic Regression Confusion Matrix : \n",log_cm)
print("Logistic Regression Classification Report : \n",log_cr)
