import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("../input/brstcancer/B-Cancer.csv")
df.head()
df.tail()
df.info()

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X = df.drop(['Outcome'],axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
X_train.head()
y_train.head()
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
X=my_imputer.fit_transform(X)
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
y
sns.lineplot(x=df["radius_mean"],y=df["perimeter_mean"], hue=df["Outcome"])
sns.countplot(df['Outcome'])
sns.barplot(df['Outcome'],df['area_mean'])
sns.scatterplot(x = df['area_mean'],y= df['smoothness_mean'],hue=df['Outcome'])
sns.regplot(x = df['area_mean'],y= df['smoothness_mean'])
sns.lmplot(x='area_mean',y='smoothness_mean',hue='Outcome',data=df)
sns.swarmplot(x=df['Outcome'],y=df['smoothness_mean'])
sns.distplot(df['Time'])
sns.jointplot(df['perimeter_mean'],df['smoothness_mean'],kind='kde')
# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
sns.pairplot(df, hue = 'Outcome', 
             vars = ['Time', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean'] )

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot = True, cmap ='coolwarm', linewidths=2)
# Support vector classifier
from sklearn.svm import SVC
svc_classifier = SVC()
svc_classifier.fit(X_train, y_train)
y_pred_scv = svc_classifier.predict(X_test)
accuracy_score(y_test, y_pred_scv)
# Train with Standard scaled Data
svc_classifier2 = SVC()
svc_classifier2.fit(X_train_sc, y_train)
y_pred_svc_sc = svc_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_svc_sc)
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
accuracy_score(y_test, y_pred_rf)
cm = confusion_matrix(y_test,y_pred_rf)
print(cm)
plt.title('Heatmap of Confusion Matrix', fontsize = 15)
sns.heatmap(cm, annot = True)
plt.show()
y_test
# Train with Standard scaled Data
rf_classifier2 = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier2.fit(X_train_sc, y_train)
y_pred_rf_sc = rf_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_rf_sc)
# XGBoost Classifier
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_xgb)
# Train with Standard scaled Data
xgb_classifier2 = XGBClassifier()
xgb_classifier2.fit(X_train_sc, y_train)
y_pred_xgb_sc = xgb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_xgb_sc)
Test = pd.DataFrame({
     "Outcome":list(y_test)
     })
Test.to_csv("Test.csv", 
          index=False)
Test.to_csv(r'Test.csv')
from IPython.display import FileLink
FileLink(r'Test.csv')