# Data analysis tools

import pandas as pd

import numpy as np



# Data Visualization Tools

import seaborn as sns

import matplotlib.pyplot as plt



# Data Pre-Processing Libraries

from sklearn.preprocessing import LabelEncoder,StandardScaler



# For Train-Test Split

from sklearn.model_selection import train_test_split



# Libraries for various Algorithms

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.tree import DecisionTreeClassifier



# Metrics Tools

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import accuracy_score, f1_score



#For Receiver Operating Characteristic (ROC)

from sklearn.metrics import roc_curve ,roc_auc_score, auc
train = pd.read_csv('../input/mobile-price-classification/train.csv')

train.head()
train.info()
train.isnull().sum()
train.columns
sns.countplot(train["price_range"])

plt.xlabel("Class")

plt.ylabel("frequency")

plt.title("Checking imbalance")
sns.distplot(train.skew(),hist=False)

plt.show()
# Printing interquartile range (IQR) for each column

Q1 = train.quantile(0.25)

Q3 = train.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
# Boxplot visualization for columns with high IQR



plt.boxplot([train["battery_power"]])

plt.xticks([1],["battery_power"])

plt.show()

plt.boxplot([train["px_height"]])

plt.xticks([1],["px_height"])

plt.show()

plt.boxplot([train["px_width"]])

plt.xticks([1],["px_width"])

plt.show()

plt.boxplot([train["ram"]])

plt.xticks([1],["ram"])

plt.show()
train.corr()
fig = plt.figure(figsize=(15,12))

sns.heatmap(train.corr(),cmap = "RdYlGn", annot = True)
train['price_range'].unique()
import seaborn as sns

sns.pairplot(data=train,y_vars=['price_range'],x_vars=['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',

       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',

       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',

       'touch_screen', 'wifi'])
corr_matrix=train.corr()

corr_matrix['price_range']
sns.countplot(train['price_range'])

plt.show()
sns.boxplot(train['price_range'],train['talk_time'])
sns.boxplot(x='ram',y='price_range',data=train,color='red')
sns.pointplot(y="int_memory", x="price_range", data=train)
labels = ["3G-supported",'Not supported']

values=train['three_g'].value_counts().values

colors = ['green', 'red']

_, ax = plt.subplots()

ax.pie(values, labels=labels,colors=colors, autopct='%1.1f%%',shadow=True,startangle=90)

plt.show()
labels = ["4G-supported",'Not supported']

values=train['four_g'].value_counts().values

colors = ['green', 'red']

_, ax = plt.subplots()

ax.pie(values, labels=labels,colors=colors, autopct='%1.1f%%',shadow=True,startangle=90)

plt.show()
plt.figure(figsize=(10,6))

train['fc'].hist(alpha=0.5,color='yellow',label='Front camera')

train['pc'].hist(alpha=0.5,color='red',label='Primary camera')

plt.legend()

plt.xlabel('MegaPixels')
sns.pointplot(y="talk_time", x="price_range", data=train)
scaler = StandardScaler()
X=train.drop('price_range',axis=1)

y=train['price_range']



scaler.fit(X)

x = scaler.transform(X)



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=0)
#Fitting the model



knn = KNeighborsClassifier(n_neighbors=35)

knn.fit(x_train,y_train)
# Applying the model to the x_test



pred_knn = knn.predict(x_test)
# Finding Accuracy



KNN = accuracy_score(pred_knn,y_test)*100
# Confusion Matrix



cm_knn=confusion_matrix(pred_knn,y_test)

print(cm_knn)
# Classification Report that computes various

# metrics like Precision, Recall and F1 Score



print(classification_report(pred_knn,y_test))
#Fitting the model



gnb=GaussianNB()

gnb.fit(x_train,y_train)
# Applying the model to the x_test



pred_gnb = gnb.predict(x_test)
# Finding Accuracy



GNB = accuracy_score(pred_gnb,y_test)*100
# Confusion Matrix



cm_gnb=confusion_matrix(pred_gnb,y_test)

print(cm_gnb)
# Classification Report that computes various 

# metrics like Precision, Recall and F1 Score



print(classification_report(pred_gnb,y_test))
#Fitting the model



svc = SVC(probability=True)

svc.fit(x_train,y_train)



# Applying the model to the x_test

pred_svc = svc.predict(x_test)
# Finding Accuracy



SVC = accuracy_score(pred_svc,y_test)*100
# Confusion Matrix



cm_svc=confusion_matrix(pred_svc,y_test)

print(cm_svc)
# Classification Report that computes various 

#metrics like Precision, Recall and F1 Score



print(classification_report(pred_svc,y_test))
#Fitting the model



dtree_en = DecisionTreeClassifier()

clf = dtree_en.fit(x_train,y_train)
# Applying the model to the x_test



pred_dt = clf.predict(x_test)
# Finding Accuracy



DTREE = accuracy_score(pred_dt,y_test)*100
# Confusion Matrix



cm_dt=confusion_matrix(y_test,pred_dt)

print(cm_dt)



# Classification Report that computes various 

# metrics like Precision, Recall and F1 Score



print(classification_report(y_test,pred_dt))
#Fitting the model



GBC=GradientBoostingClassifier(n_estimators=150)

GBC.fit(x_train,y_train)
# Applying the model to the x_test



Y_predict=GBC.predict(x_test)
# Finding Accuracy



gbc = accuracy_score(y_test,Y_predict)*100
# Confusion Matrix



cm_gbc=confusion_matrix(y_test,Y_predict)

print(cm_gbc)



# Classification Report that computes various 

# metrics like Precision, Recall and F1 Score



print(classification_report(y_test,Y_predict))
#Fitting the model



rfc = RandomForestClassifier(n_estimators=30,criterion='gini',random_state=1,max_depth=10)

rfc.fit(x_train, y_train)
# Applying the model to the x_test



pred_rf= rfc.predict(x_test)
# Finding Accuracy



RFC = accuracy_score(y_test,pred_rf)*100
# Confusion Matrix



cm_rf=confusion_matrix(pred_rf,y_test)

print(cm_rf)
# Classification Report that computes various 

# metrics like Precision, Recall and F1 Score



print(classification_report(pred_rf,y_test))
#Fitting the model. Base model is chosen to be Decision Tree



model = DecisionTreeClassifier(criterion='entropy',max_depth=1,random_state=0)

adaboost = AdaBoostClassifier(n_estimators=80, base_estimator=model,random_state=0)

adaboost.fit(x_train,y_train)
# Applying the model to the x_test



pred = adaboost.predict(x_test)
# Finding Accuracy



ada = accuracy_score(y_test,pred)*100
# Confusion Matrix



cm_ada=confusion_matrix(pred,y_test)

print(cm_ada)



# Classification Report that computes various 

# metrics like Precision, Recall and F1 Score



print(classification_report(pred,y_test))
#Fitting the model



xgb =  XGBClassifier(learning_rate =0.000001,n_estimators=1000,max_depth=5,min_child_weight=1,

                     subsample=0.8,colsample_bytree=0.8,nthread=4,scale_pos_weight=1,seed=27)

xgb.fit(x_train, y_train)
# Applying the model to the x_test





predxg = xgb.predict(x_test)



# Finding Accuracy

xg = accuracy_score(y_test,predxg)*100

# Confusion Matrix



cm_xg=confusion_matrix(predxg,y_test)

print(cm_xg)



# Classification Report that computes various 

# metrics like Precision, Recall and F1 Score



print(classification_report(predxg,y_test))
# Accuracy values for all the models

print("1)  KNN                    :",round(KNN, 2))

print("2)  Naive-Bayes            :",round(GNB, 2))

print("3)  SVM                    :",round(SVC, 2))

print("4)  Decision Tree          :",round(DTREE, 2))

print("5)  Gradient Boosting      :",round(gbc, 2))

print("6)  Random Forest          :",round(RFC, 2))

print("7)  AdaBoost               :",round(ada, 2))

print("8)  XGBoost                :",round(xg, 2))
test = pd.read_csv('../input/mobile-price-classification/test.csv')

test.head()
test = test.drop('id',axis=1) 

scaler.fit(test)

x = scaler.transform(test)
price = GBC.predict(x)
print(price)
test["price_range"]=price
test.to_csv('Submission_Price_Detection.csv')
submission = pd.read_csv('./Submission_Price_Detection.csv')
submission.head()