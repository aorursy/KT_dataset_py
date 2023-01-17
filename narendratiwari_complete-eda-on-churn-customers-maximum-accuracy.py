#importing useful libararies

import pandas as pd

import numpy as np

import seaborn as sea

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import scipy.stats as stats

from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv(r'../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
df.isna().sum()

df.columns

df.isnull().sum()

df.shape
df.drop(['customerID'],axis=1,inplace=True)
df.nunique()
for i in df.columns:

    print(df[i].value_counts())
df.isin([" "]).sum()
df['TotalCharges'] = df['TotalCharges'].replace([" "], np.nan)
df.isna().sum().sum()
df.dropna(inplace = True)
df['TotalCharges']= df['TotalCharges'].astype(float)
df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)
df['SeniorCitizen'] = df['SeniorCitizen'].replace({1:'Yes', 0:'No'})
cat = [i for i in df.columns if df[i].dtypes == 'O']

num = [i for i in df.columns if df[i].dtypes != 'O']
df.replace(['No phone service'], ['No'], inplace = True)
df.replace({'No internet service':'No'}, inplace = True)
df[cat].nunique().plot(kind='barh')
fig, ax =plt.subplots(1,2, figsize=(15,5))

plt.figure(figsize=(5,5))

sea.countplot(x ='StreamingTV', hue = 'Churn' ,data =df, ax= ax[0])

sea.countplot(x ='PaymentMethod', hue = 'Churn' ,data =df, ax= ax[1])

fig.show()
fig, ax =plt.subplots(1,2, figsize=(15,5))

plt.figure(figsize=(5,5))

sea.countplot(x ='PaperlessBilling', hue = 'Churn' ,data =df, ax= ax[0])

sea.countplot(x ='Contract', hue = 'Churn' ,data =df ,ax = ax[1])

fig.show()
fig, ax =plt.subplots(1,2, figsize=(15,5))

plt.figure(figsize=(5,5))

sea.countplot(x ='InternetService', hue = 'Churn' ,data =df, ax= ax[0])

sea.countplot(x ='gender', hue = 'Churn' ,data =df, ax = ax[1])

fig.show()
plt.figure(figsize=(10,5))

ax = sea.countplot(x="Churn", hue="Contract", data=df);

ax.set_title('Contract Type vs Churn')
df[num].head()
plt.figure(figsize=(8,5))

plt.title("Monthly C,harges VS Total Charges")

plt.scatter(x = df.MonthlyCharges, y = df.TotalCharges)

plt.xlabel('Monthly Charges')

plt.ylabel('Total Charges')

plt.show()
plt.figure(figsize=(5,5))

df[['MonthlyCharges','tenure']].head(35).plot(kind='line')

plt.title("Monthly Charges VS Tenure")

plt.xlabel('Monthly Charges')

plt.ylabel('Tenure')

plt.show()
sea.countplot(x= 'Churn' ,data=df, hue='SeniorCitizen')
df.Contract.value_counts().plot(kind='pie', legend= True)
sea.distplot(df["tenure"], color="b")
sea.distplot(df["MonthlyCharges"], color="r")
sea.distplot(df["TotalCharges"], color="g")
df['Count_OnlineServices'] = (df[['OnlineSecurity', 'DeviceProtection', 'StreamingMovies', 'TechSupport','StreamingTV', 'OnlineBackup']] == 'Yes').sum(axis=1)

plt.figure(figsize=(10,5))

sea.countplot(x= 'Count_OnlineServices', hue= 'Churn', data =df)
ax = sea.boxplot(x='Churn', y = 'tenure', data=df)

ax.set_title('Churn vs Tenure', fontsize=20)
sea.violinplot(x="MultipleLines", y="tenure", hue="Churn", kind="violin",

                 split=True, palette="pastel", data=df)
cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

df1 = pd.melt(df[df["InternetService"] != "No"][cols]).rename({'value': 'Has service'}, axis=1)

plt.figure(figsize=(10, 5))

ax = sea.countplot(data=df1, x='variable', hue='Has service')

ax.set(xlabel='Additional service', ylabel='Num of customers')

plt.show()
plt.figure(figsize = (12,5))

sea.countplot(x= 'Churn' ,data=df, hue='PaymentMethod')
plt.figure(figsize = (12,5))

sea.boxplot(x="Contract", y="MonthlyCharges", hue="Churn", data=df)
y = df.Churn
df.drop('Churn', axis =1, inplace= True)
y = pd.DataFrame(y)

y['Churn'].replace(to_replace='Yes', value=1, inplace=True)

y['Churn'].replace(to_replace='No',  value=0, inplace=True)
df_temp =df
df = pd.get_dummies(df)
df.shape
scaler = MinMaxScaler()

df[num] = scaler.fit_transform(df[num])
df.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=0)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import plot_confusion_matrix, plot_roc_curve

model_lr = LogisticRegression()

result = model_lr.fit(X_train, y_train)

prediction_test = model_lr.predict(X_test)

metrics.accuracy_score(y_test, prediction_test)
disp = plot_roc_curve(model_lr, X_test, y_test)
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators=1000)

model_rf.fit(X_train, y_train)

prediction_test = model_rf.predict(X_test)

metrics.accuracy_score(y_test, prediction_test)

disp = plot_roc_curve(model_rf, X_test, y_test)
importances = model_rf.feature_importances_

weights = pd.Series(importances,

                 index=df.columns.values)

weights.sort_values()[-20:].plot(kind = 'barh')
from sklearn.svm import SVC

model_svm = SVC(kernel='linear') 

model_svm.fit(X_train,y_train)

preds = model_svm.predict(X_test)

metrics.accuracy_score(y_test, preds)
disp = plot_roc_curve(model_svm, X_test, y_test)
import xgboost as xgb

model_gb=xgb.XGBClassifier(learning_rate=0.25,max_depth=4)

model_gb.fit(X_train, y_train)

model_gb.score(X_test,y_test)
params={

 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],   

}
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

random_search=RandomizedSearchCV(model_gb,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

random_search.fit(X_train, y_train)
random_search.best_params_
import xgboost as xgb

model_gb=xgb.XGBClassifier(learning_rate=0.05,max_depth=3)

model_gb.fit(X_train, y_train)

model_gb.score(X_test,y_test)
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import StackingClassifier
level0 = list()

level0.append(('lr', LogisticRegression()))

level0.append(('knn', KNeighborsClassifier()))

level0.append(('cart', DecisionTreeClassifier()))

level0.append(('svm', SVC()))

level0.append(('bayes', GaussianNB()))

# define meta learner model

level1 = LogisticRegression()

# define the stacking ensemble

modelx = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

modelx.fit(X_test, y_test)
preds = modelx.predict(X_test)

metrics.accuracy_score(y_test, preds)