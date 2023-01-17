import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("/kaggle/input/german-credit-data-with-risk/german_credit_data.csv",index_col=0)
# Making a copy of the original data to analyze

df = data.copy()
df.head()
df.info()
df.describe()
df_num = df[['Age','Job','Credit amount','Duration']]

df_cat = df[['Sex','Housing','Saving accounts','Checking account','Purpose']]
pd.pivot_table(df, index = 'Risk', values = df_num)
for i in df_cat:

    classLabel = df.loc[df.Risk == 'good'][i].value_counts(normalize=True).index

    plt.pie(df.loc[df.Risk == 'good'][i].value_counts(normalize=True),

            labels = classLabel, startangle=90, autopct='%.1f%%')

    plt.title(i)

    plt.show()
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,10))

ser = df.loc[df.Sex == 'female']["Risk"].value_counts(normalize = True)

ax1.pie(ser,labels = ser.index, startangle=90, autopct='%.1f%%')

ax1.set_title('Female')

ser2 = df.loc[df.Sex == 'male']["Risk"].value_counts(normalize = True)

ax2.pie(ser2,labels = ser2.index, startangle=90, autopct='%.1f%%')

ax2.set_title('Male')

fig.show()
fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,15))

ser = df.loc[df.Housing == 'own']["Risk"].value_counts(normalize = True)

ser2 = df.loc[df.Housing == 'rent']["Risk"].value_counts(normalize = True)

ser3 = df.loc[df.Housing == 'free']["Risk"].value_counts(normalize = True)



ax1.pie(ser,labels = ser.index, startangle=90, autopct='%.1f%%')

ax1.set_title('Housing Status:Own')

ax2.pie(ser2,labels = ser2.index, startangle=90, autopct='%.1f%%')

ax2.set_title('Housing Status:rent')

ax3.pie(ser3,labels = ser3.index, startangle=90, autopct='%.1f%%')

ax3.set_title('Housing Status:free')

fig.show()
fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,15))

ser = df.loc[df["Checking account"] == 'little']["Risk"].value_counts(normalize = True)

ser2 = df.loc[df["Checking account"] == 'moderate']["Risk"].value_counts(normalize = True)

ser3 = df.loc[df["Checking account"] == 'rich']["Risk"].value_counts(normalize = True)



ax1.pie(ser,labels = ser.index, startangle=90, autopct='%.1f%%')

ax1.set_title('Checking account:little')

ax2.pie(ser2,labels = ser2.index, startangle=90, autopct='%.1f%%')

ax2.set_title('Checking account:moderate')

ax3.pie(ser3,labels = ser3.index, startangle=90, autopct='%.1f%%')

ax3.set_title('Checking account:rich')

fig.show()
fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(15,15))

ser = df.loc[df["Saving accounts"] == 'little']["Risk"].value_counts(normalize = True)

ser2 = df.loc[df["Saving accounts"] == 'moderate']["Risk"].value_counts(normalize = True)

ser3 = df.loc[df["Saving accounts"] == 'rich']["Risk"].value_counts(normalize = True)

ser4 = df.loc[df["Saving accounts"] == 'quite rich']["Risk"].value_counts(normalize = True)



ax1.pie(ser,labels = ser.index, startangle=90, autopct='%.1f%%')

ax1.set_title('Checking account:little')

ax2.pie(ser2,labels = ser2.index, startangle=90, autopct='%.1f%%')

ax2.set_title('Checking account:moderate')

ax3.pie(ser3,labels = ser3.index, startangle=90, autopct='%.1f%%')

ax3.set_title('Checking account:rich')

ax4.pie(ser4,labels = ser4.index, startangle=90, autopct='%.1f%%')

ax4.set_title('Checking account:quite rich')

fig.show()
for i in df.Purpose.unique():

        ser = df.loc[df["Purpose"] == i]["Risk"].value_counts(normalize = True)

        print('applicants with Purpose: ',i)

        print("%.2f" % (ser[1]*100),'% bad')

        print("%.2f" % (ser[0]*100),'% good')
good = df.loc[df['Risk'] == 'good']

bad = df.loc[df['Risk'] == 'bad']

for i in df_num:

    good[i].hist(alpha = 0.5,label='good')

    bad[i].hist(alpha = 0.5,label='bad')

    plt.title(i)

    plt.legend(['good','bad'])

    plt.show()
for i in df_cat:

    if(i != 'Purpose'):

        print(pd.pivot_table(df, index = 'Risk',values= 'Purpose', columns = i,aggfunc = 'count'))
print(df_num.corr())

sns.heatmap(df_num.corr())
data.isnull().sum().sort_values(ascending=False)
data['Checking account'].fillna("No Info", inplace = True) 

data['Saving accounts'].fillna("No Info", inplace = True) 
# The ratio of good and bad applicants

data['Risk'].value_counts()
# set dependent and independent values

y = data['Risk']

X = data.drop('Risk',axis = 1)
# Split train and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_train.value_counts()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder



num_attribs = ['Age','Job','Credit amount','Duration']

cat_attribs = ['Sex','Housing','Saving accounts','Checking account','Purpose']



num_pipeline = Pipeline([

        ('std_scaler', StandardScaler())

    ])

cat_pipeline = Pipeline([

        ("encoding", OneHotEncoder())

    ])

full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attribs),

        ("cat", cat_pipeline, cat_attribs)

    ])
X_train = full_pipeline.fit_transform(X_train)

X_test = full_pipeline.transform(X_test)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y_train = le.fit_transform(y_train)

y_test = le.transform(y_test)
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict
lr = LogisticRegression(random_state = 42)

y_pred_lr = cross_val_predict(lr,X_train,y_train,cv = 5)

print('Accuracy score:',accuracy_score(y_train, y_pred_lr))

print('Confusion Matrix')

print(confusion_matrix(y_train, y_pred_lr))
nb = GaussianNB()

y_pred_nb = cross_val_predict(nb,X_train,y_train,cv = 5)

print('Accuracy score:',accuracy_score(y_train, y_pred_nb))

print('Confusion Matrix')

print(confusion_matrix(y_train, y_pred_nb))
knn = KNeighborsClassifier(n_neighbors = 5)

y_pred_knn = cross_val_predict(knn,X_train,y_train,cv = 5)

print('Accuracy score:',accuracy_score(y_train, y_pred_knn))

print('Confusion Matrix')

print(confusion_matrix(y_train, y_pred_knn))
svc = SVC(probability=True,kernel = 'linear', random_state = 42)

y_pred_svc = cross_val_predict(svc,X_train,y_train,cv = 5)

print('Accuracy score:',accuracy_score(y_train, y_pred_svc))

print('Confusion Matrix')

print(confusion_matrix(y_train, y_pred_svc))
svcKernel = SVC(probability=True,kernel = 'rbf', random_state = 42)

y_pred_svcKernel = cross_val_predict(svcKernel,X_train,y_train,cv = 5)

print('Accuracy score:',accuracy_score(y_train, y_pred_svcKernel))

print('Confusion Matrix')

print(confusion_matrix(y_train, y_pred_svcKernel))
dt = DecisionTreeClassifier(max_depth = 5,random_state = 42)

y_pred_dt = cross_val_predict(dt,X_train,y_train,cv = 5)

print('Accuracy score:',accuracy_score(y_train, y_pred_dt))

print('Confusion Matrix')

print(confusion_matrix(y_train, y_pred_dt))
rf = RandomForestClassifier(n_estimators = 10, random_state = 42)

y_pred_rf = cross_val_predict(rf,X_train,y_train,cv = 5)

print('Accuracy score:',accuracy_score(y_train, y_pred_rf))

print('Confusion Matrix')

print(confusion_matrix(y_train, y_pred_rf))
xgb = XGBClassifier(random_state =42)

y_pred_xgb = cross_val_predict(xgb,X_train,y_train,cv = 5)

print('Accuracy score:',accuracy_score(y_train, y_pred_xgb))

print('Confusion Matrix')

print(confusion_matrix(y_train, y_pred_xgb))
from sklearn.model_selection import GridSearchCV

param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [4, 5, 7] }]



grid_search = GridSearchCV(knn, param_grid, cv=5, verbose=3)

grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_score_
knn = KNeighborsClassifier(n_neighbors = 7, weights = 'uniform')

y_pred_knn = cross_val_predict(knn,X_train,y_train,cv = 5)

print('Accuracy score:',accuracy_score(y_train, y_pred_knn))

print('Confusion Matrix')

print(confusion_matrix(y_train, y_pred_knn))
voting_clf = VotingClassifier(estimators = [('lr',lr),

                                            ('nb',nb),

                                            ('knn',knn),

                                            ('svc',svc),

                                            ('svcKernel',svcKernel),

                                            ('rf',rf),

                                            ('dt',dt),

                                            ('xgb',xgb)], 

                              voting = 'soft') 

y_pred = cross_val_predict(voting_clf,X_train,y_train,cv = 5)

print('Accuracy score:',accuracy_score(y_train, y_pred))

print('Confusion Matrix')

print(confusion_matrix(y_train, y_pred))
voting_clf2 = VotingClassifier(estimators = [('lr',lr),

                                            ('svc',svc),

                                            ('svcKernel',svcKernel),

                                            ('dt',dt),

                                            ('xgb',xgb)], 

                              voting = 'soft') 

y_pred = cross_val_predict(voting_clf2,X_train,y_train,cv = 5)

print('Accuracy score:',accuracy_score(y_train, y_pred))

print('Confusion Matrix')

print(confusion_matrix(y_train, y_pred))
voting_clf2 = VotingClassifier(estimators = [('lr',lr),

                                            ('svc',svc),

                                            ('svcKernel',svcKernel),

                                            ('dt',dt),

                                            ('xgb',xgb)], 

                              voting = 'soft') 

voting_clf2.fit(X_train,y_train)

y_pred = voting_clf2.predict(X_test)

print('Accuracy score:',accuracy_score(y_test, y_pred))

print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred))