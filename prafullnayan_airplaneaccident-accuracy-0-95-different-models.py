# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/train.csv')

df_test=pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/test.csv')
df.head()
df_test.head()
# check the null values

# we don't have any null values

df.isnull().sum()
df.dtypes
df.info()
df.describe().transpose()
df['Severity'].value_counts()
plt.figure(figsize=(12,8))

df['Severity'].value_counts().plot(kind='bar',color='red')
plt.figure(figsize=(12,8))

order=sorted(df['Severity'].unique())

chart=sns.countplot(x='Severity',data=df,order=order)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
# So Highly_Fatal_And_Damaging are in highest percentage amomgst all the damage

(df['Severity'].value_counts())/len(df['Severity'])*100
df['class']=df['Severity'].map({'Highly_Fatal_And_Damaging':0,'Significant_Damage_And_Serious_Injuries':1,'Minor_Damage_And_Injuries':2,'Significant_Damage_And_Fatalities':3})
df.head()['class']
#Finding the correlation to target variable

df.corr()['class'].sort_values(ascending=False)
plt.figure(figsize=(12,8))

sns.heatmap(df.corr(),cmap='coolwarm')
plt.figure(figsize=(12,6))

df.corr()['class'].sort_values(ascending=False).plot(kind='bar',color='red')

# safety_score is highly correlated with the Severity
plt.figure(figsize=(12,6))

sns.distplot(df['Safety_Score'],kde=False,bins=100)
# from here we can see that Highly_Fatal_And_Damaging severity having lower Safety_Score as expected.

df.groupby('Severity').mean()['Safety_Score'].sort_values(ascending=False)
plt.figure(figsize=(12,6))

df.groupby('Severity').mean()['Safety_Score'].sort_values(ascending=False).plot(kind='bar',color='pink')
df[df['Turbulence_In_gforces']==df['Turbulence_In_gforces'].max()]['Severity'].value_counts()
df[df['Turbulence_In_gforces']==df['Turbulence_In_gforces'].min()]['Severity'].value_counts()
plt.figure(figsize=(12,8))

chart=sns.boxplot(x='Severity',y='Total_Safety_Complaints',data=df)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

# Highly_Fatal_And_Damaging have highest no of complaints
df['Accident_Type_Code'].value_counts().sort_values(ascending=False)
plt.figure(figsize=(12,12))

chart=sns.boxplot(x='Severity',y='Total_Safety_Complaints',data=df,hue='Accident_Type_Code')

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
df['Violations'].value_counts().sort_values(ascending=False)

# so maximum violations are done of type 2
# to see which type of violation results in which type of severity

plt.figure(figsize=(12,12))

chart=sns.barplot(x='Severity',y='Turbulence_In_gforces',data=df,hue='Violations')

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
# here we can see that as the turbulance increases the control of the airplane decreases.

plt.figure(figsize=(12,8))

sns.scatterplot(x='Turbulence_In_gforces',y='Control_Metric',data=df)

# Highly_Fatal_And_Damaging have less control metric as compared with others 

plt.figure(figsize=(12,8))

chart=sns.barplot(x='Severity',y='Control_Metric',data=df)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
#Highly_Fatal_And_Damaging has maximum elevation

plt.figure(figsize=(12,8))

chart=sns.barplot(x='Severity',y='Max_Elevation',data=df)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
# here we can see that in adverse weather metric there is highly fatal and damaging

plt.figure(figsize=(12,8))

chart=sns.boxplot(x='Severity',y='Adverse_Weather_Metric',data=df)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
df.groupby('Severity').mean()['Cabin_Temperature']
#Cabin_Temperature doe snot have any effect on severity as all have about same temperature 

df.groupby('Severity').mean()['Cabin_Temperature'].plot(kind='bar')
# 2nd type of violations are mostly done 

sns.countplot(df['Violations'])
df.drop('Severity',axis=1,inplace=True)

df.head()
X=df.drop('class',axis=1)

y=df['class']
X.head()
y.head()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


X=StandardScaler().fit(X).transform(X.astype(float))
X
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

print('size of train set',X_train.shape,y_train.shape)

print('size of test set',X_test.shape,y_test.shape)
from sklearn.neighbors import KNeighborsClassifier


k=4

model=KNeighborsClassifier(n_neighbors=k)
model.fit(X_train,y_train)
pred=model.predict(X_test)
pred


from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report,log_loss
print(classification_report(pred,y_test))


k_val=20

mean_acc=np.zeros((k_val-1))

for n in range(1,k_val):

    model1=KNeighborsClassifier(n_neighbors=n)

    model1.fit(X_train,y_train)

    pred=model1.predict(X_test)

    mean_acc[n-1]=accuracy_score(pred,y_test)





mean_acc
p=np.arange(1,k_val)

plt.style.use('ggplot')

with plt.style.context('dark_background'):

    plt.figure(figsize=(12,8))

    plt.plot(p,mean_acc,marker='o', markerfacecolor='red', linestyle='dashed', color='green', markersize=10)

    plt.legend(('Accuracy ', '+/- 3xstd'))

    plt.ylabel('Accuracy ')

    plt.xlabel('Number of Nabors (K)')

    plt.tight_layout()

plt.show()
print("best accuracy is",mean_acc.max(),'for k value=',mean_acc.argmax())
from sklearn import svm
clf = svm.SVC(kernel='rbf')

clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
# SVM has perform better than KNN

print(classification_report(yhat,y_test))
confusion_matrix(yhat,y_test)
# import Random Forest classifier

from sklearn.ensemble import RandomForestClassifier



# instantiate the classifier 

rfc = RandomForestClassifier(random_state=0)



# fit the model

rfc.fit(X_train, y_train)



# Predict the Test set results

y_pred = rfc.predict(X_test)



print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint
model=RandomForestClassifier(n_jobs=-1)
parameters={'max_depth':[3,5,7,10,None],

           'n_estimators':[100,200,300,400,500],

           'max_features':randint(1,13),

           'criterion':['gini','entropy'],

           'bootstrap':[True,False],

           'min_samples_leaf':randint(1,5)}
def hyperparameter_tuning(model,parameters,n_of_itern,X_train,y_train):

    random_search=RandomizedSearchCV(estimator=model,

                                    param_distributions=parameters,

                                    n_jobs=-1,

                                     n_iter=n_of_itern,

                                     cv=9)

    random_search.fit(X_train,y_train)

    params=random_search.best_params_

    score=random_search.best_score_

    return params,score
final_params,final_score=hyperparameter_tuning(model,parameters,40,X_train,y_train)
#this is our final best parameters for random forest classifier

final_params
# final accuracy with tuned parameters

final_score
# import Random Forest classifier

from sklearn.ensemble import RandomForestClassifier



# instantiate the classifier 

rfc = RandomForestClassifier(n_estimators=300,

                             criterion='entropy',

                             max_depth=None,

                             max_features=7,

                             min_samples_leaf=2,

                             bootstrap=False

                             )

                            



# fit the model

rfc.fit(X_train, y_train)



# Predict the Test set results

y_pred = rfc.predict(X_test)



print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
print(classification_report(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()

log_reg.fit(X_train,y_train)

log_predict=log_reg.predict(X_test)
print(classification_report(log_predict,y_test))
print(confusion_matrix(log_predict,y_test))
data_dict={'Algorithms':['KNN', 'SVM', 'RandomForest', 'logistic regression'],'accuracy':[0.676,0.86,0.91,0.61],'accuracy_after_hyperparameter tuning':['-',0.89,0.95,'-']}

accuracy_df=pd.DataFrame(data_dict)
accuracy_df.set_index('Algorithms',inplace=True)
accuracy_df
prediction_test=rfc.predict(X_test)

prediction_test