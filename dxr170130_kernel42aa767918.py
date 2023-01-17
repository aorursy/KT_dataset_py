import math

import numpy as np

import pandas as pd

from datetime import datetime

from scipy import stats

from sklearn.preprocessing import MultiLabelBinarizer



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

plt.style.use('seaborn-whitegrid')



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC,LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
df_receivable = pd.read_csv("../input/dataset/Data.csv")

#df.dropna(inplace=True)

df_receivable.info()
df_receivable.dropna(inplace=True)

df_receivable.head(5)

df_receivable.info()
df_receivable['InvoiceDate']= pd.to_datetime(df_receivable.InvoiceDate)
df_receivable['DueDate']= pd.to_datetime(df_receivable.DueDate)
df_receivable['Late'] = df_receivable['DaysLate'].apply(lambda x: 1 if x >0 else 0)
df_receivable['countlate']=df_receivable.Late.eq(1).groupby(df_receivable.customerID).apply(

    lambda x : x.cumsum().shift().fillna(0)).astype(int)
df_receivable.info()
df_receivable.describe()
temp = pd.DataFrame(df_receivable.groupby(['countryCode'], axis=0, as_index=False)['DaysLate'].mean())

plt.figure(figsize=(10,6))

sns.barplot(x="countryCode", y="DaysLate",data=temp,linewidth=2.5, facecolor=(1, 1, 1, 0),

                 errcolor=".4", edgecolor="red")
df_receivable.describe(include=np.object)
print(pd.crosstab(index=df_receivable["Member_Category"], columns="count"))
print(pd.crosstab(index=df_receivable["countryCode"], columns="count"))
print(pd.crosstab(index=df_receivable["Late"], columns="count"))
df_receivable.head(3)
customer_late =pd.crosstab(index=df_receivable["customerID"], columns=df_receivable['Late'])

customer_late.sort_values(by=[1], ascending = False)
df1 = df_receivable[df_receivable['DaysLate']>0].copy()
df2 = pd.DataFrame(df1.groupby(['customerID'], axis=0, as_index=False)['DaysLate'].count())
df2.columns = (['customerID','repeatCust'])
df3 = pd.merge(df_receivable, df2, how='left', on='customerID')
df3['repeatCust'].fillna(0, inplace=True)
df_receivable = df3
temp = pd.DataFrame(df_receivable.groupby(['repeatCust'], axis=0, as_index=False)['DaysLate'].mean())

plt.figure(figsize=(14,7))

sns.barplot(x="repeatCust", y="DaysLate",data=temp,color='olive')
def func_IA (x):

    if x>60: return "b. more than 60"

    else: return "a. less than 60"

df_receivable['InvoiceAmount_bin'] = df_receivable['InvoiceAmount'].apply(func_IA)
df_receivable.head(3)
temp = pd.DataFrame(df_receivable.groupby(['InvoiceAmount_bin'], axis=0, as_index=False)['DaysLate'].mean())
plt.figure(figsize=(10,6))

sns.barplot(x="InvoiceAmount_bin", y="DaysLate",data=temp,color='purple')
df_receivable['Disputed'] = df_receivable['Disputed'].map({'No':0,'Yes':1})
df_receivable['InvoiceQuarter']= pd.to_datetime(df_receivable['InvoiceDate']).dt.quarter
print(pd.crosstab(index=df_receivable["Member_Category"], columns="count"))
df_receivable['Member_Category'] = pd.Categorical(df_receivable['Member_Category'])
dfDummies = pd.get_dummies(df_receivable['Member_Category'], prefix = 'cat')
dfDummies
df_receivable = pd.concat([df_receivable, dfDummies], axis=1)
df_receivable.head(3)
plt.figure(figsize=(10,8))



ax = sns.countplot(df_receivable['countryCode'],hue=df_receivable['Late'],palette="YlGn")
plt.figure(figsize=(10,8))

sns.countplot(df_receivable['InvoiceQuarter'],hue=df_receivable['Late'],palette='bright')
plt.figure(figsize=(10,8))

sns.countplot(df_receivable['Member_Category'],hue=df_receivable['Late'],palette='YlOrRd')
plt.figure(figsize=(10,8))

sns.countplot(df_receivable['Disputed'],hue=df_receivable['Late'],palette='BuGn')
plt.figure(figsize=(8,8))

plt.figure(1)

sns.distplot(df_receivable['InvoiceAmount'],color='green')

plt.figure(figsize=(8,8))

plt.figure(2)

sns.distplot(df_receivable['DaysToSettle'],color='blue')
labels = df_receivable['customerID'].astype('category').cat.categories.tolist()
replace_map_comp = {'customerID' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
df_receivable.replace(replace_map_comp, inplace=True)
df_receivable.head(3)
df_receivable.head(3)
corremat = df_receivable.corr()

plt.figure(figsize=(10,10))

g= sns.heatmap(df_receivable.corr(),annot=True,cmap='viridis',linewidths=.5)
corremat
corremat.columns
cat_feats = ['InvoiceAmount_bin']

final_data = pd.get_dummies(df_receivable,columns=cat_feats,drop_first=True)
final_data.head(3)
features=['countryCode', 'customerID', 'InvoiceAmount',

       'Disputed','repeatCust','DaysLate', 'DaysToSettle',

       'countlate']
features
X = final_data[features]

y = final_data['Late']
y.head(5)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
# Logisitic Regression

logreg = LogisticRegression()

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)
#Checking the accuracy

logistic_accuracy = round(logreg.score(X_train,y_train)*100,2)

print(round(logistic_accuracy,2),'%')
#Decesion Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train,y_train)



y_pred = decision_tree.predict(X_test)
decision_tree_accuracy = round(decision_tree.score(X_train,y_train) * 100,2)

print(round(decision_tree_accuracy,2),'%')
# Perceptron

perceptron = Perceptron(max_iter=5)

perceptron.fit(X_train,y_train)



y_pred = perceptron.predict(X_test)
perceptron_accuracy = round(perceptron.score(X_train,y_train)* 100,2)

print(round(perceptron_accuracy,2),'%')
# Randon Forest

rand_forest = RandomForestClassifier(n_estimators=100)

rand_forest.fit(X_train,y_train)



y_pred = rand_forest.predict(X_test)
rand_forest_accuracy = round(rand_forest.score(X_train,y_train)*100,2)

print(round(rand_forest_accuracy,2),'%')
# Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train,y_train)



y_pred = gaussian.predict(X_test)
gaussian_accuracy = round(gaussian.score(X_train,y_train)*100,2)

print(round(gaussian_accuracy,2),'%')
#KNN

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)



y_pred = knn.predict(X_test)
knn_accuracy = round(knn.score(X_train,y_train)*100,2)

print(round(knn_accuracy,2),'%')
#LinearSVC

linear_svc = LinearSVC()

linear_svc.fit(X_train,y_train)



y_pred = linear_svc.predict(X_test)
linear_svc_accuracy = round(linear_svc.score(X_train,y_train)*100,2)

print(round(linear_svc_accuracy,2),'%')
#SVC

svc = SVC(gamma='auto')

svc.fit(X_train,y_train)



y_pred = svc.predict(X_test)
svc_accuracy = round(svc.score(X_train,y_train)*100,2)

print(round(svc_accuracy,2),'%')
model_evaluation = pd.DataFrame({

    'Model':['LogisticRegression','DecisionTreeClassifier','Perceptron','RandomForestClassifier',

             

             'GaussianNB','KNeighborsClassifier','LinearSVC','SVC'],

    

    'Score':[logistic_accuracy,decision_tree_accuracy,perceptron_accuracy,rand_forest_accuracy,

             gaussian_accuracy,knn_accuracy,linear_svc_accuracy,svc_accuracy,]})

model_evaluation.sort_values(by='Score',ascending = False)
from sklearn.model_selection import cross_val_score

rand_forest = RandomForestClassifier(n_estimators=100)

score = cross_val_score(rand_forest,X_train,y_train,cv = 10,scoring='accuracy')
print('Score',score)

print('Mean',round(score.mean()*100),2)

print('Satandered Deviation',score.std())
from sklearn.model_selection import cross_val_score



accuracy =cross_val_score(estimator=rand_forest,X=X_train,y=y_train,cv= 10)



accuracy.mean()
accuracy.std()
rand_forest.fit(X_train,y_train)

importance = pd.DataFrame({

    'Feature':X_train.columns,'importance':np.round(rand_forest.feature_importances_,3)})

importance = importance.sort_values('importance',ascending=False).set_index('Feature')
importance
importance.plot.bar()
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

prediction = cross_val_predict(rand_forest,X_train,y_train,cv = 3)

confusion_matrix(y_train,prediction)
from sklearn.metrics import precision_score,recall_score

print('Percision : ',precision_score(y_train,prediction))

print('Recall :',recall_score(y_train,prediction))
from sklearn.metrics import f1_score

f1_score(y_train,prediction)
y_scores = rand_forest.predict_proba(X_train)
y_scores = y_scores[:,1]
from sklearn.metrics import roc_auc_score

r_a_score = roc_auc_score(y_train,y_scores)

print('ROC-AUC-Score: ',r_a_score)
sns.set_context("talk")

sns.set_style("whitegrid", {'grid.color': '.92'})



def reformat_large_tick_values(tick_val, pos):

    """

    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the decimal).

    """

    if tick_val >= 1000000000:

        val = round(tick_val/1000000000, 1)

        new_tick_format = '{:}B'.format(val)

    elif tick_val >= 1000000:

        val = round(tick_val/1000000, 1)

        new_tick_format = '{:}M'.format(val)

    elif tick_val >= 1000:

        val = round(tick_val/1000, 1)

        new_tick_format = '{:}K'.format(val)

    elif tick_val < 1000:

        new_tick_format = round(tick_val, 1)

    else:

        new_tick_format = tick_val



    # make new_tick_format into a string value

    new_tick_format = str(new_tick_format)



    # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed

    index_of_decimal = new_tick_format.find(".")



    if index_of_decimal != -1:

        value_after_decimal = new_tick_format[index_of_decimal+1]

        if value_after_decimal == "0":

            # remove the 0 after the decimal point since it's not needed

            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal+2:]



    return new_tick_format
plt.figure(figsize=(10, 8))

sns.scatterplot(x=y_pred, y=y_test)