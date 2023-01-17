import os



from sklearn.metrics import classification_report





def classifcation_report_train_test(y_train, y_train_pred, y_test, y_test_pred):



    print('''

            =========================================

               CLASSIFICATION REPORT FOR TRAIN DATA

            =========================================

            ''')

    print(classification_report(y_train, y_train_pred))



    print('''

            =========================================

               CLASSIFICATION REPORT FOR TEST DATA

            =========================================

            ''')

    print(classification_report(y_test, y_test_pred))
import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')
# Load the Bank.csv

data=pd.read_csv("../input/bank-personal-loan-campaign/Bank.csv",skiprows=4,skipfooter=3,na_values=['Null'])
# Check the head and tail of data

data.head(6)
data.tail(6)
data=data.drop(columns=['ID','ZIP Code'],axis=1)

data.head()
#converting the column names to Lower case

data.columns = map(str.lower, data.columns)

data.columns
#replace spaces in column names with _

data.columns = [x.replace(' ', '_') for x in data.columns]

data.columns
data.head()
#check for null values

data.apply(lambda x : sum(x.isnull()))
#find unique levels in each column\

data.apply(lambda x: len(x.unique()))


def myfunc(x):

    return len(x.unique())



data.apply(myfunc)
data.apply(lambda x: len(x.unique())).sort_values()
# check the statistics of Dataframe

data.describe()
data.describe().T
#Check is there any customers with negative experience, if yes remove those rows from data 

#data[data['experience']<0].count()

print('People Having Negative Experience:',data[data['experience'] < 0]['experience'].count())

print('People Having Positive Experience:',data[data['experience'] > 0]['experience'].count())
#Check is there any customers with negative experience, if yes remove those rows from data 

data.drop(data[data['experience']<0].index,inplace=True)
data.experience.value_counts()
df = data.copy()

df.head()
for col in df.columns:

    if len(df[col].unique()) < 10:

        print(col, df[col].unique())
# Check the no of levels of all categorical columns



num_col=['age','experience','income',"ccavg",'mortgage']

cat_col=df.columns.difference(num_col)

cat_col
df[cat_col] = df[cat_col].apply(lambda x: x.astype('category'))

df[num_col] = df[num_col].apply(lambda x: x.astype('float'))

df.dtypes
df[cat_col]
for i in df[cat_col]:

    #print([i],':',df[cat_col[i]].unique())

    print(i,':',df[i].nunique())

    

num_data = data.loc[:,num_col]

cat_data = data.loc[:,cat_col]

num_data.head()
# check is there any NA values present, and if present impute them 

cat_data.isna().sum()
num_data.isna().sum()
#num_data.fillna()

num_data.fillna(num_data['age'].mean(), inplace = True) 
num_data.isna().sum()
full_data = pd.concat([num_data,cat_data],axis=1)

full_data.head()
# check the Personal loan statistics with plot which is suited

full_data['personal_loan'].value_counts().plot(kind='bar')
# Influence of income and education on personal loan and give the observations from the plot

sns.boxplot(x='education',y='income',hue='personal_loan',data=full_data)
# Influence of Credict card usage and Personal Loan  and give observations from the plot

sns.boxplot(y='ccavg',x='personal_loan',data=full_data)
# Influence of education level on persoanl loan and give the insights

full_data.groupby(['education','personal_loan']).size().plot(kind='bar')

plt.xticks(rotation=30)                                                      
# Influence of family size on persoanl loan and suggest the insights

sns.countplot(x='family',data=full_data,hue='personal_loan',palette='Set3')
# Influence of deposit account on personal loan and give the insights

sns.countplot(x='cd_account',data=full_data,hue='personal_loan')
# Influence of Security account on personal loan and give the insights

sns.countplot(x="securities_account", data=full_data,hue="personal_loan")
# Influence of Credit card on Persoanl Loan and give insights

sns.countplot(x="creditcard", data=full_data,hue="personal_loan")
#median

print('Non-Loan customers: ',full_data[full_data.personal_loan == 0]['ccavg'].median()*1000)

print('Loan customers    : ', full_data[full_data.personal_loan == 1]['ccavg'].median()*1000)
#mean

print('Non-Loan customers: ',full_data[full_data.personal_loan == 0]['ccavg'].mean()*1000)

print('Loan customers    : ', full_data[full_data.personal_loan == 1]['ccavg'].mean()*1000)
#family income personalloan realtionship

sns.boxplot(x='family',y='income',data=full_data,hue='personal_loan')
# Correlation with heat map

import matplotlib.pyplot as plt

import seaborn as sns

corr = data.corr()

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
# Correlation with heat map

import matplotlib.pyplot as plt

import seaborn as sns

corr = data.corr()

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})

plt.figure(figsize=(13,7))

# create a mask so we only see the correlation values once

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask, 1)] = True

a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')

rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)

roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
#mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask, 1)] = True
# creating` dummies for columns which have more than two levels

for col in cat_data.columns:

    if len(cat_data[col].unique()) > 2:

        print(col, cat_data[col].unique())
df = pd.get_dummies(data=full_data, columns=['family', 'education'], drop_first=True)

df.head()
# Train test split

## Split the data into X_train, X_test, y_train, y_test with test_size = 0.20 using sklearn

X = df.copy().drop("personal_loan",axis=1)

y = df["personal_loan"]



## Split the data into X_train, X_test, y_train, y_test with test_size = 0.20 using sklearn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)



## Print the shape of X_train, X_test, y_train, y_test

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
X_train.head()
X_train.isna().sum()
# after split check the proportion of target levels - train

print(y_train.value_counts(normalize=True)) 
# after split check the proportion of target levels - test

print(y_test.value_counts(normalize=True))
#you can also plot this 



import matplotlib.pyplot as plt

y_test.value_counts(normalize=True).plot(kind='bar')
# Implement ***SVM CLASSIFIER*** with grid search 
# Predict
# Apply the follwing models and show a data frame with the all the model performances

#    1. Logistic Regression - We haven't given the code, you need to explore!

#    2. Decision trees

#    3. K-nn 

    

# Please ensure you experiment with multiple hyper parameters for the each of the above algorithms

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
lrc = LogisticRegression()



lrc.fit(X_train,y_train)



y_train_pred_lrc_be = lrc.predict(X_train)

y_test_pred_lrc_be = lrc.predict(X_test)
svc = SVC()



svc.fit(X_train,y_train)



y_train_pred_svc_be = svc.predict(X_train)

y_test_pred_svc_be = svc.predict(X_test)
dtc = DecisionTreeClassifier()



dtc.fit(X_train,y_train)



y_train_pred_dt_be = dtc.predict(X_train)

y_test_pred_dt_be = dtc.predict(X_test)
knn = KNeighborsClassifier()



knn.fit(X_train,y_train)



y_train_pred_knn_be = dtc.predict(X_train)

y_test_pred_knn_be = dtc.predict(X_test)

#classifcation_report_train_test(y_train,y_train_pred_knn_be,y_test, y_test_pred_knn_be)
naive_model = GaussianNB()

naive_model.fit(X_train, y_train)



y_train_pred_nv_be = naive_model.predict(X_train)

y_test_pred_nv_be =naive_model.predict(X_test)
from sklearn.metrics import recall_score



print("Recall of DecisionTrees:",recall_score(y_test, y_test_pred_dt_be))

print("Recall of LogisticRegression:",recall_score(y_test, y_test_pred_lrc_be))

print("Recall of SupportVectorMachines:",recall_score(y_test, y_test_pred_svc_be))

print("Recall of KNearestNeighbours:",recall_score(y_test, y_test_pred_knn_be))

print("Recall of naiibeys:",recall_score(y_test, y_test_pred_nv_be))
from mlxtend.plotting import plot_learning_curves

plot_learning_curves(X_train, y_train, X_test, y_test, lrc,scoring='recall')

plt.show()
from mlxtend.plotting import plot_learning_curves

plot_learning_curves(X_train, y_train, X_test, y_test, svc,scoring='recall')

plt.show()
from mlxtend.plotting import plot_learning_curves

plot_learning_curves(X_train, y_train, X_test, y_test, dtc,scoring='recall')

plt.show()
from mlxtend.plotting import plot_learning_curves

plot_learning_curves(X_train, y_train, X_test, y_test, knn,scoring='recall')

plt.show()
from mlxtend.plotting import plot_learning_curves

plot_learning_curves(X_train, y_train, X_test, y_test, naive_model,scoring='recall')

plt.show()
#Scale the numeric attributes



from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

scaler.fit(X_train[num_col])



X_train[num_col] = scaler.transform(X_train[num_col])

X_test[num_col] = scaler.transform(X_test[num_col])

X_train.head()
#Build svc Classifier

from sklearn.svm import SVC
## Create an SVC object and print it to see the default arguments



svc = SVC(class_weight='balanced')

svc
svc.fit(X_train,y_train)
y_train_pred_svc = svc.predict(X_train)

y_test_pred_svc = svc.predict(X_test)
#from sklearn.metrics import classifcation_report_train_test

classifcation_report_train_test(y_train, y_train_pred_svc, y_test, y_test_pred_svc)
!pip install mlxtend
## Use Grid Search for parameter tuning



from sklearn.model_selection import GridSearchCV



svc_grid = SVC(class_weight='balanced')

 



param_grid = {



'C': [0.001, 0.01, 0.1, 1, 10],

'gamma': [0.001, 0.01, 0.1, 1], 

'kernel':['linear', 'rbf']}



 

svc_cv_grid = GridSearchCV(estimator = svc_grid, param_grid = param_grid, cv = 10)
svc_cv_grid.fit(X_train, y_train)
# Display the best estimator

svc_cv_grid.best_estimator_
#predicting using best_estimator

y_train_pred_svc_best = svc_cv_grid.best_estimator_.predict(X_train)

y_test_pred_svc_best = svc_cv_grid.best_estimator_.predict(X_test)
#classification reprot by using base function created on first cell

classifcation_report_train_test(y_train, y_train_pred_svc_best, y_test, y_test_pred_svc_best)
## Use Grid Search for parameter tuning



from sklearn.model_selection import GridSearchCV



svc_grid = SVC(class_weight='balanced')

 



param_grid = {



'C': [0.6,0.7,0.8,0.9,1,1.5],

'gamma': [1,2,3,4,5], 

'kernel':['linear', 'rbf']}



 

svc_cv_grid2 = GridSearchCV(estimator = svc_grid, param_grid = param_grid, cv = 10)
svc_cv_grid2.fit(X_train, y_train)
svc_cv_grid2.best_estimator_
y_train_pred_svc_best2 = svc_cv_grid2.best_estimator_.predict(X_train)

y_test_pred_svc_best2 = svc_cv_grid2.best_estimator_.predict(X_test)
classifcation_report_train_test(y_train, y_train_pred_svc_best2, y_test, y_test_pred_svc_best2)
## Use Grid Search for parameter tuning



from sklearn.model_selection import GridSearchCV



svc_grid = SVC(class_weight='balanced')

 



param_grid = {



'C': [0.6,0.7,0.8,0.9,1,1.5],

'gamma': [0.6,0.7,0.8],  

'kernel':['linear', 'rbf']}



 

svc_cv_grid3 = GridSearchCV(estimator = svc_grid, param_grid = param_grid, cv = 10)
svc_cv_grid3.fit(X_train, y_train)
svc_cv_grid3.best_estimator_
y_train_pred_svc_best1 = svc_cv_grid3.best_estimator_.predict(X_train)

y_test_pred_svc_best1 = svc_cv_grid3.best_estimator_.predict(X_test)
classifcation_report_train_test(y_train, y_train_pred_svc_best1, y_test, y_test_pred_svc_best1)
## Use Grid Search for parameter tuning



from sklearn.model_selection import GridSearchCV



svc_grid = SVC(class_weight='balanced')

 



param_grid = {



'C': [0.9,1,1.2,1.3,1.4],

'gamma': [0.6,0.7,0.8], 

'kernel':['linear', 'rbf']}



 

svc_cv_grid4 = GridSearchCV(estimator = svc_grid, param_grid = param_grid, cv = 10)
svc_cv_grid4.fit(X_train, y_train)
svc_cv_grid4.best_estimator_
svc_cv_grid4.best_params_
y_train_pred_svc_best2 = svc_cv_grid4.best_estimator_.predict(X_train)

y_test_pred_svc_best2 = svc_cv_grid4.best_estimator_.predict(X_test)
classifcation_report_train_test(y_train, y_train_pred_svc_best2, y_test, y_test_pred_svc_best2)
from mlxtend.plotting import plot_learning_curves

plot_learning_curves(X_train, y_train, X_test, y_test, svc_cv_grid4.best_estimator_,scoring='recall')

knn1 = KNeighborsClassifier(n_neighbors= 3 , weights = 'uniform', metric='euclidean')

knn1.fit(X_train, y_train)    

predicted = knn1.predict(X_test)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, predicted)

print(acc)
knn1 = KNeighborsClassifier(n_neighbors= 5 , weights = 'uniform', metric='euclidean')

knn1.fit(X_train, y_train)    

predicted = knn1.predict(X_test)

from sklearn.metrics import accuracy_score

acc = recall_score(y_test, predicted)

print(acc)
knn2 = KNeighborsClassifier(n_neighbors= 7, weights = 'uniform', metric='euclidean')

knn2.fit(X_train, y_train)    

predicted2 = knn2.predict(X_test)

from sklearn.metrics import accuracy_score

acc = recall_score(y_test, predicted2)

print(acc)
grid_params ={

    'n_neighbors':[3,5,7,9],

    'weights':['uniform','distance'],

    'metric':['euclidean','manhattan','minkowski']

}

gs_results=GridSearchCV(KNeighborsClassifier(),grid_params,verbose=1,cv=5,n_jobs=-1)

gs_results.fit(X_train,y_train)
gs_results.best_estimator_
gs_results.best_params_
gs_results.best_score_
knn_predict_train=gs_results.best_estimator_.predict(X_train)

knn_predict_test=gs_results.best_estimator_.predict(X_test)
classifcation_report_train_test(y_train, knn_predict_train, y_test, knn_predict_test)
error_rate = []



# Will take some time

for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')

plt.grid()

plt.show()
from sklearn.tree import DecisionTreeClassifier
depths = np.arange(1, 21)

depths
#num_leafs = [1, 5, 10, 20, 50, 100]

#criterion = ['gini', 'entropy']
param_grid = { 'criterion':['gini','entropy'],'max_depth': depths}

dtree_model=DecisionTreeClassifier()
gs = GridSearchCV(estimator=dtree_model, param_grid=param_grid, cv=10)
gs = gs.fit(X_train, y_train)
gs.best_estimator_
gs.best_params_
dt_predict_train=gs.best_estimator_.predict(X_train)

dt_predict_test=gs.best_estimator_.predict(X_test)
classifcation_report_train_test(y_train, dt_predict_train, y_test, dt_predict_test)
from mlxtend.plotting import plot_learning_curves

plot_learning_curves(X_train, y_train, X_test, y_test, gs.best_estimator_,scoring='recall')

plt.show()
from sklearn import tree

dt1=tree.DecisionTreeClassifier(max_depth=6)

dt1.fit(X_train,y_train)
dt1.get_depth
dt1.get_params
#dt1.predict(X_test)

dt1_predict_train=dt1.predict(X_train)

dt1_predict_test=dt1.predict(X_test)
classifcation_report_train_test(y_train, dt1_predict_train, y_test, dt1_predict_test)
tree.plot_tree(dt1.fit(X_train, y_train)) 
!pip install pydotplus
dt1.feature_importances_
f_imp = pd.Series(dt1.feature_importances_, index = X_train.columns)
## Sort importances  

f_imp_order= f_imp.nlargest(n=10)

f_imp_order
## Plot Importance

%matplotlib inline

f_imp_order.plot(kind='barh')

plt.show()
# Grid search cross validation

from sklearn.linear_model import LogisticRegression

grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge

logreg=LogisticRegression()

logreg_cv=GridSearchCV(logreg,grid,cv=10)

logreg_cv.fit(X_train,y_train)



print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)

print("accuracy :",logreg_cv.best_score_)
logreg2=LogisticRegression(C=10,penalty="l2")

logreg2.fit(X_train,y_train)

print("score",logreg2.score(X_test,y_test))

reg_predict_train=logreg2.predict(X_train)

reg_predict_test=logreg2.predict(X_test)

classifcation_report_train_test(y_train, reg_predict_train, y_test, reg_predict_test)
# Create first pipeline for base without reducing features.



#pipe = Pipeline([('classifier' , RandomForestClassifier())])

# pipe = Pipeline([('classifier', RandomForestClassifier())])



# Create param grid.

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(random_state=42)

param_grid = { 

    'n_estimators': [240,245,250],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [12,13,14],

    'criterion' :['gini', 'entropy']

}

# Create grid search object

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)

CV_rfc.fit(X_train, y_train)

CV_rfc.best_estimator_



rf_predict_train=CV_rfc.best_estimator_.predict(X_train)

rf_predict_test=CV_rfc.best_estimator_.predict(X_test)

classifcation_report_train_test(y_train, rf_predict_train, y_test, rf_predict_test)
from mlxtend.plotting import plot_learning_curves

plot_learning_curves(X_train, y_train, X_test, y_test, CV_rfc.best_estimator_,scoring='recall')

plt.show()
CV_rfc.best_estimator_.feature_importances_
feat_importances = pd.Series(CV_rfc.best_estimator_.feature_importances_, index = X_train.columns)
feat_importances_ordered = feat_importances.nlargest(n=10)

feat_importances_ordered
## Plot Importance

%matplotlib inline

feat_importances_ordered.plot(kind='barh')

plt.show()
from sklearn.metrics import recall_score



scores = pd.DataFrame(columns=['Model','Train_Recall','Test_Recall'])



def get_metrics(train_actual,train_predicted,test_actual,test_predicted,model_description,dataframe):

    

    train_recall   = recall_score(train_actual,train_predicted)

    test_recall   = recall_score(test_actual,test_predicted)

    dataframe = dataframe.append(pd.Series([model_description,train_recall,

                                            test_recall],

                                           index=scores.columns ), ignore_index=True)

    return(dataframe)
scores = get_metrics(y_train,y_train_pred_dt_be,y_test,y_test_pred_dt_be,'DecisionTrees basic model',scores)

scores = get_metrics(y_train,y_train_pred_lrc_be,y_test,y_test_pred_lrc_be,'LogisticRegression basic model',scores)

scores = get_metrics(y_train, y_train_pred_svc_be,y_test, y_test_pred_svc_be,'SupportVectorMachines basic model',scores)

scores = get_metrics(y_train, y_train_pred_knn_be,y_test, y_test_pred_knn_be,'KNearestNeighbours basic model',scores)

scores = get_metrics(y_train, y_train_pred_nv_be,y_test, y_test_pred_nv_be,'naiibeys basic model',scores)

scores = get_metrics(y_train,dt_predict_train,y_test,dt_predict_test,'Decision Tree with GridSearchCV()',scores)

scores = get_metrics(y_train,reg_predict_train,y_test,reg_predict_test,'logistic regression with GridSearchCV()',scores)

scores = get_metrics(y_train,y_train_pred_svc_best2,y_test,y_test_pred_svc_best2,'SVC using GridSearchCV()',scores)

scores = get_metrics(y_train,knn_predict_train,y_test,knn_predict_test,'KNN using GridSearchCV(),Where k=5',scores)

scores = get_metrics(y_train,rf_predict_train,y_test,rf_predict_test,'random forest using GridSearchCV',scores)



scores
scores.insert(3, "Best Tuning Parametrs",['','','','','','{criterion: gini, max_depth: 6}', '{C: 10.0, penalty: l2}', '{C: 1, gamma: 0.7, kernel: rbf}', '{metric: euclidean, n_neighbors: 3, weights: distance}','max_depth=13,max_features=auto,n_estimators=240,criterion=entropy'], True)

scores

#df.insert(2, "Age", [21, 23, 24, 21], True)