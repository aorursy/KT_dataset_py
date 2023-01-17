import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE



from sklearn.metrics import roc_auc_score,confusion_matrix,classification_report,make_scorer

from sklearn.model_selection import cross_validate,GridSearchCV,train_test_split



from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier,VotingClassifier,RandomForestClassifier,BaggingClassifier
!ls ../input/
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
for i in train_data.columns:

    print(i)

    print(train_data[i].value_counts())
train_data['Class'].value_counts()
numerical = ['Age','IC','OC','Timely Income','Gain','Loss','Stock','Weight','Weaks']

categorical = ['Worker Class','Schooling','Enrolled','Married_Life','MIC','MOC','Cast','Hispanic','Sex','MLU','Reason','Full/Part','Tax Status','Area','State','Detailed','Summary','MSA','REG','MOVE','Live','PREV','NOP','Teen','COB FATHER','COB MOTHER','COB SELF','Citizen','Own/Self','Fill','Vet_Benefits','WorkingPeriod']
train_data.groupby(['Summary','Class']).count()
len(numerical+categorical)
d= {}

for i in train_data.columns:

    if('?' in train_data[i].unique()):

        #print ("here")

        d[i]=train_data[i].value_counts()['?']/sum(train_data[i].value_counts())
d
for i in test_data.columns:

    if('?' in test_data[i].unique()):

        print(i,test_data[i].value_counts()['?']/sum(test_data[i].value_counts()))
toadd = ['Worker Class','Enrolled','Teen','Fill']
# train_data.groupby(['Fill','Class']).count()
#Drop columns with more than 50% missing values.

#Fill columns with less than 50% missing values

todrop = []

for col in toadd:

    train_data[col].replace('?','NaN')

for i in d.keys():

    if(d[i]>=0.5):

        if(i not in toadd):

            todrop.append(i)

            if(i in numerical):

                numerical.remove(i)

            if(i in categorical):

                categorical.remove(i)

#print(todrop)



train_data.replace('?',np.nan,inplace=True)

test_data.replace('?',np.nan,inplace =True)



#print(categorical)           

dtrain_data = train_data.drop(todrop,axis=1)

dtest_data = test_data.drop(todrop,axis=1)



dtrain_data[numerical] = dtrain_data[numerical].fillna(dtrain_data[numerical].median())

dtest_data[numerical] = dtest_data[numerical].fillna(dtrain_data[numerical].median())



dtrain_data[categorical] = dtrain_data[categorical].apply(lambda x:x.fillna(x.value_counts().index[0]))



for i in categorical:

    dtest_data[i] = dtest_data[i].fillna(dtrain_data[i].value_counts().index[0])
dtrain_data.head()
for i in categorical:

    if list(sorted(map(str,list(train_data[i].unique()))))!=list(sorted(map(str,list(test_data[i].unique())))):

        print(i)

        print(sorted(map(str,list(train_data[i].unique()))))

        print(sorted(map(str,list(test_data[i].unique()))))

        
#Label Encoding and Feature Scaling

sc_train = dtrain_data.copy()

sc_test = dtest_data.copy()



ss = StandardScaler()

sc_train[numerical]= ss.fit_transform(sc_train[numerical])

sc_test[numerical]= ss.transform(sc_test[numerical])





le_train = sc_train.copy()

le_test =sc_test.copy()



le = LabelEncoder()

#print (categorical)

for i in categorical:

    #print (i)

    if(i=='Detailed'):

        #Different categores in both

        le_train[i] = le_train[i].apply(lambda x:int(x[1:]))

        le_test[i] = le_test[i].apply(lambda x:int(x[1:]))

    else:

        le_train[i] = le.fit_transform(le_train[i])

        le_test[i] = le.transform(le_test[i])
le_train.head()
#Check for duplicates

dup = dict(le_train.drop('ID',axis=1).duplicated())

le_train_dd = le_train.copy()

for i in dup.keys():

    if(dup[i]==True):

        le_train_dd.drop(i,inplace=True)
le_train_dd.shape
f, ax = plt.subplots(figsize=(30,20))

corr = le_train_dd.corr()

plt.title('Fig 1. Heatmap of Correlation Matrix')

sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax, annot = True)
# threshold = 1.1

# dropcol = (corr['Class']<threshold) & (corr['Class']>-threshold)

# todrop = list(dropcol[dropcol==True].keys())

# le_train_dd.drop(todrop,axis=1,inplace=True)
le_train_dd.head()
model=PCA(n_components=2)

model_data=model.fit(le_train_dd.drop(['ID','Class'],axis=1)).transform(le_train_dd.drop(['ID','Class'],axis=1))
plt.figure(figsize=(8,6))

plt.xlabel('X')

plt.ylabel('Y')

plt.title('Fig 2. PCA Representation of Given Classes')

plt.scatter(model_data[:,0],model_data[:,1],c=le_train_dd['Class'],cmap = plt.get_cmap('rainbow_r'))
cols = list(le_train_dd.columns)

cols.remove('ID')

cols.remove('Class')

X = le_train_dd[cols]

y = le_train_dd['Class']
from imblearn.ensemble import BalanceCascade

bc = BalanceCascade(random_state=42)

X_res,y_res = bc.fit_sample(X,y)
best_model=[]

for i in range(6): 

    X_train,X_test,y_train,y_test = train_test_split(X_res[i],y_res[i],test_size=0.3,random_state=42)

    best_roc = 0 

    for n in [100,250,500]: 

        clf = AdaBoostClassifier(n_estimators=n,random_state=42)

        clf.fit(X_train,y_train)

        optimized_predictions = clf.predict(X_test) 

        auc_op = roc_auc_score(y_test, optimized_predictions)*100 

        if auc_op>best_roc: 

            best_clf=clf 

            best_roc=auc_op 

    best_model.append(best_clf)



for i in range(6,10): 

    X_train,X_test,y_train,y_test = train_test_split(X_res[i],y_res[i],test_size=0.3,random_state=42)

    best_roc = 0 

    for n in [100,250,500]: 

        for d in [18,20,22]: 

            clf = RandomForestClassifier(n_estimators=n,max_depth=d,random_state=42)

            clf.fit(X_train,y_train)

            optimized_predictions = clf.predict(X_test)

            auc_op = roc_auc_score(y_test, optimized_predictions)*100 

            if auc_op>best_roc:

                best_clf=clf

                best_roc=auc_op

    best_model.append(best_clf)
for i in best_model:

    print(i)
adb_importances = []

rf_importances = []

for i in range(6):

    print (i)

    X_train,X_test,y_train,y_test = train_test_split(X_res[i],y_res[i],test_size=0.3,random_state=42)



    print("="*50)

    best_model[i].fit(X_train,y_train)

    y_pred = best_model[i].predict(X_test)

    print("ROC_AUC",roc_auc_score(y_test,y_pred))

    adb_importances.append(best_model[i].feature_importances_)

for i in range(6,10):

    print (i)

    X_train,X_test,y_train,y_test = train_test_split(X_res[i],y_res[i],test_size=0.3,random_state=42)



    print("="*50)

    best_model[i].fit(X_train,y_train)

    y_pred = best_model[i].predict(X_test)

    print("ROC_AUC",roc_auc_score(y_test,y_pred))

    rf_importances.append(best_model[i].feature_importances_)
importances = adb_importances + rf_importances

importance = np.mean(importances,axis=0)
dictionary = dict((key,value) for (key,value)in zip(X.columns,importance))
X.columns
sorted_importances = (sorted(dictionary,key=lambda k:dictionary[k],reverse=True))
print(len(sorted_importances))
print(sorted_importances[:20])
cols = sorted_importances[:20]

if('ID' in cols):

    cols.remove('ID')

X = le_train_dd[cols]

y = le_train_dd['Class']
np.random.seed(42)
y.value_counts()
from imblearn.ensemble import BalanceCascade

bc = BalanceCascade(random_state=42)

X_res,y_res = bc.fit_sample(X,y)
X_res.shape
X_N = X_res[0]

y_N = y_res[0]
X_train,X_val,y_train,y_val = train_test_split(X_N,y_N,test_size=0.3,random_state=42)
nb = GaussianNB()

scorer = make_scorer(roc_auc_score)

cv_results = cross_validate(nb, X_N, y_N, cv=10, scoring=(scorer), return_train_score=True)

print (cv_results.keys())

print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))

print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))
nb.fit(X_train,y_train)

y_pred = nb.predict(X_val)

print(classification_report(y_val, y_pred))
X_knn = X.copy()

y_knn = y.copy()



uniquec = 5

todrop = []

col_select = list([i for i in categorical if i in cols])

print(len(col_select))

print("Unique threshold:",uniquec)

for j in col_select:

    if(len(np.unique(X_knn[j]))>uniquec):

        todrop.append(j)

print(todrop)

X_knn_ = X_knn.drop(todrop,axis=1)

for i in todrop:

    col_select.remove(i)

print(len(col_select))

print(X_knn_.shape)

for j in col_select:

    print (len(np.unique(X_knn_[j])))

X_knn_ = pd.get_dummies(X_knn_,columns=col_select)

bc = BalanceCascade(random_state=42)

X_rknn,y_rknn = bc.fit_sample(X_knn_,y_knn)



X_Nk = X_rknn[0]

y_Nk = y_rknn[0]

X_traink,X_valk,y_traink,y_valk = train_test_split(X_Nk,y_Nk,test_size=0.3,random_state=42)
train_roc = []

test_roc = []

for i in range(1,21):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_traink,y_traink)

    roc_auc_train = roc_auc_score(y_traink,knn.predict(X_traink))

    train_roc.append(roc_auc_train)

    roc_auc_test = roc_auc_score(y_valk,knn.predict(X_valk))

    test_roc.append(roc_auc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(1,21),train_roc,color='blue', linestyle='dashed', marker='o',markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(1,21),test_roc,color='red',linestyle='dashed', marker='o',markerfacecolor='blue', markersize=5)

plt.legend( [train_score, test_score],["Train ROC_AUC", "Test ROC_AUC"])

plt.title('Fig 3. ROC_AUC_Score vs K neighbors')

plt.xlabel('K neighbors')

plt.ylabel('ROC_AUC_Score')
knn = KNeighborsClassifier(n_neighbors=14)

scorer = make_scorer(roc_auc_score)

cv_results = cross_validate(knn, X_N, y_N, cv=10, scoring=(scorer), return_train_score=True)

print (cv_results.keys())

print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))

print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))
train_roc = []

test_roc = []

Cval = [0.01,0.1,1,10]

for i in Cval:

    lg = LogisticRegression(C=i)

    lg.fit(X_train,y_train)

    roc_auc_train = roc_auc_score(y_train,lg.predict(X_train))

    train_roc.append(roc_auc_train)

    roc_auc_test = roc_auc_score(y_val,lg.predict(X_val))

    test_roc.append(roc_auc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(Cval,train_roc,color='blue', linestyle='dashed', marker='o',markerfacecolor='green', markersize=5)

test_score,=plt.plot(Cval,test_roc,color='red',linestyle='dashed', marker='o',markerfacecolor='blue', markersize=5)

plt.legend( [train_score, test_score],["Train ROC_AUC", "Test ROC_AUC"])

plt.title('Fig 4. ROC_AUC_Score vs C')

plt.xlabel('C')

plt.ylabel('ROC_AUC_Score')
lg = LogisticRegression(C=1)

scorer = make_scorer(roc_auc_score)

cv_results = cross_validate(lg, X_N, y_N, cv=10, scoring=(scorer), return_train_score=True)

print (cv_results.keys())

print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))

print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))
train_roc = []

test_roc = []

for i in range(1,21):

    dt = DecisionTreeClassifier(max_depth=i,random_state=42)

    dt.fit(X_train,y_train)

    roc_auc_train = roc_auc_score(y_train,dt.predict(X_train))

    train_roc.append(roc_auc_train)

    roc_auc_test = roc_auc_score(y_val,dt.predict(X_val))

    test_roc.append(roc_auc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(1,21),train_roc,color='blue', linestyle='dashed', marker='o',markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(1,21),test_roc,color='red',linestyle='dashed', marker='o',markerfacecolor='blue', markersize=5)

plt.legend( [train_score, test_score],["Train ROC_AUC", "Test ROC_AUC"])

plt.title('Fig 5. ROC_AUC_Score vs max_depth')

plt.xlabel('max_depth')

plt.ylabel('ROC_AUC_Score')
train_roc = []

test_roc = []

for i in range(2,31):

    dt = DecisionTreeClassifier(max_depth=7,min_samples_split=i,random_state=42)

    dt.fit(X_train,y_train)

    roc_auc_train = roc_auc_score(y_train,dt.predict(X_train))

    train_roc.append(roc_auc_train)

    roc_auc_test = roc_auc_score(y_val,dt.predict(X_val))

    test_roc.append(roc_auc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(2,31),train_roc,color='blue', linestyle='dashed', marker='o',markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(2,31),test_roc,color='red',linestyle='dashed', marker='o',markerfacecolor='blue', markersize=5)

plt.legend( [train_score, test_score],["Train ROC_AUC", "Test ROC_AUC"])

plt.title('Fig 6. ROC_AUC_Score vs min_samples_split')

plt.xlabel('max_samples_split')

plt.ylabel('ROC_AUC_Score')
dt_temp = DecisionTreeClassifier() #Initialize the classifier object

parameters = {'max_depth':[4,5,7,9],'min_samples_split':[5,10,15,20,22]} #Dictionary of parameters

scorer = make_scorer(roc_auc_score) #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(dt_temp, parameters, scoring=scorer) #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train, y_train) #Fit the gridsearch object with X_train,y_train

best_rf = grid_fit.best_estimator_ #Get the best estimator. For this, check documentation of GridSearchCV object

print(grid_fit.best_params_)
dt = DecisionTreeClassifier(max_depth=5,min_samples_split=20)

scorer = make_scorer(roc_auc_score)

cv_results = cross_validate(dt, X_N, y_N, cv=10, scoring=(scorer), return_train_score=True)

print (cv_results.keys())

print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))

print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))
rf_temp = RandomForestClassifier(n_estimators = 250) #Initialize the classifier object

parameters = {'max_depth':[10,15,20,25,30],'min_samples_split':[2, 3, 4, 5]} #Dictionary of parameters

scorer = make_scorer(roc_auc_score) #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer) #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train, y_train) #Fit the gridsearch object with X_train,y_train

best_rf = grid_fit.best_estimator_ #Get the best estimator. For this, check documentation of GridSearchCV object

print(grid_fit.best_params_)
rf = RandomForestClassifier(n_estimators = 250,max_depth=15,min_samples_split=4)

scorer = make_scorer(roc_auc_score)

cv_results = cross_validate(rf, X_N, y_N, cv=10, scoring=(scorer), return_train_score=True)

print (cv_results.keys())

print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))

print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))
adb_temp = AdaBoostClassifier(random_state=42) #Initialize the classifier object

parameters = {'n_estimators':[250,300,500,600]} #Dictionary of parameters

scorer = make_scorer(roc_auc_score) #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(adb_temp, parameters, scoring=scorer) #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train, y_train) #Fit the gridsearch object with X_train,y_train

best_rf = grid_fit.best_estimator_ #Get the best estimator. For this, check documentation of GridSearchCV object

print(grid_fit.best_params_)
adb = AdaBoostClassifier(n_estimators = 600,random_state=42)

scorer = make_scorer(roc_auc_score)

cv_results = cross_validate(adb, X_N, y_N, cv=10, scoring=(scorer), return_train_score=True)

print (cv_results.keys())

print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))

print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))
from sklearn.ensemble import VotingClassifier

clf1 = rf

clf2 = adb

eclf = VotingClassifier(estimators=[('rf', clf1), ('adb', clf2)], voting='soft',weights=[0.3,0.7])

scorer = make_scorer(roc_auc_score)

cv_results = cross_validate(eclf, X_N, y_N, cv=10, scoring=(scorer), return_train_score=True)

print (cv_results.keys())

print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))

print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))
print(len(cols))
adb.fit(X_res[0],y_res[0])
le_test['pred'] = adb.predict(le_test[cols])

sub_csv = le_test[['ID','pred']]

sub_csv = sub_csv.rename(columns = {'pred':'Class'})

sub_csv.to_csv('FinalSubmission.csv',index = False)
import pandas as pd

sub_csv = pd.read_csv('FinalSubmission.csv')

sub_csv['Class'].value_counts()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html ='<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(sub_csv)