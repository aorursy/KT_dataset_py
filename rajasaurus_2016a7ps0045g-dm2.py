import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

missing_values = ["?"]

data = pd.read_csv("../input/train.csv", na_values = missing_values)

data_orig = data

data
test = pd.read_csv("../input/test.csv", na_values = missing_values)

test_orig = test

test


data.duplicated().sum()
print (data.isnull().sum())
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
null_columns = data.columns[data.isnull().any()]

null_columns
data_drop_col = data.drop([ 'ID','Enrolled', 'MLU', 'Reason',

       'Area', 'State', 'PREV',

       'Fill'], 1) # Drop Total from domain knowledge



data_drop_col.info()
test_drop_col = test.drop([ 'ID','Enrolled', 'MLU', 'Reason',

       'Area', 'State', 'PREV',

       'Fill'], 1) # Drop Total from domain knowledge



test_drop_col.info()
data_nonsense = data.drop([ 'ID','Worker Class','MIC', 'MOC','MSA', 'REG', 'MOVE', 'Live','Enrolled', 'MLU', 'Reason',

       'Area', 'State', 'PREV', 'Teen',

       'Fill'], 1) # Drop Total from domain knowledge

data_nonsense.info()
test_nonsense = test.drop([ 'ID','Worker Class','MIC', 'MOC','MSA', 'REG', 'MOVE', 'Live','Enrolled', 'MLU', 'Reason',

       'Area', 'State', 'PREV', 'Teen',

       'Fill'], 1) # Drop Total from domain knowledge

test_nonsense.info()
data_drop_teen = data.drop([ 'ID','Enrolled', 'MLU', 'Reason',

       'Area', 'State', 'PREV','Teen',

       'Fill'], 1) # Drop Total from domain knowledge

data_drop_teen.info()
test_drop_teen = test.drop([ 'ID','Enrolled', 'MLU', 'Reason',

       'Area', 'State', 'PREV','Teen',

       'Fill'], 1) # Drop Total from domain knowledge

test_drop_teen.info()
data_no_nonsense_g=data_nonsense.dropna()

data_no_nonsense_g.info()
test_no_nonsense_g=test_nonsense.dropna()

test_no_nonsense_g.info()
data_no_nonsense=data.drop(['Worker Class', 'Enrolled', 'MIC', 'MOC', 'Hispanic', 'MLU', 'Reason',

       'Area', 'State', 'MSA', 'REG', 'MOVE', 'Live', 'PREV', 'Teen',

       'COB FATHER', 'COB MOTHER', 'COB SELF', 'Fill'],1)

data_no_nonsense.info()
test_no_nonsense=test.drop(['Worker Class', 'Enrolled', 'MIC', 'MOC', 'Hispanic', 'MLU', 'Reason',

       'Area', 'State', 'MSA', 'REG', 'MOVE', 'Live', 'PREV', 'Teen',

       'COB FATHER', 'COB MOTHER', 'COB SELF', 'Fill'],1)

test_no_nonsense.info()
import seaborn as sns

f, ax = plt.subplots(figsize=(20, 16))

corr =data_no_nonsense.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
def S_method(v):

    if v=="M":

        return 0

    else:

        return 1

    

data_no_nonsense["Sex"] = data_no_nonsense["Sex"].apply(S_method)

data_no_nonsense["Sex"].value_counts()    

test_no_nonsense["Sex"] = test_no_nonsense["Sex"].apply(S_method)

test_no_nonsense["Sex"].value_counts()  
(['Worker Class', 'Enrolled', 'MIC', 'MOC', 'Hispanic', 'MLU', 'Reason',

       'Area', 'State', 'MSA', 'REG', 'MOVE', 'Live', 'PREV', 'Teen',

       'COB FATHER', 'COB MOTHER', 'COB SELF', 'Fill'],1)
def Schoo_method(v):

    #print(type(v))

    x = (v[3:])

    

    return int(x)

    

data_no_nonsense["Schooling"] = data_no_nonsense["Schooling"].apply(Schoo_method)

data_no_nonsense["Schooling"].value_counts()

test_no_nonsense["Schooling"] = test_no_nonsense["Schooling"].apply(Schoo_method)

test_no_nonsense["Schooling"].value_counts()
def C_method(v):

    if(v=="TypeA"):

        return 1

    elif(v=="TypeB"):

        return 2

    elif(v=="TypeC"):

        return 3

    elif(v=="TypeD"):

        return 4

    else:

        return 5

                

        

data_no_nonsense["Cast"] = data_no_nonsense["Cast"].apply(C_method)

data_no_nonsense["Cast"].value_counts()    



test_no_nonsense["Cast"] = test_no_nonsense["Cast"].apply(C_method)

test_no_nonsense["Cast"].value_counts()    
(['Worker Class', 'Enrolled', 'MIC', 'MOC', 'Hispanic', 'MLU', 'Reason',

       'Area', 'State', 'MSA', 'REG', 'MOVE', 'Live', 'PREV', 'Teen',

       'COB FATHER', 'COB MOTHER', 'COB SELF', 'Fill'],1)
def f_method(v):

    if(v=="FA"):

        return 1

    elif(v=="FB"):

        return 2

    elif(v=="FC"):

        return 3

    elif(v=="FD"):

        return 4

    elif(v=="FE"):

        return 5

    elif(v=="FF"):

        return 6

    elif(v=="FG"):

        return 7

    else:

        return 8

            

data_no_nonsense["Full/Part"] = data_no_nonsense["Full/Part"].apply(f_method)

data_no_nonsense["Full/Part"].value_counts()    



test_no_nonsense["Full/Part"] = test_no_nonsense["Full/Part"].apply(f_method)

test_no_nonsense["Full/Part"].value_counts()    
def M_method(v):

    

    x = (v[2:])

    

    return int(x)

    

data_no_nonsense["Married_Life"] = data_no_nonsense["Married_Life"].apply(M_method)

data_no_nonsense["Married_Life"].value_counts()



test_no_nonsense["Married_Life"] = test_no_nonsense["Married_Life"].apply(M_method)

test_no_nonsense["Married_Life"].value_counts()
def Ts_method(v):

    x = (v[-1:])

    

    return int(x)

                

data_no_nonsense["Tax Status"] = data_no_nonsense["Tax Status"].apply(Ts_method)

data_no_nonsense["Tax Status"].value_counts()   



test_no_nonsense["Tax Status"] = test_no_nonsense["Tax Status"].apply(Ts_method)

test_no_nonsense["Tax Status"].value_counts()   
def Summ_method(v):

    x = (v[-1:])

    

    return int(x)

                

data_no_nonsense["Summary"] = data_no_nonsense["Summary"].apply(Summ_method)

data_no_nonsense["Summary"].value_counts()    



test_no_nonsense["Summary"] = test_no_nonsense["Summary"].apply(Summ_method)

test_no_nonsense["Summary"].value_counts()    
def D_method(v):

    x = (v[1:])

    

    return int(x)

                

data_no_nonsense["Detailed"] = data_no_nonsense["Detailed"].apply(D_method)

data_no_nonsense["Detailed"].value_counts()



test_no_nonsense["Detailed"] = test_no_nonsense["Detailed"].apply(D_method)

test_no_nonsense["Detailed"].value_counts()
def Ci_method(v):

    x = (v[-1:])

    

    return int(x)

                

data_no_nonsense["Citizen"] = data_no_nonsense["Citizen"].apply(Ci_method)

data_no_nonsense["Citizen"].value_counts()    
test_no_nonsense["Citizen"] = test_no_nonsense["Citizen"].apply(Ci_method)

test_no_nonsense["Citizen"].value_counts()    
test_no_nonsense.head()
no_nons = pd.get_dummies(data_no_nonsense)

no_nons_t = pd.get_dummies(test_no_nonsense)

no_nons_t.info()
no_nons.info()
y=data_no_nonsense['Class']

X=data_no_nonsense.drop(['Class'],axis=1)

X.head()
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X_train)

X_train = pd.DataFrame(np_scaled)

np_scaled_val = min_max_scaler.transform(X_val)

X_val = pd.DataFrame(np_scaled_val)

X_train.head()
tr0=no_nons[no_nons['Class']==0]

tr1=no_nons[no_nons['Class']==1]

tr0_m = tr0.sample(frac=0.067)

St = pd.concat([tr0_m,tr1], ignore_index=True)
Sy=St['Class']

SX=St.drop(['Class'],axis=1)

SX.head()
np.random.seed(42)
from sklearn.naive_bayes import GaussianNB as NB

#NB?
nb = NB()

nb.fit(X_train,y_train)

nb.score(X_val,y_val)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score



y_pred_NB = nb.predict(X_val)

print(confusion_matrix(y_val, y_pred_NB))
print(classification_report(y_val, y_pred_NB))
from sklearn.linear_model import LogisticRegression

#LogisticRegression?
lg = LogisticRegression(solver = 'liblinear', C = 1, multi_class = 'ovr', random_state = 42)

lg.fit(X_train,y_train)

lg.score(X_val,y_val)
lg = LogisticRegression(solver = 'lbfgs', C = 8, multi_class = 'multinomial', random_state = 42)

lg.fit(X_train,y_train)

lg.score(X_val,y_val)
y_pred_LR = lg.predict(X_val)

print(confusion_matrix(y_val, y_pred_LR))
print(classification_report(y_val, y_pred_LR))
from sklearn.tree import DecisionTreeClassifier

#DecisionTreeClassifier?
from sklearn.tree import DecisionTreeClassifier



train_acc = []

test_acc = []

for i in range(1,15):

    dTree = DecisionTreeClassifier(max_depth=i)

    dTree.fit(X_train,y_train)

    acc_train = dTree.score(X_train,y_train)

    train_acc.append(acc_train)

    acc_test = dTree.score(X_val,y_val)

    test_acc.append(acc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(1,15),train_acc,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(1,15),test_acc,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score, test_score],["Train Accuracy", "Validation Accuracy"])

plt.title('Accuracy vs Max Depth')

plt.xlabel('Max Depth')

plt.ylabel('Accuracy')
from sklearn.tree import DecisionTreeClassifier



train_acc = []

test_acc = []

for i in range(2,30):

    dTree = DecisionTreeClassifier(max_depth = 9, min_samples_split=i, random_state = 42)

    dTree.fit(X_train,y_train)

    acc_train = dTree.score(X_train,y_train)

    train_acc.append(acc_train)

    acc_test = dTree.score(X_val,y_val)

    test_acc.append(acc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(2,30),train_acc,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(2,30),test_acc,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score, test_score],["Train Accuracy", "Validation Accuracy"])

plt.title('Accuracy vs min_samples_split')

plt.xlabel('Max Depth')

plt.ylabel('Accuracy')
dTree = DecisionTreeClassifier(max_depth=9, random_state = 42)

dTree.fit(X_train,y_train)

dTree.score(X_val,y_val)
y_pred_DT = dTree.predict(X_val)

print(confusion_matrix(y_val, y_pred_DT))
print(classification_report(y_val, y_pred_DT))
from sklearn.ensemble import RandomForestClassifier

#RandomForestClassifier?
score_train_RF = []

score_test_RF = []



for i in range(1,18,1):

    rf = RandomForestClassifier(n_estimators=i, random_state = 42)

    rf.fit(X_train, y_train)

    sc_train = rf.score(X_train,y_train)

    score_train_RF.append(sc_train)

    sc_test = rf.score(X_val,y_val)

    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(1,18,1),score_train_RF,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(1,18,1),score_test_RF,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score,test_score],["Train Score","Test Score"])

plt.title('Fig4. Score vs. No. of Trees')

plt.xlabel('No. of Trees')

plt.ylabel('Score')
rf = RandomForestClassifier(n_estimators=15, random_state = 42)

rf.fit(X_train, y_train)

rf.score(X_val,y_val)
y_pred_RF = rf.predict(X_val)

confusion_matrix(y_val, y_pred_RF)
print(classification_report(y_val, y_pred_RF))
param_grid = { 

    'n_estimators': [350,400,450,500,600],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [16,17,19 ,20,22,24,25],

    'criterion' :['gini', 'entropy']

}
xnp = min_max_scaler.fit_transform(SX)

X_n = pd.DataFrame(xnp)
X_n.head()
tnp = min_max_scaler.fit_transform(no_nons_t)

t_n = pd.DataFrame(tnp)

t_n.head()
from sklearn.model_selection import GridSearchCV

CV_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5, verbose=20,n_jobs=-1)

CV_rfc.fit(X_n,Sy)
CV_rfc.best_params_
rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 500, max_depth=22, criterion='entropy')
rfc1.fit(SX,Sy)
pred_rf = CV_rfc.predict(t_n)
temp = test['ID']

ttt=pd.DataFrame(temp)

ttt.head()
pp = {'ID':temp,'Class':pred_rf}

pred = pd.DataFrame(pp)

pred.head()
pred.to_csv("t1-7.csv", index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html='<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link("submission_DataFrame_name")