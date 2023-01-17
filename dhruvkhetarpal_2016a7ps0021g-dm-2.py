import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data_orig = pd.read_csv("../input/train.csv", sep=',')

data2_orig = pd.read_csv("../input/test.csv", sep=',')

data = data_orig

datat=data2_orig
data.head()
data2=data

data3=data

idd=datat['ID']
data = data.replace('?',np.NaN)                     #replacing ? with NaN

datat = datat.replace('?',np.NaN)
dataones=data.loc[data['Class'] == 1]                  #finding number of ones in class
dataones.shape
datazeros=data.loc[data['Class'] == 0]
datazeros.shape
datazerocurt=datazeros.sample(n=6292, random_state=2)                #random sampling
numerical=['Age','IC','OC','Weight','NOP','Vet_Benefits','Weaks','WorkingPeriod','Gain','Loss','Stock','Timely Income','Own/Self','Vet_Benefits','Class']
datan=data[numerical]
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = datan.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
categorical=['Worker Class','MIC','MOC','Cast','Hispanic','Full/Part','Tax Status','Detailed','Summary','MSA','REG','MOVE','Live','Teen','COB FATHER','COB MOTHER','COB SELF','Citizen']
for col in categorical[:]:

    print(data[col].value_counts())
data=data.drop(['ID'],1)
datat=datat.drop(['ID'],1)
data=data.drop(['MSA','REG','MOVE','COB FATHER','COB MOTHER','COB SELF','Citizen','Hispanic','Weight'],1)

datat=datat.drop(['MSA','REG','MOVE','COB FATHER','COB MOTHER','COB SELF','Citizen','Hispanic','Weight'],1)
data=data.drop(['Summary','Full/Part','Schooling','Detailed','Worker Class','Enrolled','MLU','Reason','Area','State','PREV','Fill','Married_Life','Cast'],1)

datat=datat.drop(['Summary','Full/Part','Schooling','Detailed','Worker Class','Enrolled','MLU','Reason','Area','State','PREV','Fill','Married_Life','Cast'],1)
data.head()
#data=data.drop(['Teen'],1)

data['Teen'] = data['Teen'].fillna(pd.Series(np.random.choice(['B', 'M'], 

                                                      p=[0.67,0.33], size=len(data))))

data['Live'] = data['Live'].fillna(pd.Series(np.random.choice(['YES', 'NO'], 

                                                      p=[0.85,0.15], size=len(data))))

datat['Teen'] = datat['Teen'].fillna(pd.Series(np.random.choice(['B', 'M'], 

                                                      p=[0.67,0.33], size=len(data))))

datat['Live'] = datat['Live'].fillna(pd.Series(np.random.choice(['YES', 'NO'], 

                                                      p=[0.85,0.15], size=len(data))))
data['Live'].replace({'NO':0, 'YES':1},inplace=True)
datat['Live'].replace({'NO':0, 'YES':1},inplace=True)
data=data.drop(['MIC','MOC'],1)
datat=datat.drop(['MIC','MOC'],1)
data.head()
dataonesf=data.loc[data['Class'] == 1]
datazerosf=data.loc[data['Class'] == 0]
datazerocurtf=datazerosf.sample(n=6292, random_state=2)
datazerocurtf
dataf=pd.concat([datazerocurtf,dataonesf],ignore_index=True)
dataf
data=dataf
y=data['Class']

datao=data

dataoo=data
data=data.drop(['Class'],1)
X=data
data = pd.get_dummies(data, columns=["Sex","Teen","Tax Status"])

datat = pd.get_dummies(datat, columns=["Sex","Teen","Tax Status"])

X=data
X
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
X_val.shape
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X_train)

X_train = pd.DataFrame(np_scaled)

np_scaled_val = min_max_scaler.transform(X_val)

X_val = pd.DataFrame(np_scaled_val)

X_train.head()
np.random.seed(42)
from sklearn.naive_bayes import GaussianNB as NB

#NB?
nb = NB()

nb.fit(X_train,y_train)

nb.score(X_val,y_val)
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(solver = 'liblinear', C = 1, multi_class = 'ovr', random_state = 42)

lg.fit(X_train,y_train)

lg.score(X_val,y_val)
lg = LogisticRegression(solver = 'lbfgs', C = 8, multi_class = 'multinomial', random_state = 42)

lg.fit(X_train,y_train)

lg.score(X_val,y_val)
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

    dTree = DecisionTreeClassifier(max_depth = 10, min_samples_split=i, random_state = 42)

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
dTree = DecisionTreeClassifier(max_depth=6, random_state = 42)

dTree.fit(X_train,y_train)

dTree.score(X_val,y_val)

#predy=dTree.predict(datat)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

y_pred_DT = dTree.predict(X_val)

print(confusion_matrix(y_val, y_pred_DT))
print(classification_report(y_val, y_pred_DT))
from sklearn.ensemble import RandomForestClassifier

#RandomForestClassifier?
score_train_RF = []

score_test_RF = []



for i in range(1,25,1):

    rf = RandomForestClassifier(n_estimators=i, random_state = 42)

    rf.fit(X_train, y_train)

    sc_train = rf.score(X_train,y_train)

    score_train_RF.append(sc_train)

    sc_test = rf.score(X_val,y_val)

    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(1,25,1),score_train_RF,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(1,25,1),score_test_RF,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score,test_score],["Train Score","Test Score"])

plt.title('Fig4. Score vs. No. of Trees')

plt.xlabel('No. of Trees')

plt.ylabel('Score')
from sklearn.metrics import fbeta_score, make_scorer, f1_score

from sklearn.model_selection import GridSearchCV



rf_temp = RandomForestClassifier(n_estimators = 45)        #Initialize the classifier object



parameters = {'max_depth':[3, 5, 8, 12],'min_samples_split':[2, 3, 4, 5]}    #Dictionary of parameters



scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train



best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



print(grid_fit.best_params_)
rf_best = RandomForestClassifier(n_estimators = 60,max_depth = 60, min_samples_split = 60)

rf_best.fit(X_train, y_train)

rf_best.score(X_val,y_val)

#predy=rf_best.predict(datat)
y_pred_RF_best = rf_best.predict(X_val)

confusion_matrix(y_val, y_pred_RF_best)
print(classification_report(y_val, y_pred_RF_best))
rf = RandomForestClassifier(n_estimators=45, random_state = 42)

rf.fit(X_train, y_train)

rf.score(X_val,y_val)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

y_pred_RF = rf.predict(X_val)

confusion_matrix(y_val, y_pred_RF)
print(classification_report(y_val, y_pred_RF))
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=500,learning_rate=1,random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

accuracy_score(y_val, y_pred)
clf = AdaBoostClassifier(n_estimators=500,learning_rate=1,random_state=42)

clf.fit(X, y)

predy = clf.predict(datat)

accuracy_score(y_val, y_pred)
predy.shape
dframe = pd.DataFrame(predy)

rf.score(datat,dframe)
dff=pd.DataFrame(idd)
final_data=dff.join(dframe,how='left')

final_data= final_data.rename(columns={0: "Class"})
final_data.to_csv('final_submission_2.csv', index = False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(final_data)