import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data_orig = pd.read_csv("../input/train.csv", sep=',')

data = data_orig
data.head()

#data['Worker Class'].unique()

data.drop('ID', axis=1, inplace=True)
data1=data

#data['Age'].unique()

data.dtypes
data['Worker Class'].unique()
data1.drop('Worker Class', axis=1, inplace=True)
#data1.head()

#data1['IC'].unique()

data1['OC'].unique()
data1['Schooling'].unique()
data1.drop('Schooling', axis=1, inplace=True)
data1.head()
data1.drop('Enrolled', axis=1, inplace=True)

#data1['Timely Income'].unique()
data1.head()
data1['Married_Life'].unique()
#data2 = pd.get_dummies(data1['Married_Life'])

data1.drop(['Married_Life','Cast'], axis=1, inplace=True)

#data3= pd.get_dummies(data1, prefix='Married_Life_', columns=['Married_Life'])
data3=data1

data3.head()

data3.drop(['MIC', 'MOC','Sex','MLU','Reason'], axis=1, inplace=True)
data3.head()
data_orign = pd.read_csv("../input/train.csv", sep=',')
data_orign.head()
data3.head()
data3.drop(['Area'], axis=1, inplace=True)
data3.drop(['PREV'], axis=1, inplace=True)
data3.drop(['MSA'], axis=1, inplace=True)
data3.drop(['REG'], axis=1, inplace=True)
data3.head()
data3.drop(['State'], axis=1, inplace=True)
data3.drop(['Teen'], axis=1, inplace=True)
data3.head()
#data4= pd.get_dummies(data3, prefix='Cast_', columns=['Cast'])
data4=data3

data4.head()
data4.drop(['Fill'], axis=1, inplace=True)
data4.head()
data3['Hispanic'].unique()
data5=data4.drop(['Hispanic'], axis=1, inplace=True)
data4.head()
data4['Citizen'].unique()
data.head()
data_orig.head()
data4.head()
data4= pd.get_dummies(data4, prefix='Citizen_', columns=['Citizen'])
data['COB SELF'].unique()
data4.drop(['COB SELF'], axis=1, inplace=True)
data4.head()
data4.drop(['COB MOTHER','COB FATHER'], axis=1, inplace=True)
data4.head()
data4.drop(['MOVE','Live'], axis=1, inplace=True)
data4.head()
data['Summary'].unique()
data4= pd.get_dummies(data4, prefix='Summary_', columns=['Summary'])
data4.head()
data['Detailed'].unique()
data4= pd.get_dummies(data4, prefix='Full/Part_', columns=['Full/Part'])
data4= pd.get_dummies(data4, prefix='Tax Status_', columns=['Tax Status'])
data5=data4.copy()



data5= pd.get_dummies(data5, prefix='Detailed_', columns=['Detailed'])

data5.info()
data4.drop(['Detailed'], axis=1, inplace=True)
data4.head

data4.info()
data4_sample1= data4[data4['Class']==1]

data4_sample0= data4[data4['Class']==0]

chotu = data4_sample0.sample(frac=0.07)

chotu

data66= pd.concat([data4_sample1,chotu], ignore_index=True)

data66



data66.drop(['Citizen__Case1'], axis=1, inplace=True)

data66.drop(['Citizen__Case2'], axis=1, inplace=True)

data66.drop(['Citizen__Case3'], axis=1, inplace=True)

data66.drop(['Citizen__Case4'], axis=1, inplace=True)

data66.drop(['Citizen__Case5'], axis=1, inplace=True)
data66.drop(['Summary__sum1'], axis=1, inplace=True)

data66.drop(['Summary__sum2'], axis=1, inplace=True)

data66.drop(['Summary__sum3'], axis=1, inplace=True)

data66.drop(['Summary__sum4'], axis=1, inplace=True)

data66.drop(['Summary__sum5'], axis=1, inplace=True)

data66.drop(['Summary__sum6'], axis=1, inplace=True)

data66.drop(['Summary__sum7'], axis=1, inplace=True)

data66.drop(['Summary__sum8'], axis=1, inplace=True)

data66


y=data66['Class']

X=data66.drop(['Class'],axis=1)

X.head()
from sklearn.model_selection import train_test_split

X_Train, X_val, y_Train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X_Train)

X_Train = pd.DataFrame(np_scaled)

np_scaled_val = min_max_scaler.transform(X_val)

X_val = pd.DataFrame(np_scaled_val)

X_Train.head()
np.random.seed(2000)

from sklearn.naive_bayes import GaussianNB as NB
from sklearn.linear_model import LogisticRegression
nb = NB()

nb.fit(X_Train,y_Train)



nb.score(X_val,y_val)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score



y_pred_NB = nb.predict(X_val)

print(confusion_matrix(y_val, y_pred_NB))
print(classification_report(y_val, y_pred_NB))
from sklearn.metrics import roc_auc_score



roc_auc_score(y_val, y_pred_NB)
lg = LogisticRegression(solver = 'liblinear', C = 1, multi_class = 'ovr', random_state = 42)

lg.fit(X_Train,y_Train)

lg.score(X_val,y_val)
lg = LogisticRegression(solver = 'lbfgs', C = 100, multi_class = 'multinomial', random_state = 42)

lg.fit(X_Train,y_Train)

lg.score(X_val,y_val)
y_pred_LR = lg.predict(X_val)

print(confusion_matrix(y_val, y_pred_LR))
print(classification_report(y_val, y_pred_LR))
from sklearn.metrics import roc_auc_score



roc_auc_score(y_val, y_pred_LR)
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier



train_acc = []

test_acc = []

for i in range(1,15):

    dTree = DecisionTreeClassifier(max_depth=i)

    dTree.fit(X_Train,y_Train)

    acc_train = dTree.score(X_Train,y_Train)

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

    dTree = DecisionTreeClassifier(max_depth = 7, min_samples_split=i, random_state = 42)

    dTree.fit(X_Train,y_Train)

    acc_train = dTree.score(X_Train,y_Train)

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

dTree.fit(X_Train,y_Train)

dTree.score(X_val,y_val)
y_pred_DT = dTree.predict(X_val)

print(confusion_matrix(y_val, y_pred_DT))
print(classification_report(y_val, y_pred_DT))
from sklearn.metrics import roc_auc_score



roc_auc_score(y_val, y_pred_DT)
from sklearn.ensemble import RandomForestClassifier
score_train_RF = []

score_test_RF = []



for i in range(1,18,1):

    rf = RandomForestClassifier(n_estimators=i, random_state = 42)

    rf.fit(X_Train, y_Train)

    sc_train = rf.score(X_Train,y_Train)

    score_train_RF.append(sc_train)

    sc_test = rf.score(X_val,y_val)

    score_test_RF.append(sc_test)
plt.figure(figsize=(20,12))

train_score,=plt.plot(range(1,18,1),score_train_RF,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(1,18,1),score_test_RF,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score,test_score],["Train Score","Test Score"])

plt.title('Fig4. Score vs. No. of Trees')

plt.xlabel('No. of Trees')

plt.ylabel('Score')
rf = RandomForestClassifier(n_estimators=17, random_state = 42)

rf.fit(X_Train, y_Train)

rf.score(X_val,y_val)
y_pred_RF = rf.predict(X_val)

confusion_matrix(y_val, y_pred_RF)
print(classification_report(y_val, y_pred_RF))
from sklearn.metrics import roc_auc_score



roc_auc_score(y_val, y_pred_RF)

test1 = pd.read_csv("../input/test.csv", sep=',')

test2=test1

test2.head()
test2.head()
test1.head()

test1.drop('ID', axis=1, inplace=True)


test2.head()
test1.drop('Worker Class', axis=1, inplace=True)
test1.drop('Schooling', axis=1, inplace=True)
test1.drop('Enrolled', axis=1, inplace=True)

#data1['Timely Income'].unique()
test3=test1
test3



#test3.drop(['MIC', 'MOC','Sex','MLU','Reason'], axis=1, inplace=True)
test3.drop(['Area'], axis=1, inplace=True)

test3.drop(['PREV'], axis=1, inplace=True)

test3.drop(['MSA'], axis=1, inplace=True)

test3.drop(['REG'], axis=1, inplace=True)
test3.drop(['State'], axis=1, inplace=True)

test3.drop(['Teen'], axis=1, inplace=True)
#test4= pd.get_dummies(test3, prefix='Cast_', columns=['Cast'])
test4=test3

test4.drop(['Fill'], axis=1, inplace=True)
test4.drop(['Hispanic'], axis=1, inplace=True)
test4= pd.get_dummies(test4, prefix='Citizen_', columns=['Citizen'])
test4.drop(['COB SELF'], axis=1, inplace=True)
test4.drop(['COB MOTHER','COB FATHER'], axis=1, inplace=True)
test4.drop(['MOVE','Live'], axis=1, inplace=True)
test4= pd.get_dummies(test4, prefix='Summary_', columns=['Summary'])
test4= pd.get_dummies(test4, prefix='Full/Part_', columns=['Full/Part'])
test4= pd.get_dummies(test4, prefix='Tax Status_', columns=['Tax Status'])
test4.drop(['Detailed'], axis=1, inplace=True)

test4.drop(['MIC', 'MOC','Sex','MLU','Reason'], axis=1, inplace=True)
test4.drop(['Married_Life','Cast'], axis=1, inplace=True)

test4
test4.drop(['Summary__sum1'], axis=1, inplace=True)

test4.drop(['Summary__sum2'], axis=1, inplace=True)

test4.drop(['Summary__sum3'], axis=1, inplace=True)

test4.drop(['Summary__sum4'], axis=1, inplace=True)

test4.drop(['Summary__sum5'], axis=1, inplace=True)

test4.drop(['Summary__sum6'], axis=1, inplace=True)

test4.drop(['Summary__sum7'], axis=1, inplace=True)

test4.drop(['Summary__sum8'], axis=1, inplace=True)
test4.drop(['Citizen__Case1'], axis=1, inplace=True)

test4.drop(['Citizen__Case2'], axis=1, inplace=True)

test4.drop(['Citizen__Case3'], axis=1, inplace=True)

test4.drop(['Citizen__Case4'], axis=1, inplace=True)

test4.drop(['Citizen__Case5'], axis=1, inplace=True)
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X)

X1= pd.DataFrame(np_scaled)

X1.head()

X1
X
#lg = LogisticRegression(solver = 'lbfgs', C = 100, multi_class = 'multinomial', random_state = 42)

#lg.fit(X1,y)

dTree1 = DecisionTreeClassifier(max_depth=6, random_state = 42)

dTree1.fit(X1,y)

rf = RandomForestClassifier(n_estimators=16, random_state = 42)

rf.fit(X1, y)

test4
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(test4)

test41= pd.DataFrame(np_scaled)

test41.head()
test41
predictions=lg.predict(test41)

predictions1=dTree1.predict(test41)

predictions2=rf.predict(test41)

testy = pd.read_csv("../input/test.csv", sep=',')

ids = testy[['ID']]
#esults=ids.assign(Class=predictions)

results2=ids.assign(Class=predictions2)

results2
#results.to_csv("try1.csv", index=False)

results2.to_csv("try12.csv", index=False)
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



create_download_link(results2)