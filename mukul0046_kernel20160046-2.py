import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
data_orig = pd.read_csv("../input/train.csv", sep=',')

data = data_orig
data = data.drop(['ID'], axis = 1)

data.head()
df1 = data[data.Class == 0].sample(6000)

df2 = data[data.Class == 1].sample(6290)
frames = [df1, df2]

result = pd.concat(frames)

result.head()
result.info()
y=result['Class']
result.drop(['Class', 'Enrolled', 'MIC', 'MOC', 'MLU', 'Reason', 'Area',

           'REG', 'MSA', 'State', 'Live', 'PREV', 'MOVE', 'Teen', 'Fill'],axis=1, inplace=True)
result.drop(['Worker Class', 'Detailed', 'Schooling'],axis=1,inplace=True)
result.drop(['Married_Life', 'Full/Part', 'Summary', 'Tax Status', 'COB FATHER', 'COB MOTHER', 'COB SELF', 'Hispanic'],axis=1,inplace=True)
result.info()

#data.drop(['Tax Status', 'Married_Life', 'Full/Part', 'Summary', 'Detailed', 'Schooling'],axis=1,inplace=True)
result.head()
df = pd.get_dummies(result, columns = [ "Sex",  "Own/Self",  "Citizen", "Vet_Benefits", "Cast"])
X = df

X.head()
X.info()
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
np.random.seed(42)

from sklearn.naive_bayes import GaussianNB as NB
nb = NB()

nb.fit(X_train,y_train)

nb.score(X_val,y_val)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score



y_pred_NB = nb.predict(X_val)

print(confusion_matrix(y_val, y_pred_NB))
print(classification_report(y_val, y_pred_NB))
from sklearn.neighbors import KNeighborsClassifier
train_acc = []

test_acc = []

for i in range(1,15):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    acc_train = knn.score(X_train,y_train)

    train_acc.append(acc_train)

    acc_test = knn.score(X_val,y_val)

    test_acc.append(acc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(1,15),train_acc,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(1,15),test_acc,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score, test_score],["Train Accuracy", "Test Accuracy"])

plt.title('Accuracy vs K neighbors')

plt.xlabel('K neighbors')

plt.ylabel('Accuracy')
knn = KNeighborsClassifier(n_neighbors=12)

knn.fit(X_train,y_train)

knn.score(X_val,y_val)
y_pred_KNN = knn.predict(X_val)

cfm = confusion_matrix(y_val, y_pred_KNN, labels = [0,1])

print(cfm)
print(classification_report(y_val, y_pred_KNN))
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(solver = 'liblinear', C = 2, multi_class = 'ovr', random_state = 42)

lg.fit(X_train,y_train)

lg.score(X_val,y_val)
lg = LogisticRegression(solver = 'lbfgs', C = 10, multi_class = 'multinomial', random_state = 42)

lg.fit(X_train,y_train)

lg.score(X_val,y_val)
y_pred_LR = lg.predict(X_val)

print(confusion_matrix(y_val, y_pred_LR))
print(classification_report(y_val, y_pred_LR))
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

    dTree = DecisionTreeClassifier(max_depth = 6, min_samples_split=i, random_state = 42)

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
y_pred_DT = dTree.predict(X_val)

print(confusion_matrix(y_val, y_pred_DT))
print(classification_report(y_val, y_pred_DT))
from sklearn.ensemble import RandomForestClassifier
score_train_RF = []

score_test_RF = []



for i in range(1,18,1):

    rf = RandomForestClassifier(n_estimators=i, class_weight="balanced", random_state = 42)

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
for x in range(8, 25):

    rf = RandomForestClassifier(n_estimators=x, class_weight="balanced", random_state = 42)

    rf.fit(X_train, y_train)

    #print(rf.score(X_val,y_val), x)

    y_pred_RF = rf.predict(X_val)

    print(classification_report(y_val, y_pred_RF), x)
rf = RandomForestClassifier(n_estimators=16, class_weight="balanced", random_state = 42)

rf.fit(X_train, y_train)

print(rf.score(X_val,y_val))

y_pred_RF = rf.predict(X_val)

print(classification_report(y_val, y_pred_RF))
#y_pred_RF = rf.predict(X_val)

confusion_matrix(y_val, y_pred_RF)
print(classification_report(y_val, y_pred_RF))
ddf = pd.read_csv("../input/test.csv", sep=',')
ddf = ddf.drop(['ID'], axis = 1)

ddf.head()
ddf.drop(['Enrolled', 'MIC', 'MOC', 'MLU', 'Reason', 'Area', 'REG', 'MSA', 'State', 'Live', 'PREV', 'MOVE', 'Teen', 'Fill'],axis=1, inplace=True)

ddf.drop(['Worker Class', 'COB FATHER', 'COB MOTHER', 'COB SELF', 'Hispanic', 'Detailed', 'Schooling'],axis=1,inplace=True)

ddf.drop(['Married_Life', 'Full/Part', 'Summary',  'Tax Status'],axis=1,inplace=True)
ddf = pd.get_dummies(ddf, columns = ["Sex", "Own/Self", "Cast", "Vet_Benefits", "Citizen"])
ddf.head()
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(ddf)

dataN1 = pd.DataFrame(np_scaled)

dataN1.head()
#ddf.info()
X1 = dataN1
rf = RandomForestClassifier(n_estimators=16, class_weight="balanced", random_state = 42)

rf.fit(X_train, y_train)

y_fin = rf.predict(X1)
data1 = pd.read_csv("../input/test.csv", sep=',')
dataf = data1['ID']
res1 = pd.DataFrame(y_fin)

final = pd.concat([data1["ID"], res1], axis=1).reindex()

final = final.rename(columns={0: "Class"})

len(final)
final.to_csv('submission69.csv', index = False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(final)