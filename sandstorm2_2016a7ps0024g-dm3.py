import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#missing_values = ["0"]  ,na_values = missing_values

benign_data = pd.read_csv("../input/opcode_frequency_benign.csv")

benign_data.head()
malign_data = pd.read_csv("../input/opcode_frequency_malware.csv")

malign_data.head()
benign_data['1809'] = 0

malign_data['1809'] = 1
data=[benign_data,malign_data]

traindata_orig=pd.concat(data)

traindata_orig
#first are benign and second are malign 

traindata_orig['1809'].value_counts()
traindata_orig_drop = traindata_orig
traindata_orig_drop = traindata_orig_drop.drop(['FileName'], axis = 1)

traindata_orig_drop
#data_original = data_orig_drop

#data_original.info()
#removing Class so we can predict i

#data_original = data_orig_drop

y=traindata_orig_drop['1809']

X=traindata_orig_drop.drop(columns=['1809'],axis=1)

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

#NB?
nb = NB()

nb.fit(X_train,y_train)

nb.score(X_val,y_val)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score



y_pred_NB = nb.predict(X_val)

print(confusion_matrix(y_val, y_pred_NB))
print(classification_report(y_val, y_pred_NB))
from sklearn.neighbors import KNeighborsClassifier
train_acc =[]

test_acc = []

for i in range(1,15):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    acc_train = knn.score(X_train,y_train)

    train_acc.append(acc_train)

    acc_test = knn.score(X_val,y_val)

    test_acc.append(acc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(1,15),train_acc,color='blue', linestyle='dashed',

                      marker='o',markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(1,15),test_acc,color='red',linestyle='dashed', marker='o',

markerfacecolor='blue', markersize=5)

plt.legend( [train_score, test_score],["Train Accuracy", "Test Accuracy"])

plt.title('Accuracy vs K neighbors')

plt.xlabel('K neighbors')

plt.ylabel('Accuracy')
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

knn.score(X_val,y_val)

#not completed
y_pred_KNN = knn.predict(X_val)

cfm = confusion_matrix(y_val, y_pred_KNN, labels = [0,1,2])

print(cfm)

print("True Positives of Class 0: " +str(cfm[0][0]))

print("False Positives of Class 0 wrt Class 1: "+str(cfm[1][0])) # Predicted as 0 but actually print"False Positives of Class 0 wrt Class 2: ", cfm[2][0]

print("False Negatives of Class 0 wrt Class 1: "+str(cfm[0][1])) # Precited as 1 but actually print"False Negatives of Class 0 wrt Class 2: ", cfm[0][2]

from sklearn.linear_model import LogisticRegression
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
rf = RandomForestClassifier(n_estimators=13, random_state = 42)

rf.fit(X_train, y_train)

rf.score(X_val,y_val)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

y_pred_RF = rf.predict(X_val)

confusion_matrix(y_val, y_pred_RF)
print(classification_report(y_val, y_pred_RF))
param_grid = { 

    'n_estimators': [350,400,450,500,600],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [16,17,19 ,20,22,24,25],

    'criterion' :['gini', 'entropy']

}
from sklearn.metrics import make_scorer

from sklearn.metrics import f1_score

from sklearn.model_selection import GridSearchCV

#rf_temp = RandomForestClassifier(n_estimators =300) #Initialize the classifier 

#parameters = {'max_depth':[3, 5, 8, 10],'min_samples_split':[2, 3, 4, 5]} #Dictionary 

scorer = make_scorer(f1_score, average = 'micro') #Initialize the scorer using 

#CV_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5, verbose=20,n_jobs=-1)

grid_obj = GridSearchCV(estimator=rf, param_grid=param_grid,cv=5,verbose=20,n_jobs=-1) #Initialize a GridSearchCV 

grid_fit = grid_obj.fit(X_train, y_train) #Fit the gridsearch object with X_train,

best_rf = grid_fit.best_estimator_ #Get the best estimator. For this, check documentation 

print(grid_fit.best_params_)
#rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 500, max_depth=22, criterion='entropy')

rf_best = RandomForestClassifier(random_state=42, max_features='auto',n_estimators = 500, max_depth = 19,criterion='entropy')

rf_best.fit(X_train, y_train)

rf_best.score(X_val,y_val)
y_pred_RF_best = rf_best.predict(X_val)

confusion_matrix(y_val, y_pred_RF_best)
print(classification_report(y_val, y_pred_RF_best))
test = pd.read_csv("../input/Test_data.csv")

test_orig = test

test.head()

test_drop = test.drop(columns=['FileName'],axis=1)

test_drop.info()
#testdata_orig.drop(columns=['1808'],axis=1)

test_drop1 = test_drop.drop(columns=['Unnamed: 1809'],axis=1)

test_drop1.info()

#test_drop1 = test_drop.drop(columns=['FileName'],axis=1)

#test_drop1.head()
#test_drop1.info()

from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(test_drop1)

X_test = pd.DataFrame(np_scaled)

np_scaled_val = min_max_scaler.transform(X_val)

X_test_val = pd.DataFrame(np_scaled_val)

X_test.head()
final = rf_best.predict(X_test)
final2 = pd.DataFrame({"FileName":test_orig["FileName"],"Class":final})

final2.to_csv("Assignment_submission.csv",index = False)

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

create_download_link(final2)