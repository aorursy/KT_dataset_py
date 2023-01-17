import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.utils import resample

from sklearn.metrics import fbeta_score

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn import preprocessing
# load the data frame to see at a glonce, it's relevant variables

data_frame_benign = pd.read_csv('../input/Data Mining Assignment-3/opcode_frequency_benign.csv') # load the data frame for benign

data_frame_malware = pd.read_csv('../input/Data Mining Assignment-3/opcode_frequency_malware.csv') # load the data frame for malware



# load the testing data simulatenously

data_frame_test = pd.read_csv('Test_data.csv')

data_frame_test.head()
# display information for all the variables

data_frame_benign.info()

data_frame_malware.info()

data_frame_test.info()
data_frame_benign.fillna(method='ffill',inplace=True)

data_frame_benign.fillna(method='bfill',inplace=True)



data_frame_malware.fillna(method='ffill',inplace=True)

data_frame_malware.fillna(method='bfill',inplace=True)



data_frame_test.fillna(method='ffill',inplace=True)

data_frame_test.fillna(method='bfill',inplace=True)
# Extract the values from data frame

X_train_benign = data_frame_benign.iloc[:,1:].values

X_train_malware = data_frame_malware.iloc[:,1:].values



# Extract test data

test_x = data_frame_test.iloc[:,1:-1].values

X_test = test_x



# create the labels

y_train_benign = np.zeros(X_train_benign.shape[0])

y_train_malware = np.ones(X_train_malware.shape[0])



# Create training data

train_y = np.concatenate([y_train_benign,y_train_malware])

train_x = np.concatenate([X_train_benign,X_train_malware],axis=0)



# prepare train and validation data

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.20, random_state=42)
# Performing Min Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X_train)

X_train = pd.DataFrame(np_scaled)

np_scaled_val = min_max_scaler.transform(X_val)

X_val = pd.DataFrame(np_scaled_val)

X_val = pd.DataFrame(np_scaled_val)

print(X_test.shape)

np_scaled_test = min_max_scaler.transform(X_test)

X_test = pd.DataFrame(np_scaled_test)

print(X_test.shape)
from sklearn.naive_bayes import GaussianNB as NB

nb = NB()

nb.fit(X_train,y_train)

nb.score(X_val,y_val)

preds = nb.predict(X_val)

preds.shape

fbeta_score(preds,y_val,beta=1) # fbeta score for Naive Bayes Classifier
y_pred_NB = nb.predict(X_val)

print(confusion_matrix(y_val, y_pred_NB))
print(classification_report(y_val, y_pred_NB))
from sklearn.linear_model import LogisticRegression

lg = LogisticRegression(solver = 'lbfgs', C = 8, multi_class = 'multinomial', random_state = 42)

lg.fit(X_train,y_train)

lg.score(X_val,y_val)

preds = lg.predict(X_val)

preds.shape

fbeta_score(preds,y_val,beta=1) # fbeta score for Logistic Regression Classifier
y_pred_LG = lg.predict(X_val)

print(confusion_matrix(y_val, y_pred_LG))
print(classification_report(y_val, y_pred_LG))
from sklearn.ensemble import RandomForestClassifier

# Random Forest Classifier
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
rf = RandomForestClassifier(n_estimators=12,random_state = 42)

rf.fit(X_train, y_train)

rf.score(X_val,y_val)

preds = rf.predict(X_val)

preds.shape

fbeta_score(preds,y_val,beta=1) # fbeta score for Random Forest Classifier
y_pred_RF = rf.predict(X_val)

confusion_matrix(y_val, y_pred_RF)
print(classification_report(y_val, y_pred_RF))
test_preds_rf = rf.predict(X_test)

submit_data_frame = pd.DataFrame({'FileName' : data_frame_test['FileName'].values.tolist(),'Class' : test_preds_rf})

submit_data_frame.to_csv('submissiion.csv',index=False)
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
dTree = DecisionTreeClassifier(max_depth=8, random_state = 42)

dTree.fit(X_train,y_train)

dTree.score(X_val,y_val)

preds = dTree.predict(X_val)

preds.shape

fbeta_score(preds,y_val,beta=1) # fbeta score for Decision Tree Classifier
y_pred_DT = dTree.predict(X_val)

print(confusion_matrix(y_val, y_pred_DT))
print(classification_report(y_val, y_pred_DT))
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



create_download_link(data_frame_benign)

create_download_link(data_frame_malware)
