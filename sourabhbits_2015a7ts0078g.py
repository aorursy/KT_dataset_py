import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.utils import resample

from sklearn.metrics import fbeta_score

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn import preprocessing
# load the data frame to see at a glance it's relevant variables

#data_frame = pd.read_csv('train.csv') # load the data frame

data_frame = pd.read_csv('../input/dataset.csv')

data_frame.head() # see it's initial columns



# load the testinng data simulatneously

data_frame_test = pd.read_csv('test.csv')
data_frame['Class'].unique().tolist() # find the number of classes
data_frame['Class'].value_counts()
data_frame_majority = data_frame[data_frame.Class==0]

data_frame_minority = data_frame[data_frame.Class==1]
data_frame_minority_unsampled = resample(data_frame_minority,replace=True,n_samples=93708,random_state=123)

data_frame_sampled = pd.concat([data_frame_majority,data_frame_minority_unsampled])
data_frame_sampled.Class.value_counts()
raw_train_data = data_frame_sampled.iloc[:,1:-1] # extract all the relevant training variables from the data frame

train_y = data_frame_sampled.iloc[:,-1] # extract the class labels from the data frame



raw_test_data = data_frame_test.iloc[:,1:]
# replaceing every '?' value in the data frame with the NaN value

list_of_cols = raw_train_data.columns.tolist()

for col in list_of_cols :

    if str(raw_train_data[col].dtype) == 'object' :

        #print('col {}'.format(col))

        raw_train_data[col].replace({'?' : np.nan},inplace=True)

    if str(raw_test_data[col].dtype) == 'object' :

        raw_test_data[col].replace({'?' : np.nan},inplace=True)
# now we need to remove the NaN values

raw_train_data.fillna(method='ffill',inplace=True)

raw_train_data.fillna(method='bfill',inplace=True)



# doing the same for test data

raw_test_data.fillna(method='ffill',inplace=True)

raw_test_data.fillna(method='bfill',inplace=True)
# we need to replace every categorical attribute with a numerical value

def get_replace_dict(column,data_frame) :

    column_list = data_frame[column].unique().tolist()

    column_dict = dict()

    for index, attribute in enumerate(column_list) :

        column_dict[attribute] = index

    return column_dict
list_of_cols = raw_train_data.columns.tolist()

for col in list_of_cols :

    if str(raw_train_data[col].dtype) == 'object' :

        replace_dict = get_replace_dict(col,raw_train_data)

        raw_train_data.replace(replace_dict,inplace=True)

        raw_test_data.replace(replace_dict,inplace=True)
f, ax = plt.subplots(figsize=(15, 16))

corr = raw_train_data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
cols_to_remove_set = list()

cols_list_to_check = corr.columns.tolist()

# finding cols that are co-related

for i in range(len(cols_list_to_check)-1) :

    for j in range(i+1,len(cols_list_to_check)) :

        corr_score = corr[cols_list_to_check[i]][cols_list_to_check[j]]

        if corr_score > 0.7 and (cols_list_to_check[i] not in cols_to_remove_set):

            cols_to_remove_set.append(cols_list_to_check[i])

cols_to_remove_list = list(cols_to_remove_set)

raw_train_data.drop(cols_to_remove_list,1,inplace=True)

raw_test_data.drop(cols_to_remove_list,1,inplace=True)
raw_test_data.replace({'D36' : 0},inplace=True)
X = raw_train_data

X_test = raw_test_data

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, train_y, test_size=0.20, random_state=42)
#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X_train)

X_train = pd.DataFrame(np_scaled)

np_scaled_val = min_max_scaler.transform(X_val)

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
print(X_test.shape)
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
rf = RandomForestClassifier(n_estimators=15,random_state = 42)

rf.fit(X_train, y_train)

rf.score(X_val,y_val)

preds = rf.predict(X_val)

preds.shape

fbeta_score(preds,y_val,beta=1) # fbeta score for Random Forest Classifier
y_pred_RF = rf.predict(X_val)

confusion_matrix(y_val, y_pred_RF)
print(classification_report(y_val, y_pred_RF))
test_preds_rf = rf.predict(X_test)

submit_data_frame = pd.DataFrame({'ID' : data_frame_test['ID'].values.tolist(),'Class' : test_preds_rf})

submit_data_frame.to_csv('submition.csv',index=False)
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
dTree = DecisionTreeClassifier(max_depth=12, random_state = 42)

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



create_download_link(data_frame)