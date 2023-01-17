import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
data_o = pd.read_csv('../input/train.csv' , sep=',')
data = data_o

data_test_o = pd.read_csv('../input/test.csv' , sep=',')
data_test = data_test_o

data.info()
import seaborn as sns
f, ax = plt.subplots(figsize=(20, 15))
corr = data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot = True);
data = data.drop(['ID','Worker Class','Enrolled','MIC','MOC','MLU','Reason','Area','State','MSA','REG','MOVE','Live','PREV','Teen','COB FATHER','COB MOTHER','COB SELF','Fill','Hispanic','Detailed','Vet_Benefits'], 1)
data_test = data_test.drop(['ID','Worker Class','Enrolled','MIC','MOC','MLU','Reason','Area','State','MSA','REG','MOVE','Live','PREV','Teen','COB FATHER','COB MOTHER','COB SELF','Fill','Hispanic','Detailed','Vet_Benefits'], 1)
data = data.drop(['OC','Timely Income','Weight','Own/Self','WorkingPeriod'], 1)
data_test = data_test.drop(['OC','Timely Income','Weight','Own/Self','WorkingPeriod'], 1)
y=data['Class']
X=data.drop(['Class'],axis=1)
X = pd.get_dummies(X, columns=['Schooling','Married_Life','Cast','Sex','Full/Part','Tax Status','Summary','Citizen'])
X_test = pd.get_dummies(data_test, columns=['Schooling','Married_Life','Cast','Sex','Full/Part','Tax Status','Summary','Citizen'])
X.head()
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X,y = ros.fit_resample(X,y)
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
from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X_test)
X_test = pd.DataFrame(np_scaled)
#np_scaled_val = min_max_scaler.transform(X_val)
#X_val = pd.DataFrame(np_scaled_val)
#X_train.head()
X_test.head()
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
y_pred_sagar_DT = dTree.predict(X_test)
y_pred_sagar_DT
GG_DT = y_pred_sagar_DT.tolist()
ret_DT = pd.DataFrame(GG_DT)
F_DT = pd.concat([data_test_o['ID'], ret_DT], axis=1).reindex()
F_DT = F_DT.rename(columns={0: "Class"})
F_DT['Class'] = F_DT.Class.astype(int)
F_DT.head(100)
F_DT.to_csv('submission_DT_sagar.csv', index = False,  float_format='%.f')
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html='<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(F_DT)
