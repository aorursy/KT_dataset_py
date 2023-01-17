import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data_orig = pd.read_csv("../input/data-mining-assignment-2/train.csv", sep=',')

data = data_orig
data.head()
data.info()
#CHECKING FOR DUPLCATE COLMNS

data.duplicated().sum()
#CHECKING FOR NULL COLMNS

null_columns = data.columns[data.isnull().any()]

null_columns
import seaborn as sns

f, ax = plt.subplots(figsize=(30, 24))

corr = data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
#DROPPING ID COLMN

data = data.drop(['ID'], axis = 1)
data.head()
#FINDING OUT THE OBJECT COLMNS

obj_cols=data.columns[data.dtypes=='object']

obj_cols=np.array(obj_cols)

obj_cols
#SEARCHING FOR UNIQUE VALUES IN EACH OF THE OBJECT COLMNS

col_nos=[2,11,37,44,56]

for i in col_nos:

  print(str(i)+":"+str(data["col"+str(i)].unique()))
#DROP ALL THE OBJECT COLUMNS

data2=data.drop(['col2','col11','col37','col44','col56'],axis=1)

#data2=pd.get_dummies(data2, columns=["Col189"]) #for one hot encoding

data2.head()
data2.info()
#SEPERATING X AND Y AND DROPPING Class COLMN

y=data2['Class']

X=data2.drop(['Class'],axis=1)

X.head()
#NORMALIZATION



from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X)

X_N = pd.DataFrame(np_scaled) #new dataframe is made

X_N.head()
#SPLITTING THE TRAIN AND TEST

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_N, y, test_size=0.20, random_state=42)
np.random.seed(42)
from sklearn.ensemble import RandomForestClassifier



score_train_RF = []

score_test_RF = []



for i in range(5,20,1):

    rf = RandomForestClassifier(n_estimators = 100, max_depth=i)

    rf.fit(X_train, y_train)

    sc_train = rf.score(X_train,y_train)

    score_train_RF.append(sc_train)

    sc_test = rf.score(X_test,y_test)

    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(5,20,1),score_train_RF,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(5,20,1),score_test_RF,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score,test_score],["Train Score","Test Score"])

plt.title('Fig4. Score vs. No. of Trees')

plt.xlabel('max_depth')

plt.ylabel('Score')
from sklearn.utils.class_weight import compute_class_weight

cw=compute_class_weight("balanced",[0,1,2,3],y)

print(cw)
wt_dict={0:0.77777778,1:3.80434783,2:0.79545455,3:0.83732057}

rf = RandomForestClassifier(n_estimators=1000, max_depth = 11,class_weight=wt_dict)

rf.fit(X_train, y_train)

rf.score(X_test,y_test)
rf.fit(X_train, y_train)

rf.score(X_test,y_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

y_pred_RF = rf.predict(X_test)

confusion_matrix(y_test, y_pred_RF)
print(classification_report(y_test, y_pred_RF))
#READING THE TEST DATAFRAME WHICH HAS NO CLASS LABELS

test_orig = pd.read_csv("../input/data-mining-assignment-2/test.csv", sep=',')

data_test = test_orig
data_test.head()
data_test.info()
#CHECKING FOR DUPLCATE COLMNS

print('duplicate rows',data_test.duplicated().sum())



#CHECKING FOR NULL COLMNS

null_columns = data.columns[data.isnull().any()]

print('# of null colmns',null_columns)



#DROPPING ID COLMN and ALSO MAKING THE FINAL TO BE SUBMITTED DATAFRAME

df_final=pd.DataFrame()

df_final['ID']=data_test['ID']

my_data_test = data_test.drop(['ID'], axis = 1)



#DROP ALL THE OBJECT COLUMNS

data2_test=my_data_test.drop(['col2','col11','col37','col44','col56'],axis=1)

#data2=pd.get_dummies(data2, columns=["Col189"]) #for one hot encoding



#NAMING IT AS X_unseen

X_unseen=data2_test



#NORMALIZATION



from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X_unseen)

X_unseen_N = pd.DataFrame(np_scaled) #new dataframe is made

X_unseen_N.head()
X_unseen_N.info()
y_pred=rf.predict(X_unseen_N)
y_pred.shape
df_final['Class']=y_pred
df_final.tail()
# df_final.to_csv('sub33.csv',index=False)
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

create_download_link(df_final)