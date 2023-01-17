import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

df=pd.read_csv("../input/dmassignment2/train.csv")
df.head()
df.info()
df.replace({'?':None},inplace=True)
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = df.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
df['Enrolled'].unique()
df.drop(['Weaks'],axis=1,inplace=True)
df['Schooling'].unique()
#df.drop(['PREV','Fill'],axis=1,inplace=True)
cols = ['COB FATHER','COB MOTHER','COB SELF','MIC','MOC','Worker Class','MSA','REG','MOVE']

for column in cols:

    mode = df[column].mode()[0]

    df[column] = df[column].fillna(mode)
df['COB FATHER']
# df.drop(['Gain','Loss','Stock'],axis=1,inplace=True)
#df.drop(['MIC','MOC'],axis=1,inplace=True)
df.drop(['Detailed'],axis=1,inplace=True)
#df.drop(['COB FATHER','COB MOTHER','COB SELF'],axis=1,inplace=True)
#df.drop(['MSA','REG','MOVE'],axis=1,inplace=True)
df.head()
df.info()
df['Married_Life'].unique()
Worker_Class=pd.get_dummies(df['Worker Class'],drop_first=True)
Worker_Class.head()
Married_Life=pd.get_dummies(df['Married_Life'])
Married_Life.head()
Cast=pd.get_dummies(df['Cast'])
Cast.head()
Hispanic=pd.get_dummies(df['Hispanic'],drop_first=True)
Hispanic.head()
Sex=pd.get_dummies(df['Sex'])
Sex.head()
Tax_Status=pd.get_dummies(df['Tax Status'])
Tax_Status.head()
Live=pd.get_dummies(df['Live'],drop_first=True)
Live.head()
Teen=pd.get_dummies(df['Teen'],drop_first=True)
Teen.head()
Citizen=pd.get_dummies(df['Citizen'])
Citizen.head()
Prev=pd.get_dummies(df['PREV'],drop_first=True)
Fill=pd.get_dummies(df['Fill'],drop_first=True)
COB_FATHER=pd.get_dummies(df['COB FATHER'])

COB_MOTHER=pd.get_dummies(df['COB MOTHER'])

COB_SELF=pd.get_dummies(df['COB SELF'])

MIC=pd.get_dummies(df['MIC'])

MOC=pd.get_dummies(df['MOC'])
MSA=pd.get_dummies(df['MSA'])

REG=pd.get_dummies(df['REG'])

MOVE=pd.get_dummies(df['MOVE'])

Enrolled=pd.get_dummies(df['Enrolled'])

MLU=pd.get_dummies(df['MLU'])

Reason=pd.get_dummies(df['Reason'])

Area=pd.get_dummies(df['Area'])

State=pd.get_dummies(df['State'])

Full_Part=pd.get_dummies(df['Full/Part'])

Summary=pd.get_dummies(df['Summary'])

Schooling=pd.get_dummies(df['Schooling'])
df.drop(['Worker Class','Married_Life','Hispanic','Sex','Tax Status','Live','Teen','Citizen','PREV','Fill','COB FATHER',

        'COB MOTHER','COB SELF','MIC','MOC','MSA','REG','MOVE','Enrolled','MLU','Reason','Area','State','Full/Part',

         'Summary','Schooling'],axis=1,inplace=True)
df=pd.concat([df,Worker_Class,Married_Life,Hispanic,Sex,Tax_Status,Live,Teen,Citizen,Prev,Fill,COB_FATHER,COB_MOTHER,COB_SELF

             ,MIC,MOC,MSA,REG,MOVE,Enrolled,MLU,Reason,Area,State,Full_Part,

         Summary,Schooling]

             ,axis=1)
#Changes

df.drop(['Cast'],axis=1,inplace=True)

df=pd.concat([df,Cast],axis=1)
Y=df['Class']

X=df.drop(['Class'],axis=1)
df.info()
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.20, random_state=42)
df
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
from sklearn.metrics import roc_auc_score
roc_auc_score(y_val,y_pred_NB)
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(solver = 'liblinear', C = 8, multi_class = 'ovr', random_state = 42)

lg.fit(X_train,y_train)

lg.score(X_val,y_val)

roc_auc_score(y_val,lg.predict(X_val))
y_pred_LR = lg.predict(X_val)

print(confusion_matrix(y_val, y_pred_LR))
print(classification_report(y_val, y_pred_LR))
roc_auc_score(y_val,y_pred_LR)
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
rf = RandomForestClassifier(n_estimators=11, random_state = 42)

rf.fit(X_train, y_train)

rf.score(X_val,y_val)
y_pred_RF = rf.predict(X_val)
roc_auc_score(y_val,y_pred_RF)
print(classification_report(y_val, y_pred_RF))
dff=pd.read_csv('../input/test-1/test_1.csv')
ids=dff['ID']
dff.replace({'?':None},inplace=True)
for column in dff[['ID','Age','OC','Timely Income','Stock','NOP','Own/Self','WorkingPeriod','Weight']]:

    dff[column]=dff[column].astype(float) 
for column in dff[['COB FATHER','COB MOTHER','COB SELF','MIC','MOC','Worker Class','MSA','REG','MOVE']]:

    mode = dff[column].mode()

    dff[column] = dff[column].fillna(mode)
dff.drop(['Weaks','Detailed'],axis=1,inplace=True)

Worker_Class=pd.get_dummies(dff['Worker Class'],drop_first=True)

Married_Life=pd.get_dummies(dff['Married_Life'])

Hispanic=pd.get_dummies(dff['Hispanic'],drop_first=True)

Sex=pd.get_dummies(dff['Sex'])

Tax_Status=pd.get_dummies(dff['Tax Status'])

Live=pd.get_dummies(dff['Live'],drop_first=True)

Teen=pd.get_dummies(dff['Teen'],drop_first=True)

Citizen=pd.get_dummies(dff['Citizen'])

Cast=pd.get_dummies(dff['Cast'])

Prev=pd.get_dummies(dff['PREV'],drop_first=True)

Fill=pd.get_dummies(dff['Fill'],drop_first=True)

COB_FATHER=pd.get_dummies(dff['COB FATHER'])

COB_MOTHER=pd.get_dummies(dff['COB MOTHER'])

COB_SELF=pd.get_dummies(dff['COB SELF'])

MIC=pd.get_dummies(dff['MIC'])

MOC=pd.get_dummies(dff['MOC'])

MSA=pd.get_dummies(dff['MSA'])

REG=pd.get_dummies(dff['REG'])

MOVE=pd.get_dummies(dff['MOVE'])

Enrolled=pd.get_dummies(dff['Enrolled'])

MLU=pd.get_dummies(dff['MLU'])

Reason=pd.get_dummies(dff['Reason'])

Area=pd.get_dummies(dff['Area'])

State=pd.get_dummies(dff['State'])

Full_Part=pd.get_dummies(dff['Full/Part'])

Summary=pd.get_dummies(dff['Summary'])

Schooling=pd.get_dummies(dff['Schooling'])

dff.drop(['Worker Class','Married_Life','Hispanic','Sex','Tax Status','Live','Teen','Citizen','PREV','Fill','COB FATHER',

        'COB MOTHER','COB SELF','MIC','MOC','MSA','REG','MOVE','Enrolled','MLU','Reason','Area','State','Full/Part',

         'Summary','Schooling'],axis=1,inplace=True)

dff=pd.concat([dff,Worker_Class,Married_Life,Hispanic,Sex,Tax_Status,Live,Teen,Citizen,Prev,Fill,COB_FATHER,COB_MOTHER,COB_SELF

             ,MIC,MOC,MSA,REG,MOVE,Enrolled,MLU,Reason,Area,State,Full_Part,

         Summary,Schooling]

             ,axis=1)
dff.drop(['Cast'],axis=1,inplace=True)

dff=pd.concat([dff,Cast],axis=1)
Y=df['Class']

X=df.drop(['Class'],axis=1)
pred=[]
min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X)

X = pd.DataFrame(np_scaled)

np_scaled_val = min_max_scaler.transform(dff)

dff = pd.DataFrame(np_scaled_val)

X.head()
Nb = NB()

Nb.fit(X,Y)

pred=Nb.predict(dff)
lg.fit(X,Y)

pred=lg.predict(dff)
res=pd.DataFrame(pred)

final=pd.concat([ids,res],axis=1).reindex() 

final=final.rename(columns={0:'Class'})

final.head()
final.to_csv('Final.csv',index=False)

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

    create_download_link(Final.csv) 

 

 