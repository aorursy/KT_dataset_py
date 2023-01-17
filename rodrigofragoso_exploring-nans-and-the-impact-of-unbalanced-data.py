import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm
data=pd.read_excel('/kaggle/input/covid19/dataset.xlsx')

data.head(7)
shape=data.shape

print(shape[1],'columns')

print(shape[0],'rows')
def positive_bin(x):

    if x == 'positive':

        return 1

    else:

        return 0

data['SARS-Cov-2 exam result_bin']=data['SARS-Cov-2 exam result'].map(positive_bin)
tg_values=data['SARS-Cov-2 exam result'].value_counts()

tg_values.plot.barh(color='red')

print("Negative exam results: "+"{:.2%}".format(tg_values[0]/tg_values.sum())+' ('+str(tg_values[0])+' records)')

print("Positive exam results: "+"{:.2%}".format(tg_values[1]/tg_values.sum())+'  ('+str(tg_values[1])+' records)')

print('')
data['SARS-Cov-2 exam result_Baseline']=0

print("Baseline accuracy: "+"{:.2%}".format((data['SARS-Cov-2 exam result_Baseline']==data['SARS-Cov-2 exam result_bin']).sum()/len(data['SARS-Cov-2 exam result_Baseline'])))
if data.isnull().values.any() == True:

    print('Found NaN values!!:')

else:

    print('No NaN values =):')

print(' ')



nulls=(data.isnull().sum()/len(data))*100
nulls.sort_values(ascending=False)
ax=nulls.hist(bins=90, grid=False, figsize=(10,6), color='red')

ax.set_xlabel("% of Nulls")

ax.set_ylabel("Number of variables")

print('')
pos=data[data['SARS-Cov-2 exam result_bin']==1]

neg=data[data['SARS-Cov-2 exam result_bin']==0]
if pos.isnull().values.any() == True:

    print('Found NaN values!!:')

else:

    print('No NaN values =):')

print(' ')



nulls_pos=(pos.isnull().sum().sort_values(ascending=False)/len(pos))*100

nulls_pos
ax=nulls_pos.hist(bins=80, grid=False, figsize=(10,6), color='black')

ax.set_xlabel("% of Nulls")

ax.set_ylabel("Number of variables")

print('')
if neg.isnull().values.any() == True:

    print('Found NaN values!!:')

else:

    print('No NaN values =):')

print(' ')



nulls_neg=(neg.isnull().sum().sort_values(ascending=False)/len(neg))*100

nulls_neg
ax=nulls_neg.hist(bins=80, grid=False, figsize=(10,6), color='blue')

ax.set_xlabel("% of Nulls")

ax.set_ylabel("Number of variables")

print('')
nulls.drop(['SARS-Cov-2 exam result','Patient ID','SARS-Cov-2 exam result_bin','SARS-Cov-2 exam result_Baseline'],inplace=True)
selecting_variables=nulls.loc[nulls<90]

selecting_variables
variables=selecting_variables.index.tolist()

variables.append('Patient ID')

variables.append('SARS-Cov-2 exam result_bin')
df=data[variables]

df[df['Parainfluenza 2'].notnull()].head()
import warnings

warnings.filterwarnings("ignore")



def bins(x):

    if x == 'detected' or x=='positive':

        return 1

    elif x=='not_detected' or x=='negative':

        return 0

    else:

        return x

    

for col in df.columns:

    df[col]=df[col].apply(lambda row: bins(row))
pd.set_option('display.max_columns', None)

df.describe()
variables_imputer=variables[4:18]

teste=df[variables_imputer]
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5,missing_values=np.nan)

imputer.fit(teste)

teste[:]=imputer.transform(teste)
df.drop(variables_imputer,axis=1,inplace=True)
data_final= pd.concat([teste,df],axis=1)

data_final.fillna(-1,inplace=True)

data_final.head()
X=data_final.drop(['SARS-Cov-2 exam result_bin','Patient ID'],axis=1)

y=data_final['SARS-Cov-2 exam result_bin']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,random_state=5)

print(X_train.shape, X_test.shape)
from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import log_loss, accuracy_score



resultados=[]

kf=RepeatedKFold(n_splits=10, n_repeats=1, random_state=5)

for train,valid in tqdm(kf.split(X_train)):

    

    Xtr, Xvld = X_train.iloc[train], X_train.iloc[valid]

    ytr, yvld = y_train.iloc[train], y_train.iloc[valid]

    

    rf= RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)

    rf.fit(Xtr,ytr)

    

    p=rf.predict(Xvld)

    acc=accuracy_score(yvld,p)

    resultados.append(acc)
print("Vanilla RandomForest Train accuracy: "+"{:.2%}".format(np.mean(resultados)))
from sklearn.linear_model import LogisticRegression



resultados=[]

kf=RepeatedKFold(n_splits=10, n_repeats=1, random_state=5)

for train,valid in tqdm(kf.split(X_train)):

    

    Xtr, Xvld = X_train.iloc[train], X_train.iloc[valid]

    ytr, yvld = y_train.iloc[train], y_train.iloc[valid]

    

    lr= LogisticRegression(max_iter=300)

    lr.fit(Xtr,ytr)

    

    p=lr.predict(Xvld)

    acc=accuracy_score(yvld,p)

    resultados.append(acc)
print("Vanilla Logistic Regression Train accuracy: "+"{:.2%}".format(np.mean(resultados)))
p2=rf.predict(X_test)

p2[:]=rf.predict(X_test)

acc=accuracy_score(y_test,p2)

print("Vanilla RandomForest Test accuracy: "+"{:.2%}".format(acc))

p2=lr.predict(X_test)

p2[:]=lr.predict(X_test)

acc=accuracy_score(y_test,p2)

print("Vanilla Logistic Regression Test accuracy: "+"{:.2%}".format(acc))
visual=pd.concat([X_test,y_test],axis=1)

visual['predict']=p2

visual2=visual[visual['SARS-Cov-2 exam result_bin']==visual['predict']]



print('Positive results in the test sample: ',visual[visual['SARS-Cov-2 exam result_bin']==1].shape[0])

print('Positive results correctly predicted: ',visual2[visual2['predict']==1].shape[0])

print('Positive accuracy: ',"{:.2%}".format(visual2[visual2['predict']==1].shape[0]/visual[visual['SARS-Cov-2 exam result_bin']==1].shape[0]))
sns.set(font_scale=2)
pred_prob=lr.predict_proba(X_test)



from sklearn.metrics import precision_recall_curve



scores=pred_prob[:,1]

precision, recall, thresholds = precision_recall_curve(y_test, scores)

fig, ax = plt.subplots(figsize=(10,7))

plt.plot(recall[:-1],precision[:-1],label="logistic regeression",color='red')

plt.legend(loc="center right")

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.show()
from sklearn.metrics import roc_curve



scores=pred_prob[:,1]

fpr, tpr, thresholds = roc_curve(y_test,scores)

fig, ax = plt.subplots(figsize=(10,7))

plt.plot(fpr,tpr,color='red')

plt.xlabel("False Positive Rate",size=15)

plt.ylabel("True Positive Rate",size=15)



plt.show()