import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import os
print(os.listdir("../input"))
pd.set_option('display.max_columns', 150)
tr = pd.read_csv('../input/train.csv')
ts = pd.read_csv('../input/test.csv')
tr.head()
ts.head()
print(tr.shape)
print(ts.shape)
print(tr.columns.values)
print('\n')
print(ts.columns.values)
ts['Target']=5
df = pd.concat([tr,ts],axis=0)
df.describe(include='all')
df.isna().sum()
#plt.figure(figsize=(18,6))
#sns.countplot(x='dependency',data=tr)
df['Vol']=1
pd.pivot_table(df,values='Vol',columns='dependency',aggfunc='sum')
pd.pivot_table(df,values='Vol',columns='edjefe',aggfunc='sum')
pd.pivot_table(df,values='Vol',columns='edjefa',aggfunc='sum')
df['dependency_f']=np.where(df['dependency']!='yes','no','yes')
df['dependency_f'].value_counts()
df['edjefe_f']=np.where(df['edjefe']!='yes','no','yes')
df['edjefe_f'].value_counts()
df['edjefa_f']=np.where(df['edjefa']!='yes','no','yes')
df['edjefa_f'].value_counts()
df.drop(columns=['dependency','edjefa','edjefe','v2a1','v18q1','rez_esc'],inplace=True)
sns.set_context('notebook')
sns.set_style('darkgrid')
print(df['SQBmeaned'].describe(include='all'))
df['SQBmeaned'].plot.hist(bins=30)
plt.grid(alpha=0.25)
print(df['meaneduc'].describe(include='all'))
df['meaneduc'].plot.hist(bins=30)
plt.grid(alpha=0.25)
df['SQBmeaned'].fillna(value=np.mean(df['SQBmeaned']),inplace=True)
df['meaneduc'].fillna(value=np.mean(df['meaneduc']),inplace=True)
df.isna().sum()
df.describe(include='all')
print(df['edjefa_f'].value_counts())
print(df['elimbasu4'].value_counts())
df.drop(columns=['Id','r4h3','r4m3','r4t3','paredother','pisoother','techootro','energcocinar4','elimbasu4','elimbasu5','elimbasu6','epared3','etecho3','eviv3','male','estadocivil7',
                'parentesco12','idhogar','instlevel9','tipovivi5','lugar6','area2','Vol'],inplace=True)
df['edjefe_f']=np.where(df['edjefe_f']=='yes',1,0)
df['edjefa_f']=np.where(df['edjefa_f']=='yes',1,0)
df['dependency_f']=np.where(df['dependency_f']=='yes',1,0)
sm=df.describe(include='all')
sm.iloc[7,:].sort_values(ascending=False).head(30).index
from sklearn.preprocessing import MinMaxScaler
dfs=MinMaxScaler().fit_transform(df[['agesq', 'SQBage', 'SQBmeaned', 'SQBedjefe', 'SQBescolari',
       'SQBhogar_total', 'SQBovercrowding', 'SQBhogar_nin', 'age',
       'SQBdependency', 'meaneduc', 'escolari', 'tamviv', 'rooms',
       'hogar_total', 'hhsize', 'overcrowding', 'tamhog', 'qmobilephone',
       'r4t2', 'hogar_nin', 'r4t1', 'bedrooms', 'hogar_adul', 'r4h2', 'r4m2',
       'r4m1', 'r4h1', 'hogar_mayor']])
dfsc = pd.DataFrame(data=dfs,columns=['agesq', 'SQBage', 'SQBmeaned', 'SQBedjefe', 'SQBescolari',
       'SQBhogar_total', 'SQBovercrowding', 'SQBhogar_nin', 'age',
       'SQBdependency', 'meaneduc', 'escolari', 'tamviv', 'rooms',
       'hogar_total', 'hhsize', 'overcrowding', 'tamhog', 'qmobilephone',
       'r4t2', 'hogar_nin', 'r4t1', 'bedrooms', 'hogar_adul', 'r4h2', 'r4m2',
       'r4m1', 'r4h1', 'hogar_mayor'])
dfsc.describe()
g = dfsc.plot.box(figsize=(20,4))
for item in g.get_xticklabels():
    item.set_rotation(60)
df.drop(columns=['agesq', 'SQBage', 'SQBmeaned', 'SQBedjefe', 'SQBescolari',
       'SQBhogar_total', 'SQBovercrowding', 'SQBhogar_nin', 'age',
       'SQBdependency', 'meaneduc', 'escolari', 'tamviv', 'rooms',
       'hogar_total', 'hhsize', 'overcrowding', 'tamhog', 'qmobilephone',
       'r4t2', 'hogar_nin', 'r4t1', 'bedrooms', 'hogar_adul', 'r4h2', 'r4m2',
       'r4m1', 'r4h1', 'hogar_mayor'],inplace=True)
dfsc.index=df.index
print(df.shape)
print(dfsc.shape)
ndf=pd.concat([df,dfsc],axis=1)
print(ndf.shape)
ndf.head()
ndf['Target'].value_counts()
dw = ndf[ndf['Target']!=5]
print(dw.shape)
dw['Target'].value_counts()
plt.figure(figsize=(10,8))
sns.heatmap(dw.corr(),cmap='coolwarm')
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import scikitplot as skplt
rfc = DecisionTreeClassifier(criterion='entropy',class_weight='balanced',random_state=1234)
rfc.fit(dw.drop(columns='Target'),dw['Target'])
fi = pd.DataFrame(data=rfc.feature_importances_,index=dw.drop(columns='Target').columns,columns=['Importance'])
fi.sort_values(by='Importance',ascending=False).plot.bar(figsize=(20,4))
plt.grid(alpha=0.25)
fi.sort_values(by='Importance',ascending=False).head(15).index
feats =fi.sort_values(by='Importance',ascending=False).head(15).index
from sklearn.model_selection import train_test_split
g = dw[feats].plot.box(figsize=(18,4))
for item in g.get_xticklabels():
    item.set_rotation(60)
X_train, X_test, y_train, y_test = train_test_split(dw[feats],dw['Target'], test_size=0.20, random_state=1234)
model = BaggingClassifier(n_estimators=2500,random_state=1234,warm_start=False,verbose=1,oob_score=True)
model.fit(X_train,y_train)
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
prds = model.predict(X_test)
print(classification_report(y_test,prds))
skplt.metrics.plot_confusion_matrix(y_test,prds,normalize=True)
#cvs= cross_val_score(model,dw[feats],dw['Target'],cv=5,scoring='f1_macro')
#print(cvs)
#print(np.mean(cvs))
#print(np.median(cvs))
ds = ndf[ndf['Target']==5]
ds.head()
ds.drop(columns='Target',inplace=True)
print(ds.shape)
bagprds = model.predict(ds[feats])
pdf = pd.DataFrame(data=bagprds,columns=['Target'])
pdf.index = ts.index
gsub = pd.concat([ts['Id'],pdf],axis=1)
gsub.head()
sns.countplot(gsub['Target'])
plt.grid()
gsub.to_csv('bag_model_submit.csv',index =False,index_label=False)
sns.countplot(ndf['Target'])
k = dw.drop('Target',axis=1).plot.box(figsize=(40,10))
for item in k.get_xticklabels():
    item.set_rotation(60)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
xgbc = XGBClassifier(n_estimators=7500,random_state=1234,max_depth=7)
xgbc.fit(X_train,y_train)
print(classification_report(y_test,xgbc.predict(X_test)))
skplt.metrics.plot_confusion_matrix(y_test,xgbc.predict(X_test),normalize=True)
vc = VotingClassifier(estimators=[('bag', model), ('xgb', xgbc)], voting='hard')
vc.fit(X_train,y_train)
print(classification_report(y_test,vc.predict(X_test)))
skplt.metrics.plot_confusion_matrix(y_test,vc.predict(X_test),normalize=True)
cvvs= cross_val_score(vc,dw[feats],dw['Target'],cv=5,scoring='f1_macro')
print(cvvs)
print("Average Cross Validation Score: ",np.mean(cvvs))
print("Median Cross Validation Score: ",np.median(cvvs))
bagprdss = vc.predict(ds[feats])
pdfs = pd.DataFrame(data=bagprdss,columns=['Target'])

pdfs.index = ts.index
gsubs = pd.concat([ts['Id'],pdfs],axis=1)
gsubs.head()
sns.countplot(gsubs['Target'])
plt.grid()

gsubs.to_csv('voter_submit.csv',index =False,index_label=False)