from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
downloaded = drive.CreateFile({'id':"1907ZpGGcxQx-ddQMk92E_dWZhSwNX9cH"})   # replace the id with id of file you want to access
downloaded.GetContentFile('train.csv')        # replace the file name with your file
downloaded = drive.CreateFile({'id':"1T8OerWn18N0lL1Oo1Gb_ayjbWX01Z-Sg"})   # replace the id with id of file you want to access
downloaded.GetContentFile('test.csv')        # replace the file name with your file
downloaded = drive.CreateFile({'id':"1ugSoizEczwmNhxwB4AeKHp8YH5lvsqR4"})   # replace the id with id of file you want to access
downloaded.GetContentFile('sample_submission.csv')        # replace the file name with your file

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
train_data=train_data.drop_duplicates()
train_data.head()
train_data.shape
train_data.info()
train_data.FinelineNumber.value_counts()
train_data['FinelineNumber']=train_data['FinelineNumber'].fillna(8228)
train_data.info()
train_data.dropna(inplace=True)
train_data.shape
train_data.head()
sns.heatmap(train_data.isna())
train_data.TripType.nunique()
train_data.TripType.unique()
plt.figure(figsize=(12,6))
sns.countplot(train_data.TripType)
plt.figure(figsize=(12,6))
sns.distplot(train_data.TripType,hist=False)
train_data.VisitNumber.nunique()
train_data.shape[0]
plt.figure(figsize=(12,6))
sns.distplot(train_data.VisitNumber)
train_data.head()

plt.figure(figsize=(12,6))
sns.countplot(train_data.Weekday)
train_data.groupby('Weekday')['Weekday'].count()
def weekday_to_num(x):
  if x=='Monday':
    return 0
  elif x=='Tuesday':
    return 1
  elif x=='Wednesday':
    return 2
  elif x=='Thursday':
    return 3
  elif x=='Friday':
    return 4
  elif x=='Saturday':
    return 5
  elif x=='Sunday':
    return 6 
train_data['weekday_num']=train_data.Weekday.apply(weekday_to_num)
train_data.head()
train_data.Weekday.unique()
fig,axs=plt.subplots(figsize=(10,6))
sns.distplot(train_data['weekday_num'],ax=axs)
ticks=list(range(0,7))
axs.set_xticks(ticks)
x_tick_label=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
axs.set_xticklabels(x_tick_label)
plt.show()

train_data.ScanCount.nunique()
train_data.ScanCount.value_counts()
train_data.min().ScanCount
train_data.max().ScanCount
plt.figure(figsize=(12,8))
sns.distplot(train_data.ScanCount,bins=39)
plt.figure(figsize=(10,6))
sns.distplot(train_data[train_data.ScanCount<=10].ScanCount)

plt.figure(figsize=(10,6))
sns.distplot(train_data[train_data.ScanCount>10].ScanCount,bins=4)

plt.figure(figsize=(10,6))
sns.countplot(train_data[train_data.ScanCount<=10].ScanCount)
plt.figure(figsize=(10,6))
sns.countplot(train_data[train_data.ScanCount>10].ScanCount)
train_data.head()
train_data.DepartmentDescription.nunique()	
train_data.DepartmentDescription.value_counts()	
plt.figure(figsize=(12,8))
sns.countplot(x='DepartmentDescription',data=train_data)
plt.xticks(rotation=90)
plt.show()
train_data.groupby('DepartmentDescription')['DepartmentDescription'].count()

department_list=list(train_data['DepartmentDescription'].unique())
department_enumerate=list(enumerate(department_list))
department_dict={v:k for k,v in department_enumerate}
department_dict.values()
def department_num(x):
  return department_dict[x]
  
train_data['Department_num']=train_data['DepartmentDescription'].apply(department_num)
train_data.head()
fig,axs=plt.subplots(figsize=(18,8))
sns.distplot(train_data['Department_num'],bins=68,ax=axs)
ticks=list(range(0,69))
axs.set_xticks(ticks)
x_tick_label=train_data.DepartmentDescription.unique()	
axs.set_xticklabels(x_tick_label)
plt.xticks(rotation=90)
plt.show()


train_data.nunique().FinelineNumber
plt.figure(figsize=(10,6))
sns.distplot(train_data.FinelineNumber,hist=False)
train_data.Upc.nunique()
train_data.shape
plt.figure(figsize=(8,6))
sns.distplot(train_data.Upc,hist=False)

train_data.head()
# seeing for each VisitNumber how many products were purchased based on Upc number of product purchased
products_per_visit=train_data.groupby(['VisitNumber'])['Upc'].count()
products_per_visit_dict=dict(products_per_visit)
train_data['num_of_products_for_VisitNumber']=train_data['VisitNumber'].apply(lambda x:products_per_visit_dict.get(x,0))
# train_data.drop(columns=['num_of_products'],inplace=True)
train_data.head()
train_data.num_of_products_for_VisitNumber.nunique()
plt.figure(figsize=(12,6))
sns.distplot(train_data.num_of_products_for_VisitNumber,bins=99)
plt.figure(figsize=(12,6))
sns.distplot(np.log(train_data.num_of_products_for_VisitNumber),hist=False)
train_data.head()
sns.jointplot(y='num_of_products_for_VisitNumber',x='weekday_num',data=train_data)
plt.figure(figsize=(8,8))
sns.boxplot(x='weekday_num',y='num_of_products_for_VisitNumber',data=train_data)
train_data.FinelineNumber.nunique()
train_data.Department_num.unique()
groupby_dept=train_data.groupby(['Department_num'])
fineline_dict={}
for i in range(68):
  gr=groupby_dept.get_group(i)
  c=gr['FinelineNumber'].count()
  un=gr['FinelineNumber'].nunique()
  #print(f"group: {i}, unique FinelineNumber: {un}")
  fineline_dict[i]=un


sns.jointplot(y='num_of_products_for_VisitNumber',x='TripType',data=train_data[train_data['TripType']<900])
plt.figure(figsize=(20,8))
sns.boxplot(x='TripType',y='num_of_products_for_VisitNumber',data=train_data)
train_data.head()
plt.figure(figsize=(20,8))
sns.boxplot(x='TripType',y='Department_num',data=train_data)
weekday_num_of_products=dict(train_data.groupby('weekday_num')['Upc'].count())
train_data['num_of_products_for_weekday']=train_data['weekday_num'].apply(lambda x:weekday_num_of_products.get(x))
train_data.head()
plt.figure(figsize=(20,8))
sns.boxplot(x='TripType',y='num_of_products_for_weekday',data=train_data)
train_data.drop(columns=['num_of_products_for_weekday'],inplace=True)
Department_num_of_products=dict(train_data.groupby('Department_num')['Upc'].count())
train_data['num_of_products_for_department']=train_data['Department_num'].apply(lambda x:Department_num_of_products.get(x))
plt.figure(figsize=(20,8))
sns.boxplot(x='TripType',y='num_of_products_for_department',data=train_data)
train_data.FinelineNumber.nunique()

Fineline_num_of_products=dict(train_data.groupby('FinelineNumber')['Upc'].count())
train_data['num_of_products_for_fineline']=train_data['FinelineNumber'].apply(lambda x:Fineline_num_of_products.get(x))
plt.figure(figsize=(20,8))
sns.boxplot(x='TripType',y='num_of_products_for_fineline',data=train_data)
train_data.drop(columns=['num_of_products_for_fineline'],inplace=True)
train_data.head()
train_data.Weekday.value_counts()
train_data.DepartmentDescription.value_counts()
train_data.head()
one_hot_encoded_weekday=pd.get_dummies(train_data['Weekday'],drop_first=False)
one_hot_encoded_weekday.head()
train_data=pd.concat([train_data,one_hot_encoded_weekday],axis=1)
train_data.head()

Y=train_data.TripType
X=train_data.drop(columns=['TripType'])
X['FinelineCat']=pd.cut(X['FinelineNumber'],bins=50,labels=False)
X.head()
Y.head()
train_data.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,stratify=Y,test_size=0.2)
x_train,x_cv,y_train,y_cv=train_test_split(x_train,y_train,stratify=y_train,test_size=0.2)

print(x_train.shape,x_cv.shape,x_test.shape)
print(y_train.shape,y_cv.shape,y_test.shape)
x_train_res=x_train.copy()
x_train_res['class']=y_train.values
y_train.unique()
x_train_res.head()
#function to get lookup dictionary based on train data only
from tqdm import tqdm
def get_lookup_dict(alpha,feature):
  value_count = x_train_res[feature].value_counts()
  lookup_dict = dict()
  for i, denominator in tqdm(value_count.items()):
    vec = []
    for k in y_train.unique():
      cls_cnt = x_train_res.loc[(x_train_res['class']==k) & (x_train_res[feature]==i)]
      vec.append((cls_cnt.shape[0] + alpha*10)/ (denominator + 90*alpha))
    
    lookup_dict[i]=vec
  
  return lookup_dict


from tqdm import tqdm
def get_encoded_feature(alpha,feature,df,lookup_dict):
  #lookup_dict=get_lookup_dict(alpha,feature,df)
  value_count = x_train_res[feature].value_counts()
  gv_fea = []
  for index, row in tqdm(df.iterrows()):

    if row[feature] in dict(value_count).keys():
        gv_fea.append(lookup_dict[row[feature]])
    else:
        gv_fea.append([1/38]*38)
#           gv_fea.append([-1,-1,-1,-1,-1,-1,-1,-1,-1])
  
  return gv_fea
lookup_dict_DD=get_lookup_dict(1,'DepartmentDescription')
alpha = 1
train_department_feature_responseCoding=np.array(get_encoded_feature(alpha, "DepartmentDescription", x_train,lookup_dict_DD))
test_department_feature_responseCoding=np.array(get_encoded_feature(alpha, "DepartmentDescription", x_test,lookup_dict_DD))
cv_department_feature_responseCoding=np.array(get_encoded_feature(alpha, "DepartmentDescription", x_cv,lookup_dict_DD))
train_department_feature_responseCoding.shape
test_department_feature_responseCoding.shape
cv_department_feature_responseCoding.shape
train_data.ScanCount.nunique()
lookup_dict_SS=get_lookup_dict(1,'ScanCount')
train_ScanCount_feature_responseCoding=np.array(get_encoded_feature(alpha, "ScanCount", x_train,lookup_dict_SS))
test_ScanCount_feature_responseCoding=np.array(get_encoded_feature(alpha, "ScanCount", x_test,lookup_dict_SS))
cv_ScanCount_feature_responseCoding=np.array(get_encoded_feature(alpha, "ScanCount", x_cv,lookup_dict_SS))
train_ScanCount_feature_responseCoding.shape
test_ScanCount_feature_responseCoding.shape
cv_ScanCount_feature_responseCoding.shape
train_data.FinelineNumber.nunique()
train_data.FinelineNumber
train_data['FinelineCat']=pd.cut(train_data['FinelineNumber'],bins=50,labels=False)
plt.figure(figsize=(9,6))
sns.distplot(train_data['FinelineCat'])
plt.figure(figsize=(20,8))
sns.boxplot(x='TripType',y='FinelineCat',data=train_data)
lookup_dict_FNC=get_lookup_dict(1,'FinelineCat')
train_FinelineCat_feature_responseCoding=np.array(get_encoded_feature(alpha, "FinelineCat", x_train,lookup_dict_FNC))
test_FinelineCat_feature_responseCoding=np.array(get_encoded_feature(alpha, "FinelineCat", x_test,lookup_dict_FNC))
cv_FinelineCat_feature_responseCoding=np.array(get_encoded_feature(alpha, "FinelineCat", x_cv,lookup_dict_FNC))
train_FinelineCat_feature_responseCoding.shape
test_FinelineCat_feature_responseCoding.shape
cv_FinelineCat_feature_responseCoding.shape
x_train.shape
x_train.head()
x_train.drop(columns=['Weekday','Upc','DepartmentDescription','FinelineNumber','weekday_num','Department_num','FinelineCat'],inplace=True)
x_train.head()
x_train.values
x_tr=np.hstack((x_train.values,train_ScanCount_feature_responseCoding,train_department_feature_responseCoding,train_FinelineCat_feature_responseCoding,))
x_tr.shape
y_train.shape
x_test.drop(columns=['Weekday','Upc','DepartmentDescription','FinelineNumber','weekday_num','Department_num','FinelineCat'],inplace=True)
x_test.head()
x_te=np.hstack((x_test.values,test_ScanCount_feature_responseCoding,test_department_feature_responseCoding,test_FinelineCat_feature_responseCoding,))
x_cv.drop(columns=['Weekday','Upc','DepartmentDescription','FinelineNumber','weekday_num','Department_num','FinelineCat'],inplace=True)
x_cv=np.hstack((x_cv.values,cv_ScanCount_feature_responseCoding,cv_department_feature_responseCoding,cv_FinelineCat_feature_responseCoding,))
x_cv.shape
y_cv.shape
#np.save('x_train_final',x_tr)
#np.save('x_test_final',x_te)
#np.save('x_cv_final',x_cv)
#np.save('y_train_final',y_train.values)
#np.save('y_test_final',y_test.values)
#np.save('y_cv_final',y_cv.values)
from google.colab import drive
drive.mount('/content/drive')

from scipy import sparse
x_cv=sparse.csr_matrix(np.load('/content/drive/My Drive/x_cv_final.npy'))

x_test=sparse.csr_matrix(np.load('/content/drive/My Drive/x_test_final.npy'))
x_train=sparse.csr_matrix(np.load('/content/drive/My Drive/x_train_final.npy'))
y_cv=np.load('/content/drive/My Drive/y_cv_final.npy')
y_test=np.load('/content/drive/My Drive/y_test_final.npy')
y_train=np.load('/content/drive/My Drive/y_train_final.npy')
y_train.shape[0]+y_cv.shape[0]
x_train.shape[0]+x_cv.shape[0]
print(x_train.shape,x_cv.shape,x_test.shape)
print(y_train.shape,y_cv.shape,y_test.shape)
408621+102156
408621+102156
from scipy.sparse import vstack

x_train=vstack((x_train,x_cv))
x_train.shape
y_train=np.vstack((y_train.reshape(-1,1),y_cv.reshape(-1,1)))
y_train.shape
y_train=y_train.astype('int')
y_train.dtype
random_grid = {'max_depth': [None],
               'min_samples_leaf': [100,1000],
               'min_samples_split': [100,1000],
               'n_estimators': [100,500]}
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, cv = 3, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(x_tr, y_train)
rf_random.best_params_

rf_random.best_estimator_
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV

r_cfl=RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=100,min_samples_split=100, random_state=42,n_jobs=-1)
r_cfl.fit(x_train,y_train)

pred=r_cfl.predict(x_cv)
sig_clf = CalibratedClassifierCV(r_cfl, method="sigmoid")
sig_clf.fit(x_train, y_train)
predict_y = sig_clf.predict_proba(x_cv)
loss=log_loss(y_cv, predict_y, labels=r_cfl.classes_, eps=1e-15)
loss
#test_data_len = X_test.shape[0]
cv_data_len = x_cv.shape[0]

# we create a output array that has exactly same size as the CV data
cv_predicted_y = np.zeros((cv_data_len,38))
for i in range(cv_data_len):
    rand_probs = np.random.rand(1,38)
    cv_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])
print("Log loss on Cross Validation Data using Random Model",log_loss(y_cv,cv_predicted_y, eps=1e-15))
import pickle
# save the model to disk
filename = 'rf_cv_log_loss_1.85.sav'
pickle.dump(r_cfl, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)
filename = 'rf_calibrated_cv_log_loss_1.85.sav'
pickle.dump(sig_clf, open(filename, 'wb'))