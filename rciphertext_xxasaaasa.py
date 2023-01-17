%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pandas as pd
pd.options.display.max_columns = 100

from matplotlib import pyplot as plt
import numpy as np

import seaborn as sns
sns.set(rc={'figure.figsize':(12,9)})
import pylab as plot
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss,accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
#for scaling
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print(data.shape)
data.columns
df=data
datetime=df.Dates.str.split(pat=" ",expand=True)
datetime.columns=['Date','Time']
#datetime

#------------test data-------------

df_test=test
datetime_test=df_test.Dates.str.split(pat=" ",expand=True)
datetime_test.columns=['Date','Time']
Date=datetime.Date.str.split(pat="-",expand=True)
Date.columns=['Year','Month','Day']

Time=datetime.Time.str.split(pat=":",expand=True)
Time.columns=['Hour','Minute','Second']

#------------test data-------------

Date_test=datetime_test.Date.str.split(pat="-",expand=True)
Date_test.columns=['Year','Month','Day']

Time_test=datetime_test.Time.str.split(pat=":",expand=True)
Time_test.columns=['Hour','Minute','Second']
df=pd.concat([df,Date,Time],axis=1)
#df

#-----------test data------------------

df_test=pd.concat([df_test,Date_test,Time_test],axis=1)

df=df.drop(labels=['Dates'],axis=1)

#-----------test data------------------

df_test=df_test.drop(labels=['Dates'],axis=1)

df.columns
le = preprocessing.LabelEncoder()
#y=pd.get_dummies(df.Category,columns=['Category'],prefix=" ",prefix_sep=" ",drop_first=True,)

le_res=le.fit_transform(df['Category'])
y=pd.DataFrame(le_res)
y.columns=['Category']
#y
df["rot60_X"]=(0.5) * df["Y"] + (1.732/2) * df["X"]
df["rot60_Y"]=0.5 * df["Y"] - (1.732/2) * df["X"]



df_test["rot60_X"]=(0.5) * df_test["Y"] + (1.732/2) * df_test["X"]
df_test["rot60_Y"]=0.5 * df_test["Y"] - (1.732/2) * df_test["X"]

df["rot45_X"]=0.707 * df["Y"] + 0.707 * df["X"]
df["rot45_Y"]=0.707 * df["Y"] - 0.707 * df["X"]

df_test["rot45_X"]=0.707 * df_test["Y"] + 0.707 * df_test["X"]
df_test["rot45_Y"]=0.707 * df_test["Y"] - 0.707 * df_test["X"]

df["rot30_X"]=(1.732/2) * df["Y"] + 0.5 * df["X"]
df["rot30_Y"]=(1.732/2) * df["Y"] - 0.5 * df["X"]

df_test["rot30_X"]=(1.732/2) * df_test["Y"] + 0.5 * df_test["X"]
df_test["rot30_Y"]=(1.732/2) * df_test["Y"] - 0.5 * df_test["X"]

df["radial60"]=np.sqrt(np.power(df['rot60_X'],2) + np.power(df['rot60_Y'],2))

df_test["radial60"]=np.sqrt(np.power(df_test['rot60_X'],2) + np.power(df_test['rot60_Y'],2))
df=df.drop(labels='rot60_X',axis=1)

df_test=df_test.drop(labels='rot60_X',axis=1)
df=df.drop(labels='rot60_Y',axis=1)

df_test=df_test.drop(labels='rot60_Y',axis=1)
df=df.drop(labels='Second',axis=1)

df_test=df_test.drop(labels='Second',axis=1)
df['Minute']=df['Minute'].apply(lambda x:int(x))
df['Minute']=df['Minute'].apply(lambda x : 'low' if x <31 else 'high')

df_test['Minute']=df_test['Minute'].apply(lambda x:int(x))
df_test['Minute']=df_test['Minute'].apply(lambda x : 'low' if x <31 else 'high')

df['DayOfWeek']= df['DayOfWeek'].apply(lambda x : 'WeekHigh' if x in ('Wednesday','Friday') else ('WeekMed' if x in ('Tuesday','Thursday','Saturday') else 'WeekLow'))


df_test['DayOfWeek']= df_test['DayOfWeek'].apply(lambda x : 'WeekHigh' if x in ('Wednesday','Friday') else ('WeekMed' if x in ('Tuesday','Thursday','Saturday') else 'WeekLow'))

df['Intersection']=df['Address'].apply(lambda x : 1 if '/' in x else 0)
df['Block']=df['Address'].apply(lambda x : 1 if 'Block' in x else 0)
df_test['Intersection']=df_test['Address'].apply(lambda x : 1 if '/' in x else 0)
df_test['Block']=df_test['Address'].apply(lambda x : 1 if 'Block' in x else 0)
address=pd.DataFrame(df['Address'],columns=['Address'])
address=address.Address.str.split(pat=" /",expand=True )

address.columns=['Address','Intr2']

address=address.Address.str.split(pat=" /",expand=True )
address.columns=['Address']

string=address.iloc[:,0]
string=string.str.strip()

address_fram=string.to_frame()
temp=address_fram['Address'].astype(str).str[-2:]

address=temp.to_frame()

address['Address']=address['Address'].apply(lambda x :( x if x in ("ST","AV","LN","DR","BL","HY","CT","RD","PL","PZ","TR","AL","CR","WK","EX","RW") else (("I-80" if x in ("80") else ("HWY" if x in ("WY") else ("WAY" if x in ("AY") else ("TER" if x in ("ER") else ("ALMS" if x in ("MS") else ("MAR" if x in ("AR") else ("PARK" if x in ("RK") else ("STWY" if x in ("WY") else ("VIA" if x in ("NO") else ("BLOCK")))))))))))))
df=df.drop(labels=['Address'],axis=1)
df=pd.concat([address,df],axis=1)


address_test=pd.DataFrame(df_test['Address'],columns=['Address'])
address_test=address_test.Address.str.split(pat=" /",expand=True )

address_test.columns=['Address','Intr2']

address_test=address_test.Address.str.split(pat=" /",expand=True )
address_test.columns=['Address']

string_test=address_test.iloc[:,0]
string_test=string_test.str.strip()

address_fram_test=string_test.to_frame()
temp_test=address_fram_test['Address'].astype(str).str[-2:]

address_test=temp_test.to_frame()

address_test['Address']=address_test['Address'].apply(lambda x :( x if x in ("ST","AV","LN","DR","BL","HY","CT","RD","PL","PZ","TR","AL","CR","WK","EX","RW") else (("I-80" if x in ("80") else ("HWY" if x in ("WY") else ("WAY" if x in ("AY") else ("TER" if x in ("ER") else ("ALMS" if x in ("MS") else ("MAR" if x in ("AR") else ("PARK" if x in ("RK") else ("STWY" if x in ("WY") else ("VIA" if x in ("NO") else ("BLOCK")))))))))))))
df_test=df_test.drop(labels=['Address'],axis=1)
df_test=pd.concat([address_test,df_test],axis=1)
df
df_test
Id=df['Id']
df=df.drop(['Descript','Resolution','Id'],axis=1)

#----------test data---------

Id_test=df_test['Id']
df_test=df_test.drop(['Descript','Resolution','Id'],axis=1)
le = preprocessing.LabelEncoder()

#le = preprocessing.LabelEncoder()
le_res=le.fit_transform(df['DayOfWeek'])
Day=pd.DataFrame(le_res)
Day.columns=['DayOfWeek']
df=df.drop(labels=['DayOfWeek'],axis=1)
df=pd.concat([Day,df],axis=1)

#----------test data----------

le_res_test=le.fit_transform(df_test['DayOfWeek'])
Day_test=pd.DataFrame(le_res_test)
Day_test.columns=['DayOfWeek']
df_test=df_test.drop(labels=['DayOfWeek'],axis=1)
df_test=pd.concat([Day_test,df_test],axis=1)

le_res=le.fit_transform(df['PdDistrict'])
District=pd.DataFrame(le_res)

#District=pd.get_dummies(df['PdDistrict'],drop_first=True)
District.columns=['District']
df=df.drop(labels=['PdDistrict'],axis=1)
df=pd.concat([District,df],axis=1)


le_res_test=le.fit_transform(df_test['PdDistrict'])
District_test=pd.DataFrame(le_res_test)

#District=pd.get_dummies(df['PdDistrict'],drop_first=True)
District_test.columns=['District']
df_test=df_test.drop(labels=['PdDistrict'],axis=1)
df_test=pd.concat([District_test,df_test],axis=1)



#le = preprocessing.LabelEncoder()
le_res=le.fit_transform(df['Year'])
Year=pd.DataFrame(le_res)

#District=pd.get_dummies(df['PdDistrict'],drop_first=True)
Year.columns=['Year']
df=df.drop(labels=['Year'],axis=1)
df=pd.concat([Year,df],axis=1)


#le = preprocessing.LabelEncoder()
le_res_test=le.fit_transform(df_test['Year'])
Year_test=pd.DataFrame(le_res_test)

#District=pd.get_dummies(df['PdDistrict'],drop_first=True)
Year_test.columns=['Year']
df_test=df_test.drop(labels=['Year'],axis=1)
df_test=pd.concat([Year_test,df_test],axis=1)

df_test

#le = preprocessing.LabelEncoder()
le_res=le.fit_transform(df['Month'])
Month=pd.DataFrame(le_res)

#District=pd.get_dummies(df['PdDistrict'],drop_first=True)
Month.columns=['Month']
df=df.drop(labels=['Month'],axis=1)
df=pd.concat([Month,df],axis=1)


#le = preprocessing.LabelEncoder()
le_res_test=le.fit_transform(df_test['Month'])
Month_test=pd.DataFrame(le_res_test)

#District=pd.get_dummies(df['PdDistrict'],drop_first=True)
Month_test.columns=['Month']
df_test=df_test.drop(labels=['Month'],axis=1)
df_test=pd.concat([Month_test,df_test],axis=1)


#le = preprocessing.LabelEncoder()
le_res=le.fit_transform(df['Day'])
Day=pd.DataFrame(le_res)

#District=pd.get_dummies(df['PdDistrict'],drop_first=True)
Day.columns=['Day']
df=df.drop(labels=['Day'],axis=1)
df=pd.concat([Day,df],axis=1)


#le = preprocessing.LabelEncoder()
le_res_test=le.fit_transform(df_test['Day'])
Day_test=pd.DataFrame(le_res_test)

#District=pd.get_dummies(df['PdDistrict'],drop_first=True)
Day_test.columns=['Day']
df_test=df_test.drop(labels=['Day'],axis=1)
df_test=pd.concat([Day_test,df_test],axis=1)


#le = preprocessing.LabelEncoder()
le_res=le.fit_transform(df['Hour'])
Hour=pd.DataFrame(le_res)

#District=pd.get_dummies(df['PdDistrict'],drop_first=True)
Hour.columns=['Hour']
df=df.drop(labels=['Hour'],axis=1)
df=pd.concat([Hour,df],axis=1)


#le = preprocessing.LabelEncoder()
le_res_test=le.fit_transform(df_test['Hour'])
Hour_test=pd.DataFrame(le_res_test)

#District=pd.get_dummies(df['PdDistrict'],drop_first=True)
Hour_test.columns=['Hour']
df_test=df_test.drop(labels=['Hour'],axis=1)
df_test=pd.concat([Hour_test,df_test],axis=1)


le_res=le.fit_transform(df['Minute'])
Minute=pd.DataFrame(le_res)

#District=pd.get_dummies(df['PdDistrict'],drop_first=True)
Minute.columns=['Minute']
df=df.drop(labels=['Minute'],axis=1)
df=pd.concat([Minute,df],axis=1)


le_res_test=le.fit_transform(df_test['Minute'])
Minute_test=pd.DataFrame(le_res_test)

#District=pd.get_dummies(df['PdDistrict'],drop_first=True)
Minute_test.columns=['Minute']
df_test=df_test.drop(labels=['Minute'],axis=1)
df_test=pd.concat([Minute_test,df_test],axis=1)


df["raw_radial"]=np.sqrt(np.power(df['X'],2) + np.power(df['Y'],2))

df_test["raw_radial"]=np.sqrt(np.power(df_test['X'],2) + np.power(df_test['Y'],2))
le_res=le.fit_transform(df['Category'])
cat=pd.DataFrame(le_res)
cat.columns=['Category']
df=df.drop(labels=['Category'],axis=1)
df=pd.concat([cat,df],axis=1)

df.columns
df.head()
from sklearn.cluster import KMeans
xy_scaler = StandardScaler()
#geoData = train_df.loc[:,7:9]
xy_scaler.fit(df.loc[:,['X','Y']])
xy_scaled = xy_scaler.transform(df.loc[:,['X','Y']])
kmeans = KMeans(n_clusters=26, init='k-means++')
kmeans.fit(xy_scaled);

xy_scaler_test = StandardScaler()
#geoData = train_df.loc[:,7:9]
xy_scaler_test.fit(df_test.loc[:,['X','Y']])
xy_scaled_test = xy_scaler_test.transform(df_test.loc[:,['X','Y']])
kmeans = KMeans(n_clusters=26, init='k-means++')
kmeans.fit(xy_scaled_test);
geoData = df.loc[:,['X','Y']]
df['closest_centers_f'] = kmeans.predict(geoData)
id_label=kmeans.labels_
df.loc[:,'label'] = pd.Series(kmeans.labels_)

geoData_test = df_test.loc[:,['X','Y']]
df_test['closest_centers_f'] = kmeans.predict(geoData_test)
id_label_test=kmeans.labels_
df_test.loc[:, 'label'] = pd.Series(kmeans.labels_)

le_res=le.fit_transform(df['Address'])
Address=pd.DataFrame(le_res)
Address.columns=['Address']
df=df.drop(labels=['Address'],axis=1)
df=pd.concat([Address,df],axis=1)
le_res=le.fit_transform(df_test['Address'])
Address_test=pd.DataFrame(le_res)
Address_test.columns=['Address']
df_test=df_test.drop(labels=['Address'],axis=1)
df_test=pd.concat([Address_test,df_test],axis=1)
df_test.columns
df=df[['Address', 'Minute', 'Hour', 'Day', 'Month', 'Year',
       'District', 'DayOfWeek', 'X', 'Y', 'rot45_X', 'rot45_Y', 'rot30_X',
       'rot30_Y', 'radial60', 'Intersection', 'Block', 'raw_radial',
       'closest_centers_f', 'label']]


df_test=df_test[['Address', 'Minute', 'Hour', 'Day', 'Month', 'Year', 'District',
       'DayOfWeek', 'X', 'Y', 'rot45_X', 'rot45_Y', 'rot30_X', 'rot30_Y',
       'radial60', 'Intersection', 'Block', 'raw_radial', 'closest_centers_f',
       'label']]
'''df=pd.get_dummies(df,columns=[ 'Month',
'District'],drop_first=True)
df=pd.get_dummies(df,columns=[ 'Year'],drop_first=True)
#df_test=pd.get_dummies(df_test,columns=['DayOfWeek','PdDistrict','Year','Month','Day','Hour','Minute'],drop_first=True)
'''


"""df_test=pd.get_dummies(df_test,columns=[ 'Month',
'District'],drop_first=True)
df_test=pd.get_dummies(df_test,columns=[ 'Year'],drop_first=True)"""
#df=df[['Hour', 'Day', 'Month', 'Year', 'Address',
#       'District','X','radial60','Intersection']]
#df=pd.get_dummies(df,columns=[ 'Hour'],drop_first=True)


#df_test=pd.get_dummies(df_test,columns=[ 'Hour'],drop_first=True)
#df.columns.nunique()
#df_test.columns.nunique()
#Independent Column
X=df
X.shape

#Dependent 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,shuffle=False)
'''import lightgbm as lgb
model5= lgb.LGBMClassifier(objective='multiclass')

model5.fit(X_train,y_train)
y_final=model5.predict_proba(X_test)
print (log_loss(y_test,y_final));'''
X.head()
HYPER_PARAMS = {
 
  'learning_rate': 0.02,

 'n_estimators':800,
 'max_depth': 6,
 'subsample': 0.8,
 'colsample_bytree': 0.8,
 'max_delta_step': 1,
 'objective': 'multi:softmax',
 'nthread': 4,
 'seed': 1747
 

 
 

}




model = xgb.XGBClassifier(**HYPER_PARAMS)
model.fit(X,y)


y_pred=model.predict_proba(df_test)
temp = data['Category']

le.fit_transform(temp)
le.classes_
y_pred= pd.DataFrame(y_pred, index=Id_test,columns  = le.classes_)
y_pred.to_csv("submit.csv", float_format = '%.5F')
#print (log_loss(y_test,y_pred));

"""temp = data['Category']

le.fit_transform(temp)
le.classes_
"""
# y_pred= pd.DataFrame(y_pred, index=Id_test,columns  = le.classes_)


#from sklearn.linear_model import LogisticRegression
#weight={Address:3,District:3,X:1,Day:2}
#weight={LARCENY/THEFT:35}
#classifier = LogisticRegression(penalty='l1',random_state = 0,class_weight='balanced',multi_class='multinomial', solver='saga',n_jobs=-1)
#classifier = LogisticRegression(random_state=0, penalty='l1',multi_class='multinomial', solver='saga' )
#classifier.fit(X_train[0:50000],y_train[0:50000])
#y_pred=model.predict_proba(X_test)
#print (log_loss(y_test,y_pred));
"""from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm='auto', leaf_size=60, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
knn.fit(X_train[0:100000], y_train[0:100000])

"""


#y_pred=knn.predict_proba(X_test)

#print (log_loss(y_test,y_pred));
