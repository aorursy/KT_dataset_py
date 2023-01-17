import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_csv('../input/cristiano7/data.csv',index_col=0)
plt.figure(figsize=(10,10))
sns.heatmap(df.isnull())
# sns.pairplot(df, hue='is_goal')
sns.boxplot(x='is_goal',y="remaining_min",data=df)
def append_remaining_min(dis):
    if math.isnan(dis[0]):
        if dis[1]>=2 and dis[1]<=8:
            return dis[1]
        else:
            return 4
    else:
        return dis[0]
    
df['remaining_min']=df[['remaining_min','remaining_min.1']].apply(append_remaining_min,axis=1)
sns.boxplot(x='is_goal',y="power_of_shot",data=df)
def append_power(dis):
    if math.isnan(dis[0]):
        if dis[1]>=1 and dis[1]<=4:
            return dis[1]
        else:
            return 2
    else:
        return dis[0]
        
    
df['power_of_shot']=df[['power_of_shot','power_of_shot.1']].apply(append_power,axis=1)
#Value of column knockout_match
df['knockout_match'].value_counts()
def append_ko(dis):
    if math.isnan(dis[0]):
        if dis[1]>=0 and dis[1]<=1:
            return dis[1]
        else:
            return 0
    else:
        return dis[0]
        
    
df['knockout_match']=df[['knockout_match','knockout_match.1']].apply(append_ko,axis=1)
sns.boxplot(x='is_goal',y="remaining_sec",data=df)
def append_sec(dis):
    if math.isnan(dis[0]):
        if dis[1]>=12 and dis[1]<=42:
            return dis[1]
        else:
            return 28
    else:
        return dis[0]
        
    
df['remaining_sec']=df[['remaining_sec','remaining_sec.1']].apply(append_sec,axis=1)
sns.boxplot(x='is_goal',y="distance_of_shot",data=df)
def append_dist(dis):
    if math.isnan(dis[0]):
        if dis[1]>=20 and dis[1]<=42:
            return dis[1]
        else:
            return 28
    else:
        return dis[0]
        
df['distance_of_shot']=df[['distance_of_shot',"distance_of_shot.1"]].apply(append_dist,axis=1)
def lat(st):
    if st == 'nan':
        return
    else:
        return float(st.split(',')[0])

def lng(st):
    if st == 'nan':
        return
    else:
        return float(st.split(',')[1])

df['lat']=pd.Series(map(lambda x:lat(str(x)),df.loc[:,'lat/lng']))
df['lng']=pd.Series(map(lambda x:lng(str(x)),df.loc[:,'lat/lng']))
def aw(st):
    if len(st.split()) == 3:
        return st.split()[2]

df['away']=pd.Series(map(lambda x:aw(str(x)),df['home/away']))
#list of cloumns with their index

cols=list(df.columns)

for x in cols:
    print(cols.index(x)," ",x)
sns.stripplot(df['location_x'],df['location_y'],hue=df['area_of_shot'],cmap='rainbow')
train_area=df[df['shot_basics'].isnull()==False][['location_x','location_y','shot_basics']]
test_area=df[df['shot_basics'].isnull()][['location_x','location_y','shot_basics']]

sns.stripplot(train_area['location_x'],train_area['location_y'],hue=train_area['shot_basics'],cmap='rainbow')
#filling missing values of coloumn location_x using shot_basic and area_of_shot

def locx(ax):
    ax[0]=float(ax[0])
    if math.isnan(ax[0]):
        return df[ (df['area_of_shot']==ax[1]) | (df['shot_basics']==ax[2]) ]['location_x'].mean()
    else:
        return ax[0]
    
df['location_x']= df[['location_x','area_of_shot','shot_basics']].apply(locx,axis=1)
#filling missing values of coloumn location_y using shot_basic and area_of_shot

def locy(ax):
    ax[0]=float(ax[0])
    if math.isnan(ax[0]):
        return df[ (df['area_of_shot']==ax[1])  | (df['shot_basics']==ax[2])]['location_y'].mean()
    else:
        return ax[0]
    
df['location_y']= df[['location_y','area_of_shot','shot_basics']].apply(locy,axis=1)
#filling missing values of coloumn away using match_event_id


def func2(ax):
    if ax[0] == 'nan':
        return df[df['match_event_id']==ax[1]]['away'].mode().values
    else:
        return ax[0]
    

df['away']= df[['away','match_event_id']].astype('str').apply(func2,axis=1)    
#filling missing values of coloumn date_of_game using match_event_id and match_id


def func3(ax):
    t=df[ (df['match_event_id']==ax[1]) | (df['match_id']==ax[2])]['date_of_game'].mode().values
    if ax[0] == 'nan':
        if not t:
            return 
        else:
            return t
    else:
        return ax[0]
    

df['date_of_game']= df[['date_of_game','match_event_id','match_id']].astype('str').apply(func3,axis=1)    
#filling missing values of coloumn lat(latitude) using match_event_id and match_id


def func4(ax):
    if ax[0] == 'nan':
        if len(df[df['match_event_id']==ax[1]]['lat'].mode().values)==1:
            return (df[df['match_event_id']==ax[1]]['lat'].mode().values)[0]
        else:
            return (df[df['match_event_id']==ax[1]]['lat'].mode().values)
    else:
        return ax[0]
    

df['lat']= df[['lat','match_event_id']].astype('str').apply(func4,axis=1)   
#filling missing values of coloumn lng(longitude) using match_event_id and match_id


def func5(ax):
    if ax[0] == 'nan':
        if len(df[df['match_event_id']==ax[1]]['lng'].mode().values)==1:
            return (df[df['match_event_id']==ax[1]]['lng'].mode().values)[0]
        else:
            return (df[df['match_event_id']==ax[1]]['lng'].mode().values)
    else:
        return ax[0]
    

df['lng']= df[['lng','match_event_id']].astype('str').apply(func5,axis=1)   
len(df[df.away.isnull()])
len(df[df['location_y'].isnull()])
len(df[df['location_x'].isnull()])
#filling those 5 missing values of column location_x and location_y

def rett(a,b):
    if math.isnan(a):
        return df.iloc[:,b].mean()
    else:
        return a
df['location_x']=list(map(lambda x:rett(x,1),df['location_x']))
df['location_y']=list(map(lambda x:rett(x,2),df['location_y']))
plt.figure(figsize=(22,10))
sns.heatmap(df.isnull())
train_area=df[df['area_of_shot'].isnull()==False][['location_x','location_y','area_of_shot']]
test_area=df[df['area_of_shot'].isnull()][['location_x','location_y','area_of_shot']]
sns.stripplot(df['location_x'],df['location_y'],hue=df['area_of_shot'],cmap='rainbow')
#scaling the data for KNN

from sklearn.preprocessing import StandardScaler

n=df[['location_x','location_y','area_of_shot']]

scaler = StandardScaler()

scaler.fit(n.drop('area_of_shot',axis=1))

scaled_features = scaler.transform(n.drop('area_of_shot',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=['location_x','location_y'])
#splitting the data into training and testing

newdp=pd.concat([df_feat,n['area_of_shot']],axis=1)

X_train=newdp[newdp['area_of_shot'].isnull()==False][['location_x','location_y']]
y_train=newdp[newdp['area_of_shot'].isnull()==False][['area_of_shot']]
X_test=newdp[newdp['area_of_shot'].isnull()][['location_x','location_y']]
y_test=newdp[newdp['area_of_shot'].isnull()][['area_of_shot']]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_test=list(knn.predict(X_test))
qq = list(df[df['area_of_shot'].isnull()].index)
#filling the predicted values into the dataset for the column area_of_shot


for x in qq:
    df.loc[x,'area_of_shot']=y_test[qq.index(x)]
train_area=df[df['shot_basics'].isnull()==False][['location_x','location_y','shot_basics']]
test_area=df[df['shot_basics'].isnull()][['location_x','location_y','shot_basics']]

sns.stripplot(train_area['location_x'],train_area['location_y'],hue=train_area['shot_basics'],cmap='rainbow')
#scaling the data before applying KNN

from sklearn.preprocessing import StandardScaler

n=df[['location_x','location_y','shot_basics']]

scaler = StandardScaler()

scaler.fit(n.drop('shot_basics',axis=1))

scaled_features = scaler.transform(n.drop('shot_basics',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=['location_x','location_y'])
#splitting the data into traing and testing 

newdp=pd.concat([df_feat,n['shot_basics']],axis=1)

X_train=newdp[newdp['shot_basics'].isnull()==False][['location_x','location_y']]
y_train=newdp[newdp['shot_basics'].isnull()==False][['shot_basics']]
X_test=newdp[newdp['shot_basics'].isnull()][['location_x','location_y']]
y_test=newdp[newdp['shot_basics'].isnull()][['shot_basics']]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_test=list(knn.predict(X_test))
qq = list(df[df['shot_basics'].isnull()].index)
#filling the misssing value for the column shot_basic using prediction

for x in qq:
    df.loc[x,'shot_basics']=y_test[qq.index(x)]
plt.figure(figsize=(10,7))
sns.stripplot(x='distance_of_shot',y='location_y',data=df,hue='range_of_shot')
#scaling the data for KNN

from sklearn.preprocessing import StandardScaler

n=df[['location_y','distance_of_shot','range_of_shot']]

scaler = StandardScaler()

scaler.fit(n.drop('range_of_shot',axis=1))

scaled_features = scaler.transform(n.drop('range_of_shot',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=['location_y','distance_of_shot'])
#splitting the data into train and test

newdp=pd.concat([df_feat,n['range_of_shot']],axis=1)

X_train=newdp[newdp['range_of_shot'].isnull()==False][['location_y','distance_of_shot']]
y_train=newdp[newdp['range_of_shot'].isnull()==False][['range_of_shot']]
X_test=newdp[newdp['range_of_shot'].isnull()][['location_y','distance_of_shot']]
y_test=newdp[newdp['range_of_shot'].isnull()][['range_of_shot']]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_test=list(knn.predict(X_test))
qq = list(df[df['range_of_shot'].isnull()].index)
#filling the values of column range_of_shot using prediction

for x in qq:
    df.loc[x,'range_of_shot']=y_test[qq.index(x)]
plt.figure(figsize=(10,7))
sns.stripplot(x='match_id',y='game_season',data=df)
n=df[['match_id','game_season']]
X_train=n[n['game_season'].isnull()==False][['match_id']]
y_train=n[n['game_season'].isnull()==False][['game_season']]
X_test=n[n['game_season'].isnull()][['match_id']]
y_test=n[n['game_season'].isnull()][['game_season']]
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_test=list(knn.predict(X_test))
qq = list(df[df['game_season'].isnull()].index)
#filling the knn predictions into the column game_season

for x in qq:
    df.loc[x,'game_season']=y_test[qq.index(x)]
plt.figure(figsize=(22,10))
sns.heatmap(df.isnull())
#moving on to next step
fd=df[df['type_of_shot'].isnull()==False]

sns.stripplot(fd['area_of_shot'],fd['location_x'],hue=fd['type_of_shot'],cmap='rainbow')
#we can evidentlly see a pattern of relationship of type_of_combined_shot with (location_x,location_y)
ff=df[df['type_of_combined_shot'].isnull()==False]

sns.stripplot(ff['location_y'],ff['location_x'],hue=ff['type_of_combined_shot'],cmap='rainbow')
# value counts for the column type_of_combined_shot
ff['type_of_combined_shot'].value_counts()
from sklearn.preprocessing import StandardScaler

n=df[['location_x','location_y','type_of_combined_shot']]

scaler = StandardScaler()

scaler.fit(n.drop('type_of_combined_shot',axis=1))

scaled_features = scaler.transform(n.drop('type_of_combined_shot',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=['location_x','location_y'])
newdp=pd.concat([df_feat,n['type_of_combined_shot']],axis=1)

X_train=newdp[newdp['type_of_combined_shot'].isnull()==False][['location_x','location_y']]
y_train=newdp[newdp['type_of_combined_shot'].isnull()==False][['type_of_combined_shot']]
X_test=newdp[newdp['type_of_combined_shot'].isnull()][['location_x','location_y']]
y_test=newdp[newdp['type_of_combined_shot'].isnull()][['type_of_combined_shot']]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train,y_train)
y_test=list(knn.predict(X_test))
qq = list(df[df['type_of_combined_shot'].isnull()].index)
#filling in the predicted values for the column type_of_combined_shot

for x in qq:
    df.loc[x,'type_of_combined_shot']=y_test[qq.index(x)]
def aw(st):
    if len(st.split('-')) == 2:
        return st.split('-')[1]

df['type_shot']=pd.Series(map(lambda x:aw(str(x)),df['type_of_shot']))
def funbun(x):
    if x!=None:
        return int(x)
        
df['type_shot']=df['type_shot'].apply(funbun)
#plotting the new created column (tye_shot)


fd=df[df['type_shot'].isnull()==False]
plt.figure(figsize=(15,10))
sns.stripplot(fd['power_of_shot'],fd['distance_of_shot'],hue=fd['type_shot'])
plt.figure(figsize=(22,10))
sns.heatmap(df.isnull())
season= pd.get_dummies(df['game_season'],drop_first=True)
shot_area= pd.get_dummies(df['area_of_shot'],drop_first=True)
basic = pd.get_dummies(df['shot_basics'],drop_first=True)
ranges =  pd.get_dummies(df['range_of_shot'],drop_first=True)
com =  pd.get_dummies(df['type_of_combined_shot'],drop_first=True)
df.drop(['match_event_id','lat/lng','team_id', 'remaining_min.1', 'power_of_shot.1',
       'knockout_match.1', 'remaining_sec.1', 'distance_of_shot.1', 'away',
       'type_shot','team_name','home/away'],axis=1,inplace=True)
df
df=pd.concat([df[['location_x', 'location_y','remaining_sec', 'distance_of_shot']]
              ,ranges,basic,shot_area,season,df[['is_goal','shot_id_number']]],axis=1)
from keras.models import Sequential
from keras.layers import Dense,Dropout,LeakyReLU
from keras.layers import Activation, Dense
model=Sequential()
model.add(Dense(64,input_shape=(38,)))
model.add(Activation('relu'))
model.add(Dense(60))
model.add(Activation('relu'))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()
#splitting the dataset

xmos=df[df['is_goal'].isnull()==False]
ymos=df[df['is_goal'].isnull()==True]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(xmos.drop(['is_goal','shot_id_number'],axis=1),xmos[['is_goal','shot_id_number']] , test_size=0.33)
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
#fitting the model
#with 40  epoch

model.fit(X_train,y_train['is_goal'],10,40,validation_data=(X_test,y_test['is_goal']))
X_train=xmos.drop(['is_goal','shot_id_number'],axis=1)
y_train=xmos[['is_goal','shot_id_number']]
X_test=ymos.drop(['is_goal','shot_id_number'],axis=1)
y_test=ymos[['is_goal','shot_id_number']]
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
model.fit(X_train,y_train['is_goal'],10,20,validation_data=(X_test,y_test['is_goal']))
#predicting

val=pd.DataFrame(model.predict(X_test),columns=['is_goal'])
y_test.drop('is_goal',axis=1,inplace=True)
val.index=y_test.index
#filling the Y-test column

y_test=pd.concat([val,y_test['shot_id_number']],axis=1)
y_mux=pd.concat([X_test,y_test],axis=1)
x_mux=pd.concat([X_train,y_train],axis=1)
#final DataFrame
total=pd.concat([x_mux,y_mux],axis=0)