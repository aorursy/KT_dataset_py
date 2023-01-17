import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv('../input/metro-bike-share-trip-data.csv')

df.head(10)
df.shape
df.info()
df.isna().sum()
df.drop(columns=['Starting Lat-Long',

                 'Ending Lat-Long',

                 'Neighborhood Councils (Certified)',

                 'Council Districts',

                 'Zip Codes',

                 'LA Specific Plans',

                 'Precinct Boundaries',

                 'Census Tracts'],

       inplace=True)

df.Duration=df.Duration/60
df.head()
df.set_index('Trip ID', inplace=True)

df.dropna(inplace=True)

df['Start Time']= pd.to_datetime(df['Start Time'])

df['End Time']=pd.to_datetime(df['End Time'])



df.head()
df.describe()
df.loc[df['Starting Station Latitude']==0]['Starting Station ID'].value_counts()
df.loc[df['Starting Station Longitude']==0]['Starting Station ID'].value_counts()
df.loc[df['Ending Station Latitude']==0]['Ending Station ID'].value_counts()
df.loc[df['Ending Station Longitude']==0]['Ending Station ID'].value_counts()
stat_4108_lat= df.loc[df['Starting Station ID']==4108]['Starting Station Latitude'].max() #use max to avoid the zeros

stat_4108_long= df.loc[df['Starting Station ID']==4108]['Starting Station Longitude'].min()

 #all non-zeroes are the same anyway no need to find and replace

df['Starting Station Latitude'].replace(0,stat_4108_lat,inplace=True)

df['Ending Station Latitude'].replace(0,stat_4108_lat,inplace=True)

df['Starting Station Longitude'].replace(0,stat_4108_long,inplace=True)

df['Ending Station Longitude'].replace(0,stat_4108_long,inplace=True)
df.describe()
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)

plt.scatter(df['Starting Station Latitude'],df['Starting Station Longitude'],alpha=0.3)

plt.title("Starting Station Latitude and Longitude")

plt.subplot(1,2,2)

plt.scatter(df['Ending Station Latitude'],df['Ending Station Longitude'],alpha=0.3)

plt.title("Ending Station Latitude and Longitude")

plt.show()
sorted(df['Starting Station ID'].unique())==sorted(df['Ending Station ID'].unique())
import folium

bike_map1=folium.Map([df['Starting Station Latitude'].values[0],df['Starting Station Longitude'].values[0]])

for station in df['Starting Station ID'].unique():

    lat=df.loc[df['Starting Station ID']==station]['Starting Station Latitude'].values[0]

    lon=df.loc[df['Starting Station ID']==station]['Starting Station Longitude'].values[0]

    marker=folium.Marker([lat,lon],popup=str(station))

    marker.add_to(bike_map1)



bike_map1
df.loc[df['Starting Station ID']==3039]['Ending Station ID'].value_counts()
df.loc[df['Ending Station ID']==3039]['Starting Station ID'].value_counts()
df.loc[df['Starting Station ID']==3009]['Ending Station ID'].value_counts()
df.loc[df['Ending Station ID']==3009]['Starting Station ID'].value_counts()
print('The first trip made FROM bike Station 3039 (Culver City) was on : ',df.loc[df['Starting Station ID']==3039]['Start Time'].min())

print('The last trip made FROM bike Station 3039 (Culver City) was on : ',df.loc[df['Starting Station ID']==3039]['Start Time'].max())

print('The first trip made FROM bike Station 3009 (Venice) was on : ',df.loc[df['Starting Station ID']==3009]['Start Time'].min())

print('The last trip made FROM bike Station 309 (Venice) was on : ',df.loc[df['Starting Station ID']==3009]['Start Time'].max())



print('The first trip made TO bike Station 3039 (Culver City) was on : ',df.loc[df['Ending Station ID']==3039]['Start Time'].min())

print('The last trip made TO bike Station 3039 (Culver City) was on : ',df.loc[df['Ending Station ID']==3039]['Start Time'].max())

print('The first trip made TO bike Station 3009 (Venice) was on : ',df.loc[df['Ending Station ID']==3009]['Start Time'].min())

print('The last trip made TO bike Station 309 (Venice) was on : ',df.loc[df['Ending Station ID']==3009]['Start Time'].max())
df['Starting Station ID'].value_counts().tail(10)
df['Ending Station ID'].value_counts().tail(10)
df.loc[df['Starting Station ID']==4108]['Trip Route Category'].value_counts(normalize=True)
df.loc[df['Ending Station ID']==4108]['Trip Route Category'].value_counts(normalize=True)
df.loc[df['Starting Station ID']==3053]['Start Time']
df.loc[df['Ending Station ID']==3053]['Start Time']
print('The first trip made FROM bike Station 3053 was on : ',df.loc[df['Starting Station ID']==3053]['Start Time'].min())

print('The last trip made FROM bike Station 3053 was on : ',df.loc[df['Starting Station ID']==3053]['Start Time'].max())

print('The first trip made TO bike Station 3053 was on : ',df.loc[df['Ending Station ID']==3053]['Start Time'].min())

print('The last trip made TO bike Station 30053 was on : ',df.loc[df['Ending Station ID']==3053]['Start Time'].max())
df['Start Time'].min()
df['Start Time'].max()
df['Start Time'].hist(figsize=(15,4))

plt.title('Ride Timeline Histogram')

plt.show()
df['Start Month']=df['Start Time'].dt.month_name()



df['Start Month'].value_counts()
plt.figure(figsize=(14,5))

plt.bar(df['Start Month'].value_counts().index,df['Start Month'].value_counts().values)

plt.title('Number of Rides by Month of Year')

plt.show()
df['Start Day']=df['Start Time'].dt.day_name()

df['Start Day'].value_counts()
plt.figure(figsize=(14,5))



plt.bar(df['Start Day'].value_counts().index,df['Start Day'].value_counts().values)

plt.xticks(rotation=45)

plt.title('Number of Rides by Day of Week')

plt.show()
df['Time Only']= df['Start Time'].dt.round('H')

df['Time Only']=pd.to_datetime(df['Time Only'],format= '%H:%M:%S' ).dt.time

df.head()
plt.figure(figsize=(9,4))

plt.scatter(df['Time Only'].value_counts().index,df['Time Only'].value_counts().values)

#on peak between the bars

plt.vlines(x='7:30:00',ymin=0,ymax=12000,color='red')

plt.vlines(x='20:30:00',ymin=0,ymax=12000,color='red')

plt.title('Bike Usage by Time of Day')

plt.show()
plt.figure(figsize=(14,8))

days=['Monday','Tuesday','Wednesday','Thursday','Friday']

for i in range(len(days)):

    plt.subplot(2,4,i+1)

    plt.scatter(df.loc[df['Start Day'] == days[i]]['Time Only'].value_counts().index,df.loc[df['Start Day'] == days[i]]['Time Only'].value_counts().values)

    plt.title(days[i])

    plt.vlines(x='7:30:00',ymin=0,ymax=1800,color='red')

    plt.vlines(x='20:30:00',ymin=0,ymax=1800,color='red')

    

#Offset peak start by 2 hours for the weekend

plt.subplot(2,4,6)

plt.scatter(df.loc[df['Start Day'] == 'Saturday']['Time Only'].value_counts().index,df.loc[df['Start Day'] == 'Saturday']['Time Only'].value_counts().values)

plt.title('Saturday')

plt.vlines(x='9:30:00',ymin=0,ymax=1800,color='red')

plt.vlines(x='20:30:00',ymin=0,ymax=1800,color='red')



plt.subplot(2,4,7)

plt.scatter(df.loc[df['Start Day'] == 'Sunday']['Time Only'].value_counts().index,df.loc[df['Start Day'] == 'Sunday']['Time Only'].value_counts().values)

plt.title('Sunday')

plt.vlines(x='9:30:00',ymin=0,ymax=1800,color='red')

plt.vlines(x='20:30:00',ymin=0,ymax=1800,color='red')



plt.tight_layout()
def to_hour_int(x):

    #convert hh:mm:ss to hh integar

    x=str(x)

    x=x[:2]

    x=int(x)

    return x
df['Time Only Int']=df['Time Only']

df['Time Only Int']=df['Time Only Int'].astype('str')

df['Time Only Int']=df['Time Only Int'].apply(lambda x: to_hour_int(x))
df['Peak']=1

df.loc[df['Time Only Int']>20.5,'Peak']=0

df.loc[df['Time Only Int']<7.5,'Peak']=0

df.loc[(df['Start Day']=='Saturday')&(df['Time Only Int']<9.5),'Peak']=0

df.loc[(df['Start Day']=='Sunday')&(df['Time Only Int']<9.5),'Peak']=0



df['Peak'].describe()
print(df.Peak.value_counts(normalize=True))

plt.bar(df.Peak.value_counts().index,df.Peak.value_counts().values)

plt.xticks(ticks=[0,1],labels=['Off-Peak','On-Peak'])

plt.title('On-Peak vs. Off-Peak Rides')

plt.show()
df.info()
df['Trip Route Category'].value_counts()
df['Trip Route Category'].value_counts(normalize=True)
df['Passholder Type'].value_counts()
df['Passholder Type'].value_counts(normalize=True)
df['Plan Duration'].value_counts()
df.drop(columns=['Plan Duration'],inplace=True)
len(df['Bike ID'].unique())
df['Bike ID'].value_counts().hist()

plt.title('Bike Usage')

plt.show()
df.groupby('Trip Route Category')['Duration'].describe()
df.groupby('Passholder Type')['Duration'].describe()
df.groupby('Passholder Type')['Trip Route Category'].value_counts(normalize=True)
df.groupby('Peak')['Duration'].describe()
df.groupby('Peak')['Trip Route Category'].value_counts(normalize=True)
df.groupby('Peak')['Passholder Type'].value_counts(normalize=True)
df.Duration.value_counts(normalize=True).head(10)

plt.figure(figsize=(15,4))

df.Duration.hist(bins=29)

plt.title('Ride Duration Histogram')

plt.show()
plt.figure(figsize=(15,4))

df.loc[df['Duration']<30].Duration.hist(bins=29)

plt.title('Ride Duration for rides less than 30 minutes')

plt.show()
round(len(df.loc[df.Duration<31])/len(df),2)
df['Duration']=df['Duration'].clip(upper=30)
plt.figure(figsize=(15,4))

df.Duration.hist(bins=29)

plt.title('Ride Duration clipped to 30 minutes')

plt.show()

# plt.vlines(x=15,ymin=0,ymax=10000,color='red')

# plt.vlines(x=10,ymin=0,ymax=10000,color='red')

# plt.vlines(x=5,ymin=0,ymax=10000,color='red')

# plt.ylim(0,30000)

# plt.xlim(0,5000)
df.columns
y=df['Passholder Type']

X=df.drop(columns=['Bike ID','Time Only','Start Time','End Time','Passholder Type'])
X['Starting Station ID']=X['Starting Station ID'].astype('str')

X['Ending Station ID']=X['Ending Station ID'].astype('str')

X=pd.get_dummies(X)
X.shape
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit_transform(X)
from sklearn.decomposition import PCA

pca=PCA()

pca.fit(X)

tot = sum(pca.explained_variance_)

var_exp = [(i/tot)*100 for i in sorted(pca.explained_variance_, reverse=True)] 

print(var_exp[0:5])

print(sum(var_exp))

cum_var_exp = np.cumsum(var_exp) 

plt.style.use('ggplot')

plt.figure(figsize=(15, 8))

plt.plot(cum_var_exp)

plt.title('Cumulative Explained Variance as a Function of the Number of Components')

plt.vlines(x=20,ymin=var_exp[0],ymax=100)
pca=PCA(n_components=20)

pca.fit(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y)
X_train=pca.transform(X_train)

X_test=pca.transform(X_test)
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

param_grid_rfc={'n_estimators':[10,100],

                'criterion': ['entropy', 'gini'], 

                'max_depth': [2, 5, 10, None],  

                'min_samples_leaf':[0.05 ,0.1, 0.2], 

                'min_samples_split':[0.05 ,0.1, 0.2]}

grid_rfc=GridSearchCV(estimator=RandomForestClassifier(),

                     param_grid=param_grid_rfc,

                     cv=3)

grid_rfc.fit(X_train,y_train)
grid_rfc.best_params_
grid_rfc.best_score_
from sklearn.metrics import confusion_matrix, classification_report

y_pred = grid_rfc.best_estimator_.predict(X_test)

print(classification_report(y_test,y_pred))
import itertools

def show_cf(y_true, y_pred, class_names=None, model_name=None):

    '''Stylized Visual Confusion Matrix provided by Flatiron School'''

    cf = confusion_matrix(y_true, y_pred)

    plt.imshow(cf, cmap=plt.cm.Blues)

    

    if model_name:

        plt.title("Confusion Matrix: {}".format(model_name))

    else:

        plt.title("Confusion Matrix")

    plt.ylabel('True Label')

    plt.xlabel('Predicted Label')

    

#     class_names = set(y_true)

    tick_marks = np.arange(len(class_names))

    if class_names:

        plt.xticks(tick_marks, class_names)

        plt.yticks(tick_marks, class_names)

    

    thresh = cf.max() / 2.

    

    for i, j in itertools.product(range(cf.shape[0]), range(cf.shape[1])):

        plt.text(j, i, cf[i, j], horizontalalignment='center', color='white' if cf[i, j] > thresh else 'black')



    plt.colorbar()

    

show_cf(y_test,y_pred,class_names=['Flex Pass','Monthly Pass','Walk Up'],model_name='Tuned Random Forest Classifier')
from sklearn.ensemble import AdaBoostClassifier

param_grid_ada={'n_estimators': [30, 50, 70],

                'learning_rate': [1.0, 0.5, 0.1]}

grid_ada=GridSearchCV(estimator=AdaBoostClassifier(),

                     param_grid=param_grid_ada,

                     cv=3)

grid_ada.fit(X_train,y_train)
grid_ada.best_params_
grid_ada.best_score_
y_pred = grid_ada.best_estimator_.predict(X_test)

# print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
show_cf(y_test,y_pred, class_names=['Flex Pass','Monthly Pass','Walk Up'],model_name='Tuned ADABoost Classifier')
import xgboost as xgb

param_grid_xgb= {

    "learning_rate": [0.3,0.5,0.7],

    'max_depth': [5,6,7],

    'min_child_weight': [0,1,3],

    'n_estimators': [10,100],

}

grid_xgb=GridSearchCV(estimator=xgb.XGBClassifier(),

                      param_grid=param_grid_xgb,

                     cv=3)

grid_xgb.fit(X_train,y_train)

grid_xgb.best_params_
grid_xgb.best_score_
y_pred = grid_xgb.best_estimator_.predict(X_test)

# print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
show_cf(y_test, y_pred,class_names=['Flex Pass','Monthly Pass','Walk Up'],model_name='Tuned XGBoost Classifier')