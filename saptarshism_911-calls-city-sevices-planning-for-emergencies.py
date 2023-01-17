import numpy as np #for mathematical manipulation

import pandas as pd #for database manipulation

import matplotlib.pyplot as plt #for plotting

import seaborn as sns #better plotting library

%matplotlib inline
data=pd.read_csv('../input/911.csv') #read data from csv
data.head()
data.info()
# Drop dummy variable e

data=data.drop('e',axis=1)
data.head(2)
top_10_zip=pd.DataFrame(data['zip'].value_counts().head(10))

top_10_zip.reset_index(inplace=True)

top_10_zip.columns=['ZIP','Count']

top_10_zip
top_20_zip=pd.DataFrame(data['zip'].value_counts().head(20))

top_20_zip.reset_index(inplace=True)

top_20_zip.columns=['ZIP','Count']

fig1=plt.figure(figsize=(12,6))

sns.barplot(data=top_20_zip,x='ZIP',y='Count',palette="viridis")

fig1.tight_layout()
top_10_twp=pd.DataFrame(data['twp'].value_counts().head(10))

top_10_twp.reset_index(inplace=True)

top_10_twp.columns=['Township','Count']

top_10_twp
top_20_twp=pd.DataFrame(data['twp'].value_counts().head(20))

top_20_twp.reset_index(inplace=True)

top_20_twp.columns=['Township','Count']

fig2=plt.figure(figsize=(12,6))

g=sns.barplot(data=top_20_twp,x='Township',y='Count',palette="viridis")

g.set_xticklabels(g.get_xticklabels(),rotation=45)

fig2.tight_layout()
data['title'].nunique()
data['Reason']=data['title'].apply(lambda v:v.split(':')[0])
data['Reason'].nunique()
data['Reason'].value_counts()
fig3=plt.figure(figsize=(12,6))

g=sns.countplot(data=data[(data['twp'].isin(top_10_twp['Township']))],x='twp',hue='Reason',palette='viridis')

g_x=g.set_xticklabels(g.get_xticklabels(),rotation=30)

fig3.tight_layout()
data['Station']=data['desc'].apply(lambda v:v.split(';')[2])
data['timeStamp']=pd.to_datetime(data['timeStamp'])
data['Hour']=data['timeStamp'].apply(lambda v:v.hour)

data['DayOfWeek']=data['timeStamp'].apply(lambda v:v.dayofweek)

data['Month']=data['timeStamp'].apply(lambda v:v.month)

data['Date']=data['timeStamp'].apply(lambda v:v.date())
# Map day values to proper strings

dmap1 = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

data['DayOfWeek']=data['DayOfWeek'].map(dmap1)
data.head(2)
fig4=plt.figure(figsize=(12,8))

sns.countplot(x='DayOfWeek',hue='Reason',palette='viridis',data=data)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig5=plt.figure(figsize=(12,8))

sns.countplot(x='Month',hue='Reason',palette='viridis',data=data)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
databyMonth_EMS = data[data['Reason']=='EMS'].groupby('Month').count()

databyMonth_Fire = data[data['Reason']=='Fire'].groupby('Month').count()

databyMonth_Traffic = data[data['Reason']=='Traffic'].groupby('Month').count()

databyMonth_Cumul = data.groupby('Month').count()



databyMonth_EMS['twp'].plot(figsize=(12,8),label='EMS',lw=5,ls='--')

databyMonth_Fire['twp'].plot(figsize=(12,8),label='Fire',lw=5,ls='--')

databyMonth_Traffic['twp'].plot(figsize=(12,8),label='Traffic',lw=5,ls='--')

databyMonth_Cumul['twp'].plot(figsize=(12,8),label='Total',lw=5)



fig=plt.xticks(np.arange(1,13),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

plt.title("Emergency Call Rates vs Month")

plt.legend()
sns.lmplot(data=databyMonth_Cumul.reset_index(),x='Month',y='twp')

plt.title("Regression plot of Emergency calls vs Month")

plt.xlabel('Months')

plt.ylabel('Counts')
data.groupby('Date').count()['twp'].plot(figsize=(15,3))

plt.tight_layout()
data[data['Reason']=='EMS'].groupby('Date').count()['twp'].plot(figsize=(15,3),label='EMS')

data[data['Reason']=='Fire'].groupby('Date').count()['twp'].plot(figsize=(15,3),label='Fire')

data[data['Reason']=='Traffic'].groupby('Date').count()['twp'].plot(figsize=(15,3),label='Traffic')

plt.tight_layout()

plt.legend()
strange_increase=data[(data['Reason']=='Traffic') & ( (data['timeStamp']>pd.to_datetime("2016-01-1")) &  (data['timeStamp']<pd.to_datetime("2016-02-1")))].reset_index().drop('index',axis=1)
strange_increase['title'].value_counts()
normal_counts=data[(data['Reason']=='Traffic') & ( (data['timeStamp']>pd.to_datetime("2016-02-1")) &  (data['timeStamp']<pd.to_datetime("2016-08-1")))].reset_index().drop('index',axis=1)

normal_counts['title'].value_counts()/6
sns.jointplot(data=data,x='lng',y='lat',kind='scatter')
data_geog=data[(np.abs(data["lat"]-data["lat"].mean())<=(4.5*data["lat"].std())) & (np.abs(data["lng"]-data["lng"].mean())<=(10*data["lng"].std()))]

data_geog.reset_index().drop('index',axis=1,inplace=True)

sns.jointplot(data=data_geog,x='lng',y='lat',kind='scatter')
data_geog[['lat','lng']].head()
pd.options.mode.chained_assignment = None #Remove Error Message

x_mean=data_geog['lng'].mean()

y_mean=data_geog['lat'].mean()

data_geog['x']=data_geog['lng'].map(lambda v:v-x_mean)

data_geog['y']=data_geog['lat'].map(lambda v:v-y_mean)
theta=np.pi/3

rot_mat=np.array([np.cos(theta),-np.sin(theta),np.sin(theta),np.cos(theta)]).reshape(2,2)

data_geog[['x','y']]=data_geog[['x','y']].apply(lambda v:np.dot(v.as_matrix(),rot_mat),axis=1)
sns.jointplot(data=data_geog,x='x',y='y',kind='scatter',xlim=(-0.3,0.3))
sns.jointplot(data=data_geog,x='x',y='y',kind='kde',xlim=(-0.3,0.3))
sns.jointplot(data=data_geog[data_geog['Reason']=='EMS'],x='x',y='y',kind='kde',color='green',xlim=(-0.3,0.3))

plt.title('EMS Distribution')

plt.tight_layout()
sns.jointplot(data=data_geog[data_geog['Reason']=='Fire'],x='x',y='y',kind='kde',color='red',xlim=(-0.3,0.3))

plt.title('Fire Distribution')

plt.tight_layout()
sns.jointplot(data=data_geog[data_geog['Reason']=='Traffic'],x='x',y='y',kind='kde',color='purple',xlim=(-0.3,0.3))

plt.title('Traffic Distribution')

plt.tight_layout()
fig=plt.figure(figsize=(10,10))

twp_group=data_geog.groupby('twp')

for name, group in twp_group:

    plt.plot(group.x, group.y, marker='o', linestyle='', label=name)

plt.xlim(-0.3,0.3)

plt.title("Townships")
from sklearn.cluster import KMeans
X=data_geog[['x','y']].reset_index().drop('index',axis=1)
kmeans=KMeans(n_clusters=10)
kmeans.fit(X)
fig=plt.figure(figsize=(7,7))

plt.scatter(X['x'],X['y'],c=kmeans.labels_,cmap='rainbow')

plt.xlim(-0.3,0.3)
fig=plt.figure(figsize=(12,12))

for i in range(3,12):

    kmeans=KMeans(n_clusters=i)

    kmeans.fit(X)

    fig.add_subplot(3,3,i-2)

    plt.scatter(X['x'],X['y'],c=kmeans.labels_,cmap='rainbow')

    plt.title("Number of Clusters = {}".format(i))

    plt.xlim(-0.3,0.3)
latsin_dist=np.abs(np.sin(np.max(data_geog["lat"])/180*np.pi)-np.sin(np.min(data_geog["lat"])/180*np.pi))

lng_dist=np.abs(np.max(data_geog["lng"])-np.min(data_geog["lng"]))
def ll2area(latsin,lng):

    return 2*np.pi*(6371**2)*latsin*lng/360

A=ll2area(latsin_dist,lng_dist)

print("The Area of the Township is Appoximately {} sq. km".format(A))
pop=np.int(A*314)

print("The Avg Population of the Township is Appoximately {}".format(pop))
final_kmeans=KMeans(n_clusters=6)

final_kmeans.fit(X)

fig=plt.figure(figsize=(7,7))

plt.scatter(X['x'],X['y'],c=final_kmeans.labels_,cmap='rainbow')

plt.xlim(-0.3,0.3)
data_clus=pd.concat([data_geog.reset_index().drop('index',axis=1),pd.DataFrame(final_kmeans.labels_,columns=['Cluster'])],axis=1)
data_clus.drop(['desc','title','timeStamp'],axis=1,inplace=True)
data_clus.tail(2)
data_block=data_clus.copy()
data_block['lat']=np.rint(data_block['lat']*100)/100 #reduce to 1 km block

data_block['lng']=np.rint(data_block['lng']*100)/100 #reduce to 1 km block
data_block.head(2)
model_data=data_block.groupby(['lat','lng']).count().reset_index().drop(['zip','twp','Reason','Hour','DayOfWeek','Month','Date','Station','Cluster','x','y'],axis=1)

print("This is the form of our new data reduced to 1 sqkm blocks")

model_data.head(1)
X2=model_data[['lat','lng']]

y2=model_data.drop(['lat','lng'],axis=1)
from sklearn.neighbors import KernelDensity

kd=KernelDensity()

kd.fit(X2,y2)

y2=pd.DataFrame(np.exp(kd.score_samples(X2)))

mean_kernel=np.exp(kd.score_samples(X2)).mean()

min_kernel=np.exp(kd.score_samples(X2)).min()

mean_pop=0.95*pop/A

def kern2pop(ker):

    return (((mean_kernel+np.sign(ker-mean_kernel)*(np.abs(ker-mean_kernel))**0.59)/mean_kernel))*mean_pop

y2=pd.DataFrame(y2.apply(kern2pop))

print("Our data is now ready")

pd.DataFrame(pd.concat([X2,y2],axis=1).head(5))
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.linear_model import LinearRegression
lin_model=LinearRegression()

print("R2 Score: {} ".format(cross_val_score(lin_model,X2,y2,scoring='r2',cv=10).mean()))

print("Root Mean Squared Error: {}".format(np.sqrt(-cross_val_score(lin_model,X2,y2,scoring='neg_mean_squared_error',cv=10).mean())))

predicted=cross_val_predict(lin_model,X2,y2,cv=10)

plt.scatter(y2,predicted)

plt.plot([y2.min(), y2.max()], [y2.min(), y2.max()], 'k--', lw=4)

plt.title('Residual Error')

plt.xlabel('Measured')

plt.ylabel('Predicted')
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(2)

X_quad=pd.DataFrame(poly.fit_transform(X2),columns=['1','lat','lng','lat^2','lat*lng','lng^2'])
quad_model=LinearRegression()

print("R2 Score: {}".format(cross_val_score(quad_model,X_quad,y2,scoring='r2',cv=10).mean()))

print("Root Mean Squared Error: {}".format(np.sqrt(-cross_val_score(quad_model,X_quad,y2,scoring='neg_mean_squared_error',cv=10).mean())))

predicted=cross_val_predict(quad_model,X_quad,y2,cv=10)

plt.scatter(y2,predicted)

plt.plot([y2.min(), y2.max()], [y2.min(), y2.max()], 'k--', lw=4)

plt.title('Residual Error')

plt.xlabel('Measured')

plt.ylabel('Predicted')
from sklearn.tree import DecisionTreeRegressor
dtree_model=DecisionTreeRegressor()

print("R2 Score: {}".format(cross_val_score(dtree_model,X2,y2,scoring='r2',cv=10).mean()))

print("Root Mean Squared Error: {}".format(np.sqrt(-cross_val_score(dtree_model,X2,y2,scoring='neg_mean_squared_error',cv=10).mean())))

predicted=cross_val_predict(dtree_model,X2,y2,cv=10)

plt.scatter(y2,predicted)

plt.plot([y2.min(), y2.max()], [y2.min(), y2.max()], 'k--', lw=4)

plt.title('Residual Error')

plt.xlabel('Measured')

plt.ylabel('Predicted')
from sklearn.ensemble import RandomForestRegressor
randf_model=RandomForestRegressor(n_jobs=-1)

yx=y2.values.ravel()

print("R2 Score: {}".format(cross_val_score(randf_model,X2,yx,scoring='r2',cv=10).mean()))

print("Root Mean Squared Error: {}".format(np.sqrt(-cross_val_score(randf_model,X2,yx,scoring='neg_mean_squared_error',cv=10).mean())))

predicted=cross_val_predict(randf_model,X2,yx,cv=10)

plt.scatter(yx,predicted)

plt.plot([yx.min(), yx.max()], [yx.min(), yx.max()], 'k--', lw=4)

plt.title('Residual Error')

plt.xlabel('Measured')

plt.ylabel('Predicted')
from sklearn.svm import SVR 
svm_model=SVR()

print("R2 Score: {}".format(cross_val_score(svm_model,X2,yx,scoring='r2',cv=10).mean()))

print("Root Mean Squared Error: {}".format(np.sqrt(-cross_val_score(svm_model,X2,yx,scoring='neg_mean_squared_error',cv=10).mean())))

predicted=cross_val_predict(svm_model,X2,yx,cv=10)

plt.scatter(yx,predicted)

plt.plot([yx.min(), yx.max()], [yx.min(), yx.max()], 'k--', lw=4)

plt.title('Residual Error')

plt.xlabel('Measured')

plt.ylabel('Predicted')
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}

grid=GridSearchCV(SVR(),param_grid,verbose=0)

grid.fit(X2,yx)

print('Best Score is {} at {}'.format(grid.best_score_,grid.best_params_))
gridsvm=SVR(C=1000,gamma=1)

print("R2 Score: {}".format(cross_val_score(gridsvm,X2,yx,scoring='r2',cv=10).mean()))

print("Root Mean Squared Error: {}".format(np.sqrt(-cross_val_score(gridsvm,X2,yx,scoring='neg_mean_squared_error',cv=10).mean())))

predicted=cross_val_predict(gridsvm,X2,yx,cv=10)

plt.scatter(yx,predicted)

plt.plot([yx.min(), yx.max()], [yx.min(), yx.max()], 'k--', lw=4)

plt.title('Residual Error')

plt.xlabel('Measured')

plt.ylabel('Predicted')
clus_info=pd.DataFrame(final_kmeans.cluster_centers_,columns=['x','y'])

print("Cluster Centers in local coordinate are:")

clus_info
fig=plt.figure(figsize=(7,7))

plt.scatter(X['x'],X['y'],c=final_kmeans.labels_,cmap='summer')

plt.xlim(-0.3,0.3)

plt.scatter(clus_info['x'],clus_info['y'],marker='^',color='black')

n=[' C1',' C2',' C3',' C4',' C5',' C6']

for i,txt in enumerate(n):

    plt.annotate(txt,xy=(clus_info['x'][i],clus_info['y'][i]),color='black')
poly=PolynomialFeatures(2)

clus_quad=pd.DataFrame(poly.fit_transform(data_clus[['lat','lng']]),columns=['1','lat','lng','lat^2','lat*lng','lng^2'])
quad_model.fit(X_quad,y2)

popdense=pd.DataFrame(quad_model.predict(clus_quad).ravel(),columns=["Pop. Density"])

data_clus=pd.concat([data_clus,popdense],axis=1)

data_clus.head()
pope=data_clus.groupby('Cluster').mean()['Pop. Density'].as_matrix()

pope
print("The predicted population densities are:\n")

#popdense=quad_model.predict(clus_quad).ravel()

#popdense=gridsvm.predict(data_clus[['lat','lng']]).ravel()

print(pope)

print("\nSo we can see that {} has the maximum density and correspondingly a small cluster size.\nAlso {} has least population density but much larger size. This generaly means that the populations have been managed equally".format(n[pope.argmax()],n[pope.argmin()]))
areas=[]

print("The approximate cluster areas are:\n")

for i in range(0,6):

    tempdata=data_clus[data_clus['Cluster']==i]

    lats=np.abs(np.sin(np.max(tempdata["lat"])/180*np.pi)-np.sin(np.min(tempdata["lat"])/180*np.pi))

    lngs=np.abs(np.max(tempdata["lng"])-np.min(tempdata["lng"]))

    pops=(2/3)*ll2area(lats,lngs)

    areas.append(pops)

    print("Cluster {} : {:.2f} sq km".format(i+1,pops))

print("\nThe predicted cluster populations are:")

print(areas*pope)
def getname(v):

    if len(v.split('Station'))>1:

        if v.split('Station')[1][0]==':':

            return v.split('Station')[1][1:]

        else:

            return v.split('Station')[1]

    else:

        return 0

data_geog['Station Name']=data_geog['Station'].apply(getname)

station_base=data_geog[data_geog['Station Name'] != 0].copy().drop(['timeStamp','title','desc'],axis=1)

station_base.head(3)
station_base['Station Name'].nunique()
station_list =station_base.groupby('Station Name').mean().reset_index().drop(['zip','Hour','Month'],axis=1).drop(0)

station_list.head(2)
fig=plt.figure(figsize=(7,7))

plt.scatter(X['x'],X['y'],c=final_kmeans.labels_,cmap='summer')

plt.scatter(station_list['x'],station_list['y'],marker='^',color='black')

plt.xlim(-0.3,0.3)
dummies=pd.get_dummies(station_base['Reason'])

dummies.head(2)
emergencies=pd.concat([station_base,dummies],axis=1).groupby('Station Name').sum().drop(['lat','lng','x','y','zip','Hour','Month'],axis=1).reset_index().drop(0)

emergencies=pd.concat([emergencies,station_list[['lat','lng','x','y']]],axis=1)

popdenseStation=pd.DataFrame(quad_model.predict(pd.DataFrame(poly.fit_transform(emergencies[['lat','lng']]),columns=['1','lat','lng','lat^2','lat*lng','lng^2'])).ravel(),columns=["Pop. Density"])

emergencies=pd.concat([emergencies.reset_index().drop('index',axis=1),popdenseStation],axis=1)

emergencies.head(3)
emergencies.tail(3)
def makemap(ems,fire,lat,lng,size,alpha):

    if fire>ems:

        plt.scatter(lat,lng,marker='o',color='red',s=size,alpha=alpha)

        plt.xlim(-0.3,0.3)

    else:

        ems=plt.scatter(lat,lng,marker='o',color='blue',s=size,alpha=alpha)

        plt.xlim(-0.3,0.3)

fig=plt.figure(figsize=(7,7))

plt.scatter(X['x'],X['y'],c=final_kmeans.labels_,cmap='summer')

plt.xlim(-0.3,0.3)





for index,row in emergencies.iterrows():

    makemap(row['EMS'],row['Fire'],row['x'],row['y'],50,1)



fire_station=emergencies[emergencies["Fire"]>0].drop(['EMS','Fire'],axis=1)

ems_station=emergencies[emergencies["EMS"]>0].drop(['EMS','Fire'],axis=1)
fig=plt.figure(figsize=(7,7))

plt.scatter(X['x'],X['y'],c=final_kmeans.labels_,cmap='summer')

plt.xlim(-0.3,0.3)



for index,row in emergencies.iterrows():

    makemap(row['EMS'],row['Fire'],row['x'],row['y'],3* (550-row['Pop. Density']),0.4)