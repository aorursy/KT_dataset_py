import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score
#!ls crop
#files = !ls crop/*Soybean*
all = os.listdir(path='../input/')
files = [fname for fname in all if 'Soybean' in fname]
files
path_prefix='../input/'
soybean = pd.concat([pd.read_csv(path_prefix+f) for f in files], keys=files)
#soybean.head()
close = pd.read_csv(path_prefix+'soybean_JUL14.txt')
#close.head()
nearby = pd.read_csv(path_prefix+'soybean_nearby.txt')
#nearby.head()
soybean.shape
soybean.columns
#soybean.dtypes
soybean.describe().transpose()[:27]
soybean.head(5)
#soybean.tail(5)
close.shape
close.columns
close.dtypes
close.describe().transpose()
close.head(3)
#close.tail(3)
nearby.shape
nearby.columns
#nearby.dtypes
nearby.describe().transpose()
nearby.head(5)
#nearby.tail(5)
#any dataset containing np.nan values?
sum(soybean.isna().any(axis=1))
sum(close.isna().any(axis=1))
sum(nearby.isna().any(axis=1))
#it's unclear why the dataset has lots of duplicate dates? 
#It might be due to the way the script is aggregating the raw data from the USDA website
print(soybean.shape[0], len(soybean.Date.unique()))
print('before drop duplicates: ', soybean.shape[0])
soybean.drop_duplicates(inplace=True)
print('after drop duplicates: ', soybean.shape[0])
print('before drop duplicates: ', close.shape[0])
close.drop_duplicates(inplace=True)
print('after drop duplicates: ', close.shape[0])
print('before drop duplicates: ', nearby.shape[0])
nearby.drop_duplicates(inplace=True)
print('after drop duplicates: ', nearby.shape[0])
#reset df index
soybean.reset_index(drop=True, inplace=True)
#parse Date object type to datetime type
soybean.Date = pd.to_datetime(soybean.Date)
#soybean.dtypes['Date']
#add 'year' and 'month' new columns to the df to facilitate following DV and analysis
#ref.: Extracting just Month and Year from Pandas Datetime column (Python)
#https://stackoverflow.com/questions/25146121/extracting-just-month-and-year-from-pandas-datetime-column-python
soybean['year'] = soybean.Date.dt.year
soybean['month'] = soybean.Date.dt.month
soybean.loc[:3, ['Date','year','month']]
plt.scatter(soybean['United States Exports'], soybean['China Imports'])
plt.xlabel('US Exports')
plt.ylabel('China Imports')
plt.title('US Soybean Exports vs. China Imports')
plt.scatter(soybean['United States Exports'], soybean['Japan Imports'])
plt.xlabel('US Exports')
plt.ylabel('Japan Imports')
plt.title('US Soybean Exports vs. Japan Imports')
#since data contains different number of months for different years, I use the average over months per year as a common factor for comparision
soybean_yearly = soybean.groupby('year').mean()
#drop 'month' column 
soybean_yearly.drop(axis=1, labels=['month'], inplace=True)
#print(soybean_yearly.shape)
#soybean_yearly
#pd.unique(soybean_yearly.index)
#option A
soybean_yearly.Yield.plot(kind='bar',figsize=(6,4))
#option B
#plt.bar(soybean_yearly.index, soybean_yearly['Yield'])
plt.xlabel('Year')
plt.ylabel('Yield')
plt.title('Soybean Yearly Yield')
#option A
def plot_trends1 (df,f1, f2, title):
    #set figure style
    linestyle = ['b--','g-s']
    linewidth = 1.8
    fig = plt.gcf()
    fig.set_size_inches(8, 6)   
    
    #plot multi plots in the same figure 
    #option A (preferred)
    plt.plot(df[f1], linestyle[0], linewidth=linewidth, label=f1)
    plt.plot(df[f2], linestyle[1], linewidth=linewidth, label=f2)
    #option B
    #df[f1].plot(color='r',marker='.', label=f1)
    #df[f2].plot(color='b', marker='*',label=f2)
    
    #set figure annotation info (xlabel, ylabel, title, legend, etc.)
    plt.xlabel('Year')
    plt.title(title+f1+' & ' +f2)  
    plt.legend(loc='best')
    plt.show()
plot_trends1(soybean_yearly,'Area Planted','Area Harvested', 'Soybean Yearly ')
#Option B
def plot_trends1B (df,f1,f2, title): 

    #set figure style
    color=['green','coral']
    linewidth = 1.8
    fig = plt.gcf()
    fig.set_size_inches(8, 6)

    #plot multi plots in the same figure 
    ax = df[f1].plot(linewidth=linewidth, marker='*', color=color[0])
    df[f2].plot(linewidth=linewidth, marker='s', color=color[1])
    
    #set figure annotation info (xlabel, ylabel, title, legend, xticks, xticklabels with rotation, etc.)
    ax.set_xticks(df.index)
    ax.set_xticklabels(df.index, rotation=45) 
    plt.xlabel('Year')
    plt.title(title+f1+' & ' +f2)  
    plt.legend(loc='best')    
    plt.show()
plot_trends1B(soybean_yearly,'Area Planted','Area Harvested', 'Soybean Yearly ')
#multiple plots in same figure with one axis via python matplotlib
#ref.: https://stackoverflow.com/questions/43179027/multiple-plots-in-same-figure-with-one-axis-via-python-matplotlib
def plot_trends2 (df,f1,f2,f3,title): 
    #color=['blue','green','red']
    linestyle=['r-.','b--','g-*']
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    fig.set_size_inches(8, 6)

    ax.plot(df[f2], linestyle[1], label=f2)
    ax.plot(df[f3], linestyle[2], label=f3)
    ax2.plot(df[f1], linestyle[0],label=f1)

    ax.set_title(title+f1+' vs. '+f2+' & '+f3)

    ax.legend(loc=2)
    ax2.legend(loc=4)
    plt.show()
plot_trends2(soybean_yearly,'Yield','Area Planted','Area Harvested','Soybean Yearly ')
plot_trends2(soybean_yearly,'Production','Area Planted','Area Harvested','Soybean Yearly ')
plot_trends2(soybean_yearly,'United States Exports','Area Planted','Area Harvested','Soybean Yearly ')
plot_trends2(soybean_yearly,'China Exports','Area Planted','Area Harvested','Soybean Yearly ')
plot_trends2(soybean_yearly,'China Imports','Area Planted','Area Harvested','Soybean Yearly ')
soybean_monthly = soybean.groupby(['year','month']).mean()
#soybean_monthly.shape
#soybean_monthly.head(3) # this is a multi-index (i.e. multi index levels) df
#soybean_monthly.index.names
#soybean_monthly.index.levels
#soybean_monthly.index.labels
#soybean_monthly.index.get_level_values('year')
#soybean_monthly.index.get_level_values('month')
def plot_trends1B_multi_index (df,f1,f2,title): 

    #set figure style
    color=['green','coral']
    marker=['*','s']
    linewidth = 1.8
    fig = plt.gcf()
    fig.set_size_inches(8, 6)

    #plot multi plots in the same figure 
    #ax = df.Yield.plot(linewidth=linewidth, color=color[0])
    ax = df[f1].plot(linewidth=linewidth, marker=marker[0], color=color[0])
    df[f2].plot(linewidth=linewidth, marker=marker[1], color=color[1])
    
    #set figure annotation info (xlabel, ylabel, title, legend, xticks, xticklabels with rotation, etc.)
    dur = np.arange(len(df.index.labels[0])) # [0,....129]
    #print(dur)
    every10sampling = (dur%10 == 0) #boolean vector for filtering, 1/10 sampling rate
    #print(every10sampling)
    lessdur = dur[every10sampling]
    #print(lessdur)
    
    ax.set_xticks(lessdur)
    
    lvv = df.index.get_level_values('year') #for each row, get its corresponding level value (specified by level name) in a multi index (levels) df!  
    #print(lvv)
    lesslvv = lvv[every10sampling]
    #print(lesslvv)
    
    #ax.set_xticklabels(df.index.labels, rotation=45) #Notice the ; (remove it and see what happens !)
    ax.set_xticklabels(lesslvv, rotation=45)
    
    plt.xlabel('Year')
    plt.title(title+f1+' & ' +f2)  
    plt.legend(loc='best')    
    plt.show()
plot_trends1B_multi_index(soybean_monthly,'Area Planted','Area Harvested','Soybean Monthly ')
plot_trends1B_multi_index(soybean_monthly, 'United States Exports', 'China Imports','Soybean Monthly ')
def plot_trends2_multi_index (df,f1,f2,f3,title): 
    color=['blue','green','red']
    marker=['*','s','d']
    linewidth=1.8
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    fig.set_size_inches(8, 6)

    ax.plot(range(len(df.index)),df[f2], linewidth=linewidth, marker=marker[1],color=color[1], label=f2)
    ax.plot(range(len(df.index)),df[f3], linewidth=linewidth, marker=marker[2],color=color[2], label=f3)
    ax2.plot(range(len(df.index)),df[f1], linewidth=linewidth, marker=marker[0], color=color[0],  label=f1)

    dur = np.arange(len(df.index.labels[0])) # [0,....129]
    #print(dur)
    every10sampling = (dur%10 == 0) #boolean vector for filtering, 1/10 sampling rate
    #print(every10sampling)
    lessdur = dur[every10sampling]
    #print(lessdur)
    
    ax.set_xticks(lessdur)
    
    lvv = df.index.get_level_values('year') #for each row, get its corresponding level value (specified by level name) in a multi index (levels) df!  
    #print(lvv)
    lesslvv = lvv[every10sampling]
    #print(lesslvv)
    
    #ax.set_xticklabels(df.index.labels, rotation=45) #Notice the ; (remove it and see what happens !)
    ax.set_xticklabels(lesslvv, rotation=45)
    
    ax.set_title(title+f1+' vs. '+f2+' & '+f3)

    ax.legend(loc=2)
    ax2.legend(loc=4)
    plt.show()
plot_trends2_multi_index(soybean_monthly,'Yield','Area Planted','Area Harvested','Soybean Monthly ')
plot_trends2_multi_index(soybean_monthly,'Production','Area Planted','Area Harvested','Soybean Monthly ')
plot_trends2_multi_index(soybean_monthly,'United States Exports','Area Planted','Area Harvested','Soybean Monthly ')
plot_trends2_multi_index(soybean_monthly,'China Imports','Area Planted','Area Harvested','Soybean Monthly ')
#tranform nearby dataset's dates field to datetime and add year and month fields
nearby.dates = pd.to_datetime(nearby.dates)
nearby['year'] = nearby.dates.dt.year
nearby['month']= nearby.dates.dt.month
#nearby.head(3)
#group and average nearby price per year, month
nearby_monthly = nearby.groupby(['year','month']).mean()
#nearby_monthly.head(3)
#nearby_monthly.shape
#Option A: use df1.merge(df2, ...) default is inner join
merged0 = nearby_monthly.merge(soybean_monthly, how='inner', left_index=True, right_index=True)
#print(nearby_monthly.shape, soybean_monthly.shape,merged0.shape)
#Option B: use df1.join(df1, ...) default is left join
merged = nearby_monthly.join(soybean_monthly, how = 'inner', lsuffix='_x')
#print(nearby_monthly.shape, soybean_monthly.shape,merged.shape)
#Option C: use concat([df1,df2], axis=1) default is full outer join
merged2 = pd.concat([nearby_monthly,soybean_monthly], join='inner', axis=1)
#print(nearby_monthly.shape, soybean_monthly.shape,merged2.shape)
#merged.head(5)
#merged.tail(5)
plot_trends2_multi_index(merged, 'nearby_close', 'Area Planted','Area Harvested', 'Soybean Monthly ')
#group and average nearby price per year
nearby_yearly = nearby.groupby('year').mean()
nearby_yearly.drop(axis=1,labels=['month'],inplace=True)
#nearby_yearly.head(3)
#merged with soybean yearly data
merged_yearly = nearby_yearly.merge(soybean_yearly, how='inner', left_index=True, right_index=True)
#print(nearby_yearly.shape, soybean_yearly.shape, merged_yearly.shape)
#merged_yearly
plot_trends2(merged_yearly,'nearby_close','Area Planted','Area Harvested','Soybean Yearly ')
# create a function for plotting a series with index and numeric values

def plot_series(ser, y_label,title):  
    color='m'
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.ylabel(y_label)
    plt.title(title)
    
    ax = ser.plot(linewidth=3.3, color=color)
    ax.set_xticks(range(len(ser)))
    ax.set_xticklabels(ser.index,rotation=90)
    plt.show()
plot_series(soybean.corr()['United States Exports'].sort_values(ascending=False)[1:11],'United States Exports','Top 10 Correlations')
top10 = soybean.corr()['United States Exports'].sort_values(ascending=False)[0:10]
top10
sub_df = soybean[top10.index]
#sub_df
def draw_imshow(df):
    plt.imshow(df, cmap=plt.cm.Blues, interpolation='nearest')
    plt.colorbar()
    tick_marks = [i for i in range(len(df.columns))]
    plt.xticks(tick_marks, df.columns, rotation='vertical')
    plt.yticks(tick_marks, df.columns)
draw_imshow(sub_df.corr())
USs = soybean_yearly.columns[soybean_yearly.columns.str.contains('United States')]
#USs
#subset a df only containing us soybean columns 
df_us = soybean_yearly [USs]
#df_us.head()
ss = StandardScaler()
X = ss.fit_transform(df_us)
#print(df_us.shape, X.shape)
pd.DataFrame(X, columns=USs).describe().transpose()
#set a range of no. of clusters to explore
nc = range(2,11) #2~10

wsse_list = []
km_list = []

for i in nc:
    km=KMeans(n_clusters=i)
    _ = km.fit(X)
    wsse = km.inertia_
    wsse_list.append(wsse)
    km_list.append(km)
#print(wsse_list)
#print(km_list)   
plt.plot(nc,wsse_list)
plt.xlabel('no. of cluster')
plt.ylabel('WSSE (Within-Cluster Sum of Square Errors)')
plt.title('Elbow Curve for Selecting n_clusters value')
final_model = km_list[nc.index(6)]
final_model
uscentroids = final_model.cluster_centers_
#print(uscentroids.shape, uscentroids)
print(final_model.labels_)
pd.value_counts(final_model.labels_, sort=False)
# create a df of centroids with labels

def centroids_df(names, centroids):
    df = pd.DataFrame(centroids, columns=names)
    df['label'] = df.index
    return df
def draw_parallel(df):
    colors = ['r','y', 'g', 'c', 'b', 'm','k']    
    ax = plt.figure(figsize=(12,9)).gca().axes.set_ylim([-3,+3])
    plt.xticks(rotation=90) 
    parallel_coordinates(df, 'label', color = colors, marker='s')
centroidsDF = centroids_df(USs, uscentroids)
#centroidsDF
draw_parallel(centroidsDF)
year_label = zip(df_us.index,final_model.labels_)
#year_label
pd.DataFrame(list(year_label), columns=['Year','Label']).transpose()
data = merged.copy().drop(axis=1, labels=('nearby_close'))
#data.head()
labels = merged['nearby_close']
#labels.head()
data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=.2, random_state=7)
#print(data_train.shape, data_test.shape, label_train.shape, label_test.shape)
lm = LinearRegression()
lm.fit(data_train,label_train)
#np.array(zip(data.columns,lm.coef_)).reshape(-1,2)
#option A
revf = pd.DataFrame(list(zip(data.columns,lm.coef_)),columns=['feature','coef'])
#option B
#revf = pd.DataFrame(np.array(zip(data.columns,lm.coef_)).reshape(-1,2),columns=['feature','coef'])
revf.coef = revf.coef.astype(np.float64)
#revf
#pandas - sort by absolute value without changing the data
#ref.: https://stackoverflow.com/questions/30486263/pandas-sort-by-absolute-value-without-changing-the-data

#top 10 most important features for lm model prediction
revf.reindex(revf.coef.abs().sort_values(ascending=False).index)[:10]
label_pred_lm = lm.predict(data_test)
#print(label_pred_lm.shape, type(label_pred_lm))
#print(label_pred_lm)
RMSE_lm = sqrt(mean_squared_error(y_true = label_test, y_pred = label_pred_lm))
print(RMSE_lm, label_test.mean(), label_test.std())
obs_pred_df_lm = pd.DataFrame({'observ':label_test,'pred':label_pred_lm})
opdf_lm = obs_pred_df_lm.reset_index()
#opdf_lm
#How to sort a dataFrame in python pandas by two or more columns?
#Ref.: https://stackoverflow.com/questions/17141558/how-to-sort-a-dataframe-in-python-pandas-by-two-or-more-columns
sorted_opdf_lm = opdf_lm.sort_values(['year','month'], ascending=[True,True])
sorted_opdf_lm['year_month']=list(zip(sorted_opdf_lm['year'].values,sorted_opdf_lm['month'].values))
sorted_opdf_lm.reset_index(drop=True, inplace=True)
def plot_pred_vs_obs (df,f1,f2,title): 

    #set figure style
    color=['green','coral']
    marker=['*','s']
    linewidth = 1.8
    fig = plt.gcf()
    fig.set_size_inches(8, 6)

    #plot multi plots in the same figure 
    ax = df[f1].plot(linewidth=linewidth, marker=marker[0], color=color[0])
    df[f2].plot(linewidth=linewidth, marker=marker[1], color=color[1])
    
    #set figure annotation info (xlabel, ylabel, title, legend, xticks, xticklabels with rotation, etc.)
    ax.set_xticks(range(len(df.index)))
    ax.set_ylim([0,1600])
    
    #ax.set_xticklabels(df.index.labels, rotation=45) #Notice the ; (remove it and see what happens !)
    ax.set_xticklabels(df.year_month, rotation=60)
    
    plt.xlabel('(Year, Month)')
    plt.ylabel('nearby_close')
    plt.title(title+f1+' vs. ' +f2)  
    plt.legend(loc='best')    
    plt.show()
plot_pred_vs_obs(sorted_opdf_lm,'observ','pred', 'LM Nearby_close ' )
dt = DecisionTreeRegressor(random_state=123)
dt.fit(data_train,label_train)
feature_relevance = pd.Series(dt.feature_importances_, index=data.columns)
#top 10 most important features for DT model prediction
feature_relevance.sort_values(ascending=False)[:10]
label_pred_dt = dt.predict(data_test)
#print(label_pred_dt.shape, type(label_pred_dt))
#print(label_pred_dt)
RMSE_dt = sqrt(mean_squared_error(y_true = label_test, y_pred = label_pred_dt))
RMSE_dt
pd.DataFrame({'target mean':label_test.mean(),'target std.':label_test.std(),'Linear regression RMSE':RMSE_lm,'Decisition Tree RMSE':RMSE_dt},index=['Compare'])
obs_pred_df_dt = pd.DataFrame({'observ':label_test,'pred':label_pred_dt})
#print(obs_pred_df_dt.head(3))
opdf_dt = obs_pred_df_dt.reset_index()
#print(opdf_dt.head(3))
sorted_opdf_dt = opdf_dt.sort_values(['year','month'], ascending=[True,True])
#print(sorted_opdf_dt.head(3))
sorted_opdf_dt['year_month']=list(zip(sorted_opdf_dt['year'].values,sorted_opdf_dt['month'].values))
#print(sorted_opdf_dt.head(3))
sorted_opdf_dt.reset_index(drop=True, inplace=True)
#print(sorted_opdf_dt.head(3))
plot_pred_vs_obs(sorted_opdf_dt,'observ','pred', 'DT Nearby_close ' )
#let's use the whole common period except the last month as new data: 2008-02 ~ 2017-11
#For example: use the data at time t-1 (year: 2008 month: 02) to predict the nearby price at year: 2008 month: 03  
data2 = data[:-1]
print(data.shape, data2.shape)
#let's use the whole common period except the first month as new label: 2008-03 ~ 2017-12
labels2 = labels[1:]
print(labels.shape,labels2.shape)
#train / rest set split
data_train2, data_test2, label_train2, label_test2 = train_test_split(data2, labels2, test_size=.2, random_state=7)
print(data_train2.shape, data_test2.shape, label_train2.shape, label_test2.shape)
#Re-train the DT model using 'new' data
dt2 = DecisionTreeRegressor(random_state=123)
dt2.fit(data_train2,label_train2)
feature_relevance2 = pd.Series(dt2.feature_importances_, index=data.columns)
#top 10 most important features for DT model prediction
feature_relevance2.sort_values(ascending=False)[:10]
label_pred_dt2 = dt2.predict(data_test2)
print(label_pred_dt2.shape, type(label_pred_dt2))
print(label_pred_dt2)
RMSE_dt2 = sqrt(mean_squared_error(y_true = label_test2, y_pred = label_pred_dt2))
RMSE_dt2
pd.DataFrame({'Decisition Tree RMSE  (data T to predict price T)':RMSE_dt,'Decisition Tree RMSE (data T-1 to predict price T)':RMSE_dt2},index=['Compare'])
obs_pred_df_dt2 = pd.DataFrame({'observ':label_test2,'pred':label_pred_dt2})
#print(obs_pred_df_dt.head(3))
opdf_dt2 = obs_pred_df_dt2.reset_index()
#print(opdf_dt.head(3))
sorted_opdf_dt2 = opdf_dt2.sort_values(['year','month'], ascending=[True,True])
#print(sorted_opdf_dt.head(3))
sorted_opdf_dt2['year_month']=list(zip(sorted_opdf_dt2['year'].values,sorted_opdf_dt2['month'].values))
#print(sorted_opdf_dt.head(3))
sorted_opdf_dt2.reset_index(drop=True, inplace=True)
#print(sorted_opdf_dt.head(3))
plot_pred_vs_obs(sorted_opdf_dt,'observ','pred', 'DT Nearby_close ' )
