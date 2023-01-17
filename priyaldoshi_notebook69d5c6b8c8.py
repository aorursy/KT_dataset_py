import pandas as pd,numpy as np  
import matplotlib.pyplot as plt
import itertools as it
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')

# sklearn and models
from sklearn import preprocessing, ensemble, metrics, feature_selection, model_selection, pipeline
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
from numpy import sqrt
from numpy import argmax
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
train =  pd.read_csv("/kaggle/input/dataset/train.csv")
test =  pd.read_csv("/kaggle/input/dataset/test.csv")
weather =  pd.read_csv("/kaggle/input/dataset/weather.csv")
spray =  pd.read_csv("/kaggle/input/dataset/spray.csv")
# Aggregate train data  since nummosquitos is split into multiple rows if they are greater than 50  
train  =  pd.DataFrame(train.groupby(by=[x for x in train.columns if x != 'NumMosquitos'])['NumMosquitos'].sum()).reset_index()
train['Train_Ind'] = 1
test['Train_Ind'] = 0
# Merge Train test data  
idata =  pd.concat([train,test], axis=0, ignore_index=True)
idata.drop(columns = ['Address','AddressNumberAndStreet','AddressAccuracy','NumMosquitos'],inplace=True)
# Species most likely carriers of WNV 
species  = train.groupby('Species')['WnvPresent'].sum().reset_index()
species_wnv  =  np.unique(species[species['WnvPresent'] > 0]['Species'])
species
# Positive traps with virus per week 
idata['Date'] = pd.to_datetime(idata['Date'])
idata['Year'] = idata['Date'].map(lambda x: x.year)
idata['Week'] = idata['Date'].map(lambda x: x.week)
train =  idata[idata['Train_Ind'] ==1]
traps = train[['Species','WnvPresent','Date','Trap','Year','Week']].groupby(['Trap','Date','Year','Week','Species'])['WnvPresent'].max().reset_index()
traps[:3]
# High class imbalance  
fig,  ax = plt.subplots(1,1)
sns.countplot(train['WnvPresent'], ax = ax)
for i in ax.patches:
    height = i.get_height()
    ax.text(i.get_x()+i.get_width()/2,height,'{:.2f}{}'.format(height/len(train)*100,'%'))
plt.show()
traps = traps[['Year','Week','WnvPresent','Species']]
checks = traps[['Week', 'Year','WnvPresent']].groupby(['Week','Year']).count().reset_index()

weekly_postives = traps.groupby(['Year','Week', 'Species']).sum().reset_index()
weekly_postives_species = weekly_postives.set_index(['Year','Week', 'Species']).unstack()
weekly_postives_species.columns = weekly_postives_species.columns.get_level_values(1)
weekly_postives_species['total_positives'] = weekly_postives_species.sum(axis=1)
weekly_postives_species = weekly_postives_species.reset_index().fillna(0)
    
weekly_checks = checks.groupby(['Year','Week']).sum()
weekly_checks.columns = ['checks']
weekly_checks = weekly_checks.reset_index()
weekly_checks['positive'] = weekly_postives_species['total_positives']
weekly_checks['trap_infection_rate'] = weekly_checks['positive'] / weekly_checks['checks'] * 100
weekly_checks_years = weekly_checks.pivot(index='Week', columns='Year', values='trap_infection_rate')

ax = weekly_checks_years.interpolate().plot(title='Trap infection rate', figsize=(10,6))
ax.set_ylabel('Perctenage traps infected');
plt.savefig('postive_trap_rate.png')

fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True, sharex=True)
fig.tight_layout()
axit = (ax for ax in it.chain(*axes))

for m, group in weekly_postives_species.groupby('Year'):
    ax = next(axit); ax.xaxis.grid(); ax.yaxis.grid()
    for species in species_wnv:
        ax.plot(group['Week'], group[species], label=species)
    ax.legend(loc='upper left');
    ax.set_title(m)
plt.savefig('postive_traps.png')
# WNV starts adn rises from June  - october  ( approx 5-6 months of the year)  during summer time 
# In trap ,if we find any species other than above 3 identified , can be safely assumed they wont be having WNV 
# Insight  , 2007 and 2013 has seen the highest no of WNV across all traps and areas 
# Mosquito species PIPIENS/RESTUANS is seen the most dominant one across years  
weather['Date'] = pd.to_datetime(weather['Date'])
# Replace missing value with  NA 
weather.replace(dict.fromkeys([' ','M','-'],np.NaN),inplace=True)
# weather  =  weather.replace('T', 0.005)
weather[:10]
# Impute Precipitation
# precip_med =  weather[weather['PrecipTotal'] != '  T'].PrecipTotal.median()
precip_med = 0.05

precip_totals = []
for val in weather.PrecipTotal:
    if val == '  T':
        precip_totals.append(precip_med)
    else:
        precip_totals.append(val)

weather.PrecipTotal = pd.to_numeric(precip_totals)
# Summarizing key statistics 
stats = pd.concat([weather[weather['Station']==1].dtypes,weather.isna().sum(),pd.DataFrame(weather[weather['Station']==1].describe(include='all').T).loc[:,['unique','top','freq','mean','std','min','max','50%']]],axis=1)
stats.columns = ['dtype','missing_count','unique','top','freq','mean','std','min','max','median']
stats = stats.reindex(columns=['dtype','missing_count','unique','top','freq','mean','median','std','min','max'])
stats.index.name = 'Features'
stats.sort_values(by='dtype')
stats
# Drop columns with no information from weather data   
# 1. Water1  is has only null values 
# 2. Depth has only constant value 
# 3. Snowfall has only 2 unique values , 50% data is missing , remaining data majority has only 0 value  
weather.drop(columns=['Water1','Depth','SnowFall'],inplace=True)
weather.columns
removeelements =  list(['Station', 'Date'])
imputeweathercols  =  [ele for ele in list(weather.columns) if ele not in removeelements]
weather[imputeweathercols] =  weather[imputeweathercols].fillna(method = 'ffill')
# time conversion lambda function
time_func = lambda x: pd.Timestamp(pd.to_datetime(x, format = '%H%M'))
hours_RiseSet_func = lambda x: x.minute/60.0 + float(x.hour)

weather.Sunset = weather.Sunset.replace('\+?60', '59', regex = True)
weather.Sunrise = weather.Sunrise.apply(time_func)
weather.Sunset = weather.Sunset.apply(time_func)
weather['DayDuration'] = (weather.Sunset - weather.Sunrise).astype('timedelta64[m]')/60
weather['NightDuration'] = 24 - weather['DayDuration']
weather['Sunrise_hours'] = weather.Sunrise.apply(hours_RiseSet_func)
weather['Sunset_hours'] = weather.Sunset.apply(hours_RiseSet_func)
weather['Tmax_Tmin'] = weather.Tmax-weather.Tmin
weather[:3]
# new_df.dtypes
cols_to_change = ['Tmax', 'Tmin', 'Tavg', 'Depart', 'DewPoint','WetBulb', 'Heat', 'Cool',
                  'PrecipTotal', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir',
                  'AvgSpeed', 'DayDuration', 'NightDuration','Tmax_Tmin'] # columns of type object that can be coerced to numeric values
for col in cols_to_change:
    weather[col] = pd.to_numeric(weather[col])#changing columns above to numeric
weathers1 =  weather[weather['Station']==1]
weathers1.drop(columns='Station',inplace=True)
findata =  pd.merge(idata,weathers1,on = 'Date',how='left')
# binning the data 
bins = [0,17,18,19,20]
labels = [1,2,3,4]
findata['Sunset_bin'] = pd.cut(findata['Sunset_hours'], bins=bins, labels=labels)

bins = [0,5,6,8]
labels = [1,2,3]
findata['Sunrise_bin'] = pd.cut(findata['Sunrise_hours'], bins=bins, labels=labels)
set(list(pd.Series(findata['CodeSum']).str.cat(sep=' ').split(" ")))
dummies = findata['CodeSum'].str.get_dummies(sep=' ')
findata = pd.concat([findata, dummies], axis=1)

# fields_to_drop = ['CodeSum']
# findata.drop(fields_to_drop, axis=1,inplace=True)
corvars =  ['WnvPresent','Tmax', 'Tmin', 'Tavg',
       'Depart', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset',
       'PrecipTotal', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir',
       'AvgSpeed', 'DayDuration', 'NightDuration','Tmax_Tmin']
fig, ax = plt.subplots(1,1, figsize=(20,10))
sns.heatmap(findata[corvars].corr(), annot=True, cmap="YlGnBu" ,ax = ax)
plt.show()

### weather variables like temp 
# Relation of temperature and wnv virus 
fig, ax = plt.subplots(1,1, figsize=(20,5))
findatatrain = findata[findata['Train_Ind']==1]
sns.distplot(findatatrain.loc[findatatrain.Year==2007,'Tavg'].apply(int), ax=ax,color='g',hist=False)
sns.distplot(findatatrain.loc[findatatrain.Year==2009,'Tavg'].apply(int), ax=ax,color='b',hist=False)
sns.distplot(findatatrain.loc[findatatrain.Year==2011,'Tavg'].apply(int), ax=ax,color='r',hist=False)
sns.distplot(findatatrain.loc[findatatrain.Year==2013,'Tavg'].apply(int), ax=ax,color='y',hist=False)
ax.xaxis.set(ticks=np.arange(40,90,5))
ax.set_title('Tavg')

plt.show()
# Insight : The higher the temperature higher the WNV carriers 
# Temprature plots are left skewed , showing the avg temperature is higher most of the times across hi
f, axes = plt.subplots(4,4, figsize=(20,5))

sns.distplot(findatatrain["Tmin"] , color="skyblue", ax=axes[0,0])
axes[0,0].set_title('Tmin')
sns.distplot(findatatrain["Tmax"] , color="olive", ax=axes[0,1])
axes[0,1].set_title('Tmax')
sns.distplot(findatatrain["Tavg"] , color="gold", ax=axes[0,2])
axes[0,2].set_title('Tavg')
sns.distplot(findatatrain["Depart"] , color="teal", ax=axes[0,3])
axes[0,3].set_title('Depart')
sns.distplot(findatatrain["DewPoint"] , color="skyblue", ax=axes[1,0])
axes[1,0].set_title('DewPoint')
sns.distplot(findatatrain["WetBulb"] , color="olive", ax=axes[1,1])
axes[1,1].set_title('WetBulb')
sns.distplot(findatatrain["Cool"] , color="gold", ax=axes[1,2])
axes[1,2].set_title('Cool')
sns.distplot(findatatrain["PrecipTotal"] , color="teal", ax=axes[1,3])
axes[1,3].set_title('PrecipTotal')
sns.distplot(findatatrain["StnPressure"] , color="skyblue", ax=axes[2,0])
axes[2,0].set_title('StnPressure')
sns.distplot(findatatrain["SeaLevel"] , color="olive", ax=axes[2,1])
axes[2,1].set_title('SeaLevel')
sns.distplot(findatatrain["ResultSpeed"] , color="gold", ax=axes[2,2])
axes[2,2].set_title('ResultSpeed')
sns.distplot(findatatrain["ResultDir"] , color="teal", ax=axes[2,3])
axes[2,3].set_title('ResultDir')
sns.distplot(findatatrain["AvgSpeed"] , color="skyblue", ax=axes[3,0])
axes[3,0].set_title('AvgSpeed')
sns.distplot(findatatrain["DayDuration"] , color="olive", ax=axes[3,1])
axes[3,1].set_title('DayDuration')
sns.distplot(findatatrain["NightDuration"] , color="gold", ax=axes[3,2])
axes[3,2].set_title('NightDuration')
# sns.distplot(findatatrain["Heat"] , color="gold", ax=axes[3,3])
# axes[3,3].set_title('Heat')
# which street has the most WNV carriers 
fig, ax = plt.subplots(1,1, figsize = (20,5))
wnvstreet =  findatatrain.groupby(['Street'])['WnvPresent'].sum().reset_index()
wnvstreet.sort_values('WnvPresent',inplace=True,ascending=False)
wnvhighstreet =  wnvstreet[wnvstreet['WnvPresent']>5]
sns.pointplot(x="Street", y="WnvPresent",kind="point", data=wnvhighstreet,ax=ax)
ax.tick_params('x',labelrotation=90, labelsize='small', )
ax.set_title('Virus spread in streets in 2007, 2009, 2011 & 2013', fontdict = {'fontsize':20})
plt.legend()
plt.show()
# Effect of weather on Virus
print(f'Count of distinct weather condition: {findatatrain["CodeSum"].nunique()}')
fig, ax = plt.subplots(1,1, figsize=(20,5))
tmp = pd.crosstab(findatatrain['CodeSum'],findatatrain['WnvPresent'],normalize='index')
tmp.reset_index()

g = sns.countplot(findatatrain['CodeSum'],order = list(tmp.index), ax =ax)
g.tick_params(axis='x',labelrotation=90)
g.set_title('Effect of Weather Condition on Virus', fontsize=20)
g1 = g.twinx()
sns.pointplot(tmp.index,tmp[1], ax=g1)

plt.show()
# Effect of sunrise and sunset on Virus
print(f'Count of distinct weather condition: {findatatrain["Sunrise_bin"].nunique()}')
fig, ax = plt.subplots(1,1, figsize=(20,5))
tmp = pd.crosstab(findatatrain['Sunrise_bin'],findatatrain['WnvPresent'],normalize='index')
tmp.reset_index()

g = sns.countplot(findatatrain['Sunrise_bin'],order = list(tmp.index), ax =ax)
g.tick_params(axis='x',labelrotation=90)
g.set_title('Effect of Sunrise on Virus', fontsize=20)
g1 = g.twinx()
sns.pointplot(tmp.index,tmp[1], ax=g1)

plt.show()
# Effect of sunrise and sunset on Virus
print(f'Count of distinct weather condition: {findatatrain["Sunset_bin"].nunique()}')
fig, ax = plt.subplots(1,1, figsize=(20,5))
tmp = pd.crosstab(findatatrain['Sunset_bin'],findatatrain['WnvPresent'],normalize='index')
tmp.reset_index()

g = sns.countplot(findatatrain['Sunset_bin'],order = list(tmp.index), ax =ax)
g.tick_params(axis='x',labelrotation=90)
g.set_title('Effect of Sunset on Virus', fontsize=20)
g1 = g.twinx()
sns.pointplot(tmp.index,tmp[1], ax=g1)

plt.show()
# Effect of sunrise and sunset on Virus
print(f'Count of distinct weather condition: {findatatrain["PrecipTotal"].nunique()}')
fig, ax = plt.subplots(1,1, figsize=(20,5))
tmp = pd.crosstab(findatatrain['PrecipTotal'],findatatrain['WnvPresent'],normalize='index')
tmp.reset_index()

g = sns.countplot(findatatrain['PrecipTotal'],order = list(tmp.index), ax =ax)
g.tick_params(axis='x',labelrotation=90)
g.set_title('Effect of Precipitation on Virus', fontsize=20)
g1 = g.twinx()
sns.pointplot(tmp.index,tmp[1], ax=g1)

plt.show()
# Effect  of Wind Speed and Dir
fig, (ax1) = plt.subplots(1,1, figsize=(20,5))
sns.boxplot(findatatrain['Year'], findatatrain['ResultSpeed'].replace(np.NaN,0.0).apply(float), hue = findatatrain['WnvPresent'], ax=ax1)
# ax1.set_ylim(28,30)
ax1.set_title('ResultSpeed', fontdict= {'fontsize':15})
plt.show()
# Effect  of Wind Speed and Dir
fig, (ax1) = plt.subplots(1,1, figsize=(20,5))
sns.boxplot(findatatrain['Year'], findatatrain['ResultDir'].replace(np.NaN,0.0).apply(float), hue = findatatrain['WnvPresent'], ax=ax1)
# ax1.set_ylim(28,30)
ax1.set_title('ResultDir', fontdict= {'fontsize':15})
plt.show()
# Effect  of station pressure 
fig, (ax1) = plt.subplots(1,1, figsize=(20,5))
sns.boxplot(findatatrain['Year'], findatatrain['StnPressure'].replace(np.NaN,0.0).apply(float), hue = findatatrain['WnvPresent'], ax=ax1)
# ax1.set_ylim(28,30)
ax1.set_title('StnPressure', fontdict= {'fontsize':15})
plt.show()
# Effect  of sea level
fig, (ax1) = plt.subplots(1,1, figsize=(20,5))
sns.boxplot(findatatrain['Year'], findatatrain['SeaLevel'].replace(np.NaN,0.0).apply(float), hue = findatatrain['WnvPresent'], ax=ax1)
# ax1.set_ylim(28,30)
ax1.set_title('SeaLevel', fontdict= {'fontsize':15})
plt.show()
day_func= lambda x: x.day
day_of_year_func = lambda x: x.dayofyear
week_of_year_func = lambda x: x.week
month_func = lambda x: x.month

# train
findata['month'] = findata.Date.apply(month_func)
findata['day'] = findata.Date.apply(day_func)
findata['day_of_year'] = findata.Date.apply(day_of_year_func)
findata['week'] = findata.Date.apply(week_of_year_func)
# Create dummy categorical variables '
dummy = ['Species','Block','Street','Trap','Sunset_bin','Sunrise_bin']
for each in dummy:
    dummies = pd.get_dummies(findata[each], prefix=each, drop_first=False)
    findata = pd.concat([findata, dummies], axis=1)

# TODO: Drop the previous rank column
fields_to_drop = ['Species','Block','Street','Trap','Sunset_bin','Sunrise_bin']
findata.drop(fields_to_drop, axis=1,inplace=True)
# tagging risk months from July to september 
findata['RiskMonth'] = np.where(findata.month < 7 , 1, np.where(findata.month > 9,1,0))
findata['RainyDayInd'] = np.where(findata['CodeSum'].str.contains(u'RA') == True,1,0)
findata['DryWet'] = findata['DewPoint']-findata['WetBulb']
findata.drop(columns= ['CodeSum','Id'],inplace=True)
mrd =  findata.drop(columns=['Date','Sunrise','Sunset','Sunrise_hours','Sunset_hours','WnvPresent'])
mrdtrain  =  mrd[mrd['Train_Ind']==1]
mrdtrain.drop(columns=['Train_Ind'],inplace=True)
mrdtest  =  mrd[mrd['Train_Ind']==0]
mrdtest.drop(columns=['Train_Ind'],inplace=True)
# get label
labels = findata[findata['Train_Ind']==1].pop('WnvPresent').values
train_split, val_split, label_train_split, label_val_split = model_selection.train_test_split(mrdtrain, 
                                      labels, test_size = 0.33, random_state = 42, stratify= labels)
clf = ensemble.RandomForestClassifier(random_state= 42)
clf.fit(train_split,label_train_split)
# create predictions and submission file
val_pred = clf.predict_proba(val_split)[:,1]
# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')
fpr, tpr, thresholds = metrics.roc_curve(label_val_split,val_pred)
roc_auc = metrics.auc(fpr, tpr)
scores = [f1_score(label_val_split,to_labels(val_pred, t)) for t in thresholds]
ix = argmax(scores)
print('Best Threshold=%f, FScore=%.3f' % (thresholds[ix], scores[ix]))

print('roc_auc for Validation Dataset',roc_auc)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
val_predlabels =  to_labels(val_pred,thresholds[ix])
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(label_val_split, val_predlabels) # Calulate Confusion matrix for test set.
print(cm)
print(accuracy_score(label_val_split, val_predlabels))
print(classification_report(label_val_split, val_predlabels))
# Hyperparameter tuning 
from sklearn.model_selection import RandomizedSearchCV
# number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# number of features at every split
max_features = ['auto', 'sqrt']
# max depth
max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
max_depth.append(None)
# create random grid
random_grid = {
 'n_estimators': n_estimators,
 'max_features': max_features,
 'max_depth': max_depth
 }
# Random search of parameters
rfc_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the model
rfc_random.fit(train_split,label_train_split)
# print results
print(rfc_random.best_params_)
# predictions for test data 
test_pred = clf.predict_proba(mrdtest)[:,1]