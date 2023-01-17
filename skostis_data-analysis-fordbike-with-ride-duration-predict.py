import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
bike2019 =pd.read_csv('../input/ford-gobike-2019feb-tripdata/201902-fordgobike-tripdata.csv')
bike2019.head()
bike2019.describe()
bike2019.info()
bike2019['start_time']=pd.to_datetime(bike2019['start_time'])
bike2019['end_time']=pd.to_datetime(bike2019['end_time'])
bike2019['start_time'].duplicated().value_counts()
bike2019[bike2019['start_time'].duplicated()]
bike2019[bike2019.duplicated(subset=['end_time'])]
bike2019.dropna(inplace=True)
bike2019.info()
bike2019['Customer_age']=2019-bike2019['member_birth_year']
bike2019=bike2019.drop('member_birth_year',axis=1)
bike2019['duration_sec']=bike2019['duration_sec']/60
bike2019=bike2019.rename(columns={'duration_sec':'duration_min'})
bin_edges=[bike2019['duration_min'].min(),bike2019['duration_min'].mean(),bike2019['duration_min'].max()]
bin_names=['casual','long']
bike2019['general_runtime']=pd.cut(bike2019['duration_min'],bin_edges,labels=bin_names)
bin_edges=[0,18.000000,30,40,50,60,70,140]
bin_names=['underage','18s','30s','40s','50s','60s','>70']
bike2019['Customer_decade']=pd.cut(bike2019['Customer_age'],bin_edges,labels=bin_names)
bike2019['hour_start']=bike2019['start_time'].dt.hour
bike2019['day_start']=bike2019['start_time'].dt.weekday_name
bike2019['month_start']=bike2019['start_time'].dt.month
bike2019=bike2019.drop('start_time',axis=1)
bike2019=bike2019.drop('end_time',axis=1)
bike2019['hour_start'].describe()
bin_edges=[0,6,12,16,19,23]
bin_names=['aftermidnigt','morning','midday','afternoon','night']
bike2019['day_period']=pd.cut(bike2019['hour_start'],bin_edges,labels=bin_names)
bike2019.head()
bike2019.to_csv('clean_bikes.csv')
bike2019=pd.read_csv('clean_bikes.csv')
bike2019.hist(figsize=(9,8));
bike2019.info()
# there's a long tail in the distribution, so let's put it on a log scale instead
log_binsize = 0.025
bins = 10 ** np.arange(2.4, np.log10(bike2019['duration_min'].max())+log_binsize, log_binsize)

plt.figure(figsize=[8, 5])
plt.hist(data = bike2019, x = 'duration_min', bins = bins)
plt.xscale('log')

plt.xlabel('duration (min)')
plt.show()
a=bike2019[bike2019['duration_min']<60]
a.hist(['duration_min'],bins = 1000);
# let's plot all some categorical variables together to get an idea of each  variable's distribution.

fig, ax = plt.subplots(nrows=2, figsize = [8,8])

default_color = sb.color_palette()[0]
sb.countplot(data = bike2019, x = 'Customer_decade', color = default_color, ax = ax[0])
sb.countplot(data = bike2019, x = 'member_gender', color = default_color, ax = ax[1])

plt.show()
sb.regplot(data=a, x='duration_min',y='Customer_age',scatter_kws={'alpha': 1/20})
plt.figure(figsize = [8, 8])

# subplot 1: color vs cut
plt.subplot(2, 1, 1)
sb.countplot(data = bike2019, x = 'member_gender', hue = 'Customer_decade', palette = 'Blues')

# subplot 2: clarity vs. cut
ax = plt.subplot(2, 1, 2)
sb.countplot(data = bike2019, x = 'hour_start', hue = 'member_gender', palette = 'Greens')
ax.legend(ncol = 2) # re-arrange legend to reduce overlapping



plt.show()
a=bike2019[bike2019['duration_min']<60]
base_color = sb.color_palette()[3]
sb.boxplot(data = a, x='duration_min', y='member_gender', color = base_color);
y=a[a['Customer_age']<30]
base_color = sb.color_palette()[3]
sb.boxplot(data = y, x='duration_min', y='member_gender', color = base_color);
young=bike2019[bike2019['Customer_age']<30]

gender = [['Male', '^'],['Female','o'],
               ['Other', 's']]

for gen, marker in gender:
    df_gender = young[young['member_gender'] == gen]
    plt.scatter(data = df_gender, x = 'duration_min', y = 'Customer_age',
                marker = marker,alpha = 1/5)
    
plt.legend(['Male','Female','Other']);
bike2019['duration_min'].describe(),bike2019['Customer_age'].describe(),bike2019['month_start'].describe()
ccounts=bike2019.groupby(['member_gender','Customer_decade']).size()
ccounts = ccounts.reset_index(name = 'count')
ccounts = ccounts.pivot(index = 'Customer_decade', columns = 'member_gender', values = 'count')

dcounts=bike2019.groupby(['month_start','day_start']).size()
dcounts = dcounts.reset_index(name = 'count')
dcounts = dcounts.pivot(index = 'day_start', columns = 'month_start', values = 'count')
dcounts
sb.heatmap(dcounts)
hcounts=bike2019.groupby(['day_start','hour_start']).size()
hcounts = hcounts.reset_index(name = 'count')
hcounts = hcounts.pivot(index = 'hour_start', columns = 'day_start', values = 'count')
plt.figure(figsize=(14, 7))
b=sb.heatmap(hcounts,square=False)
b.set_title('Bike usage per hour')
b.set_ylabel('Hour')
b.set_xlabel('Day');
plt.figure(figsize=(14, 7))
mcount=bike2019.groupby('hour_start').size()
mcount = mcount.reset_index(name = 'count')
g=sb.barplot(data = mcount, x = 'hour_start',y='count');
g.set_title('Rides per hour')
g.set_ylabel('Rides')
g.set_xlabel('Hour');
plt.figure(figsize=(16, 6))
a=bike2019[bike2019['duration_min']<=60]
g=sb.violinplot(x=a['day_start'],y=a['duration_min'],hue='user_type',data=a, order=["Monday", "Tuesday","Wednesday","Thursday",
                                                                                    "Friday","Saturday","Sunday"],scale="width",cut=0)
g.set_title('Duration per day')
g.set_ylabel('Duration (min)')
g.set_xlabel('Day')
plt.xticks(rotation=90);
plt.figure(figsize=(14, 6))
b=sb.countplot(data = bike2019, x = 'day_start',hue='user_type',order=["Monday", "Tuesday","Wednesday","Thursday",
                                                                                  "Friday","Saturday","Sunday"]);
b.set_title('Rides amount per month seperated by rides length')
b.set_ylabel('Rides')
b.set_xlabel('Day')
plt.xticks(rotation=90)
plt.show();
plt.figure(figsize=(16, 6))
a=bike2019[bike2019['duration_min']<=60]
g=sb.violinplot(x=a['day_start'],y=a['duration_min'],hue='member_gender',order=["Monday", "Tuesday","Wednesday","Thursday",
                                                                                  "Friday","Saturday","Sunday"],data=a)
plt.xticks(rotation=90)
g.set_title('Duration per day separated by gender')
g.set_ylabel('Duration (min)')
g.set_xlabel('Day');
mcount=bike2019.groupby('start_station_name').size().sort_values().tail()
mcount = mcount.reset_index(name = 'count')
pl=sb.barplot(data = mcount, x ='start_station_name',y='count');
pl.set_title('Most popular start stations')
pl.set_ylabel('Rides')
pl.set_xlabel('Station name');
plt.xticks(rotation=90);
mcount=bike2019.groupby('end_station_name').size().sort_values().tail()
mcount = mcount.reset_index(name = 'count')
pl=sb.barplot(data = mcount, x ='end_station_name',y='count');
pl.set_title('Most popular destination stations')
pl.set_ylabel('Rides')
pl.set_xlabel('Station name');
plt.xticks(rotation=90);
X=bike2019.copy()
#https://www.ncdc.noaa.gov/cdo-web/quickdata#
X.drop(['Unnamed: 0', 'start_station_name','end_station_id','end_station_name', 'end_station_latitude', 'end_station_longitude','general_runtime'], axis=1,inplace=True)
X=X[X['user_type']=='Subscriber']
X.info()
X=pd.get_dummies(X)
corr = X.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Remove rows with missing target, separate target from predictors

y = X.duration_min              
X.drop(['duration_min'], axis=1, inplace=True)

# Break off validation set from training data

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.9, test_size=0.1, random_state=1)
X_train.info(),X_test.info()
y_val
def score_dataset(XT, XV, YT, YV,n,r):
    model= XGBRegressor(n_estimators=n,verbosity =0,learning_rate=0.1)
    model.fit(XT, YT, 
             early_stopping_rounds=r, 
             eval_set=[(XV, YV)], 
             verbose=False)
    preds = model.predict(XV)
    mae = mean_absolute_error(YV, preds)
    return mae
stopping_rounds=[5,10,50,100]
estimators=[100,500,1000,2000]
learning_rate=[0.1,0.3,0.5,0.7,1]
#imputed_results = pd.DataFrame(columns = ['stopping_rounds','estimators','MAE'])

#for n in estimators: 
#    for r in stopping_rounds: 
#        m=score_dataset(X_train, X_val, y_train, y_val,n,r)
#        imputed_results = imputed_results.append({"stopping_rounds":r,"estimators":n,"MAE":m},ignore_index=True)
#imputed_results.sort_values(by=['MAE'])
model= XGBRegressor(n_estimators=500,verbosity =0,learning_rate=0.1)
model.fit(X_train, y_train, 
        early_stopping_rounds=50, 
          eval_set=[(X_val, y_val)],verbose=False)
preds = model.predict(X_val)
mae = mean_absolute_error(y_val, preds)
print(mae)
model= XGBRegressor(n_estimators=500,verbosity =0,learning_rate=0.1)
model.fit(X_train, y_train, 
             early_stopping_rounds=50, 
             eval_set=[(X_val, y_val)],verbose=False)
    
final_predictions=model.predict(X_test)
#final_predictions=round(final_predictions,2)
#y_test=round(y_test,2)
X_test['prediction']=final_predictions.astype('float64')
X_test['actual_duration']=y_test
X_test.info()
labels=testval['Id']
men_means = X_test['prediction']
women_means = y_test

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(100,20))
rects1 = ax.bar(x - width/2, men_means, width, label='prediction')
rects2 = ax.bar(x + width/2, women_means, width, label='actual_duration')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Duration')
ax.set_title('Actual and predicted ride duration')
ax.set_xticks(x)

ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()


