import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

sns.set_style('darkgrid')

warnings.filterwarnings('ignore')

df=pd.read_csv('../input/kobe-bryant-shot-selection/data.csv', header=0,sep=',')
df.shot_made_flag.value_counts()
df.info()
plt.figure(figsize=(15,5))

df.shot_made_flag.value_counts().plot(kind='bar')

plt.title('misses vs baskets')

plt.show()
#missing values



df.shot_made_flag.isnull().sum()
# analysis of missing values

plt.figure(figsize=(18,5))

sns.heatmap(df.isnull())
# no missing values except shot_made_flag column, which is our target.

# rows with null values form the test data, the remaining part is our training data.
#lat and lon columns



plt.figure(figsize=(15,6))

plt.subplot(121)

plt.scatter(df.lon, df.lat, marker='x', alpha=0.1)

plt.title('scatter plot of lon and lat values')

plt.subplot(122)

plt.scatter(df.loc_x, df.loc_y, marker='x', alpha=0.1)

plt.title('scatter plot of loc_x and loc_y values')
# It seems that lat and lon values are linear transformation of loc_x and loc_y values, so delete lat and lon columns



if 'lat' in df.columns:

    df.drop(labels='lat',axis=1,inplace=True)

if 'lon' in df.columns:

    df.drop(labels='lon',axis=1,inplace=True)
#team_name

df.team_name.value_counts()
# team_name column contains just one name, so drop this column



if 'team_name' in df.columns:

    df.drop(labels='team_name', inplace=True, axis=1)
#shot_id



# the column shot_id is almost the same as the index of the dataframe, so delete it



if 'shot_id' in df.columns:

    df.drop(labels='shot_id', inplace=True, axis=1)
# game_id and game_event_id



df.game_id.value_counts().head(10)
# we see that the game_id 21501228 is repeated 50 times in game_id column.

# This means that  Kobe had 50 attempts in this match and corresponding to each attempt we have one id 

# in game_event_id column. By game_id and game event_id we can order the attempts. But such information

# may lead to data leakage. I think we cannot use general performance information in one match to

# predict whether an attempt in that match is a basket or miss. So delete game_id and game_event_id



if 'game_id' in df.columns:

    df.drop(labels='game_id', inplace=True, axis=1)

if 'game_event_id' in df.columns:

    df.drop(labels='game_event_id', inplace=True, axis=1)
#matchup and opponent



df.matchup.head(20)
# check whether each matchup starts with 'LAL' or not

# maybe this column gives an information about the location of match, home or away?



df.matchup.str.startswith('LAL').value_counts()
# every matchup starts with LAL. However, every matchup has two kind of separators: @ or vs.

# A little  internet search gives that 'vs.' means Lakers is home team





df['home_or_away']=df.matchup.apply(lambda x: 'home' if x.find('@')==-1 else 'away')

df['home_or_away']=df['home_or_away'].astype('category')
# check whether  the second part of matchup column matches with opponent column



def second_part(x):

    # x:string

    if x.find('@')>=0:

        return x.split('@')[1].strip().upper()

    else:

        return x.split('vs.')[1].strip().upper()

    





(df.matchup.apply(func=second_part).values!=df.opponent.str.upper().values).sum()        


df.matchup.apply(func=second_part).unique()
df.opponent.unique()
set(df.matchup.apply(func=second_part).unique()).difference(df.opponent.unique())
set(df.opponent.unique()).difference(df.matchup.apply(func=second_part).unique())
# there are a few  differences in abbreviations, maybe due to some name changes of nba teams in time

# we cannot gain any real information from the matchup column, opponent column is sufficient, so delete the former one



if 'matchup' in df.columns:

    df.drop(labels='matchup', axis=1,inplace=True)
average_successful_attempt=df.shot_made_flag.mean()

plt.figure(figsize=(15,6))

df.groupby('opponent')['shot_made_flag'].mean().plot(kind='bar')

plt.hlines(average_successful_attempt, 0,33, colors='red')

plt.title('average successful attempt against each opponent')

plt.show()
#convert opponent column to a categorical variable

df['opponent']=df['opponent'].astype('category')
#team_id 

df.team_id.value_counts()
# since team_id feature is constant, delete it



if 'team_id' in df.columns:

    df.drop(labels='team_id', axis=1,inplace=True)
#shot_type

df.shot_type.value_counts()
plt.figure(figsize=(15,5))

sns.countplot('shot_made_flag',hue='shot_type',data=df[df.shot_made_flag.notnull()])
# rate of successfull attempts in two categories of shot-type

plt.figure(figsize=(15,6))

df.groupby('shot_type')['shot_made_flag'].mean().plot(kind='bar')

plt.hlines(average_successful_attempt, -5,30, colors='red',linestyles='dashed')

plt.show()
#convert shot_type into categorical variable



df['shot_type']=df.shot_type.astype('category')
#game_date and season



# convert game_date column to datetime object



df['game_date']=pd.to_datetime(df['game_date'])
#convert season into categorical variable



df['season']=df.season.astype('category')
# rate of successfull attempts in each season

plt.figure(figsize=(15,6))

df.groupby('season')['shot_made_flag'].mean().plot(kind='bar')

plt.hlines(average_successful_attempt, -5,30, colors='red',linestyles='dashed')

plt.title('rate of successfull attempts in each season')

plt.show()
#convert game_date into new categorical variables such as year, day, month,...



df['weekofyear']=df.game_date.apply(lambda x:x.weekofyear)

df['dayofweek']=df.game_date.apply(lambda x:x.dayofweek)

df['year']=df.game_date.apply(lambda x:x.year)

df['month']=df.game_date.apply(lambda x:x.month)



df['weekofyear']=df['weekofyear'].astype('category')

df['dayofweek']=df['dayofweek'].astype('category')

df['year']=df['year'].astype('category')

df['month']=df['month'].astype('category')

#drop game_date column



if 'game_date' in df.columns:

    df.drop(labels='game_date',axis=1,inplace=True)



#shot_zone_range



df.shot_zone_range.value_counts()
plt.figure(figsize=(15,5))

sns.countplot('shot_zone_range',hue='shot_made_flag',data=df[df.shot_made_flag.notnull()])

plt.title('misses and baskets from each zone_range')

plt.show()
plt.figure(figsize=(15,6))



for zone in df.shot_zone_range.unique():

    plt.scatter(df.loc[df.shot_zone_range==zone,'loc_x'],df.loc[df.shot_zone_range==zone,'loc_y'], label="{}".format(zone))

    tx,ty=df.loc[df.shot_zone_range==zone,'loc_x'].mean(),df.loc[df.shot_zone_range==zone,'loc_y'].mean()

    avg_basket=df.loc[df.shot_zone_range==zone,'shot_made_flag'].mean()

    plt.text(tx,ty+50,round(avg_basket,3),bbox=dict(facecolor='white', alpha=0.5))



plt.title("average successful attempts from each zone")

plt.legend()

plt.show()
#convert shot_zone_range into categorical variable



df['shot_zone_range']=df['shot_zone_range'].astype('category')
#shot_zone_basic



df.shot_zone_basic.value_counts()
plt.figure(figsize=(15,5))

sns.countplot('shot_zone_basic',hue='shot_made_flag',data=df[df.shot_made_flag.notnull()])

plt.title('misses and baskets from each zone_basic')

plt.show()
plt.figure(figsize=(10,10))



for zone in df.shot_zone_basic.unique():

    plt.scatter(df.loc[df.shot_zone_basic==zone,'loc_x'],df.loc[df.shot_zone_basic==zone,'loc_y'], label="{}".format(zone))

    tx,ty=df.loc[df.shot_zone_basic==zone,'loc_x'].mean(),df.loc[df.shot_zone_basic==zone,'loc_y'].mean()

    avg_basket=df.loc[df.shot_zone_basic==zone,'shot_made_flag'].mean()

    if zone=='Above the Break 3' or zone=='Mid-Range':

        plt.text(tx,ty+70,round(avg_basket,3),bbox=dict(facecolor='white', alpha=0.5))

    else:

        plt.text(tx,ty,round(avg_basket,3),bbox=dict(facecolor='white', alpha=0.5))



plt.title("average successful attempts from each zone")

plt.legend()

plt.show()
#convert shot_zone_basic into categorical variable



df['shot_zone_basic']=df['shot_zone_basic'].astype('category')

#shot_zone_area



df.shot_zone_area.value_counts()
plt.figure(figsize=(15,5))

sns.countplot('shot_zone_area',hue='shot_made_flag',data=df[df.shot_made_flag.notnull()])

plt.title('misses and baskets from each zone_area')

plt.show()
plt.figure(figsize=(10,10))



for zone in df.shot_zone_area.unique():

    plt.scatter(df.loc[df.shot_zone_area==zone,'loc_x'],df.loc[df.shot_zone_area==zone,'loc_y'], label="{}".format(zone))

    tx,ty=df.loc[df.shot_zone_area==zone,'loc_x'].mean(),df.loc[df.shot_zone_area==zone,'loc_y'].mean()

    avg_basket=df.loc[df.shot_zone_area==zone,'shot_made_flag'].mean()

    plt.text(tx,ty+30,round(avg_basket,3),bbox=dict(facecolor='white', alpha=0.5))

    

plt.title("average successful attempts from each zone area")

plt.legend()

plt.show()
#convert shot_zone_area into categorical variable



df['shot_zone_area']=df['shot_zone_area'].astype('category')
#action_type



df.action_type.value_counts()
# rate of successfull attempts in each action_type

plt.figure(figsize=(15,6))

df.groupby('action_type')['shot_made_flag'].mean().plot(kind='bar')

plt.hlines(average_successful_attempt, -5,60, colors='red',linestyles='dashed')

plt.title('rate of successfull attempts in each action_type')

plt.show()
#convert shot_type into categorical variable



df['action_type']=df.action_type.astype('category')
#combined_shot_type



df.combined_shot_type.value_counts()
plt.figure(figsize=(15,5))

sns.countplot(x='combined_shot_type',hue='shot_made_flag',data=df[df.shot_made_flag.notnull()])
# rate of successfull attempts in each combined_shot_type

plt.figure(figsize=(15,6))

df.groupby('combined_shot_type')['shot_made_flag'].mean().plot(kind='bar')

plt.hlines(average_successful_attempt, -5,60, colors='red',linestyles='dashed')

plt.title('rate of successfull attempts in each combined_shot_type')

plt.show()
#convert combined_shot_type into categorical variable



df['combined_shot_type']=df.combined_shot_type.astype('category')
#loc_x and loc_y



# locx and locy are the coordinates on the court where the shot took place.



df.loc_x.describe()
df.loc_y.describe()
plt.figure(figsize=(15,6))

plt.scatter(df.loc_x, df.loc_y, marker='x', alpha=0.1)

plt.title('locations of attempts of Kobe')

plt.show()
# The boundary of 3pt is very clear from the above plot. As the distance from the basket increases, the number of attempts decreases

#as expected.



plt.figure(figsize=(15,6))

sns.scatterplot(x='loc_x',y='loc_y', hue='shot_made_flag',data=df[df.shot_made_flag.notnull()],alpha=0.2)

plt.show()
plt.figure(figsize=(15,6))

plt.subplot(121)

plt.scatter(df.loc[df.shot_made_flag==0,'loc_x'], df.loc[df.shot_made_flag==0,'loc_y'], marker='x', alpha=0.3)

plt.title('locations of misses of Kobe')

plt.ylim(-100,800)

plt.subplot(122)

plt.scatter(df.loc[df.shot_made_flag==1,'loc_x'], df.loc[df.shot_made_flag==1,'loc_y'], marker='x', alpha=0.3)

plt.title('locations of baskets of Kobe')

plt.ylim(-100,800)

plt.show()
#minutes_remaining and seconds_remaining



df.minutes_remaining.value_counts()
df.seconds_remaining.describe()
#instead of these two columns, create one column showing remaining time to the end of a period

#in terms of seconds



df['total_seconds_remaining']=df[['minutes_remaining','seconds_remaining']].apply(lambda x:x[0]*60+x[1], axis=1).values

bins_=[0]+list(np.linspace(6,715,71))

df['time_intervals']=pd.cut(df.total_seconds_remaining,bins=bins_,labels=list(range(1,72))).values

plt.figure(figsize=(16,6))

df.groupby('time_intervals')['shot_made_flag'].mean().plot(kind='line')

plt.show()
#there is a sharp decrease in the last 5 seconds of periods, no pattern in the remaining parts



df['in_last_five_seconds']=[1 if val==1 else 0 for val in df.time_intervals.values]

df['in_last_five_seconds']=df['in_last_five_seconds'].astype('category')





if 'minutes_remaining' in df.columns:

    df.drop(labels='minutes_remaining',axis=1,inplace=True)

if 'seconds_remaining' in df.columns:

    df.drop(labels='seconds_remaining',axis=1,inplace=True)

if 'time_intervals' in df.columns:

    df.drop(labels='time_intervals',axis=1,inplace=True)

    

    

#period and playoffs



df.period.value_counts()
# rate of successfull attempts in each period

plt.figure(figsize=(15,6))

df.groupby('period')['shot_made_flag'].mean().plot(kind='bar')

plt.hlines(average_successful_attempt, -5,60, colors='red',linestyles='dashed')

plt.title('rate of successfull attempts in each period')

plt.show()
df.playoffs.value_counts()
# rate of successfull attempts in two different terms

plt.figure(figsize=(15,6))

df.groupby('playoffs')['shot_made_flag'].mean().plot(kind='bar')

plt.hlines(average_successful_attempt, -5,60, colors='red',linestyles='dashed')

plt.title('rate of successfull attempts in normal term/playoffs')

plt.show()


df['period']=df['period'].astype('category')

df['playoffs']=df['playoffs'].astype('category')

#shot_distance



df.shot_distance.describe()
df.shot_distance.nunique()
df.shot_distance.value_counts().head(15)
# rate of successfull attempts in each distance

plt.figure(figsize=(15,6))

df.groupby('shot_distance')['shot_made_flag'].mean().plot(kind='bar')

plt.hlines(average_successful_attempt, -5,60, colors='red',linestyles='dashed')

plt.title('rate of successfull attempts in each distance')

plt.show()
df['shot_made_flag']=df['shot_made_flag'].astype('category')

# the last preparation before machine learning algorithms

#convert categorical variables into dummies



df_with_dummies=pd.get_dummies(df.drop(labels='shot_made_flag',axis=1),drop_first=True)

xtrain=df_with_dummies[df.shot_made_flag.notnull()]

ytrain=df.shot_made_flag[df.shot_made_flag.notnull()].values

test=df_with_dummies[df.shot_made_flag.isnull()]
# RANDOM FOREST



from sklearn.ensemble import RandomForestClassifier



# tuning the parameters of  random forest



from collections import OrderedDict



# aim is to optimize 'name_of_parameter'

def optimization_of_parameter_of_rf(X,y,dict_of_param,name_of_parameter, list_of_values, min_estimators, max_estimators):

    list_of_parameter_dicts=[(value,{**dict_of_param,**{'n_estimators':100, 'warm_start':True, 'oob_score':True, 'n_jobs':-1,

                               'random_state':434,name_of_parameter:value}}) for value in list_of_values]

    

    #if x and y are two dictionaries, {**x,**y} is a way of merging these two

    

    ensemble_clfs = [( "{x}={y}".format(x=name_of_parameter, y=value), RandomForestClassifier(**param_dict))

                 for value, param_dict in list_of_parameter_dicts]



    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.

    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)







    for label, clf in ensemble_clfs:

        for i in range(min_estimators, max_estimators + 1):

            clf.set_params(n_estimators=i)

            clf.fit(X, y)



            # Record the OOB error for each `n_estimators=i` setting.

            oob_error = 1 - clf.oob_score_

            error_rate[label].append((i, oob_error))



    # Generate the "OOB error rate" vs. "n_estimators" plot.

    plt.figure(figsize=(15,8))

    for label, clf_err in error_rate.items():

        xs, ys = zip(*clf_err)

        plt.plot(xs, ys, label=label)



    plt.xlim(min_estimators, max_estimators)

    plt.xlabel("n_estimators")

    plt.ylabel("OOB error rate")

    plt.legend(loc="upper right")

    plt.show()
#tune the parameter max_features



optimization_of_parameter_of_rf(xtrain.values, ytrain,{'min_samples_leaf':9},'max_features',[50,70,90],30,250)
#tune the parameter min_samples_leaf   



optimization_of_parameter_of_rf(xtrain.values, ytrain,{'max_features':50},'min_samples_leaf',[5,9,11,13],30,250)
#tune the parameter max_depth



optimization_of_parameter_of_rf(xtrain.values, ytrain,{'max_features':50, 'min_samples_leaf':13},'max_depth',[15,20,25],30,250)
#set up random forest model with the above parameters



rfc=RandomForestClassifier(n_estimators=400,max_features=50,min_samples_leaf=13, max_depth=20)



rfc.fit(xtrain.values, ytrain)



preds=rfc.predict_proba(test)

# prepare a file containing probabilities of being 1 ( shot made) of every shot in test data



preds_df=pd.DataFrame({'shot_made_flag':preds[:,1]},index=df[df.shot_made_flag.isnull()].index+1)

preds_df.index.name='shot_id'

preds_df.to_csv('submission.csv')