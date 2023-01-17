import numpy as np

import pandas as pd

from scipy.stats import zscore

from fbprophet import Prophet

print('Required libraries have been imported')
path ='../input/'

dfs = {

    'air_visit_data': pd.read_csv(path+'air_visit_data.csv'),

    'air_store_info': pd.read_csv(path+'air_store_info.csv'),

    'sample_submission': pd.read_csv(path+'sample_submission.csv'),

    'date_info': pd.read_csv(path+'date_info.csv')

}

print('files read:{}'.format(list(dfs.keys())))

for key, name in dfs.items(): locals()[key] = name

print('Data captured')
outliers = (air_visit_data.groupby( ['air_store_id'])['visitors'].transform(zscore) > 3)

air_visit_data[outliers]=np.nan

air_visit_data.dropna(inplace=True)

print(str(outliers.sum())+' outliers have been removed')
#Fill in Nans where possible with average in cluster on that day adjusted by the size of the particular restaurants 

def fill_nans_in_cluster(genre_name,area_name):

    #get list of the same type of restaurants in the neighborhood

    neighbors_bool = air_store_info.apply(lambda x:(x.air_genre_name==genre_name and x.air_area_name==area_name), axis=1)

    neighbors_ids=pd.DataFrame((air_store_info[neighbors_bool]))

    neighbors_restaurants= air_visit_data.merge(neighbors_ids,on='air_store_id',how='inner')[['air_store_id','visit_date','visitors']]

 

    #pivot neighbors_restaurants to easy fill in possible missing dates.

    neighbors_restaurants=neighbors_restaurants.pivot_table(index='visit_date',columns='air_store_id', values='visitors',aggfunc=sum)

    

    #Fill in missing dates(if any) with Nans

    idx = pd.date_range('2016-01-01', '2017-04-22')

    neighbors_restaurants.index = pd.DatetimeIndex(neighbors_restaurants.index)

    neighbors_restaurants = neighbors_restaurants.reindex(idx, fill_value=np.nan)



    # Get visitors rate, normalized to the avarage number of visitors per day 

    neighbors_restaurants_average= neighbors_restaurants.mean(axis=0).tolist()

    normalized_neighbors_restaurants = neighbors_restaurants.div(neighbors_restaurants_average,axis=1)



    # Fill in Nans with avarge number of visiotrs in nighbour restaurants 

    #axis argument to fillna is Not Implemented, so have to use transpond

    normalized_neighbors_restaurants_with_filled_nans=normalized_neighbors_restaurants.T.fillna(normalized_neighbors_restaurants.mean(axis=1))

    

    #replace normalized values with real vistors by multipliyng back on average per restaurant

    neighbors_restaurants_with_filled_nans = normalized_neighbors_restaurants_with_filled_nans.mul(neighbors_restaurants_average,axis=0).reset_index()



    #return visit data in the original format 

    df_columns = neighbors_restaurants_with_filled_nans.columns[1:]

    return  pd.melt(neighbors_restaurants_with_filled_nans,id_vars=['air_store_id'], value_vars=df_columns)
clusters_names= air_store_info.apply(lambda x:(x.air_genre_name + '_' + x.air_area_name), axis=1).unique().tolist()

full_data = pd.DataFrame(columns=air_visit_data.columns)



for cluster in clusters_names:

    cluster_data = fill_nans_in_cluster (cluster.split('_')[0],cluster.split('_')[1])

    cluster_data.rename(columns={'variable':'visit_date','value':'visitors'},inplace=True )

    full_data=full_data.append(cluster_data,ignore_index=True)

print('Missing data filling complete')
target_restaurants= pd.DataFrame({'air_store_id':sample_submission['id'].str[:-len('_2017-04-23')].unique()})

full_data=full_data.merge(target_restaurants,on='air_store_id',how='inner')
visit_data = full_data.merge(air_store_info[['air_store_id','air_genre_name','air_area_name']],

                             left_on='air_store_id',right_on='air_store_id',how='left')

date_info['calendar_date']= pd.to_datetime(date_info['calendar_date'])



visit_data = visit_data.merge(date_info, left_on='visit_date', right_on='calendar_date', how='left')

visit_data.drop('calendar_date', axis=1, inplace=True)



print('Data ready for analysis')
#Make a copy of visit daеa so we could use it later 

simple_visit_data = visit_data.copy()



#If there is a holyday mark day of the weel as a Holiday

simple_visit_data.loc[simple_visit_data['holiday_flg']==1,'day_of_week'] = 'Holiday'



# Calculate average number of the visitors per day of the week. Holiday is treated as day of the week

visitors_per_day_of_the_week = simple_visit_data.groupby(['air_store_id', 'day_of_week']).mean().reset_index()

visitors_per_day_of_the_week.drop('holiday_flg', axis=1, inplace=True)
#Make a copy of visit daеa so we could use it later 

simple_submission = sample_submission.copy()



#extraxt required restaurant ids and required dates

simple_submission['air_store_id'] = simple_submission['id'].str[:-len('_2017-04-23')] 

simple_submission['calendar_date'] = simple_submission['id'].str[-len('2017-04-23'):] 

simple_submission.drop(['visitors','id'], axis=1, inplace=True)

simple_submission['calendar_date']= pd.to_datetime(simple_submission['calendar_date'])



# Using visitors_per_day_of_the_week fill in required position in the submission file

simple_submission = simple_submission.merge(date_info, on='calendar_date', how='left')

simple_submission.loc[simple_submission['holiday_flg']==1,'day_of_week'] = 'Holiday'

simple_submission = simple_submission.merge(visitors_per_day_of_the_week, on=['air_store_id', 'day_of_week'], how='left')



print('simple submission file is ready')

simple_submission['id']= simple_submission.apply(lambda row: str(row.air_store_id)+'_' + str(row.calendar_date)[:len('2017-04-23')], axis=1)

simple_submission[['id', 'visitors']].to_csv('simple_submission.csv', index=None)

print("Submission for simple method is done")
#Holidays are presented in the format required by Prophet 

new_year_day = pd.DataFrame({

  'holiday': 'new_year_day',

  'ds': pd.to_datetime(['2016-01-01', '2017-01-01']),

  'lower_window': -2, #how many days before holiday are significant 

  'upper_window': 1,  #how many days after holiday are significant 

})

bank_holiday = pd.DataFrame({

  'holiday': 'bank_holiday',

  'ds': pd.to_datetime(['2016-01-02','2016-01-03', '2016-12-31', '2017-01-02','2017-01-03']),

  'lower_window': 0,

  'upper_window': 0,

})

coming_of_age_day = pd.DataFrame({

  'holiday': 'coming_of_age_day',

  'ds': pd.to_datetime(['2016-01-09','2017-01-11']),

  'lower_window': 0,

  'upper_window': 0,

})

national_foundation_day = pd.DataFrame({

  'holiday': 'national_foundation_day',

  'ds': pd.to_datetime(['2016-02-11','2017-02-11']),

  'lower_window': 0,

  'upper_window': 0,

})

valentines_day = pd.DataFrame({

  'holiday': 'valentines_day',

  'ds': pd.to_datetime(['2016-02-14','2017-02-14']),

  'lower_window':-1,

  'upper_window': 1,

})

dolls_girls_festival = pd.DataFrame({

  'holiday': 'dolls_girls_festival',

  'ds': pd.to_datetime(['2016-03-03','2017-03-03']),

  'lower_window':-1,

  'upper_window': 1,

})

equinox = pd.DataFrame({

  'holiday': 'equinox',

  'ds': pd.to_datetime(['2016-03-20','2016-03-21','2016-09-22','2016-06-20','2017-03-20']),

  'lower_window': 0,

  'upper_window': 0,

})

golden_week = pd.DataFrame({

  'holiday': 'golden_week',

  'ds': pd.to_datetime(['2016-04-29','2016-05-03','2016-05-04','2016-05-05','2017-04-29','2017-05-03','2017-05-04','2017-05-05']),

  'lower_window':-2,

  'upper_window': 1,

})

star_festival = pd.DataFrame({

  'holiday': 'star_festival',

  'ds': pd.to_datetime(['2016-07-07']),

  'lower_window': 0,

  'upper_window': 0,

})

sea_day = pd.DataFrame({

  'holiday': 'sea_day',

  'ds': pd.to_datetime(['2016-07-18']),

  'lower_window': 0,

  'upper_window': 0,

})

mountain_day = pd.DataFrame({

  'holiday': 'mountain_day',

  'ds': pd.to_datetime(['2016-08-11']),

  'lower_window': 0,

  'upper_window': 0,

})

respect_for_the_aged = pd.DataFrame({

  'holiday': 'respect_for_the_aged',

  'ds': pd.to_datetime(['2016-09-19']),

  'lower_window': 0,

  'upper_window': 0,

})

sports_day = pd.DataFrame({

  'holiday': 'sports_day',

  'ds': pd.to_datetime(['2016-10-10']),

  'lower_window': 0,

  'upper_window': 0,

})

culture_day = pd.DataFrame({

  'holiday': 'culture_day',

  'ds': pd.to_datetime(['2016-11-03']),

  'lower_window': 0,

  'upper_window': 0,

})

day_7_5_3 = pd.DataFrame({

  'holiday': 'day_7_5_3',

  'ds': pd.to_datetime(['2016-11-15']),

  'lower_window': 0,

  'upper_window': 0,

})

labor_thanksgiving_day = pd.DataFrame({

  'holiday': 'labor_thanksgiving_day',

  'ds': pd.to_datetime(['2016-11-23']),

  'lower_window':-1,

  'upper_window': 1,

})

christmas = pd.DataFrame({

  'holiday': 'christmas',

  'ds': pd.to_datetime(['2016-12-21','2016-12-23','2016-12-25']),

  'lower_window':-2,

  'upper_window': 1,

})



holidays = pd.concat((new_year_day, bank_holiday,coming_of_age_day,national_foundation_day,\

                      valentines_day,dolls_girls_festival,equinox,golden_week,\

                     star_festival,sea_day,mountain_day,respect_for_the_aged,\

                     sports_day,culture_day,day_7_5_3,labor_thanksgiving_day,\

                     christmas))

# get restaraunts with missed data on '2017-04-22'

missings= visit_data['visitors'].isnull() & (visit_data['visit_date'] == '2017-04-22')



#get average number of visitors on saturdays for these restaurants

visitors_per_day_of_the_week = visit_data.groupby(['air_store_id', 'day_of_week']).mean().reset_index()

visitors_on_Saturdays = visitors_per_day_of_the_week[visitors_per_day_of_the_week['day_of_week']== 'Saturday']



#apply to visit data

visit_data.loc[missings, 'visitors'] = visit_data[missings].merge(

    visitors_on_Saturdays[['air_store_id', 'visitors']], on='air_store_id', how='left')['visitors_y'].values

print('Data ready for advanced method')

def prophet_prediction (air_id):

    #get restaurant data

    restaurant_visit_data = visit_data[visit_data['air_store_id'] == air_id]

    

    #fill it into Prophet model and fit the model

    df=pd.DataFrame()

    df['ds']=restaurant_visit_data['visit_date']

    df['y'] = np.log(restaurant_visit_data['visitors'])

    model = Prophet(changepoint_prior_scale=0.5, yearly_seasonality=False)

    model.fit(df)

    

    #run prediction for the next 39 days

    future_data = model.make_future_dataframe(periods=39)

    forecast_data = model.predict(future_data)

    return forecast_data.iloc[-39:,:][['ds', 'yhat']]

#prepare submission file

submission= pd.DataFrame(columns=('id','visitors'))

for row in target_restaurants['air_store_id']:

    submission_to_append= pd.DataFrame(columns=('id','visitors'))

    temp_submission=prophet_prediction(row)

    submission_to_append['id']= temp_submission['ds'].map(lambda x: str(row)+'_'+str(x)[:len('2017-04-23')])

    submission_to_append['visitors']= np.exp(temp_submission['yhat'])

    submission = submission.append(submission_to_append,ignore_index=True)

prophet_submission= pd.DataFrame(submission).reset_index()

submission[['id','visitors']].to_csv('prophet_submission.csv', index=None)

print("Submission for advanced method is done")