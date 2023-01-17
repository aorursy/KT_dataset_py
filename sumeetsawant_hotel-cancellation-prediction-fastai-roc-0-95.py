import pandas as pd 

import numpy as np 



import matplotlib.pyplot as plt 

import seaborn as sns 

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly.figure_factory import create_table

import plotly.express as px



plt.style.use('fivethirtyeight')



import warnings 

warnings.filterwarnings('ignore')

df=pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
df.head()
df.isnull().sum()
df.shape
df.dtypes
df['required_car_parking_spaces'].unique()
df['adr'].unique()
df['reserved_room_type'].value_counts()
df['assigned_room_type'].value_counts()
df['country'].unique()
df['booking_changes'].unique()
df['distribution_channel'].unique()
df['is_repeated_guest'].value_counts()
df['deposit_type'].value_counts()
df['adults'].unique()
df['children'].unique()
df['babies'].unique()
df['previous_cancellations'].unique()
df['meal'].unique()
df['total_of_special_requests'].value_counts()
df['customer_type'].value_counts()
color=sns.color_palette()[1]

sns.countplot(data=df,x='is_canceled',color=color);

plt.title('Distribution of Cancelations')
color=sns.color_palette()[1]

sns.countplot(data=df,x='hotel',color=color);

plt.title('Hotel Type Count')
df.shape[0]
sns.set_style('whitegrid')

row=df.shape[0]

color=sns.color_palette()[0]

sns.countplot(data=df,x='is_canceled',hue='hotel',color=color);

#df.plot(kind='bar',stacked=True,legend=False)

plt.yticks([10000,20000,30000,40000],[8,17,25,34]);

plt.ylabel('percentage');

plt.title('Cancellation count in Hotel vs Resort');
df['arrival_date_year'].min(),df['arrival_date_year'].max()
fig = px.bar(x=df["arrival_date_year"].value_counts().index, 

             y=(df.groupby('arrival_date_year').aggregate('sum')['is_canceled']/df.groupby('arrival_date_year').aggregate('count')['is_canceled'])*100,

             

             )



fig.update_layout(

    title={

        'text': "Percent Cancellation vs Year",

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    xaxis_title="Year",

    yaxis_title="Percent Cancellation",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="RebeccaPurple"

    ))



fig.update_xaxes(tickvals=[2015,2016,2017])

fig.show()
timelapse=df.groupby(['arrival_date_year','arrival_date_month','country']).aggregate('sum')['is_canceled'].to_frame().reset_index()

timelapse['month-year']=timelapse.arrival_date_month.astype(str)+' '+timelapse.arrival_date_year.astype(str)



px.choropleth(timelapse, locations="country", color='is_canceled', hover_name="country", animation_frame='month-year',

              color_continuous_scale=px.colors.sequential.Plasma, projection="natural earth")
#df_month=df.groupby('arrival_date_month').aggregate('sum').reset_index()



# Got the below values from the above groupby dataframe 



month = ['January', 'February', 'March', 'April', 'May', 'June', 'July',

         'August', 'September', 'October', 'November', 'December']



value=[1807,2696,3149,4524,4677,4535,4742,5239,4116,4246,2122,2371]



fig=px.line(x=month,y=value)

fig.show()
df_percent=df.groupby('arrival_date_month').aggregate('mean').reset_index()



#Values taken from above dataframe 



month = ['January', 'February', 'March', 'April', 'May', 'June', 'July',

         'August', 'September', 'October', 'November', 'December']



value=[30.4,33.4,32.15,40.79,39.66,41.45,37.45,37.75,39.17,38.04,31.23,34.97]



fig=px.line(x=month,y=value)

fig.show()
deposit=df.groupby('deposit_type').aggregate('sum')['is_canceled'].to_frame().reset_index()



fig=(px.pie(deposit, values=deposit['is_canceled'], names='deposit_type',color='deposit_type', color_discrete_map=

                                {'Non Refund':'gold',

                                 'No Deposit':'darkorange',

                                 'Refundable':'lightgreen'

                                 }

                            ))

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
deposit=df.groupby('deposit_type').mean()['is_canceled'].to_frame().reset_index()





fig=(px.pie(deposit, values=deposit['is_canceled'], names='deposit_type',color='deposit_type', color_discrete_map=

                                {'Non Refund':'gold',

                                 'No Deposit':'darkorange',

                                 'Refundable':'lightgreen'

                                 }

                                

                                   ))

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()

plt.figure(figsize=(20,20));

sns.heatmap(df.corr(),annot=True,linewidths=0.1,cbar=False);

#Lets compute a new column in the above dataset to compute number of days stay booked in the hotel 



df['stays_total_duration']=df['stays_in_week_nights']+df['stays_in_weekend_nights']
plt.figure(figsize=(15,8))



plt.subplot(1,2,1)

sns.boxplot(x=df['babies'],y=df['stays_total_duration'],hue=df['is_canceled']);





plt.subplot(1,2,2)

sns.boxplot(x=df['children'],y=df['stays_total_duration'],hue=df['is_canceled']);
plt.figure(figsize=(10,10)),

sns.boxplot(x=df['adults'],y=df['stays_total_duration'],hue=df['is_canceled']);
plt.figure(figsize=(15,8))



plt.subplot(1,2,1)

sns.boxplot(x=df['customer_type'],y=df['stays_total_duration'],hue=df['is_canceled']);



plt.subplot(1,2,2)

sns.countplot(x=df['customer_type'],hue=df['is_canceled']);





from fastai.tabular import * 
df.shape
# Lets train the model for 2015 and 2016 and validate it for 2017 



df_train=df[df['arrival_date_year']!=2017]

df_test=df[df['arrival_date_year']==2017]
# Classify the variables into response , catergorical and numerical 



response='is_canceled'



catergorical=['hotel','meal','country','market_segment','customer_type','deposit_type','assigned_room_type','reserved_room_type','is_repeated_guest'\

              ,'distribution_channel' ,'arrival_date_year','arrival_date_month','arrival_date_week_number','arrival_date_day_of_month']



numerical=['adults','children','babies','stays_total_duration','stays_in_weekend_nights','stays_in_week_nights','required_car_parking_spaces'\

          ,'total_of_special_requests','booking_changes','previous_bookings_not_canceled','previous_cancellations','days_in_waiting_list']



procs = [FillMissing, Categorify, Normalize]



path='/kaggle/working'
data = (TabularList.from_df(df_train, path=path, cat_names=catergorical, cont_names=numerical, procs=procs)

                           .split_by_rand_pct(0.15)

                           .label_from_df(cols=response)

                           .add_test(df_test)

                           .databunch())
data.show_batch(rows=10)
learn = tabular_learner(data, layers=[200,100], metrics=[AUROC()])
learn.fit(8, 1e-2)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, 1e-3, wd=0.2)
learn.fit_one_cycle(5, 1e-4)
learn.fit_one_cycle(5, 1e-5, wd=0.2)
learn.save('Hotel_Cancellation')