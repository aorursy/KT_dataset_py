# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('white')

plt.rcParams["patch.force_edgecolor"] = True

from plotly.offline import iplot, init_notebook_mode

import cufflinks as cf

import plotly.graph_objs as go

# import chart_studio.plotly as py



init_notebook_mode(connected=True)

cf.go_offline(connected=True)



# Set global theme

cf.set_config_file(world_readable=True, theme='pearl')

df_original = pd.read_csv("/kaggle/input/hourly-weather-surface-brazil-southeast-region/sudeste.csv", )

sample_df = df_original.sample(300000,random_state=101)


print(sample_df.iloc[:,:10].head(1))

print("=="*20)

print(sample_df.iloc[:,10:20].head(1))

print("=="*20)



print(sample_df.iloc[:,20:].head(1))
sample_df.drop(['wsnm','wsid','inme'],inplace=True, axis=1)
# Padding of columns

# i.e  1->01, 2-> 02, so on.

sample_df.mo = sample_df.mo.astype(str).str.zfill(2)

sample_df.da = sample_df.da.astype(str).str.zfill(2)
sample_df['Date'] =  sample_df[['yr','mo','da']].astype(str).agg("-".join,axis=1)
sample_df['Date'] = pd.to_datetime(sample_df['Date'], format='%Y-%m-%d')
sample_df.drop(['mdct','yr','da','hr','mo','date'], axis= 1, inplace=True)
sample_df.set_index("Date", inplace=True,drop=True)
null_cols = sample_df.columns[sample_df.isnull().sum()>0]

not_null_cols = sample_df.columns[sample_df.notnull().sum()==300000]

null_df= sample_df[null_cols]

not_null_df= sample_df[not_null_cols]

def customized_heatmap(df):

    corr_df = df.corr()

#     print(corr_df)

    missing_df =corr_df.iloc[1:,:-1].copy()  

#     print(missing_df)



    

    # Get only half portion of corr_df to avoid df, so create mask    

    mask = np.triu(np.ones_like(missing_df), k=1)

    

     

    # plot a heatmap of the values

    plt.figure(figsize=(20,14))

    ax = sns.heatmap(missing_df, vmin=-1, vmax=1, cbar=False,

                     cmap='coolwarm', mask=mask, annot=True)

    

    # format the text in the plot to make it easier to read

    for text in ax.texts:

        t = float(text.get_text())

        if -0.25 < t < 0.25:

            text.set_text('')

        else:

            text.set_text(round(t, 2))

        text.set_fontsize('x-large')

    plt.xticks( size='x-large')

    plt.yticks(rotation=0, size='x-large')

#     plt.savefig("Heatmap DF")

    plt.show()

    
customized_heatmap(null_df)
null_counts = null_df.isnull().sum()

null_counts_pct = (null_counts / sample_df.shape[0])*100



null_pct_df = pd.DataFrame({'null_counts': null_counts, 'null_pct': null_counts_pct})



print(null_pct_df.T.astype(int))
#Plot by sorting the values by gust

sorted_df_by_temp = sample_df[['gust','wdsp',"temp"]].sort_values(['temp'] )

sorted_df_by_gust = sample_df[['gust','wdsp',"temp"]].sort_values(['gust'] )
plt.figure(figsize=(10,5))

sns.heatmap(sorted_df_by_gust.isnull(),cmap='coolwarm', cbar=False, yticklabels=False);

plt.figure(figsize=(10,5))

sns.heatmap(sorted_df_by_temp.isnull(),cmap='coolwarm', cbar=False, yticklabels=False);
mask_gust_wdsp = sample_df['gust'].notnull() & sample_df['wdsp'].isnull()
# Create interval from mask

sample_df[mask_gust_wdsp].gust.value_counts(bins=3)
# Create three mean 

first_range_mean_wdsp_by_gust= sample_df[sample_df['gust']<7.5]['wdsp'].mean()

second_range_mean_wdsp_by_gust=sample_df[((sample_df['gust']>=7.5) &(sample_df['gust']<15.0))]['wdsp'].mean()

third_range_mean_wdsp_by_gust=sample_df[sample_df['gust']>=15.0]['wdsp'].mean()
#math is imported to check nan, np.nan wont work

def fill_wdsp_by_gust(col): 

    

    

#Initialize relevant cols

    gust = col[0]

    wdsp = col[1]

    

    # If the value is nan

    #Assign by ranges declared above

    import math

    if (math.isnan(wdsp)):

        # Make sure gust in not nan

        if math.isnan(gust):

            pass

        elif (gust<7.5):

            return first_range_mean_wdsp_by_gust

        elif (gust>=7.5 ) and (gust<15.0):

            return second_range_mean_wdsp_by_gust

        elif (gust>=15.0):

            return third_range_mean_wdsp_by_gust

          #if not nan return as it is

    else:

        return wdsp

    
sample_df['wdsp'] = sample_df[['gust','wdsp']].apply(fill_wdsp_by_gust,axis=1)

sample_df.dropna(subset=['tmax','temp','tmin'],inplace=True)
print(sample_df['temp'].value_counts(bins=5).sort_index())

# this funciton will take two cols, temp and column to fill values in this order

# the second argument is the context

# From correlation table it seems temperature has high correlation with many columns so, context will clarify which value to fill 

#by column name

#math is imported to check nan, np.nan wont work



# Create five conditions

cond_1 = sample_df[sample_df['temp']<5.96]

cond_2 = sample_df[((sample_df['temp']>=5.96) & (sample_df['temp']<15.32))]

cond_3= sample_df[((sample_df['temp']>=15.32) & (sample_df['temp']<24.68))  ]

cond_4 = sample_df[((sample_df['temp']>=24.68) & (sample_df['temp']<34.04))  ]

cond_5 = sample_df[sample_df['temp']>=34.04]





# Create five ranges of mean according to above interval for windspeed



first_range_mean_wdsp_by_temp=cond_1['wdsp'].mean()

second_range_mean_wdsp_by_temp= cond_2['wdsp'].mean()

third_range_mean_wdsp_by_temp= cond_3['wdsp'].mean()

fourth_range_mean_wdsp_by_temp= cond_4['wdsp'].mean()

fifth_range_mean_wdsp_by_temp= cond_5['wdsp'].mean()





# Create five ranges of mean according to above interval for gust



first_range_mean_gust_by_temp= cond_1['gust'].mean()

second_range_mean_gust_by_temp=  cond_2['gust'].mean()

third_range_mean_gust_by_temp= cond_3['gust'].mean()

fourth_range_mean_gust_by_temp=  cond_4['gust'].mean()

fifth_range_mean_gust_by_temp=  cond_5['gust'].mean()





# Create five ranges of mean according to above interval for solar radiation



first_range_mean_radiation_by_temp= cond_1['gbrd'].mean()

second_range_mean_radiation_by_temp=cond_2['gbrd'].mean()

third_range_mean_radiation_by_temp= cond_3['gbrd'].mean()

fourth_range_mean_radiation_by_temp=  cond_4['gbrd'].mean()

fifth_range_mean_radiation_by_temp=  cond_5['gbrd'].mean()







def fill_missing_by_temp(col, context):

    import math

    #Initialize relevant cols

    temp = col[0]

    col_1_val = col[1]

    

    # Divide the task by context

    #Either for windspeed or for gust

    

    if context == "wdsp":

      

        # If the value is nan

        #Assign by ranges declared above

        if math.isnan(col_1_val):

            if(temp<5.96):

                return first_range_mean_wdsp_by_temp

            elif(temp>=5.96) and (temp<15.32):

                return second_range_mean_wdsp_by_temp

            elif(temp>=15.32) and (temp<24.68):

                return third_range_mean_wdsp_by_temp

            elif(temp>=24.68) and (temp<34.04):

                return fourth_range_mean_wdsp_by_temp

            elif(temp>=34.04):

                return fifth_range_mean_wdsp_by_temp

            #if not nan return as it is

        else:

            return col_1_val

        

    elif context=="gbrd":

         # If the value is nan

        #Assign by ranges declared above

        if math.isnan(col_1_val):

            if(temp<5.96):

                return first_range_mean_radiation_by_temp

            elif(temp>=5.96) and (temp<15.32):

                return second_range_mean_radiation_by_temp

            elif(temp>=15.32) and (temp<24.68):

                return third_range_mean_radiation_by_temp

            elif(temp>=24.68) and (temp<34.04):

                return fourth_range_mean_radiation_by_temp

            elif(temp>=34.04):

                return fifth_range_mean_radiation_by_temp

            #if not nan return as it is

        else:

            return col_1_val

        

    else:

         # If the value is nan

        #Assign by ranges declared above

        if math.isnan(col_1_val):

            if(temp<5.96):

                return first_range_mean_gust_by_temp

            elif(temp>=5.96) and (temp<15.32):

                return second_range_mean_gust_by_temp

            elif(temp>=15.32) and (temp<24.68):

                return third_range_mean_gust_by_temp

            elif(temp>=24.68) and (temp<34.04):

                return fourth_range_mean_gust_by_temp

            elif(temp>=34.04):

                return fifth_range_mean_gust_by_temp

            #if not nan return as it is

        else:

            return col_1_val

    

        

    

    
sample_df['wdsp'] = sample_df[['temp','wdsp']].apply(fill_missing_by_temp,context ="wdsp",axis=1)

sample_df['gust'] = sample_df[['temp','gust']].apply(fill_missing_by_temp,context= "gust", axis=1)



#Plot by sorting the values by temp

gbrd_df_by_temp = sample_df[['temp',"gbrd" ]].sort_values(['temp'] )

plt.figure(figsize=(10,5))

sns.heatmap(gbrd_df_by_temp[['temp',"gbrd"]].isnull(),cmap='coolwarm', cbar=False, yticklabels=False);

sample_df['gbrd'] = sample_df[['temp','gbrd']].apply(fill_missing_by_temp,context= "gbrd", axis=1)

# First drop hmin and hmax na values



sample_df.dropna(subset=['hmax','hmin'],inplace=True)
print(null_pct_df.T.astype(int))
# Check out intervals



sample_df.hmdy.value_counts(bins=3).sort_index()
# Create three mean 

first_range_mean_dwep_by_humidity= sample_df[sample_df['hmdy']<33.333]['dewp'].mean()

second_range_mean_dwep_by_humidity=sample_df[((sample_df['hmdy']>=33.333) &(sample_df['hmdy']<66.667))]['dewp'].mean()

third_range_mean_dwep_by_humidity=sample_df[sample_df['hmdy']>= 66.667 ]['dewp'].mean()
#math is imported to check nan, np.nan wont work

def fill_dewp_by_humidity(col): 

    

    

#Initialize relevant cols

    hmdy = col[0]

    dewp = col[1]

    

    # If the value is nan

    #Assign by ranges declared above

    import math

    if math.isnan(dewp):

        if (hmdy<33.333):

            return first_range_mean_dwep_by_humidity

        elif (hmdy>=33.333 ) and (hmdy<66.667):

            return second_range_mean_dwep_by_humidity

        elif (hmdy>=66.667):

            return third_range_mean_dwep_by_humidity

          #if not nan return as it is

    else:

        return dewp

    
sample_df['dewp'] = sample_df[["hmdy", "dewp"]].apply(fill_dewp_by_humidity, axis=1)

sample_df['dmin'] = sample_df[["hmdy", "dmin"]].apply(fill_dewp_by_humidity, axis=1)

sample_df['dmax'] = sample_df[["hmdy", "dmax"]].apply(fill_dewp_by_humidity, axis=1)
# New df without prcp

df = sample_df[['elvt', 'lat', 'lon','city', 'prov', 'stp', 'smax', 'smin', 'gbrd', 'temp',

       'dewp', 'tmax', 'dmax', 'tmin', 'dmin', 'hmdy', 'hmax', 'hmin', 'wdsp',

       'wdct', 'gust']].copy()



from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()



#Make a copy from the original df first



normed_df = df.copy()

# EXtract cols with non-string values

float_cols = df.columns[df.dtypes =="float64" ].tolist()

# Normed df 

normed_df[float_cols] =scaler.fit_transform(normed_df[float_cols])
df['prov'].value_counts(normalize=True).plot.pie(figsize=(8,10),autopct = '%.1f%%',

                                                 labels=['Minas Gerais','São Paulo','Rio de Janeiro','Espírito Santo'])

plt.xlabel("")

plt.ylabel("")

plt.title('Weather Data by Province');

# plt.savefig("Provinces Proportions")



gr_by_city = normed_df[['prov','city', 'stp', 'smax', 'smin', 'gbrd', 'temp',

                  'dewp', 'tmax', 'dmax', 'tmin', 'dmin', 'hmdy', 

                  'hmax', 'hmin', 'wdsp','wdct', 'gust']].groupby('city').mean()



gr_by_prov = normed_df[['prov','city', 'stp', 'smax', 'smin', 'gbrd', 'temp',

                             'dewp', 'tmax', 'dmax', 'tmin', 'dmin','hmdy', 

                        'hmax', 'hmin', 'wdsp','wdct', 'gust']].groupby(['prov']).mean()
gr_by_city.columns
gr_by_yr_prov = normed_df.groupby([normed_df.index.year,'prov']).mean().unstack(level=0)
layout = dict(xaxis_title="City",

              yaxis_title="Frequency Normalized",

              title="Avg. Temp by City")

gr_by_city[('temp')].iplot(kind="bar", layout=layout);


# Change name for better labels in legend

# 'stp', 'smax', 'smin', 'gbrd', 'temp', 'dewp', 'tmax', 'dmax', 'tmin',

#        'dmin', 'hmdy', 'hmax', 'hmin', 'wdsp', 'wdct', 'gust'

cols_to_plot =['stp','gbrd', 'temp', 'dewp', 'hmdy', 'wdsp', 'gust']

cols_changed_name ={"Air Pressure","Solar Radiation", "Temperature","Dew Point","Humidity", "Windspeed","Gust"}





temp_df = gr_by_prov[cols_to_plot]



temp_df.columns = cols_changed_name

layout = dict(xaxis = dict( tickvals =["ES","MG","RJ","SP"],

                           ticktext=['Espírito Santo','Minas Gerais','Rio de Janeiro','São Paulo']),

             legend_title_text='Weather Factors',

              xaxis_title="Provinces",

              yaxis_title="Frequency Normalized",

              title = "Avg Weather Factors by Provinces",)

temp_df.iplot(kind="bar",  layout=layout);

#names = ["Air pressure","Solar Radiation", "Temperature", "Dewpoint", "Humidity","Windspeed","Gust"])



          
# sample_df[['prcp',"city"]].groupby(['city', df.index.year]).mean().unstack(level=1)

#

temp_yearly=pd.pivot_table(data= sample_df,index="city", columns=sample_df.index.year,values="temp")

# temp_yearly_normed=pd.pivot_table(data= normed_df,index="city", columns=sample_df.index.year,values="temp")



l= dict(showlegend=False, title="Yearly Average Temperature", xaxis_title="Year", yaxis_title="Frequecny")

temp_yearly.iplot(kind="box",layout=l)


# temp_yearly.iplot(secondary_y = min_values)

layout= dict(yaxis_title="Frequency",xaxis_title="City", title="Average Yearly Temperature by Cities",)

temp_yearly.iplot(kind="scatter", mode="lines+markers", layout=layout)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()



# Take a copy of sample_df

sample_df_norm = sample_df.copy()



#Columns to normalize

cols_to_norm = ['elvt', 'lat', 'lon', 'prcp', 'stp', 'smax', 'smin','gbrd', 'temp', 'dewp', 'tmax', 'dmax', 'tmin', 'dmin',

                'hmdy', 'hmax','hmin', 'wdsp', 'wdct', 'gust']



#Normalize numeric scales

sample_df_norm.loc[:,cols_to_norm] = scaler.fit_transform(sample_df.loc[:,cols_to_norm])
### Seasonal Fluctuations- Tempearature



# Assuming 4 mnths seasons

# Lets obsorve seasoal fluctuation

layout= dict(yaxis_title="Frequency",xaxis_title="Years", title="Average Seasonal Temperature Distribution",)

sample_df_norm[['temp','tmin',"tmax"]].resample('M').mean().rolling(4).mean().iplot(kind="bar",layout=layout )

temperature_df=sample_df[['city','temp','tmax','tmin']]
def get_temp_by_choice(yr,mo,c):

    # Get temperature record by given month and year

    temp_m_y_c=temperature_df[(temperature_df.index.year==yr) & (temperature_df.index.month==mo) & (temperature_df.city == c)]

    

    

    # Extract given city from  above record

    temp_max = temp_m_y_c.max()

    temp_min = temp_m_y_c.min()

    print(temp_max)

    print(temp_min)

    print("=="*20)

#     return temp_max, temp_min
get_temp_by_choice(2008,10,"Viçosa")

get_temp_by_choice(2008,10,"Valença")

get_temp_by_choice(2008,10,"Montalvânia")

get_temp_by_choice(2008,10,"Itaobim")
timeline={"Weekly":"W","Monthly":"M","Yearly":"Y"}





def temp_analysis(cities, time_line="Yearly"):

    """provide timeline among weekly, monthly or yearly. The default value is Yearly.

     Function takes  two  arguments 

     1. cities, which is list of cites

     2. time_line, which has three values, weekly, yearly, or monthly

    for eg: temp_analysis(['Viçosa','Mantena','Formiga',"São João del Rei","Juiz de Fora" ], "Yearly")



"""

    print(temp_analysis.__doc__)

    

    c =pd.DataFrame()

    # Extract given city from  above record

    for city in cities:

        c= c.append(temperature_df[temperature_df['city'] == city])

    

    # Resample it on average

    #title() takes care of uppercase and lowercase confusion

    d=c.resample(timeline[time_line.title()]).mean()

    print("Temperature of cities {}".format(cities))



   

    # Plot the data

    layout= dict( title='Avg Temperatures {}'.format(time_line.upper()), xaxis_title="Years", yaxis_title="Temperatures")

    d[['tmax','tmin','temp']].iplot(kind="bar", layout=layout)

temp_analysis(['Viçosa','Mantena','Formiga',"São João del Rei","Juiz de Fora" ], "Weekly")
# Assuming 4 mnths seasons

# Lets obsorve seasoal fluctuation

layout= dict(yaxis_title="Frequency",xaxis_title="Years", title="Average Seasonal Pressure Distribution",)

sample_df_norm[['stp','smin',"smax"]].resample('M').mean().rolling(4).mean().iplot(kind="bar",layout=layout )

# df for airpressure only

air_df=sample_df[['city','prov','stp','smax','smin']]
def monthly_plot(m,y,cities):

    '''  Function takes  three arguments month in number, year in int and list of cities to compare

    for eg: monthly_plot(10,2008,['Viçosa','Mantena','Formiga',"São João del Rei","Juiz de Fora" ])





    '''

    print(monthly_plot.__doc__)

  

    # Get pressure record by given month and year

    air_m_y=air_df[(air_df.index.month == m) & (air_df.index.year == y)]

    air_m_y_c =pd.DataFrame()

    # Extract given city from  above record

    for c in cities:

        air_m_y_c = air_m_y_c.append(air_m_y[air_m_y['city'] == c])

    

    grp_by_c = air_m_y_c.groupby('city').mean()

    layout= dict( title='Avg Air pressure of city {}-{}'.format(m,y), xaxis_title="Cities", yaxis_title="Air pressure")

    grp_by_c.iplot(kind="bar",layout=layout)
# For prov "MG"

minas_cities = air_df[air_df.prov=="MG"].city

# lets just use 5 cities at random

minas_cities[:5]
monthly_plot(10,2008,['Viçosa','Mantena','Formiga',"São João del Rei","Juiz de Fora" ])
gust_yr_city = sample_df_norm.pivot_table(columns=sample_df_norm.index.year,index='city',values=['gust'],aggfunc='mean').stack()

gusty_city= gust_yr_city.loc[gust_yr_city['gust']==gust_yr_city.gust.max()].index[0]

gusty_city
customized_heatmap(normed_df)
# fig = plt.figure(figsize=(10,10))

sns.jointplot(normed_df['hmdy'],normed_df['dewp']);

# plt.savefig("Humidity vs Dewpoint")
fig = plt.figure(figsize=(10,10))

sns.jointplot(normed_df['wdsp'],normed_df['hmdy'], kind='kde');

plt.savefig("Windspped vs Humidty Density Plot")