import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('dark_background')
df = pd.read_csv('../input/homicide-reports/database.csv')
df.info()
df.columns
print(df.describe())

df.describe().plot()
import seaborn as sns
sns.pairplot(df)

plt.title('Crime Happened Between 1980-2014')

plt.show()
crime_describe = df.describe()

type(crime_describe)
crime_describe = crime_describe.drop(columns = ['Record ID'])
crime_describe
df['Incident'].sum()
states = df['State'].unique()
len(states)



#so in all states
cities = df['City'].unique()

len(cities)
df.head()
more_focused = df[['Victim Sex', 'Victim Age', 'Perpetrator Sex','Perpetrator Age', 'Relationship', 

                  'Weapon', 'Victim Count', 'Perpetrator Count', 'Crime Type', 'City', 'State','Year', 

                  'Month', 'Incident']]
more_focused.head()
more_focused['Crime Type'].unique()



# so we have just murdered..
more_focused['Victim Count'].sum()
more_focused['Perpetrator Count'].sum()
sns.scatterplot(more_focused['Victim Count'], more_focused['Perpetrator Count'])

plt.grid()
plt.figure(figsize=(10,8))

sns.barplot(more_focused['Year'],more_focused['Incident'])

plt.xticks(rotation= 'vertical')

plt.title('Yearly Incident(Murdered) In US\n1980-2014')

plt.show()
plt.figure(figsize=(10,8))

sns.barplot(more_focused['State'], more_focused['Incident'])

plt.xticks(rotation= 'vertical')

plt.title('State Wise Incident(Murdered) In US\n1980-2014')

plt.show()
plt.figure(figsize=(10,8))

sns.barplot(more_focused['Month'], more_focused['Incident'])

plt.xticks(rotation= 'vertical')

plt.title('Monthly Incident(Murdered) In US\n1980-2014')

plt.show()
cities =  more_focused.groupby('City')['Incident'].sum()
cities_describe = cities.describe()

cities_describe
below_100 =  cities.loc[cities < 100].values

below_1000 =  cities.loc[cities < 1000].values

below_10000 =  cities.loc[cities < 10000].values

below_100000 =  cities.loc[cities < 100000].values

below_1000000 =  cities.loc[cities < 1000000].values



above_1M_incident_city = cities.loc[cities < 1000000]

above_1lakh_to_1M_incident_city = cities.loc[(cities > 100000) & (cities <1000000)]

above_10th_to_1lakh_incident_city = cities.loc[(cities > 10000) & (cities <100000)]

above_1th_to_10th_incident_city = cities.loc[(cities > 1000) & (cities <10000)]

above_100_to_1th_incident_city = cities.loc[(cities > 100) & (cities <1000)]

above_10_to_100_incident_city = cities.loc[(cities > 10) & (cities <100)]

above_0_to_10_incident_city = cities.loc[(cities > 0) & (cities <10)]
print('Amount Of Cities Having Number Of Incident\n')

print(f'> 1M :              {len(above_1M_incident_city)}')

print(f'1lakh> <1M :        {len(above_1lakh_to_1M_incident_city)}')

print(f'10th> <1lakh :      {len(above_10th_to_1lakh_incident_city)}')

print(f'1th> <10th :        {len(above_1th_to_10th_incident_city)}')

print(f'100> <1th :         {len(above_100_to_1th_incident_city)}')

print(f'10> <100 :          {len(above_10_to_100_incident_city)}')

print(f'0> <10 :            {len(above_0_to_10_incident_city)}')
range_list = [below_100, below_1000, below_10000, below_100000, below_1000000]

j = 0



while j < 5:

    for i in range_list:

        

        plt.figure(figsize=(10,8))

        sns.kdeplot(i, shade = True)

        plt.show()

        

        

        plt.figure(figsize=(10,8))

        sns.distplot(i,bins= 50,rug=True, rug_kws={'color':'purple'}, hist_kws= {'color': 'red'}, kde_kws= {'color': 'blue'})

        plt.show()

        

        j +=1
std = []

mean_cities = cities.mean()

std_cities = cities.std()



for i in cities:

    standardizise =( i - mean_cities )/ std_cities  

    std.append(standardizise)
sns.boxplot(cities_describe, palette='winter')

plt.xlim([0,30000])

plt.title('Cities Incident')

plt.show()
sns.kdeplot(std)
cities_describe = cities.describe()

cities_describe
i = 0

range_j = range(1,18)

while i < 1700:

    for j in range_j:

    

        plt.figure(figsize=(20,8))

        sns.barplot(cities.index[i:j*100], cities[i:j*100])

        plt.xticks(rotation= 'vertical')

        plt.ylim([0,1000])

        plt.title(f'Cities Incident(Murdered) In US\n1980-2014\n\nSummary: \n\n{cities_describe}')

        plt.show()



        i += 100

        j += 1

    

    

plt.figure(figsize=(20,8))

sns.barplot(cities.index[1700:], cities[1700:])

plt.xticks(rotation= 'vertical')

plt.ylim([0,1000])

plt.title(f'Cities Incident(Murdered) In US\n1980-2014\n\nSummary: \n\n{cities_describe}')

plt.show()
data_matrix = df[['Victim Sex', 'Perpetrator Sex','Victim Age','Perpetrator Age', 

                  'Victim Count', 'Perpetrator Count', 'City', 'State','Year', 

                  'Month', 'Incident', 'Relationship', 'Weapon','Crime Type','Crime Solved']]
data_matrix.head()
import seaborn as sns

sns.barplot(data_matrix['Crime Type'], data_matrix['Incident'])
crime_type_gb = data_matrix.groupby('Crime Type')['Incident'].sum()
plt.figure(figsize=(10,8))

sns.barplot(crime_type_gb.index, crime_type_gb, palette= 'spring')

plt.title(f'{crime_type_gb}')

plt.show()
total_incident = crime_type_gb[0] + crime_type_gb[1]

negligence_percentage = (crime_type_gb[0] / total_incident) * 100

not_negligence_percentge = (crime_type_gb[1] / total_incident) *100
plt.figure(figsize=(20,8))

sns.barplot(crime_type_gb.index, crime_type_gb, palette= 'spring')

plt.title(f'{crime_type_gb}\n\nBy Willingly: {not_negligence_percentge}%\nBy Negligence: {negligence_percentage}%')

plt.show()
data_matrix.head()
def perpetartor_count_wise(i, name):

    perpetrator = data_matrix.loc[data_matrix['Perpetrator Count'] == i]

    perpetrator_percentage = (len(perpetrator) / len(data_matrix)) * 100

    perpetrator_percentage

    print(f'{name} Perpetrator: \t{perpetrator_percentage}%')



    

perpetartor_count_wise(0, '0')    

perpetartor_count_wise(1, 'Individual')

perpetartor_count_wise(2, 'Two')

perpetartor_count_wise(3, 'Three')

perpetartor_count_wise(4, 'Four')

perpetartor_count_wise(5, 'Five')

perpetartor_count_wise(6, 'Six')

perpetartor_count_wise(7, 'Seven')

perpetartor_count_wise(8, 'Eight')

perpetartor_count_wise(9, 'Nine')

perpetartor_count_wise(10, 'Ten')
def perpetartor_count_df(i):

    perpetrator = data_matrix.loc[data_matrix['Perpetrator Count'] == i]

    return perpetrator





perpetrator0 = perpetartor_count_df(0)

perpetrator1 = perpetartor_count_df(1)

perpetrator2 = perpetartor_count_df(2)

perpetrator3 = perpetartor_count_df(3)

perpetrator4 = perpetartor_count_df(4)

perpetrator5 = perpetartor_count_df(5)

perpetrator6 = perpetartor_count_df(6)

perpetrator7 = perpetartor_count_df(7)

perpetrator8 = perpetartor_count_df(8)

perpetrator9 = perpetartor_count_df(9)

perpetrator10 = perpetartor_count_df(10)
def perpetartor_gb(df):

    perpetrator_gb = df.groupby('Relationship')['Incident'].sum()

    return perpetrator_gb



perpetrator0_gb = perpetartor_gb(perpetrator0)

perpetrator1_gb = perpetartor_gb(perpetrator1)

perpetrator2_gb = perpetartor_gb(perpetrator2)

perpetrator3_gb = perpetartor_gb(perpetrator3)

perpetrator4_gb = perpetartor_gb(perpetrator4)

perpetrator5_gb = perpetartor_gb(perpetrator5)

perpetrator6_gb = perpetartor_gb(perpetrator6)

perpetrator7_gb = perpetartor_gb(perpetrator7)

perpetrator8_gb = perpetartor_gb(perpetrator8)

perpetrator9_gb = perpetartor_gb(perpetrator9)

perpetrator10_gb = perpetartor_gb(perpetrator10)
perpetrator_all = data_matrix.groupby('Relationship')['Incident'].sum()

perpetrator_all
def perpetartor_count_wise_(i, name):

    perpetrator = data_matrix.loc[data_matrix['Perpetrator Count'] == i]

    perpetrator_percentage = (len(perpetrator) / len(data_matrix)) * 100

    perpetrator_percentage

    formated = f'{i} Perpetrator: {perpetrator_percentage}%'

    return formated



title = perpetartor_count_wise_(0, 'individual')

title
overall_perpetrator = data_matrix.groupby('Relationship')['Incident'].sum()





def relationship_plot(var, name, i):

    plt.figure(figsize = (20,8))



    sns.barplot(var.index, var)

    percentages = perpetartor_count_wise_(i, name)

    title = f'{name} Perpetrator Relationship With Victim\n\n{percentages}'

    plt.xticks(rotation = 45)

    plt.title(title)

    plt.show()



    

    

relationship_plot(perpetrator0_gb, 'Undefined',0)

relationship_plot(perpetrator1_gb, 1, 1)

relationship_plot(perpetrator2_gb, 2,2)

relationship_plot(perpetrator3_gb, 3,3)

relationship_plot(perpetrator4_gb, 4,4)

relationship_plot(perpetrator5_gb, 5,5)

relationship_plot(perpetrator6_gb, 6,6)

relationship_plot(perpetrator7_gb, 7,7)

relationship_plot(perpetrator8_gb, 8,8)

relationship_plot(perpetrator9_gb, 9,9)

relationship_plot(perpetrator10_gb, 10,10)







plt.figure(figsize = (20,8))



sns.barplot(overall_perpetrator.index, overall_perpetrator)

plt.xticks(rotation = 'vertical')

title = 'Overall Perpetrator Relationship With Victim'

plt.title(title)

plt.show()

def return_arguments_var(data, series1, value1, series2 = None, value2= None):

    if series2:

        args = data.loc[(data[series1] == value1) & (data[series2] == value2)]

        

    else:

        args = data.loc[(data[series1] == value1)]

    return args





unsolved_cases = return_arguments_var(data_matrix,'Crime Solved', 'No')

all_cases_done_by_unknown = return_arguments_var(data_matrix, 'Relationship', 'Unknown')
print(f'Number of Cases Didnt Solved: {len(unsolved_cases)}')

print(f'Percentage Crime Didnt Solved: {(len(unsolved_cases)/len(data_matrix))*100}%')

print(f'Percentage Crime Done by Unknown: {(len(all_cases_done_by_unknown)/len(data_matrix))*100}%')



unknown_and_not_solved = data_matrix.loc[(data_matrix['Relationship'] == 'Unknown') & (data_matrix['Crime Solved'] == 'No')]

print(f'Number of Crime has done by Unknown and Hasnt Solved: {len(unknown_and_not_solved)}')

print(f'Percentage Crime Done By Unknown and Hasnt Solved: {(len(unknown_and_not_solved)/len(data_matrix))*100}%')
print(len(perpetrator0))

print(len(all_cases_done_by_unknown))
perpetrator0_unknown = return_arguments_var(data_matrix, 'Relationship','Unknown', 'Perpetrator Count', 0)
perpetrator0_unknown.shape
print(f'Percentage of Crime Unknown Vs 0 Perpetarator:\n {(len(all_cases_done_by_unknown) / len(perpetrator0))*100}%')
year1980 = return_arguments_var(data_matrix, 'Year', 1980)

year1981 = return_arguments_var(data_matrix, 'Year', 1981)

year1982 = return_arguments_var(data_matrix, 'Year', 1982)

year1983 = return_arguments_var(data_matrix, 'Year', 1983)

year1984 = return_arguments_var(data_matrix, 'Year', 1984)

year1985 = return_arguments_var(data_matrix, 'Year', 1985)

year1986 = return_arguments_var(data_matrix, 'Year', 1986)

year1987 = return_arguments_var(data_matrix, 'Year', 1987)

year1988 = return_arguments_var(data_matrix, 'Year', 1988)

year1989 = return_arguments_var(data_matrix, 'Year', 1989)

year1990 = return_arguments_var(data_matrix, 'Year', 1990)

year1991 = return_arguments_var(data_matrix, 'Year', 1991)

year1992 = return_arguments_var(data_matrix, 'Year', 1992)

year1993 = return_arguments_var(data_matrix, 'Year', 1993)

year1994 = return_arguments_var(data_matrix, 'Year', 1994)

year1995 = return_arguments_var(data_matrix, 'Year', 1995)

year1996 = return_arguments_var(data_matrix, 'Year', 1996)

year1997 = return_arguments_var(data_matrix, 'Year', 1997)

year1998 = return_arguments_var(data_matrix, 'Year', 1998)

year1999 = return_arguments_var(data_matrix, 'Year', 1999)

year2000 = return_arguments_var(data_matrix, 'Year', 2000)

year2001 = return_arguments_var(data_matrix, 'Year', 2001)

year2002 = return_arguments_var(data_matrix, 'Year', 2002)

year2003 = return_arguments_var(data_matrix, 'Year', 2003)

year2004 = return_arguments_var(data_matrix, 'Year', 2004)

year2005 = return_arguments_var(data_matrix, 'Year', 2005)

year2006 = return_arguments_var(data_matrix, 'Year', 2006)

year2007 = return_arguments_var(data_matrix, 'Year', 2007)

year2008 = return_arguments_var(data_matrix, 'Year', 2008)

year2009 = return_arguments_var(data_matrix, 'Year', 2009)

year2010 = return_arguments_var(data_matrix, 'Year', 2010)

year2011 = return_arguments_var(data_matrix, 'Year', 2011)

year2012 = return_arguments_var(data_matrix, 'Year', 2012)

year2013 = return_arguments_var(data_matrix, 'Year', 2013)

year2014 = return_arguments_var(data_matrix, 'Year', 2014)











def groupby(data, series1, on_groupby):

    year = data.groupby(series1)[on_groupby].sum()

    return year





year_wise_incident = groupby(data_matrix, 'Year', 'Incident')

year_wise_victim = groupby(data_matrix, 'Year', 'Victim Count')

year_wise_perpetrator = groupby(data_matrix, 'Year', 'Perpetrator Count')

year_wise_crime_solved = groupby(data_matrix, 'Year', 'Crime Solved')
plt.figure(figsize=(20,8))

year_wise_incident.plot(c='r')

plt.title('Incident Each Year')

plt.show()
plt.figure(figsize=(20,8))

year_wise_victim.plot(label = 'Victim Count Each Year')

year_wise_perpetrator.plot(label = 'Perpetrator Count Each Year')

plt.legend(loc = 'best')
def year_month_df(data):

    year_month = data.groupby('Month')['Incident'].sum()

    year_month = pd.DataFrame(year_month)

    return year_month

def year_month_decode(data, decode_list):

    year_month['MonthCode'] = decode_list

    return year_month
data_matrix['Month']=data_matrix['Month'].replace(['January','February', 'March', 'April', 'May', 'June', 'July',

                              'August','September', 'October', 'November','December', ],

                            [1,2,3,4,5,6,7,8,9,10,11,12])
data_matrix['new']= data_matrix['Year'].map(str) + ' ' + data_matrix['Month'].map(str)
data_matrix['new'] = pd.to_datetime(data_matrix['new'])
data_matrix.groupby('new')['Incident'].sum()
data_matrix['new1'] = data_matrix['new'].dt.to_period('M')
data_matrix.columns = ['Victim Sex', 'Perpetrator Sex', 'Victim Age', 'Perpetrator Age',

       'Victim Count', 'Perpetrator Count', 'City', 'State', 'Year', 'Month',

       'Incident', 'Relationship', 'Weapon', 'Crime Type', 'Crime Solved',

       'new', 'YMonth']
data_matrix.drop(columns = ['Year', 'Month', 'new'], inplace= True)
data_matrix.head()
monthly_incident = data_matrix.groupby('YMonth')['Incident'].sum()

monthly_perpetrator = data_matrix.groupby('YMonth')['Perpetrator Count'].sum()

monthly_victim = data_matrix.groupby('YMonth')['Victim Count'].sum()
plt.figure(figsize=(20,8))

monthly_incident.plot(c='r')

plt.title('Incident Per Month In Each Year')

plt.show()
plt.figure(figsize=(20,8))

monthly_victim.plot(label = 'Victim', c= 'r')

monthly_perpetrator.plot(label= 'Perpetrator', c = 'w')

plt.legend(loc='best')

plt.title('Analysis Per Month In each year')

plt.show()
monthly_victim[120:]
monthly_victim[180:]
monthly_victim.describe()
plt.figure(figsize=(20,8))

monthly_perpetrator.plot()

plt.show()
monthly_perpetrator = pd.DataFrame(monthly_perpetrator)

monthly_perpetrator.reset_index(inplace = True)
monthly_perpetrator.reset_index(inplace=True)
monthly_perpetrator
time = monthly_perpetrator['index'].values

time[:10]

series = monthly_perpetrator['Perpetrator Count'].values

series[:10]
split_time = 380

train_time = time[:split_time]

train_x = series[:split_time]



valid_time = time[split_time:]

valid_x = series[split_time:]



window_size = 12

batch_size = 10

shuffle_buffer_size = 420
import tensorflow as tf 



def windowed_dataset(series, window_size, batch_size, shuffle_buffer):

    dataset = tf.data.Dataset.from_tensor_slices(series)

    dataset = dataset.window(window_size+1, shift=1, drop_remainder = True)

    dataset = dataset.flat_map(lambda window: window.batch((window_size + 1 )))

    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))

    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset



perpetrator_dataset = windowed_dataset(train_x, window_size, batch_size, shuffle_buffer_size)
model_sl = tf.keras.Sequential([

            tf.keras.layers.Dense(1, input_shape = [window_size])

])
model_sl.compile(loss= 'mse', optimizer = tf.keras.optimizers.RMSprop(1e-6,momentum= 0.9))
history = model_sl.fit(perpetrator_dataset, epochs= 500, verbose=1)
def plot_series(time, series, format = '', start= 0, end= None, label = None):

  plt.plot(time[start:end], series[start:end], format, label= label, )

  plt.xlabel('Time')

  plt.ylabel('Value')

  if label:

    plt.legend(fontsize = 10)

plt.figure(figsize=(20,8))

loss = history.history['loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'w', label = 'Training Loss')

plt.show()
plot_loss = loss[200:]

plt.plot(epochs[200:], plot_loss, 'b', label = 'Training Loss')

plt.show()
forecast = []

for time in range(len(series) - window_size):

  forecast.append(model_sl.predict(series[time: time + window_size][np.newaxis]))



forecast = forecast[split_time - window_size:]

results = np.array(forecast)[:, 0,0]



plt.figure(figsize =(20,8) )



plot_series(valid_time, valid_x)

plot_series(valid_time, results)
tf.keras.metrics.mean_absolute_error(valid_x, results).numpy()
model_sl.save('model_sl.h5')
from tensorflow import keras as tk

from tensorflow.keras import layers as tkl

from tensorflow.keras.layers import Dense as tkld
# as our dataset is 2 dim but rnn need 3 dim we can expand dim using Lambda layer

import tensorflow as tf



model_rnn = tk.models.Sequential([

                              tkl.Lambda(lambda x: tf.expand_dims(x, axis = -1), input_shape = [None]),

                              tkl.SimpleRNN(20, return_sequences= True, input_shape = [None, 1]),

                              tkl.SimpleRNN(20),

                              tkld(1),

                              tkl.Lambda(lambda x: x*100.0)

])



# as default activation function of rnn is tanh, our output range from -1,1.. so we are scaling to normal sequence like 30,40 as time series are like that
lr_schedule = tk.callbacks.LearningRateScheduler(lambda epoch: 1e-8* 10**(epoch /20))



optimizer = tk.optimizers.RMSprop(lr = 1e-6, momentum= 0.9)
model_rnn.compile(loss = 'mse', optimizer= optimizer, metrics= ['mae'])
history = model_rnn.fit(perpetrator_dataset, epochs= 500,verbose= 1)
plt.figure(figsize=(20,8))

loss = history.history['loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'w', label = 'Training Loss')

plt.show()
plt.figure(figsize=(20,8))

loss = history.history['loss']

epochs = range(120,len( loss))

plot_loss = loss[120:]



plt.plot(epochs, plot_loss, 'w', label = 'Training Loss')

plt.show()
forecast = []

for time in range(len(series) - window_size):

  forecast.append(model_rnn.predict(series[time: time + window_size][np.newaxis]))



forecast = forecast[split_time - window_size:]

results = np.array(forecast)[:, 0,0]



plt.figure(figsize =(20,8) )



plot_series(valid_time, valid_x)

plot_series(valid_time, results)
tk.metrics.mean_absolute_error(valid_x, results).numpy()