#import libraries
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Operations
def Avgperday(data):
  avgperday = data.groupby('Day')['AvgTemperature'].mean()
  return avgpermonth
def Avgpermonth(data):
  avgpermonth = data.groupby('Month')['AvgTemperature'].mean()
  return avgpermonth
def Avgperyear(data):  
  avgperyear = data.groupby('Year')['AvgTemperature'].mean()
  return avgperyear

# chart function
def plot_chart(data,info): 
  plt.figure(figsize=(15,5))
  plt.plot(data.reset_index()[info],data.values,"o-",linestyle='solid',label="Temperature")
  plt.xlabel('{}'.format(info),fontsize=16)
  plt.ylabel('Average Temperature [°C]',fontsize=16)
  plt.title('Temperature per {}'.format(info),fontsize=16)
  plt.grid(color='r', linestyle='dotted', linewidth=0.5)
  plt.legend()
  plt.show()
def GlobalTemperature_visual(chart_time,inicial_date = '1995-01-01',final_date = '2019-12-31',Region=None,Country=None,City=None):  
  """
   Funtion which returns a chart of the Average Temperature from a given location and time.

  If any location is specified the function returns a chart related with the data from the whole world.
  To know what is the exactly list of Regions/Country/City for the parameters, involke the GlobalTemperature_dataInfo.
  The Parameter chart_time only accepts the following string names: [Day, Month, Year].
  The correct format for the inicial_date and final_date is: Year-Month-Day.
  If not specified, the inicial_date and final_date are: '1995-01-01', '2019-12-31'.
  The Parameters inicial_date and final_date only accepts values in the interval of: 1995-01-01 ------ 2019-12-31.
  """
  data = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv',low_memory=False) #import data
  data.drop(columns = 'State',inplace=True) #drop column state
  data['AvgTemperature'] = (data['AvgTemperature']-32)*(5/9) #transforming in Celsius
  remove = data.loc[(data['AvgTemperature']< -50)] #removing outliers
  data.drop(remove.index,inplace=True)
  remove = data.loc[(data['Year'] == 2020)] #removing data from incomplete year
  data.drop(remove.index,inplace=True)
  date = pd.to_datetime(data[['Month','Day','Year']],errors='coerce') #data format
  data['date'] = date #new column

#Location choice
  if Region != None and Country == None and City == None:
    if any(data['Region'].unique() == Region) == False:
      return print('Please check the list of locations and the spelling accepted using the funtion: GlobalTemperature_dataInfo ')
    data = data[data['Region'] == Region]

  elif Region == None and Country != None and City == None:
    if any(data['Country'].unique() == Country) == False:
      return print('Please check the list of locations and the spelling accepted using the funtion: GlobalTemperature_dataInfo ')
    data = data[data['Country'] == Country]

  elif Region == None and Country == None and City != None:
    if any(data['City'].unique() == City) == False:
      return print('Please check the list of locations and the spelling accepted using the funtion: GlobalTemperature_dataInfo ')
    data = data[data['City'] == City]

  elif Region == None and Country == None and City == None:
    data = data
  else:
    return print('Please select just one of types of location: Region, Country, City or let None in all for World data')



  #Date choice
  if inicial_date<'1995-01-01' or inicial_date>'2019-12-31':
    return print('Please choose a initial_date greater than 1995-01-01 and lesser than 2019-12-31.')
  elif final_date<'1995-01-01' or final_date>'2019-12-31':
    return print('Please choose a initial_date greater than 1995-01-01 and lesser than 2019-12-31.')

  data = data[(data['date'] >= inicial_date) & (data['date'] <= final_date)]



  #chart period choice
  if chart_time == 'Day':
    return plot_chart(Avgperday(data),chart_time)

  elif chart_time == 'Month':
    return plot_chart(Avgpermonth(data),chart_time)

  elif chart_time == 'Year':
    return plot_chart(Avgperyear(data),chart_time)

  elif chart_time == None:
    return plot_chart(Avgperyear(data),chart_time )
  
  else:
    return print('Please type one of the following: Day,Month,Year or None ')
def GlobalTemperature_values(chart_time,inicial_date = '1995-01-01',final_date = '2019-12-31',Region=None,Country=None,City=None):  
  """
  Funtion which returns the values of the Average Temperature from a given location and time.

  If any location is specified the function returns a chart related with the data from the whole world.
  To know what is the exactly list of Regions/Country/City for the parameters, involke the GlobalTemperature_dataInfo.
  
  The Parameter chart_time only accepts the following string names: [Day, Month, Year].

  The correct format for the inicial_date and final_date is: Year-Month-Day.
  If not specified, the inicial_date and final_date are: '1995-01-01', '2019-12-31'.
  The Parameters inicial_date and final_date only accepts values in the interval of: 1995-01-01 ------ 2019-12-31.

  """
  data = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv',low_memory=False) #import data
  data.drop(columns = 'State',inplace=True) #drop column state
  data['AvgTemperature'] = (data['AvgTemperature']-32)*(5/9) #transforming in Celsius
  remove = data.loc[(data['AvgTemperature']< -50)] #removing outliers
  data.drop(remove.index,inplace=True)
  remove = data.loc[(data['Year'] == 2020)] #removing data from incomplete year
  data.drop(remove.index,inplace=True)
  date = pd.to_datetime(data[['Month','Day','Year']],errors='coerce') #data format
  data['date'] = date #new column

   #Location choice
  if Region != None and Country == None and City == None:
    if any(data['Region'].unique() == Region) == False:
      return print('Please check the list of locations and the spelling accepted using the funtion: GlobalTemperature_dataInfo ')
    data = data[data['Region'] == Region]

  elif Region == None and Country != None and City == None:
    if any(data['Country'].unique() == Country) == False:
      return print('Please check the list of locations and the spelling accepted using the funtion: GlobalTemperature_dataInfo ')
    data = data[data['Country'] == Country]

  elif Region == None and Country == None and City != None:
    if any(data['City'].unique() == City) == False:
      return print('Please check the list of locations and the spelling accepted using the funtion: GlobalTemperature_dataInfo ')
    data = data[data['City'] == City]
  
  elif Region == None and Country == None and City == None:
    data = data

  else:
    return print('Please select just one of types of location: Region, Country, City or let None in all for World data')



  #Date choice
  if inicial_date<'1995-01-01' or inicial_date>'2019-12-31':
    return print('Please choose a initial_date greater than 1995-01-01 and lesser than 2019-12-31.')
  elif final_date<'1995-01-01' or final_date>'2019-12-31':
    return print('Please choose a initial_date greater than 1995-01-01 and lesser than 2019-12-31.')

  data = data[(data['date'] >= inicial_date) & (data['date'] <= final_date)]

  #chart period choice

  if chart_time == 'Day':
    return Avgperday(data)

  elif chart_time == 'Month':
    return Avgpermonth(data)

  elif chart_time == 'Year':
    return Avgperyear(data)

  elif chart_time == None:
    return Avgperyear(data)
  
  else:
    return print('Please type one of the following: Day,Month,Year or None ')
def GlobalTemperature_dataInfo(locations):
  """
  Function which returns a list of Region/Country/city accepted. 
  The Parameter chart_time only accepts the following string names: [Region,Country,City].
  """
  data = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv',low_memory=False) #import data
  data.drop(columns = 'State',inplace=True) #drop column state
  data['AvgTemperature'] = (data['AvgTemperature']-32)*(5/9) #transforming in Celsius
  remove = data.loc[(data['AvgTemperature']< -50)] #removing outliers
  data.drop(remove.index,inplace=True)
  remove = data.loc[(data['Year'] == 2020)] #removing data from incomplete year
  data.drop(remove.index,inplace=True)
  date = pd.to_datetime(data[['Month','Day','Year']],errors='coerce') #data format
  data['date'] = date #new column

  if locations == 'City':
    a = list(data['City'].unique())
    return sorted(a) 
  elif locations == 'Country':
    a = list(data['Country'].unique())
    return sorted(a)
  elif locations == 'Region':
    a = list(data['Region'].unique())
    return sorted(a)
  else:
    return print('Please select just one of types of location: Region, Country, City')
GlobalTemperature_dataInfo(locations = 'Regions')
GlobalTemperature_dataInfo(locations = 'Region')
GlobalTemperature_visual('Month',Region='Africa',Country=None,City='Sao Paulo')
GlobalTemperature_visual('Month',Region='Africaw',Country=None,City=None)
GlobalTemperature_visual('Year',Region='Africa',Country=None,City=None)
GlobalTemperature_visual('Year',Region='Africa',Country=None,City=None,inicial_date = '2000-01-01')
GlobalTemperature_visual('Year',Region=None,Country=None,City=None)
GlobalTemperature_values('Year',Region=None,Country=None,City=None)
Regions = GlobalTemperature_dataInfo(locations = 'Region') #taking the regions 
# datasets which will be used to make a chart of the avg_temp/regions/periods_of_time
first_t = pd.DataFrame(columns=Regions)
sec_t = pd.DataFrame(columns=Regions)
third_t = pd.DataFrame(columns=Regions)
fourth_t = pd.DataFrame(columns=Regions)
fifth_t = pd.DataFrame(columns=Regions)
last_t = pd.DataFrame(columns=Regions)

#using the previous functions to select the data from different Regions and periods of time 
for r in Regions:
  first_t[r] = GlobalTemperature_values('Year',Region=r,Country=None,City=None,inicial_date = '1995-01-01',final_date = '2000-01-01')
  sec_t[r] = GlobalTemperature_values('Year',Region=r,Country=None,City=None,inicial_date = '2000-01-01',final_date = '2005-01-01')
  third_t[r] = GlobalTemperature_values('Year',Region=r,Country=None,City=None,inicial_date = '2005-01-01',final_date = '2010-12-29')
  fourth_t[r] = GlobalTemperature_values('Year',Region=r,Country=None,City=None,inicial_date = '2010-01-01',final_date = '2015-12-29')
  fifth_t[r] = GlobalTemperature_values('Year',Region=r,Country=None,City=None,inicial_date = '2015-01-01',final_date = '2018-12-29')
  last_t[r] = GlobalTemperature_values('Year',Region=r,Country=None,City=None,inicial_date = '2019-01-01',final_date = '2019-12-29')

#adding data from the world for comparison 
first_t['World'] = GlobalTemperature_values('Year',Region=None,Country=None,City=None,inicial_date = '1995-01-01',final_date = '2000-01-01')
sec_t['World'] = GlobalTemperature_values('Year',Region=None,Country=None,City=None,inicial_date = '2000-01-01',final_date = '2005-01-01')
third_t['World'] = GlobalTemperature_values('Year',Region=None,Country=None,City=None,inicial_date = '2005-01-01',final_date = '2010-12-29')
fourth_t['World'] = GlobalTemperature_values('Year',Region=None,Country=None,City=None,inicial_date = '2010-01-01',final_date = '2015-12-29')
fifth_t['World'] = GlobalTemperature_values('Year',Region=None,Country=None,City=None,inicial_date = '2015-01-01',final_date = '2018-12-29')
last_t['World'] = GlobalTemperature_values('Year',Region=None,Country=None,City=None,inicial_date = '2019-01-01',final_date = '2019-12-29')

#join all the data into a new dataset
df = pd.DataFrame(columns = ['Regions','1995-01-01 -- 2000-01-01', '2000-01-01 -- 2005-01-01','2005-01-01 -- 2010-01-01','2010-01-01 -- 2015-01-01','2015-01-01 -- 2018-12-29','2019-01-01 -- 2019-12-29'])
df['Regions'] = first_t[:].mean().reset_index()['index']
df['1995-01-01 -- 2000-01-01'] = first_t[:].mean().reset_index()[0]
df['2000-01-01 -- 2005-01-01'] = sec_t[:].mean().reset_index()[0]
df['2005-01-01 -- 2010-01-01'] = third_t[:].mean().reset_index()[0]
df['2010-01-01 -- 2015-01-01'] = fourth_t[:].mean().reset_index()[0]
df['2015-01-01 -- 2018-12-29'] = fifth_t[:].mean().reset_index()[0]
df['2019-01-01 -- 2019-12-29'] = last_t[:].mean().reset_index()[0]

#converting the form of this dataset into a way that seaborn barplot get it
df = pd.melt(df,
             id_vars = ['Regions'],
             value_vars = ['1995-01-01 -- 2000-01-01', '2000-01-01 -- 2005-01-01','2005-01-01 -- 2010-01-01','2010-01-01 -- 2015-01-01','2015-01-01 -- 2018-12-29','2019-01-01 -- 2019-12-29'],
             var_name = 'Label',
             value_name = 'Value')
import seaborn as sns
plt.figure(figsize=(25,9))
sns.barplot(x ='Regions',y='Value',hue='Label',data=df)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=20)
plt.title('Avg Temperature [°C]/Region/Year', fontsize=20)
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
degrees = 35
plt.xticks(rotation=degrees)
plt.xlabel('Regions', fontsize=20)
plt.ylabel('Avg Temperature [°C]', fontsize=20)
plt.grid(color='r', linestyle='dotted', linewidth=0.5)
plt.tight_layout()
wd = GlobalTemperature_values('Year',Region=None,Country=None,City=None,inicial_date = '1995-01-01',final_date = '2019-12-31')
X = wd.axes[0].values
y = wd.values

#different models
model = np.poly1d(np.polyfit(X, y, 1))
model2 = np.poly1d(np.polyfit(X, y, 2))
model3 = np.poly1d(np.polyfit(X, y, 3))

#For the Next 10 years
Xnew =np.array([x for x in range(1995,2030,1)]).reshape(-1,1)
ynew = model(Xnew)

Xnew2 =np.array([x for x in range(1995,2030,1)]).reshape(-1,1)
ynew2 = model2(Xnew)

Xnew3 =np.array([x for x in range(1995,2030,1)]).reshape(-1,1)
ynew3 = model3(Xnew)


plt.figure(figsize=(12,7))
plt.scatter(X, y,label = 'Original Data')
plt.plot(Xnew, ynew,color='r', label = 'Prediction model (1°)')
plt.plot(Xnew2, ynew2,color='k', label = 'Prediction model (2°)')
plt.plot(Xnew3, ynew3,color='g', label = 'Prediction model (3°)')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Avg Temperature [°C]/Year', fontsize=20)
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.xlabel('Years', fontsize=15)
plt.ylabel('Avg Temperature [°C]', fontsize=15)
plt.grid(color='r', linestyle='dotted', linewidth=0.5)
plt.tight_layout()
plt.legend()
plt.show()