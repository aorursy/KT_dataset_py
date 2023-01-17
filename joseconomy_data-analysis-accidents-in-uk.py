#importing the libraries 
import numpy as np 
import pandas as pd 
import os 
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
#Importing the dataset
casualties=pd.read_csv("../input/Casualties0514.csv")

#Info of the columns
casualties.info()
#Checking the null values
casualties.isnull().sum()
casualties.head()
#Heatmap to see correlations
plt.figure(figsize=(15,10))
corr=casualties.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, linewidths=.5,annot=True,mask=mask)

#Values of the variable gender
casualties['Sex_of_Casualty'].value_counts()
#I'm going to drop the values -1 which are the null values
casualties = casualties[casualties.Sex_of_Casualty != -1]

#Transforming the variable into a categorical one
def map_sex(sex):
    if sex == 1:
        return 'Male'
    elif sex == 2:
        return 'Female'

casualties['Sex_of_Casualty'] = casualties['Sex_of_Casualty'].apply(map_sex)
sns.set(style="darkgrid")
plt.figure(figsize=(12,8))
genderplot = sns.countplot(x='Sex_of_Casualty',data=casualties)
genderplot.set(xlabel='Sex', ylabel='Count')
for p in genderplot.patches: 
    height = p.get_height() 
    genderplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(casualties))*100)+'%',  
      ha="center") 

#Turning the variable into a categorical one
def map_age(age):
    if age == 1:
        return '0-5'
    elif age == 2:
        return '6-10'
    elif age == 3:
        return '11-15'
    elif age == 4:
        return '16-20'
    elif age == 5:
        return '21-25'
    elif age == 6:
        return '26-35'
    elif age == 7:
        return '36-45'
    elif age == 8:
        return '46-55'
    elif age == 9:
        return '56-65'
    elif age == 10:
        return '66-75'
    elif age == 11:
        return 'over 75'
    elif age == -1:
        return "Don't know"
    

casualties['Age_Band_of_Casualty'] = casualties['Age_Band_of_Casualty'].apply(map_age)
sns.set(style="darkgrid")
plt.figure(figsize=(15,10))
ageplot=sns.countplot(x='Age_Band_of_Casualty',data=casualties,order=['0-5','6-10','11-15','16-20','21-25','26-35','36-45','46-55','56-65','66-75','over 75'])
for p in ageplot.patches: 
    height = p.get_height() 
    ageplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(casualties))*100)+'%',  
      ha="center") 

#obtaining the values of the severity variable
casualties['Casualty_Severity'].value_counts()
# turning the variable into a categorical one
def map_severity(severity):
    if severity == 1:
        return 'Fatal'
    elif severity == 2:
        return 'Serious'
    elif severity == 3:
        return 'Slight'
    
casualties['Casualty_Severity'] = casualties['Casualty_Severity'].apply(map_severity)
sns.set(style="darkgrid")
plt.figure(figsize=(15,5))
severityplot = sns.countplot(x='Casualty_Severity',hue='Sex_of_Casualty',data=casualties,order=['Slight','Serious','Fatal'])
severityplot.set(xlabel='Severity', ylabel='Count')
for p in severityplot.patches: 
    height = p.get_height() 
    severityplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(casualties))*100)+'%',  
      ha="center") 

accidents= pd.read_csv("../input/Accidents0514.csv")
accidents = accidents[accidents.Weather_Conditions != -1]
accidents = accidents[accidents.Road_Surface_Conditions != -1]
accidents.info()
accidents.head()
#Heatmap to see correlations
plt.figure(figsize=(15,10))
corr=casualties.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, linewidths=.5,annot=True,mask=mask)
acc_time = accidents[['Date','Day_of_Week','Time']]

acc_time.head()
acc_time.info()
acc_time.dropna(axis=0,inplace=True)
#creating the column, hour,day,month and year
#creating year column
def year(string):
    return int(string[6:10])
acc_time['Year']=acc_time['Date'].apply(lambda x: year(x))
#creating month column
def month(string):
    return int(string[3:5])
acc_time['Month']=acc_time['Date'].apply(lambda x: month(x))
#creating day column
def day(string):
    return int(string[0:2])
acc_time['Day']=acc_time['Date'].apply(lambda x: day(x))
#creating hour column
def hour(string):
    s=string[0:2]
    return int(s)
acc_time['Hour']=acc_time['Time'].apply(lambda x: hour(x))






sns.set(style="darkgrid")
plt.figure(figsize=(15,5))
yearplot = sns.countplot(x='Year',data=acc_time)
yearplot.set(xlabel='Year', ylabel='Count')
for p in yearplot.patches: 
    height = p.get_height() 
    yearplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(acc_time))*100)+'%',  
      ha="center") 
plt.show()
sns.set(style="darkgrid")
plt.figure(figsize=(15,5))
monthplot = sns.countplot(x='Month',data=acc_time)
monthplot.set(xlabel='Month', ylabel='Count')
for p in monthplot.patches: 
    height = p.get_height() 
    monthplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(acc_time))*100)+'%',  
      ha="center") 
plt.show()
sns.set(style="darkgrid")
plt.figure(figsize=(15,5))
weekplot = sns.countplot(x='Day_of_Week',data=acc_time)
weekplot.set(xlabel='Day of week', ylabel='Count')
for p in weekplot.patches: 
    height = p.get_height() 
    weekplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(acc_time))*100)+'%',  
      ha="center") 
weekplot.set(xticklabels=['Monday','Tuesday','Wesnesday','Thursday','Friday','Saturday','Sunday'])
plt.show()
sns.set(style="darkgrid")
plt.figure(figsize=(15,8))
Hourplot = sns.countplot(x='Hour',data=acc_time)
Hourplot.set(xlabel='Hour', ylabel='Count')
for p in Hourplot.patches: 
    height = p.get_height() 
    Hourplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(acc_time))*100)+'%',  
      ha="center") 
plt.show()

#Creating a pivot table to get a heatmap with the concentration of accidents by month over the years
#I decide to create a column of ones to get a count of the accidents
acc_time['Ones']=1
table = pd.pivot_table(acc_time, values='Ones', index=['Month'],columns=['Year'], aggfunc=np.sum)
plt.figure(figsize=(20,10))
yticks = np.array(['January','February','March','April','May','June','July','August','September','October','November','December'])
sns.set(rc={"axes.labelsize":36},font_scale=2)
sns.heatmap(table, yticklabels=yticks,linewidths=.1,annot=False,cmap='magma')
df_conditions = accidents[['Light_Conditions','Weather_Conditions','Road_Surface_Conditions',]]

df_conditions.info()
df_conditions['Severity']=casualties['Casualty_Severity']

sns.set(style="darkgrid")
plt.figure(figsize=(15,5))
lightplot = sns.countplot(x='Light_Conditions',data=df_conditions,hue='Severity',hue_order=['Slight','Serious','Fatal'])
lightplot.set(xlabel='Light conditions', ylabel='Count',xticklabels=['Daylight','Darkness Light-Lit','Darkness Light-Unlit','Darkness-No light','Darkness unknown light'])
for p in lightplot.patches: 
    height = p.get_height() 
    lightplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(acc_time))*100)+'%',  
      ha="center")     
plt.show()

sns.set(style="darkgrid")
plt.figure(figsize=(15,8))
weatherplot = sns.countplot(x='Weather_Conditions',data=df_conditions)
weatherplot.set(xlabel='Weather conditions', ylabel='Count',xticklabels=['fine','Raining','Snowing','Fine/winds',
                                                                         'Raining/winds','Snowing/winds','Fog','Other','Unknown'])
for p in weatherplot.patches: 
    height = p.get_height() 
    weatherplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(acc_time))*100)+'%',  
      ha="center") 
plt.show()

sns.set(style="darkgrid")
plt.figure(figsize=(15,8))
roadplot = sns.countplot(x='Road_Surface_Conditions',data=df_conditions)
roadplot.set(xlabel='Road Surface conditions', ylabel='Count',xticklabels=['Dry','Wet','Snow','Frost','flood','oil','mud'])
for p in roadplot.patches: 
    height = p.get_height() 
    roadplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(acc_time))*100)+'%',  
      ha="center") 
plt.show()
#Loading the dataset
vehicles=pd.read_csv('../input/Vehicles0514.csv')
print(vehicles.shape)
vehicles.head()
#list of columns
list(vehicles)
#dropping the columns that we are not going to use
vehicles.drop(['Age_of_Driver', 'Age_Band_of_Driver','Sex_of_Driver' ], axis=1,inplace=True)
#Heatmap to see correlations
plt.figure(figsize=(15,10))
corr=vehicles.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, linewidths=.5,annot=False,mask=mask)
vehicles['Vehicle_Manoeuvre'].value_counts()
manoeuvre = vehicles
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != -1]
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != 8]
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != 15]
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != 6]
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != 11]
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != 12]
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != 14]
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != 1]
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != 10]
manoeuvre = manoeuvre[manoeuvre.Vehicle_Manoeuvre != 13]



plt.figure(figsize=(15,8))

manoplot = sns.countplot(x='Vehicle_Manoeuvre',data=manoeuvre)
manoplot.set(xlabel='Vehicle_Manoeuvre',ylabel='Count',xticklabels=['Parked','Waiting to go','Slowing/stoping','moving off','turning left','turning right','Going ahead \n left hand bend','Going ahead \n right hand bend','Going ahead \n other'])
for p in manoplot.patches: 
    height = p.get_height() 
    manoplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(manoeuvre))*100)+'%',  
      ha="center") 
plt.show()
vehicles['Junction_Location'].value_counts()
location = vehicles
location = location[location.Junction_Location != -1]
plt.figure(figsize=(15,8))

junctionplot = sns.countplot(x='Junction_Location',data=location)
junctionplot.set(xlabel='Junction_Location',ylabel='Count',xticklabels=['Not in\n junction','aproaching/parked \n junction','cleared \n junction','leaving \n roundabout','entering \n roundabout','leaving \n main road','entering \n main road','entering from \n slip road','mid junction'])
for p in junctionplot.patches: 
    height = p.get_height() 
    junctionplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(location))*100)+'%',  
      ha="center") 
plt.show()
vehicles['1st_Point_of_Impact'].value_counts()
vehicles['first_point_of_impact']=vehicles['1st_Point_of_Impact']
vehicles = vehicles[vehicles.first_point_of_impact != -1]
plt.figure(figsize=(15,8))

impactplot = sns.countplot(x='first_point_of_impact',data=vehicles)
impactplot.set(xlabel='first_point_of_impact',ylabel='Count',xticklabels=['did not \n impact','front','back','offside','nearside'])
for p in impactplot.patches: 
    height = p.get_height() 
    impactplot.text(p.get_x()+p.get_width()/2., 
      height + 3, 
      '{:1.2f}'.format((height/len(vehicles))*100)+'%',  
      ha="center") 
plt.show()
