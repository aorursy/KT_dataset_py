

from scipy.stats import boxcox



import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import os

from sklearn.decomposition import PCA

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv(os.path.join(dirname, filename))

df.shape

df.head()
df.info
df.columns
missing = df.isnull().sum(axis=0).reset_index()

missing.columns = ['columns_name','missing_count']

missing['missing_ratio'] = round(missing['missing_count'] /df.shape[0],2)

missing
df.Description.unique()

#1780093 unique values
df.TMC.unique()
df.TMC.nunique()

df.TMC.mode()
df = df[['Severity','Start_Time', 'End_Time','Start_Lat', 'Start_Lng','Distance(mi)','Side','City', 'State','County',

       'Zipcode', 'Timezone', 'Airport_Code', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',

       'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)','Wind_Chill(F)',

       'Precipitation(in)','Weather_Timestamp','Weather_Condition', 'Crossing',

       'Junction', 'Station','Traffic_Signal', 

       'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight',

       'Astronomical_Twilight' ]]
#fill missing values



#fill categorical variables with most popular category

df['Weather_Condition']=df['Weather_Condition'].fillna("Other")  

df['Wind_Direction']=df['Wind_Direction'].fillna("Other") 

df['Sunrise_Sunset']=df['Sunrise_Sunset'].fillna("Day") 

df['Civil_Twilight']=df['Civil_Twilight'].fillna("Day")

df['Nautical_Twilight']=df['Nautical_Twilight'].fillna("Day")

df['Astronomical_Twilight']=df['Astronomical_Twilight'].fillna("Day")





#fill continuous variables with median



m1 = df['Wind_Speed(mph)'].median()

df['Wind_Speed(mph)']=df['Wind_Speed(mph)'].fillna(m1) 

m2= df['Visibility(mi)'].median()

df['Visibility(mi)']=df['Wind_Speed(mph)'].fillna(m2) 

m3 = df['Pressure(in)'].median()

df['Pressure(in)']=df['Wind_Speed(mph)'].fillna(m3)

m4 = df['Humidity(%)'].median()

df['Humidity(%)']=df['Humidity(%)'].fillna(m4)

m5 = df['Humidity(%)'].median()

df['Humidity(%)']=df['Humidity(%)'].fillna(m5)

m6 = df['Temperature(F)'].median()

df['Temperature(F)']=df['Temperature(F)'].fillna(m6)



#Wind_Chill(F) missing a lot 

#Precipitation(in) missing a lot



df = df.loc[df['Wind_Chill(F)'].notnull()]

df = df.loc[df['Precipitation(in)'].notnull()]

# df = df.loc[df['TMC'].notnull()]

print(df.count())
from matplotlib import pyplot

pyplot.hist(df['Pressure(in)'])

pyplot.show()

pyplot.hist(df['Visibility(mi)'])

pyplot.show()

pyplot.hist(df['Wind_Speed(mph)'])

pyplot.show()

pyplot.hist(df['Precipitation(in)'])

pyplot.show()
#transform the data

# df['log_Pressure(in)'] = df['Pressure(in)'].apply(lambda x: np.log(x+0.000001))

# df['log_Visibility(mi)'] = df['Visibility(mi)'].apply(lambda x: np.log(x+0.000001))

# df['log_Wind_Speed(mph)'] = boxcox(df['Wind_Speed(mph)'].apply(lambda x: x+0.000001),0)

# df['log_Precipitation(in)'] = boxcox(df['Precipitation(in)'].apply(lambda x: x+0.000001),0)

x = df[['Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)']].values

x = StandardScaler().fit_transform(x)

normalized =pd.DataFrame(x,columns=['Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)'])

df['c_Pressure(in)'] =normalized['Pressure(in)']

df['c_Visibility(mi)'] =normalized['Visibility(mi)']

df['c_Wind_Speed(mph)'] =normalized['Wind_Speed(mph)']

df['c_Precipitation(in)'] =normalized['Precipitation(in)']
df['c_Pressure(in)'].std()
from matplotlib import pyplot

pyplot.hist(df['c_Pressure(in)'])

pyplot.show()

pyplot.hist(df['c_Visibility(mi)'])

pyplot.show()

pyplot.hist(df['c_Wind_Speed(mph)'])

pyplot.show()

pyplot.hist(df['c_Precipitation(in)'])

pyplot.show()
#change data type

df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')

df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

df['Weather_Timestamp'] = pd.to_datetime(df['Weather_Timestamp'], errors='coerce')

df['Year']=df['Start_Time'].dt.year

df['Month']=df['Start_Time'].dt.strftime('%b')

df['Day']=df['Start_Time'].dt.day

df['Hour']=df['Start_Time'].dt.hour

df['Weekday']=df['Start_Time'].dt.strftime('%a')

# df['TMC']=df['TMC'].astype(str)

#generate duration of accident

df['Duration']=(df['End_Time']-df['Start_Time'])/np.timedelta64(1,'m')



#drop negative duration

df = df.loc[df['Duration']>0]
#simplize categorical variables

# df.groupby("Weather_Condition").count()[['ID']].sort_values(['ID'], ascending=False).head(20)  #get top 5 condition and leave the other as other

df.loc[df["Weather_Condition"].str.contains("Cloudy|Clouds|Overcast", case=False),"Weather_Condition"]="Cloudy"

df.loc[df["Weather_Condition"].str.contains("Clear|Fair", case=False),"Weather_Condition"]="Clear/Fair"

df.loc[df["Weather_Condition"].str.contains("Rain", case=False),"Weather_Condition"]="Rain"

df.loc[df["Weather_Condition"].str.contains("Heavy Rain|Storm|Shower", case=False),"Weather_Condition"]="Heavy Rain"

df.loc[df["Weather_Condition"].str.contains("Snow|Sleet|Ice", case=False),"Weather_Condition"]="Snow"

df.loc[df["Weather_Condition"].str.contains("Heavy Snow|Heavy Sleet|Heavy Ice Pellets|Snow Showers|Squalls", case=False),"Weather_Condition"]="Heavy Snow"

df.loc[df["Weather_Condition"].str.contains("Haze", case=False),"Weather_Condition"]="Haze"

df.loc[df["Weather_Condition"].str.contains("Fog", case=False),"Weather_Condition"]="Fog"

df.loc[~df["Weather_Condition"].isin(["Clear/Fair","Cloudy","Light Rain","Light Snow","Rain","Snow","Haze","Fog"]),"Weather_Condition"]="Other"
df.Weather_Condition.unique()
#simplize categorical variables



df.loc[df["Wind_Direction"].str.contains("West|WSW|WNW|W", case=False),"Weather_Condition"]="West"

df.loc[df["Wind_Direction"].str.contains("E|East|ENE|ESE", case=False),"Weather_Condition"]="East"

df.loc[df["Wind_Direction"].str.contains("North|NNW|NNE", case=False),"Weather_Condition"]="North"

df.loc[df["Wind_Direction"].str.contains("South|SSW|SSE", case=False),"Weather_Condition"]="South"

df.loc[~df["Wind_Direction"].isin(["West","Calm","North","South","East","VAR"]),"Weather_Condition"]="Other"

df.Wind_Direction.unique()
# Generate dummies for categorical data

# prefix=['Side','Timezone','Wind_Direction','Weather_Condition','Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop','Sunrise_Sunset','Civil_Twilight','Nautical_Twilight', 'Astronomical_Twilight']

# df_dummy = pd.get_dummies(df,drop_first=True)
for i in df.columns:

    print(df[i].dtype)
#explorative analysis on correlation with severity

# column=[]

# correlation=[]

# for i in df.columns:

#     if df[i].dtype=="float64" or df[i].dtype=="int64":

#         column.append(i)

#         correlation.append(np.corrcoef(df[i].values, df.Severity.values)[0,1])

        

# d = pd.DataFrame({'column':column, 'correlation':correlation})

# d
# corr = df[d.column.tolist()].corr(method='spearman')

# fig, ax = plt.subplots(figsize=(10,10))

# sns.heatmap(corr,vmax=1,square = True)
# Export the data

df.to_csv('./cleaning.csv',index=False)
#split training and test dataset

y = df['Severity']

X = df.drop('Severity', axis=1)



# Split the data set into training and testing data sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)