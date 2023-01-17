import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
alcdata = pd.read_csv("/kaggle/input/iiitb-ai511ml2020-assignment-1/Assignment/alcoholism/student-mat.csv", low_memory=False)
fifadata = pd.read_csv("/kaggle/input/iiitb-ai511ml2020-assignment-1/Assignment/fifa18/data.csv", low_memory=False)
accidata1 = pd.read_csv("/kaggle/input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2005_to_2007.csv", low_memory=False)
accidata2 = pd.read_csv("/kaggle/input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2009_to_2011.csv", low_memory=False)
accidata3 = pd.read_csv("/kaggle/input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2012_to_2014.csv", low_memory=False)
alcdata.info()
alcdata.isnull().sum()
alcdata.columns
le = LabelEncoder()

le_cols = ['sex', 'address', 'famsize', 'Pstatus']

alcdata[le_cols] = alcdata[le_cols].apply(lambda col: le.fit_transform(col))
alcdata[le_cols].head()
bin_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

for col in bin_cols:
    alcdata[col] = np.where(alcdata[col].str.contains('yes'), 1,0)
    
alcdata[bin_cols].head()
alcdata = pd.get_dummies(alcdata, prefix_sep='_')
alcdata.head()
alcdata.columns
alcdata['G'] = alcdata[['G1', 'G2', 'G3']].mean(axis=1)

cat_vars1 = ['age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu']

fig, axes =  plt.subplots(2, 3, sharex=False, sharey= False, figsize=(24,16))
for i, ax in zip(cat_vars1, axes.flatten()):
    sns.violinplot(split=True, data = alcdata, x=i, y='G', hue='sex', ax=ax)
plt.show()
cat_vars2 =  ['traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid']

fig, axes =  plt.subplots(2, 3, sharex=False, sharey= False, figsize=(24,16))
for i, ax in zip(cat_vars2, axes.flatten()):
    sns.violinplot(split=True, data = alcdata, x=i, y='G', hue='sex', ax=ax)
plt.show()
cat_vars3 = ['activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel']

fig, axes =  plt.subplots(2, 3, sharex=False, sharey= False, figsize=(24,16))
for i, ax in zip(cat_vars3, axes.flatten()):
    sns.violinplot(split=True, data = alcdata, x=i, y='G', hue='sex', ax=ax)
plt.show()
cat_vars4 = ['freetime', 'goout', 'Dalc', 'Walc', 'health']

fig, axes =  plt.subplots(2, 3, sharex=False, sharey= False, figsize=(24,16))
for i, ax in zip(cat_vars4, axes.flatten()):
    sns.violinplot(split=True, data = alcdata, x=i, y='G', hue='sex', ax=ax)
plt.show()
cat_vars5 = ['school_GP', 'school_MS']

fig, axes =  plt.subplots(1, 2, sharex=False, sharey= False, figsize=(12,5))

for i, ax in zip(cat_vars5, axes.flatten()):
    sns.violinplot(split=True, data = alcdata, x=i, y='G', hue='sex', ax=ax)
plt.show()
cat_vars6 = ['Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher']

fig, axes =  plt.subplots(2, 3, sharex=False, sharey= False, figsize=(24,16))
for i, ax in zip(cat_vars6, axes.flatten()):
    sns.violinplot(split=True, data = alcdata, x=i, y='G', hue='sex', ax=ax)
plt.show()
cat_vars7 = ['Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher']

fig, axes =  plt.subplots(2, 3, sharex=False, sharey= False, figsize=(24,16))
for i, ax in zip(cat_vars7, axes.flatten()):
    sns.violinplot(split=True, data = alcdata, x=i, y='G', hue='sex', ax=ax)
plt.show()
cat_vars8 = ['reason_course', 'reason_home', 'reason_other', 'reason_reputation']

fig, axes =  plt.subplots(2, 2, sharex=False, sharey= False, figsize=(24,16))
for i, ax in zip(cat_vars8, axes.flatten()):
    sns.violinplot(split=True, data = alcdata, x=i, y='G', hue='sex', ax=ax)
plt.show()
cat_vars9 = ['guardian_father', 'guardian_mother', 'guardian_other']

fig, axes =  plt.subplots(1, 3, sharex=False, sharey= False, figsize=(24,10))
for i, ax in zip(cat_vars9, axes.flatten()):
    sns.violinplot(split=True, data = alcdata, x=i, y='G', hue='sex', ax=ax)
plt.show()
alcdata_enc = alcdata

alcdata_enc['U_M'] = np.where((alcdata_enc['address'] == 1) & (alcdata_enc['sex'] == 0),1,0)
alcdata_enc['U_F'] = np.where((alcdata_enc['address'] == 1) & (alcdata_enc['sex'] == 1),1,0)
alcdata_enc['R_M'] = np.where((alcdata_enc['address'] == 0) & (alcdata_enc['sex'] == 0),1,0)
alcdata_enc['R_F'] = np.where((alcdata_enc['address'] == 0) & (alcdata_enc['sex'] == 1),1,0)
plt.figure(figsize=(15, 8))
sns.violinplot(data = alcdata, x='famrel', y='G', hue='Pstatus')

#0 = Away
#1 = Together
plt.hist(alcdata.age)
plt.hist(alcdata.absences, bins=20)
plt.figure(figsize=(20, 5))
ax = plt.subplot(1,3,1)
alcdata1 = alcdata
alcdata1['absences'].replace({0.000000: 0.000001}, inplace=True)
plt.title('log-skew-remove')
plt.hist(np.log(alcdata1.absences), bins=20)

ax = plt.subplot(1,3,2)
plt.title('sqrt-skew-remove')
plt.hist(np.power(alcdata.absences, 1/2), bins=20)

ax = plt.subplot(1,3,3)
plt.title('cbrt-skew-remove')
plt.hist(np.power(alcdata.absences, 1/3), bins=20)

plt.show()
#Check the columns in the data given to us
fifadata.info()
#Columns 7,8,11,12
fifadata.isnull().sum()[7:9], fifadata.isnull().sum()[11:13]
import re

for i in fifadata['Overall']:
    if not re.match(r"^\d+$", str(i)):
        print(i)
for i in fifadata['Potential']:
    if not re.match(r"^\d+$", str(i)):
        print(i)
for i in fifadata['Value']:
    if not re.match(r"^€\d+\.?\d*(M|K)?$", i):
        print(i, count)
for i in fifadata['Wage']:
    if not re.match(r"^€\d+\.?\d*(M|K)?$", i):
        print(i)
def curr_clear(inp):
    inp = inp.strip("€")
    if(inp[-1] == 'M'):
        inp = float(inp[:-1])*(10**6)
    elif(inp[-1] == 'K'):
        inp = float(inp[:-1])*(10**3)
    else:
        inp = float(inp)
    
    return inp

def curr_clear_1(inp):
    if(type(inp) != float):
        inp = inp.strip("€")
        if(inp[-1] == 'M'):
            inp = float(inp[:-1])*(10**6)
        elif(inp[-1] == 'K'):
            inp = float(inp[:-1])*(10**3)
        else:
            inp = float(inp)
            
    return inp
fifadata['Release Clause'] = fifadata['Release Clause'].apply(lambda x : curr_clear_1(x))
fifadata['Value'] = fifadata['Value'].apply(lambda x : curr_clear(x))
fifadata['Wage'] = fifadata['Wage'].apply(lambda x : curr_clear(x))
plt.figure(figsize=(12, 8))
ax = plt.subplot(1,2,1)
sns.scatterplot(data=fifadata, x="Unnamed: 0", y="Wage", ax=ax)
ax = plt.subplot(1,2,2)
sns.scatterplot(data=fifadata, x="Unnamed: 0", y="Value", ax=ax)
#Number of Players in a club
club_count = fifadata.groupby("Club").size()

#Sum of Wages of Players in a Club
club_wage = fifadata.groupby("Club")['Wage']
club_wage.sum().sort_values(ascending=False)
club_value = fifadata.groupby("Club")['Value']

#Descending sort based on club valuation
club_value.sum().sort_values(ascending=False)
club_potential = fifadata.groupby("Club")['Potential']

#Descending sort based on club potential
club_potential.sum().sort_values(ascending=False)
club_overall = fifadata.groupby("Club")['Overall']

#Descending sort based on club overall skills
club_overall.sum().sort_values(ascending=False)
club_count['WO'] = club_wage.sum()/club_overall.sum()
club_count['WP'] = club_wage.sum()/club_potential.sum()
club_count['WO'].sort_values().head()
club_count['WP'].sort_values().head()
plt.figure(figsize=(15, 3))

ax = plt.subplot(1,2,1)
club_count['WO'].sort_values().head().plot(kind='bar')


ax = plt.subplot(1,2,2)
club_count['WP'].sort_values().head().plot(kind='bar')
plt.figure(figsize=(13, 8))
ax = plt.subplot(1,2,1)
sns.lineplot(data=fifadata, x="Overall", y="Wage")

#Approximate function to the graph on left
x= np.array([x for x in range(50, 95)])
y=np.power(1.15, x)
ax = plt.subplot(1,2,2)
plt.plot(x,y)
plt.show()
#Predicting the Average Predicted Wage for a given club
club_count['club_pred_wage'] = (1.15**(club_overall.sum()/club_count))*club_count
#Subtracting Prediction with the Club's Actual Wage
club_count['economy'] = club_wage.sum() - club_count['club_pred_wage']

#Converting Series Dataframe into dictionary
a = club_count['economy'].to_dict()
#Removing useless values in dictionary(values which are not floats)
prediction = {k:v for k,v in a.items() if type(v) == float}
#Sort the dictionary as per values
prediction = {k: v for k, v in sorted(prediction.items(), key=lambda item: item[1])}
prediction = dict(list(prediction.items())[0:5])
prediction
#Checking for missing values
fifadata['Age'].isnull().sum()
#Outlier Detection
sns.boxplot(x=fifadata['Age'])
plt.figure(figsize=(18, 8))

sns.violinplot(x="Age", y="Potential", data=fifadata)
plt.figure(figsize=(18, 8))

sns.boxplot(x="Potential", y="Age", data=fifadata, palette='viridis_r')
plt.figure(figsize=(15, 8))
sns.catplot(x="Age", y="Value", data=fifadata, height=6, aspect=1.9, palette='viridis_r')
fifadata['SprintSpeed'].isnull().sum()
fifadata['SprintSpeed'].fillna(fifadata['SprintSpeed'].median(), inplace = True)
sns.scatterplot(x="Unnamed: 0", y = 'SprintSpeed', data = fifadata)
plt.figure(figsize=(15, 8))
sns.boxplot(x='Age', y='SprintSpeed', data = fifadata, palette='viridis_r')
def height(inp):
    h_foot = float(inp.split("\'")[0])
    h_inch = float(inp.split("\'")[1])
    h_inch += (h_foot)*12
    #print(inp, h_inch, h_foot)
    h_cm = round(h_inch * 2.54, 1)
    return float(h_cm)
    
def weight(inp):
    return float(inp[:-3])
fifadata[['Height', 'Weight']].isnull().sum()
#Missing value removal with modal data
fifadata['Height'] = fifadata['Height'].fillna(fifadata['Height'].mode().iloc[0])
fifadata['Weight'] = fifadata['Weight'].fillna(fifadata['Weight'].mode().iloc[0])
#Data Sanitisation
fifadata['Height'] = fifadata['Height'].apply(lambda x : height(x))
fifadata['Weight'] = fifadata['Weight'].apply(lambda x : weight(x))
fifadata[['Height', 'Weight']].head()
fifadata['bmi'] = fifadata.apply(lambda row: round((row['Weight']*0.453592)/((row['Height']/100) ** 2), 2), axis=1)
fifadata['bmi'].head()
pot_cols = ['Potential', 'Age', 'Release Clause', 'bmi', 'Preferred Foot', 'Dribbling', 'Finishing', 'ShortPassing', 'BallControl', 'SprintSpeed', 'Agility', 'Stamina', 'Vision', 'Work Rate', 'Body Type']
fifadata[pot_cols].isnull().sum()
fifadata[pot_cols]['Release Clause'].fillna(0)
pot_df = fifadata.dropna(subset = ["Preferred Foot"])
pot_df[pot_cols].isnull().sum()
sns.lmplot(data=pot_df, x='bmi', y='Potential', scatter_kws={'alpha':0.3, 'color':'y'})

sns.lmplot(data=pot_df, x='bmi', y='Potential', order = 2, scatter_kws={'alpha':0.3, 'color':'y'})
corr = pot_df[pot_cols].corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, annot=True, vmin=-1, cmap="coolwarm")
sns.lmplot(data=fifadata, x='Overall', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'})

sns.lmplot(data=fifadata, x='Overall', y='Wage', order=2, scatter_kws={'alpha':0.3, 'color':'y'})
sns.lmplot(data=fifadata, x='Potential', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'})

sns.lmplot(data=fifadata, x='Potential', y='Wage', order = 2, scatter_kws={'alpha':0.3, 'color':'y'})

sns.lmplot(data=fifadata, x='Potential', y='Wage', order = 3, scatter_kws={'alpha':0.3, 'color':'y'})
sns.lmplot(data=fifadata, x='International Reputation', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'})
sns.lmplot(data=fifadata, x='Age', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'})

sns.lmplot(data=fifadata, x='Age', y='Wage', order = 2, scatter_kws={'alpha':0.3, 'color':'y'})
sns.lmplot(data=fifadata, x='Release Clause', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'})
sns.lmplot(data=fifadata, x='Weight', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'})

sns.lmplot(data=fifadata, x='Weight', y='Wage', order = 2, scatter_kws={'alpha':0.3, 'color':'y'})
sns.lmplot(data=fifadata, x='Height', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'})

sns.lmplot(data=fifadata, x='Height', y='Wage', order = 2, scatter_kws={'alpha':0.3, 'color':'y'})
sns.lmplot(data=fifadata, x='Skill Moves', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'})

sns.lmplot(data=fifadata, x='Skill Moves', y='Wage', order = 2, scatter_kws={'alpha':0.3, 'color':'y'})
num_cols = ['Age', 'Overall', 'Potential', 'Value', 'Wage', 'Special','International Reputation', 'Release Clause']
corr = fifadata[num_cols].corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, annot=True, vmin=-1, cmap="coolwarm")
club_age = fifadata.groupby("Club")['Age']
club_age.describe()
club_age.median().describe()
sorted_age = club_age.median().sort_values()
sorted_age.head()
sorted_age.tail()
youth = 23
fifadata1 = fifadata.apply(lambda x : 1 if x['Age'] <= youth else 0, axis = 1)
fifadata.assign(Count = fifadata1).groupby('Club')['Count'].sum().sort_values(ascending=False)
accidata1.info()
accidata2.info()
accidata3.info()
frames = [accidata1, accidata2, accidata3]
accidata = pd.concat(frames)
accidata.info()
accidata[['Number_of_Casualties', 'Day_of_Week']].isnull().sum()
sns.boxplot(accidata['Number_of_Casualties'])
day_grp = accidata.groupby('Day_of_Week')
day_grp['Number_of_Casualties'].describe()
plt.figure(figsize=(15, 6))
sns.lineplot(x = 'Day_of_Week', y = 'Number_of_Casualties', data = accidata, estimator=lambda x: len(x))
#Sorting number of causalties per day 
day_grp['Number_of_Casualties'].describe().sort_values(by='count', ascending=False)[['count', 'min', 'max']]
accidata['Speed_limit'].isnull().sum()
day_grp['Speed_limit'].describe()[['min', 'max', 'mean']]
accidata['Light_Conditions'].unique()
accidata['Weather_Conditions'].unique()
print('Number of missing values in Light_Conditions column is, ',accidata['Light_Conditions'].isnull().sum())
print('Number of missing values in Weather_Conditions column is, ',accidata['Weather_Conditions'].isnull().sum())
#Filling the missing values in Weather_Conditions with Unknown
accidata['Weather_Conditions'].fillna('Unknown', inplace=True)
plt.figure(figsize=(20, 6))
sns.countplot(data=accidata, x='Weather_Conditions', hue ='Accident_Severity', palette="rocket_r")
# accidata['Weather_Conditions'].value_counts().plot.pie()
plt.figure(figsize=(18, 6))
sns.countplot(data=accidata, x='Light_Conditions', hue = 'Accident_Severity', palette="rocket_r")
accidata.drop(['Accident_Index', 'Location_Easting_OSGR', 'Location_Northing_OSGR', 'Junction_Detail', 'Junction_Control', 'LSOA_of_Accident_Location', 'Year'], axis=1, inplace=True)
accidata.info()
#Dropped with null because can't use mode here as location will completely change
accidata.dropna(subset = ['Latitude', 'Longitude'], inplace=True)

#Used the mode of the data here
acci_cols = ['Weather_Conditions', 'Light_Conditions', 'Road_Surface_Conditions', 'Did_Police_Officer_Attend_Scene_of_Accident', 'Time', 'Pedestrian_Crossing-Human_Control', 'Pedestrian_Crossing-Physical_Facilities', 'Special_Conditions_at_Site', 'Carriageway_Hazards']
accidata[acci_cols] = accidata[acci_cols].fillna(accidata[acci_cols].mode().iloc[0])

accidata.isnull().sum()
#Accident Severity on the map based on latitudes and longitudes
plt.figure(figsize=(5,10))
sns.scatterplot(x='Longitude',y='Latitude',data=accidata, hue = 'Accident_Severity')
sns.catplot(x='Police_Force',y='Accident_Severity',data=accidata, kind="point", height = 5, aspect = 2)
plt.figure(figsize=(20, 5))
sns.violinplot(x='Number_of_Vehicles', y='Accident_Severity', data = accidata)
accidata['Date'] = pd.to_datetime(accidata['Date'])
accidata.nunique()
# accidata['Did_Police_Officer_Attend_Scene_of_Accident'].unique()
accidata['Did_Police_Officer_Attend_Scene_of_Accident'] = np.where(accidata['Did_Police_Officer_Attend_Scene_of_Accident'].str.contains('Yes'), 1,0)
accidata.info()
le_acc_cols = ['Weather_Conditions', 'Light_Conditions', 'Pedestrian_Crossing-Human_Control', 'Road_Surface_Conditions','Local_Authority_(Highway)', 'Road_Type', 'Pedestrian_Crossing-Physical_Facilities', 'Special_Conditions_at_Site', 'Carriageway_Hazards']

accidata[le_acc_cols] = accidata[le_acc_cols].apply(lambda col: le.fit_transform(col))
accidata[le_acc_cols].head()
# accidata.info()
# edit_cols = ['Road_Surface_Conditions', 'Pedestrian_Crossing-Human_Control']
accidata_enc = accidata
accidata_enc.dtypes