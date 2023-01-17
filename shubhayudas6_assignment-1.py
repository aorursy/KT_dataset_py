import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import gc   # To clean up dataframes
import warnings   # Hate these
warnings.filterwarnings('ignore')
alcdata = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/alcoholism/student-mat.csv")
fifadata = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/fifa18/data.csv")
# Never hurts to look at all the data first

print(alcdata.info())
alcdata.sample(10)   # Yes I am not using df.head(), so as to look at random data
# I like green and yellow :-) ... sorry
correlations = alcdata.corr().round(2)
plt.figure(figsize=(12., 12.))
sns.heatmap(correlations, cmap='viridis', vmin=-1, annot=correlations)
plt.title("Correlation matrix of all the non-categorical data")
plt.show()
print(f"Mean of G1: {alcdata.G1.mean(axis=0)}")
print(f"Mean of G2: {alcdata.G2.mean(axis=0)}")
print(f"Mean of G3: {alcdata.G3.mean(axis=0)}")
# Get the mean for every student
alcdata['overall_grade'] = alcdata[['G1', 'G2', 'G3']].mean(axis=1)

# Remove the individual grades, in another dataframe, thus preserving the original data
alcdata_modified = alcdata.drop(columns=['G1', 'G2', 'G3'])

# Go ahead and plot the correlations now
correlations = alcdata_modified.corr().round(2)
plt.figure(figsize=(14., 12.))
sns.heatmap(correlations, cmap='viridis', vmin=-1, annot=correlations)
plt.title("Correlation matrix of all the non-categorical data")
plt.show()
# First find out all the categorical columns and then find the unique values

# I needed a loop for this, but it is only for selecting a columns, so no trouble with performance!
for col in alcdata.select_dtypes('object').columns:
    print(alcdata[col].value_counts(), end='\n\n')
# Ok so let me make a list of most of the binary columns. These can be label encoded as: yes -> 1; no -> 0
binary_encoded = ['romantic', 'internet', 'higher', 'nursery', 'activities', 'paid', 'famsup', 'schoolsup']

# Now a list of columns, which needs to be one hot encoded. This is remove any sort of ordering
categorical = ['school', 'sex', 'address', 'famsize', 'Mjob', 'Fjob', 'reason', 'guardian', 'Pstatus']

# Some of the columns were already Label Encoded, I need to convert them to categorical data
pre_label_encoded = ['Medu', 'Fedu', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']
for col in pre_label_encoded:
    alcdata_modified[col] = alcdata_modified[col].astype('category')
    
    if col in alcdata.columns:
        alcdata[col] = alcdata[col].astype('category')
sns.boxplot(data=alcdata, x='reason', y='overall_grade')
plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
sns.boxplot(data=alcdata, x='Walc', y='overall_grade', ax=ax1)
sns.boxplot(data=alcdata, x='Dalc', y='overall_grade', palette='viridis', ax=ax2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
sns.boxplot(data=alcdata, x='Medu', y='overall_grade', ax=ax1)
sns.boxplot(data=alcdata, x='Fedu', y='overall_grade', palette='viridis', ax=ax2)
# Let me get the correlations for the encoded famrel and Pstatus

for i in range(1, 6):
    print(f"Correlation of (famrel == {i}) with overall grades: {ordered[np.where(ordered[:, 1] == 'famrel_'+str(i))[0][0], 0]}")

print()
for i in ['T', 'A']:
    print(f"Correlation of (Pstatus == {i}) with overall grades: {ordered[np.where(ordered[:, 1] == 'Pstatus_'+i)[0][0], 0]}")
plt.figure(figsize=(14,7))
sns.boxplot(data=alcdata, x='famrel', y='overall_grade')
plt.show()
plt.figure(figsize=(14,7))
sns.boxplot(data=alcdata, x='Pstatus', y='overall_grade')
plt.xticks(range(2), labels=['Living apart', 'Living together'])
plt.show()
# First get all the column names

alcdata.drop(columns=['G1', 'G2', 'G3'], inplace=True)
alcdata.columns.values
plt.hist(alcdata.overall_grade, histtype='step')
plt.title('Distribution of grades')
plt.show()
for feature in alcdata.select_dtypes('int').columns.values:
    plt.figure()
    sns.countplot(alcdata[feature])
    plt.title(feature.capitalize())
    plt.show()
print(fifadata.info())
pd.set_option('max_columns', None)
fifadata.sample(5)
# Removing some irrelavant columns, along with the positions of the players
# I don't really know what they mean
drop_cols = ['Unnamed: 0', 'ID', 'Photo', 'Flag', 'Club Logo',
             'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM',
           'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM',
           'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Real Face', 'Jersey Number']

fifadata.drop(columns=drop_cols, inplace=True)
price_cols = ['Value', 'Wage', 'Release Clause']
fifadata[price_cols].fillna({'Release Clause': '$0K', 'Wage': '$0K', 'Value': '$0K'}, inplace=True)

# Convert currency into floats, with proper scaling for number of zeros
def price_process(x):
    if type(x) == str:
        x = x.strip()
        if x.endswith('K'):
            return x[1:-1]
        elif x.endswith('M'):
            return str(float(x[1:-1]) * 1000)
        else:
            return x[1: ]
    elif type(x) == float:
        return x
    else:
        return 0

# Convert weight from objects to floats
def weight_process(x):
    if type(x) == str:
        x = x.strip()
        if x.endswith('lbs'):
            return x.replace('lbs', '')
    elif type(x) == float:
        return x
    else:
        return 0

# Convert height from feets + inches to inches only 
def height_process(x):
    if type(x) == str:
        x = x.strip()
        if "'" in x:
            temp = [int(t) for t in x.split("'")]
            return temp[0]*12+temp[1]
    elif type(x) == float:
        return x
    else:
        return 0
    
# Get year from date string
def date_process(x):
    if type(x) == str:
        x = x.strip()
        if "," in x:
            return x.split(',')[-1].strip()
        elif x.startswith('2'):
            return x
        else:
            return "0"
    else:
        return "0"
    
for col in price_cols:
    fifadata[col] = fifadata[col].apply(price_process)
    fifadata[col] = fifadata[col].astype('float')
    
fifadata.Weight = fifadata.Weight.apply(weight_process).astype('float')
fifadata.Height = fifadata.Height.apply(height_process).astype('float')
fifadata.Joined = fifadata.Joined.apply(date_process).astype('int')
fifadata['Contract Valid Until'] = fifadata['Contract Valid Until'].apply(date_process).astype('int')

fifadata['Contract Valid Until'].fillna(fifadata['Contract Valid Until'].mean(), inplace=True)
fifadata['Joined'].fillna(fifadata.Joined.mean(), inplace=True)
fifadata['Loaned From'].fillna('None', inplace=True)
pd.set_option('max_columns', None)
fifadata.sample(5)
temp = fifadata[['Club', 'Overall', 'Potential', 'Value', 'International Reputation', 'Work Rate', 'Wage']]
temp['Contract_Duration'] = fifadata.loc[:, 'Contract Valid Until'] - fifadata.loc[:, 'Joined']

temp['Work Rate'] = temp['Work Rate'].astype('category')
temp.dropna(subset=['Work Rate'], inplace=True)
temp['Work Rate'] = temp['Work Rate'].cat.codes
temp['n_players'] = 1
temp = temp.groupby('Club').agg({
    'Overall': 'mean',
    'Potential': 'mean',
    'Value': 'mean',
    'International Reputation': 'mean',
    'Work Rate': 'mean',
    'Wage': 'mean',
    'Contract_Duration': 'mean',
    'n_players': 'sum'
})

temp['Wage_per_player'] = temp.Wage / temp.n_players
temp.drop(columns=['Wage', 'n_players'], inplace=True)

temp.sample(5)
temp.sort_values(by=['Overall', 'Potential', 'Value', 'Wage_per_player', 'International Reputation', 'Contract_Duration'],
                ascending=[False, False, False, True, False, False]).head(10)
plt.figure(figsize=(10,8))
sns.lineplot(data=fifadata, y='Potential', x='Age', legend='full')
plt.title("Relationship between a player's potential and age")
plt.show()
plt.figure(figsize=(10,8))
sns.lineplot(data=fifadata, y='Value', x='Age', legend='full')
plt.title("Relationship between a player's value and age")
plt.ylabel("Value in million euros")
plt.show()
kwargs = {'alpha': 0.8, 'color': 'k', 'linestyle':'--'}

plt.figure(figsize=(16,8))
sns.lineplot(data=fifadata, y='SprintSpeed', x='Age')
sns.lineplot(data=fifadata, y='Stamina', x='Age')
plt.title("Estimation of a player's pace and age")
plt.ylabel("Amplitude")
plt.legend(['Sprint Speed', 'Stamina'])
plt.xticks(range(14, 46, 1))
plt.axvline(26, **kwargs)
plt.axvline(27, **kwargs)
plt.show()
cols = ['Age', 'Overall', 'Value', 'Wage', 'Special',
       'International Reputation', 'Weak Foot', 'Skill Moves',
       'Crossing', 'Finishing', 'HeadingAccuracy',
       'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',
       'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',
       'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping',
       'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',
       'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking',
       'StandingTackle', 'Potential']
from sklearn.preprocessing import MinMaxScaler

fifadata[cols] = MinMaxScaler().fit_transform(fifadata[cols])
# I could make out better from the weird color scheme
correlations = fifadata[cols].corr().round(3)
plt.figure(figsize=(24., 12.))
sns.heatmap(correlations, cmap='hot', vmin=-1, annot=correlations)
plt.title("Correlation matrix of all possible relevant features")
plt.show()
plt.figure(figsize=(16,8))
sns.lineplot(data=fifadata, y='Overall', x='Potential')
sns.lineplot(data=fifadata, y='Reactions', x='Potential')
sns.lineplot(data=fifadata, y='Composure', x='Potential')
sns.lineplot(data=fifadata, y='ShortPassing', x='Potential')
sns.lineplot(data=fifadata, y='Special', x='Potential')
plt.title("Estimation of what affects a player's potential")
plt.ylabel("Amplitude")
plt.legend(['Overall', 'Reactions', 'Composure', 'ShortPassing', 'Special'])
plt.show()
plt.figure(figsize=(16,8))
sns.lineplot(data=fifadata, y='International Reputation', x='Wage')
sns.lineplot(data=fifadata, y='Reactions', x='Wage')
sns.lineplot(data=fifadata, y='Overall', x='Wage')
sns.lineplot(data=fifadata, y='Potential', x='Wage')
sns.lineplot(data=fifadata, y='Composure', x='Wage')
plt.title("Estimation of what affects a player's wage")
plt.ylabel("Amplitude")
plt.legend(['International Reputation', 'Reactions', 'Overall', 'Potential', 'Composure'])
plt.show()
temp = fifadata[['Club', 'Age']].groupby('Club').agg([('Age_Min', 'min'), 
                                                     ('Age_Mean', 'mean'),
                                                     ('Age_Max', 'max')])
temp.columns = temp.columns.droplevel(0)
temp = temp.reset_index().sort_values(by='Age_Mean')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10))
sns.countplot(data=temp, x='Age_Min', ax=ax1)
ax1.set_title('Minimum age of players across clubs')

sns.countplot(data=temp, x='Age_Max', ax=ax2)
ax2.set_title('Maximum age of players across clubs')

sns.boxplot(data=temp, x='Age_Mean', ax=ax3)
ax3.set_title('Average age of players across clubs')

plt.tight_layout()
plt.show()
temp = fifadata[['Club', 'Age']]
temp['Ave_Age'] = 0

def filtfunc(x):
    club_sel = fifadata[fifadata['Club'] == x][['Age']]
    age_sel = club_sel[club_sel.Age == club_sel.Age.min()]
    return age_sel.count()

temp = temp.groupby('Club').agg({'Age': [('Min_Age', 'min')], 'Ave_Age': 'mean'})
temp.columns = temp.columns.droplevel()
temp = temp.reset_index()
temp['n_Min_Age'] = temp['Club'].apply(lambda x: filtfunc(x))

temp.reset_index().sort_values(by=['Min_Age', 'n_Min_Age'], ascending=[True, False])
temp.reset_index().sort_values(by=['n_Min_Age', 'Min_Age'], ascending=[False, True])
accidata1 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2005_to_2007.csv")
accidata2 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2009_to_2011.csv")
accidata3 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2012_to_2014.csv")
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
accidata = pd.concat([accidata1, accidata2, accidata3])
accidata.shape, accidata1.shape, accidata2.shape, accidata3.shape
del accidata1, accidata2, accidata3
gc.collect()

# The output column is categorical, not an integer. So typecast.
accidata[['Accident_Severity']] = accidata[['Accident_Severity']].astype('category')

# Paranthesis aren irritating, remove them!
# Also shorten the extremely elaborate column names
accidata = accidata.rename({'Location_Easting_OSGR': 'Easting_OSGR',
                            'Location_Northing_OSGR': 'Northing_OSGR',
                            'Local_Authority_(District)': 'Local_Authority_District',
                           'Local_Authority_(Highway)': 'Local_Authority_Highway',
                           'Did_Police_Officer_Attend_Scene_of_Accident': 'Officer_Attend',
                           'Pedestrian_Crossing-Human_Control': 'PC_Human_Control',
                           'Pedestrian_Crossing-Physical_Facilities': 'PC_Physical_Facilities',
                           'Special_Conditions_at_Site': 'Special_Conditions'}, axis=1)
accidata.info()
accidata.Junction_Detail.unique()
accidata.drop('Junction_Detail', axis=1, inplace=True)
accidata.sample(5)
temp = accidata[['Day_of_Week', 'Number_of_Casualties']].groupby('Day_of_Week').sum().reset_index()
temp = temp.sort_values('Number_of_Casualties', ascending=False)
temp
assert temp.Number_of_Casualties.sum() == accidata.Number_of_Casualties.sum()
accidata[['Day_of_Week', 'Speed_limit']].groupby('Day_of_Week').agg([np.max, np.min]).reset_index()
# First of all these are categorical values, so let me find out all the unique values
print(accidata.Light_Conditions.value_counts(), end="\n\n")
print(accidata.Weather_Conditions.value_counts(), end="\n\n")

print("Count of accidents, grouped by severity level")
print(accidata.Accident_Severity.value_counts())
temp = accidata[['Light_Conditions', 'Accident_Severity']].dropna()
temp = pd.get_dummies(data=temp, columns=['Accident_Severity'])
temp = temp.groupby('Light_Conditions').agg(np.count_nonzero).reset_index()
temp
cat_cols = ['Accident_Severity_'+str(i) for i in range(1, 4)]
temp[cat_cols] = temp[cat_cols].div(temp[cat_cols].sum(axis=1), axis=0)

plt.figure(figsize=(10, 8))
plt.plot(temp[cat_cols], temp.Light_Conditions, 'o', markersize=12)
plt.grid()
plt.legend(["Severity 1", "Severity 2", "Severity 3"])
plt.xlabel("Probabilty of accident")
plt.ylabel("Light Conditions")
plt.show()

temp
temp = accidata[['Weather_Conditions', 'Accident_Severity']].dropna()
temp = pd.get_dummies(data=temp, columns=['Accident_Severity'])
temp = temp.groupby('Weather_Conditions').agg(np.count_nonzero).reset_index()
temp
cat_cols = ['Accident_Severity_'+str(i) for i in range(1, 4)]
temp[cat_cols] = temp[cat_cols].div(temp[cat_cols].sum(axis=1), axis=0)

plt.figure(figsize=(10, 8))
plt.plot(temp[cat_cols], temp.Weather_Conditions, 'o', markersize=12)
plt.grid()
plt.legend(["Severity 1", "Severity 2", "Severity 3"])
plt.xlabel("Probabilty of accident")
plt.ylabel("Weather Conditions")
plt.show()

temp
# Take a look at the data
print(accidata.columns)

# These columns didn't show up, so manually selecting them
accidata[['Time',
       'Local_Authority_District', 'Local_Authority_Highway', '1st_Road_Class',
       '1st_Road_Number', 'Road_Type', 'Speed_limit',
       'Junction_Control', '2nd_Road_Class', '2nd_Road_Number',
       'PC_Human_Control']].sample(10)
accidata.Accident_Severity.value_counts()
latlong_OSGR = accidata[['Easting_OSGR', 'Northing_OSGR', 'Latitude', 'Longitude']].corr()
sns.heatmap(latlong_OSGR, annot=latlong_OSGR,
            vmin=-1, linewidths=0.8)
plt.figure(figsize=(20, 8))
sns.countplot(data=accidata, x='Police_Force', hue="Accident_Severity", palette='viridis')
plt.show()
# Proving that there are just too many dates to be useful
# Would be useful if I was doing a time-series analysis

accidata.Date.unique().shape, accidata.Time.unique().shape
accidata['Week_Of_Year'] = accidata[['Date']].apply(lambda x: pd.to_datetime(x, format="%d/%m/%Y"))
accidata['Week_Of_Year'] = accidata['Week_Of_Year'].apply(lambda x: x.weekofyear)

accidata['Quadrant'] = pd.to_datetime(accidata['Time'], format="%H:%M").dt.hour
accidata['Quadrant'] = pd.cut(accidata['Quadrant'], bins=[-1., 5., 11., 17., 23.], labels=['q1', 'q2', 'q3', 'q4'])
accidata['Quadrant'] = accidata['Quadrant'].astype('category')

accidata[['Date', 'Week_Of_Year', 'Year', 'Time', 'Quadrant']].sample(5)
temp = accidata[['Day_of_Week', 'Accident_Severity']].dropna()
temp = pd.get_dummies(data=temp, columns=['Accident_Severity'])
temp = temp.groupby('Day_of_Week').agg(np.count_nonzero).reset_index()
cat_cols = ['Accident_Severity_'+str(i) for i in range(1, 4)]
temp[cat_cols] = temp[cat_cols].div(temp[cat_cols].sum(axis=1), axis=0)

plt.figure(figsize=(10, 8))
plt.plot(temp[cat_cols], temp.Day_of_Week, 'o', markersize=12)
plt.grid()
plt.title('Severity of accidents grouped by road class')
plt.legend(["Severity 1", "Severity 2", "Severity 3"])
plt.xlabel("Probabilty of accident")
plt.ylabel("Day of week")
plt.show()

temp
accidata.Local_Authority_District.unique().shape, accidata.Local_Authority_Highway.unique().shape
accidata['1st_Road_Class'].unique(), accidata['2nd_Road_Class'].unique()
accidata['1st_Road_Number'].unique().shape, accidata['2nd_Road_Number'].unique().shape
temp = accidata[['1st_Road_Class', 'Accident_Severity']].dropna()
temp = pd.get_dummies(data=temp, columns=['Accident_Severity'])
temp = temp.groupby('1st_Road_Class').agg(np.count_nonzero).reset_index()
cat_cols = ['Accident_Severity_'+str(i) for i in range(1, 4)]
temp[cat_cols] = temp[cat_cols].div(temp[cat_cols].sum(axis=1), axis=0)

plt.figure(figsize=(10, 8))
plt.plot(temp[cat_cols], temp['1st_Road_Class'], 'o', markersize=12)
plt.grid()
plt.title('Severity of accidents grouped by road class')
plt.legend(["Severity 1", "Severity 2", "Severity 3"])
plt.xlabel("Probabilty of accident")
plt.ylabel("1st_Road_Class")
plt.show()

temp
temp = accidata[['2nd_Road_Class', 'Accident_Severity']].dropna()
temp = pd.get_dummies(data=temp, columns=['Accident_Severity'])
temp = temp.groupby('2nd_Road_Class').agg(np.count_nonzero).reset_index()
cat_cols = ['Accident_Severity_'+str(i) for i in range(1, 4)]
temp[cat_cols] = temp[cat_cols].div(temp[cat_cols].sum(axis=1), axis=0)

plt.figure(figsize=(10, 8))
plt.plot(temp[cat_cols], temp['2nd_Road_Class'], 'o', markersize=12)
plt.grid()
plt.title('Severity of accidents grouped by road class')
plt.legend(["Severity 1", "Severity 2", "Severity 3"])
plt.xlabel("Probabilty of accident")
plt.ylabel("2nd_Road_Class")
plt.show()

temp
accidata.Road_Type.unique(), accidata.Road_Surface_Conditions.unique()
temp = accidata[['Road_Type', 'Accident_Severity']].dropna()
temp = pd.get_dummies(data=temp, columns=['Accident_Severity'])
temp = temp.groupby('Road_Type').agg(np.count_nonzero).reset_index()
cat_cols = ['Accident_Severity_'+str(i) for i in range(1, 4)]
temp[cat_cols] = temp[cat_cols].div(temp[cat_cols].sum(axis=1), axis=0)

plt.figure(figsize=(10, 8))
plt.plot(temp[cat_cols], temp.Road_Type, 'o', markersize=12)
plt.grid()
plt.legend(["Severity 1", "Severity 2", "Severity 3"])
plt.xlabel("Probabilty of accident")
plt.ylabel("Road type")
plt.show()

temp
temp = accidata[['Road_Surface_Conditions', 'Accident_Severity']].dropna()
temp = pd.get_dummies(data=temp, columns=['Accident_Severity'])
temp = temp.groupby('Road_Surface_Conditions').agg(np.count_nonzero).reset_index()
cat_cols = ['Accident_Severity_'+str(i) for i in range(1, 4)]
temp[cat_cols] = temp[cat_cols].div(temp[cat_cols].sum(axis=1), axis=0)

plt.figure(figsize=(10, 8))
plt.plot(temp[cat_cols], temp.Road_Surface_Conditions, 'o', markersize=12)
plt.grid()
plt.legend(["Severity 1", "Severity 2", "Severity 3"])
plt.xlabel("Probabilty of accident")
plt.ylabel("Road Surface Conditions")
plt.show()

temp
accidata.Junction_Control.unique()
temp = accidata[['Junction_Control', 'Accident_Severity']].dropna()
temp = pd.get_dummies(data=temp, columns=['Accident_Severity'])
temp = temp.groupby('Junction_Control').agg(np.count_nonzero).reset_index()
cat_cols = ['Accident_Severity_'+str(i) for i in range(1, 4)]
temp[cat_cols] = temp[cat_cols].div(temp[cat_cols].sum(axis=1), axis=0)

plt.figure(figsize=(10, 8))
plt.plot(temp[cat_cols], temp.Junction_Control, 'o', markersize=12)
plt.grid()
plt.legend(["Severity 1", "Severity 2", "Severity 3"])
plt.xlabel("Probabilty of accident")
plt.ylabel("Junction control method")
plt.show()

temp
temp = accidata[['Speed_limit', 'Accident_Severity']].dropna()
temp = pd.get_dummies(data=temp, columns=['Accident_Severity'])
temp = temp.groupby('Speed_limit').agg(np.count_nonzero).reset_index()

temp
cat_cols = ['Accident_Severity_'+str(i) for i in range(1, 4)]
temp[cat_cols] = temp[cat_cols].div(temp[cat_cols].sum(axis=1), axis=0)

plt.figure(figsize=(10, 8))
plt.plot(temp[cat_cols], temp.Speed_limit, 'o', markersize=12)
plt.grid()

plt.legend(["Severity 1", "Severity 2", "Severity 3"])
plt.xlabel("Probabilty of accident")
plt.ylabel("Speed limit")
plt.show()

temp
print(accidata.PC_Physical_Facilities.value_counts(), end='\n\n')
print(accidata.PC_Human_Control.value_counts())
temp = accidata[['PC_Physical_Facilities', 'Accident_Severity']].dropna()
temp = pd.get_dummies(data=temp, columns=['Accident_Severity'])
temp = temp.groupby('PC_Physical_Facilities').agg(np.count_nonzero).reset_index()
temp
cat_cols = ['Accident_Severity_'+str(i) for i in range(1, 4)]
temp[cat_cols] = temp[cat_cols].div(temp[cat_cols].sum(axis=1), axis=0)

plt.figure(figsize=(10, 8))
plt.plot(temp[cat_cols], temp.PC_Physical_Facilities, 'o', markersize=12)
plt.grid()
plt.legend(["Severity 1", "Severity 2", "Severity 3"])
plt.xlabel("Probabilty of accident")
plt.ylabel("Pedestrian Crossing Physical Facilities")
plt.show()
temp = accidata[['PC_Human_Control', 'Accident_Severity']].dropna()
temp = pd.get_dummies(data=temp, columns=['Accident_Severity'])
temp = temp.groupby('PC_Human_Control').agg(np.count_nonzero).reset_index()
temp
cat_cols = ['Accident_Severity_'+str(i) for i in range(1, 4)]
temp[cat_cols] = temp[cat_cols].div(temp[cat_cols].sum(axis=1), axis=0)

plt.figure(figsize=(10, 8))
plt.plot(temp[cat_cols], temp.PC_Human_Control, 'o', markersize=12)
plt.grid()
plt.legend(["Severity 1", "Severity 2", "Severity 3"])
plt.xlabel("Probabilty of accident")
plt.ylabel("Pedestrian Crossing Physical Facilities")
plt.show()
accidata.Carriageway_Hazards.value_counts()
temp = accidata[['Carriageway_Hazards', 'Accident_Severity']].dropna()
temp = temp[temp.Carriageway_Hazards != 'None']

temp = pd.get_dummies(data=temp, columns=['Accident_Severity'])
temp = temp.groupby('Carriageway_Hazards').agg(np.count_nonzero).reset_index()

temp
cat_cols = ['Accident_Severity_'+str(i) for i in range(1, 4)]
temp[cat_cols] = temp[cat_cols].div(temp[cat_cols].sum(axis=1), axis=0)

plt.figure(figsize=(10, 8))
plt.plot(temp[cat_cols], temp.Carriageway_Hazards, 'o', markersize=12)
plt.grid()
plt.legend(["Severity 1", "Severity 2", "Severity 3"])
plt.xlabel("Probabilty of accident")
plt.ylabel("Carriageway related hazards")
plt.show()
accidata.Special_Conditions.value_counts()
temp = accidata[['Special_Conditions', 'Accident_Severity']].dropna()
temp = temp[temp.Special_Conditions != 'None']

temp = pd.get_dummies(data=temp, columns=['Accident_Severity'])
temp = temp.groupby('Special_Conditions').agg(np.count_nonzero).reset_index()
cat_cols = ['Accident_Severity_'+str(i) for i in range(1, 4)]
temp[cat_cols] = temp[cat_cols].div(temp[cat_cols].sum(axis=1), axis=0)

plt.figure(figsize=(10, 8))
plt.plot(temp[cat_cols], temp.Special_Conditions, 'o', markersize=12)
plt.grid()
plt.axvline(x=0.01)
plt.axvline(x=0.13)
plt.axvline(x=0.87)
plt.legend(["Severity 1", "Severity 2", "Severity 3"])
plt.xlabel("Probabilty of accident")
plt.ylabel("Existance of any special conditions at site")
plt.show()

temp
accidata.Urban_or_Rural_Area.value_counts()
temp = accidata[['Urban_or_Rural_Area', 'Accident_Severity']].dropna()

temp = pd.get_dummies(data=temp, columns=['Accident_Severity'])
temp = temp.groupby('Urban_or_Rural_Area').agg(np.count_nonzero).reset_index()
cat_cols = ['Accident_Severity_'+str(i) for i in range(1, 4)]
temp[cat_cols] = temp[cat_cols].div(temp[cat_cols].sum(axis=1), axis=0)

plt.figure(figsize=(10, 8))
plt.plot(temp[cat_cols], temp.Urban_or_Rural_Area, 'o', markersize=12)
plt.grid()
plt.legend(["Severity 1", "Severity 2", "Severity 3"])
plt.xlabel("Probabilty of accident")
plt.ylabel("Urban_or_Rural_Area")
plt.show()

temp
print(accidata.Officer_Attend.value_counts())
print(accidata.LSOA_of_Accident_Location.unique().shape)
temp = accidata[['Officer_Attend', 'Accident_Severity']].dropna()

temp = pd.get_dummies(data=temp, columns=['Accident_Severity'])
temp = temp.groupby('Officer_Attend').agg(np.count_nonzero).reset_index()
cat_cols = ['Accident_Severity_'+str(i) for i in range(1, 4)]
#temp[cat_cols] = temp[cat_cols].div(temp[cat_cols].sum(axis=1), axis=0)

plt.figure(figsize=(10, 8))
plt.plot(temp[cat_cols], temp.Officer_Attend, 'o', markersize=12)
plt.grid()
plt.legend(["Severity 1", "Severity 2", "Severity 3"])
plt.xlabel("Probabilty of accident")
plt.ylabel("Officer_Attend")
plt.show()

temp
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
categorical_cols = ['Day_of_Week', 'Year', '2nd_Road_Class', 'Road_Type',
                    'PC_Human_Control', 'PC_Physical_Facilities', 'Light_Conditions',
                    'Road_Surface_Conditions', 'Officer_Attend', 'Quadrant',
                   'Junction_Control', 'Weather_Conditions', 'Urban_or_Rural_Area']

drop_cols = ['Easting_OSGR', 'Northing_OSGR', 'LSOA_of_Accident_Location', '1st_Road_Number',
            '2nd_Road_Number', 'Local_Authority_District', 'Local_Authority_Highway', 'Date', 'Time',
            'Special_Conditions', 'Carriageway_Hazards', 'Accident_Index', '1st_Road_Class']

normalize_cols = ['Police_Force', 'Number_of_Vehicles', 'Number_of_Casualties']
scale_cols = ['Longitude', 'Latitude', 'Speed_limit']

output = "Accident_Severity"

accidata[normalize_cols] = accidata[normalize_cols].astype('float')
accidata[scale_cols] = accidata[scale_cols].astype('float')
accidata[output] = accidata[output].astype('category')
accidata[categorical_cols] = accidata[categorical_cols].astype('category')
accidata[normalize_cols] = StandardScaler().fit_transform(accidata[normalize_cols])
accidata[scale_cols] = MinMaxScaler().fit_transform(accidata[scale_cols])
accidata.drop(columns=drop_cols, inplace=True)
accidata.drop_duplicates(inplace=True, ignore_index=True)
types = pd.api.types.CategoricalDtype(categories=
                                      ['Missing', 'Automatic traffic signal', 'Giveway or uncontrolled', 'Stop Sign', 'Authorised person'])
accidata.Junction_Control = accidata.Junction_Control.astype(types)
accidata.Junction_Control.fillna('Missing', inplace=True)

accidata.dropna(subset=['PC_Human_Control', 'PC_Physical_Facilities', 'Road_Surface_Conditions',
                        'Officer_Attend', 'Weather_Conditions', 'Quadrant', 'Latitude', 'Longitude'],
                how='any', inplace=True)
for feature in categorical_cols:
    print(f"{feature}: {accidata[feature].unique()}", end='\n\n')
accidata['Road_Type'].cat.rename_categories({
    'Single carriageway': 'Single_Cway',
    'Dual carriageway': 'Dual_Cway',
    'One way street': 'One_Way',
    'Slip road': 'Slip_Road'
}, inplace=True)

accidata['PC_Human_Control'].cat.rename_categories({
    'None within 50 metres': 'None_LTE_50',
    'Control by other authorised person': 'Auth_Person',
    'Control by school crossing patrol': 'School_Person', 
}, inplace=True)

accidata['PC_Physical_Facilities'].cat.rename_categories({
    'No physical crossing within 50 meters': 'None_LTE_50',
    'Pedestrian phase at traffic signal junction': 'Pedes_TSJnc',
    'non-junction pedestrian crossing': 'Pedes_NJnc',
    'Zebra crossing': 'Zebra',
    'Central refuge': 'Central_Refuge',
    'Footbridge or subway': 'Bridge_Subway'
}, inplace=True)

accidata['Light_Conditions'].cat.rename_categories({
    'Daylight: Street light present': 'Daylight',
    'Darkness: Street lights present and lit': 'Dark_Light_Lit',
    'Darkeness: No street lighting': 'Dark_No_Light',
    'Darkness: Street lighting unknown': 'Dark_Light_NA',
    'Darkness: Street lights present but unlit': 'Dark_Light_Unlit'
}, inplace=True)

accidata['Road_Surface_Conditions'].cat.rename_categories({
    'Flood (Over 3cm of water)': 'Flood'
}, inplace=True)

accidata['Junction_Control'].cat.rename_categories({
    'Giveway or uncontrolled': 'Giveway',
    'Automatic traffic signal': 'Automatic',
    'Stop Sign': 'Stop',
    'Authorised person': 'Auth_Person'
}, inplace=True)

accidata['Weather_Conditions'].cat.rename_categories({
    'Fine without high winds': 'Fine_NHW',
    'Fine with high winds': 'Fine_HW',
    'Raining without high winds': 'Rain_NHW',
    'Raining with high winds': 'Rain_HW',
    'Snowing without high winds': 'Snow_NHW',
    'Snowing with high winds': 'Snow_HW',
    'Fog or mist': 'Fog_Mist'
}, inplace=True)
def getCompleteDataframe(data):
    y = data[output]
    encoder = OneHotEncoder()
    temp = data[categorical_cols]

    temp = pd.DataFrame(encoder.fit_transform(temp).toarray())
    temp.columns = encoder.get_feature_names(categorical_cols)

    data.drop(columns=categorical_cols + [output], inplace=True)
    data = pd.concat([data.reset_index(), temp], axis=1)
    
    return data, y
X, y = getCompleteDataframe(accidata)
X = X.drop(columns=['index']).reset_index()
X.isna().sum().sort_values(ascending=False)
X.columns
# Very few rows belong to this category
X.drop(['Urban_or_Rural_Area_3'], axis=1, inplace=True)
train_X, test_X, train_Y, test_Y = train_test_split(X, y, 
                                                    test_size=0.1, random_state=43, shuffle=True)

train_X.shape, test_X.shape, train_Y.shape, test_Y.shape
from sklearn.linear_model import LogisticRegressionCV
model = LogisticRegressionCV(cv=5, n_jobs=-1, verbose=1)
model.fit(train_X, train_Y)
model.score(test_X, test_Y)
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model, test_X, test_Y)
plt.show()
train_Y.value_counts(), train_Y.shape
balanced_model = LogisticRegressionCV(cv=5, class_weight="balanced",
                                      n_jobs=-1, multi_class='multinomial')

balanced_model.fit(train_X, train_Y)
print(f"Accuracy score: {balanced_model.score(test_X, test_Y)}")

plot_confusion_matrix(balanced_model, test_X, test_Y)
plt.show()