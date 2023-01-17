import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



sns.set(style="ticks", context="talk")
alcdata = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/alcoholism/student-mat.csv")

fifadata = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/fifa18/data.csv")

accidata1 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2005_to_2007.csv")

accidata2 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2009_to_2011.csv")

accidata3 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2012_to_2014.csv")
alcdata.info()

alcdata.columns
grades = alcdata[ ['G1', 'G2', 'G3']]

display( grades )
grade = pd.DataFrame( grades.sum(axis = 1)/60 )
alcdata = alcdata.drop( ['G1', 'G2', 'G3'], axis = 1 )

alcdata['grade'] = grade
con_variables = ['age', 'absences', 'grade']

alcdata_con = alcdata[ con_variables ]

alcdata_corr = alcdata_con.corr()

fig, ax = plt.subplots(figsize=(11, 9))

sns.set( font_scale = 1 )

cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

sns.heatmap( alcdata_corr, cmap=cmap, annot=True, fmt='0.2f', square=True, linewidths = 2, vmin = -0.7, vmax = 0.7 )
cat_variables = [ var for var in list(alcdata) if var not in con_variables ]

display( cat_variables )
for var in cat_variables:

    fig, axs = plt.subplots( ncols = 2, figsize = [10,5] )

    sns.boxplot( data = alcdata, x = alcdata[str(var)], y = alcdata.grade, ax = axs[0])

    sns.pointplot( data = alcdata, x = alcdata[str(var)], y = alcdata.grade, ax = axs[1] )

    plt.ylim(0, 1)
alcdata.loc[ alcdata.Pstatus == 'A', 'Pstatus'] = 0

alcdata.loc[ alcdata.Pstatus == 'T', 'Pstatus'] = 1

alcdata.Pstatus.value_counts()
alcdata.Pstatus.value_counts()

fig, axs = plt.subplots( figsize = [10,5] )

sns.distplot( alcdata.famrel )
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 



for var in con_variables:

    fig, axs = plt.subplots( figsize = [10,5] )

    sns.distplot( alcdata[str(var)] )

    

    

absences_no_skew = [ np.log( val ) for val in alcdata.absences if val > 0]

fig, axs = plt.subplots( figsize = [10,5] )

sns.distplot( absences_no_skew )
def convert(Value):

    val = str(Value).replace('€', '')

    if 'M' in val:

        val = float(val.replace('M', ''))*1000000

    elif 'K' in str(Value):

        val = float(val.replace('K', ''))*1000

    return float(val)



fifadata['Value'] = fifadata['Value'].apply(lambda x: convert(x))

fifadata['Wage'] = fifadata['Wage'].apply(lambda x: convert(x))

fifadata['Release Clause'] = fifadata['Release Clause'].apply(lambda x: convert(x))



club_rc = pd.DataFrame( fifadata.groupby('Club')['Release Clause'].sum() )

# display( club_rc )



club_val = pd.DataFrame( fifadata.groupby('Club')['Value'].sum() )

# display( club_val )



club_wage = pd.DataFrame( fifadata.groupby('Club')['Wage'].sum() )

# display( club_wage )

# display( fifadata['Release Clause'] )
# Release Clause : Value of the assets( players ) of the club

# Wage : Amount paid to the player

# Value : Value of the player in the market

# We can say that if the release clause of the player is lesser than the value of the player,

# the club is performing bad economically

# This can be subtracted from the value



# economy = club_rc - club_wage + club_value - club_rc

# economy = club_value - club_wage



club_rc['wage'] = club_wage

club_rc['value'] = club_val



club_rc['economy'] = club_rc['value'] - club_rc['wage']

# display( club_rc.economy)

club_rc = club_rc.sort_values(by=['economy'], ascending = False)



club_rc.head()
# display( fifadata.Age.value_counts() )

# display( fifadata.Potential.value_counts() )



vals = ['Age', 'Potential', 'Value', 'Acceleration', 'SprintSpeed', 'Agility']

plt.figure( figsize=(11, 9) )

sns.scatterplot( x = fifadata.Age, y = fifadata.Potential )



fig, ax = plt.subplots(figsize=(11, 9))

sns.set( font_scale = 1 )

cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

sns.heatmap( fifadata[vals].corr(), cmap=cmap, annot=True, fmt='0.2f', square=True, linewidths = 2, vmin = -1, vmax = 1 )
fifadata.info()

# fifadata['Skill Moves'].value_counts()
skills = ['Potential', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Interceptions', 'StandingTackle', 'SlidingTackle']

traits = ['Potential', 'Composure', 'Vision', 'Aggression', 'Strength', 'Stamina', 'Jumping', 'ShotPower', 'Balance', 'Reactions', 'Agility', 'SprintSpeed', 'Acceleration']

plt.figure( figsize=(20, 10) )

fig, ax = plt.subplots(figsize=(11, 9))

sns.set( font_scale = 1 )

cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

sns.heatmap( fifadata[skills].corr(), cmap=cmap, annot=True, fmt='0.2f', square=True, linewidths = 2, vmin = -1, vmax = 1 )



plt.figure( figsize=(20, 10) )

fig, ax = plt.subplots(figsize=(11, 9))

sns.set( font_scale = 1 )

cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

sns.heatmap( fifadata[traits].corr(), cmap=cmap, annot=True, fmt='0.2f', square=True, linewidths = 2, vmin = -1, vmax = 1 )
# Too similar as well as not applicable to evreyone as it is about Goal Keeping

fifadata_1 = fifadata.drop( ['GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes'], axis = 1)

fig, ax = plt.subplots(figsize=(15, 15))

sns.set( font_scale = 1 )

cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

sns.heatmap( fifadata_1.corr(), cmap=cmap, annot=False, fmt='0.2f', square=True, linewidths = 0, vmin = -1, vmax = 1 )
club_age = pd.DataFrame( fifadata.groupby('Club').Age.describe() )

# display(club_age)

club_age = club_age.sort_values( by = ['75%'] )

display(club_age)

# display( club_age['mean'] )

fig, ax = plt.subplots(figsize=(11, 9))

sns.distplot( club_age['mean'])
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 

datasets = [accidata1, accidata2, accidata3]

datasets_sizes = [ x.shape[0] for x in datasets ]

# display( datasets_sizes)

accidata = pd.concat( [accidata1, accidata2, accidata3] )



if accidata.shape[0] == sum( datasets_sizes ):

    print("Datasets merged successfully")
accidata.info()
# str = accidata.dtypes[ accidata.dtypes == "object" ].index

accidata.isna().sum()



# accidata.Junction_Control.value_counts()
drop_list = ['Junction_Detail', 'Junction_Control', 'LSOA_of_Accident_Location']

accidata = accidata.drop( drop_list , axis = 1)
display( accidata.iloc[0].Date, accidata.iloc[0].Day_of_Week )
dow = { 1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}

accidata_dow_count = pd.DataFrame( accidata.groupby( ['Day_of_Week'] ).Number_of_Casualties.sum() )

# display( accidata_dow_count )

casualties_dow = { dow[x]: y.item() for x, y in accidata_dow_count.iterrows() }

display( casualties_dow )
#enter code/answer in this cell. You can add more code/markdown cells below for your answer.

accidata_dow = accidata.groupby( ['Day_of_Week'] )

display( accidata_dow.Speed_limit.describe()["min"] )

display( accidata_dow.Speed_limit.describe()["max"] )
plt.figure( figsize=(20, 9) )

sns.set( font_scale = 1)

sns.countplot( accidata.Light_Conditions, hue=accidata.Accident_Severity )
plt.figure( figsize=(20, 9) )

sns.countplot( accidata.Weather_Conditions, hue=accidata.Accident_Severity )
lol = pd.get_dummies( accidata.Accident_Severity, prefix = "Severity")

lol.head()



copy = accidata

copy['severity_1'] = lol.Severity_1

copy['severity_2'] = lol.Severity_2

copy['severity_3'] = lol.Severity_3

# display( copy )

# copy = copy.drop( 'Accident_Severity', axis = 1)

# display( copy )
fig, ax = plt.subplots(figsize=(20, 20))

sns.set( font_scale = 1 )

cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

sns.heatmap( copy.corr(), cmap=cmap, annot=True, fmt='0.2f', square=True, linewidths = 2, vmin = -1.0, vmax = 1.0 )
copy = copy.drop( ['Longitude', 'Latitude', 'Accident_Index', 'Local_Authority_(District)'], axis =1 )

# display( copy )
copy.info()
copy['Pedestrian_Crossing-Human_Control'].value_counts()

copy['Pedestrian_Crossing-Physical_Facilities'].value_counts()

copy['Did_Police_Officer_Attend_Scene_of_Accident'].value_counts()

# copy['Carriageway_Hazards'].value_counts()

# copy['Special_Conditions_at_Site'].value_counts()



drop_list = ['Date', 'Urban_or_Rural_Area', 'Special_Conditions_at_Site', 'Carriageway_Hazards' ]

copy = copy.drop( drop_list, axis = 1)

copy.info()
copy['Local_Authority_(Highway)'].value_counts()

copy['Road_Type'].value_counts()
copy.info()

copy_2 = copy

list = ['Time', 'Did_Police_Officer_Attend_Scene_of_Accident', 'Pedestrian_Crossing-Human_Control', 'Pedestrian_Crossing-Physical_Facilities', ]

copy_2 = copy_2.drop( list, axis= 1)

copy_2.info()
light_dummy = pd.get_dummies( copy_2.Light_Conditions, prefix="Light")



copy_2 = pd.concat( [copy_2, light_dummy], axis = 1 )



copy_2 = copy_2.drop( 'Light_Conditions', axis = 1)

copy_2.info()
weather_dummy = pd.get_dummies( copy_2.Weather_Conditions, prefix="Weather")



copy_2 = pd.concat( [copy_2, weather_dummy], axis = 1 )



copy_2

copy_2 = copy_2.drop( 'Weather_Conditions', axis = 1)

copy_2.info()
copy_2 = pd.concat( [copy_2, pd.get_dummies( copy_2.Road_Surface_Conditions, prefix="Road")], axis = 1 )



copy_2

copy_2 = copy_2.drop( 'Road_Surface_Conditions', axis = 1)

copy_2.info()
# copy_2 = pd.concat( [copy_2, pd.get_dummies( copy_2['Local_Authority_(Highway)'], prefix="High")], axis = 1 )



copy_2

copy_2 = copy_2.drop( 'Local_Authority_(Highway)', axis = 1)

copy_2.info()
copy_2 = pd.concat( [copy_2, pd.get_dummies( copy_2['Road_Type'], prefix="Road_type")], axis = 1 )



copy_2

copy_2 = copy_2.drop( 'Road_Type', axis = 1)
copy_2.info()
copy_2 = copy_2.dropna()



y = copy_2[ ['severity_1', 'severity_2', 'severity_3'] ]



copy_2 = copy_2.drop( ['severity_1', 'severity_2', 'severity_3'], axis = 1)
y.info()

copy_2.info()
copy_2.shape

y = copy_2.Accident_Severity

y.shape

X_main = copy_2

y_main = y

X_main = X_main.drop( 'Accident_Severity', axis = 1)

print( X_train.shape, y_train.shape)
# from sklearn.linear_model import LogisticRegression

# from sklearn.model_selection import cross_val_score



# score = cross_val_score( LogisticRegression(max_iter = 500, verbose = True), X_main, y_main, cv = 5)