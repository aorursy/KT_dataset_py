import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# Lets load the data and start having a look
data = pd.read_csv("../input/skiResort.csv", encoding = "ISO-8859-1")
data.head()

# Lets have a look at the feature name
list(data)
# For those features that are types of lifts, we will just replace the NaNs with 0 
# to indicate there are no lifts of this type at these ski resorts.

# Here is a list of all the lift types.
listOfLifts = ['Funicular',
               'Circulating ropeway/gondola lift',
               'Chairlift', 
               'T-bar lift/platter/button lift ', 
               'Sunkid Moving Carpet', 
               'Aerial tramway/reversible ropeway', 
               'Rope tow/beginner lift', 
               'People mover', 
               'Combined installation (gondola and chair)', 
               'Cog railway', 
               'Helicopter for Heli-skiing', 
               'Snow caterpillars for Cat-skiing']

data[listOfLifts] = data[listOfLifts].fillna(0)
data.head()
# Collect the total number of missing values for each feature.
missing = data.isnull().sum()
percent = (data.isnull().sum()/data.isnull().count()) * 100
missing_data = pd.concat([missing, percent], axis=1, keys=['Total', 'Percent'])

# Show the total missing values.
plt.figure(figsize=(20,10))
ax = sns.barplot(y=missing_data.index, x='Percent' ,data=missing_data, palette="gist_heat", orient='h')
plt.xlabel('% Missing')
plt.title('Total missing values by feature', loc='Center', fontsize=14)
plt.show()

totalMissing = data.isnull().sum().sum() / (np.shape(data)[0]*np.shape(data)[1]) *100
print ("Total missing value consist of {0:5.2f} % of the total data".format(totalMissing))
# For the features that are ratings, the missing values indicate a site visit has not been completed by 
# skiresort.info staff. So we will replace these values with the average value for each rating according 
# to each country. If there are no values for an entire country, the value will be an overall average 
# rating of 2.5.

# List the ratings features.
listOfRatings = ['Ski resort size ', 
                 'Slope offering, variety of runs ', 
                 'Lifts and cable cars ', 
                 'Snow reliability ', 
                 'Slope preparation ', 
                 'Access, on-site parking ', 
                 'Orientation (trail map, information boards, sign-postings) ', 
                 'Cleanliness and hygiene ', 
                 'Environmentally friendly ski operation ', 
                 'Friendliness of staff ', 
                 'Mountain restaurants, ski huts, gastronomy ', 
                 'Après-ski ', 
                 'Accommodation offering directly at the slopes and lifts ', 
                 'Families and children ', 
                 'Beginners ', 
                 'Advanced skiers, freeriders ', 
                 'Snow parks ',
                 'Cross-country skiing and trails ']

# Fill the missing ratings with the average rating for each feature according to the country 
# location. Fill any further missing averages with an average rating of 2.5
averageRatings = data.groupby('Country')[listOfRatings].transform('mean').fillna(2.5)
data[listOfRatings] = data[listOfRatings].fillna(averageRatings)

data.head()

# There are still some features that contain missing values.
data.isnull().sum().head(10)
# We will fill in State/Province missing values with a non descript label "Not Specified". 
data['State/Province'] = data['State/Province'].fillna('Not Specified')

# Next, fill in the missing values for the slope difficulty. These missing values means that 
# skiresort.info could not obtain the information about the distance of runs in each category.
# So we will replace the values with, 0
difficulty = ['Easy', 'Intermediate ', 'Difficult']
data[difficulty]= data[difficulty].fillna(0)

data.isnull().sum()
# Group the ski resorts by country
skiResort_Country = data.groupby('Country')[['Country']].count()
skiResort_Country = skiResort_Country.rename(columns={"Country": "Count"})
skiResort_Country = skiResort_Country.sort_values(by='Count', ascending=False)
# Lets just take a look at the top 15 countries
skiResort_Country_short = skiResort_Country.head(15)

plt.figure(figsize=(20,6))
sns.barplot(y=skiResort_Country_short.index, x='Count' ,data=skiResort_Country_short, palette="gist_heat", orient='h')
plt.xlabel('Count')
plt.ylabel('Country')
plt.title('Total number of Resorts by Country', loc='Center', fontsize=14)
plt.show()


# Lets group the data by continent
skiResort_Continent = data.groupby('Continent')[['Continent']].count()
skiResort_Continent = skiResort_Continent.rename(columns={"Continent": "Count"})
skiResort_Continent = skiResort_Continent.sort_values(by='Count', ascending=False)

plt.figure(figsize=(20,6))
sns.barplot(y=skiResort_Continent.index, x='Count' ,data=skiResort_Continent, palette="gist_heat", orient='h')
plt.xlabel('Count')
plt.ylabel('Continent')
plt.title('Total number of Resorts by Continent', loc='Center', fontsize=14)
plt.show()
# Lets have a look the actual numbers by country.
skiResort_Country['Percent'] = skiResort_Country/skiResort_Country.sum()*100
skiResort_Country.head(10)
# Lets have a look the actual numbers by continent.
skiResort_Continent['Percent'] = skiResort_Continent/skiResort_Continent.sum()*100
skiResort_Continent
Europe = data[data['Continent'] == 'Europe']
Europe.head()
Europe['Ski resort size '][Europe['Ski resort size '] <= 1] = 1
Europe['Ski resort size '][(Europe['Ski resort size '] > 1) & (Europe['Ski resort size '] <= 2)] = 2
Europe['Ski resort size '][(Europe['Ski resort size '] > 2) & (Europe['Ski resort size '] <= 3)] = 3
Europe['Ski resort size '][(Europe['Ski resort size '] > 3) & (Europe['Ski resort size '] <= 4)] = 4
Europe['Ski resort size '][Europe['Ski resort size '] > 4] = 5

plt.figure(figsize=(15,10))
sns.countplot(x='Ski resort size ',data = Europe)
plt.title('Count of ratings values',fontsize=15)
plt.show()
# Sort the data by altitude.
altitude = data.sort_values(by='Altitude', ascending=False)#.head(15)
sample = altitude.head(15)

# Display the top 15 countries with high elevation ski resorts
plt.figure(figsize=(20,10))
sns.barplot(y="Country", x="Altitude" ,data=sample, palette="gist_heat", orient='h')
plt.xlabel('Altitude')
plt.ylabel('Country')
plt.title('Highest altitude resort by country', loc='Center', fontsize=14)
plt.show()
# Get the slope difficulty columns and combine them into one data series.
terrain = [data['Easy'], data['Intermediate '], data['Difficult']]
totalSkiableTerrain = pd.concat(terrain)

# Display the distribution of each ability level and the combination of all three levels as one
fig, axes = plt.subplots(4, figsize=(20, 7), sharex=True)
sns.distplot(data['Easy'],color='skyblue',ax=axes[0])
axes[0].set_ylim(0,0.005)
sns.distplot(data['Intermediate '],color='red',ax=axes[1])
axes[1].set_ylim(0,0.005)
sns.distplot(data['Difficult'],color='teal',ax=axes[2])
axes[2].set_ylim(0,0.005)
sns.distplot(totalSkiableTerrain,color='magenta',ax=axes[3])
axes[3].set_xlabel('All terrain')
axes[3].set_ylim(0,0.005)
axes[0].set_title('Distribution of slope difficulty')

plt.show()
# Lets find out what resort has the largest amount of skiable terrain for beginners.
data[data['Easy'] == data['Easy'].max()]
#Lets find out what resort has the largest amount of skiable terrain for intermediate skiers.
data[data['Intermediate '] == data['Intermediate '].max()]
#Lets find out what resort has the largest amount of skiable terrain for advanced skiers.
data[data['Difficult'] == data['Difficult'].max()]
# Get the ratio of skiable terrain that is less that 20 km in each ability level.
Easy20 = data[data['Easy'] <= 20]['Easy'].sum()/data['Easy'].sum()
Intermediate20 = data[data['Intermediate '] <= 20]['Intermediate '].sum()/data['Intermediate '].sum()
Difficult20 = data[data['Difficult'] <= 20]['Difficult'].sum()/data['Difficult'].sum()

# Display the results in terms of percentage.
print ("Easy terrain < 20km: {0:5.2f} %".format(Easy20*100))
print ("Intermediate terrain < 20km: {0:5.2f}  %".format(Intermediate20*100))
print ("Difficult terrain < 20km: {0:5.2f}  %".format(Difficult20*100))