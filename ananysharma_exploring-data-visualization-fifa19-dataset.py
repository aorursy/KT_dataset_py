import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/fifa19/data.csv')
def country(x):
    return data[data['Nationality'] == x][['Name','Overall','Potential','Position']]


# let's check the Indian Players 
country('India')
def club(x):
    return data[data['Club'] == x][['Name','Jersey Number','Position','Overall','Nationality','Age','Wage',
                                    'Value','Contract Valid Until']]

club('Manchester United')
# df1 = df.drop(labels = "ID",inplace = False ,axis =1)
# df1.shape[:]
# df1.drop(labels = ["Photo","Flag",],axis=1,inplace = True)
data['ShortPassing'].fillna(data['ShortPassing'].mean(), inplace = True)
data['Volleys'].fillna(data['Volleys'].mean(), inplace = True)
data['Dribbling'].fillna(data['Dribbling'].mean(), inplace = True)
data['Curve'].fillna(data['Curve'].mean(), inplace = True)
data['FKAccuracy'].fillna(data['FKAccuracy'], inplace = True)
data['LongPassing'].fillna(data['LongPassing'].mean(), inplace = True)
data['BallControl'].fillna(data['BallControl'].mean(), inplace = True)
data['HeadingAccuracy'].fillna(data['HeadingAccuracy'].mean(), inplace = True)
data['Finishing'].fillna(data['Finishing'].mean(), inplace = True)
data['Crossing'].fillna(data['Crossing'].mean(), inplace = True)
data['Weight'].fillna('200lbs', inplace = True)
data['Contract Valid Until'].fillna(2019, inplace = True)
data['Height'].fillna("5'11", inplace = True)
data['Loaned From'].fillna('None', inplace = True)
data['Joined'].fillna('Jul 1, 2018', inplace = True)
data['Jersey Number'].fillna(8, inplace = True)
data['Body Type'].fillna('Normal', inplace = True)
data['Position'].fillna('ST', inplace = True)
data['Club'].fillna('No Club', inplace = True)
data['Work Rate'].fillna('Medium/ Medium', inplace = True)
data['Skill Moves'].fillna(data['Skill Moves'].median(), inplace = True)
data['Weak Foot'].fillna(3, inplace = True)
data['Preferred Foot'].fillna('Right', inplace = True)
data['International Reputation'].fillna(1, inplace = True)
data['Wage'].fillna('€200K', inplace = True)
data.fillna(0, inplace = True)
def defending(data):
    return int(round((data[['Marking', 'StandingTackle', 
                               'SlidingTackle']].mean()).mean()))

def general(data):
    return int(round((data[['HeadingAccuracy', 'Dribbling', 'Curve', 
                               'BallControl']].mean()).mean()))

def mental(data):
    return int(round((data[['Aggression', 'Interceptions', 'Positioning', 
                               'Vision','Composure']].mean()).mean()))

def passing(data):
    return int(round((data[['Crossing', 'ShortPassing', 
                               'LongPassing']].mean()).mean()))

def mobility(data):
    return int(round((data[['Acceleration', 'SprintSpeed', 
                               'Agility','Reactions']].mean()).mean()))
def power(data):
    return int(round((data[['Balance', 'Jumping', 'Stamina', 
                               'Strength']].mean()).mean()))

def rating(data):
    return int(round((data[['Potential', 'Overall']].mean()).mean()))

def shooting(data):
    return int(round((data[['Finishing', 'Volleys', 'FKAccuracy', 
                               'ShotPower','LongShots', 'Penalties']].mean()).mean()))
data.rename(columns={'Club Logo':'Club_Logo'}, inplace=True)

# adding these categories to the data

data['Defending'] = data.apply(defending, axis = 1)
data['General'] = data.apply(general, axis = 1)
data['Mental'] = data.apply(mental, axis = 1)
data['Passing'] = data.apply(passing, axis = 1)
data['Mobility'] = data.apply(mobility, axis = 1)
data['Power'] = data.apply(power, axis = 1)
data['Rating'] = data.apply(rating, axis = 1)
data['Shooting'] = data.apply(shooting, axis = 1)
data.head()
players = data[['Name','Defending','General','Mental','Passing',
                'Mobility','Power','Rating','Shooting','Flag','Age',
                'Nationality', 'Photo', 'Club_Logo', 'Club']]

data["International Reputation"]
plt.rcParams['figure.figsize'] = (10,10)
sns.countplot(data["Preferred Foot"],palette = "YlOrRd")
plt.title("Most Preferred Foot",fontsize = 20)
plt.show()
labels = ['1', '2', '3', '4', '5']
sizes = data['International Reputation'].value_counts()
colors = plt.cm.copper(np.linspace(0, 1, 5))
explode = [0.1, 0.1, 0.2, 0.5, 0.9]
print(sizes)
plt.rcParams['figure.figsize'] = (9, 9)
plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True)
plt.title('International Repuatation for the Football Players', fontsize = 20)
plt.legend()
plt.show()

data["Weak Foot"]

labels = ['5', '4', '3', '2', '1'] 
sizes = data["Weak Foot"].value_counts()
colors = plt.cm.copper(np.linspace(0, 1, 5))
explode = [0, 0, 0, 0, 0.1]

# plt.rcParams["figure.figsize"] = (10,10)
plt.pie(sizes,labels = labels,colors = colors,explode = explode,shadow = True,startangle = 90)
plt.title("Weak Foot Analysis")
plt.legend()
plt.show()
data.head()
data["Position"]
plt.rcParams["figure.figsize"] = (20,8)
sns.countplot(data['Position'],palette = 'bone')
plt.title("Position Analysis",fontsize = 12)
plt.show()
def clean_weight(x):
    outp = x.replace('lbs','')
    return float(outp)
data["Weight"]
data['Weight'] = data["Weight"].apply(lambda x: clean_weight(x))
data["Weight"]
data["Wage"]
def wage_clean(x):
    out = x.replace('€','')
    if 'M' in out:
        out = float(out.replace("M",""))*1000000
    elif 'K' in out:
        out = float(out.replace('K',''))*1000
    return out    
data["Wage"] = data["Wage"].apply(lambda x: wage_clean(x))
data["Wage"]
plt.rcParams['figure.figsize'] = (15,8)
sns.distplot(data["Wage"],color = 'blue')
plt.xlabel(xlabel = "Wage range",fontsize = 10)
plt.ylabel(ylabel = "Count distribution",fontsize = 10)
plt.title("Distribution PLot",fontsize =20)
plt.legend()
plt.xticks(rotation = 90)
plt.show()
plt.figure(figsize = (10, 8))
ax = sns.countplot(x = 'Skill Moves', data = data, palette = 'pastel')
ax.set_title(label = 'Count of players on Basis of their skill moves', fontsize = 20)
ax.set_xlabel(xlabel = 'Number of Skill Moves', fontsize = 16)
ax.set_ylabel(ylabel = 'Count', fontsize = 16)
plt.show()
plt.rcParams['figure.figsize'] = (10,10)
sns.countplot(data["Height"],palette = 'dark')
plt.title("Count on the basis of height",fontsize = 20)
plt.show
plt.rcParams['figure.figsize'] = (15,8)
sns.distplot(data["Weight"],color = 'blue')
plt.xlabel(xlabel = "Weight range",fontsize = 10)
plt.ylabel(ylabel = "Count distribution",fontsize = 10)
plt.title("Distribution PLot",fontsize =20)
plt.legend()
plt.xticks(rotation = 90)
plt.show()
plt.style.use('dark_background')
data['Nationality'].value_counts().head(80).plot.bar(color = 'orange', figsize = (20, 7))
plt.title('Different Nations Participating in FIFA 2019', fontsize = 30, fontweight = 20)
plt.xlabel('Name of The Country')
plt.ylabel('count')
plt.show()
data["Nationality"]
plt.rcParams['figure.figsize'] = (20,10)
plt.style.use("dark_background")
sns.countplot((data["Nationality"]),palette = "dark")
plt.title("Nations Participation")
plt.xticks(rotation = 90)
plt.show()
sns.set(palette = 'dark',style = "dark")
plt.figure(figsize = (15,10))
sns.distplot(data['Age'],bins = 40,kde = False,color = 'g')
ax.set_xlabel("Age Measure",fontsize = 12)
ax.set_ylabel("Distribution",fontsize = 12)
ax.set_title("Age Histogram",fontsize = 20)
plt.show()

selected_columns = ['Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value',
                    'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot',
                    'Skill Moves', 'Work Rate', 'Body Type', 'Position', 'Height', 'Weight',
                    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
                    'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
                    'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
                    'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
                    'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
                    'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
                    'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']

data_selected = pd.DataFrame(data, columns = selected_columns)
data_selected.columns
data_selected.head()
plt.rcParams['figure.figsize'] = (20,10)
sns.boxenplot(data["Age"],data["Overall"],hue = data["Preferred Foot"],palette = "twilight")
ax.set_xlabel("Age",fontsize = 12)
ax.set_ylabel("Overall",fontsize = 12)
ax.set_title("BOX PLOT",fontsize= 20)
plt.show()
plt.scatter(data['Overall'], data['International Reputation'], s = data['Age']*10, c = 'pink')
plt.xlabel('Overall Ratings', fontsize = 20)
plt.ylabel('International Reputation', fontsize = 20)
plt.title('Ratings vs Reputation', fontweight = 20, fontsize = 20)
#plt.legend('Age', loc = 'upper left')
plt.show()
data_selected.sample(10)
data_selected[:5]
import matplotlib.cm as cm
plt.rcParams['figure.figsize'] = (20,10)
# colors = cm.rainbow(np.linspace(0, 1, len(data)))
plt.scatter(data["Overall"],data["Potential"])
plt.xlabel("Wage")
plt.ylabel("Release Clause")
plt.title("Scatter plot")
plt.show()
plt.rcParams["figure.figsize"] = (20,10)
sns.heatmap(data_selected[:].corr(),annot = True)
plt.title("Histogram of data",fontsize= 15)
plt.show()
data.iloc[data.groupby(data['Position'])['Overall'].idxmax()][['Position', 'Name', 'Age', 'Club', 'Nationality','Overall']]

data.iloc[data.groupby(data['Position'])['Overall'].idxmin()][['Position', 'Name', 'Age', 'Club', 'Nationality',"Overall"]]

# best players per each position with their age, club, and nationality based on their overall scores

data.iloc[data.groupby(data['Position'])['Potential'].idxmax()][['Position', 'Name', 'Age', 'Club', 'Nationality']]

some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia')
data_countries = data.loc[data['Nationality'].isin(some_countries) & data['Weight']]
data_countries

plt.rcParams['figure.figsize'] = (15, 7)
ax = sns.violinplot(x = data_countries['Nationality'], y = data_countries['Weight'], palette = 'Reds')
ax.set_xlabel(xlabel = 'Countries', fontsize = 9)
ax.set_ylabel(ylabel = 'Weight in lbs', fontsize = 9)
ax.set_title(label = 'Distribution of Weight of players from different countries', fontsize = 20)
plt.show()
plt.rcParams['figure.figsize'] = (15, 7)
ax = sns.boxplot(x = data_countries['Nationality'], y = data_countries['Weight'], palette = 'Reds')
ax.set_xlabel(xlabel = 'Countries', fontsize = 9)
ax.set_ylabel(ylabel = 'Weight in lbs', fontsize = 9)
ax.set_title(label = 'Distribution of Weight of players from different countries', fontsize = 20)
plt.show()
plt.rcParams['figure.figsize'] = (15, 7)
ax = sns.barplot(x = data_countries['Nationality'], y = data_countries['Weight'], palette = 'dark')
ax.set_xlabel(xlabel = 'Countries', fontsize = 9)
ax.set_ylabel(ylabel = 'Weight in lbs', fontsize = 9)
ax.set_title(label = 'Distribution of Weight of players from different countries', fontsize = 20)
plt.show()
sns.jointplot(x = data[data["Preferred Foot"]=="Left"]["BallControl"], y = data[data["Preferred Foot"]=="Left"]["Dribbling"], data = data,kind = 'regg')
plt.show()
sns.jointplot(x = data[data["Preferred Foot"]=="Right"]["BallControl"], y = data[data["Preferred Foot"]=="Right"]["Dribbling"], data = data,kind = 'regg')
plt.show()