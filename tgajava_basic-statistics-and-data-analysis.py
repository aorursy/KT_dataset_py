# importing the neccessary packages
import pandas as pd
import statistics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# the first column can be used as index of the dataframe
fifa18 = pd.read_csv("../input/CompleteDataset.csv", index_col=0)
# all columns
print(fifa18.columns)
# Take a look at the data
fifa18.head()
print("mixed types: \n", fifa18.columns[23:35])
print("unique values, column 23\n\n", fifa18.iloc[:,23].unique())
print("unique values, column 23\n\n", fifa18.iloc[:,30].unique())
print("unique values, column 66\n\n", fifa18.iloc[:,66].unique())
# #This operation took more than 10 minutes

# for i, v in enumerate(fifa18['Value']):
#     if('M' in v):
#         # 1:-1 is slicing string from one to size - 1
#         # you know that slicing doesn't include the last element
#         fifa18['Value'][i] = float(fifa18['Value'][i][1:-1]) * 1e6
#     elif('K' in v):
#         fifa18['Value'][i] = float(fifa18['Value'][i][1:-1]) * 1e3
#     else:
#         fifa18['Value'][i] = float(fifa18['Value'][i][1:])
# df.to_csv("fifa18_mod1.csv")
# for i, w in enumerate(df['Wage']):
#     if('K' in w):
#         df['Wage'][i] = float(df['Wage'][i][1:-1]) * 1e3
#     elif('€0' in w):
#         df['Wage'][i] = float(df['Wage'][i][1:])     
# df.to_csv("fifa18_mod1.csv")
# So I am using apply instead
def fix(x):
    # evaluate sum
    if('+' in str(x).strip()):
        calc = x.split('+')
        return int(calc[0]) + int(calc[1])
    # evaluate subtraction
    elif('-' in str(x).strip()):
        calc = x.split('-')
        return int(calc[0]) + int(calc[1])
    # convert to integer if string contains a valid number
    elif str(x).strip().isdigit():
        return int(x)
    # return as it is, for example null values
    else:
         return x
for column in fifa18.iloc[:,11:74]:
    fifa18[column] = fifa18[column].apply(fix)
# save the modfied dataframe
fifa18.to_csv("fifamod3.csv")
fifa18.head(5)
print("mixed types: \n", fifa18.columns[23:35])
print("unique values, column 23\n\n", fifa18.iloc[:,23].unique())
print("unique values, column 23\n\n", fifa18.iloc[:,30].unique())
print("unique values, column 66\n\n", fifa18.iloc[:,66].unique())
print("Number of nulls in RWB column: ", fifa18["RWB"].isna().sum())
print("Number of nulls in RWB column: ", fifa18["RWB"].isnull().sum())
# positional attribute
print("type of personal attribute: ", fifa18['Name'].dtype)
print("type of performance attribute: ", fifa18['Agility'].dtype)
print("type of positional attribute: ", fifa18['LS'].dtype)
print("type of prefered position attribute: ", fifa18['Preferred Positions'].dtype)
# silenting warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
def convert(v):
        if('M' in str(v)):
            return float(v[1:-1]) * 1e6
        elif('K' in str(v)):
            return float(v[1:-1]) * 1e3  
        elif('€0' in str(v)):
            return float(v[1:])
        else:
            return v

fifa18['Value'] = fifa18['Value'].apply(convert)
fifa18['Wage'] = fifa18['Wage'].apply(convert)
fifa18.to_csv("fifamod3.csv")
print("type of Value: ", fifa18['Value'].dtype)
print("type of Wage: ", fifa18['Wage'].dtype)
all_perfo_attribs = np.concatenate([np.arange(5,7), np.arange(11,46)])
print(all_perfo_attribs)
ten_perfo_attribs = np.random.choice(all_perfo_attribs, 10)
print("randomly selected: ", ten_perfo_attribs)
fifa18.iloc[:, ten_perfo_attribs].describe()
# Making sure the data is consistent on Club, Value and Wage columns
wage_0 = fifa18[(fifa18['Wage'] == 0)]
value_0 = fifa18[(fifa18['Value'] == 0)]
fifa18_has_clubs = fifa18.dropna(subset = ['Club'])
consistent = fifa18[(fifa18['Wage'] == 0) & (fifa18['Value'] == 0)]

print("Players with no Clubs: ", len(fifa18) - len(fifa18_has_clubs))
print("Players with no Clubs:", len(consistent))
fifa18[(fifa18['Wage'] == 0) & (fifa18['Value'] == 0)]['Name'].count()
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,4), sharey=True) 
wages = fifa18['Wage']
sns.distplot(fifa18['Wage'], kde=True, rug=False, ax=ax1);
sns.distplot(wages[wages < 100000], kde=True, rug=False, ax=ax2);
length = len(fifa18)
wages = fifa18['Wage']
highest_payed = wages.sort_values(ascending=False)[0:5].index
# lowest payed players ignoring 0
lowest_payed = (wages[wages != 0]).sort_values(ascending=True)[0:5].index
print("highest paid\n" ,fifa18[['Name','Wage']].loc[highest_payed])
print("lowest paid\n" ,fifa18[['Name','Wage']].loc[lowest_payed])

# group the data by football club
data_group_by_club = fifa18.groupby('Club')
# find the mean of each attribute and select the Overall column
clubs_average_overall = data_group_by_club.mean()['Overall']
# sort the average overall in descending order and slice the top 5
top_clubs_top_5 = clubs_average_overall.sort_values(ascending = False)[:5]
# filter the big dataframe to include only players from top clubs
fifa18_top_5 = fifa18.loc[fifa18['Club'].isin(top_clubs_top_5.index)]
# create seaborn FacetGrid object, it will contain cell per club
g = sns.FacetGrid(fifa18_top_5, col='Club')
# In each column plot the age distrubtion of a club
g.map(sns.distplot, "Age")
plt.show()
g = sns.FacetGrid(fifa18_top_5, col='Club')
g.map(sns.boxplot, "Age", order='')
plt.show()
# Categorize ages into two
fifa18_top_5['Age_Cat'] = fifa18_top_5['Age'].apply(lambda x: "Above 30" if x >= 30 else "Below 30")
g = sns.FacetGrid(fifa18_top_5, col = 'Club', row='Age_Cat', height=4, margin_titles=True)
#plt.subplots_adjust(top=0.6)
#g.fig.suptitle('Players Stamina') 
g.map(sns.distplot, "Stamina")
plt.show()
print(fifa18.columns[46:])
print(len(fifa18.columns[46:]) - 2)
forward = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW']
# All the midfield positions end with an M
midfield = ['LAM', 'CAM', 'RAM', 'LM', 'LCM', 'RCM','CM', 'RM', 'LDM', 'CDM', 'RDM']
defense = ['LB', 'LWB', 'LCB', 'CB', 'RCB', 'RB', 'RWB']

assert(len(forward) + len(midfield) + len(defense) == 26)
# if all are true
d = True
m = True
f = True
output = int("".join([str(int(d)), str(int(m)), str(int(f))]), 2)
print("player likes to play in all positions: ", output)
d = True
m = True
f = False
output = int("".join([str(int(d)), str(int(m)), str(int(f))]), 2)
print("player likes to play in defense and midfield: ", output)
def check(p):
    pos = p.split()
    # does it contain defender position
    d = any(i.strip() in defense for i in pos) 
    # does it contain midfield position
    m = any(i.strip() in midfield for i in pos)   
    # does it contain forward position
    f = any(i.strip() in forward for i in pos)  
    
    # The outer int is used to represent a binary number
    # and convert to decimal
    return int("".join([str(int(d)), str(int(m)), str(int(f))]), 2)

fifa18["Pref_Pos_Encoded"] = fifa18['Preferred Positions'].apply(check)
# Test the check function
two_columns = fifa18[['Preferred Positions','Pref_Pos_Encoded']]
# select a random starting point in the data
start = np.random.randint(len(fifa18) - 3)
print(two_columns[start: start+5])
sns.set(style="whitegrid")
ax = sns.countplot(x="Pref_Pos_Encoded", data=fifa18)
plt.xticks(np.arange(8), ('GK', 'Forward', 'Midfiled', 'M & F', 'Defender','D & F','D & M','D, M, F'))
ax.set_title("Players Preferred Position")
plt.xlabel('')
plt.show()
print("D & F: ", len(two_columns[two_columns['Pref_Pos_Encoded'] == 5]))
print("D, M & F: ", len(two_columns[two_columns['Pref_Pos_Encoded'] == 7]))
rare_players = fifa18[(fifa18['Pref_Pos_Encoded'] == 5) | (fifa18['Pref_Pos_Encoded'] == 7)]
indices = fifa18[fifa18['Pref_Pos_Encoded'] == 7]['Overall'].sort_values(ascending = False)
fifa18.loc[indices].drop_duplicates().head(3)
fifa18.columns
# Column indices of performance attributes
all_perfo_attribs
# create a dataframe containing only player performance attributes(columns)
fifa18_performance = fifa18.iloc[:,all_perfo_attribs]
fifa18_performance.columns
corr = fifa18_performance.corr()
#Three steps needed to have resized diagram
fig, ax = plt.subplots(figsize=(10,8)) 
sns.heatmap(corr, linewidths=.5, ax=ax)
ax.set_title("Correlation Between Performance Attributes")
ball_control = corr[['Ball control']].sort_values(ascending = False, by = "Ball control")
top_correlated = ball_control.head(5)
print("Highest Positive Correlation to Ball Control\n", top_correlated)
fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(16,8)) 

ax1.set_title("Ball Control vs. Dribbling")
ax2.set_title("Agility vs. Strength")
ax3.set_title("Balance vs. Strength")

sns.regplot(x="Ball control", y="Dribbling", ax = ax1, data=fifa18, color='b')
sns.regplot(x="Agility", y="Strength", ax = ax2, data=fifa18, color="g")
sns.regplot(x="Balance", y="Strength", ax = ax3, data=fifa18, color="y")
fifa18.loc[(fifa18["Strength"] > 80) & (fifa18["Balance"] > 80)]
fifa18_filtered = fifa18.loc[fifa18['Pref_Pos_Encoded'].isin([4,2,1])]
# convert the numerical encoding into representative text
fifa18_filtered['Pref_Pos_String'] = fifa18_filtered['Pref_Pos_Encoded'].apply(lambda x: "Defense" if x == 4 else ("Midfield" if x == 2 else "Attack"))
# create 3 by 3 grid of axes
fig, axes= plt.subplots(3,3, figsize=(16,8),sharey = True) 
# find preformance columns
p_columns = fifa18_performance.columns
# select 9 attributes
under_study = list(p_columns[3:12])
# flatten the axis list
axes_list = axes.ravel()
for i, ax in enumerate(axes_list):
    ax.xaxis.label.set_visible(False)
    ax.grid = False
    sns.barplot(x="Pref_Pos_String", y=under_study[i], data=fifa18_filtered, ax = ax)
grouped_mean = fifa18_filtered.iloc[:, np.append(all_perfo_attribs,75)].groupby("Pref_Pos_String").mean()
grouped_mean
grouped_mean.loc["Attack"][5:].sort_values(ascending = False).head(5)
grouped_mean.loc["Midfield"][3:].sort_values(ascending = False).head(5)
grouped_mean.loc["Defense"][3:].sort_values(ascending = False).head(5)