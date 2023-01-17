# Load the Data and get familiar with it
# Source: https://www.kaggle.com/c/titanic/data
# Refer to the Kaggle website for Variable Descriptions
# Note that for this Udacity project, the train.csv data is used and has been renamed 'titanic-data.csv'

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import re

%matplotlib inline

data = pd.read_csv('../input/train.csv')
data.head()
# Display summary statistics quickly
data.describe()
# Describe the DataFrame
data.info()
# Categorical Data Summary
data.describe(include=['O'])
# Convert variables into integers
# To improve code, used dictionaries

# Create new boolean variables for sex
data['Female'] = (data['Sex']=='female').astype(int)
data['Male'] = (data['Sex']=='male').astype(int)

# Create new boolean variables for passenger class
data['Pc1'] = (data['Pclass']==1).astype(int)
data['Pc2'] = (data['Pclass']==2).astype(int)
data['Pc3'] = (data['Pclass']==3).astype(int)

# Create new numerical variables to describe embarkation ports
data['Emb_C'] = (data['Embarked']=='C').astype(int)
data['Emb_Q'] = (data['Embarked']=='Q').astype(int)
data['Emb_S'] = (data['Embarked']=='S').astype(int)
(data.corr().Survived).sort_values(ascending= False)
# Plot a frequency histogram with survivors and victims grouped by age
# Use the fillna function to differentiate those whose ages are NaN
# Set NaN to 100 which is much greater than max('Age') = 80

nan_sub = 100
survivors_age = data[data.Survived == True].Age.fillna(nan_sub)
victims_age = data[data.Survived == False].Age.fillna(nan_sub)


bin_width = 5
lower_bound = -5
upper_bound = 105
bar_width = 4.5
survivor_color = 'b'
victim_color = 'r'
rate_color = 'g'

bins = np.arange(lower_bound, upper_bound, bin_width)
index = bins[0:-1]

fig, (ax1, ax2) = plt.subplots(2,1, figsize = (15,10))
ax1.bar(index + 0.5*bar_width, np.histogram(survivors_age, bins)[0],
       color = survivor_color, width = bar_width)
ax1.bar(index + 0.5*bar_width, np.histogram(victims_age, bins)[0]*-1,
        color = victim_color, width = bar_width, hatch = 'x')

ax1.set_ylabel('Count')
ax1.set_xlabel('Age')
ax1.set_title('Age and Survival')
ax1.legend(['Survivor', 'Victim'])
ax1.axhline(0, color = 'k', linestyle = '--')
ax1.set_xticklabels(['',0, 20, 40, 60, 80, 'Age = NaN'])
ticks =  ax1.get_yticks()
ax1.set_yticklabels([int(abs(tick)) for tick in ticks])

# Plot Age vs. Survival rate

nan_sub = 100
survivors_age = data[data.Survived == True].Age.fillna(nan_sub)
all_age = data.Age.fillna(nan_sub)
age_rate = np.nan_to_num(np.histogram(survivors_age,bins)[0]/np.histogram(all_age,bins)[0])


ax2.bar(index+0.5*bar_width, age_rate, 
        color = rate_color, width = bar_width)


ax2.set_ylabel('Survival Rate')
ax2.set_xlabel('Age')
ax2.set_title('Age and Survival Rate')
ax2.set_xticklabels(['',0, 20, 40, 60, 80, 'Age = NaN'])
plt.subplots_adjust(hspace = 0.3)
plt.show()


# Explore passengers presume to be children
age_cutoff = 18
data[data.Age<age_cutoff].head()
# Explore passengers presumed to be adults
data[data.Age>=age_cutoff].head()
# Explore passengers whose 'Age' is NaN
data[data.Age.isnull()].head()
# Create a new variable 'Title'
# extract the string that preceeds a "."
# Quickly confirm that the operation was performed properly
import re
regex = re.compile('([A-Za-z]+)\.')
data['Title'] = data.Name.str.extract(regex, expand = True)
data['Title'].describe()
# Compare 'Title' and 'Sex'
data.pivot_table('Name', index = 'Title', columns = ['Sex'], aggfunc =('count')).fillna(0)
data[data.Title == 'Capt']
# Define Child and Adult status by age
age_cutoff = 18
data['Child'] = (data.Age < age_cutoff).astype(int)
data['Woman'] = (data[data.Sex=='female'].Age>=age_cutoff).astype(int)
data['Man'] = (data[data.Sex=='male'].Age>=age_cutoff).astype(int)
data['Woman'] = (data['Woman']).fillna(0)
data['Man'] = (data['Man']).fillna(0)
# How much of the sample has been categorized?
data['Child'].sum()+data['Woman'].sum()+data['Man'].sum()
data.Age.isnull().sum()
mask_m = (data.Age.isnull()) & (data.Title == 'Mr')
mask_w = (data.Age.isnull()) & (data.Title == 'Mrs')
mask_c = (data.Age.isnull()) & (data.Title =='Master')
data.loc[mask_m,'Man'] = 1
data.loc[mask_w, 'Woman'] =1
data.loc[mask_c, 'Child'] = 1

# How much of the sample has been categorized?
data['Child'].sum()+data['Woman'].sum()+data['Man'].sum()
# What unique titles remain in our group of uncategorized passengers?
data[data.Man ==0].loc[data.Woman ==0].loc[data.Child==0].Title.unique()
# Categorize those without the 'Title' 'Miss' as adults, either 'Man' or 'Woman'

mask_m = ((data.Man == 0) & (data.Woman == 0) & (data.Child == 0) & (data.Title != 'Miss') 
          & (data.Sex == 'male'))
mask_w = ((data.Man == 0) & (data.Woman == 0) & (data.Child == 0) & (data.Title != 'Miss') 
          & (data.Sex == 'female'))
data.loc[mask_m, 'Man'] = 1
data.loc[mask_w, 'Woman'] =1

# How much of the sample has been categorized?
data['Child'].sum()+data['Woman'].sum()+data['Man'].sum()
# Categorize 'Miss' passengers who have SibSp or Parch greater than 0 as children
mask_c = (data.Child ==0) & (data.Woman ==0) & (data.Man ==0) & (data.SibSp > 0)
data.loc[mask_c, 'Child'] =1
mask_c = (data.Child ==0) & (data.Woman ==0) & (data.Man ==0) & (data.Parch > 0)
data.loc[mask_c, 'Child'] =1

# How much of the sample has been categorized?
data['Child'].sum()+data['Woman'].sum()+data['Man'].sum()
# Are women with the 'Title' "Miss" more likely to travel solo than children with 'Title' of "Miss"
miss_child = data[data.Child == 1].loc[data.Title == 'Miss'].PassengerId.count()
# 65 children with Title Miss
solo_miss_child = (data[data.Child == 1].loc[data.Title == 'Miss'].loc[data.SibSp == 0]
                   .loc[data.Parch ==0].PassengerId.count())
# 11 of them are travelling solo most are teenagers, one is 5 years old.

miss_woman = data[data.Woman == 1].loc[data.Title == 'Miss'].PassengerId.count()
# 95 Women with the Title Miss
solo_miss_woman = (data[data.Woman == 1].loc[data.Title == 'Miss'].loc[data.SibSp == 0]
                   .loc[data.Parch ==0].PassengerId.count())
# 67 are travelling solo
(solo_miss_woman/miss_woman)/(solo_miss_child/miss_child)
# Women with title of Miss are 4.2 times more likely to travel solo than girls
# Are women with the 'Title' "Miss" more likely to travel solo than those with 'Title' "Mrs"
woman = data[data.Woman == 1].PassengerId.count()
# 233 women
solo_woman = data[data.Woman ==1].loc[data.SibSp ==0].loc[data.Parch ==0].PassengerId.count()
# 93 are travelling solo
(solo_miss_woman/miss_woman)/(solo_woman/woman)
# Women with title of Miss are 1.7 times more likely to travel solo those with title of Mrs
# Do men travel solo more than boys?
boy = data[data.Child == 1].loc[data.Sex == 'male'].PassengerId.count()
# 62 male children
solo_boy = (data[data.Child == 1].loc[data.Sex == 'male'].loc[data.SibSp == 0]
            .loc[data.Parch ==0].PassengerId.count())
# 12 of them are travelling solo, all have the title Mr. ages 11-17
man = data[data.Man ==1].PassengerId.count()
# 515 Men
solo_man = data[data.Man == 1].loc[data.SibSp == 0].loc[data.Parch ==0].PassengerId.count()
# 399 are travelling solo
(solo_man/man)/(solo_boy/boy)
# Men are 4 times more likely to travel solo than boys
# Create a mask for solo travelling Miss who are NaN old
data[data.Man ==0].loc[data.Woman ==0].loc[data.Child==0]
mask_miss_NaN = (data.Age.isnull()) & (data.Title == 'Miss') & (data.SibSp == 0) & (data.Parch ==0)

# Assign these passengers to the 'Woman' group
data.loc[mask_miss_NaN, 'Woman'] = 1

# How much of the sample has been categorized?
data['Child'].sum()+data['Woman'].sum()+data['Man'].sum()

# Survival Rates of Adults and Children by Sex
data.pivot_table('Survived', index = 'Sex', columns = 'Child', aggfunc = 'mean')
# Create dataframes of survivors and victims by boarding groups


survivors = data.loc[data.Child == 0].loc[data.Survived == 1].pivot_table('Survived', index = 'Sex',
                                                                          aggfunc = 'count')
survivors.loc['Children']=data.loc[data.Child == 1].loc[data.Survived == 1]['Survived'].count()
survivors = Series(survivors.values[:,0], index = ['Women', 'Men', 'Children'], name = 'Group')

victims = data.loc[data.Child == 0].loc[data.Survived == 0].pivot_table('Survived', index = 'Sex', 
                                                                        aggfunc = 'count')
victims.loc['Children']= data.loc[data.Child ==1].loc[data.Survived ==0]['Survived'].count()
victims = Series(victims.values[:,0], index = ['Women', 'Men', 'Children'], name = 'Group')
# Plot survival of boarding groups

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize= (12,8))
fig = survivors.plot.bar(ax = ax1)
victims.apply(lambda x: -x).plot.bar(ax= ax1, hatch = 'x', color = 'r')

ax1.legend([ 'Survivors', 'Victims'])
ax1.axhline(0, color = 'k', linestyle = '--')
ax1.set_ylim([-500, 250])
ticks =  ax1.get_yticks()
ax1.set_yticklabels([int(abs(tick)) for tick in ticks])
ax1.set_xticklabels(ax1.xaxis.get_majorticklabels(), rotation =0)
ax1.set_ylabel('Count')
ax1.set_title('Survival by Boarding Group')

x_offset = -0.03
y_offset = 10
y_drop = -40

for p in [0,1,2]:
    b = ax1.patches[p].get_bbox()
    val = "{:.2f}".format(b.y1 + b.y0)        
    ax1.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset))

for p in [3,4,5]:
    b = ax1.patches[p].get_bbox()
    val = "{:.2f}".format(abs(b.y1 + b.y0))        
    ax1.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y0 + y_drop))


# Plot survival rates of Women, Men, and Children

survival_rate = data[data.Child == 0].pivot_table('Survived', index = 'Sex', aggfunc = 'mean')
survival_rate.loc['Children']=data[data.Child == 1]['Survived'].mean()
survival_rate = Series(survival_rate.values[:,0], index = ['Women', 'Men', 'Children'], name = 'Group')

survival_rate.plot.bar(ax = ax2, color = 'g')
ax2.set_ylim([0,1])
ax2.set_ylabel('Proportion of Survivors')
ax2.set_title('Survival Rates by Boarding Group')
ax2.set_xticklabels(ax2.xaxis.get_majorticklabels(), rotation =0)


x_offset = -0.03
y_offset = 0.02
for p in ax2.patches:
    b = p.get_bbox()
    val = "{:.2f}".format(b.y1 + b.y0)        
    ax2.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset))

plt.tight_layout()
plt.show()
# Plot proportion of 'Pclass' in pie chart

df = None
df = data.pivot_table('PassengerId', index = 'Pclass', aggfunc = 'count')

df.plot.pie(subplots = True, labels = ['1st Class', '2nd Class', '3rd Class'], legend = False, 
           autopct = '%.1f')
plt.title('Proportion of Passengers by Passenger Class')
plt.ylabel('')
plt.tight_layout()
plt.show()
data[['Survived', 'Pclass']].groupby('Pclass').mean()
# Survival rate of 1st class vs. 3rd
data[data.Pclass ==1].Survived.mean()/data[data.Pclass==3].Survived.mean()
# What percentage of victims were in third class
data[data.Survived ==0].Pc3.mean()
# Create dataframes for survivors, victims and survival rate

survivors = DataFrame(columns = np.unique(data.Pclass.values))

survivors = survivors.append(data[data.Woman ==1].pivot_table('Survived', columns = 'Pclass',
                                                             aggfunc = 'sum'))
survivors = survivors.append(data[data.Man ==1].pivot_table('Survived', columns = 'Pclass',
                                                         aggfunc = 'sum'))
survivors = survivors.append(data[data.Child ==1].pivot_table('Survived', columns = 'Pclass',
                                                         aggfunc = 'sum'))
survivors.columns = ['1st Class', '2nd Class', '3rd Class']


survivors.index = ['Women', 'Men', 'Children']

victims = DataFrame(columns = np.unique(data.Pclass.values))
victims = victims.append(data[data.Woman ==1].loc[data.Survived ==0].pivot_table('Survived', columns = 'Pclass',
                                                                               aggfunc = 'count'))
victims = victims.append(data[data.Man ==1].loc[data.Survived ==0].pivot_table('Survived', columns = 'Pclass', 
                                                                           aggfunc = 'count'))
victims = victims.append(data[data.Child ==1].loc[data.Survived ==0].pivot_table('Survived', 
                                                                                  columns = 'Pclass', 
                                                                                  aggfunc = 'count'))
victims.index = ['Women', 'Men', 'Children']

survival_rate = DataFrame(columns = np.unique(data.Pclass.values))
survival_rate = survival_rate.append(data[data.Woman == 1].pivot_table('Survived', columns = 'Pclass',
                                                               aggfunc = 'mean'))
survival_rate = survival_rate.append(data[data.Man == 1].pivot_table('Survived', columns = 'Pclass',
                                                               aggfunc = 'mean'))
survival_rate = survival_rate.append(data[data.Child == 1].pivot_table('Survived', columns = 'Pclass',
                                                                 aggfunc = 'mean'))
survival_rate.columns = ['1st Class', '2nd Class', '3rd Class']
survival_rate.index = ['Women', 'Men', 'Children']

# Plot survival for passenger class and boarding group

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols =1, figsize = (12,8))
fig = survivors.plot.bar(ax = ax1)
victims.apply(lambda x: -1*x).plot.bar(ax = ax1, hatch = 'x')

ax1.legend(['1st Class Survivors', '2nd Class Survivors', '3rd Class Survivors', 
            '1st Class Victims', '2nd Class Victims', '3rd Class Victims'])
ax1.axhline(0, color = 'k', linestyle = '--')
ax1.set_xticklabels(ax1.xaxis.get_majorticklabels(), rotation = 0)
ax1.set_ylabel('Frequency')
ax1.set_ylim([-300, 100])
ticks =  ax1.get_yticks()
ax1.set_yticklabels([int(abs(tick)) for tick in ticks])

ax1.set_title('Survival of Boarding Groups by Passenger Class')

x_offset = -0.04
y_offset = 10
y_drop = -20
for p in np.arange(0,9):
    b = ax1.patches[p].get_bbox()
    val = "{:.0f}".format(b.y1 + b.y0)        
    ax1.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset))

for p in np.arange(9,18):
    b = ax1.patches[p].get_bbox()
    val = "{:.0f}".format(abs(b.y1 + b.y0))        
    ax1.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y0 + y_drop))

# Plot survival rates of Boarding Groups by Passenger Class

survival_rate.plot.bar(ax = ax2)
ax2.set_ylim([0,1])
ax2.set_ylabel('Proportion of Survivors')
ax2.legend()
ax2.set_title('Survival Rates of Boarding Groups by Passenger Class')
ax2.set_xticklabels(ax2.xaxis.get_majorticklabels(), rotation = 0)

x_offset = -0.04
y_offset = 0.01
for p in ax2.patches:
    b = p.get_bbox()
    val = "{:.2f}".format(b.y1 + b.y0)        
    ax2.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset))

plt.tight_layout()
plt.show()
# How does 'Fare' correlate with other variables
data.corr().Fare.sort_values(ascending = False)
# Plot 'Fare' as a histogram, include median fares for each 'Pclass'

bin_width = 25
lower_bound = -25
upper_bound = 300
bar_width = 20
col_fare = 'y'
col_rate = 'g'
col_c1 = 'b'
col_c2 = 'g'
col_c3 = 'r'


bins = np.arange(lower_bound, upper_bound, bin_width)
index = bins[0:-1]


fig, (ax1, ax2) = plt.subplots(2,1, figsize = (15,12))

ax1.bar(index+0.5*bar_width, np.histogram(data.Fare, bins)[0], 
         width = bar_width, color = col_fare)
ax1.set_xlim(lower_bound, upper_bound)

ax1.axvline(data[data.Pclass == 1].Fare.median(), color = col_c1)
ax1.axvline(data[data.Pclass == 2].Fare.median(), color = col_c2)
ax1.axvline(data[data.Pclass == 3].Fare.median(), color = col_c3)

ax1.set_ylabel('Count')
ax1.set_xlabel('Fare ($)')
ax1.set_title('Fare Histogram')


# Plot fares vs. survival rate, include average survival rate for each 'Pclass'

fare_rate = np.nan_to_num(np.histogram(data[data.Survived ==1].Fare,bins )[0]
                          /np.histogram(data.Fare, bins)[0])


ax2.bar(index+0.5*bar_width, fare_rate, 
        color = col_rate, width = bar_width)
ax2.set_xlim(lower_bound, upper_bound)


ax2.axvline(data[data.Pclass == 1].Fare.median(), color = col_c1)
ax2.axvline(data[data.Pclass == 2].Fare.median(), color = col_c2)
ax2.axvline(data[data.Pclass == 3].Fare.median(), color = col_c3)
ax2.axhline(data[data.Pclass ==1].Survived.mean(), color = col_c1, linestyle = '--')
ax2.axhline(data[data.Pclass ==2].Survived.mean(), color = col_c2, linestyle = '--')
ax2.axhline(data[data.Pclass ==3].Survived.mean(), color = col_c3, linestyle = '--')
ax2.set_ylabel('Survival Rate')
ax2.set_xlabel('Fare ($)')
ax2.set_title('Survival Rate vs. Fare')
ax2.legend(['1st Class Median Fare', 
            '2nd Class Median Fare',
            '3rd Class Median Fare',
            '1st Class Survival Rate',
            '2nd Class Survival Rate',
            '3rd Class Survival Rate'])
ax2.set_ylim(0,1)
plt.subplots_adjust(hspace = .3)
plt.show()

# Box plot 'Fare' for 'Man', 'Woman', 'Child' and 'Pclass'

df = DataFrame(columns = np.unique(data.Pclass.values))
df['Fare'] = data['Fare']
df[1] = data['Fare'].loc[data.Pclass ==1]
df[2] = data['Fare'].loc[data.Pclass ==2]
df[3] = data['Fare'].loc[data.Pclass ==3]
df['Women'] = data['Fare'].loc[data.Woman ==1]
df['Men'] = data['Fare'].loc[data.Man ==1]
df['Children'] = data['Fare'].loc[data.Child==1]
df.drop(['Fare'], axis =1, inplace = True)

fig, ax = plt.subplots(figsize=(12, 4))
fig = df.plot.box(vert = False, ax = ax)
ax.set_xlim([0,175])
ax.set_ylabel('Boarding Group or Passenger Class')
ax.set_xlabel('Fare ($)')
ax.set_title('Box Plots of Fares vs. Boarding Group or Passenger Class')
plt.show()
# Summarize survival rates
# Need to learn how to write a function for this

survival_rank = DataFrame(columns = ['Survival Rate', 'Count', 'Proportion Sample', 'Proportion Victims'])
victims = data[data.Survived==0].PassengerId.count()

survival_rank.loc['Ship Total'] = [data.Survived.mean(),
                                     data.Survived.count(),
                                  1,
                                  1]

survival_rank.loc['Women'] = [data['Survived'][data.Woman ==1].mean(),
                              data['Survived'][data.Woman ==1].count(),
                             data.Woman.mean(),
                             data[data.Survived==0].Woman.mean()]

survival_rank.loc['Men'] = [data['Survived'][data.Man == 1].mean(),
                            data['Survived'][data.Man == 1].count(),
                           data.Man.mean(),
                           data[data.Survived==0].Man.mean()]

survival_rank.loc['Children'] = [data['Survived'][data.Child == 1].mean(),
                            data['Survived'][data.Child == 1].count(),
                           data.Child.mean(),
                           data[data.Survived==0].Child.mean()]

survival_rank.loc['First Class'] = [data['Survived'][data.Pc1 ==1].mean(),
                                    data['Survived'][data.Pc1 ==1].count(),
                                   data.Pc1.mean(),
                                   data[data.Survived==0].Pc1.mean()]

survival_rank.loc['Second Class'] = [data['Survived'][data.Pc2 ==1].mean(),
                                    data['Survived'][data.Pc2 ==1].count(),
                                    data.Pc2.mean(),
                                    data[data.Survived==0].Pc2.mean()]

survival_rank.loc['Third Class'] = [data['Survived'][data.Pc3 ==1].mean(),
                                    data['Survived'][data.Pc3 ==1].count(),
                                   data.Pc3.mean(),
                                   data[data.Survived==0].Pc3.mean()]


#survival_rank.sort_values(by = 'Survival Rate', ascending = False)

survival_rank['Death Premium'] = survival_rank['Proportion Victims']/survival_rank['Proportion Sample']-1
survival_rank.sort_values(by = 'Death Premium', ascending = True)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.info()
print('Training Data')
train.info()
print('-'*20)
print('Test Data')
test.info()
def create_titles(df):
    regex = re.compile('([A-Za-z]+)\.')
    df['Title'] = df.Name.str.extract(regex, expand = True)
    return df

def create_boarding_groups(df, age_cutoff = 18):
    df['Child'] = (df.Age < age_cutoff).astype(int)
    df['Woman'] = (df[df.Sex == 'female'].Age >= age_cutoff).astype(int)
    df['Man'] = (df[df.Sex == 'male'].Age >= age_cutoff).astype(int)
    df['Woman'] = (df['Woman']).fillna(0)
    df['Man'] = (df['Man']).fillna(0)
    
    mask_m = (df.Age.isnull()) & (df.Title == 'Mr')
    mask_w = (df.Age.isnull()) & (df.Title == 'Mrs')
    mask_c = (df.Age.isnull()) & (df.Title =='Master')
    df.loc[mask_m,'Man'] = 1
    df.loc[mask_w, 'Woman'] =1
    df.loc[mask_c, 'Child'] = 1
    
    mask_m = ((df.Man == 0) & (df.Woman == 0) & (df.Child == 0) & (df.Title != 'Miss') 
          & (df.Sex == 'male'))
    mask_w = ((df.Man == 0) & (df.Woman == 0) & (df.Child == 0) & (df.Title != 'Miss') 
          & (df.Sex == 'female'))
    df.loc[mask_m, 'Man'] = 1
    df.loc[mask_w, 'Woman'] =1
    
    mask_c = (df.Child ==0) & (df.Woman ==0) & (df.Man ==0) & (df.SibSp > 0)
    df.loc[mask_c, 'Child'] =1
    mask_c = (df.Child ==0) & (df.Woman ==0) & (df.Man ==0) & (df.Parch > 0)
    df.loc[mask_c, 'Child'] =1
    
    # Create a mask for solo travelling Miss who are NaN old
    df[df.Man ==0].loc[df.Woman ==0].loc[df.Child==0]
    mask_miss_NaN = (df.Age.isnull()) & (df.Title == 'Miss') & (df.SibSp == 0) & (df.Parch ==0)

    # Assign these passengers to the 'Woman' group
    df.loc[mask_miss_NaN, 'Woman'] = 1

    #print(df['Child'].sum() + df['Woman'].sum() + df['Man'].sum())
    return df

def encode_passenger_class(df):
    # Create new boolean variables for passenger class
    df['Pc1'] = (df['Pclass']==1).astype(int)
    df['Pc2'] = (df['Pclass']==2).astype(int)
    df['Pc3'] = (df['Pclass']==3).astype(int)
    return df

def encode_sex(df):
    # Create new boolean variables for sex
    df['Female'] = (df['Sex']=='female').astype(int)
    df['Male'] = (df['Sex']=='male').astype(int)
    return df

def create_ml_df(df):
    df = create_titles(df)
    df = create_boarding_groups(df)
    df = encode_passenger_class(df)
    df = encode_sex(df)
    df = df.drop(columns=['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 
                        'Cabin', 'Embarked', 'Title'])
    return df
# Load datasets and convert them into machine learning dataframes

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train_df = create_ml_df(train)
test_df = create_ml_df(test)

train_y = train_df['Survived']
train_x = train_df.drop(columns = ['Survived'])
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression()
lr_clf.fit(train_x, train_y)
from sklearn.model_selection import cross_val_score
lr_scores = cross_val_score(lr_clf, train_x, train_y, cv = 10)
lr_scores.mean()
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(train_x, train_y)
dt_scores = cross_val_score(dt_clf, train_x, train_y, cv = 10)
dt_scores.mean()
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
forest_scores = cross_val_score(forest_clf, train_x, train_y, cv=10)
forest_scores.mean()
y_predictions = lr_clf.predict(test_df)
submission = pd.DataFrame({
    "PassengerId": test['PassengerId'],
    "Survived": y_predictions
})

submission.to_csv('titanic.csv', index = False)
