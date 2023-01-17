# Introduction about the data set - I have taken this dataset from Kaggle and this data set is about UFC fights with 

    #lot of details like player's name ,wight, height , reach, Winner, Win by Knockout, 

    #Below has some info on the columns I have analyzed from the dataset

    #R_ and B_ prefix signifies red and blue corner fighter stats respectively

    #_opp_ containing columns is the average of damage done by the opponent on the fighter

    #SIG_STR is no. of significant strikes 'landed of attempted'

    #TOTAL_STR is total strikes 'landed of attempted'

    #HEAD is no. of significant strinks to the head 'landed of attempted'

    #BODY is no. of significant strikes to the body 'landed of attempted'

    #Format is the format of the fight (3 rounds, 5 rounds etc.)

    #location is the location in which the event took place

    #Fight_type is which weight class and whether it's a title bout or not

    #Winner is the winner of the fight

    #Stance is the stance of the fighter (orthodox, southpaw, etc.)

    #Height_cms is the height in centimeter

    #Reach_cms is the reach of the fighter (arm span) in centimeter

    #Weight_lbs is the weight of the fighter in pounds (lbs)

    #age is the age of the fighter

    #weight_class is which weight class the fight is in (Bantamweight, heavyweight, Women's flyweight, etc.)

    #no_of_rounds is the number of rounds the fight was scheduled for

    #draw is the number of draws in the fighter's ufc career

    #wins is the number of wins in the fighter's ufc career

    #losses is the number of losses in the fighter's ufc career



# Objective - I know nothing about the UFC fights other than the fact 2 people fight. My goal from this anlysis is 

            # to know , that how much info I can find out by just analysing the data set. This is ongoing analysis but 

            # I just wanted to post this here and will be updating the jupyter notebook as I progress.

from pandas import ExcelWriter
import pandas as pd

import os

from os import listdir
import pandas as pd

data = pd.read_csv("../input/ufcdata/data.csv")

preprocessed_data = pd.read_csv("../input/ufcdata/preprocessed_data.csv")

raw_fighter_details = pd.read_csv("../input/ufcdata/raw_fighter_details.csv")

raw_total_fight_data = pd.read_csv("../input/ufcdata/raw_total_fight_data.csv")
print(data.shape, preprocessed_data.shape,  raw_fighter_details.shape, raw_total_fight_data.shape)

len(data['R_fighter'].unique())
data.dtypes.head()
from dateutil.parser import parse

def is_date(string, fuzzy=False):  ## To check if certain column is date or not

    try: 

        parse(string, fuzzy=fuzzy)

        return True



    except ValueError:

        return False
### Converting column datatype to date - if any date column is present - After the import the date colums comes as Object

for cols in data.columns:

    coltype = data[cols].dtype

    if (coltype == 'object'):

        if(is_date(data.loc[0,cols])==True):

            data[cols] = pd.to_datetime(data[cols])

        else:

            data[cols] = data[cols].apply(str)        
### This Function is designed to do data profiling for any data set. 

def prodf(df):

    cat_var_value_cnt =10

    col_info = pd.DataFrame(columns=['col_name','data_type','variable_type','total_rows','unique_val_count',

                                     'non_missing_count','missing_count','most_common_val','max_value','min_value',

                                     'max_value_length','min_value_length'])

    col_info['col_name'] = df.columns

    col_info['total_rows'] = df.shape[0]

    row_cnt =0

    for item in df.columns:

        col_info.loc[row_cnt,'unique_val_count'] = len(df[item].unique())

        col_info.loc[row_cnt,'variable_type'] = 'Categorical' if len(df[item].unique())<=10 else 'Non-Categorical'

        col_info.loc[row_cnt,'data_type'] = df[item].dtype

        col_info.loc[row_cnt,'missing_count'] = df[item].isna().sum()

        col_info.loc[row_cnt,'non_missing_count'] = df.shape[0]-df[item].isna().sum()

        col_info.loc[row_cnt,'max_value'] = df.loc[df[item].isna()==False,[item]].max().max()

        col_info.loc[row_cnt,'min_value'] = df.loc[df[item].isna()==False,[item]].min().min()

        col_info.loc[row_cnt,'max_value_length'] = df[item].astype(str).str.len().max()

        col_info.loc[row_cnt,'min_value_length'] = df[item].astype(str).str.len().min()

        col_info.loc[row_cnt,'most_common_val'] = df[item].value_counts().idxmax()

        

        ### Check the data type and populate the columns like "most common val" and "mean" column accordingly

        ##print(df[item].max())

        row_cnt = row_cnt+1

        

    return col_info
data_profile = prodf(data)

data_profile.head(5)
## Removing columnes which has missing values for more than 1000 recods

del_cols =data_profile[data_profile['missing_count']>1000]['col_name']

data.head()

df = data.drop(del_cols,axis=1)
import matplotlib.pyplot as plt

%matplotlib inline

df.shape
##Number of winners based on red and blue - Bar Chart

fig, ax= plt.subplots()

#Count orrurence of red and blue wins

wdata = df['Winner'].value_counts()

#Get X and y Data

points= wdata.index

frequency=wdata.values

#Create bar charts

ax.bar(points,frequency)

#Set title and labels

ax.set_title('Red_Blue_Scores')

ax.set_xlabel('points')

ax.set_ylabel('frequency')

plt.show()
##Different type of weight class # Leighweight champianship is the most popular UFC game



fig, ax = plt.subplots(figsize=(10, 10))

wdata = df['weight_class'].value_counts()

points = wdata.index

frequency = wdata.values

ax.bar(points,frequency, color='tan')

ax.set_title('Diffirent_types_of_Weight_Scores')

ax.set_xlabel('Weight Class')

ax.set_ylabel('frequency')

plt.xticks(rotation=90)

plt.show()

fig, ax = plt.subplots(figsize=(15,6))

gdata = df['location'].value_counts()

### remove 2 entry from the top in the list Las Vegas with 1200 Match, London with 110 matches - 

        ###other places have smaller number of matches and it was causing the plot to skew

y_data = gdata[2:].head(20).values

x_data = gdata[2:].head(20).index

ax.plot(x_data, y_data)

plt.xlabel('Number of Fight')

plt.ylabel('Location')

plt.title('Number of fight in that location')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

plt.scatter(df['R_Height_cms'],df['R_Reach_cms'],color='Red',alpha=0.1)

plt.xlabel('Height')

plt.ylabel('Reach')

plt.title ('Red Player Height Vs Reach')



plt.subplot(1,2,2)

plt.scatter(data['B_Height_cms'],data['B_Reach_cms'],color='Blue',alpha=0.1)

plt.xlabel('Height')

plt.ylabel('Reach')

plt.title ('Blue Player Height Vs Reach')



plt.show()



## creating the data set to determine which all player best at Knockout

## This data is normalized  and need to prepare smaller data set for any further analysis - Scenario by Scenario

data.columns

## Merging 2 data set vertically

r_fight_ko = data[['R_fighter','R_win_by_KO/TKO']].rename(columns={'R_fighter':'fighter','R_win_by_KO/TKO':'win_by_ko'})

b_fight_ko = data[['B_fighter','B_win_by_KO/TKO']].rename(columns={'B_fighter':'fighter','B_win_by_KO/TKO':'win_by_ko'})



kodata = pd.concat([r_fight_ko,b_fight_ko], axis=0)

print("Fighter & their win by KO from the dataset:- "+ str(kodata.shape))
## Since I prepared the dataset from original dataset - it has lots of duplicate - Need to remove the duplicate

kodata.drop_duplicates(inplace=True)
## Removing players who never win fight by ko

kodata = kodata[kodata['win_by_ko']>0] ## There are 1546 player who won the fight by KO. Only few of them have won the fight by KO Many times.

kodata.sort_values(by='win_by_ko', ascending=False)
## Who is the best fighter 

## Which color wins most of the time in game - Red

red_winner_fighter = data[data['Winner']=='Red']['R_fighter'] 

blue_winner_fighter = data[data['Winner']=='Blue']['B_fighter'] 

print('Number of Red fighter winner :-' + str(red_winner_fighter.shape[0]) + ' \nBlue Winner fighter :- ' + str(blue_winner_fighter.shape[0]) )
## top 30 player with most wins

all_winner = red_winner_fighter.append(blue_winner_fighter)

all_winnerdf = pd.DataFrame(all_winner, columns=['Player'])

plotdf = all_winnerdf['Player'].value_counts().head(30)

xd = plotdf.index

yd = plotdf.values



fig, ax = plt.subplots(figsize=(15,6))

ax.bar(xd, yd, color=['lightgreen','lightblue'])

ax.set_title('Number of Wins / Player')

ax.set_xlabel('player')

ax.set_ylabel('Number of Wins')

plt.xticks(rotation=90)

plt.show()
raw_fighter_details.head()

raw_fighter_details[raw_fighter_details['fighter_name']=='Donald Cerrone']

winnerdf = pd.DataFrame(columns=['fighter_name','number_of_win'])

winnerdf['fighter_name'], winnerdf['number_of_win'] = plotdf.index, plotdf.values

winnerdf.head()

#### Getting the fighter detail into the winner dataset

winnerplayerdetail = pd.merge(winnerdf,raw_fighter_details,how='left', on='fighter_name')

winnerplayerdetail.head()
### Dividing the winner dataset and find which "Stance" has lead to most of the win

stancedf = winnerplayerdetail[['number_of_win','Stance']].groupby('Stance').sum().reset_index()

xd = stancedf['Stance']

yd = stancedf['number_of_win']



plt.bar(xd,yd, color='orange')

plt.title('Win based on Stance')

plt.xlabel('Stance')

plt.ylabel('Number of Win')

plt.show()

### Orthodox has higher chance of win