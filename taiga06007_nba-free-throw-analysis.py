import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

#read csv by panda

df = pd.read_csv("../input/nba-free-throws/free_throws.csv")

print(df.head())
# make the df for creating the chart

succes_by_quater =  df.groupby(['period', 'playoffs']).shot_made.sum().unstack()

total_by_quater = df.groupby(['period', 'playoffs']).shot_made.count().unstack()

succes_by_quater['playoff_rate'] = succes_by_quater['playoffs']/total_by_quater['playoffs']

succes_by_quater['regular_rate'] = succes_by_quater['regular'] / total_by_quater['regular']



succes_by_quater = succes_by_quater.drop([6,7,8])

succes_by_quater = succes_by_quater.assign(period = list(range(1,6)))







print(succes_by_quater.head(10))

print(succes_by_quater.columns)
#create plots

f,ax = plt.subplots(figsize=(10,5))

sns.set(style="darkgrid")

sns.lineplot(x="period", y="playoff_rate",data=succes_by_quater)

sns.lineplot(x="period", y="regular_rate",data=succes_by_quater)

plt.legend(['Playoffs','Regular'])

ax.set_xticks(succes_by_quater["period"])

ax.set_xticklabels(['1Q','2Q','3Q','4Q','OT'])

ax.set_title('Free throw success rate between regular season and playoffs by period')

# make the df for creating the chart

#categorize seasons column

name_split2 = df.season.str.split(' - ')

df['season_cleaned'] = name_split2.str.get(0)

df['season_cleaned'] = df.season_cleaned.astype(int)



succes_by_season =  df.groupby(['season_cleaned', 'playoffs']).shot_made.sum().unstack()

total_by_season = df.groupby(['season_cleaned', 'playoffs']).shot_made.count().unstack()

succes_by_season['playoff_rate'] = succes_by_season['playoffs']/total_by_season['playoffs']

succes_by_season['regular_rate'] = succes_by_season['regular'] / total_by_season['regular']





succes_by_season = succes_by_season.assign(season = list(range(2006,2016)))





print(succes_by_season)

print(succes_by_season.columns)
#create plots

f,ax = plt.subplots(figsize=(10,5))

sns.set(style="darkgrid")

sns.lineplot(x="season", y="playoff_rate",data=succes_by_season)

sns.lineplot(x="season", y="regular_rate",data=succes_by_season)

plt.legend(['Playoffs','Regular'])

ax.set_title('Free throw success rate between regular season and playoffs by season')



""" 

Examples of data cleaning

end result: split into two value 

game : change team names to numeric values

playoffs: regular:0 playoff:1

time: 11:59 ~ 11:00 → 12, 10:59 ~ 10:00 → 11

make the ID for each players

"""



# split end_result into home, away

name_split = df.end_result.str.split(' - ')

df['home_score'] = name_split.str.get(0)

df['away_score'] = name_split.str.get(1)

df['home_score'] = df.home_score.astype(int)

df['away_score'] = df.away_score.astype(int)



#split game into home team and away team 

name_split = df.game.str.split(' - ')

df['home_team'] = name_split.str.get(0)

df['away_team'] = name_split.str.get(1)



#change team name to numeric value

#print(df.home_team.value_counts())

team_mapping = {'BOS': 0,'UTAH': 1,'CLE': 2,'GS': 3,'DEN': 4,'LAL': 5,'MIA': 6,'IND': 7,'LAC': 8,'HOU': 9,'CHI': 10,'ORL': 11,'TOR': 12,

                'MEM': 13,'SAC': 14,'SA': 15,'ATL': 16,'DAL': 17, 'WSH': 18,'PHX': 19,'DET': 20,'MIL': 21,'PHI': 22,'NY': 23,'POR': 24,     

                'MIN': 25,'CHA': 26,'NO': 27,'OKC': 28,'NJ': 29,'BKN': 30,'SEA': 31,'EAST': 32,'WEST': 33}







df['home_team_cleaned'] = df.home_team.map(team_mapping)

df['away_team_cleaned'] = df.away_team.map(team_mapping)



#change playoff to numeric value

df['playoffs_int'] = df.playoffs.map({'regular': 0,'playoffs': 1})





#categorize time columns 

name_split1 = df.time.str.split(':')

df['time_cleaned'] = name_split1.str.get(0)

df['time_cleaned'] = df.time_cleaned.astype(int)



#change period value to int 

df['period'] = df.period.astype(int)



#make the ID for each players

df['id'] = df.groupby(['player']).ngroup()



# Make the column that wether player can meke both shot or not 

name_split = df.play.str.split(' ')

df['made'] = name_split.str.get(5)



def int_str(x):

    if len(x) == 1:

        return int(x)

    else:

        return np.nan

    

df['made_cleaned'] = df.made.apply(lambda x:int_str(str(x)))

df['tried'] = name_split.str.get(7)

df['tried_cleaned'] = df.tried.apply(lambda x:int_str(str(x)))



def made_two_generator(made,tried):

    if made == 2 and tried == 2:

        return 1

    elif made <2 and tried == 2:

        return 0

    elif made <=1 and tried == 1:

        return np.nan

    else:

        return np.nan

        







df['made_two'] = df.apply(lambda x: made_two_generator(x.made_cleaned, x.tried_cleaned), axis=1)



# see if there are any null values

print(df.isna().any())



# see data types

print(df.dtypes)

# This is for next ml model

df_ml = df[['home_score','away_score','home_team_cleaned','away_team_cleaned','player','shot_made','playoffs_int','time_cleaned','season_cleaned','period']]

#Drop nan value on made_two column

df = df.dropna(subset = ['made_two'])

print(len(df))

#specify players. This time, my favorite player are Lebron, Kobe, carmelo, Dwight haward

# Lebron James 

lebron_james = df_ml.loc[df_ml['player'].isin(['LeBron James'])]



#Kobe Bryant

kobe_bryant = df_ml.loc[df_ml['player'].isin(['Kobe Bryant'])]



#Carmelo Anthony

carmelo_anthony = df_ml.loc[df_ml['player'].isin(['Carmelo Anthony'])]



#Dwight Howard

dwight_howard = df_ml.loc[df_ml['player'].isin(['Dwight Howard'])]
#The model can predict whether player can make freethrow. 

#Doesn't take into account the number of shots.





def logisttic_model(df):

    # divide into features and labels

    features = df[['home_score','away_score','home_team_cleaned','away_team_cleaned','playoffs_int','time_cleaned','season_cleaned','period']]

    labels = df['shot_made']

    

    #divide into train and test sets

    train_data,test_data,train_labels,test_labels = train_test_split(features,labels,test_size = 0.20, random_state = 50)

    

    #normalize data

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train_data)

    test_scaled = scaler.transform(test_data)

    

    #create and evaluate model

    model = LogisticRegression()

    model.fit(train_scaled,train_labels)

    print(model.score(test_scaled,test_labels))

    print(list(zip(['home_score','away_score','home_team_cleaned','away_team_cleaned','playoffs_int','time_cleaned','season_cleaned','period'],model.coef_[0])))
#load player dataframe into logistic_model

#Lebron James

logisttic_model(lebron_james)
#Kobe Bryant

logisttic_model(kobe_bryant)
#Carmelo Anthony

logisttic_model(carmelo_anthony)
#Dwight Howard

logisttic_model(dwight_howard)

#Only dwight_howard. External factors may not affect the success rate of free throws
#find id based on player's name

def id_generator(df,name):

    name_df = df.loc[df['player'].isin([name])]

    id_list = []

    name_df = name_df.id.apply(lambda x:id_list.append(x))

    return id_list[0]

    



print(id_generator(df,'LeBron James'))

print(id_generator(df,'Ben Wallace'))

print(id_generator(df,'Dwight Howard'))

print(id_generator(df,'Kobe Bryant'))

print(id_generator(df,'Dirk Nowitzki'))

print(id_generator(df,'Carmelo Anthony'))







#try to improve lebrons score

# can the model predict whether player make both of their free throws



# divide into features and labels



features = df[['home_team_cleaned','away_team_cleaned','playoffs_int','time_cleaned','season_cleaned','period','id']]

labels = df['made_two']



#divide into train and test sets

train_data,test_data,train_labels,test_labels = train_test_split(features,labels,test_size = 0.20,random_state = 50)



#normalize data



scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_data)

test_scaled = scaler.transform(test_data)





#load sample data for prediction



"""

1

home team : CLE

away team : BOS

playoffs : regular

period : 4

Player : Lebron James



2

home team : DTL

away team : BOS

playoffs : regular

period : 4

Player : Ben Wallace



3

home team : ORL

away team : BOS

playoffs : regular

period : 4

Player : Dwight Howard



4



home team : LAL

away team : BOS

playoffs : regular

period : 4

Player : Kobe Bryant



5

home score : 110

away score : 109

home team : DAL

away team : BOS

playoffs : regular

period : 4

Player : Dirk Nowitzki



6

home team : NYC

away team : BOS

playoffs : regular

period : 4

Player : Carmelo Anthony

"""



sample1 = np.array([2,0,0,0,2008,4,661])

sample2 = np.array([20,0,0,0,2008,4,93])

sample3 = np.array([11,0,0,0,2008,4,321])

sample4 = np.array([5,0,0,0,4,2008,628])

sample5 = np.array([17,0,0,0,4,2008,303])

sample6 = np.array([150,0,0,0,4,2008,150])

sample_score = np.array([sample1,sample2,sample3,sample4,sample5,sample6])

sample = scaler.transform(sample_score)





#create and evaluate model

model = LogisticRegression(C= 0.3,random_state = 50)

model.fit(train_scaled,train_labels)

labels_pred = model.predict(test_scaled)

print(model.score(test_scaled,test_labels))

print(list(zip(['home_team_cleaned','away_team_cleaned','playoffs_int','time_cleaned','season_cleaned','period','id'],model.coef_[0])))



#put the data and predict whther Lebron can make a shot or not

print(model.predict(sample))

free_throw_probability = model.predict_proba(sample)

print(free_throw_probability)



#Check Lebron's sucess rate of free thorws

made_shot = []

for i in free_throw_probability :

    made_shot.append(i[1])

made_shot = np.array(made_shot)

rate = np.mean(made_shot)*100.0

rate = np.round_(rate)

print(str(rate)+"%")





#little bit improved 



# import the metrics class

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(test_labels, labels_pred)

cnf_matrix
class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
print("Accuracy:",metrics.accuracy_score(test_labels, labels_pred))

print("Precision:",metrics.precision_score(test_labels, labels_pred))

print("Recall:",metrics.recall_score(test_labels, labels_pred))

print("F1:",metrics.f1_score(test_labels, labels_pred, average = 'micro'))
labels_pred_proba = model.predict_proba(test_scaled)[::,1]

fpr, tpr, _ = metrics.roc_curve(test_labels,  labels_pred_proba)

auc = metrics.roc_auc_score(test_labels, labels_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()