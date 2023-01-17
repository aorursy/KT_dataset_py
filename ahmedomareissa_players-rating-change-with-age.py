import matplotlib.ticker as mtick

from sqlite3_util_py import * #this util file contains the imported libraries and some useful functions to explore data base and table
desc() 
#importing data from sqlite

playerskills = pd.read_sql("""SELECT a.player_api_id,player_name ,date,birthday,overall_rating,potential,crossing,finishing,heading_accuracy,short_passing,volleys,dribbling,curve,free_kick_accuracy,long_passing,ball_control,acceleration,sprint_speed,agility,reactions,balance,shot_power,jumping,stamina,strength,long_shots,aggression,interceptions,positioning,vision,penalties,marking,standing_tackle,sliding_tackle  FROM Player_Attributes a join player b on a.player_api_id = b.player_api_id  ;""", conn)
playerskills.head()
playerskills.info()
#Drop NA

playerskills.dropna(inplace= True)
#Fix the dates datetype. 

playerskills[['date','birthday']] = playerskills[['date','birthday']].apply(pd.to_datetime)
#Creating age column by substracting date of rating vs birthday

playerskills['age'] = (playerskills['date'] - playerskills['birthday']).dt.days // 365

#count the age years we have per plater

player_years = playerskills.groupby([ 'player_name','player_api_id']).nunique()[['age']]

player_years['age'].plot.hist(bins = 20).set_xlabel("Age");
#get players with 9 years of history and more

players1 = player_years[player_years['age']>=9].reset_index()['player_api_id']

#get the palyer ratings between 15 and 35 years old

mask = (playerskills['age'] >= 15) & (playerskills['age'] <= 35)



#apply filters

playerskills = playerskills[playerskills['player_api_id'].isin(players1)]

playerskills = playerskills[mask]
playerskills.head()
playerskills[playerskills['player_name'].isin(['Wayne Rooney','Cristiano Ronaldo'])].groupby(['age','player_name']).mean().reset_index().pivot(index='age', columns='player_name', values='overall_rating').plot().set_ylabel("Rating")
#Average player rating per age year

average_rating_per_year = playerskills.groupby(['age','player_name','player_api_id']).mean().reset_index()



#Average player rating 

average_rating = playerskills.drop(columns = 'age').groupby(['player_api_id']).mean().reset_index()



arpy = average_rating_per_year.merge(average_rating,suffixes=('_yearly', ''), right_on = 'player_api_id', left_on = 'player_api_id' )



for i in list(average_rating.columns)[1:]: 

    arpy[i + '_change%'] = (100*(arpy[i + '_yearly'] - arpy[i])/arpy[i])

arpy[arpy['player_name'].isin(['Wayne Rooney','Cristiano Ronaldo'])].pivot(index='age', columns='player_name', values='overall_rating_change%')
plot = arpy[arpy['player_name'].isin(['Wayne Rooney','Cristiano Ronaldo'])].pivot(index='age', columns='player_name', values='overall_rating_change%').plot.bar()

plot.yaxis.set_major_formatter(mtick.PercentFormatter())

plot.set_ylabel("overall_rating_change%");
arpy.plot.scatter(x='age',y = 'overall_rating_change%');
plot = arpy.boxplot(column='overall_rating_change%',by = 'age',figsize=[10,5])

plot.yaxis.set_major_formatter(mtick.PercentFormatter())

plot.set_ylabel("overall_rating_change%");
plot = arpy.groupby('age')['overall_rating_change%'].mean().plot.bar()

plot.yaxis.set_major_formatter(mtick.PercentFormatter())

plot.set_ylabel("overall_rating_change%");
#all what we need to do then is to draw heatmap to show the age vs %change in performance per attribute

import seaborn as sns

col = [ i + '_change%' for i in average_rating_per_year.drop(columns = ['player_name','player_api_id','age']).columns]

df = arpy.groupby('age').mean()[col].astype(int).T

sns.set(rc={'figure.figsize':(10

                              ,8)})

sns.heatmap(df, annot=False)
