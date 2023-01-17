#This is my first kernel

#The aim is to analyze and predict a player's performance in IPL



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import preprocessing,cross_validation

from sklearn import linear_model



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Input data files are available in the "../input/" directory.

delivery_data=pd.read_csv("../input/deliveries.csv")

match_data=pd.read_csv("../input/matches.csv")



delivery_data.head()
match_data.head()  
player_team="Royal Challengers Bangalore"

name='V Kohli'



#extracting batsman_data

batsman_data=delivery_data[delivery_data.batsman==name]



#creating a dataframe containing names of all team except player's own team

teams=pd.DataFrame(index=match_data.team1.unique())

teams=teams.drop('Royal Challengers Bangalore')

teams
#taking data where player team has batted first

data=match_data[match_data.team1==player_team]

batting_first=list()

for team in teams.index:

    for venue in data.venue.unique():

        matches=data[(data.venue==venue)&(data.team2==team)].id

        runs=0

        balls=0

        for match in matches:

            t=batsman_data[batsman_data.match_id==match].batsman_runs.sum()

            runs=runs+t

            balls=balls+len(batsman_data[(batsman_data.match_id==match)&(batsman_data.wide_runs==0)&(batsman_data.noball_runs==0)])

        batting_first=batting_first+[[team,venue,1,balls,runs]]



#taking data where batted second

data=match_data[match_data.team2==player_team]

batting_second=list()

for team in teams.index:

    for venue in data.venue.unique():

        matches=data[(data.venue==venue)&(data.team1==team)].id

        runs=0

        balls=0

        for match in matches:

            t=batsman_data[batsman_data.match_id==match].batsman_runs.sum()

            runs=runs+t

            balls=balls+len(batsman_data[(batsman_data.match_id==match)&(batsman_data.wide_runs==0)&(batsman_data.noball_runs==0)])

        batting_second=batting_second+[[team,venue,0,balls,runs]]

        

#merging the two data sets

batting_first=batting_first+batting_second

df=pd.DataFrame(data=batting_first,columns=['team','venue','batting_first','balls','runs'])



df=df[df.runs!=0]

df.head()

        
fig = plt.figure(figsize=(16,8))  

fig.add_subplot(221)

df.groupby('team').runs.sum().plot(kind='bar',title='runs scored against each team')



#The plot shows that Virat has scored most against Chennai Super Kings
fig = plt.figure(figsize=(16,8))  

ax=fig.add_subplot(221)

df.groupby('venue').runs.sum().plot(kind='bar',title='runs scored at each venue')

#Every player scores well on his homeground and Virat is no exception.He scores most at M chinnaswamy 

#is his homeground
fig = plt.figure(figsize=(16,8))  

fig.add_subplot(221)



df.groupby('batting_first').runs.sum().plot(kind='bar',title='batting_first vs batting_second')

plt.xticks(rotation=0)



#Even though in International cricket he is one of the best while chasing in IPL he performs well 

#while batting_first as compared to batting_second
#predicting player's performance using Linear Regression



#encoding data into a suitable format

processed_df = df.copy()

le = preprocessing.LabelEncoder()

processed_df.team = le.fit_transform(processed_df.team)

processed_df.venue = le.fit_transform(processed_df.venue)



X = processed_df.drop(['runs'], axis=1).values

y = processed_df['runs'].values



X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)



reg=linear_model.LinearRegression()

reg.fit(X_train,y_train)

plt.plot(reg.predict(X_test))

plt.plot(y_test)

plt.legend(['actual_runs','predicted_runs'])

plt.show()









#ANY SUGGESTIONS ARE MOST WELCOMED