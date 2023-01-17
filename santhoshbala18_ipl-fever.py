from IPython.display import HTML



HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

</script>

<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

os.listdir("../input/")



# Any results you write to the current directory are saved as output.
df = pd.read_excel("../input/iplallseasons_refined.xlsx")
df.head()
df.shape
df['Match_Year'] = df['Match_date'].apply(lambda x: str(x).split(",")[-1])
df['Team1_score'] = df['Team1_score'].apply(lambda x: str(x).split("/")[0])

df['Team2_score'] = df['Team2_score'].apply(lambda x: str(x).split("/")[0])
df.loc[df['Team2_score']=='nan']
df.dropna(inplace=True)
df['Team1_score'] = df['Team1_score'].astype('int')

df['Team2_score'] = df['Team2_score'].astype('int')
sns.countplot(df['Match_Year'])

plt.xticks(rotation=90)

plt.ylabel("Number of Matches")

plt.show()
df['Match_venue'] = df['Match_venue'].str.replace('M Chinnaswamy Stadium, Bangalore','M Chinnaswamy Stadium, Bengaluru')
plt.figure(figsize=(15,15))

sns.countplot(df['Match_venue'],palette='viridis')

plt.xticks(rotation=90)

plt.tight_layout()

plt.title("Number of Matches Per Venue")

plt.xlabel("")

plt.ylabel("Number of Matches")

plt.show()
plt.figure(figsize=(10,10))

win_perc = df.groupby('Winning_team')['Winning_team'].count()

win_perc = (win_perc.loc[win_perc>5])

win_perc.plot(kind='pie',autopct='%1.1f%%',startangle=90,pctdistance=0.85,cmap=plt.get_cmap('BrBG'))



#draw circle

centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle) 

plt.ylabel("")

plt.tight_layout()

plt.title("Win Percentage for each Team")

plt.show()
toss_match_win = df[df['Toss_winner']==df['Winning_team']].Match_number.count()

perc = toss_match_win/len(df)*100

print("Percentage of win after winning Toss {:.2f}%".format(perc))
winning = df[df['Toss_winner']==df['Winning_team']]

sns.countplot(winning['Toss_decision'],palette='magma_r')

plt.title("Decision of Winning Team After Winning the toss")

plt.ylabel("Matches")

plt.show()
toss_winner = df.groupby(['Toss_winner','Toss_decision'])['Match_number'].count().reset_index()

toss_winner = toss_winner.loc[toss_winner.Match_number>10]
plt.figure(figsize=(10,10))

sns.barplot(x='Toss_winner',y='Match_number',hue='Toss_decision',data=toss_winner,palette='gist_earth')

plt.xticks(rotation=90)

plt.ylabel("Number of Matches")

plt.tight_layout()
plt.figure(figsize=(15,10))

team1_score_mean = df.groupby('Match_venue').Team1_score.mean().reset_index()

team2_score_mean = df.groupby('Match_venue').Team2_score.mean().reset_index()

total = pd.merge(team1_score_mean,team2_score_mean,on='Match_venue')





sns.barplot(total['Match_venue'],total['Team1_score'])

plt.xticks(rotation=90)

plt.title("Team Average Score")

plt.ylabel("Average Score")

plt.show()
plt.figure(figsize=(15,5))

team1_score_mean = df.groupby('Match_Year').Team1_score.mean().reset_index()

team2_score_mean = df.groupby('Match_Year').Team2_score.mean().reset_index()

total = pd.merge(team1_score_mean,team2_score_mean,on='Match_Year')



sns.barplot(total['Match_Year'],total['Team1_score'])

sns.barplot(total['Match_Year'],total['Team2_score'])

plt.xticks(rotation=90)

plt.title("Team Average Score")

plt.ylabel("Average Score")

plt.show()
plt.figure(figsize=(15,5))

team1_score_max = df.groupby('Match_Year').Team1_score.max().reset_index()



sns.barplot(team1_score_max['Match_Year'],team1_score_max['Team1_score'])

plt.xticks(rotation=90)

plt.title("Maximum Score per Year")

plt.ylabel("Maximum Score")

plt.show()
plt.figure(figsize=(15,5))

team1_score_min = df.groupby('Match_Year').Team1_score.min().reset_index()



sns.barplot(team1_score_min['Match_Year'],team1_score_min['Team1_score'])

plt.xticks(rotation=90)

plt.title("Minimum Score per Year")

plt.ylabel("Minimum Score")

plt.show()