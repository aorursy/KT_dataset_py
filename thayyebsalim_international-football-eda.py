import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns
results = pd.read_csv("../input/international-football-results-from-1872-to-2017/results.csv", engine="python")

print(results.dtypes)

display(results)
conditions=[ results['home_score']>results['away_score'],results['home_score']==results['away_score'],results['home_score']<results['away_score']]

match_result=[ "home_win", 'draw', 'away_win']



results["result"] = np.select(conditions, match_result)

results
def home_away_draws(df): #returns number of home wins, away wins, draws and total games played 

    home_wins=len(df[df.away_score<df.home_score])

    away_wins=len(df[df.away_score>df.home_score])

    draws=len(df[df.away_score==df.home_score])

    total_games_played=home_wins+away_wins+draws

    avg_home=df['home_score'].mean()

    avg_away=df['away_score'].mean()

    return home_wins,away_wins,draws,total_games_played,avg_home,avg_away
def draw_pie(title, pie):

    labels = ['Home wins', 'Away wins', 'Draws']

    sizes = pie

    colors = ['#39375b','#745c97','#d597ce']

    plt.figure(figsize=(15,10))

    fig1, ax1 = plt.subplots()

    plt.title(title)

    patches, texts, autotexts = ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)

    for text in texts:

        text.set_color('grey')

    for autotext in autotexts:

        autotext.set_color('white')

    ax1.axis('equal')  

    plt.tight_layout()

    plt.show()
major_tournaments = ['FIFA World Cup','Confederations Cup','Oceania Nations Cup','CCCF Championship','AFC Asian Cup','Copa América','Gold Cup','UEFA Euro','NAFU Championship','African Cup of Nations']



major=results.loc[results['tournament'].isin(major_tournaments) , :]
#All games including matches that were played in nuetral grounds

m_home_wins, m_away_wins, m_draws, m_total_games_played, m_home_score_avg, m_away_score_avg = home_away_draws(major)

print(f'Away wins: {m_away_wins}')

print(f'Home wins: {m_home_wins}')

print(f'Draws: {m_draws}')

print(f'Home side goal per game: {m_home_score_avg}')

print(f'Away side goal per game: {m_away_score_avg}')



m_home_win_percent=m_home_wins/m_total_games_played

m_away_win_percent=m_away_wins/m_total_games_played

m_draw_percent=m_draws/m_total_games_played



print(m_home_win_percent)

print(m_away_win_percent)

print(m_draw_percent)



draw_pie("Home wins vs Away Wins in Neutral Games", [m_home_win_percent,m_away_win_percent,m_draw_percent])
#Matches that were played in either home or away grounds (non-neutral games)

major_nn = major[major['neutral']==False]

nn_m_home_wins, nn_m_away_wins, nn_m_draws, nn_m_total_games_played, nn_m_home_score_avg, nn_m_away_score_avg = home_away_draws(major_nn)



print(f'Away wins: {nn_m_away_wins}')

print(f'Home wins: {nn_m_home_wins}')

print(f'Draws: {nn_m_draws}')

print(f'Home side goal per game: {nn_m_home_score_avg}')

print(f'Away side goal per game: {nn_m_away_score_avg}')



nn_m_home_win_percent=nn_m_home_wins/nn_m_total_games_played

nn_m_away_win_percent=nn_m_away_wins/nn_m_total_games_played

nn_m_draw_percent=nn_m_draws/nn_m_total_games_played



print(nn_m_home_win_percent)

print(nn_m_away_win_percent)

print(nn_m_draw_percent)



draw_pie("Home wins vs Away Wins in Non-Neutral Games",[nn_m_home_win_percent,nn_m_away_win_percent,nn_m_draw_percent])
#Goals by tournaments 

avg_goal=major.pivot_table(['home_score','away_score'],columns='tournament',aggfunc=np.sum)

avg_goal.loc['total']=avg_goal.sum(axis=0)

avg_goal=avg_goal.T

avg_goal
#Average Goals per Tournament 

count=major['tournament'].value_counts()

count=count.sort_index()

avg_goal['count']=count

avg_goal['avg']=avg_goal['total']/avg_goal['count']

avg_goal
#Above table visualized

plt.figure(figsize=(20,5))

plt.title("Average Goal per Game Based on Tournament")

sns.set_style("darkgrid")

sns.barplot(x=avg_goal.index,y=avg_goal['avg'],color='#d597ce');
#Host country 

matches_hosted=pd.DataFrame()

matches_hosted['counts']=major['country'].value_counts()

matches_hosted['country']=major['country'].value_counts().index

matches_hosted=matches_hosted.reset_index()

matches_hosted.drop(columns=['index'],inplace=True)

matches_hosted
#Conversion of counrtry names to ISO-3661-alpha-3 format for plotly

import pycountry



input_countries = matches_hosted['country']

change={"Netherlands Antilles":"Aruba","United Arab Republic":"Egypt","South Korea":"Korea, Republic of","Bolivia":"Bolivia, Plurinational State of","England":"United Kingdom","Iran":"Iran, Islamic Republic of","Venezuela":"Venezuela, Bolivarian Republic of","Yugoslavia":"Serbia","Ivory Coast":"Côte d'Ivoire","China PR":"Taiwan, Province of China","Vietnam":"Viet Nam","Russia":"Russian Federation"}



for i,country in enumerate(input_countries):

    if country in change:

        input_countries[i]=change[country]



countries = {}

for country in pycountry.countries:

    countries[country.name] = country.alpha_3



codes = [countries.get(country, 'Unknown code') for country in input_countries]

matches_hosted['codes']=codes

matches_hosted
#Cholorpleth of countries based on number of matches hosted

import plotly.graph_objects as go



plt.figure(figsize=(20,20))

fig = go.Figure(data=go.Choropleth(

    locations = matches_hosted['codes'],

    z = matches_hosted['counts'],

    text = "red",

    colorscale = 'reds',

    autocolorscale=False,

    reversescale=False,

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_title = 'Games hosted',

))



fig.update_layout(

    title_text='Games hosted by Country',

    geo=dict(

        showframe=False,

        showcoastlines=False,

        projection_type='equirectangular'

    ),

    annotations = [dict(

        x=0.55,

        y=0.1,

        xref='paper',

        yref='paper',

        showarrow = False

    )]

)



fig.show()
#Tournament that has had the most games

plt.figure(figsize=(20,5))

plt.title("Tournament games count")

sns.set_style("darkgrid")

sns.barplot(x=avg_goal.index,y=avg_goal['count'],color='#d597ce');

#Goals scored by each team

home_scored=major.pivot_table(['home_score'],columns=["home_team"],aggfunc=np.sum)

home_scored=home_scored.T

away_scored=major.pivot_table(['away_score'],columns=["away_team"],aggfunc=np.sum)

away_scored=away_scored.T

home_scored=home_scored.rename(columns={"home_score":"scored"})

away_scored=away_scored.rename(columns={"away_score":"scored"})

scored_df=home_scored.append(away_scored)

scored_df=scored_df.groupby(level=0).sum()

scored_df
#Goals conceded by each team

home_conceded=major.pivot_table(['away_score'],columns=["home_team"],aggfunc=np.sum)

home_conceded=home_conceded.T

away_conceded=major.pivot_table(['home_score'],columns=["away_team"],aggfunc=np.sum)

away_conceded=away_conceded.T

home_conceded=home_conceded.rename(columns={"away_score":"conceded"})

away_conceded=away_conceded.rename(columns={"home_score":"conceded"})

conceded_df=home_conceded.append(away_conceded)

conceded_df=conceded_df.groupby(level=0).sum()

conceded_df
# Number of games each team has played

home_games=major['home_team'].value_counts()

away_games=major['away_team'].value_counts()

games_count=home_games.append(away_games)

games_count=games_count.groupby(level=0).sum()
#Making a new table out of the data

countries_goals=pd.DataFrame(index=scored_df.index)

countries_goals['scored']=scored_df['scored']

countries_goals['conceded']=conceded_df['conceded']

countries_goals['games_played']=games_count

countries_goals['scored_avg']=countries_goals['scored']/countries_goals['games_played']

countries_goals['conceded_avg']=countries_goals['conceded']/countries_goals['games_played']

countries_goals
#Finding who won,drew or lost from every match 

home_wins_draws_loses=major[['home_team','result']]

home_wins_draws_loses.rename(columns={'home_team':'team'},inplace=True)

home_wins_draws_loses['result']=["win" if x=="home_win" else "draw" if x=="draw" else "lose" for x in home_wins_draws_loses['result']]

away_wins_draws_loses=major[['away_team','result']]

away_wins_draws_loses.rename(columns={'away_team':'team'},inplace=True)

away_wins_draws_loses['result']=["win" if x=="away_win" else "draw" if x=="draw" else "lose" for x in away_wins_draws_loses['result']]

wins_draws_loses_df=home_wins_draws_loses[['team','result']]

wins_draws_loses_df=wins_draws_loses_df.append(away_wins_draws_loses)

wins_draws_loses_df
#Aggregating the counts from the above table 

wins=wins_draws_loses_df.loc[(wins_draws_loses_df['result']=='win')]

wins=wins.groupby('team').count()

draws=wins_draws_loses_df.loc[(wins_draws_loses_df['result']=='draw')]

draws=draws.groupby('team').count()

loses=wins_draws_loses_df.loc[(wins_draws_loses_df['result']=='lose')]

loses=loses.groupby('team').count()



#Another option /major.pivot_table(index=['team','result'],aggfunc='count')['away_score']

#draws

wins

#loses
countries=pd.DataFrame(index=countries_goals.index)

countries['wins']=[wins.loc[x]['result'] if x in wins.index else 0 for x in countries.index]

countries['draws']=[draws.loc[x]['result'] if x in draws.index else 0 for x in countries.index]

countries['loses']=[loses.loc[x]['result'] if x in loses.index else 0 for x in countries.index]

countries['games_played']=countries_goals['games_played']

countries['win_percentage']=((countries['wins']/countries['games_played'])*100).round(2)

countries
#Somewhat of a metric for finding best teams

countries['weight']=countries['games_played']/100

countries['best']=(2*countries['win_percentage'])*countries['weight']
countries.loc[:,['wins','games_played','win_percentage','weight','best']].sort_values('best',ascending=False).head(30)
#Just checking the tail end of the data 

countries.loc[:,['wins','games_played','win_percentage','weight','best']].sort_values('best',ascending=False).tail(60)
#Goals scored

plot_data=countries_goals.sort_values("scored",ascending=False).head(10)

plt.figure(figsize=(15,10))

plt.title("Number of Goals scored")

sns.set_style('darkgrid')

sns.barplot(x=plot_data.index,y=plot_data['scored'],color="#39375b")
#Number of goals conceded 

plot_data=countries_goals.sort_values("conceded",ascending=False).head(10)

plt.figure(figsize=(15,10))

plt.title("Number of Goals conceded")

sns.set_style('darkgrid')

sns.barplot(x=plot_data.index,y=plot_data['conceded'],color="#39375b")
plot_data=countries.sort_values("win_percentage",ascending=False).head(10)

plt.figure(figsize=(15,10))

plt.title("Win Percentage")

sns.set_style('darkgrid')

sns.barplot(x=plot_data.index,y=plot_data['win_percentage'],color="#39375b")
plot_data=countries.sort_values("best",ascending=False).head(10)

plt.figure(figsize=(15,10))

plt.title("Custom Best team metric")

sns.set_style('darkgrid')

sns.barplot(x=plot_data.index,y=plot_data['best'],color="#39375b")