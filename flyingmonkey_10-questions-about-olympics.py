import pandas as pd
from matplotlib import pyplot as plt
path = ('../input/athlete_events.csv')
athlete = pd.read_csv(path)
athlete = athlete[athlete['Season'] == 'Summer']
def functioncountry(df):
    athlete_winners = athlete[athlete.Medal.notnull()]
    winner_team = athlete_winners.groupby('Team').size().to_frame('medal_count')
    winner_team = winner_team.reset_index().sort_values('medal_count', ascending=False)
    top_10_winner = winner_team.head(10)
    result=top_10_winner
    return result

def plotfunction(abcd):
    abcd.plot.bar(x= 'Team', y= 'medal_count')
    plt.title("The country who won most medal")
    plt.xlabel("Team")
    plt.ylabel("medal_count")
    plt.show()
    
teammedal= functioncountry(athlete)
plotfunction(teammedal)
def get_medals_in_one_year_for_country(df):
    athlete_winners = athlete[athlete.Medal.notnull()]
    winner_teamYear = athlete_winners.groupby(['Year', 'Team']).size().to_frame('medal_count').reset_index()
    winner_teamYear = winner_teamYear.sort_values('medal_count', ascending=False)
    result=winner_teamYear.head(10)
    return result

get_medals_in_one_year_for_country(athlete)
def get_medals_per_year_for_country(df, teamName):
    athlete_winners = df[df.Medal.notnull()]
    winner_teamYear = athlete_winners.groupby(['Year', 'Team']).size().to_frame('medal_count').reset_index()
    winner_teamYear = winner_teamYear.sort_values('medal_count', ascending=False)
    result = winner_teamYear[ winner_teamYear.Team == teamName]
    result = result.sort_values('Year')
    return result

def plotfunction(abcd):
    plt.plot(abcd.Year,abcd.medal_count)
    plt.title("Year wise medal for United states")
    plt.xlabel("Year")
    plt.ylabel("Medal_count")
    plt.show

us = get_medals_per_year_for_country(athlete, 'United States')
plotfunction(us)
def functionhost(df):
    athlete_city = df.groupby(['Year','City'])['City'].size()
    athlete_city = athlete_city.to_frame('count') 
    athlete_city = athlete_city.reset_index()
    hostcity = athlete_city['City'].value_counts(dropna=False)
    hostcity = hostcity.to_frame('count')
    result = hostcity.head(5).reset_index().rename(columns={'index':'city'})
    return result

def plotfunction(abcd):
    abcd.plot.bar(x= 'city', y= 'count')
    plt.title("Top 5 city have hosted most olympics")
    plt.xlabel("city")
    plt.ylabel("count")
    plt.show()
    
Host = functionhost(athlete)
plotfunction(Host)
events = athlete['Event'].nunique()
events
def functionplayer (df):
    athlete_player = athlete.groupby(['Year','Sex']).size().unstack()
    athlete_player.plot()
    plt.xlabel("Year")
    plt.ylabel("count")
    result = plt.show()
    return result

functionplayer(athlete)
def functionplayer(df, sex):
    Player = athlete[athlete['Sex'] == sex]
    result= Player.groupby('Event').size().sort_values(ascending=False).head(10)
    return  result
functionplayer(athlete, "M")
def functionplayer(df, sex):
    Player = athlete[athlete['Sex'] == sex]
    result = Player.groupby('Event').size().sort_values(ascending=False).head(10)
    return  result
functionplayer(athlete, "F")
def functionlastevent (df):
    lastevents = athlete[['Event', 'Year']]
    lastevents = lastevents.sort_values(['Event','Year'], ascending=True)
    lastevents = lastevents.drop_duplicates(subset=['Event'], keep='first')
    lastevents = lastevents.sort_values(["Year"], ascending= False)
    result = lastevents.head(5)
    return result

functionlastevent (athlete)
def functioneventyear (df):
    lastevents = athlete[['Event', 'Year']]
    lastevents = lastevents.sort_values(['Event','Year'], ascending=False)
    lastevents = lastevents.drop_duplicates(subset=['Event'], keep='first')
    lastevents = lastevents.sort_values(["Year"], ascending= False)
    lasteventsyear = lastevents["Year"]
    lasteventsyear = lasteventsyear.max(axis=0)
    result = lastevents[lastevents.Year != 2016].head(10)
    return result

functioneventyear(athlete)
