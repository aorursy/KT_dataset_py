import pandas

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Import of Ballon d'or data

df_ballon = pandas.read_csv("../input/ballon-d-or.csv", sep=",", header=0)

df_ballon.head()
df_ballon.info()
#Train data description

df_ballon.describe()
# Check for any NaN values

df_ballon.isnull().any()
#Filling the NaN by mean of the whole column values

df_ballon['points'].fillna(value = df_ballon['points'].mean(), inplace = True)
#Names of columns in DF

df_ballon.columns
#Choosen Parameters

chosen_columns = [

    'year',

    'rank',

    'player',

    'team',

    'points',

    'percentages'

]
#Main DF

df = pandas.DataFrame(df_ballon, columns = chosen_columns)

df.sample(5)
# Transferring rank into float and making an auxiliary variable called Winning_index

premier_league = ['Arsenal', 'Blackpool', 'Chelsea', 'Fulham', 'Liverpool', 'Manchester United', 'Newcastle United', 'Tottenham Hotspur', 'West Ham United', 'Wolverhampton Wanderers']



primera_division = ['Atlético Madrid', 'Barcelona', 'Real Madrid']



serie_a = ['Cagliari', 'Hellas Verona', 'Internazionale', 'Juventus', 'Milan']



ligue_1 = ['Bordeaux', 'Marseille', 'Nancy', 'Paris Saint-Germain', 'Saint-Étienne', 'Stade de Reims']



bundesliga = ['Bayern Munich', 'Borussia Dortmund', 'Borussia Mönchengladbach', 'Hamburg', 'Köln', 'Rot-Weiss Essen']





def club_to_league(df_value):

    try:

        element = df_value



        if element in premier_league:

            value = 'Premier League'

        elif element in primera_division:

            value = 'Primera Division'

        elif element in serie_a:

            value = 'Serie A'

        elif element in ligue_1:

            value = 'Ligue 1'

        elif element in bundesliga:

            value = 'Bundesliga'

        else: 

            value = 'Others'

    except ValueError:

        value = 0

    return value







def value_to_int(df_value):

    try:

        element = df_value



        if element == '1st':

            value = 1

        elif element == '2nd':

            value = 2

        else: 

            value = 3

    except ValueError:

        value = 0

    return value



def winning_index(df_value):

    try:

        element = df_value



        if element == '1st':

            value = 1

        else: 

            value = 0

    except ValueError:

        value = 0

    return value



df['League'] = df['team'].apply(club_to_league)

df['rank_float'] = df['rank'].apply(value_to_int)

df['Winning_index'] = df['rank'].apply(winning_index)

df.head()
# The best player per each club history by ranking points

df.iloc[df.groupby(df['team'])['points'].idxmax()][['team', 'player', 'points']]
#Last 5 Years Winners

df.loc[(df['year'] > 2013) & (df['Winning_index'] == 1)]
#Real Madrid Winners

df.loc[(df['team'] == "Real Madrid") & (df['Winning_index'] == 1)]
#Barcelona Winners

df.loc[(df['team'] == "Barcelona") & (df['Winning_index'] == 1)]
#Manchester United Winners

df.loc[(df['team'] == "Manchester United") & (df['Winning_index'] == 1)]
# Top five the most points collected Clubs

df_ts = df.groupby(['team'])['points'].sum().sort_values(ascending = False).head(5)

df_ts = pandas.DataFrame(data = df_ts, columns = ['points'])

index_list = list(df_ts.index)

df_ts.insert(loc = 1, column = 'Team', value = index_list)

sns.set(rc={'figure.figsize':(12,9)})

sns.set(style="whitegrid", palette="RdBu")

sns.set_color_codes("dark")

sns.barplot(x="Team", y="points", data=df_ts, hue='Team')

plt.title('Top 5 teams with highest points score')

# Top five the most rewarded Leagues by Reward Winning Index

df_leagues = df.groupby(['League'])['Winning_index'].sum().sort_values(ascending = False).head(5)

df_leagues = pandas.DataFrame(data = df_leagues, columns = ['Winning_index'])

index_list = list(df_leagues.index)

df_leagues.insert(loc = 1, column = 'League', value = index_list)



sns.barplot(x="League", y="Winning_index", data=df_leagues, hue='League', color="b")

plt.title('Most rewarded Leagues')

# Top five the most rewarded Clubs by Reward Winning Index

df_winning_team = df.groupby(['team'])['Winning_index'].sum().sort_values(ascending = False).head(5)

df_winning_team = pandas.DataFrame(data = df_winning_team, columns = ['Winning_index'])

index_list = list(df_winning_team.index)

df_winning_team.insert(loc = 1, column = 'Team', value = index_list)



sns.barplot(x="Team", y="Winning_index", data=df_winning_team, hue='Team', color="g")

plt.title('Top 5 teams with highest amout of rewards')
# Top five the most rewarded Players

df_winning = df.groupby(['player'])['Winning_index'].sum().sort_values(ascending = False).head(5)

df_winning = pandas.DataFrame(data = df_winning, columns = ['Winning_index'])

index_list = list(df_winning.index)

df_winning.insert(loc = 1, column = 'Player', value = index_list)





sns.barplot(x="Player", y="Winning_index", data=df_winning, hue='Player', color="grey")

plt.title('Top 5 players with highest amount of rewards')
# The most unquestionable victories

df_maxper = df.groupby(['player'])['percentages'].max().sort_values(ascending = False).head(5)

df_maxper = pandas.DataFrame(data = df_maxper, columns = ['percentages'])

index_list = list(df_maxper.index)

df_maxper.insert(loc = 1, column = 'Player', value = index_list)

sns.relplot(x="Player", y="percentages", hue="Player", size="percentages",

            sizes=(200, 400), alpha=.5, palette="dark",

            height=11, data=df_maxper, legend="brief")

plt.title('Most unquestionable victories by percentage')