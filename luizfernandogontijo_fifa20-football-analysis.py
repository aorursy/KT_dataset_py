import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns
df_fifa20 = pd.read_csv('../input/fifa-20-complete-player-dataset/players_20.csv')
df_fifa20.head()
df_fifa20.shape 
df_fifa20.describe()
df_fifa20.columns
bundesliga = [

  "1. FC Nürnberg", "1. FSV Mainz 05", "Bayer 04 Leverkusen", "FC Bayern München",

  "Borussia Dortmund", "Borussia Mönchengladbach", "Eintracht Frankfurt",

  "FC Augsburg", "FC Schalke 04", "Fortuna Düsseldorf", "Hannover 96",

  "Hertha BSC", "RB Leipzig", "SC Freiburg", "TSG 1899 Hoffenheim",

  "VfB Stuttgart", "VfL Wolfsburg", "SV Werder Bremen"

]



premierLeague = [

  "Arsenal", "Bournemouth", "Brighton & Hove Albion", "Burnley",

  "Cardiff City", "Chelsea", "Crystal Palace", "Everton", "Fulham",

  "Huddersfield Town", "Leicester City", "Liverpool", "Manchester City",

  "Manchester United", "Newcastle United", "Southampton", 

  "Tottenham Hotspur", "Watford", "West Ham United", "Wolverhampton Wanderers"

]



laliga = [

  "Athletic Club de Bilbao", "Atlético Madrid", "CD Leganés",

  "Deportivo Alavés", "FC Barcelona", "Getafe CF", "Girona FC", 

  "Levante UD", "Rayo Vallecano", "RC Celta", "RCD Espanyol", 

  "Real Betis", "Real Madrid", "Real Sociedad", "Real Valladolid CF",

  "SD Eibar", "SD Huesca", "Sevilla FC", "Valencia CF", "Villarreal CF"

]



seriea = [

  "Atalanta","Bologna","Cagliari","Chievo Verona","Empoli", "Fiorentina","Frosinone","Genoa",

  "Inter","Juventus","Lazio","Milan","Napoli","Parma","Roma","Sampdoria","Sassuolo","SPAL",

  "Torino","Udinese"

]



superlig = [

  "Akhisar Belediyespor","Alanyaspor", "Antalyaspor","Medipol Başakşehir FK","BB Erzurumspor","Beşiktaş JK",

  "Bursaspor","Çaykur Rizespor","Fenerbahçe SK", "Galatasaray SK","Göztepe SK","Kasimpaşa SK",

  "Kayserispor","Atiker Konyaspor","MKE Ankaragücü", "Sivasspor","Trabzonspor","Yeni Malatyaspor"

]



ligue1 = [

  "Amiens SC", "Angers SCO", "AS Monaco", "AS Saint-Étienne", "Dijon FCO", "En Avant de Guingamp",

  "FC Nantes", "FC Girondins de Bordeaux", "LOSC Lille", "Montpellier HSC", "Nîmes Olympique", 

  "OGC Nice", "Olympique Lyonnais","Olympique de Marseille", "Paris Saint-Germain", 

  "RC Strasbourg Alsace", "Stade Malherbe Caen", "Stade de Reims", "Stade Rennais FC", "Toulouse Football Club"

]



eredivisie = [

  "ADO Den Haag","Ajax", "AZ Alkmaar", "De Graafschap","Excelsior","FC Emmen","FC Groningen",

  "FC Utrecht", "Feyenoord","Fortuna Sittard", "Heracles Almelo","NAC Breda",

  "PEC Zwolle", "PSV","SC Heerenveen","Vitesse","VVV-Venlo","Willem II"

]



liganos = [

  "Os Belenenses", "Boavista FC", "CD Feirense", "CD Tondela", "CD Aves", "FC Porto",

  "CD Nacional", "GD Chaves", "Clube Sport Marítimo", "Moreirense FC", "Portimonense SC", "Rio Ave FC",

  "Santa Clara", "SC Braga", "SL Benfica", "Sporting CP", "Vitória Guimarães", "Vitória de Setúbal"

]
df_fifa20['league'] = df_fifa20['club'].apply(lambda row: 'bundesliga' if row in bundesliga 

                                              else ('premierLeague' if row in premierLeague else ('laliga' if row in laliga 

                                                                                                  else ('seriea' if row in seriea 

                                                                                                        else ('superlig' if row in superlig 

                                                                                                              else ('ligue1' if row in ligue1 

                                                                                                                   else ('eredivisie' if row in eredivisie 

                                                                                                                        else ('liganos' if row in liganos

                                                                                                                             else 'NaN'))))))))
df_fifa20 = df_fifa20[df_fifa20.league != 'NaN'] #take off the Nan data from league column
df_fifa20['value_eur'].fillna('0', inplace = True)
df_fifa20['value_eur'].isnull().sum()
df_fifa20[df_fifa20['league'] == 'premierLeague'].count()
df_fifa20[['value_eur', 'league']].groupby('league').mean()
def plot_histogram(league,title,column):

    df = df_fifa20.loc[df_fifa20['league']==league]

    sns.distplot(df['value_eur'], bins= 30, kde=True, rug=False)

    plt.xlabel(column)

    plt.ylabel("frequency")

    plt.title(title)

    return plt.show()
plot_histogram('premierLeague', 'Players values in Premier League', 'value_eur')
def plot_hist_subplots(atribute, kde):

    sns.set(style="white", palette="muted", color_codes=False)

    rs = np.random.RandomState(10)



    f, axes = plt.subplots(3, 3, figsize=(13, 13), sharex=True)

    sns.despine(right=True)



    sns.distplot(df_fifa20.loc[df_fifa20['league']=='bundesliga'][atribute], kde=kde, color="r", ax=axes[0, 0]).set_title("bundesliga")

    sns.distplot(df_fifa20.loc[df_fifa20['league']=='eredivisie'][atribute], kde=kde, color="k", ax=axes[0, 1]).set_title("eredivisie")

    sns.distplot(df_fifa20.loc[df_fifa20['league']=='laliga'][atribute], kde=kde, color="y", ax=axes[0, 2]).set_title("laliga")

    sns.distplot(df_fifa20.loc[df_fifa20['league']=='liganos'][atribute], kde=kde, color="g", ax=axes[1, 0]).set_title("liganos")

    sns.distplot(df_fifa20.loc[df_fifa20['league']=='ligue1'][atribute], kde=kde, color="m", ax=axes[1, 1]).set_title("ligue1")

    sns.distplot(df_fifa20.loc[df_fifa20['league']=='premierLeague'][atribute], kde=kde, color="b", ax=axes[1, 2]).set_title("premierLeague")

    sns.distplot(df_fifa20.loc[df_fifa20['league']=='seriea'][atribute], kde=kde, color="k", ax=axes[2, 0]).set_title("seriea")

    sns.distplot(df_fifa20.loc[df_fifa20['league']=='superlig'][atribute], kde=kde, color="c", ax=axes[2, 1]).set_title("superlig")

    #sns.distplot(df_fifa20.loc[df_fifa20['league']=='none'][atribute], kde=kde, color="k", ax=axes[2, 2]).set_title("none")





    return plt.tight_layout()
plot_hist_subplots('value_eur', True)
plot_hist_subplots('overall', True)
def plot_bars(atribute):

    df = df_fifa20[df_fifa20.league != 'NaN']

    df = df.groupby('league', as_index=False, sort=False).sum().sort_values(by=[atribute], ascending=False)

    return sns.barplot(x=atribute, y='league', data=df)
plot_bars('potential')
sklis_vec =['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 

             'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 

             'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 

             'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 

             'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 

             'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 

             'mentality_composure', 'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving', 

             'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']



name_sklis_vec = ['short_name', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 

                 'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 

                 'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 

                 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 

                 'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 

                 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 

                 'mentality_composure', 'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving', 

                 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']
df_fifa20_attributes_names = df_fifa20[name_sklis_vec]

df_fifa20_attributes_names = df_fifa20_attributes_names.dropna()

df_fifa20_attributes_names.head()
def plot_attributes(short_name):

    df_fifa20_attributes_names = df_fifa20[name_sklis_vec]

    df_fifa20_attributes_names = df_fifa20_attributes_names.dropna()

    df_fifa20_attributes_names = df_fifa20_attributes_names.loc[df_fifa20['short_name']=='Neymar Jr'].reset_index()

    df_fifa20_attributes_names_transpose = df_fifa20_attributes_names.T

    y = df_fifa20_attributes_names_transpose.iloc[2:].index

    x = df_fifa20_attributes_names_transpose[0].iloc[2:]

    plt.figure(figsize= (10,15))

    plt.barh(y,x)
plot_attributes('L. Messi')
def plot_attributes(short_name1, short_name2):

    

    df_fifa20_attributes_names1 = df_fifa20[name_sklis_vec]

    df_fifa20_attributes_names1 = df_fifa20_attributes_names1.dropna()

    df_fifa20_attributes_names1 = df_fifa20_attributes_names1.loc[df_fifa20['short_name']==short_name1].reset_index()

    df_fifa20_attributes_names_transpose1 = df_fifa20_attributes_names1.T

    y1 = df_fifa20_attributes_names_transpose1.iloc[2:].index

    x1 = df_fifa20_attributes_names_transpose1[0].iloc[2:]

    

    df_fifa20_attributes_names2 = df_fifa20[name_sklis_vec]

    df_fifa20_attributes_names2 = df_fifa20_attributes_names2.dropna()

    df_fifa20_attributes_names2 = df_fifa20_attributes_names2.loc[df_fifa20['short_name']==short_name2].reset_index()

    df_fifa20_attributes_names_transpose2 = df_fifa20_attributes_names2.T

    y2 = df_fifa20_attributes_names_transpose2.iloc[2:].index

    x2 = df_fifa20_attributes_names_transpose2[0].iloc[2:]

    

    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(12,12))



    axes[0].barh(y1, x1, align='center', color='red', zorder=10)

    

    axes[0].set(title=short_name1)



    axes[1].barh(y2, x2, align='center', color='black', zorder=10)



    axes[1].set(title=short_name2)

    

    axes[0].invert_xaxis()

    #axes[0].set(yticks=df_male_1['age'])

    #axes[0].yaxis.tick_right()

    

    plt.figure(figsize= (10,15))

    

    fig.tight_layout()

    fig.subplots_adjust(wspace=0.09)

    plt.show()
plot_attributes('Neymar Jr', "L. Messi")
plot_attributes('Cristiano Ronaldo', 'V. van Dijk')
def comp_attr(club, attribute1, attribute2):

    df_fifa20_select = df_fifa20.loc[df_fifa20['club']==club].reset_index()

    

    plt.figure(figsize= (10,10))

    p1 = sns.regplot(x=attribute1, y=attribute2, data=df_fifa20_select,

           fit_reg=True, marker = '+' # No regression line

           )   # Color by evolution stage

    

    for line in range(0,df_fifa20_select.shape[0]):

        p1.text(df_fifa20_select[attribute1][line], df_fifa20_select[attribute2][line], 

                df_fifa20_select.short_name[line], horizontalalignment='left', size='medium', color='black')



    return 
comp_attr('Liverpool', 'pace', 'dribbling')
comp_attr('Liverpool', 'defending_sliding_tackle', 'defending_marking')