# import all libs
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import plotly.express as px
path = '../input/soccer/database.sqlite'
conn = sqlite3.connect(path)
tables = pd.read_sql("SELECT * FROM sqlite_master WHERE type='table';", conn)
tables
num_of_contests = pd.read_sql("""SELECT name, season, count(DISTINCT stage) as num_of_stages
                         FROM Match
                         LEFT JOIN League
                         ON Match.league_id = League.id
                         GROUP BY league_id, season 
                         LIMIT 10;""", conn)
num_of_contests
Average_player_score = pd.read_sql("""SELECT Country.name as country_name, League.name as league_name, season, avg(home_team_goal+away_team_goal) as avg_score
                                      FROM Match
                                      LEFT JOIN Country
                                      ON Country.id = Match.country_id
                                      LEFT JOIN League
                                      ON League.id = Match.league_id
                                      GROUP BY Match.country_id, season;""",conn)
Average_player_score
country_list = pd.read_sql("""SELECT name
                              FROM Country;""", conn)
season_list = pd.read_sql("""SELECT DISTINCT season
                        FROM Match;""", conn)
color_list = ['gold', 'olivedrab', 'darkred', 'dimgray', 'seagreen', 'slateblue', 'orange', 'darkgreen', 'indigo', 'lightpink', 'steelblue']
i = 0
for item in country_list.name:
    cur = Average_player_score[Average_player_score.country_name == item]
    X = cur.season
    Y = cur.avg_score
    plt.plot(X, Y, label=item, color=color_list[i])
    i += 1
plt.xticks(X, list(season_list.season), rotation=90)
plt.legend(loc='best',bbox_to_anchor=(0.9, 0.2, 0.5, 0.5))
plt.show()
increase_rate = pd.read_sql("""WITH weights AS(
                               SELECT '2009'AS 'year', 0.05 AS 'weight'
                               UNION
                               SELECT '2010'AS 'year', 0.07 AS 'weight'
                               UNION
                               SELECT '2011'AS 'year', 0.09 AS 'weight'
                               UNION
                               SELECT '2012'AS 'year', 0.11 AS 'weight'
                               UNION
                               SELECT '2013'AS 'year', 0.14 AS 'weight'
                               UNION
                               SELECT '2014'AS 'year', 0.16 AS 'weight'
                               UNION
                               SELECT '2015'AS 'year', 0.18 AS 'weight'
                               UNION
                               SELECT '2016'AS 'year', 0.20 AS 'weight'),
                               
                               basis AS (SELECT Country.name AS country_name, substr(season, 6, 5) AS basis_year, avg(home_team_goal+away_team_goal) AS total_goal_basis
                               FROM Match
                               LEFT JOIN Country
                               ON Country.id = Match.country_id
                               GROUP BY Match.country_id, season),
                               
                               end AS (SELECT Country.name AS country_name, substr(season, 0, 5) AS next_year, avg(home_team_goal+away_team_goal) AS total_goal_end
                               FROM Match
                               LEFT JOIN Country
                               ON Country.id = Match.country_id
                               GROUP BY Match.country_id, season)
                               
                               SELECT basis.country_name, sum((total_goal_end - total_goal_basis)/total_goal_end * weight) as avg_increase_rate
                               FROM basis 
                               LEFT JOIN end
                               ON basis.country_name = end.country_name 
                               AND basis.basis_year = end.next_year
                               LEFT JOIN weights
                               ON weights.year = basis.basis_year
                               GROUP BY basis.country_name
                               HAVING total_goal_end IS NOT NULL;
                               """,conn)
increase_rate
X = increase_rate.country_name
Y = increase_rate.avg_increase_rate
fig, ax = plt.subplots()
ax.scatter(X, Y)
for i, txt in enumerate(X):
    ax.annotate(txt, (X[i], Y[i]))
plt.xticks(["Country"])
plt.show()
player_api_id = 155782
player_query = """SELECT strftime('%Y', date) as year, round(avg(overall_rating),2) as avg_rate, round(avg(potential),2) as avg_potential
                  FROM Player_Attributes
                  WHERE player_api_id = {}
                  GROUP BY year;
                  """.format(player_api_id)
player = pd.read_sql(player_query, conn)
player
X = player.year
Y_rate = player.avg_rate
Y_potential = player.avg_potential
plt.plot(X, Y_rate, label = 'average rating')
plt.plot(X, Y_potential, label = 'average potential')
plt.legend(loc='best',bbox_to_anchor=(0.9, 0.2, 0.5, 0.5))
plt.show()
player_ability = pd.read_sql("""SELECT player_fifa_api_id as id, avg(heading_accuracy) AS head_acc , avg(curve) AS curve, avg(ball_control) AS ball_control, avg(acceleration) AS acceleration, avg(agility) AS agility, avg(balance) AS balance, avg(positioning) AS positioning, avg(vision) AS vision
                                FROM Player_Attributes
                                GROUP BY player_fifa_api_id;""", conn)
player_ability
# data standardize 
for col_name in player_ability.columns[1:]:
    col_max = max(player_ability[col_name])
    col_min = min(player_ability[col_name])
    player_ability[col_name] = player_ability[col_name].apply(lambda x: round((x-col_min)/(col_max - col_min),2))
player_ability
id = 233885
player_metric = player_ability[player_ability.id == id].values[0][1:]
col_name = player_ability.columns[1:]
df = pd.DataFrame(dict(r = player_metric,theta = col_name))
fig = px.line_polar(df, r='r', theta='theta', line_close=True)

fig.update_traces(fill='toself')
fig.show()