import sqlite3

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy

from bs4 import BeautifulSoup

import datetime

sns.set()

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)
characters = pd.read_csv("../input/simpsons_characters.csv")
episodes = pd.read_csv("../input/simpsons_episodes.csv")
locations = pd.read_csv("../input/simpsons_locations.csv")
script = pd.read_csv("../input/simpsons_script_lines.csv",error_bad_lines=False)

# line 8084, 52607, 59910, 71801, 73539, 77230, 78953, 81138, 101154, 115438, 117573, 130610, 152970, 153017, 153018, 

# 154080, 154082, 154084, 154086, 154089, 154165, 156872 were skipped due to errors
# creating a connection to sql database, simpsons.sqlite

conn = sqlite3.connect("simpsons.sqlite")
characters.to_sql("characters", conn, index = False,if_exists = "replace")

episodes.to_sql("episodes", conn, index = False, if_exists = "replace")

locations.to_sql("locations", conn, index = False, if_exists = "replace")

script.to_sql("script", conn, index = False, if_exists = "replace")
pd.read_sql_query("""SELECT *

                     FROM episodes

                     LIMIT 1""",conn)
pd.read_sql_query("""SELECT MAX(CAST (original_air_date AS DATE)),

                            MIN(CAST (original_air_date AS DATE))

                     FROM episodes""",conn)
pd.read_sql_query("""SELECT MAX(season) AS "Total Seasons", 

                            MAX(number_in_season) AS "Max No. of Episodes in One Season", 

                            MAX(number_in_series) AS "Totoal No. of Episodes",

                            MIN(us_viewers_in_millions) AS "Min US Viewers in Millions for one episode",

                            MAX(us_viewers_in_millions) AS "Max US viewers in Millions for one episode",

                            ROUND(AVG(us_viewers_in_millions),2) AS "Average Number of Viewers in Millions",

                            MAX(imdb_rating) AS "Max imdb Rating",

                            MIN(imdb_rating) AS "Min imdb Rating",

                            ROUND(AVG(imdb_rating),2) AS "Avg imdb Rating",

                            ROUND(AVG(imdb_votes),2) AS "Avg imdb votes"

                     FROM episodes;""", conn)
graphing = pd.read_sql_query("""SELECT season,

                            MAX(number_in_season) AS "episodes",

                            MAX(us_viewers_in_millions) AS "viewer_high",

                            MIN(us_viewers_in_millions) AS "viewer_low",

                            AVG(us_viewers_in_millions) AS "viewer_avg",

                            MAX(imdb_rating) AS "imdb_max_rating",

                            MIN(imdb_rating) AS "imdb_min_rating",

                            AVG(imdb_rating) AS "imdb_avg_rating",

                            AVG(imdb_votes) AS "avg_voters"

                     FROM episodes

                     WHERE season < 28

                     GROUP BY season""", conn)

# season 28 was excluded cause it was only four episodes in the dataset
pd.read_sql_query("""SELECT MAX(number_in_series)/MAX(season) AS "avg_episodes_season"

                     FROM episodes

                     WHERE season < 28""", conn)
graph_viewer = pd.melt(graphing, id_vars=['season'], value_vars=['viewer_high', 'viewer_low','viewer_avg'])
graph_viewer.head(3)
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
graph_rating = pd.melt(graphing, id_vars = ['season'], value_vars = ['imdb_max_rating','imdb_min_rating','imdb_avg_rating'])
plt.figure(figsize=(20, 16), dpi= 80, facecolor='w', edgecolor='k')

plt.subplot(2,2,1)

sns.pointplot(x = "season", y = "episodes", data = graphing, markers="")

plt.title("Number of Episodes")

plt.subplot(2,2,2)

sns.pointplot(x = "season", y = "value", data = graph_viewer, hue = "variable", markers="")

plt.title("Viewership")

plt.subplot(2,2,3)

sns.pointplot(x = "season", y = "value", data = graph_rating, hue = "variable", markers="")

plt.title("IMDB Ratings")

plt.subplot(2,2,4)

sns.pointplot(x = "season", y = "avg_voters", data = graphing, markers="")

plt.title("Number of IMDB votes")

plt.show()
plt.figure(figsize=(20, 8), dpi= 80, facecolor='w', edgecolor='k')

plt.subplot(1,2,1)

sns.heatmap(episodes[["us_viewers_in_millions", "imdb_rating","imdb_votes"]].corr(), annot= True)

plt.title("No. of Viewers, IMDB Rating, No. of IMDB Reviewers")

plt.yticks(rotation=0)

plt.subplot(1,2,2)

sns.heatmap(graphing[["viewer_avg","imdb_avg_rating","avg_voters"]].corr(), annot= True)

plt.title("Correlation between Seasonal Viewers, IMDB Rating, and number of IMDB Reviewers")

plt.yticks(rotation=0)

plt.show()
plt.figure(figsize=(20, 16), dpi= 80, facecolor='w', edgecolor='k')

plt.subplot(2,2,1)

sns.regplot(x = "viewer_avg", y = "imdb_avg_rating", data = graphing, ci=False)

plt.title("Avg number of viewers and avg rating")

plt.subplot(2,2,2)

sns.regplot(x = "avg_voters", y = "imdb_avg_rating", data = graphing, ci=False)

plt.title("Avg number of reviewers and avg rating")

plt.subplot(2,2,3)

sns.regplot(x = "us_viewers_in_millions", y = "imdb_rating", data = episodes, ci= False)

plt.title("number of viewers and rating per episode")

plt.subplot(2,2,4)

sns.regplot(x = "imdb_votes", y = "imdb_rating", data = episodes, ci = False)

plt.title("number of imdb reviews and imdb rating")

plt.show()
# Low correlation between number of episodes and avg_rating

graphing[["episodes","imdb_avg_rating"]].corr()
plt.figure(figsize=(10, 8), dpi= 80, facecolor='w', edgecolor='k')

sns.regplot(x = "episodes", y = "imdb_avg_rating", data = graphing, ci=False)

plt.title("No. of Episodes vs. avg rating")

plt.show()
date = pd.read_sql_query("""SELECT original_air_date, 

                                   title,

                                   season

                            FROM episodes""", conn)
date["original_air_date"] = pd.to_datetime(date["original_air_date"], format = "%Y-%m-%d")
date["weekday"] = date["original_air_date"].dt.weekday
date["weekday"].value_counts()

# 0 monday and 6 sunday
air_date = pd.pivot_table(date, values = "original_air_date", index = "season", columns="weekday", aggfunc = "count")
air_date[0] = 0

air_date[5] = 0
air_date_graph = air_date[[0,1,2,3,4,5,6]]

air_date_graph = air_date_graph.rename(columns = {0:"Monday",1:"Tuesday", 2:"Wednesday", 3:"Thursday", 4: "Friday", 5: "Saturday", 6:"Sunday"})

air_date_graph = air_date_graph.fillna(0)
plt.figure(figsize=(10, 8), dpi= 80, facecolor='w', edgecolor='k')

sns.set_style("whitegrid")

sns.heatmap(air_date_graph, cmap="Blues")

plt.title("Air Date")

plt.show()
most_popular_episodes = pd.read_sql_query("""SELECT title, 

                            original_air_date, 

                            season, 

                            number_in_season,

                            us_viewers_in_millions,

                            SUBSTR(original_air_date, 1, 4) AS year,

                            SUBSTR(original_air_date, 6, 2) AS month,

                            SUBSTR(original_air_date, 9, 2) AS date

                     FROM episodes

                     GROUP BY season

                     HAVING us_viewers_in_millions = MAX(us_viewers_in_millions);""", conn)
most_popular_episodes[["title","month"]].groupby("month").agg("count").reset_index().sort_values(["title"])
most_popular_episodes[["title","original_air_date","season","number_in_season","month","year"]].sort_values("month")
characters_gender = pd.read_sql_query("""SELECT * FROM characters WHERE gender IS NOT NULL """, conn)
pd.read_sql_query("""SELECT COUNT(normalized_name) FROM characters

                     UNION ALL

                     SELECT COUNT(normalized_name) FROM characters WHERE gender IS NULL;""", conn)
characters_gender[["id","gender"]].groupby("gender").agg("count")
gender_group = characters_gender[["id","gender"]].groupby("gender").agg("count").reset_index()
gender_group["percent"] = round(gender_group["id"]/np.sum(gender_group["id"])*100,1)
plt.figure(figsize=(9, 8), dpi= 80, facecolor='w', edgecolor='k')

sns.barplot(x="gender", y="id", data=gender_group, order = ["m","f"])

plt.title("Gender Distribution")

plt.show()
script_with_new_id = pd.read_sql_query("""SELECT CASE WHEN INSTR(CAST(character_id AS VARCHAR),'.')>0 THEN SUBSTR(character_id ,1, INSTR(CAST(character_id AS VARCHAR),'.')-1)

                                                 ELSE character_id END as new_id,

                                                 CASE WHEN INSTR(CAST(location_id AS VARCHAR),'.')>0 THEN SUBSTR(location_id ,1, INSTR(CAST(location_id AS VARCHAR),'.')-1)

                                                 ELSE location_id END as new_loc_id,

                                                 *

                                          FROM script""", conn)
script_with_new_id.to_sql("script", conn, index = False, if_exists = "replace")
pd.read_sql_query("""SELECT gender, COUNT(*)

                     FROM (SELECT a.new_id, 

                            a.episode_id,

                            b.gender

                     FROM script a

                     LEFT JOIN (SELECT id, gender FROM characters) b

                     ON a.new_id = b.id)

                     GROUP BY gender""", conn)
(31212+87227)/(31212+87227+39809)*100
top_10_lines = pd.read_sql_query("""SELECT new_id, name, gender, lines

                     FROM (SELECT new_id, count(id) AS lines 

                     FROM script 

                     WHERE new_id IS NOT NULL

                     GROUP BY new_id 

                     ORDER BY lines DESC) a

                     LEFT JOIN (SELECT id, name, gender FROM characters) b

                     ON a.new_id = b.id

                     LIMIT 10""", conn)
plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

sns.reset_defaults()

sns.barplot(x="lines", y="name", data=top_10_lines, orient= 'h', hue = "gender")

plt.title("Top 10 characters with the most lines")

plt.show()
pd.read_sql_query("""SELECT season, id, number_in_season, number_in_series

                     FROM episodes

                     WHERE id = 500""", conn)
pd.read_sql_query("""SELECT MIN(episode_id), MAX(episode_id) FROM script""", conn)
# calculating the number of lines by each of the member of the simpson family

line_len = pd.read_sql_query("""SELECT episode_id,  SUM(CASE WHEN new_id = '2' THEN 1 ELSE 0 END) AS homer_lines,

                            SUM(CASE WHEN new_id = '1' THEN 1 ELSE 0 END) AS marge_lines,

                            SUM(CASE WHEN new_id = '8' THEN 1 ELSE 0 END) AS bart_lines,

                            SUM(CASE WHEN new_id = '9' THEN 1 ELSE 0 END) AS lisa_lines,

                            SUM(CASE WHEN new_id = '2' THEN word_count ELSE 0 END) AS homer_len,

                            SUM(CASE WHEN new_id = '1' THEN word_count ELSE 0 END) AS marge_len,

                            SUM(CASE WHEN new_id = '8' THEN word_count ELSE 0 END) AS bart_len,

                            SUM(CASE WHEN new_id = '9' THEN word_count ELSE 0 END) AS lisa_len,

                            COUNT(*) AS total, 

                            SUM(word_count) AS total_len

                     FROM script

                     GROUP BY episode_id""", conn)
# calculate the percentage of lines for each characters

line_len["homer_perc"] = round(line_len["homer_lines"]/line_len["total"]*100,2)

line_len["marge_perc"] = round(line_len["marge_lines"]/line_len["total"]*100,2)

line_len["bart_perc"] = round(line_len["bart_lines"]/line_len["total"]*100,2)

line_len["lisa_perc"] = round(line_len["lisa_lines"]/line_len["total"]*100,2)
# finding out who has the most lines in each episode

line_len["max_perc"] = line_len[["homer_perc","marge_perc","bart_perc","lisa_perc"]].apply(lambda x: max(x), axis = 1)
def compare(df):

    if df["max_perc"] == df["homer_perc"]:

        return "homer"

    elif df["max_perc"] == df["marge_perc"]:

        return "marge"

    elif df["max_perc"] == df["lisa_perc"]:

        return "lisa"

    elif df["max_perc"] == df["bart_perc"]:

        return "bart"
line_len["featuring"] = line_len.apply(lambda x: compare(x), axis = 1)
plt.figure(figsize=(20, 7.5), dpi= 80, facecolor='w', edgecolor='k')

plt.subplot(1,2,1)

sns.heatmap(line_len[["homer_lines","marge_lines","bart_lines","lisa_lines"]].corr(), annot=True)

plt.title("Simpson Family no. of Lines")

plt.subplot(1,2,2)

sns.heatmap(line_len[["homer_len","marge_len","bart_len", "lisa_len"]].corr(), annot=True)

plt.title("Simpson Family Line Lengths")

plt.show()
plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

plt.subplot(2,3,1)

sns.regplot(x = "homer_lines", y = "marge_lines", data = line_len, ci = False)

plt.title("Homer vs. Marge")

plt.subplot(2,3,2)

sns.regplot(x = "homer_lines", y = "lisa_lines", data = line_len, ci = False)

plt.title("Homer vs. Lisa")

plt.subplot(2,3,3)

sns.regplot(x = "homer_lines", y = "bart_lines", data = line_len, ci = False)

plt.title("Homer vs. Bart")

plt.subplot(2,3,4)

sns.regplot(x = "marge_lines", y = "lisa_lines", data = line_len, ci = False)

plt.title("Marge vs. Lisa")

plt.subplot(2,3,5)

sns.regplot(x = "marge_lines", y = "bart_lines", data = line_len, ci = False)

plt.title("Marge vs. Bart")

plt.subplot(2,3,6)

sns.regplot(x = "lisa_lines", y = "bart_lines", data = line_len, ci = False)

plt.title("Lisa vs. Bart")

plt.show()
# parent pair - homer + marge, children pair - lisa + bart. if the parent pair had more lines than the children pair

# I consider the episode to be featuring the parents. female_key = lisa + marge, male_ key = bart + homer. If an 

# episodes had more combined lines from the female_key, I considered this episode to feature the mother daughtor pair,

# and vice versa.

line_len["parent"] = line_len["homer_lines"] + line_len["marge_lines"]

line_len["children"] = line_len["bart_lines"] + line_len["lisa_lines"]

line_len["parent_feature"] = line_len["parent"] > line_len["children"]

line_len["female_key"] = line_len["marge_lines"] + line_len["lisa_lines"]

line_len["male_key"] = line_len["homer_lines"] + line_len["bart_lines"]

line_len["female_feature"] = line_len["female_key"] > line_len["male_key"]
line_len["parent_feature"].value_counts()
line_len["female_feature"].value_counts()
line_len.to_sql("line_len", conn, index = False, if_exists='replace')
# calculating per_female - percentage of lines that belonged to a female character

gender = pd.read_sql_query("""SELECT episode_id, SUM(CASE WHEN gender = 'm' THEN 1 ELSE 0 END) AS male_no, 

                     SUM(CASE WHEN gender = 'f' THEN 1 ELSE 0 END) AS female_no, 

                     COUNT(*) AS total_line,

                     ROUND(SUM(CASE WHEN gender = 'f' THEN 1 ELSE 0 END)/CAST(COUNT(*) AS FLOAT),2)*100 AS per_female

                     FROM(SELECT a.episode_id, a.new_id, b.gender

                     FROM (SELECT episode_id, new_id FROM script) a

                     LEFT JOIN (SELECT id, gender FROM characters) b

                     ON a.new_id = b.id)

                     GROUP BY episode_id""", conn)
plt.figure(figsize=(15, 10), dpi= 80, facecolor='w', edgecolor='k')

plt.hist(x = gender["per_female"], bins = 13)

plt.xlabel("Percent of female characters")

plt.ylabel("Count")

plt.title("Histogram of percent of Female Characters")

plt.show()
gender.to_sql("gender", conn, index=False, if_exists = 'replace')
total = pd.read_sql_query("""SELECT *

                     FROM (SELECT a.episode_id, a.male_no, a.female_no, a.total_line, a.per_female, b.season, 

                            b.number_in_season, b.us_viewers_in_millions, b.imdb_rating

                     FROM gender a

                     LEFT JOIN (SELECT id, season, number_in_season, us_viewers_in_millions, imdb_rating

                     FROM episodes) b

                     ON a.episode_id = b.id) c

                     LEFT JOIN (SELECT episode_id, homer_lines, marge_lines, bart_lines, lisa_lines, total, 

                                       homer_perc, marge_perc, bart_perc, lisa_perc, featuring, parent_feature,

                                       female_feature

                                FROM line_len) d

                     ON c.episode_id = d.episode_id""", conn)
total["featuring"].value_counts()

# Homer had the most lines in 368 episoes followed by Bart (90), Lisa (62), and Marge (44).
sns.boxplot(total["imdb_rating"], total["featuring"],order=["bart","homer","lisa","marge"])

plt.show()
def yes_no(x):

    if x == 1:

        return "Yes"

    else:

        return "No"

total["parent_yes"] = total["parent_feature"].apply(lambda x : yes_no(x))

total["female_yes"] = total["female_feature"].apply(lambda x: yes_no(x))
sns.boxplot(total["imdb_rating"], total["parent_yes"])

plt.show()
sns.boxplot(total["imdb_rating"], total["female_yes"], order = ["Yes","No"])

plt.show()
featuring = total.pivot_table(values = "male_no", index = "season", columns = "featuring", aggfunc = "count").fillna(0).reset_index()
featuring_prop = pd.merge(featuring, graphing[["season", "episodes"]], how = "inner", on = "season")

featuring_prop["bart_prop"] = round(featuring_prop["bart"]/featuring_prop["episodes"]*100, 2)

featuring_prop["homer_prop"] = round(featuring_prop["homer"]/featuring_prop["episodes"]*100, 2)

featuring_prop["lisa_prop"] = round(featuring_prop["lisa"]/featuring_prop["episodes"]*100, 2)

featuring_prop["marge_prop"] = round(featuring_prop["marge"]/featuring_prop["episodes"]*100, 2)
pd.read_sql_query("""SELECT MAX(episode_id), MIN(episode_id) FROM script""", conn)
pd.read_sql_query("""SELECT * FROM episodes WHERE number_in_series = 568""", conn)

# the script only contains data up to season 26 number 16 - exclude season 26 in the graphs
featuring_prop_25 = featuring_prop[featuring_prop["season"] < 26]
plt.figure(figsize=(20, 18), dpi= 80, facecolor='w', edgecolor='k')

plt.subplot(2,2,1)

sns.pointplot(x = "season", y = "homer_prop", data = featuring_prop_25, markers="")

plt.title("Homer-featured Episodes")

plt.subplot(2,2,2)

sns.pointplot(x = "season", y = "marge_prop", data = featuring_prop_25, markers="")

plt.title("Marge-featured Episodes")

plt.subplot(2,2,3)

sns.pointplot(x = "season", y = "lisa_prop", data = featuring_prop_25, markers="")

plt.title("Lisa-featured Episodes")

plt.subplot(2,2,4)

sns.pointplot(x = "season", y = "bart_prop", data = featuring_prop_25, markers="")

plt.title("bart-featured Episodes")

plt.show()
feature_graph = featuring_prop_25[["season", "homer_prop","marge_prop","bart_prop","lisa_prop"]]

feature_graph = feature_graph.set_index("season")
plt.figure(figsize=(10, 5), dpi= 80, facecolor='w', edgecolor='k')

sns.heatmap(feature_graph.T, cmap="Blues")

plt.yticks(rotation=0)

plt.xticks(rotation=0)

plt.show()
season_info = total[["season","us_viewers_in_millions","imdb_rating"]].groupby("season").agg("mean").reset_index()
all_season = pd.merge(featuring_prop_25, season_info, how="inner", on = "season")
plt.figure(figsize=(20, 8), dpi= 80, facecolor='w', edgecolor='k')

sns.regplot(x = "homer_prop", y = "imdb_rating", data = all_season, ci= False)

plt.title("Homer-featured Episodes")

plt.show()
all_season[["homer_prop","imdb_rating"]].corr()
plt.figure(figsize=(20, 8), dpi= 80, facecolor='w', edgecolor='k')

sns.regplot(x = "bart_prop", y = "imdb_rating", data = all_season, ci= False)

plt.title("Bart-featured Episodes")

plt.show()
all_season[["bart_prop","imdb_rating"]].corr()
plt.figure(figsize=(20, 8), dpi= 80, facecolor='w', edgecolor='k')

sns.regplot(x = "marge_prop", y = "imdb_rating", data = all_season, ci= False)

plt.title("Marge-featured Episodes")

plt.show()
all_season[["marge_prop","imdb_rating"]].corr()
plt.figure(figsize=(20, 8), dpi= 80, facecolor='w', edgecolor='k')

sns.regplot(x = "lisa_prop", y = "imdb_rating", data = all_season, ci= False)

plt.title("Lisa-featured Episodes")

plt.show()
all_season[["lisa_prop","imdb_rating"]].corr()
top_10_locations = pd.read_sql_query("""SELECT a.new_loc_id, a.line_counts, b.normalized_name

                     FROM (SELECT new_loc_id, 

                                  COUNT(new_id) AS line_counts

                     FROM script

                     GROUP BY new_loc_id

                     ORDER BY line_counts DESC) a 

                     LEFT JOIN locations b

                     ON a.new_loc_id = b.id

                     LIMIT 10""", conn)
plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

sns.barplot(x="normalized_name", y="line_counts", data=top_10_locations)

plt.title("Top 10 locations with the most lines")

plt.show()