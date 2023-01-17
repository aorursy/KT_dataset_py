import numpy as np 

import pandas as pd 

import os



data_dir = '../input/data_competition/data_competition'



relation_users = pd.read_csv(data_dir + "/UserUser.txt", sep="\t", header=None)

relation_users.columns = ["follower", "followed"]



labels_training = pd.read_csv(data_dir + "/labels_training.txt", sep=",")

labels_training.columns = ["news", "label"]



news_users = pd.read_csv(data_dir + "/newsUser.txt", sep="\t", header=None)

news_users.columns = ["news", "user", "times"]
import sqlite3

conn = sqlite3.connect('test.db')

c = conn.cursor()



### REMOVE EXISTING DATA TABLES

print("Dropping tables")

c.execute("DROP TABLE IF EXISTS users")

c.execute("DROP TABLE IF EXISTS news")

c.execute("DROP TABLE IF EXISTS news_labels")

c.execute("DROP TABLE IF EXISTS news_propagation")

c.execute("DROP TABLE IF EXISTS followers")

print("Creating tables")
### users DATA TABLE ##########################################################



sql = '''

CREATE TABLE IF NOT EXISTS users (

 userID INT

 )

'''

c.execute(sql)

print("users DT created")
### news DATA TABLE ###########################################################



sql = '''

CREATE TABLE IF NOT EXISTS news (

 newsID INT,

 newsTitle TEXT,

 newsText TEXT,

 PRIMARY KEY (newsID)

 )

'''

c.execute(sql)

print("news DT created")
### news_labels DATA TABLE ####################################################



sql = '''

CREATE TABLE IF NOT EXISTS news_labels (

 newsID INT,

 label TEXT,

 PRIMARY KEY (newsID),

 FOREIGN KEY (newsID) REFERENCES news(newsID)

 )

'''

c.execute(sql)

print("news_labels DT created")
### news_propagation DATA TABLE ###############################################



sql = '''

CREATE TABLE IF NOT EXISTS news_propagation (

 newsID INT,

 userID INT,

 propagCount INT,

 PRIMARY KEY (newsID, userID),

 FOREIGN KEY (newsID) REFERENCES news(newsID),

 FOREIGN KEY (userID) REFERENCES users(userID)

 )

'''

c.execute(sql)

print("news_propagation DT created")
### follower DATA TABLE #######################################################



sql = '''

CREATE TABLE IF NOT EXISTS followers (

 userID INT,

 userID_followed INT,

 PRIMARY KEY (userID, userId_followed),

 FOREIGN KEY (userID) REFERENCES users(userID),

 FOREIGN KEY (userId_followed) REFERENCES users(userID)

 )

'''

c.execute(sql)

print("follower DT created")
### INSERT DATA IN THE users DATA TABLES ######################################



list_unique_user_id = pd.DataFrame(list(np.union1d(relation_users.follower, relation_users.followed)))



list_unique_user_id.columns = ["userID"]

list_unique_user_id.to_sql(name='users', con=conn, if_exists='append', index = False)



#for id_user in list_unique_user_id[0:10] :

#    print("inserting data...")

#    c.execute("INSERT INTO users (userID) VALUES (%s)" % id_user)
### INSERT DATA IN THE news DATA TABLES #######################################



list_news = []

for filename in os.listdir(data_dir + "/news/training/"):

    id_news = filename.split(".")[0]

    with open (data_dir + "/news/training/" + filename, "r", encoding="utf8") as myfile:

        news = myfile.readlines()

    news = [x.strip() for x in news] 

    title_news = news[0]

    text_news = ' '.join(news[2:len(news)])

    list_news.append([id_news, title_news, text_news])

    

for filename in os.listdir(data_dir + "/news/test/"):

    id_news = filename.split(".")[0]

    with open (data_dir + "/news/test/" + filename, "r", encoding="utf8") as myfile:

        news = myfile.readlines()

    news = [x.strip() for x in news] 

    title_news = news[0]

    text_news = ' '.join(news[2:len(news)])

    list_news.append([id_news, title_news, text_news])



list_news = pd.DataFrame(list_news)



list_news.columns = ["newsID", "newsTitle", "newsText"]

list_news.to_sql(name='news', con=conn, if_exists='append', index = False)
### INSERT DATA IN THE news_labels DATA TABLES ################################



labels_training["label"] = labels_training["label"].astype(str)



for filename in os.listdir(data_dir + "/news/test/"):

    id_news = filename.split(".")[0]

    label_news = 'unlabelled'

    labels_training.loc[len(labels_training.index)+1] = [id_news, label_news]



labels_training = pd.DataFrame(labels_training)



labels_training.columns = ["newsID", "label"]

labels_training.to_sql(name='news_labels', con=conn, if_exists='append', index = False)
### INSERT DATA IN THE news_propagation DATA TABLES ###########################



news_users.columns = ["newsID", "userID", "propagCount"]

news_users.to_sql(name='news_propagation', con=conn, if_exists='append', index = False)
### INSERT DATA IN THE follower DATA TABLES ###################################



relation_users.columns = ["userID", "userID_followed"]

relation_users.to_sql(name='followers', con=conn, if_exists='append', index = False)
conn.commit()
instruc_nb_rows = {

 "Number of rows of users": 

    ''' 

    SELECT COUNT(userID)

    FROM users 

    ''',

 "Number of rows of news": 

    ''' 

    SELECT COUNT(*) 

    FROM news ''',

 "Number of rows of news_labels": 

    ''' 

    SELECT COUNT(*) 

    FROM news_labels 

    ''',

 "Number of rows of news_propagation": 

    ''' 

    SELECT COUNT(*) 

    FROM news_propagation 

    ''',

 "Number of rows of followers": 

    ''' 

    SELECT COUNT(*) 

    FROM followers 

    '''

 }



instructionSequence = ["Number of rows of users", 

                       "Number of rows of news", 

                       "Number of rows of news_labels", 

                       "Number of rows of news_propagation", 

                       "Number of rows of followers"]

for instruction in instructionSequence:

    c.execute(instruc_nb_rows[instruction])

    print(instruction)

    print(c.fetchone()[0])
instruc_stat_des = {

 "Number of users that have shared at least one news": 

    ''' 

    SELECT COUNT(DISTINCT userID) 

    FROM users 

    ''',

 "Number of news that have been shared at least one time": 

    ''' 

    SELECT COUNT(DISTINCT newsID) 

    FROM news 

    ''',

 "Number of news per label": 

    ''' 

    SELECT label, COUNT(label) 

    FROM news_labels 

    GROUP BY label

    ''',

 "Number of users with more than 500 followers" : 

    ''' 

    SELECT COUNT(DISTINCT userID) 

    FROM users 

    JOIN (

        SELECT * 

        FROM (

            SELECT userID_followed, COUNT(userID_followed) AS nb_follower 

            FROM followers 

            GROUP BY userID_followed

            ) 

        WHERE nb_follower > 500

        ) AS most_followed 

    ON most_followed.userID_followed = users.userID 

    ''',

 "Number of users with less than 5 followers" : 

    ''' 

    SELECT COUNT(DISTINCT userID) 

    FROM users 

    JOIN (

        SELECT * 

        FROM (

            SELECT userID_followed, COUNT(userID_followed) AS nb_follower 

            FROM followers 

            GROUP BY userID_followed

            ) 

        WHERE nb_follower < 6

        ) AS one_followed 

    ON one_followed.userID_followed = users.userID 

    ''',

 "Number of users that follow more than 500 times" : 

    ''' 

    SELECT COUNT(DISTINCT userID) 

    FROM users 

    JOIN (

        SELECT * 

        FROM (

            SELECT userID AS userID0, COUNT(userID) AS nb_follow 

            FROM followers 

            GROUP BY userID0

            ) 

        WHERE nb_follow > 500) AS most_follow 

    ON most_follow.userID0 = users.userID 

    ''',

 "Number of users that follow at least one other user": 

    ''' 

    SELECT COUNT(DISTINCT userID) 

    FROM followers 

    ''',

 "Number of users that have at least one follower": 

    ''' 

    SELECT COUNT(DISTINCT userID_followed) 

    FROM followers 

    '''

 }



instructionSequence = ["Number of news per label",

                       "Number of news that have been shared at least one time",

                       "Number of users that have shared at least one news",

                       "Number of users that follow more than 500 times",

                       "Number of users that follow at least one other user",

                       "Number of users with more than 500 followers",

                       "Number of users that have at least one follower",

                       "Number of users with less than 5 followers"]

for instruction in instructionSequence:

    c.execute(instruc_stat_des[instruction])

    print(instruction)

    if instruction == "Number of news per label" :

        print(c.fetchall())

    else :

        print(c.fetchall()[0][0])
### STATISTICS ON SHARING ############################################



instruc_stat_propa = {

 "Mean of number of sharing": 

    ''' 

    SELECT CAST(AVG(propagCount) as int) 

    FROM news_propagation 

    ''',

 "Median of number of sharing": 

    ''' 

    SELECT propagCount 

    FROM news_propagation  

    ORDER BY propagCount  

    LIMIT 1  

    OFFSET (

        SELECT COUNT(*) 

        FROM news_propagation

        ) / 2 

    ''',

 "Distribution of number of sharing": 

    ''' 

    SELECT propagCount, COUNT(*) propa_count 

    FROM news_propagation 

    GROUP BY propagCount 

    ORDER BY propagCount ASC 

    '''

 }



instructionSequence = ["Mean of number of sharing", 

                       "Median of number of sharing", 

                       "Distribution of number of sharing"]

for instruction in instructionSequence:

    c.execute(instruc_stat_propa[instruction])

    print(instruction)

    if instruction == "Distribution of number of sharing" :

        print(c.fetchall()[0:10])

    else :

        print(c.fetchall())





instruc_stat_label = {

 "Distribution by label for the news shared by 10 most sharing": 

    ''' 

    SELECT label, COUNT(label) 

    FROM (

        SELECT * 

        FROM news_labels 

        JOIN (

            SELECT * 

            FROM news_propagation 

            JOIN (

                SELECT userID, SUM(propagCount) AS sum_propa 

                FROM news_propagation 

                GROUP BY userID 

                ORDER BY sum_propa DESC 

                LIMIT 10

                ) AS nb_share_table 

            ON news_propagation.userID = nb_share_table.userID

            ) AS nb_propa_10share 

        ON news_labels.newsID = nb_propa_10share.newsID 

        ) 

    GROUP BY label 

    ''',

 "Distribution by label for the news shared by 100 most sharing": 

    ''' 

        SELECT label, COUNT(label) 

        FROM (

            SELECT * 

            FROM news_labels 

            JOIN (

                SELECT * 

                FROM news_propagation 

                JOIN (

                    SELECT userID, SUM(propagCount) AS sum_propa 

                    FROM news_propagation 

                    GROUP BY userID 

                    ORDER BY sum_propa DESC 

                    LIMIT 100

                    ) AS nb_share_table 

                ON news_propagation.userID = nb_share_table.userID

                ) AS nb_propa_100share 

            ON news_labels.newsID = nb_propa_100share.newsID 

            ) 

        GROUP BY label 

        ''',

 "Distribution by label for the 10 news the most shared": 

    ''' 

    SELECT label, COUNT(label) 

    FROM (

        SELECT * 

        FROM news_labels 

        JOIN (

            SELECT newsID, SUM(propagCount) AS sum_propa 

            FROM news_propagation 

            GROUP BY newsID 

            ORDER BY sum_propa DESC 

            LIMIT 10

            ) AS shared10_news 

        ON news_labels.newsID = shared10_news.newsID 

        ) 

    GROUP BY label 

    ''',

 "Distribution by label for the 50 news the most shared": 

    ''' 

    SELECT label, COUNT(label) 

    FROM (

        SELECT * 

        FROM news_labels 

        JOIN (

            SELECT newsID, SUM(propagCount) AS sum_propa 

            FROM news_propagation 

            GROUP BY newsID 

            ORDER BY sum_propa DESC 

            LIMIT 50

            ) AS shared50_news 

        ON news_labels.newsID = shared50_news.newsID 

        ) 

    GROUP BY label 

    '''

 }



instructionSequence = ["Distribution by label for the news shared by 10 most sharing",

                       "Distribution by label for the news shared by 100 most sharing",

                       "Distribution by label for the 10 news the most shared", 

                       "Distribution by label for the 50 news the most shared"]

for instruction in instructionSequence:

    c.execute(instruc_stat_label[instruction])

    print(instruction)

    print(c.fetchall())
### STATISTICS ON FOLLOW ###################################################



instruc_stat_following = {

 "Mean of number of following": 

    ''' 

    SELECT CAST(AVG(nb_follow) as int) 

    FROM (

        SELECT userID, COUNT(userID) AS nb_follow 

        FROM followers 

        GROUP BY userID

        ) 

    ''',

 "Distribution of number of following": 

    ''' 

    SELECT nb_follow, COUNT(*) propa_count 

    FROM (

        SELECT userID, COUNT(userID) AS nb_follow 

        FROM followers 

        GROUP BY userID

        ) 

    GROUP BY nb_follow 

    ORDER BY nb_follow ASC '''

 }



instructionSequence = ["Mean of number of following", 

                       "Distribution of number of following"]

for instruction in instructionSequence:

    c.execute(instruc_stat_following[instruction])

    print(instruction)

    if instruction == "Distribution of number of following" :

        print(c.fetchall()[0:10])

    else :

        print(c.fetchall())





instruc_stat_followed = {

 "Mean of number of followed": 

    ''' 

    SELECT CAST(AVG(nb_followed) as int) 

    FROM (

        SELECT userID, COUNT(userID) AS nb_followed 

        FROM followers 

        GROUP BY userID_followed

        ) 

    ''',

 "Distribution of number of followed": 

    ''' 

    SELECT nb_followed, COUNT(*) propa_count 

    FROM (

        SELECT userID, COUNT(userID_followed) AS nb_followed 

        FROM followers 

        GROUP BY userID_followed

        ) 

    GROUP BY nb_followed 

    ORDER BY nb_followed ASC '''

 }



instructionSequence = ["Mean of number of followed", 

                       "Distribution of number of followed"]

for instruction in instructionSequence:

    c.execute(instruc_stat_followed[instruction])

    print(instruction)

    if instruction == "Distribution of number of followed" :

        print(c.fetchall()[0:10])

    else :

        print(c.fetchall())



        

        ### STATISTICS ON LABELS ######################################################



instruc_stat_label = {

 "Distribution by label for the news shared by 100 most followed": 

    '''

    SELECT label, COUNT(label) 

    FROM (

        SELECT * 

        FROM news_labels 

        JOIN (

            SELECT * 

            FROM news_propagation 

            JOIN (

                SELECT userID_followed, COUNT(userID_followed) AS nb_follower 

                FROM followers 

                GROUP BY userID_followed 

                ORDER BY nb_follower DESC 

                LIMIT 100

                ) AS nb_follow_table 

            ON news_propagation.userID = nb_follow_table.userID_followed

            ) AS nb_propa_100follow 

        ON news_labels.newsID = nb_propa_100follow.newsID 

        ) 

    GROUP BY label 

    ''',

 "Distribution by label for the news shared by 1000 most followed": 

    ''' 

    SELECT label, COUNT(label) 

    FROM (

        SELECT * 

        FROM news_labels 

        JOIN (

            SELECT * 

            FROM news_propagation 

            JOIN (

                SELECT userID_followed, COUNT(userID_followed) AS nb_follower 

                FROM followers 

                GROUP BY userID_followed 

                ORDER BY nb_follower DESC 

                LIMIT 1000

                ) AS nb_follow_table 

            ON news_propagation.userID = nb_follow_table.userID_followed

            ) AS nb_propa_1000follow 

        ON news_labels.newsID = nb_propa_1000follow.newsID 

        ) 

    GROUP BY label 

    ''',

 "Distribution by label for the news shared by followed by one": 

    ''' 

    SELECT label, COUNT(label) 

    FROM (

        SELECT * 

        FROM news_labels 

        JOIN (

            SELECT * 

            FROM news_propagation 

            JOIN (

                SELECT * 

                FROM (

                    SELECT userID_followed, COUNT(userID_followed) AS nb_follower 

                    FROM followers 

                    GROUP BY userID_followed

                    ) 

                WHERE nb_follower = 1

                ) AS nb_follow_table 

            ON news_propagation.userID = nb_follow_table.userID_followed

            ) AS nb_propa_100follow 

        ON news_labels.newsID = nb_propa_100follow.newsID 

        ) 

    GROUP BY label 

    '''

}



instructionSequence = ["Distribution by label for the news shared by 100 most followed",

                       "Distribution by label for the news shared by 1000 most followed", 

                       "Distribution by label for the news shared by followed by one"]



for instruction in instructionSequence:

    c.execute(instruc_stat_label[instruction])

    print(instruction)

    print(c.fetchall())
instruc_stat_label = {

 "The 10 users that follow the most": 

    ''' 

    SELECT userID, COUNT(userID) AS nb_follow 

    FROM followers 

    GROUP BY userID 

    ORDER BY nb_follow DESC 

    LIMIT 10 

    ''',

 "The 10 users that are followed the most": 

    ''' 

    SELECT userID_followed, COUNT(userID_followed) AS nb_follower 

    FROM followers 

    GROUP BY userID_followed 

    ORDER BY nb_follower DESC 

    LIMIT 10 

    '''

 }





instructionSequence = ["The 10 users that follow the most", 

                       "The 10 users that are followed the most"]

for instruction in instructionSequence:

 c.execute(instruc_stat_label[instruction])

 print(instruction)

 print(c.fetchall())





instruc_stat_label = {

 "The 10 news that were the most shared": 

    ''' 

    SELECT newsID, SUM(propagCount) AS sum_propa 

    FROM news_propagation 

    GROUP BY newsID 

    ORDER BY sum_propa DESC 

    LIMIT 10 

    ''',

 "The 10 users that shared the most": 

    ''' 

    SELECT userID, SUM(propagCount) AS sum_propa 

    FROM news_propagation 

    GROUP BY userID 

    ORDER BY sum_propa DESC 

    LIMIT 10 

    '''

 }





instructionSequence = ["The 10 news that were the most shared", 

                       "The 10 users that shared the most"]

for instruction in instructionSequence:

    c.execute(instruc_stat_label[instruction])

    print(instruction)

    print(c.fetchall())
c.execute('''

    WITH RECURSIVE

      under_part(user, follower, level) AS (

         VALUES('User', '13973', 0)

         UNION

         SELECT followers.userID_followed, followers.userID, under_part.level+1 

             FROM followers, under_part

         WHERE followers.userID_followed = under_part.follower AND  under_part.level <= 1

    )

    SELECT SUBSTR('..........',1,level*3) || "(" || user || ", " || follower || ")" 

    FROM under_part

    ''').fetchall()[0:40]

instruc_stat_label = {

    "News propagation from user 22 (3 generations)": 

    '''

    WITH RECURSIVE

      under_part(user, follower, level) AS (

         VALUES('?', '22', 0)

         UNION

         SELECT followers.userID_followed, followers.userID, under_part.level+1 

             FROM followers, under_part

         WHERE followers.userID_followed = under_part.follower AND  under_part.level <= 2

    )

    SELECT COUNT(DISTINCT follower)

    FROM under_part

    GROUP BY follower

    ''',

    "News propagation from user 22 (5 generations)": 

    '''

    WITH RECURSIVE

      under_part(user, follower, level) AS (

         VALUES('?', '22', 0)

         UNION

         SELECT followers.userID_followed, followers.userID, under_part.level+1 

             FROM followers, under_part

         WHERE followers.userID_followed = under_part.follower AND  under_part.level <= 4

    )

    SELECT COUNT(DISTINCT follower)

    FROM under_part

    GROUP BY follower

    ''',

    "News propagation from user 22 (6 generations)": 

    '''

    WITH RECURSIVE

      under_part(user, follower, level) AS (

         VALUES('?', '22', 0)

         UNION

         SELECT followers.userID_followed, followers.userID, under_part.level+1 

             FROM followers, under_part

         WHERE followers.userID_followed = under_part.follower AND  under_part.level <= 5

    )

    SELECT COUNT(DISTINCT follower)

    FROM under_part

    GROUP BY follower

    ''',

    "News propagation from user 229 (3 generations)": 

    '''

    WITH RECURSIVE

      under_part(user, follower, level) AS (

         VALUES('?', '229', 0)

         UNION

         SELECT followers.userID_followed, followers.userID, under_part.level+1 

             FROM followers, under_part

         WHERE followers.userID_followed = under_part.follower AND  under_part.level <= 2

    )

    SELECT COUNT(DISTINCT follower)

    FROM under_part

    GROUP BY follower

    ''',

    "News propagation from user 229 (5 generations)": 

    '''

    WITH RECURSIVE

      under_part(user, follower, level) AS (

         VALUES('?', '229', 0)

         UNION

         SELECT followers.userID_followed, followers.userID, under_part.level+1 

             FROM followers, under_part

         WHERE followers.userID_followed = under_part.follower AND  under_part.level <= 4

    )

    SELECT COUNT(DISTINCT follower)

    FROM under_part

    GROUP BY follower

    '''

}





instructionSequence = ["News propagation from user 22 (3 generations)", 

                       "News propagation from user 22 (5 generations)", 

                       "News propagation from user 22 (6 generations)", 

                       "News propagation from user 229 (3 generations)",

                       "News propagation from user 229 (5 generations)"]

for instruction in instructionSequence:

    c.execute(instruc_stat_label[instruction])

    print(instruction)

    print(len(c.fetchall()))

#close the connection (cursor "c" will be closed as well)

conn.close()

print("inserting task finished")