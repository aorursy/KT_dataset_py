import pandas as pd 
import sqlite3

path = "../input/"  
database = path + 'database.sqlite'

conn = sqlite3.connect(database)

# Do review scores for individual artists generally improve over time? / Full results table 

query_1A = pd.read_sql("""SELECT 
                            reviews.score, artists.artist, years.year
                        FROM reviews 
                        JOIN artists 
                            ON reviews.reviewid = artists.reviewid 
                        JOIN years
                            ON artists.reviewid = years.reviewid
                        ORDER BY artists.artist, years.year
                        ;""", conn)

query_1A
import pandas as pd 
import sqlite3

path = "../input/"  
database = path + 'database.sqlite'

conn = sqlite3.connect(database)

# Do review scores for individual artists generally improve over time? / Individual Trends for Top 3 Most Reviewed Artists 

query_1B = pd.read_sql(""" SELECT reviews.score AS 'review_score', reviews.reviewid, artists.artist AS 'artist', years.year AS 'year'
                        FROM reviews
                        JOIN artists
                            ON reviews.reviewid = artists.reviewid
                         JOIN years
                            ON artists.reviewid = years.reviewid
                        WHERE artists.artist = 'guided by voices'
                            OR artists.artist = 'neil young'
                            OR artists.artist = 'bonnie prince billy'
                        GROUP BY artists.artist, years.year
                        ORDER BY artists.artist, years.year
                        ;""", conn)

query_1B
import pandas as pd 
import sqlite3

path = "../input/"  
database = path + 'database.sqlite'

conn = sqlite3.connect(database)

# Query_1C Do average reviews increase over time / visualization 

query_1C = pd.read_sql("""SELECT AVG(reviews.score) AS 'average review', years.year
                        FROM reviews
                        JOIN years
                            ON reviews.reviewid = years.reviewid
                        GROUP BY years.year 
                        ;""", conn)

query_1C.plot(x=['year'],y=['average review'],figsize=(12,5),title='Average Reviews Over Time')
import pandas as pd 
import sqlite3

path = "../input/"  
database = path + 'database.sqlite'

conn = sqlite3.connect(database)

# How has Pitchfork's review genre selection changed over time?

query_2 = pd.read_sql("""WITH t1 AS (
                        SELECT COUNT(reviews.score) AS 'count',genres.genre, years.year AS 'year'
                        FROM reviews
                        JOIN genres 
                            ON reviews.reviewid = genres.reviewid
                        JOIN years 
                            ON reviews.reviewid = years.reviewid
                        GROUP BY genre, year 
                        ORDER BY year, count DESC)
                        
                      SELECT DISTINCT(year), genre, count
                      FROM t1 
                      WHERE  count=(SELECT MAX(t2.count)
                                        FROM t1 AS t2
                                        WHERE t1.year = t2.year)
                      ;""", conn)

query_2
import pandas as pd 
import sqlite3

path = "../input/"  
database = path + 'database.sqlite'

conn = sqlite3.connect(database)

# Who are the most highly rated artists and the least highly rated artists?

query_3 = pd.read_sql("""WITH highest AS (
                            SELECT AVG(reviews.score) AS 'average_review', artists.artist
                            FROM reviews
                            JOIN artists ON reviews.reviewid = artists.reviewid
                            GROUP BY artists.artist
                            ORDER BY AVG(reviews.score) DESC
                            LIMIT 10
                            ),
                            lowest AS ( 
                            SELECT AVG(reviews.score) AS 'average_review', artists.artist
                            FROM reviews
                            JOIN artists ON reviews.reviewid = artists.reviewid
                            GROUP BY artists.artist
                            ORDER BY AVG(reviews.score)
                            LIMIT 10)
                            
                            SELECT *
                            FROM highest 
                            UNION ALL 
                            SELECT *
                            FROM lowest
                            ORDER BY average_review DESC;""", conn)
query_3
import pandas as pd 
import sqlite3

path = "../input/"  
database = path + 'database.sqlite'

conn = sqlite3.connect(database)

# Which genre has the highest average rating? 

query_4 = pd.read_sql("""SELECT AVG(reviews.score) AS 'average review', genres.genre
                        FROM reviews
                        JOIN genres ON reviews.reviewid = genres.reviewid
                        GROUP BY genre
                        ORDER BY AVG(reviews.score) DESC;""", conn)

query_4