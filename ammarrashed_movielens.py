import ibmos2spark

# @hidden_cell
# credentials = {#### #####}

# configuration_name = '#### ####'
# cos = ibmos2spark.CloudObjectStorage(sc, credentials, configuration_name, 'bluemix_cos')
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
tags = spark.read\
  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
  .option('header', 'true')\
  .load(cos.url('tag.csv', 'cs340spring20184b12f92a8d204278af3959660617b691'))
tags.show()
genome_tags = spark.read\
  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
  .option('header', 'true')\
  .load(cos.url('genome_tags.csv', 'cs340spring20184b12f92a8d204278af3959660617b691'))
genome_tags.show()
genome_scores = spark.read\
  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
  .option('header', 'true')\
  .load(cos.url('genome_scores.csv', 'cs340spring20184b12f92a8d204278af3959660617b691'))
genome_scores.show()
ratings = spark.read\
  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
  .option('header', 'true')\
  .load(cos.url('rating.csv', 'cs340spring20184b12f92a8d204278af3959660617b691'))
ratings.show()
movies = spark.read\
  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
  .option('header', 'true')\
  .load(cos.url('movie.csv', 'cs340spring20184b12f92a8d204278af3959660617b691'))
movies.show()
user_tag_movie_joined = tags.join(genome_tags, on="tag")
user_tag_movie_joined.show()
user_tag_movie = user_tag_movie_joined.select(["userId","tagId","movieId"])
user_tag_movie.show()
user_tag_movie_relevance = user_tag_movie.join(genome_scores, ((user_tag_movie.tagId==genome_scores.tagId)
                                                               and (user_tag_movie.movieId == genome_scores.movieId)))
user_tag_movie_relevance.show()
user_tag_movie_relevance = user_tag_movie.join(genome_scores, ((user_tag_movie.tagId==genome_scores.tagId) 
                                                               & (user_tag_movie.movieId == genome_scores.movieId)))
user_tag_movie_relevance.show()
user_tag_movie_relevance = user_tag_movie.join(genome_scores, on=["tagId","movieId"])
user_tag_movie_relevance.show()
from pyspark.sql.functions import col, tanh
users_reliability_flat = user_tag_movie_relevance.withColumn("reliability", tanh(user_tag_movie_relevance.relevance*20-12))\
                                            .select(["userId","reliability"])
users_reliability_flat.show()
users_reliability_flat.count()
users_reliability = users_reliability_flat.groupBy("userId").avg()
users_reliability.show()
users_reliability.count()
from pyspark.sql.functions import desc

users_reliability_sorted = users_reliability.select(["userId", col("avg(reliability)").alias("reliability")])\
                                            .sort(desc("reliability"))
users_reliability_sorted.show()
users_reliability_ratings_movies = ratings.join(users_reliability_sorted, on="userId")
users_reliability_ratings_movies.show()
movieId_reliableRating = users_reliability_ratings_movies.withColumn("reliable rating", col("rating")*col("reliability"))\
                                                         .select(["movieId", "reliable rating"])
movieId_reliableRating.show()
movieId_reliableRating_grouped = movieId_reliableRating.groupBy("movieId").avg()
movieId_reliableRating_grouped.show()
movieTitle_reliableRating = movieId_reliableRating_grouped.join(movies, on="movieId").select(["title","avg(reliable rating)"])
movieTitle_reliableRating.show()
top_movies = movieTitle_reliableRating.sort(desc("avg(reliable rating)"))\
                                        .select(["title", col("avg(reliable rating)").alias("actual rating")])
top_movies.show()
