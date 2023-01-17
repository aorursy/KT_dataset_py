# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json # JSON encoder and decoder

# Input data files are available in the "../input/" directory.
# Any results written to the current directory are saved as output.
# Pandas(
#0    Index (int),
#1    userId (int),
#2    movieId (int),
#3    rating (float),
#4    timestamp (int)
# )
ratings_df = pd.read_csv("../input/ratings.csv")

# Pandas(
#0      Index (int),
#1      adult (bool),
#2   x  belongs_to_collection ({id (str), name (str), poster_path (str), backdrop_path(str)}), 
#3   x  budget (str),
#4   x  genres ([{id (str), name (str)}]),
#5      homepage (str),
#6   x  id (str),
#7      imdb_id (str),
#8      original_language (str),
#9      original_title (str),
#10     overview (str),
#11     popularity (float),
#12     poster_path (str),
#13  x  production_companies ([{name (str), id (str)}]),
#14  x  production_countries ([{iso_3166_1 (str), name (str)}]),
#15  x  release_date (str),
#16     revenue (float),
#17  x  runtime (float),
#18  x  spoken_languages ([{iso_639_1 (str), name (str)}]),
#19     status (str),
#20     tagline (str),
#21     title (str),
#22     video (bool),
#23     vote_average (float),
#24     vote_count (float)
# )
movies_df = pd.read_csv("../input/movies_metadata.csv", low_memory=False)
total_ratings = 0
ratings_sum = 0

for row in ratings_df.itertuples():
    # If the userId has surpassed "1", end the loop
    if row[1] > 1:
        break
        
    # Get the movie information using the movieId in the given ratings_df row
    movie_df = movies_df[movies_df.id == str(row[2])]
    # Ignore ratings of movies with a non-existant movieId
    if movie_df.empty:
        continue
        
    print("Movie ID: ",row[2])
    print("User Rating: ", row[3])
    print("Movie Title: ", movie_df["original_title"].values[0])
    print("Movie Genres: ", movie_df["genres"].values[0])
    print("Movie Budget: ", movie_df["budget"].values[0])
    print("Movie Production Companies: ", movie_df["production_companies"].values[0])
    print("Movie Production Countries: ", movie_df["production_countries"].values[0])
    print("Movie Runtime: ", movie_df["runtime"].values[0])
    print("Movie Release Date: ", movie_df["release_date"].values[0])
    print("Movie Spoken Languages: ", movie_df["spoken_languages"].values[0])
    print("\n\n")
    
    total_ratings += 1
    ratings_sum += row[3]

# Print the total number of movies the user rated, and their average rating for them all.
print("User 1 had %d ratings with an average of %d out of 5 stars." % (total_ratings, (ratings_sum / total_ratings)))
def is_similar(movie_df):
    """
    For this function, a data frame for a movie (a row from movies_df) is taken in,
    and it is determined whether or not the movie is similar to the one being advertised.
    
    Below are the attributes for the movie being advertised ("desired_...")
    """
    desired_genre = "comedy"
    desired_runtime = 105 # minutes
    desired_language = "en" # iso_639_1 code
    
    # A list of genres (lower case strings) that the movie includes
    genres_list = []
    # Create a JSON object of the list of genres from the data frame value
    try:
        genres_list_json = json.loads('{"genres": ' + str(movie_df["genres"].values[0]).replace("'", "\"") + '}')
        # Append each value (the genre name) of the JSON object into genres_list
        for genre in genres_list_json["genres"]:
            genres_list.append(genre["name"].lower())
    except ValueError:
        # Ignore JSONDecodeError and leave genres_list empty (assume desired language is there)
        pass
    # The movie is not similar, as the desired genre is not in the genres list that was created
    if len(genres_list) != 0 and desired_genre.lower() not in genres_list:
        return False
    
    # The runtime of the movie
    runtime = movie_df["runtime"].values[0]
    # The movie is not similar, as the runtime is not within a 30 minute range of the new movie's runtime
    if runtime != 0 and desired_runtime < runtime - 30 and desired_runtime > runtime + 30:
        return False
    
    # The original language of the movie
    original_language = movie_df["original_language"].values[0]
    # The movie is not similar, as it is in a different language
    if original_language and original_language.lower() != desired_language.lower():
        return False
    
    # The list of languages spoken in the movie (not dubbed languages, but actual languages spoken in the movie)
    # These are represented as lower case languages codes (ex. "en")
    spoken_languages_list = []
    # Like the genres, create a JSON object of the list of languages from the data frame value
    try:
        spoken_languages_list_json = json.loads('{"languages": ' + str(movie_df["spoken_languages"].values[0]).replace("'", "\"") + '}')
        # Append each value (the language code) of the JSON object into spoken_languages_list 
        for language in spoken_languages_list_json["languages"]:
            spoken_languages_list.append(language["iso_639_1"].lower())
    except ValueError:
        # Ignore JSONDecodeError and leave spoken_languages_list empty (assume desired language is there)
        pass
    # The movie is not similar, as the desired language is not spoken within the movie
    if len(spoken_languages_list) != 0 and desired_language.lower() not in spoken_languages_list:
        return False
    
    # The movie in question has passed all of the tests. It is similar to the one being advertised!
    return True
def classify_users(num_users, similar_movie_ratio_threshold, rating_threshold):
    """
    This function takes in 3 arguments:
     * num_users: the number of users to analyze
     * similar_movie_ratio_threshold: the percentage of movies that need to be similar for a primary target
     * rating_threshold: the average rating for similar movies needed for a secondary target
     
    It returns a 3D array of userIds (primary targets, secondary targets, non-targets)
    """
    primary_targets = []
    secondary_targets = []
    non_targets = []
    
    total_movies = 0
    similar_movies = 0
    similar_movies_ratings_sum = 0
    user = 0
    
    for row in ratings_df.itertuples():
        # Set initial userId
        if user == 0:
            user = row[1]
        # End function after we have analyzed a certain number of users (num_users)
        if user > num_users:
            break
        # Determine whether user is a target or not after we have analyzed all of their ratings (user number has changed)
        if user != row[1]:
            # If user has rated a given amount of similar movies, they are a primary target
            if total_movies != 0 and similar_movies / total_movies >= similar_movie_ratio_threshold:
                primary_targets.append(user)
            # If user has rated (on average) the small amount of similar movies above a given threshold, they are a secondary target
            elif similar_movies != 0 and similar_movies_ratings_sum / similar_movies >= rating_threshold:
                secondary_targets.append(user)
            # Note which users are non-target audience members for possible future analysis
            else:
                non_targets.append(user)
            
            user = row[1]
            total_movies = 0
            similar_movies = 0
            similar_movies_ratings_sum = 0
        
        # Get the movie information using the movieId in the given ratings_df row
        movie_df = movies_df[movies_df.id == str(row[2])]
        # Ignore ratings of movies with a non-existant movieId
        if movie_df.empty:
            continue
        # Determine if movie is similar to the one being advertised
        if is_similar(movie_df):
            similar_movies += 1
            similar_movies_ratings_sum += row[3]
        total_movies += 1
        
    return [primary_targets, secondary_targets, non_targets]
total = 0
similar = 0
similar_sum = 0
user_id = 0
for row in ratings_df.itertuples():
    if user_id == 0:
        user_id = row[1]
    if user_id > 10:
        break
    if user_id != row[1]:
        print("User %d had %d total ratings with %d being similar movies (%.1f%%)." % (user_id, total, similar, ( 100 * (similar / total))))
        print("The user rated similar movies with an average of ", (similar_sum / similar), " out of 5 stars.")
        
        if total != 0 and similar / total >= 0.25:
            print("User %d is a primary target.\n" % (user_id))
        elif similar != 0 and similar_sum / similar >= 4:
            print("User %d is a secondary target.\n" % (user_id))
        else:
            print("User %d is a non-target.\n" % (user_id))
            
        user_id = row[1]
        total = 0
        similar = 0
        similar_sum = 0

    movie_df = movies_df[movies_df.id == str(row[2])]
    if movie_df.empty:
        continue
    if is_similar(movie_df):
        similar += 1
        similar_sum += row[3]
    total += 1
targets = classify_users(1000, 0.35, 4)

primary_percentage = ("%.1f%%" % (100 * len(targets[0]) / 1000))
secondary_percentage = ("%.1f%%" % (100 * len(targets[1]) / 1000))
non_percentage = ("%.1f%%" % (100 * len(targets[2]) / 1000))

print("Primary targets (", primary_percentage, "):")
print(targets[0], "\n")
print("Secondary targets (", secondary_percentage, "):")
print(targets[1], "\n")
print("Non-targets (", non_percentage, "):")
print(targets[2], "\n")