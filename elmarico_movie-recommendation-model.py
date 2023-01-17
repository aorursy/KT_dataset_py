# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from selenium import webdriver

from selenium.webdriver.common.by import By

from selenium.webdriver.support.ui import WebDriverWait

from selenium.webdriver.support import expected_conditions as EC

import csv



path = "path_to_chrome_driver"

driver = webdriver.Chrome(path)



# Function to scrape informations from movie

def scrape_movie():



    try:

        title = WebDriverWait(driver, 30).until(

            EC.presence_of_element_located((By.CLASS_NAME, "title_wrapper"))

        )

        movie_title = title.text



        rating = WebDriverWait(driver, 30).until(

            EC.presence_of_element_located((By.CLASS_NAME, "imdbRating"))

        )

        movie_rating = rating.text



        duration = WebDriverWait(driver, 30).until(

            EC.presence_of_element_located((By.XPATH, "//*[@id='title-overview-widget']/div[1]/div[2]/div/div[2]/div[2]/div/time"))

        )

        movie_duration = duration.text



        genre = WebDriverWait(driver, 30).until(

            EC.presence_of_element_located((By.XPATH, "//*[@id='title-overview-widget']/div[1]/div[2]/div/div[2]/div[2]/div/a[1]"))

        )

        movie_genre = genre.text



        summary = WebDriverWait(driver, 30).until(

            EC.presence_of_element_located((By.CLASS_NAME, "summary_text"))

        )

        movie_summary = summary.text



        year = WebDriverWait(driver, 30).until(

            EC.presence_of_element_located((By.ID, "titleYear"))

        )

        movie_release = year.text



        credits = WebDriverWait(driver, 30).until(

            EC.presence_of_all_elements_located((By.CLASS_NAME, "credit_summary_item"))

        )

        director = credits[0].text

        writers = credits[1].text

        stars = credits[2].text



        keywords = WebDriverWait(driver, 30).until(

            EC.presence_of_all_elements_located((By.CLASS_NAME, "itemprop"))

        )

        keywords_of_movie = []

        for i in keywords:

            keywords_of_movie.append(i.text)



        movie = (movie_title,movie_rating,director,writers,stars,keywords_of_movie,movie_duration,movie_genre,movie_release,movie_summary)

        driver.back()



        return movie



    except:

        driver.quit()





driver.get("https://www.imdb.com/search/title/?groups=top_1000&sort=user_rating,desc&start=0&ref_=adv_nxt")



scraped_movies = []



# Start of every page is increased by 50. So, start of 1st page is 1, 2nd is 51, 3rd is 101,...

start_page = []

start_page.append(0)

number = 1

for i in range(16):

    number += 50

    start_page.append(number)



page = 0

while page < 14:

    for i in range(0,50): #0,50

        try:

            items = WebDriverWait(driver, 30).until(

                EC.presence_of_all_elements_located((By.CLASS_NAME, "loadlate"))

            )



            items[i].click()

            movie_scraped = scrape_movie()

            scraped_movies.append(movie_scraped)



        except:

            driver.implicitly_wait(5)

    page+=1

    new_link = "https://www.imdb.com/search/title/?groups=top_1000&sort=user_rating,desc&start={}&ref_=adv_nxt".format(start_page[page])

    driver.get(new_link)



with open("imdb_top_800_movies_4.csv", "w", newline='',encoding='utf-8') as f:

    writer = csv.writer(f)

    writer.writerow(["Title","Rating","Director","Writers","Stars","Keywords","Movie_Duration","Genre","Release_Year","Description"])

    writer.writerows(scraped_movies)
import pandas as pd



df = pd.read_csv('/kaggle/input/imdb-top-800-moviescsv/imdb_top_800_movies.csv', encoding='ISO-8859-1')

print(df.head())
print(df.dtypes)
df["Title"] = df["Title"].apply(lambda x: x.split("(")[0])

df["Director"] = df["Director"].apply(lambda x: x.split(":")[1])

df["Writers"] = df["Writers"].apply(lambda x: x.split(":")[1].split("|")[0].split("(")[0])

df["Stars"] = df["Stars"].apply(lambda x: x.split(":")[0].split("|")[0])

df.head()
df["duration"] = df["Movie_Duration"].apply(pd.Timedelta)

df["hour"] = df["duration"].apply(lambda x: str(x).split("days")[1].split(":")[0])

df["hour"] = pd.to_numeric(df["hour"]) * 60

df["minutes"] = df["duration"].apply(lambda x: str(x).split("days")[1].split(":")[1])

df["minutes"] = pd.to_numeric(df["minutes"])

df["Movie_Duration"] = df["hour"] + df["minutes"]

df.drop(columns=["duration","hour","minutes"],inplace=True)
df.head(3)
df["Num_of_ratings"] = df["Rating"].apply(lambda x: x.split("\r")[1].replace(",",""))
df["Num_of_ratings"] = pd.to_numeric(df["Num_of_ratings"])
df["Rating"] = df["Rating"].apply(lambda x: x.split("\r")[0].split("/")[0])
df["Rating"] = pd.to_numeric(df["Rating"])

print(df.dtypes)
to_convert = ["Title","Director","Writers","Stars","Keywords"]

def convert_to_lower(columns):

    for i in columns:

        df[i] = df[i].apply(lambda x: x.lower())



convert_to_lower(to_convert)
spec_chars = ["!",'"',"#","ç","","%","&","'","(",")",

              "*","+",",","-",".","/",":",";","<",

              "=",">","?","@","[","\\","]","^","_",

              "`","{","|","}","~","–",",","[","]",

              "à","á","â","ã","ó", "+","Ù","Û",

              "ä", "©", "í", "ì","ç","\r"]





for char in spec_chars:

    df["Stars"] = df["Stars"].str.replace(char, "")

    df["Keywords"] = df["Keywords"].str.replace(char, "")

    df["Writers"] = df["Writers"].str.replace(char, "")

    df["Title"] = df["Title"].str.replace(char, "")

    df["Director"] = df["Director"].str.replace(char, "")
print(df.isnull().values.any())

print(df.isnull().sum().sum())

df.drop(columns="Description",inplace=True)
df.to_csv("imdb_top_800_cleaned.csv", index=False,header=True)
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer
df_cleaned = pd.read_csv("/kaggle/input/cleaned/imdb_top_800_cleaned.csv")

df_cleaned.reset_index(inplace=True)

df_cleaned.rename(columns={"index" : "Id"},inplace=True)
df_cleaned.head()
features = ["Title", "Director", "Writers", "Stars", "Keywords", "Genre"]
def combine_features(features):

    features_combined = []

    for i in range(features.shape[0]):

        features_combined.append(features["Director"][i] + " " +

                                 features["Writers"][i] + " " +

                                 features["Stars"][i] + " " +

                                 features["Keywords"][i] + " " +

                                 features["Genre"][i].lower())



    return features_combined
df_cleaned["features"] = combine_features(df_cleaned)

df_cleaned.head(3)
cv = CountVectorizer()

vectorized = cv.fit_transform(df_cleaned["features"])
cs = cosine_similarity(vectorized)

print(cs)

print(cs.shape)
movie_title = "the godfather " # User's choice of move - can be any movie from df



movie_id = df_cleaned[df_cleaned["Title"] == movie_title]["Id"].values[0] # finding Id of movie
scores = list(enumerate(cs[movie_id]))

sorted_scores = sorted(scores,key = lambda x: x[1], reverse=True)
counter = 0

similar_movies = []

for i in sorted_scores:

    similar_movie = df_cleaned[df_cleaned["Id"] == i[0]]["Id"].values[0]

    similar_movies.append(similar_movie)

    counter+=1

    if(counter == 6):

        break
similar_movies_expanded = []

for id in similar_movies:

    movie_to_append = df_cleaned[df_cleaned["Id"] == id][["Title","Genre","Release_Year","Rating"]].values[0]

    similar_movies_expanded.append(movie_to_append)

    

print("Similar movies to the movie '{}' are: "

      "\n1)Title:{} | Genre:{} | Year:{} | Rating: {} "

      "\n2)Title:{} | Genre:{} | Year:{} | Rating: {}"

      "\n3)Title:{} | Genre:{} | Year:{} | Rating: {}"

      "\n4)Title:{} | Genre:{} | Year:{} | Rating: {}"

      "\n5)Title:{} | Genre:{} | Year:{} | Rating: {}".format(movie_title,

          similar_movies_expanded[1][0], similar_movies_expanded[1][1], similar_movies_expanded[1][2], similar_movies_expanded[1][3],

          similar_movies_expanded[2][0], similar_movies_expanded[2][1], similar_movies_expanded[2][2], similar_movies_expanded[2][3],

          similar_movies_expanded[3][0], similar_movies_expanded[3][1], similar_movies_expanded[3][2], similar_movies_expanded[3][3],

          similar_movies_expanded[4][0], similar_movies_expanded[4][1], similar_movies_expanded[4][2], similar_movies_expanded[4][3],

          similar_movies_expanded[5][0], similar_movies_expanded[5][1], similar_movies_expanded[5][2], similar_movies_expanded[5][3],

      ))