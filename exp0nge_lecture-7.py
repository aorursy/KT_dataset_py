import requests
base_url = "https://cat-fact.herokuapp.com"
response = requests.get("{}/facts".format(base_url))
response
response.content
response.json()
type(response.json())
json = response.json()
json['all']
type(json['all'])
json.keys()
json['all'][0]
json['all'][1]
type(json['all'][0]['user'])
json['all'][0]['user'].keys()
json['all'][0]['user']['name']
json['all'][0]['user']['name'].keys()
json['all'][0].values()
json['all'][0]['user']['name'].get('first')
print(json['all'][0]['user']['name'].get('first_name', 'N/A'))
empty_dict = {}
empty_dict.get('name')
all_facts = json['all']
first_fact = all_facts[0]
user = first_fact['user']
name = user['name']
first_name = name['first']
first_name
type(json['all'][0])
json['all'][0]['text']
fact = json['all'][0]['text']
author = json['all'][0]['user']['name']['first']

'The fact "{}" and it was submitted by {}'.format(fact, author)
for cat_fact in json['all']:
    fact = cat_fact['text']
    author = cat_fact.get('user')
    if author:
        author_name = author['name']['first']
    else:
        # if author == None
        author_name = "annon"

    print('The fact "{}" and it was submitted by {}'.format(fact, author_name))
base_url = "https://api.pokemontcg.io/v1/"
response = requests.get("{}/{}".format(base_url, "cards"))
response
json = response.json()
json
response = requests.get("{}/{}/{}".format(base_url, "cards", "base5-20"))
response.content
json.keys()
json['cards'][1]
json['cards'][1]['imageUrlHiRes']
# https://www.dictionaryapi.com/api/v3/references/thesaurus/json/umpire?key=your-api-key
base_url = "https://www.dictionaryapi.com/api/v3/references/thesaurus/json/"
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("API_KEY")
api_key = "helloworld"
base_url
"{}{}?key={}".format(base_url, "learning", 123445)
response = requests.get("{}{}?key={}".format(base_url, "cards", api_key))
response
response.content
response.json()
response.json()[0]['meta']
response.json()[0]['meta']['syns'][0]
import json
base_url_v3 = "https://api.themoviedb.org/3"
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("TMDB_BEARER")
data = { "query": "Adventure Time", "hello": "world" }
print(data)
headers = {"Authorization": "Bearer {}".format(api_key)}
full_url = "{}{}".format(base_url_v3, "/search/movie")
print(full_url)
response = requests.get(full_url, params = data, headers=headers)
response
response.request.url
response.json()
tending_daily = "/trending/all/day"
full_url = "{}{}".format(base_url_v3, tending_daily)
print(full_url)
response = requests.get(full_url, headers={"Authorization": "Bearer {}".format(api_key)})
response
response.json()
json_response = response.json()
type(json_response)
json_response.keys()
total_pages = json_response['total_pages']
page = json_response['page']
total_results = json_response['total_results']
print("Total Pages: {}\nCurrent Page: {}\nTotal Results: {}".format(total_pages, page, total_results))
per_page_results = len(json_response['results'])
per_page_results
import pandas as pd
df = pd.DataFrame(json_response['results'])
df
df.sort_values("popularity", ascending=False)
df.dropna()
movie_df = df[df['media_type'] == 'movie']
movie_df
movie_df.dtypes
movie_df['release_date'] = pd.to_datetime(movie_df['release_date'])
movie_df.dtypes
movie_df.plot(x = 'release_date', y = 'popularity')
recommendations = "/movie/{}/recommendations".format(545609)
full_url = "{}{}".format(base_url_v3, recommendations)
full_url

response = requests.get(full_url, headers={"Authorization": "Bearer {}".format(api_key)})
response
response.json()
