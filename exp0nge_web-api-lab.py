api_key = '2Hg2yNZz4S0s0DoOVp8ZqDv1cNKIN0FO'
api_key
base_url = 'https://api.nytimes.com/'
article_base_url = 'svc/search/v2/articlesearch.json'
import requests
requests.get("https://api.nytimes.com/svc/search/v2/articlesearch.json?q=election&api-key=2Hg2yNZz4S0s0DoOVp8ZqDv1cNKIN0FO").json()
params = {
    'q': 'technology',
    'api-key': api_key
}
print(params)
my_string = '{}{}'.format(base_url, article_base_url)
my_string
search_response = requests.get(my_string, params = params)
search_response
search_response.json()
search_response.request.url
search_response.json()
json_content = search_response.json()

# Print the keys of the dictionary
json_content.keys()
json_content['response'].keys()
json_content['response']['meta']
type(json_content['response']['docs'])
first_doc = json_content['response']['docs'][0]
first_doc
first_doc.keys()
first_doc['abstract']
params = {
    'q': 'technology',
    'api-key': api_key,
    'fq': 'pub_year: (2019)'
}
search_response = requests.get('{}{}'.format(base_url, article_base_url), params=params)
json_content = search_response.json()
json_content
json_content['response']['docs'][0]['pub_date']
second_article = json_content['response']['docs'][1]
second_article
second_article.keys()
second_article['pub_date']
second_article['abstract']
# https://api.nytimes.com/svc/books/v3/lists/current/hardcover-fiction.json?api-key=yourkey
books_url = 'svc/books/v3/lists/names.json'
params = {
    'api-key': api_key
}
full_url = '{}{}'.format(base_url, books_url)
full_url
books_response = requests.get(full_url, params=params)
books_response
json_content = books_response.json()
json_content
json_content['results'][0]
import pandas as pd
df = pd.DataFrame(____['_____'])
df
books_url = 'svc/________/{}/{}.json'.format('current', '____')
params = {
    'api-key': api_key
}
full_url = '{}{}'.format(base_url, books_url)
books_response = requests.get(full_url, ____=params)
books_response
___.___()
df = pd.DataFrame(books_response._____()['____']['____'])
df
df.plot(____)
movie_url = 'svc/_____'
params = {
    'api-key': api_key,
    '____': 'fast'
}
full_url = '{}{}'.format(base_url, movie_url)
movie_response = requests.get(full_url, params=params)
movie_response
____.___()
pd.DataFrame(______)
