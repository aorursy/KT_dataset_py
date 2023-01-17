from google.cloud import bigquery

import pandas as pd



def stack_user_answers(x):

    '''Returns a data frame of Stack Overflow answers for requested user IDs

    Args:

    * x - (list) List of user ids as strings to be queried

     

    Return

    * (pd.DataFrame) - DataFrame of Stack Overflow answers

    '''



    # create our bigquery client

    client = bigquery.Client()



    # create our SQL string query

    query = """

    SELECT

      id, owner_display_name, body, creation_date, owner_user_id, score, tags

    FROM

      `bigquery-public-data.stackoverflow.posts_answers`

    WHERE 

      owner_user_id IN (""" + ",".join(x) + """)

    """



    # make our query using the client

    query_job = client.query(query)



    # iterate through the result

    iterator = query_job.result()

    rows = list(iterator)



    # Transform the rows into a nice pandas dataframe

    data = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

    

    return(data)
import requests

from bs4 import BeautifulSoup

url = 'https://stackoverflow.com/tags/python/topusers'

article_response = requests.get(url)

souped_page = BeautifulSoup(article_response.text, 'html.parser')

info = souped_page.find_all('div',{'user-details'})

info[0]

css_selector = ".grid--cell+ .grid--cell .user-details a"
import requests

import lxml.html

import numpy as np

from lxml import html

!pip install cssselect

from lxml.cssselect import CSSSelector

import re



def get_top_x_answerers(url, x, quiet = False):

    '''Returns a list of the user ids 

     Args:

     * url - (str) URL that we are querying for user details

     * x - (int) The number of users we want to return

     * quiet -(bool) Do we want to print the info of the top x users. Default = False

     

     Return

     * (list) - List of integers for the user

     '''

    

    # Get out page of html

    article_response = requests.get(url)

    

    # Get the html tree from the page

    tree = html.fromstring(article_response.content)

    

    # Pass in the selected CSS

    answerers = tree.cssselect(css_selector)

    

    # Build a dictionary of the user names and ids

    top_users = {}

    for i in range(x):

        

        s = answerers[i].get("href")

        if not quiet:

            print(answerers[i].get("href"))

        

        # use a regex string to grab the name of the user

        user = re.findall("/[\D]+/[\d]+/([\D|\d]+)",s)[0]

        

        # another regex for their id

        user_id = (re.findall("[\d]+",s)[0])

        

        # fill in our dictionary with the key being their name and the value as the user_id

        top_users[user] = user_id

    

    # let's just return the user ids, i.e. the values in our dict that we built above

    return(top_users.values())
# Let's grab the top 10 python users

python_url = 'https://stackoverflow.com/tags/python/topusers'

python = get_top_x_answerers(url = python_url, x = 10, quiet = False)



r_url = 'https://stackoverflow.com/tags/r/topusers'

r = get_top_x_answerers(url = r_url, x = 10, quiet = True)
# get the answers from our top users

try:

    r_df = stack_user_answers(r)

except:

    r_df = pd.read_csv("../input/stack-answers-r-python/r_10.csv")

    

try:    

    python_df = stack_user_answers(python)

except:

    python_df = pd.read_csv("../input/stack-answers-r-python/python_10.csv")
# mark the answers as either r or python and bind the data frames together

r_df['language'] = "r"

python_df['language'] = "python"



data = r_df.append(python_df)

data.head(5)
def accepted_answer(id):

    '''Returns a boolean value (0 or 1) for if the answer ID was accepted

    Args:

    * id - (Numeric) Answer ID

     

    Return

    * 1 or 0

    '''

    # convert our answer ID to a string

    id = str(id)

    

    # create the url of the answer

    url = "https://stackoverflow.com/a/" + id

    

    # grab the url html page

    article_response = requests.get(url)

    

    # turn the content of the html page into a string

    tree = html.fromstring(article_response.content)

    

    # search for the answer in the page

    div = tree.get_element_by_id("answer-" + id)

    

    # for the answer check the itemprop class for whether it was accepted

    if div.get("itemprop") == 'acceptedAnswer':

        return 1

    else:

        return 0

        
import json



def accepted_api(ids):

    '''Returns a list of Booleans for whether the answer was accepted

    Args:

    * ids - (Pandas series) Series of numeric answer ids

     

    Return

    * List of Booleans

    '''

    

    # Build our API request URL

    base_url = "http://api.stackexchange.com/answers/"

    end_url = "?order=desc&sort=activity&site=stackoverflow"

    

    # Concatentate our numeric answer IDs as a string 

    ids = ";".join(ids.map(str))

    url_req = '{}{}{}'.format(base_url,ids,end_url)

    resp = requests.get(url_req).json()

    return([c['is_accepted'] for c in resp['items']])

import itertools



def chunks(seq, size):

    '''Returns a generator object with each element representing the next size x elements from seq

    Args:

    * seq - (List) List of elements to break up into chunks of size = size

    * size - (Numeric) Numeric for length of each chunk

     

    Return

    * (Pandas series) Series of numeric answer ids

    '''

    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
try:

    accepteds = [accepted_api(group) for group in chunks(data['id'], 100)]

    data['accepted'] = pd.Series(list(itertools.chain.from_iterable(accepteds)))

except:

    accepteds = pd.read_csv("../input/stack-answers-r-python/accepted.csv")

    data = pd.merge(data, accepteds, on='id')

    

data.head(5)
data['body'][0]
bs = BeautifulSoup(data['body'][0], 'html.parser')

print(bs.text)
from html.parser import HTMLParser



class MLStripper(HTMLParser):

    def __init__(self):

        self.reset()

        self.strict = False

        self.convert_charrefs= True

        self.fed = []

    def handle_data(self, d):

        self.fed.append(d)

    def get_data(self):

        return ''.join(self.fed)



def just_text(x):

    return(re.sub("\n<pre><code>.*?</code></pre>|</p>|<p>|<code>.*?</code>|\n"," ", x,flags=re.DOTALL))

    

def strip_tags(html):

    s = MLStripper()

    s.feed(html)

    return s.get_data()



print(data['body'][0:1].apply(just_text).apply(strip_tags)[0])
def clean_soup(x):

    '''Clean a block of text using BeautifulSoup

    Args:

    * x - (str) Text block to be cleaned

     

    Return

    * Cleaned str

    '''

    bs = BeautifulSoup(x, 'html.parser')

    return(bs.text)



data['text'] = data['body'].apply(clean_soup)
from textblob import TextBlob

import nltk.sentiment.vader

from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()



demo = "This is a terrific meetup :)"

blob = TextBlob(demo)

print(blob.sentiment)



scores = analyzer.polarity_scores(demo)

print(scores)
from textblob import TextBlob

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()



def nlp_pol_subj(text):

    '''Carries out sentiment analysis on text

    Args:

    * x - (str) Text block to be analysed

     

    Return

    * Tuple of length 3 containing the polarity, subjectivity and compound score

    '''

    

    blob = TextBlob(text)

    scores = analyzer.polarity_scores(text)

    return blob.sentiment.polarity, blob.sentiment.subjectivity, scores['compound']
## This is the code we would use to run this but it will be too slow for a meetup, so we will have this commented out for the moment

nlp_df = data['text'].map(nlp_pol_subj).apply(pd.Series, index = ["Polarity", "Subjectivity","Compound"])

data = pd.concat([data, nlp_df], axis = 1)
data.to_pickle("data.pkl")
import pandas as pd

data = pd.read_pickle("../input/stack-overflow-output/data.pkl")

data.creation_date = [pd.to_datetime(p) for p in data.creation_date.values]
data.head(5)
from matplotlib import dates

import seaborn as sns

import numpy as np

from sklearn import preprocessing



# convert the user id into a string for grouping

data['person'] = data['owner_user_id'].apply(str)



# conver the date into a numeric

data['datenum'] = dates.date2num(data['creation_date'])

data['year'] = [y.year for y in data['creation_date']]

data['hour'] = [y.hour for y in data['creation_date']]

data['minute'] = [y.minute for y in data['creation_date']]



# also let's scale the datenum and the score

data['datenum_scaled'] = preprocessing.scale(data['datenum'])

data['score_scaled'] = preprocessing.scale(data['score'])



# lastly let's give each person an id from 0 - 9 for the 10 python and 10 r answerers

classnames, indices = np.unique(data['person'], return_inverse=True)

data['uid_by_language'] = indices

r_values = data.loc[data['language']=="r","uid_by_language"].unique()

python_values = data.loc[data['language']=="python","uid_by_language"].unique()



new_r = [r_values.tolist().index(x) if x in r_values else None for x in data.loc[data['language'] == "r", 'uid_by_language'].values ] 

data.loc[data['language']=="r","uid_by_language"] = new_r



new_python = [python_values.tolist().index(x) if x in python_values else None for x in data.loc[data['language'] == "python" , 'uid_by_language'].values ] 

data.loc[data['language']=="python","uid_by_language"] = new_python
%matplotlib inline

import matplotlib.pyplot as plt

from matplotlib import dates

import seaborn as sns

from sklearn import preprocessing



pl = sns.lmplot(x="datenum_scaled", y="Compound", data=data, col = "language", height=4, sharex=False)
sns.lmplot(x="datenum_scaled", y="Compound", data=data, col = "language", fit_reg=True, x_estimator=np.mean, x_bins=20, sharex=False)
#!conda install -c districtdatalabs yellowbrick

import yellowbrick

from sklearn.linear_model import Ridge

from yellowbrick.regressor import ResidualsPlot

from sklearn.linear_model import LinearRegression



# construct our linear regression model

model = LinearRegression(fit_intercept=True)

x = data.datenum_scaled

y = data.Compound



# fit our model to the data

model.fit(x[:, np.newaxis], y)





# Instantiate the linear model and visualizer

visualizer = ResidualsPlot(model = model)



visualizer.fit(x[:, np.newaxis], y)  # Fit the training data to the model

visualizer.poof()                    # Draw/show/poof the data
violins = sns.catplot(x="person", y="Compound", data=data, col='language', kind = "violin", width=6)

violins.set_xticklabels(rotation=90)
sns.lmplot(x="datenum_scaled", y="Compound", data=data, 

           row = "uid_by_language", col = "language", height=4, 

           fit_reg=True, x_estimator=np.mean, x_bins=20, sharex=False)
import statsmodels.api as sm

import statsmodels.formula.api as smf



# construct our mixed effect model, with a random slope and intercept for each person

md = smf.mixedlm("Compound ~ 0 + datenum_scaled + language", data, groups=data["person"], re_formula="~datenum_scaled")

mdf = md.fit()

print(mdf.summary())
# construct our mixed effect model, with a random slope and intercept for each person

md = smf.mixedlm("Compound ~ 0 + datenum_scaled*language", data, groups=data["person"], re_formula="~datenum_scaled")

mdf = md.fit()

print(mdf.summary())
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score



languages = data.language.unique()

data = pd.concat([data,pd.get_dummies(data.language)],axis=1)



training_colummns = ["Subjectivity", "Compound", "r", "python", "hour"]

X = data[training_colummns]

y = data['accepted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = LogisticRegression(random_state=0, solver='lbfgs',

                         multi_class='multinomial')

clf.fit(X_train, y_train)

y_pred_log_reg = clf.predict(X_test)
print('f1 score {}'.format(f1_score(y_test, y_pred_log_reg, average='weighted')))

print('recall score {}'.format(recall_score(y_test, y_pred_log_reg, average='weighted')))

print('precision score {}'.format(precision_score(y_test, y_pred_log_reg, average='weighted')))
{key:value for key, value in zip(sorted(data['language'].unique()), f1_score(y_test, y_pred_log_reg, average=None))}
rf = RandomForestClassifier()

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print('f1 score {}'.format(f1_score(y_test, y_pred_rf, average='weighted')))

print('recall score {}'.format(recall_score(y_test, y_pred_rf, average='weighted')))

print('precision score {}'.format(precision_score(y_test, y_pred_rf, average='weighted')))

{key:value for key, value in zip(sorted(data['language'].unique()), f1_score(y_test, y_pred_rf, average=None))}
feats = {} # a dict to hold feature_name: feature_importance

for feature, importance in zip(X.columns, rf.feature_importances_):

    feats[feature] = importance #add the name/value pair 



importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})

importances.sort_values(by='Gini-importance').plot(kind='bar', rot=45)