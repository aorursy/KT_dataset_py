import numpy as np

import pandas as pd

import re

from plotly.subplots import make_subplots

import plotly.graph_objects as go

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
# Drop unnamed column (index?)

df = df.drop(df.columns[0],1)

df['Easy Apply'] = df['Easy Apply'].str.replace('True','1').str.replace('-1','0') # Convert Easy Apply (True=1,False=0)
df['Job Title'] = df['Job Title'].str.replace('[^\w\s]','') # Remove Punctuation

df['Job Title'] = df['Job Title'].str.replace('[0-9]+','') # Remove isolated digits
# UPDATE AND ANALYSE

text_rem = ['Quality',

            ' Center on Immigration and Justice CIJ',

            ' Insights Analytics Team Customer',

            ' Merchant Health',

           'FPA'] # Remove Irregular job title subtexts

for t in text_rem:

    df['Job Title'] = df['Job Title'].str.replace(t,'')

    

df['Job Title'].to_csv('mycsvfile.csv',index=False)
# Restructure 'Salary Estimate'

test = df["Salary Estimate"].str.split("-", n = 1, expand = True) 

df['Min Salary'] = test[0].str.replace('[^0-9^-]+','')

df['Max Salary'] = test[1].str.replace('[^0-9^-]+','')

df = df.drop('Salary Estimate',1)

df['Min Salary'] = pd.to_numeric(df['Min Salary'])

df['Max Salary'] = pd.to_numeric(df['Max Salary'])
df = df.replace(-1, np.nan) # Numerical Columns

df = df.replace('-1', np.nan) # String Columns
# Remove the (ratings?) from company name

df["Company Name"] = df["Company Name"].str.split("\n", n = 1, expand = True)[0]
# Restructure Size

new_size = df['Size'].str.replace('1 to 50 employees','Start-up').replace('51 to 200 employees','Small').replace('201 to 500 employees','Medium').replace('501 to 1000 employees','Big').replace('1001 to 5000 employees','Very Big').replace('5001 to 10000 employees','Huge').replace('10000\+ employees','Titanic')

df['Size'] = new_size
# Fix Revenue

new_rev = df['Revenue'].str.replace('Less than $1 million (USD)', '<$1M').replace('$1 to $5 million (USD)', '$1-5M').replace('$5 to $10 million (USD)', '$5-10M').replace('$10 to $25 million (USD)', '$10-25M' ).replace('$25 to $50 million (USD)', '$25-50M').replace('$50 to $100 million (USD)', '$50-100M').replace('$100 to $500 million (USD)', '$100-500M').replace('$500 million to $1 billion (USD)',  '$0.5-1B').replace('$1 to $2 billion (USD)',  '$1-2B').replace('$2 to $5 billion (USD)', '$2-5B').replace('$5 to $10 billion (USD)', '$5-10B').replace('$10+ billion (USD)',  '>$10B').replace('Unknown / Non-Applicable',  'NaN')

df['Revenue'] = new_rev
most_pos = df.groupby(by=['Company Name','Easy Apply'])['Job Title'].count().reset_index().sort_values(by=['Company Name'],ascending=False).rename(columns = {'Job Title': 'Positions'}, inplace = False)

most_pos_easy = most_pos[ most_pos['Easy Apply'] == '1' ].sort_values(by=['Positions'],ascending=False).head(7)

most_pos_no_easy = most_pos[ most_pos['Easy Apply'] == '0' ].sort_values(by=['Positions'],ascending=False).head(7)
# Most Open Roles

fig = go.Figure(data=[

    go.Bar(name='Easy Apply', 

           x = most_pos_easy['Company Name'], 

           y = most_pos_easy['Positions']

          ),

    go.Bar(name='No Easy Apply', 

           x = most_pos_no_easy['Company Name'],

           y = most_pos_no_easy['Positions']

          )

])



fig.update_layout(

    template="plotly_dark",

    title_text = 'Open Positions, by Company',

    barmode='group'

)



fig.show()
avg_rating = df.groupby(by=['Sector','Easy Apply'])['Rating'].mean().reset_index().sort_values(by=['Sector','Easy Apply'],ascending=False)

avg_rating_easy = avg_rating[ avg_rating['Easy Apply'] == '1' ].sort_values(by=['Sector'],ascending=False)

avg_rating_no_easy = avg_rating[ avg_rating['Easy Apply'] == '0' ].sort_values(by=['Sector'],ascending=False)
# Most Open Roles

fig = go.Figure(data=[

    go.Bar(name='Easy Apply', 

           x = avg_rating_easy['Sector'], 

           y = avg_rating_easy['Rating']

          ),

    go.Bar(name='No Easy Apply', 

           x = avg_rating_no_easy['Sector'],

           y = avg_rating_no_easy['Rating']

          )

])



fig.update_layout(

    template="plotly_dark",

    title_text = 'Sector Ratings, by Easy Apply',

    barmode='group'

)



fig.show()
avg_min = df.groupby(by=['Industry'])['Min Salary'].mean().reset_index().sort_values(by=['Min Salary'],ascending=False)

avg_max = df.groupby(by=['Industry'])['Max Salary'].mean().reset_index().sort_values(by=['Max Salary'],ascending=False)



# Most Open Roles

fig = go.Figure(data=[

    go.Bar(name='Max Salary', 

           x = avg_max['Industry'],

           y = avg_max['Max Salary']

          ),

    go.Bar(name='Min Salary', 

           x = avg_min['Industry'], 

           y = avg_min['Min Salary']

          ),

])



fig.update_layout(

    template="plotly_dark",

    title_text = 'Avg Salary, by Industry',

    barmode='group'

)



fig.show()
test = df.groupby(by=['Company Name','Competitors'])['Job Title'].count().reset_index()

test['Competitors'] = test['Competitors'].str.strip()

test = test.sort_values(by=['Company Name'],ascending=False)

test = test['Competitors'].str.split(",", expand = True) 

test = test.values.tolist()



flat_list = [item for sublist in test for item in sublist]

flat_list = ['Not Listed' if x is np.nan else x for x in flat_list]

flat_list = ['Not Listed' if x is None else x for x in flat_list]

flat_list = [x.strip(' ') for x in flat_list]

flat_list.sort()



from itertools import groupby

data = [(key, len(list(group))) for key, group in groupby(flat_list)] or {key: len(list(group)) for key, group in groupby(flat_list)}

competitors = pd.DataFrame.from_records(data).rename(columns = {0: 'Competitor',1:'Count'}, inplace = False).sort_values(by='Count', ascending=False).head(15)

competitors = competitors.iloc[1:]
# Most Competitors

fig = go.Figure(data=[

    go.Bar(name='Max Salary', 

           x = competitors['Competitor'],

           y = competitors['Count']

          )

])



fig.update_layout(

    template="plotly_dark",

    title_text = 'Most Competitors',

    barmode='group'

)



fig.show()
df['Avg Salary'] = (df['Min Salary'] + df['Max Salary'])/2

df['State'] = df['Location'].str[-2:]

state_salary = df.groupby(by=['State'])['Avg Salary'].mean().reset_index()



import plotly.graph_objects as go



fig = go.Figure(data=go.Choropleth(

    locations=state_salary['State'],

    z = state_salary['Avg Salary'].astype(float),

    locationmode = 'USA-states',

    colorbar_title = "000's $",

    colorscale = 'Reds',

    zmin=37,

    zmax=89,

))



fig.update_layout(

    template="plotly_dark",

    title_text = 'Average Salary, by State',

    geo_scope='usa',

)



fig.show()
text = ''.join([i for i in df['Job Description'] if not i.isdigit()]) # Remove Digits

test = []

for k in text.split("\n"):

    test.append( re.sub(r"[^a-zA-Z0-9]+", ' ', k))
text = test

text = ''.join(text)

text = re.sub(r'==.*?==+', '', text)



from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

  

all_stopwords = stopwords.words('english')

word_tokens = word_tokenize(text)

filtered_sentence = [w for w in word_tokens if not w in all_stopwords] 
def plot_cloud(wordcloud):

    plt.figure(figsize=(20, 15))

    plt.imshow(wordcloud) 

    plt.axis("off");
STOPWORDS.update(['play','will','within','one','use','working','provide','benefit','partner','internal','external',

                     'high','protected','across','written','need','care','help','must','area','office','state','related',

                     'people','member','may','well','using','etc','make','year','us','change','benefits','part','national',

                 'access','time','applications','able','issue','task','practice','duties','candidate','maintain','day','field',

                  'meet','ensure','decision','best','sexual','initiative','gender','world','relevant','race','preferred',

                  'looking','re','document','ad','self','highly','include','veteran','key','source','request','full','result',

                  'build','provides','technique','Governance','end','color','years','work','including'])

wordcloud = WordCloud(

    width = 1500,

    height = 1000, 

    random_state=1, 

    background_color='salmon', 

    colormap='Pastel2', 

    collocations=False, 

    stopwords = STOPWORDS).generate(text)

# Plot

plot_cloud(wordcloud)
!pip install vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



analyser = SentimentIntensityAnalyzer()



def sentiment_scores(sentence,full_list): 

    sid_obj = SentimentIntensityAnalyzer() 

    sentiment_dict = sid_obj.polarity_scores(sentence) 

      

    t = [(sentiment_dict['neg']*100), (sentiment_dict['neu']*100), (sentiment_dict['pos']*100), (sentiment_dict['compound']*100)]

    full_list.append(t)
sentiments = []

for job in df['Job Description']:

    job.replace('\n','')

    re.sub('\d', '', job)

    sentiment_scores(job,sentiments)

    

sentiment_df = pd.DataFrame(sentiments,columns=['negative','neutral','positive','compound'])

sentiment_df = pd.concat([df, sentiment_df], axis=1)

sentiment_df['Founded'] = sentiment_df['Founded'].fillna(0)
# Round to decade of year founded

import math

years = []

for year in sentiment_df['Founded']:

    years.append( int(math.ceil( year / 10.0)) * 10 ) 

sentiment_df['Founded'] = years
industry_sent = sentiment_df.groupby(by=['Industry'])['compound'].mean().reset_index().sort_values(by=['compound'],ascending=True).head(10)

company_sent = sentiment_df.groupby(by=['Size'])['compound'].mean().reset_index().sort_values(by=['compound'],ascending=True).head(10)

founded_sent = sentiment_df.groupby(by=['Founded'])['compound'].mean().reset_index().sort_values(by=['compound'],ascending=True)

founded_sent = founded_sent.drop(0) # drop founded year of 0

ownership_sent = sentiment_df.groupby(by=['Type of ownership'])['compound'].mean().reset_index().sort_values(by=['compound'],ascending=True)



fig = make_subplots(rows=2, 

                    cols=2,

                    subplot_titles=("Score by Company Size", 'Score by Founded Decade',"Score by Industry (Lowest 10)",'Score by Ownership Type'),

                   )



fig.add_trace(

    go.Bar(name='Company', 

           x = company_sent['Size'],

           y = company_sent['compound']

          ),

    row=1, col=1

)



fig.add_trace(

    go.Bar(name='Industry', 

           x = industry_sent['Industry'], 

           y = industry_sent['compound']

          ),

    row=2, col=1

)



fig.add_trace(

    go.Bar(name='Founded', 

           x = founded_sent['Founded'], 

           y = founded_sent['compound']

          ),

    row=1, col=2

)



fig.add_trace(

    go.Bar(name='Ownership', 

           x = ownership_sent['Type of ownership'], 

           y = ownership_sent['compound']

          ),

    row=2, col=2

)



fig.update_yaxes(range=[92, 100], row=1, col=1,)

fig.update_yaxes(range=[92, 100], row=1, col=2,)

fig.update_yaxes(range=[62, 100], row=2, col=1,)

fig.update_yaxes(range=[89, 100], row=2, col=2,)



fig.update_layout(

    template="plotly_dark",

    margin=dict(l=50, r=50, t=80, b=80),

    title_text = 'Sentiment of Job Description (Compound Score)',

    height=800,

)



fig.show()
test = sentiment_df[ sentiment_df['compound'].astype(int) >=80 ]

test['compound'] = pd.cut(x = test['compound'],

                        bins = range(80,101), 

                        labels =  range(80,100))

test = test.groupby(by=['compound'])['Rating'].mean().reset_index()



# Most Open Roles

fig = go.Figure(data=[

    go.Bar(name='Rating by Sentiment', 

           x = test['compound'], 

           y = test['Rating']

          ),

])



fig.update_layout(

    template="plotly_dark",

    title_text = 'Company Ratings, by Compound Sentiment Score',

    barmode='group'

)



fig.show()