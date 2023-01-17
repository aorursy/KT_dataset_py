# pip install squarify
#Basic exploratory libraries
import numpy as np
import pandas as pd

#Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import squarify as sq

#Word cloud libraries
from os import path
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator

#Library to supress unnecessary warnings
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
kiva_loan = pd.read_csv('/kaggle/input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
kiva_themes = pd.read_csv('/kaggle/input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv')
kiva_mpi = pd.read_csv('/kaggle/input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')

kiva_lt = kiva_loan.merge(kiva_themes, on = 'id')
philippines_loan = kiva_lt[kiva_lt['country'] == 'Philippines']
philippines_mpi = kiva_mpi[kiva_mpi['country'] == 'Philippines']

philippines = philippines_loan.merge(philippines_mpi, on = 'country')
philippines.head(1)
philippines.columns
philippines = philippines[['region_x', 'sector', 'activity', 'Loan Theme Type', 'use', 'date', 'loan_amount',
                           'funded_amount', 'borrower_genders', 'lender_count', 'term_in_months',
                           'repayment_interval', 'geo', 'lat', 'lon']]
philippines.columns.values[0] = 'region'
philippines.columns.values[3] = 'theme'
arts = philippines[philippines['sector'] == 'Arts']
arts.head(2)
arts.columns
arts.count()
sns.set_context('talk')
sns.jointplot(x = 'loan_amount', y = 'lender_count', data = arts)
plt.figure(figsize = (20,5))
plt.xticks(rotation = 75)

sns.distplot(arts['lender_count'], bins = 10)
plt.figure(figsize = (20,5))
plt.xticks(rotation = 75)

sns.distplot(arts['loan_amount'], bins = 10)
arts_activity = arts.groupby(['borrower_genders', 'activity'])['loan_amount'].sum().sort_values(ascending = False).reset_index()
arts_activity
sizes = arts_activity['loan_amount']
label = np.array(arts_activity['activity']) + '\n' + sizes.astype('str')

plt.style.use('ggplot')
plt.figure(figsize = (20,15))

sq.plot(sizes = sizes, label = label, alpha = 0.6, text_kwargs={'fontsize':10})
male = len(arts[arts['borrower_genders'] == 'male'])
female = len(arts[arts['borrower_genders'] == 'female'])

print('There are {} females versus {} males who have received loans in the arts in Philippines.\
 A difference of {}.'.format(female, male, (female - male)))
fig = px.bar(arts_activity, x = 'activity', y = 'loan_amount',
            hover_data = ['loan_amount', 'borrower_genders'], color = 'loan_amount',
            labels = {'loan_amount': 'Loan Amount'}, height = 500)

fig.show()
arts_repayment = arts.groupby(['borrower_genders', 'activity', 'repayment_interval'])['loan_amount'].sum()\
.sort_values(ascending = False).reset_index()
arts_repayment
fig = px.bar(arts_repayment, x = 'repayment_interval', y = 'loan_amount',
            hover_data = ['loan_amount', 'borrower_genders', 'activity'], color = 'loan_amount',
            labels = {'loan_amount': 'Loan Amount'}, height = 500)

fig.show()
mapbox_access_token = 'pk.eyJ1IjoiZ2l0b25nYSIsImEiOiJjazBueDZsN2cwNGE3M21xcnl0bGg0cWUxIn0.9-jwOGyzRkCFbcPfafeoMw'

size = arts['loan_amount'] / 100

fig = go.Figure(go.Scattermapbox(
    lat = arts['lat'],
    lon = arts['lon'],
    mode = 'markers',
    marker = go.scattermapbox.Marker(
        size = size,
        color = 'red',
            ),
    text = arts['region'],
        )
    )

fig.update_layout(
    autosize = True,
    hovermode = 'closest',
    mapbox = go.layout.Mapbox(
        accesstoken = mapbox_access_token,
        bearing = 0,
        center = go.layout.mapbox.Center(
            lat = 12.2445, 
            lon = 125.0388,
            ),
        pitch = 0,
        zoom = 4,
        )
    )

fig.show()
text = " ".join(review for review in arts['use'])
wordcloud = WordCloud(max_words=1000, background_color="white").generate(text)

plt.figure(figsize=[15,10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()