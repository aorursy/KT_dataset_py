import pandas as pd



import plotly_express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



from plotly.offline import init_notebook_mode

import plotly.io as pio

pio.templates.default = "plotly_white"

init_notebook_mode()
episodesData = pd.read_csv('../input/the-office-dataset/the_office_series.csv')

transcripts = pd.read_csv('../input/the-office-us-complete-dialoguetranscript/The-Office-Lines-V3.csv', encoding = 'latin-1')
episodesData.head()
transcripts.head()
averageDurationRating = episodesData.groupby(['Season'])[['Duration','Ratings']].mean().reset_index()

fig = px.scatter(averageDurationRating, x = 'Season', y = 'Ratings',trendline = 'lowess',size = 'Duration',

                 title = '<b>Ratings over each Season</b>')

fig.show()
views_votes = episodesData.groupby('Season')[['Votes','Viewership']].mean().reset_index()

fig = make_subplots(rows = 1, cols = 2, horizontal_spacing=0.2,

                    subplot_titles=['Average Number of Votes Each Season', 'Average Viewership Each Season'])



trace0 = go.Bar(

    x = views_votes['Votes'],

    y = views_votes['Season'],

    orientation = 'h',

    marker_color = px.colors.sequential.tempo[::-1],

    name = 'Votes'

)



trace1 = go.Bar(

    x = views_votes['Viewership'],

    y = views_votes['Season'],

    orientation = 'h',

    marker_color = px.colors.sequential.deep[::-1],

    name = 'Viewership'

)



fig.add_trace(trace0, row = 1, col = 1)

fig.add_trace(trace1, row = 1, col = 2)

fig.update_layout(showlegend = False)

fig.show()
episodesData['TotalRating'] = episodesData['Ratings'] * episodesData['Votes']

averageDurationRating = episodesData.groupby(['Season'])[['Duration','TotalRating']].mean().reset_index()

fig = px.scatter(averageDurationRating, x = 'Season', y = 'TotalRating',trendline = 'ols',size = 'Duration', 

                 title = '<b>Total Rating across Seasons</b>')

fig.show()
directedEpisodes = episodesData['Director'].value_counts().reset_index()

directedEpisodes.columns = ['Director','Number of Episodes']

directedEpisodes = directedEpisodes[directedEpisodes['Director'] != 'See full summary']



directorAvgRating = episodesData.groupby('Director')['Ratings'].mean().reset_index()

directorAvgRating = directorAvgRating[directorAvgRating['Director'] != 'See full summary']

directorAvgRating = directorAvgRating.sort_values(by = 'Ratings',ascending = False)



fig = make_subplots(rows = 1, cols = 2,

                    subplot_titles=['Number of Episodes Directed', 'Average Rating of Episodes'], 

                   horizontal_spacing=0.2)



trace0 = go.Bar(

    x = directedEpisodes['Number of Episodes'][:10],

    y = directedEpisodes['Director'][:10],

    orientation = 'h',

    name = 'Directors'

)



fig.add_trace(trace0, row = 1, col = 1)



trace1 = go.Bar(

    x = directorAvgRating['Ratings'][:10],

    y = directorAvgRating['Director'][:10],

    orientation = 'h',

    name = 'Ratings'

)



fig.add_trace(trace1, row = 1, col = 2)

fig.update_layout(showlegend = False)

fig.show()
castDirectors = ['Paul Lieberstein', 'B.J. Novak','Steve Carell', 'John Krasinski','Rainn Wilson','Mindy Kaling']

fig = px.bar(directorAvgRating[directorAvgRating['Director'].isin(castDirectors)],

            x = 'Ratings',y='Director',orientation='h',color='Ratings',color_continuous_scale='peach'

            )

fig.update_layout(coloraxis_showscale=False)

fig.show()
fig = make_subplots(rows = 3, cols = 2, specs = [[{"type":"table"}] * 2]*3,

                                subplot_titles = castDirectors,

                                horizontal_spacing=0.03,vertical_spacing = 0.07)

for i, director in enumerate(castDirectors):

    df = episodesData[episodesData['Director'] == director].sort_values(by = 'Ratings',ascending = False)

    trace = go.Table(header = dict(values = ['<b>Episode Name</b>','<b>Rating</b>']), cells = dict(values = [df['EpisodeTitle'], df['Ratings']]))

    fig.add_trace(trace, row = (i//2)+1 , col = (i%2)+1)

fig.update_layout(height = 500,margin=dict(l=80, r=80, t=100, b=20),

                  title = { 'text' : '<b>Cast Directed Episodes</b>', 'x' : 0.5})

fig.show()
def countNumberofGuestStars(guestStars):

    if(guestStars == ''):

        return 0

    else:

        stars = guestStars.split(',')

        return len(stars)



episodesData['GuestStars'] = episodesData['GuestStars'].fillna('')

episodesData['Number of Guests'] = episodesData['GuestStars'].apply(lambda x: countNumberofGuestStars(str(x)))
df = episodesData.sort_values(by = 'Number of Guests', ascending = False)

trace = go.Table(header = dict(values = ['<b>Episode Name</b>','<b>Rating</b>','<b>Number of Guests</b>']), 

                 cells = dict(values = [df['EpisodeTitle'][:3], df['Ratings'][:3], df['Number of Guests'][:3]]))

fig = go.Figure([trace])

fig.update_layout(title = '<b>Top 3 Episodes with the Most number of Guest Stars',height = 300)

fig.show()
numOfGuests = episodesData.groupby('Season')['Number of Guests'].mean().reset_index()

fig = px.bar(numOfGuests, x = 'Season', y = 'Number of Guests', title = '<b>Average Number of Guests per Season</b>')

fig.show()
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

writerDf = pd.DataFrame({})

writerDf['WriterList'] = episodesData['Writers'].apply(lambda x: [y.strip() for y in x.split('|')])



mlb.fit(writerDf['WriterList'])

#creating columns = the classes of the multilabelbinarizer

writerDf[mlb.classes_] = mlb.transform(writerDf['WriterList'])

writerDf.drop('WriterList',axis = 1, inplace = True)
writerEpisodes = writerDf.sum().reset_index()

writerEpisodes.columns = ['Writer', 'Number of Episodes']

writerEpisodes = writerEpisodes.sort_values(by = 'Number of Episodes')

fig = px.bar(writerEpisodes,x = 'Number of Episodes', y = 'Writer', title = 'Number of Episodes Written',height  = 1000)

fig.show()