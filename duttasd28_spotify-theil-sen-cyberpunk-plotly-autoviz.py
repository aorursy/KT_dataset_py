import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

url = 'https://raw.githubusercontent.com/Dutta-SD/Images_Unsplash/master/Kaggle/bews_self_edit_1.png'

from IPython.display import Image

Image(url)
## We might get some warnings later due to unknown characters. So we import this cell

import warnings

warnings.filterwarnings('ignore')
# Latin 1 encoding seems to be needed for this dataset

import pandas as pd

spData = pd.read_csv('../input/top50spotify2019/top50.csv',encoding = 'latin_1', index_col=0)

spData.head()
# Get information

spData.info()
# We will remove the + using regex(credits : Stack Overflow) :P

from wordcloud import WordCloud

from collections import Counter

allSongs = [trackname for trackname in spData['Track.Name']]

wc_dict = Counter(allSongs)



wordcloud = WordCloud(width=1000, height=500).generate_from_frequencies(wc_dict)

plt.figure(figsize = (20, 10))

plt.imshow(wordcloud)

plt.axis('off');
# We will remove the + using regex(credits : Stack Overflow) :P

from wordcloud import WordCloud

from collections import Counter

allSongs = [trackname for trackname in spData['Artist.Name']]

wc_dict = Counter(allSongs)



wordcloud = WordCloud(width=1000, height=500, background_color = 'white').generate_from_frequencies(wc_dict)

plt.figure(figsize = (20, 10))

plt.imshow(wordcloud)

plt.axis('off');
!pip -q --disable-pip-version-check install mplcyberpunk
# Plotting Libraries

import matplotlib.pyplot as plt

import seaborn as sns

import mplcyberpunk

%matplotlib inline
plt.style.use('cyberpunk')

plt.figure(figsize = (6, 30))

sns.barplot(data = spData, y = 'Track.Name', x= 'Popularity');

mplcyberpunk.make_lines_glow()
plt.figure(figsize = (20, 5));

sns.countplot(data = spData, x = 'Beats.Per.Minute', palette = 'winter');

mplcyberpunk.add_glow_effects()

plt.title('Count of Beats Per Minute');
## Danceability

##sns.set(style = 'whitegrid')



plt.style.use('cyberpunk')

plt.figure(figsize = (25, 5))

sns.pointplot(data = spData, x = 'Track.Name',y = 'Danceability',hue = 'Popularity', palette = 'inferno');

# remove lines

sns.despine(offset = 10)

# Rotate text by 90

plt.xticks(rotation = 90)

plt.title('Danceability');

# Move Legend

plt.legend(loc=8, ncol = 18);



mplcyberpunk.add_glow_effects()
# Genre plots using Plotly

import plotly.express as px

fig = px.pie(spData, values = 'Popularity', names='Genre', hole = 0.3)

fig.update_layout(annotations=[dict(text='Genre',font_size=20, showarrow=False)])
import plotly.graph_objects as go



# Generate Charts with Plotly

fig = go.Figure(data = [go.Scatter3d(

    x = spData['Energy'],

    y = spData['Loudness..dB..'],

    z = spData['Liveness'],

    text = spData['Track.Name'],  ## Additional texts which will be shown

    mode = 'markers',

    marker = dict(

    color = spData['Popularity'],

    colorbar_title = 'Popularity',

    colorscale = 'blues'

    )

)])



# Set variables and size

fig.update_layout(width=800, height=800, title = 'Energy, Liveness, Acousticness plot of Songs',

                  scene = dict(xaxis=dict(title='Energy'),

                               yaxis=dict(title='Liveness'),

                               zaxis=dict(title='Acousticness')

                               )

                 )



fig.show()
# Visualise the Valence

plt.figure(figsize = (10, 4))

plt.style.use('cyberpunk')

sns.distplot(spData['Valence.'],

             rug=True,

             hist_kws={"histtype": "stepfilled",

                       'linewidth' : 2,

                       'color':'r',

                      'alpha' : 0.11});

plt.title('Valence. Distribution');

mplcyberpunk.add_glow_effects()
# Energy plot

sns.set(style = 'white')

plt.figure(figsize = (10, 10));

sns.despine(offset = 10, left = True)

sns.jointplot(data = spData, 

              x = 'Energy',

              y = 'Loudness..dB..',

              kind = 'kde',

              color = '#3a4e0b',

              space = 1);
# Speechiness

fig = px.histogram(spData,

                   x="Speechiness.",

                  opacity = 1,

                  title = 'Speechiness Histogram',

                  color = 'Artist.Name')

fig.show()

# Length

fig = px.line(spData,

              y = 'Length.',

              x = 'Track.Name',

              title = 'Length of Popular Songs(Hover to See Name)')

fig.update_xaxes(visible=False)



fig.show()

# review the data once again

spData.tail()
# Retain numeric columns

spData2 = spData.drop(['Track.Name', 'Artist.Name', 'Genre'], axis = 1)

spData2.head()
sns.pairplot(data = spData2, corner = True)
plt.figure(figsize = (11, 11))

plt.title('Correlation between all features')

sns.heatmap(data = spData2.corr(),

            annot = True,

            cmap = 'copper_r',

            square = True,

           linewidths = 0.9);
!pip -q --disable-pip-version-check install autoviz
# Autoviz class

from autoviz.AutoViz_Class import AutoViz_Class



av = AutoViz_Class();

#av.AutoViz? if you need help
plt.style.use('cyberpunk')

spData3 = av.AutoViz('../input/top50spotify2019/top50.csv', dfte = spData2, max_cols_analyzed=50)
from sklearn.linear_model import TheilSenRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
X = spData2.iloc[: , :-1] # All rows, all columns upto column `-1` ie popularity

y = spData2.iloc[:, -1] # The last row , 'Popularity'
y.plot();
# Split The data

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = 42)
# define Model

lRegrssModel = TheilSenRegressor()



# Fit Model

lRegrssModel.fit(X_train, y_train)



# Get Predictions

y_preds = lRegrssModel.predict(X_val)
mean_squared_error(y_val, y_preds)
print('True\tPred')

for (trueVal, predVal) in zip(y_val, y_preds):

    print(f"{trueVal}\t{predVal:.3f}")