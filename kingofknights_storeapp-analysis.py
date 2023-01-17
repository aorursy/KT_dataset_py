import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.offline as plt
import plotly
plotly.offline.init_notebook_mode(connected=True)
#storeApp = pd.read_csv("AppleStore.csv", index_col='id')
storeApp = pd.read_csv("../input/AppleStore.csv", index_col='id')
storeApp.drop('Unnamed: 0', axis=1, inplace=True)
storeApp.head()
storeApp.info()
storeApp.describe()
storeApp.drop('currency', axis=1, inplace=True)
trace = [go.Histogram(x=storeApp.prime_genre, opacity=0.75)]
layout = go.Layout(title='Distribution of prime_genre',
    xaxis=dict(title='Type' ),
    yaxis=dict(title='Count'),
    bargap=0.2,
    bargroupgap=0.1)
trace = go.Figure(data=trace, layout=layout)
#plotly.plotly.iplot(trace)
plt.iplot(trace)
storeApp['F2P'] = storeApp.price == 0
maping = {
    True:'YES',
    False:'NO'
}
storeApp.F2P = storeApp.F2P.map(maping)
storeApp.drop('price', axis=1, inplace=True)
trace = [go.Histogram(x=storeApp.F2P, opacity=0.75, histnorm='probability')]
layout = go.Layout(title='Percentage of Paid vs Free App',
    xaxis=dict(title='F2P Yes or Not' ),
    yaxis=dict(title='Precentage %'),
    bargap=0.2,
    bargroupgap=0.1)
trace = go.Figure(data=trace, layout=layout)
#plotly.plotly.iplot(trace)
plt.iplot(trace)
storeApp['catogory'] = storeApp.prime_genre == 'Games'
maping = {
    True:'Game',
    False:'Other'
}
storeApp.catogory = storeApp.catogory.map(maping)
figure = ff.create_facet_grid(
    storeApp,
    x='user_rating_ver',
    facet_col='F2P',
    facet_row='catogory',
    facet_row_labels='C:',
    facet_col_labels='F2P :',
    trace_type='histogram',
    marker={'color': 'rgb(255,165,128)'}
)

#plotly.plotly.iplot(figure)
plt.iplot(figure)
Games = storeApp[storeApp.prime_genre == 'Games']
NonGames = storeApp[storeApp.prime_genre != 'Games']
storeApp.drop('prime_genre', axis=1, inplace=True)
TopFreeGames = Games[Games.F2P == 'YES']
TopPaidGames = Games[Games.F2P == 'NO']
TopFreeNonGames = NonGames[NonGames.F2P == 'YES']
TopPaidNonGames = NonGames[NonGames.F2P == 'NO']
def getTop(data):
    data.sort_values(['user_rating', 'rating_count_tot'],ascending=False, inplace=True)
    data = data.head()
    print(data[['track_name', 'user_rating', 'rating_count_tot']])
getTop(TopFreeGames)

getTop(TopPaidGames)
getTop(TopFreeNonGames)

getTop(TopPaidNonGames)
MB = 1024 * 1024
storeApp.loc[ storeApp['size_bytes'] <= MB * 10, 'size_bytes'] = 10
storeApp.loc[(storeApp['size_bytes'] > MB * 10) & (storeApp['size_bytes'] <= MB * 50), 'size_bytes'] = 50
storeApp.loc[(storeApp['size_bytes'] > MB * 50) & (storeApp['size_bytes'] <= MB * 100), 'size_bytes'] = 100
storeApp.loc[(storeApp['size_bytes'] > MB * 100) & (storeApp['size_bytes'] <= MB * 200), 'size_bytes'] = 200
storeApp.loc[(storeApp['size_bytes'] > MB * 200) & (storeApp['size_bytes'] <= MB * 500), 'size_bytes'] = 500
storeApp.loc[(storeApp['size_bytes'] > MB * 500) & (storeApp['size_bytes'] <= MB * 1000), 'size_bytes'] = 1000
storeApp.loc[ storeApp['size_bytes'] > MB * 1000, 'size_bytes'] = 2000
storeApp.size_bytes = storeApp.size_bytes.astype('str') + 'MB'
trace = [go.Histogram(x=storeApp.size_bytes, opacity=0.75)]
layout = go.Layout(title='Distribution of size_bytes',
    xaxis=dict(title='Size' ),
    yaxis=dict(title='Count'),
    bargap=0.2,
    bargroupgap=0.1)
trace = go.Figure(data=trace, layout=layout)
#plotly.plotly.iplot(trace)
plt.iplot(trace)
trace = [go.Histogram(x=storeApp.cont_rating, opacity=0.75)]
layout = go.Layout(title='Distribution of Content Rating',
    xaxis=dict(title='Rating' ),
    yaxis=dict(title='Count'),
    bargap=0.2,
    bargroupgap=0.1)
trace = go.Figure(data=trace, layout=layout)
#plotly.plotly.iplot(trace)
plt.iplot(trace)
storeApp.loc[ storeApp['sup_devices.num'] <=  5, 'sup_devices.num'] = 5
storeApp.loc[(storeApp['sup_devices.num'] > 5) & (storeApp['sup_devices.num'] <= 10), 'sup_devices.num'] = 10
storeApp.loc[(storeApp['sup_devices.num'] > 10) & (storeApp['sup_devices.num'] <= 15), 'sup_devices.num'] = 15
storeApp.loc[(storeApp['sup_devices.num'] > 15) & (storeApp['sup_devices.num'] <= 20), 'sup_devices.num'] = 20
storeApp.loc[(storeApp['sup_devices.num'] > 20) & (storeApp['sup_devices.num'] <= 25), 'sup_devices.num'] = 25
storeApp.loc[(storeApp['sup_devices.num'] > 25) & (storeApp['sup_devices.num'] <= 30), 'sup_devices.num'] = 30
storeApp.loc[(storeApp['sup_devices.num'] > 30) & (storeApp['sup_devices.num'] <= 35), 'sup_devices.num'] = 35
storeApp.loc[(storeApp['sup_devices.num'] > 35) & (storeApp['sup_devices.num'] <= 40), 'sup_devices.num'] = 40
storeApp.loc[ storeApp['sup_devices.num'] > 40, 'sup_devices.num'] = 45
figure = ff.create_facet_grid(
    storeApp,
    x='sup_devices.num',
    facet_col='F2P',
    facet_row='catogory',
    facet_row_labels='C:',
    facet_col_labels='F2P :',
    trace_type='histogram',
    marker={'color': 'rgb(255,165,128)'}
)

#plotly.plotly.iplot(figure)
plt.iplot(figure)
maping = {1:'YES', 0:'NO'}
storeApp.vpp_lic = storeApp.vpp_lic.map(maping)
trace = [go.Histogram(x=storeApp.vpp_lic, opacity=0.75, histnorm='probability')]
layout = go.Layout(title='Percentage of App has vpp_lic',
    xaxis=dict(title='vpp_lic Yes or Not' ),
    yaxis=dict(title='Precentage %'),
    bargap=0.2,
    bargroupgap=0.1)
trace = go.Figure(data=trace, layout=layout)
#plotly.plotly.iplot(trace)
plt.iplot(trace)
storeApp.loc[ storeApp['lang.num'] <=  5, 'lang.num'] = 5
storeApp.loc[(storeApp['lang.num'] > 5) & (storeApp['lang.num'] <= 10), 'lang.num'] = 10
storeApp.loc[(storeApp['lang.num'] > 10) & (storeApp['lang.num'] <= 15), 'lang.num'] = 15
storeApp.loc[(storeApp['lang.num'] > 15) & (storeApp['lang.num'] <= 20), 'lang.num'] = 20
storeApp.loc[(storeApp['lang.num'] > 20) & (storeApp['lang.num'] <= 25), 'lang.num'] = 25
storeApp.loc[(storeApp['lang.num'] > 25) & (storeApp['lang.num'] <= 30), 'lang.num'] = 30
storeApp.loc[(storeApp['lang.num'] > 30) & (storeApp['lang.num'] <= 35), 'lang.num'] = 35
storeApp.loc[(storeApp['lang.num'] > 35) & (storeApp['lang.num'] <= 40), 'lang.num'] = 40
storeApp.loc[ storeApp['lang.num'] > 40, 'lang.num'] = 45
figure = ff.create_facet_grid(
    storeApp,
    x='lang.num',
    facet_col='F2P',
    facet_row='catogory',
    facet_row_labels='C:',
    facet_col_labels='F2P :',
    trace_type='histogram',
    marker={'color': 'rgb(255,165,128)'}
)

#plotly.plotly.iplot(figure)
plt.iplot(figure)
trace = [go.Histogram(x=storeApp['ipadSc_urls.num'], opacity=0.75)]
layout = go.Layout(title='Distribution of Screenshot displayed',
    xaxis=dict(title='Number of Screenshot' ),
    yaxis=dict(title='Count'),
    bargap=0.2,
    bargroupgap=0.1)
trace = go.Figure(data=trace, layout=layout)
#plotly.plotly.iplot(trace)
plt.iplot(trace)
storeApp.head()