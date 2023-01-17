import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import plotly.graph_objs as go
import plotly.tools as tools
import plotly.offline as ply
ply.init_notebook_mode(connected=True)
import seaborn as sns 
color = sns.color_palette()
import matplotlib.pyplot as plt
%matplotlib inline
# Any results you write to the current directory are saved as output.
print(os.listdir("../input/"))
locations = pd.read_csv('../input/kiva_mpi_region_locations.csv')
locations.head()
Y=locations.country.value_counts().index[::-1]
X=locations.country.value_counts().values[::-1]
data = go.Bar(
    x = X,
    y = Y,
    orientation = 'h',
    marker=dict(
        color=X,
        colorscale = 'Jet',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Countries Around the world Kiva fund with Loans',
    width=800,
    height=1200,
    )
figure = go.Figure(data=[data], layout=layout)
ply.iplot(figure, filename="LoansbyKiva")
data = [dict(
  type = 'scatter',
  x = X,
  y = Y,
  mode = 'markers',
  transforms = [dict(
    type = 'groupby',
    groups = X
  )]
)]

ply.iplot({'data': data}, validate=False)
Y=locations.world_region.value_counts().index[::-1]
X=locations.world_region.value_counts().values[::-1]
data = go.Bar(
    x = Y,
    y = X,
    orientation = 'v',
    marker=dict(
        color=X,
        colorscale = 'Jet',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Regions Kiva Fund',
    width=700,
    height=500,
    )
figure = go.Figure(data=[data], layout=layout)
ply.iplot(figure, filename="PerRegionLoans")
map_df = pd.DataFrame(locations['country'].value_counts()).reset_index()
map_df.columns=['country', 'loans']
map_df = map_df.reset_index().drop('index', axis=1)
data = [ dict(
        type = 'choropleth',
        locations = map_df['country'],
        locationmode = 'country names',
        z = map_df['loans'],
        text = map_df['country'],
        colorscale = [[0,"rgb(5, 50, 172)"],[0.85,"rgb(40, 100, 190)"],[0.9,"rgb(70, 140, 245)"],
            [0.94,"rgb(90, 160, 245)"],[0.97,"rgb(106, 177, 247)"],[1,"rgb(220, 250, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Number of Loans'),
      ) ]

layout = dict(
    title = 'Number of Loans Per Country',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

figure = dict( data=data, layout=layout )
ply.iplot(figure, validate=False, filename='countryandloans')
trace = []
for name, group in locations.groupby("country"):

    trace.append ( 
        go.Box(
            x=group["MPI"].values,
            name=name
        )
    )
layout = go.Layout(
    title='Multidimensional Poverty Index(MPI) for earch Country',
    width = 1000,
    height = 2000
)
figure = go.Figure(data=trace, layout=layout)
ply.iplot(figure, filename="ContryMPIndex")
trace = []
for name, group in locations.groupby("world_region"):

    trace.append ( 
        go.Box(
            y=group["MPI"].values,
            name=name
        )
    )
layout = go.Layout(
    title='Multidimensional Poverty Index(MPI) for each Region',
    width = 750,
    height = 800,
    orientation= 'v',
)
figure = go.Figure(data=trace, layout=layout)
ply.iplot(figure, filename="WorldRegionMPI")
loans = pd.read_csv('../input/kiva_loans.csv')
loans.head()
Y=loans.activity.value_counts().index[::-1]
X=loans.activity.value_counts().values[::-1]
data = go.Bar(
    x = X,
    y = Y,
    orientation = 'h',
    marker=dict(
        color=X,
        colorscale = 'Jet',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Loans by Activity',
    width=850,
    height=1000,
    )
figure = go.Figure(data=[data], layout=layout)
ply.iplot(figure, filename="LoansSeries")
Y=loans.sector.value_counts().index[::-1]
X=loans.sector.value_counts().values[::-1]
data = go.Bar(
    x = X,
    y = Y,
    orientation = 'h',
    marker=dict(
        color=X,
        colorscale = 'Jet',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Loans by Sector',
    width=900,
    height=600,
    )
figure = go.Figure(data=[data], layout=layout)
ply.iplot(figure, filename="SectorLoans")
from wordcloud import WordCloud
wordcloud = WordCloud(width=1440, height=1080).generate(" ".join(loans.use.astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(wordcloud)
plt.axis('off')
loans.info()
loans.date = pd.to_datetime(loans.date)
from wordcloud import WordCloud
wordcloud = WordCloud(width=1440, height=1080).generate(" ".join(loans.region.astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(wordcloud)
plt.axis('off')
# borrower_genders 	repayment_interval
Y=loans.repayment_interval.value_counts().index[::-1]
X=loans.repayment_interval.value_counts().values[::-1]
data = go.Bar(
    x = X,
    y = Y,
    orientation = 'h',
    marker=dict(
        color=X,
        colorscale = 'Jet',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Loans distribution by Gender',
    width=900,
    height=600,
    )
fig = go.Figure(data=[data], layout=layout)
ply.iplot(fig, filename="Gender")
placeholder = list(loans.borrower_genders)
loans.borrower_genders = ['female' if str(gender).find('female') else 'male' for gender in loans.borrower_genders ]
loans.head()
import plotly.figure_factory as ff
group_labels = ['Terms in Months', 'Lender count']
trace1 = go.Histogram(
    x=loans.term_in_months[:1000],
    opacity=0.75,
    histnorm='count',
    name='control'
)
trace2 = go.Histogram(
    x=loans.lender_count[:1000],
    opacity=0.75,
    histnorm='count',
    name='control'
)

data = [trace1, trace2]
layout = go.Layout(barmode='overlay')
figure = go.Figure(data=data, layout=layout)

ply.iplot(figure, filename='histogram')
from collections import Counter
def gender_rank(text):
    if text == 'male':
        return 'male'
    elif text == 'female':
        return 'female'
    else:
        text = Counter(str(text).split(',')).most_common()[0][0]
        if text.replace(' ', '') == 'nan':
            return np.NaN
        return text.replace(' ', '')
d = [gender_rank(x) for x in placeholder]
# borrower_genders 	repayment_interval
Y=loans.borrower_genders.value_counts().index[::-1]
X=loans.borrower_genders.value_counts().values[::-1]
data = go.Bar(
    x = X,
    y = Y,
    orientation = 'h',
    marker=dict(
        color=X,
        colorscale = 'Jet',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Loans by Gender',
    width=700,
    height=300,
    )
figure = go.Figure(data=[data], layout=layout)
ply.iplot(figure, filename="GenderType")
# borrower_genders 	repayment_interval
Y=loans[loans.borrower_genders == 'male'].repayment_interval.value_counts().index[::-1]
X=loans[loans.borrower_genders == 'male'].repayment_interval.value_counts().values[::-1]
data = go.Bar(
    x = X,
    y = Y,
    orientation = 'h',
    marker=dict(
        color=X,
        colorscale = 'Jet',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Loans distribution by Males',
    width=700,
    height=300,
    )
fig = go.Figure(data=[data], layout=layout)
ply.iplot(fig, filename="GenderMale")
# borrower_genders 	repayment_interval
Y=loans[loans.borrower_genders == 'female'].repayment_interval.value_counts().index[::-1]
X=loans[loans.borrower_genders == 'female'].repayment_interval.value_counts().values[::-1]
data = go.Bar(
    x = X,
    y = Y,
    orientation = 'h',
    marker=dict(
        color=X,
        colorscale = 'Jet',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Loans by Females',
    width=700,
    height=300,
    )
fig = go.Figure(data=[data], layout=layout)
ply.iplot(fig, filename="GenderFemale")
X=loans[loans.borrower_genders == 'male'].sector.value_counts()
trace = go.Pie(labels=X.index, values=X.values)
ply.iplot([trace], filename='basic_pie_chart')
X=loans[loans.borrower_genders == 'female'].sector.value_counts()
trace = go.Pie(labels=X.index, values=X.values)
ply.iplot([trace], filename='basic_pie_chart')
loans[['country', 'region', 'currency', 'borrower_genders', 'repayment_interval', 'activity']].groupby('country').head()
region = pd.read_csv('../input/loan_themes_by_region.csv')
region.head()
region_df = pd.DataFrame(region['country'].value_counts()).reset_index()
region_df.columns=['country', 'loans']
region_df = region_df.reset_index().drop('index', axis=1)
data = [ dict(
        type = 'choropleth',
        locations = region_df['country'],
        locationmode = 'country names',
        z = region_df['loans'],
        text = region_df['country'],
        colorscale = [[0,"rgb(5, 50, 172)"],[0.85,"rgb(40, 100, 190)"],[0.9,"rgb(70, 140, 245)"],\
            [0.94,"rgb(90, 160, 245)"],[0.97,"rgb(106, 177, 247)"],[1,"rgb(220, 250, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Number of Loans'),
      ) ]

layout = dict(
    title = 'Loans by Country',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

figure = dict( data=data, layout=layout )
ply.iplot( figure, validate=False, filename='regionship')
trace0 = go.Scatter(
    x=region['country'].value_counts().index,
    y=region['country'].value_counts().values,
    text=region['country'].value_counts().index,
    mode='markers',
    marker=dict(
        color = np.random.randn(500), #set color equal to a variable
        colorscale='Jet',
        showscale=True,
        size=[i/5  if i < 550 else i/50 for i in region['country'].value_counts().values],
    )
)

data = go.Data([trace0])
ply.iplot(data, filename='mpl-7d-bubble')
trace0 = go.Scatter(
    x=region['country'].value_counts().index,
    y=region['country'].value_counts().values,
    text=region['country'].value_counts().index,
    mode='markers',
    marker=dict(
        color = np.random.randn(500), #set color equal to a variable
        colorscale='Jet',
        showscale=True,
        size=[i/2  if i < 550 else i/50 for i in region['country'].value_counts().values],
    )
)

data = go.Data([trace0])
ply.iplot(data, filename='mplbubble')
themeids = pd.read_csv('../input/loan_theme_ids.csv')
themeids.head()
X = themeids['Loan Theme Type'].value_counts().index[::-1]
Y = themeids['Loan Theme Type'].value_counts().values[::-1]
data = go.Bar(
    x = Y,
    y = X,
    orientation = 'h',
    marker=dict(
        color=Y,
        colorscale = 'Jet',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Loans by Loan Theme Type',
    width=900,
    height=1000,
    )
fig = go.Figure(data=[data], layout=layout)
ply.iplot(fig, filename="Loans")
