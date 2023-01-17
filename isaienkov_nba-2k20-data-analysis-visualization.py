import numpy as np

import pandas as pd

import plotly.express as px

from datetime import date

import plotly.graph_objs as go

from iso3166 import countries
WIDTH=800
df = pd.read_csv('/kaggle/input/nba2k20-player-dataset/nba2k20-full.csv')

df.head()
print('Top rating presented in dataset: ', df['rating'].max())

print('Low rating presented in dataset: ', df['rating'].min())
def plot_value_count(dataframe, column, width, height, title):

    ds = dataframe[column].value_counts().reset_index()

    ds.columns = [column, 'count']

    fig = px.bar(

        ds, 

        x=column, 

        y="count", 

        orientation='v', 

        title=title, 

        width=width,

        height=height

    )

    fig.show()
plot_value_count(df, 'rating', WIDTH, 600, 'Players and their rating')
plot_value_count(df, 'team', WIDTH, 600, 'Teams by number of players')
df[df['team'].isna()]
ds = df[df['team'].notnull()]

ds = ds['jersey'].value_counts().reset_index()

ds.columns = ['jersey', 'count']

ds['jersey'] = 'number ' + ds['jersey']

ds = ds.sort_values('count')



fig = px.bar(

    ds, 

    x='count', 

    y="jersey", 

    orientation='h', 

    title="Player's jersey distribution", 

    width=WIDTH,

    height=1000

)



fig.show()
df[(df['jersey'] == '#0') & (df['team'].notnull())]
ds = df[(df['jersey'] == '#0') & (df['team'].notnull())]

ds = ds['team'].value_counts().reset_index()

ds.columns = ['team', 'count']

ds = ds.sort_values('count')



fig = px.bar(

    ds, 

    x='count', 

    y="team", 

    orientation='h', 

    title="Number of jerseys #0 for every team", 

    width=WIDTH,

    height=600

)



fig.show()
df.loc[df['position'] == 'C-F', 'position'] = 'F-C'

df.loc[df['position'] == 'F-G', 'position'] = 'G-F'



plot_value_count(df, 'position', WIDTH, 500, "Players position distribution")
plot_value_count(df, 'country', WIDTH, 600, "Players country distribution")
ds = df['college'].value_counts().reset_index()

ds.columns = ['college', 'count']

ds = ds.sort_values('count').tail(30)



fig = px.bar(

    ds, 

    x='count',

    y="college", 

    orientation='h', 

    title="Top 30 colleges by number of players", 

    width=WIDTH, 

    height=800

)



fig.show()
plot_value_count(df, 'draft_year', WIDTH, 600, "Players draft year distribution")
ds = df['draft_round'].value_counts().reset_index()

ds.columns = ['draft_round', 'count']

ds.loc[ds['draft_round']=='1', 'draft_round'] = '1-st'

ds.loc[ds['draft_round']=='2', 'draft_round'] = '2-nd'



fig = px.pie(

    ds, 

    names='draft_round', 

    values="count", 

    title="Players draft round pie chart", 

    width=WIDTH, 

    height=500

)



fig.show()
ds = df[df['draft_peak']!='Undrafted']

ds = ds['draft_peak'].value_counts().reset_index()

ds.columns = ['draft_peak', 'count']



fig = px.bar(

    ds, 

    x='draft_peak', 

    y="count", 

    orientation='v', 

    title="Players draft peak distribution", 

    width=WIDTH

)



fig.show()
df['salary'] = df['salary'].str.replace('$', '')

df['salary'] = df['salary'].astype(np.float64)



fig = px.histogram(

    df, 

    "salary", 

    nbins=100, 

    title='Salary distribution', 

    width=WIDTH,

    height=600

)



fig.show()
weight = df['weight'].str.split('/',expand=True)

weight.columns = ['weight_lbs', 'weight_kg']

df = pd.concat([df, weight], axis=1)

df = df.drop(['weight'], axis=1)

df['weight_lbs'] = df['weight_lbs'].str.replace('lbs.', '')

df['weight_kg'] = df['weight_kg'].str.replace('kg.', '')

df['weight_lbs'] = df['weight_lbs'].astype(np.int32)

df['weight_kg'] = df['weight_kg'].astype(np.float64)

df
height = df['height'].str.split('/',expand=True)

height.columns = ['height_feet', 'height_m']

df = pd.concat([df, height], axis=1)

df = df.drop(['height'], axis=1)

df['height_m'] = df['height_m'].astype(np.float64)

df
fig = px.histogram(

    df, 

    "weight_kg", 

    nbins=50, 

    title='Weight distribution', 

    width=WIDTH

)



fig.show()
fig = px.histogram(

    df, 

    "height_m", 

    nbins=20, 

    title='Height distribution', 

    width=WIDTH

)



fig.show()
def calculate_age(born): 

    today = date.today() 

    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
df['b_day'] = pd.to_datetime(df['b_day'])

df['age'] = df['b_day'].apply(lambda row : calculate_age(row))
df['b_year'] = df['b_day'].dt.year

df['b_month'] = df['b_day'].dt.month
ds = df['b_month'].value_counts().reset_index()

ds.columns = ['month', 'count']



fig = px.bar(

    ds, 

    x='month', 

    y="count", 

    orientation='v', 

    title="Players month of birth", 

    width=WIDTH,

    height=600

)



fig.show()
fig = px.histogram(

    df, 

    "age", 

    nbins=25, 

    title='Age distribution', 

    width=800

)



fig.show()
df[df['draft_peak']=='1']
team = df.groupby('team')['rating'].mean().reset_index().sort_values('rating', ascending=True)



fig = px.bar(

    team, 

    x="rating", 

    y="team", 

    orientation='h',

    title='Average rating of players for evety team',

    width=800, 

    height=800

)



fig.show()
position = df.groupby('position')['rating'].mean().reset_index().sort_values('rating', ascending=True)



fig = px.bar(

    position, 

    x="rating",

    y="position", 

    orientation='h',

    title='Average rating of players by position',

    width=800, 

    height=400

)



fig.show()
ds = df['country'].value_counts().reset_index()

ds.columns = ['country', 'count']

ds = ds[ds['count']>=5]

countries_list = ds['country'].unique()



position = df[df['country'].isin(countries_list)]

position = position.groupby('country')['rating'].mean().reset_index().sort_values('rating', ascending=True)



fig = px.bar(

    position, 

    x="rating", 

    y="country", 

    orientation='h',

    title='Average rating of players by country (5+ players)',

    width=800, 

    height=500

)



fig.show()
position = df.groupby('draft_year')['rating'].mean().reset_index().sort_values('rating', ascending=True)

position['draft_year'] = position['draft_year'].astype(str) + ' year'

fig = px.bar(

    position, 

    x="rating", 

    y="draft_year", 

    orientation='h',

    title='Average rating of players by draft year',

    width=800, 

    height=600

)

fig.show()
position = df.groupby('draft_peak')['rating'].mean().reset_index().sort_values('rating', ascending=True)

position['draft_peak'] = position['draft_peak'].astype(str) + ' peak'



fig = px.bar(

    position, 

    x="rating", 

    y="draft_peak", 

    orientation='h',

    title='Average rating of players by draft peak',

    width=800,

    height=1200

)



fig.show()
position = df.sort_values(['age', 'rating'], ascending=True).tail(20)



fig = px.bar(

    position, 

    x="rating", 

    y="full_name", 

    color='age', 

    orientation='h',

    title='Top 20 old players',

    width=800, 

    height=600

)



fig.show()
position = df.sort_values(['age', 'rating'], ascending=False).tail(20)



fig = px.bar(

    position, 

    x="rating", 

    y="full_name",     

    color='age', 

    orientation='h', 

    title='Top 20 young players',

    width=800, 

    height=600

)



fig.show()
position = df.sort_values(['height_m', 'rating'], ascending=True).tail(20)



fig = px.bar(

    position, 

    x="rating", 

    y="full_name", 

    color='height_m', 

    orientation='h', 

    width=800, 

    height=600, 

    title='Top 20 high players'

)



fig.show()
position = df.sort_values(['height_m', 'rating'], ascending=False).tail(20)



fig = px.bar(

    position, 

    x="rating", 

    y="full_name", 

    color='height_m', 

    orientation='h', 

    width=800, 

    height=600, 

    title='Top 20 short players'

)



fig.show()
country_dict = {}

for c in countries:

    country_dict[c.name] = c.alpha3

    

df['alpha3'] = df['country']

df = df.replace({"alpha3": country_dict})



data = df.groupby(['alpha3', 'country'])['rating'].max().reset_index()

data.columns = ['alpha3', 'nationality', 'max_rating']



fig = px.choropleth(

    data, 

    locations="alpha3",

    hover_name='nationality',

    color="max_rating",

    projection="natural earth",

    color_continuous_scale=px.colors.sequential.Plasma,

    title='Max rating for every country',

    width=800, 

    height=600

)



fig.show()
data = df.groupby(['alpha3', 'country'])['salary'].max().reset_index()

data.columns = ['alpha3', 'nationality', 'max_salary']



fig = px.choropleth(

    data, 

    locations="alpha3",

    hover_name='nationality',

    color="max_salary",

    projection="natural earth",

    color_continuous_scale=px.colors.sequential.Plasma,

    title='Max salary for players from every country',

    width=800, 

    height=600

)



fig.show()
data = df.groupby(['alpha3', 'country'])['height_m'].max().reset_index()

data.columns = ['alpha3', 'nationality', 'height_m']



fig = px.choropleth(

    data, 

    locations="alpha3",

    hover_name='nationality',

    color='height_m',

    projection="natural earth",

    color_continuous_scale=px.colors.sequential.Plasma,

    title='Max height for players from every country',

    width=800, 

    height=600

)



fig.show()
data = df.groupby(['alpha3', 'country'])['weight_kg'].max().reset_index()

data.columns = ['alpha3', 'nationality', 'weight_kg']



fig = px.choropleth(

    data, 

    locations="alpha3",

    hover_name='nationality',

    color='weight_kg',

    projection="natural earth",

    color_continuous_scale=px.colors.sequential.Plasma,

    title='Max weight for players from every country',

    width=800, 

    height=600

)



fig.show()
data = df['alpha3'].value_counts().reset_index()

data.columns=['alpha3', 'national_count']

df = pd.merge(df, data, on='alpha3')

data = df[df['national_count']>=5]

df = df.drop(['national_count'], axis=1)

data = data.groupby(['alpha3', 'country'])['rating'].mean().reset_index()

data.columns = ['alpha3', 'nationality', 'mean_rating']



fig = px.choropleth(

    data, 

    locations="alpha3",

    hover_name='nationality',

    color="mean_rating",

    projection="natural earth",

    color_continuous_scale=px.colors.sequential.Plasma,

    title='Mean rating for sportsmen for every country (minimum 5 players)',

    width=900, 

    height=700

)



fig.show()
def draw_plotly_court(fig, fig_width=600, margins=10):

    def ellipse_arc(x_center=0.0, y_center=0.0, a=10.5, b=10.5, start_angle=0.0, end_angle=2 * np.pi, N=200, closed=False):

        t = np.linspace(start_angle, end_angle, N)

        x = x_center + a * np.cos(t)

        y = y_center + b * np.sin(t)

        path = f'M {x[0]}, {y[0]}'

        for k in range(1, len(t)):

            path += f'L{x[k]}, {y[k]}'

        if closed:

            path += ' Z'

        return path



    fig_height = fig_width * (470 + 2 * margins) / (500 + 2 * margins)

    fig.update_layout(width=fig_width, height=fig_height)



    # Set axes ranges

    fig.update_xaxes(range=[-250 - margins, 250 + margins])

    fig.update_yaxes(range=[-52.5 - margins, 417.5 + margins])



    threept_break_y = 89.47765084

    three_line_col = "#000000"

    main_line_col = "#000000"



    fig.update_layout(

        # Line Horizontal

        margin=dict(l=20, r=20, t=20, b=20),

        paper_bgcolor="white",

        plot_bgcolor="white",

        yaxis=dict(

            scaleanchor="x",

            scaleratio=1,

            showgrid=False,

            zeroline=False,

            showline=False,

            ticks='',

            showticklabels=False,

            fixedrange=True,

        ),

        xaxis=dict(

            showgrid=False,

            zeroline=False,

            showline=False,

            ticks='',

            showticklabels=False,

            fixedrange=True,

        ),

        shapes=[

            dict(

                type="rect", x0=-250, y0=-52.5, x1=250, y1=417.5,

                line=dict(color=main_line_col, width=1),

                fillcolor='#55AF55',

                layer='below'

            ),

            dict(

                type="rect", x0=-80, y0=-52.5, x1=80, y1=137.5,

                line=dict(color=main_line_col, width=1),

                fillcolor='#333333',

                layer='below'

            ),

            dict(

                type="rect", x0=-60, y0=-52.5, x1=60, y1=137.5,

                line=dict(color=main_line_col, width=1),

                fillcolor='#333333',

                layer='below'

            ),

            dict(

                type="circle", x0=-60, y0=77.5, x1=60, y1=197.5, xref="x", yref="y",

                line=dict(color=main_line_col, width=1),

                fillcolor='#dddddd',

                layer='below'

            ),

            dict(

                type="line", x0=-60, y0=137.5, x1=60, y1=137.5,

                line=dict(color=main_line_col, width=1),

                layer='below'

            ),



            dict(

                type="rect", x0=-2, y0=-7.25, x1=2, y1=-12.5,

                line=dict(color="#ec7607", width=1),

                fillcolor='#ec7607',

            ),

            dict(

                type="circle", x0=-7.5, y0=-7.5, x1=7.5, y1=7.5, xref="x", yref="y",

                line=dict(color="#ec7607", width=1),

            ),

            dict(

                type="line", x0=-30, y0=-12.5, x1=30, y1=-12.5,

                line=dict(color="#ec7607", width=1),

            ),



            dict(type="path",

                 path=ellipse_arc(a=40, b=40, start_angle=0, end_angle=np.pi),

                 line=dict(color=main_line_col, width=1), layer='below'),

            dict(type="path",

                 path=ellipse_arc(a=237.5, b=237.5, start_angle=0.386283101, end_angle=np.pi - 0.386283101),

                 line=dict(color=main_line_col, width=1), layer='below'),

            dict(

                type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,

                line=dict(color=three_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,

                line=dict(color=three_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=220, y0=-52.5, x1=220, y1=threept_break_y,

                line=dict(color=three_line_col, width=1), layer='below'

            ),



            dict(

                type="line", x0=-250, y0=227.5, x1=-220, y1=227.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=250, y0=227.5, x1=220, y1=227.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=-90, y0=17.5, x1=-80, y1=17.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=-90, y0=27.5, x1=-80, y1=27.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=-90, y0=57.5, x1=-80, y1=57.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=-90, y0=87.5, x1=-80, y1=87.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=90, y0=17.5, x1=80, y1=17.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=90, y0=27.5, x1=80, y1=27.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=90, y0=57.5, x1=80, y1=57.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=90, y0=87.5, x1=80, y1=87.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),



            dict(type="path",

                 path=ellipse_arc(y_center=417.5, a=60, b=60, start_angle=-0, end_angle=-np.pi),

                 line=dict(color=main_line_col, width=1), layer='below'),



        ]

    )

    return True
fig = go.Figure()

draw_plotly_court(fig)

fig.add_trace(

    go.Scatter(

        x=[-110, 40, 0, -160, 160],

        y=[70, 45, 260, 200, 150],

        mode='markers+text',

        marker=dict(

            size=[30, 30, 30, 30, 30]

        ),

        text=['PF', 'C', 'PG', 'SG', 'SF']

    )

)



fig.show()
sorted_df = df.sort_values(['rating'], ascending=False)

positions = ['F', 'G-F', 'G', 'F-C', 'C']

best_by_rating = list()



for pos in positions:

    part = sorted_df[sorted_df['position']==pos]

    best_by_rating.append(part.head(1))

    

best_by_rating = pd.concat(best_by_rating)





fig = go.Figure()

draw_plotly_court(fig)

fig.add_trace(

    go.Scatter(

        x=[-110, 40, 0, -160, 160],

        y=[70, 45, 260, 200, 150],

        mode='markers+text',

        marker=dict(

            size=[30, 30, 30, 30, 30]

        ),

        text=['PF', 'C', 'PG', 'SG', 'SF']

    )

)





fig.add_annotation(

            x=-110,

            y=95,

            text=best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['rating']))

fig.add_annotation(

            x=40,

            y=70,

            text=best_by_rating[best_by_rating['position']=='C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='C'].iloc[0]['rating']))

fig.add_annotation(

            x=0,

            y=285,

            text=best_by_rating[best_by_rating['position']=='G'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='G'].iloc[0]['rating']))

fig.add_annotation(

            x=-160,

            y=225,

            text=best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['rating']))

fig.add_annotation(

            x=160,

            y=175,

            text=best_by_rating[best_by_rating['position']=='F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='F'].iloc[0]['rating']))

fig.update_annotations(dict(

            xref="x",

            yref="y",

            showarrow=False,

            font=dict(

                family="sans serif",

                size=16,

                color="#FF0000"

            )

))



fig.update_layout(showlegend=False)



fig.show()
sorted_df = df.sort_values(['salary'], ascending=False)

positions = ['F', 'G-F', 'G', 'F-C', 'C']

best_by_rating = list()



for pos in positions:

    part = sorted_df[sorted_df['position']==pos]

    best_by_rating.append(part.head(1))

    

best_by_rating = pd.concat(best_by_rating)





fig = go.Figure()

draw_plotly_court(fig)

fig.add_trace(

    go.Scatter(

        x=[-110, 40, 0, -160, 160],

        y=[70, 45, 260, 200, 150],

        mode='markers+text',

        marker=dict(

            size=[30, 30, 30, 30, 30]

        ),

        text=['PF', 'C', 'PG', 'SG', 'SF']

    )

)





fig.add_annotation(

            x=-110,

            y=95,

            text=best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['rating']))

fig.add_annotation(

            x=40,

            y=70,

            text=best_by_rating[best_by_rating['position']=='C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='C'].iloc[0]['rating']))

fig.add_annotation(

            x=0,

            y=285,

            text=best_by_rating[best_by_rating['position']=='G'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='G'].iloc[0]['rating']))

fig.add_annotation(

            x=-160,

            y=225,

            text=best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['rating']))

fig.add_annotation(

            x=160,

            y=175,

            text=best_by_rating[best_by_rating['position']=='F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='F'].iloc[0]['rating']))

fig.update_annotations(dict(

            xref="x",

            yref="y",

            showarrow=False,

            font=dict(

                family="sans serif",

                size=16,

                color="#FF0000"

            )

))



fig.update_layout(showlegend=False)



fig.show()
sorted_df = df.sort_values(['age'], ascending=False)

positions = ['F', 'G-F', 'G', 'F-C', 'C']

best_by_rating = list()



for pos in positions:

    part = sorted_df[sorted_df['position']==pos]

    best_by_rating.append(part.head(1))

    

best_by_rating = pd.concat(best_by_rating)





fig = go.Figure()

draw_plotly_court(fig)

fig.add_trace(

    go.Scatter(

        x=[-110, 40, 0, -160, 160],

        y=[70, 45, 260, 200, 150],

        mode='markers+text',

        marker=dict(

            size=[30, 30, 30, 30, 30]

        ),

        text=['PF', 'C', 'PG', 'SG', 'SF']

    )

)





fig.add_annotation(

            x=-110,

            y=95,

            text=best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['age']))

fig.add_annotation(

            x=40,

            y=70,

            text=best_by_rating[best_by_rating['position']=='C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='C'].iloc[0]['age']))

fig.add_annotation(

            x=0,

            y=285,

            text=best_by_rating[best_by_rating['position']=='G'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='G'].iloc[0]['age']))

fig.add_annotation(

            x=-160,

            y=225,

            text=best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['age']))

fig.add_annotation(

            x=160,

            y=175,

            text=best_by_rating[best_by_rating['position']=='F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='F'].iloc[0]['age']))

fig.update_annotations(dict(

            xref="x",

            yref="y",

            showarrow=False,

            font=dict(

                family="sans serif",

                size=16,

                color="#FF0000"

            )

))



fig.update_layout(showlegend=False)



fig.show()
sorted_df = df.sort_values(['height_m'], ascending=False)

positions = ['F', 'G-F', 'G', 'F-C', 'C']

best_by_rating = list()

for pos in positions:

    part = sorted_df[sorted_df['position']==pos]

    best_by_rating.append(part.head(1))

    

best_by_rating = pd.concat(best_by_rating)





fig = go.Figure()

draw_plotly_court(fig)

fig.add_trace(

    go.Scatter(

        x=[-110, 40, 0, -160, 160],

        y=[70, 45, 260, 200, 150],

        mode='markers+text',

        marker=dict(size=[30, 30, 30, 30, 30]),

        text=['PF', 'C', 'PG', 'SG', 'SF']

    )

)





fig.add_annotation(

            x=-110,

            y=95,

            text=best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['height_m']))

fig.add_annotation(

            x=40,

            y=70,

            text=best_by_rating[best_by_rating['position']=='C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='C'].iloc[0]['height_m']))

fig.add_annotation(

            x=0,

            y=285,

            text=best_by_rating[best_by_rating['position']=='G'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='G'].iloc[0]['height_m']))

fig.add_annotation(

            x=-160,

            y=225,

            text=best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['height_m']))

fig.add_annotation(

            x=160,

            y=175,

            text=best_by_rating[best_by_rating['position']=='F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='F'].iloc[0]['height_m']))

fig.update_annotations(dict(

            xref="x",

            yref="y",

            showarrow=False,

            font=dict(

                family="sans serif",

                size=16,

                color="#FF0000"

            )

))



fig.update_layout(showlegend=False)



fig.show()
sorted_df = df.sort_values(['height_m'], ascending=True)

positions = ['F', 'G-F', 'G', 'F-C', 'C']

best_by_rating = list()

for pos in positions:

    part = sorted_df[sorted_df['position']==pos]

    best_by_rating.append(part.head(1))

    

best_by_rating = pd.concat(best_by_rating)





fig = go.Figure()

draw_plotly_court(fig)

fig.add_trace(

    go.Scatter(

        x=[-110, 40, 0, -160, 160],

        y=[70, 45, 260, 200, 150],

        mode='markers+text',

        marker=dict(size=[30, 30, 30, 30, 30]),

        text=['PF', 'C', 'PG', 'SG', 'SF']

    )

)





fig.add_annotation(

            x=-110,

            y=95,

            text=best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['height_m']))

fig.add_annotation(

            x=40,

            y=70,

            text=best_by_rating[best_by_rating['position']=='C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='C'].iloc[0]['height_m']))

fig.add_annotation(

            x=0,

            y=285,

            text=best_by_rating[best_by_rating['position']=='G'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='G'].iloc[0]['height_m']))

fig.add_annotation(

            x=-160,

            y=225,

            text=best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['height_m']))

fig.add_annotation(

            x=160,

            y=175,

            text=best_by_rating[best_by_rating['position']=='F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='F'].iloc[0]['height_m']))

fig.update_annotations(dict(

            xref="x",

            yref="y",

            showarrow=False,

            font=dict(

                family="sans serif",

                size=16,

                color="#FF0000"

            )

))



fig.update_layout(showlegend=False)



fig.show()
sorted_df = df[df['team'].isnull()]

sorted_df = sorted_df.sort_values(['rating'], ascending=False)

positions = ['F', 'G-F', 'G', 'F-C', 'C']

best_by_rating = list()

for pos in positions:

    part = sorted_df[sorted_df['position']==pos]

    best_by_rating.append(part.head(1))

    

best_by_rating = pd.concat(best_by_rating)





fig = go.Figure()

draw_plotly_court(fig)

fig.add_trace(

    go.Scatter(

        x=[-110, 40, 0, -160, 160],

        y=[70, 45, 260, 200, 150],

        mode='markers+text',

        marker=dict(size=[30, 30, 30, 30, 30]),

        text=['PF', 'C', 'PG', 'SG', 'SF']

    )

)





fig.add_annotation(

            x=-110,

            y=95,

            text=best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['rating']))

fig.add_annotation(

            x=40,

            y=70,

            text=best_by_rating[best_by_rating['position']=='C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='C'].iloc[0]['rating']))

fig.add_annotation(

            x=0,

            y=285,

            text=best_by_rating[best_by_rating['position']=='G'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='G'].iloc[0]['rating']))

fig.add_annotation(

            x=-160,

            y=225,

            text=best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['rating']))

fig.add_annotation(

            x=160,

            y=175,

            text=best_by_rating[best_by_rating['position']=='F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='F'].iloc[0]['rating']))

fig.update_annotations(dict(

            xref="x",

            yref="y",

            showarrow=False,

            font=dict(

                family="sans serif",

                size=16,

                color="#FF0000"

            )

))



fig.update_layout(showlegend=False)



fig.show()
sorted_df = df[df['country']=='USA']

sorted_df = sorted_df.sort_values(['rating'], ascending=False)

positions = ['F', 'G-F', 'G', 'F-C', 'C']

best_by_rating = list()

for pos in positions:

    part = sorted_df[sorted_df['position']==pos]

    best_by_rating.append(part.head(1))

    

best_by_rating = pd.concat(best_by_rating)





fig = go.Figure()

draw_plotly_court(fig)

fig.add_trace(

    go.Scatter(

        x=[-110, 40, 0, -160, 160],

        y=[70, 45, 260, 200, 150],

        mode='markers+text',

        marker=dict(size=[30, 30, 30, 30, 30]),

        text=['PF', 'C', 'PG', 'SG', 'SF']

    )

)





fig.add_annotation(

            x=-110,

            y=95,

            text=best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['rating']))

fig.add_annotation(

            x=40,

            y=70,

            text=best_by_rating[best_by_rating['position']=='C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='C'].iloc[0]['rating']))

fig.add_annotation(

            x=0,

            y=285,

            text=best_by_rating[best_by_rating['position']=='G'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='G'].iloc[0]['rating']))

fig.add_annotation(

            x=-160,

            y=225,

            text=best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['rating']))

fig.add_annotation(

            x=160,

            y=175,

            text=best_by_rating[best_by_rating['position']=='F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='F'].iloc[0]['rating']))

fig.update_annotations(dict(

            xref="x",

            yref="y",

            showarrow=False,

            font=dict(

                family="sans serif",

                size=16,

                color="#FF0000"

            )

))



fig.update_layout(showlegend=False)



fig.show()
sorted_df = df[df['country']=='Canada']

sorted_df = sorted_df.sort_values(['rating'], ascending=False)

positions = ['F', 'G-F', 'G', 'F-C', 'C']

best_by_rating = list()

for pos in positions:

    part = sorted_df[sorted_df['position']==pos]

    best_by_rating.append(part.head(1))

    

best_by_rating = pd.concat(best_by_rating)





fig = go.Figure()

draw_plotly_court(fig)

fig.add_trace(

    go.Scatter(

        x=[-110, 40, 0, -160, 160],

        y=[70, 45, 260, 200, 150],

        mode='markers+text',

        marker=dict(size=[30, 30, 30, 30, 30]),

        text=['PF', 'C', 'PG', 'SG', 'SF']

    )

)





fig.add_annotation(

            x=-110,

            y=95,

            text=best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['rating']))

fig.add_annotation(

            x=40,

            y=70,

            text=best_by_rating[best_by_rating['position']=='C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='C'].iloc[0]['rating']))

fig.add_annotation(

            x=0,

            y=285,

            text=best_by_rating[best_by_rating['position']=='G'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='G'].iloc[0]['rating']))

fig.add_annotation(

            x=-160,

            y=225,

            text=best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['rating']))

fig.add_annotation(

            x=160,

            y=175,

            text=best_by_rating[best_by_rating['position']=='F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='F'].iloc[0]['rating']))

fig.update_annotations(dict(

            xref="x",

            yref="y",

            showarrow=False,

            font=dict(

                family="sans serif",

                size=16,

                color="#FF0000"

            )

))



fig.update_layout(showlegend=False)



fig.show()
sorted_df = df[df['country']!='USA']

sorted_df = sorted_df.sort_values(['rating'], ascending=False)

positions = ['F', 'G-F', 'G', 'F-C', 'C']

best_by_rating = list()

for pos in positions:

    part = sorted_df[sorted_df['position']==pos]

    best_by_rating.append(part.head(1))

    

best_by_rating = pd.concat(best_by_rating)





fig = go.Figure()

draw_plotly_court(fig)

fig.add_trace(

    go.Scatter(

        x=[-110, 40, 0, -160, 160],

        y=[70, 45, 260, 200, 150],

        mode='markers+text',

        marker=dict(size=[30, 30, 30, 30, 30]),

        text=['PF', 'C', 'PG', 'SG', 'SF']

    )

)





fig.add_annotation(

            x=-110,

            y=95,

            text=best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['rating']))

fig.add_annotation(

            x=40,

            y=70,

            text=best_by_rating[best_by_rating['position']=='C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='C'].iloc[0]['rating']))

fig.add_annotation(

            x=0,

            y=285,

            text=best_by_rating[best_by_rating['position']=='G'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='G'].iloc[0]['rating']))

fig.add_annotation(

            x=-160,

            y=225,

            text=best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['rating']))

fig.add_annotation(

            x=160,

            y=175,

            text=best_by_rating[best_by_rating['position']=='F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='F'].iloc[0]['rating']))

fig.update_annotations(dict(

            xref="x",

            yref="y",

            showarrow=False,

            font=dict(

                family="sans serif",

                size=16,

                color="#FF0000"

            )

))



fig.update_layout(showlegend=False)



fig.show()
sorted_df = df[df['draft_peak']=='2']

sorted_df = sorted_df.sort_values(['rating'], ascending=False)

positions = ['F', 'G-F', 'G', 'F-C', 'C']

best_by_rating = list()

for pos in positions:

    part = sorted_df[sorted_df['position']==pos]

    best_by_rating.append(part.head(1))

    

best_by_rating = pd.concat(best_by_rating)





fig = go.Figure()

draw_plotly_court(fig)

fig.add_trace(

    go.Scatter(

        x=[-110, 40, 0, -160, 160],

        y=[70, 45, 260, 200, 150],

        mode='markers+text',

        marker=dict(size=[30, 30, 30, 30, 30]),

        text=['PF', 'C', 'PG', 'SG', 'SF']

    )

)





fig.add_annotation(

            x=-110,

            y=95,

            text=best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['rating']))

fig.add_annotation(

            x=40,

            y=70,

            text=best_by_rating[best_by_rating['position']=='C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='C'].iloc[0]['rating']))

fig.add_annotation(

            x=0,

            y=285,

            text=best_by_rating[best_by_rating['position']=='G'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='G'].iloc[0]['rating']))

fig.add_annotation(

            x=-160,

            y=225,

            text=best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['rating']))

fig.add_annotation(

            x=160,

            y=175,

            text=best_by_rating[best_by_rating['position']=='F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='F'].iloc[0]['rating']))

fig.update_annotations(dict(

            xref="x",

            yref="y",

            showarrow=False,

            font=dict(

                family="sans serif",

                size=16,

                color="#FF0000"

            )

))



fig.update_layout(showlegend=False)



fig.show()
sorted_df = df[df['draft_peak']=='3']

sorted_df = sorted_df.sort_values(['rating'], ascending=False)

positions = ['F', 'G-F', 'G', 'F-C', 'C']

best_by_rating = list()

for pos in positions:

    part = sorted_df[sorted_df['position']==pos]

    best_by_rating.append(part.head(1))

    

best_by_rating = pd.concat(best_by_rating)





fig = go.Figure()

draw_plotly_court(fig)

fig.add_trace(

    go.Scatter(

        x=[-110, 40, 0, -160, 160],

        y=[70, 45, 260, 200, 150],

        mode='markers+text',

        marker=dict(size=[30, 30, 30, 30, 30]),

        text=['PF', 'C', 'PG', 'SG', 'SF']

    )

)





fig.add_annotation(

            x=-110,

            y=95,

            text=best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['rating']))

fig.add_annotation(

            x=40,

            y=70,

            text=best_by_rating[best_by_rating['position']=='C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='C'].iloc[0]['rating']))

fig.add_annotation(

            x=0,

            y=285,

            text=best_by_rating[best_by_rating['position']=='G'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='G'].iloc[0]['rating']))

fig.add_annotation(

            x=-160,

            y=225,

            text=best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['rating']))

fig.add_annotation(

            x=160,

            y=175,

            text=best_by_rating[best_by_rating['position']=='F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='F'].iloc[0]['rating']))

fig.update_annotations(dict(

            xref="x",

            yref="y",

            showarrow=False,

            font=dict(

                family="sans serif",

                size=16,

                color="#FF0000"

            )

))



fig.update_layout(showlegend=False)



fig.show()
sorted_df = df[df['draft_peak']=='Undrafted']

sorted_df = sorted_df.sort_values(['rating'], ascending=False)

positions = ['F', 'G-F', 'G', 'F-C', 'C']

best_by_rating = list()

for pos in positions:

    part = sorted_df[sorted_df['position']==pos]

    best_by_rating.append(part.head(1))

    

best_by_rating = pd.concat(best_by_rating)





fig = go.Figure()

draw_plotly_court(fig)

fig.add_trace(

    go.Scatter(

        x=[-110, 40, 0, -160, 160],

        y=[70, 45, 260, 200, 150],

        mode='markers+text',

        marker=dict(size=[30, 30, 30, 30, 30]),

        text=['PF', 'C', 'PG', 'SG', 'SF']

    )

)





fig.add_annotation(

            x=-110,

            y=95,

            text=best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='F-C'].iloc[0]['rating']))

fig.add_annotation(

            x=40,

            y=70,

            text=best_by_rating[best_by_rating['position']=='C'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='C'].iloc[0]['rating']))

fig.add_annotation(

            x=0,

            y=285,

            text=best_by_rating[best_by_rating['position']=='G'].iloc[0]['full_name'] + ', ' + 

                str(best_by_rating[best_by_rating['position']=='G'].iloc[0]['rating']))

fig.add_annotation(

            x=-160,

            y=225,

            text=best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='G-F'].iloc[0]['rating']))

fig.add_annotation(

            x=160,

            y=175,

            text=best_by_rating[best_by_rating['position']=='F'].iloc[0]['full_name']+ ', ' + 

                str(best_by_rating[best_by_rating['position']=='F'].iloc[0]['rating']))

fig.update_annotations(dict(

            xref="x",

            yref="y",

            showarrow=False,

            font=dict(

                family="sans serif",

                size=16,

                color="#FF0000"

            )

))



fig.update_layout(showlegend=False)



fig.show()