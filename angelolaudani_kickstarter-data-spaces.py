import numpy as np

import pandas as pd



import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go

from plotly.offline import iplot



import plotly.io as pio

pio.templates.default = "ggplot2"



from collections import defaultdict

from xgboost import XGBRegressor

from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

from sklearn.tree import DecisionTreeClassifier



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
color1 = '#2bd9de'

color2 = '#2bde73'

color3 = '#081245'

color4 = '#122906'

color5 = '#2b2727'

color6 = '#783bbd'

color7 = '#ff2052'

color8 = '#ffcc33'

color9 = '#800000'

colors = [color1, color2, color3, color4, color5, color6, color7, color8, color9]

colors_map = [color1, color2, color3, color4]



df = pd.read_csv("../input/ks-projects-201801.csv")

print("The dataset has %d rows and %d columns." % df.shape)
print(df.shape)

print()

print(df.info())

print()

print(df.isnull().sum())
df.head(5)
def preprocess_data(df):

    new_df = df.drop(columns=['ID','currency','goal','pledged','usd pledged'])

    new_df.rename(inplace=True, columns={'usd_goal_real': 'goal', 'usd_pledged_real': 'pledged'})

    new_df['launched'] = pd.to_datetime(new_df['launched'].str.split(' ',expand=True)[0], format='%Y-%m-%d', errors='coerce')

    new_df['deadline'] = pd.to_datetime(new_df['deadline'].str.split(' ',expand=True)[0], format='%Y-%m-%d', errors='coerce')

    new_df = new_df.reindex(columns=(['country','name','main_category','category','launched','deadline','goal','pledged','backers','state']))



    return new_df
df = preprocess_data(df)

df.dropna(inplace=True)

df.head(3)
state = df['state'].value_counts()

print("State Count: ")

print(state)

print()



percentual_success = round(state / len(df["state"]) * 100,2)



print("State Percentual: ")

print(percentual_success)



fig = px.pie(df, values=state.values, names=state.index, color_discrete_sequence=colors, title='Distribution of States')

fig.show()
canceled_success = df[(df.state == 'canceled') & (df.pledged >= df.goal) & (df.pledged != 0)]

print("Canceled with pledged > goal (success): ")

print(canceled_success.count()[0])

print()



canceled_fail = df[(df.state == 'canceled') & (df.pledged < df.goal)]

print("Canceled with pledged < goal (fail): ")

print(canceled_fail.count()[0])



labels = ["canceled successful", "canceled failed"]

values = [canceled_success.count()[0], canceled_fail.count()[0]]



fig = px.pie(df, values=values, names=labels, color_discrete_sequence=colors, title='Canceled Successful/Failed')

fig.show()
def preprocess_state(df):

    new_df = df.drop(df[(df.state == 'undefined') | (df.state == 'live') | (df.state == 'suspended')].index)

    new_df.loc[(new_df.state == 'canceled') & (new_df.pledged >= new_df.goal) & (new_df.pledged != 0), 'state'] = 'successful'

    new_df.loc[(new_df.state == 'canceled') & (new_df.pledged < new_df.goal), 'state'] = 'failed'



    return new_df
df = preprocess_state(df)

state = df['state'].value_counts()

print("Kickstarter Project Results up to 2018: ")

print(state)



labels = list(state.index)

values = list(state.values)

                   

trace1 = go.Bar(x=labels, y=values, marker = dict(color = colors[:len(values)]))

layout = go.Layout(title='Kickstarter Project Results up to 2018', xaxis=dict(title='status'), yaxis=dict(title='projects'), autosize=False, width=600, height=500)

                   

fig = go.Figure(data=[trace1], layout=layout)

fig.show()
percentual_failed = round(df[df.state == 'failed'].count()[0] / len(df['state']) *100,2)

print("Fail Percentage:", percentual_failed, "%")
main_category = df['main_category'].value_counts()

main_category_success = df[df.state == 'successful']['main_category'].value_counts()

main_category_fail = df[df.state == 'failed']['main_category'].value_counts()



trace1 = go.Bar(x=main_category.index, y=main_category.values, marker = dict(color = color3), name='Projects for Main Category')

trace2 = go.Bar(x=main_category_success.index, y=main_category_success.values, marker = dict(color = color2), name='Success for Main Category')

trace3 = go.Bar(x=main_category_fail.index, y=main_category_fail.values, marker = dict(color = color1), name='Fail for Main Category')



fig = make_subplots(rows=2, cols=2, specs=[[{'colspan': 2}, None], [{}, {}]], subplot_titles=('Main Category','Successful', "Failed"), print_grid=False)



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace3, 2, 2)



fig.update_layout(showlegend=True, title="Main Category Distribuition", xaxis={'tickangle':45})

fig.show()
v = df.groupby("main_category")["state"].value_counts().unstack()

fail_ratio = v.div(v.sum(axis="columns"), axis="rows").sort_values('failed', ascending=False)



categories = [fail_ratio.index.values]

labels = [fail_ratio.columns.values]

values = [fail_ratio.values]



fig = {

  "data": [

    {

      "values": values[0][0],

      "labels": labels[0],

      "domain": {"row": 0, "column": 0},

      "name": categories[0][0],

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie",

      "title": categories[0][0],

      "marker": {"colors" : colors}

    },

    {

      "values": values[0][1],

      "labels": labels[0],

      "domain": {"row": 0, "column": 1},

      "name": categories[0][1],

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie",

      "title": categories[0][1]

    },

    {

      "values": values[0][2],

      "labels": labels[0],

      "domain": {"row": 0, "column": 2},

      "name": categories[0][2],

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie",

      "title": categories[0][2]

    },

    {

      "values": values[0][14],

      "labels": labels[0],

      "domain": {"row": 1, "column": 0},

      "name": categories[0][14],

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie",

      "title": categories[0][14]

    },

    {

      "values": values[0][13],

      "labels": labels[0],

      "domain": {"row": 1, "column": 1},

      "name": categories[0][13],

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie",

      "title": categories[0][13]

    },

    {

      "values": values[0][12],

      "labels": labels[0],

      "domain": {"row": 1, "column": 2},

      "name": categories[0][12],

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie",

      "title": categories[0][12]

    }],

  "layout": {

        "title":"Success/Fail Ratio by Main Category",

        "grid": {"rows": 2, "columns": 3},

    }

}

iplot(fig)
top_categories = df['category'].value_counts().head(15)

categories_list = top_categories.index



df_top = df.loc[df['category'].isin(categories_list)]



sub_category_success = df_top[df_top.state == 'successful']['category'].value_counts()

sub_category_fail = df_top[df_top.state == 'failed']['category'].value_counts()



trace1 = go.Bar(x=sub_category_success.index, y=sub_category_success.values, marker = dict(color = color2), name='successful')

trace2 = go.Bar(x=sub_category_fail.index, y=sub_category_fail.values, marker = dict(color = color1), name='failed')



data = [trace1, trace2]



layout = go.Layout(title='Success/Fail per Category', barmode='stack',

    xaxis={'title':'categories','categoryorder':'total descending'},

    yaxis={'title':'projects'})

fig = go.Figure(data=data, layout=layout)

fig.show()
v = df_top.groupby("category")["state"].value_counts().unstack()

fail_ratio = v.div(v.sum(axis="columns"), axis="rows").sort_values('failed', ascending=False)



categories = [fail_ratio.index.values]

labels = [fail_ratio.columns.values]

values = [fail_ratio.values]



fig = {

  "data": [

    {

      "values": values[0][0],

      "labels": labels[0],

      "domain": {"row": 0, "column": 0},

      "name": categories[0][0],

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie",

      "title": categories[0][0],

      "marker": {"colors" : colors}

    },

    {

      "values": values[0][1],

      "labels": labels[0],

      "domain": {"row": 0, "column": 1},

      "name": categories[0][1],

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie",

      "title": categories[0][1]

    },

    {

      "values": values[0][2],

      "labels": labels[0],

      "domain": {"row": 0, "column": 2},

      "name": categories[0][2],

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie",

      "title": categories[0][2]

    },

    {

      "values": values[0][14],

      "labels": labels[0],

      "domain": {"row": 1, "column": 0},

      "name": categories[0][14],

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie",

      "title": categories[0][14]

    },

    {

      "values": values[0][13],

      "labels": labels[0],

      "domain": {"row": 1, "column": 1},

      "name": categories[0][13],

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie",

      "title": categories[0][13]

    },

    {

      "values": values[0][12],

      "labels": labels[0],

      "domain": {"row": 1, "column": 2},

      "name": categories[0][12],

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie",

      "title": categories[0][12]

    }],

  "layout": {

        "title":"Success/Fail Ratio by Category",

        "grid": {"rows": 2, "columns": 3},

    }

}

iplot(fig)
launch_year = df.launched.dt.strftime('%Y')



df_succ = df[df.state == 'successful']

launch_year_succ = df_succ.launched.dt.strftime('%Y')



df_fail = df[df.state == 'failed']

launch_year_fail = df_fail.launched.dt.strftime('%Y')





year_count = launch_year.value_counts().sort_index()

year_count = year_count.drop(labels=['1970','2018'])



year_count_succ = launch_year_succ.value_counts().sort_index()



year_count_fail = launch_year_fail.value_counts().sort_index()

year_count_fail = year_count_fail.drop(labels=['1970','2018'])





trace0 = go.Scatter(

    x = year_count.index,

    y = year_count.values,

    name = 'Total Project/Year',

    marker = dict(color = color3)

)



trace1 = go.Scatter(

    x = year_count.index,

    y = year_count_fail.values,

    name = 'Failed Project/Year',

    marker = dict(color = color1)

)



trace2 = go.Scatter(

    x = year_count.index,

    y = year_count_succ.values,

    name = 'Successful Project/Year',

    marker = dict(color = color2)

)



data = [trace0, trace1, trace2]



layout = go.Layout(title='Year Statistics', barmode='stack',

    xaxis={'title':'year'},

    yaxis={'title':'projects'})

fig = go.Figure(data=data, layout=layout)



fig.show()
import calendar

fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]])

df_succ = df[df.state == 'successful']

df_fail = df[df.state == 'failed']

new_order = ['December', 'November', 'October', 'September', 'August', 'July', 'June', 'May', 'April', 'March', 'February', 'January']



# Launch Month Chart

launch_month = df.launched.dt.strftime('%B')

launch_month_succ = df_succ.launched.dt.strftime('%B')

launch_month_fail = df_fail.launched.dt.strftime('%B')



month_count = launch_month.value_counts().reindex(new_order, axis=0)

month_count_succ = launch_month_succ.value_counts().reindex(new_order, axis=0)

month_count_fail = launch_month_fail.value_counts().reindex(new_order, axis=0)



trace1 = go.Bar(y=month_count_succ.index, x=month_count_succ.values, marker = dict(color = color2), name='successful', orientation='h')

trace2 = go.Bar(y=month_count_fail.index, x=month_count_fail.values, marker = dict(color = color1), name='failed', orientation='h')



# Deadline Month Chart

dead_month = df.deadline.dt.strftime('%B')

dead_month_succ = df_succ.deadline.dt.strftime('%B')

dead_month_fail = df_fail.deadline.dt.strftime('%B')



dead_month_count = dead_month.value_counts().reindex(new_order, axis=0)

dead_month_count_succ = dead_month_succ.value_counts().reindex(new_order, axis=0)

dead_month_count_fail = dead_month_fail.value_counts().reindex(new_order, axis=0)



trace3 = go.Bar(y=dead_month_count_succ.index, x=dead_month_count_succ.values, marker = dict(color = color2), name='successful', orientation='h', showlegend=False)

trace4 = go.Bar(y=dead_month_count_fail.index, x=dead_month_count_fail.values, marker = dict(color = color1), name='failed', orientation='h', showlegend=False)



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 1)

fig.append_trace(trace3, 1, 2)

fig.append_trace(trace4, 1, 2)



annotations = []

for td, xd, yd in zip(month_count, month_count_succ, range(0,12)):

    annotations.append(dict(xref='x', yref='y',

                                x=xd / 2, y=yd,

                                text=str(round(xd / td *100)) + '%',

                                font=dict(family='Arial', size=14,

                                          color='rgb(248, 248, 255)'),

                                showarrow=False))

    

for td, xd, yd, space in zip(month_count, month_count_fail, range(0,12), month_count_succ):

    annotations.append(dict(xref='x', yref='y',

                                x=(xd / 2) + space, y=yd,

                                text=str(round(xd / td *100)) + '%',

                                font=dict(family='Arial', size=14,

                                          color='rgb(248, 248, 255)'),

                                showarrow=False))



for td, xd, yd in zip(dead_month_count, dead_month_count_succ, range(0,12)):

    annotations.append(dict(xref='x2', yref='y2',

                                x=xd / 2, y=yd,

                                text=str(round(xd / td *100)) + '%',

                                font=dict(family='Arial', size=14,

                                          color='rgb(248, 248, 255)'),

                                showarrow=False))

    

for td, xd, yd, space in zip(dead_month_count, dead_month_count_fail, range(0,12), dead_month_count_succ):

    annotations.append(dict(xref='x2', yref='y2',

                                x=(xd / 2) + space, y=yd,

                                text=str(round(xd / td *100)) + '%',

                                font=dict(family='Arial', size=14,

                                          color='rgb(248, 248, 255)'),

                                showarrow=False))

    

fig.update_layout(title='Month Statistics', barmode='stack', height=750,

    xaxis={'title':'projects'},

    yaxis={'title':'launch month'},

    xaxis2={'title':'projects'},

    yaxis2={'title':'deadline month'})

fig.update_layout(annotations=annotations)

fig.show()
fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_yaxes=True, vertical_spacing=0.001)

df_succ = df[df.state == 'successful']

df_fail = df[df.state == 'failed']

new_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']



# Launch Day Chart

launch_day = df.launched.dt.day_name()

launch_day_succ = df_succ.launched.dt.day_name()

launch_day_fail = df_fail.launched.dt.day_name()



day_count = launch_day.value_counts().reindex(new_order, axis=0)

day_count_succ = launch_day_succ.value_counts().reindex(new_order, axis=0)

day_count_fail = launch_day_fail.value_counts().reindex(new_order, axis=0)



trace1 = go.Bar(x=day_count_fail.index, y=day_count_fail.values, marker = dict(color = color1), name='failed')

trace2 = go.Bar(x=day_count_succ.index, y=day_count_succ.values, marker = dict(color = color2), name='successful')



# Deadline Day Chart

dead_day = df.deadline.dt.day_name()

dead_day_succ = df_succ.deadline.dt.day_name()

dead_day_fail = df_fail.deadline.dt.day_name()



dead_day_count = dead_day.value_counts().reindex(new_order, axis=0)

dead_day_count_succ = dead_day_succ.value_counts().reindex(new_order, axis=0)

dead_day_count_fail = dead_day_fail.value_counts().reindex(new_order, axis=0)



trace3 = go.Bar(x=dead_day_count_fail.index, y=dead_day_count_fail.values, marker = dict(color = color1), name='failed', showlegend=False)

trace4 = go.Bar(x=dead_day_count_succ.index, y=dead_day_count_succ.values, marker = dict(color = color2), name='successful', showlegend=False)



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 1)

fig.append_trace(trace3, 1, 2)

fig.append_trace(trace4, 1, 2)



annotations = []

for td, xd, yd in zip(day_count, range(0,7), day_count_fail):

    annotations.append(dict(xref='x', yref='y',

                                x=xd, y=yd / 2,

                                text=str(round(yd / td *100)) + '%',

                                font=dict(family='Arial', size=14,

                                          color='rgb(248, 248, 255)'),

                                showarrow=False))

    

for td, xd, yd, space in zip(day_count, range(0,7), day_count_succ, day_count_fail):

    annotations.append(dict(xref='x', yref='y',

                                x=xd, y=(yd / 2) + space,

                                text=str(round(yd / td *100)) + '%',

                                font=dict(family='Arial', size=14,

                                          color='rgb(248, 248, 255)'),

                                showarrow=False))

    

for td, xd, yd in zip(dead_day_count, range(0,7), dead_day_count_fail):

    annotations.append(dict(xref='x2', yref='y2',

                                x=xd, y=yd / 2,

                                text=str(round(yd / td *100)) + '%',

                                font=dict(family='Arial', size=14,

                                          color='rgb(248, 248, 255)'),

                                showarrow=False))

    

for td, xd, yd, space in zip(dead_day_count, range(0,7), dead_day_count_succ, dead_day_count_fail):

    annotations.append(dict(xref='x2', yref='y2',

                                x=xd, y=(yd / 2) + space,

                                text=str(round(yd / td *100)) + '%',

                                font=dict(family='Arial', size=14,

                                          color='rgb(248, 248, 255)'),

                                showarrow=False))



fig.update_layout(title='Week Statistics', barmode='stack', height=750,

    xaxis={'title':'launch day'},

    yaxis={'title':'projects'},

    xaxis2={'title':'deadline day'},

    yaxis2={'title':'projects'})

fig.update_layout(annotations=annotations)

fig.show()
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

labels = ["Weekday", "Weekend"]



# Launch Day Chart

launch_day = df.launched.dt.weekday.apply(lambda x: 1 if x > 4 else 0)

day_count = launch_day.value_counts()



dead_day = df.deadline.dt.weekday.apply(lambda x: 1 if x > 4 else 0)

dead_day_count = dead_day.value_counts()



trace1 = go.Pie(labels=labels, values=day_count.values, title='Launch Day', pull=[0, 0.1], hole=.5, marker=dict(colors=colors))

trace2 = go.Pie(labels=labels, values=dead_day_count.values, title='Deadline Day', pull=[0, 0.1], hole=.5)



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)



fig.update_layout(title='Weekday/Weekend')

fig.show()
import datetime

project_days = df.deadline-df.launched



display(project_days.value_counts())

print()

display(project_days.value_counts().index)



project_days = project_days.dt.days

project_days_count = project_days.value_counts()



outliers_indx = []

for x in range(len(project_days_count)):

    if project_days_count.index[x] > 100:

        outliers_indx.append(x)



project_days_count.drop(project_days_count.index[outliers_indx], inplace=True)
project_days_count.sort_index(inplace=True)

v = (project_days_count.values/project_days_count.sum() * 100)



color=np.array(['rgb(255,255,255)']*v.shape[0])

color[v<2]=color1

color[v>=2]=color2



trace1 = go.Bar(x=project_days_count.index, y=v, marker = dict(color = color), name='Projects for Main Category')

layout = go.Layout(title='Project Duration', xaxis=dict(title='days'), yaxis=dict(title='% of total projects', type='log'), bargap=0)



day_int = np.arange(0,100,5)

fig = go.Figure(data=[trace1], layout=layout)

fig.update_layout(yaxis_tickformat = '.2f', yaxis = dict(tickmode = 'auto'), xaxis = dict(tickmode = 'array', tickvals = day_int))

fig.show()
import pycountry

#Remove projects with undefined country for the sake of next graph

df_c = df[df.country != 'N,0"']

country_count = round((df_c.country.value_counts() / project_days_count.sum() * 100),3)

country_list = []



for x in range(len(country_count)):

    country_list.append(pycountry.countries.get(alpha_2=country_count.index[x]))



for x in range(len(country_list)):

    country_count.rename(index={country_count.index[x]:country_list[x].alpha_3},inplace=True)





fig = go.Figure(data=go.Choropleth(

    locations=country_count.index,

    z = country_count.values,

    colorscale = colors_map,

    colorbar_title = "Total Project %",

    autocolorscale=False,

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_ticksuffix = '%',

))

fig.update_layout(title='Project % by Country', height=600, geo=dict(showframe=False, showcoastlines=False, projection_type='equirectangular'))

fig.show()
df_c = df[df.country != 'N,0"']

df_c_succ = df[(df.country != 'N,0"') & (df.state == 'successful')]

df_c_fail = df[(df.country != 'N,0"') & (df.state == 'failed')]



country_count_succ = pd.DataFrame(df_c_succ['country'].value_counts())

country_count_fail = pd.DataFrame(df_c_fail['country'].value_counts())

country_count = country_count_succ.join(country_count_fail, lsuffix='_successful', rsuffix='_failed')



results = ["successful", "failed"]

labels = list(country_count.index)

values = list(country_count.values)



fig = make_subplots(rows=4, cols=6, subplot_titles=labels)





for i, country in enumerate(values):

    i = i+1

    succ = country[0]

    fail = country[1]

    tot = succ + fail

    text_lab = [str(round(succ/tot*100)) + "%", str(round(fail/tot*100)) + "%"] 

    

    trace1 = go.Bar(x=results, y=country, name=labels[i-1], text=text_lab, marker = dict(color = colors))

    if i < 7:

        fig.append_trace(trace1, 1, i)

    if i >= 7 and i < 13:

        fig.append_trace(trace1, 2, i-6)

    if i >= 13 and i < 19:

        fig.append_trace(trace1, 3, i-12)

    if i >= 19 and i <= 22:

        fig.append_trace(trace1, 4, i-18)



fig.update_layout(title='Success/Fail per Country', showlegend=False, height=700)

fig.show()
goal_diff = df.pledged-df.goal

df['goal_diff'] = goal_diff

df = df.reindex(columns=(['country','name','main_category','category','launched','deadline','goal','pledged','goal_diff','backers','state']))

df.head()
def getbudget(money):

    if(money >= 100000):

        return 'Extra Large'

    if(money >= 10000):

        return 'Large'

    if(money >= 1000):

        return 'Medium'

    if(money >= 100):

        return 'Small'

    else:

        return 'Extra Small'



df['goal'] = df.apply(lambda x: getbudget(x['goal']), axis = 1 )

df['pledged'] = df.apply(lambda x: getbudget(x['pledged']), axis = 1 )



df.head(5)
goal_count = df['goal'].value_counts()

print("Goal:")

display(goal_count)

print()

print("Pledged:")

pledged_count = df['pledged'].value_counts()

display(pledged_count)
df_succ = df[(df.state == 'successful')]

df_fail = df[(df.state == 'failed')]



goal_count_succ = df_succ['goal'].value_counts()

pledged_count_succ = df_succ['pledged'].value_counts()



goal_count_fail = df_fail['goal'].value_counts()

pledged_count_fail = df_fail['pledged'].value_counts()



fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], vertical_spacing=0.001)



trace1 = go.Bar(y=goal_count_succ.index, x=goal_count_succ.values, marker = dict(color = color2), name='successful', orientation='h')

trace2 = go.Bar(y=goal_count_fail.index, x=goal_count_fail.values, marker = dict(color = color1), name='failed', orientation='h')

trace3 = go.Bar(y=pledged_count_succ.index, x=pledged_count_succ.values, marker = dict(color = color2), name='successful', orientation='h', showlegend=False)

trace4 = go.Bar(y=pledged_count_fail.index, x=pledged_count_fail.values, marker = dict(color = color1), name='failed', orientation='h', showlegend=False)



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 1)

fig.append_trace(trace3, 1, 2)

fig.append_trace(trace4, 1, 2)



fig.update_layout(title='Success/Fail per Goal and Pledged Category',

    xaxis={'title':'goal category', 'categoryorder':'total descending'},

    yaxis={'title':'launch month'},

    xaxis2={'title':'pledged category', 'categoryorder':'total descending'})

fig.show()
#df["state"]=LabelEncoder().fit_transform(df["state"])

fig = px.parallel_categories(df, dimensions=['goal', 'pledged', 'state'],

                range_color=[0,1],

                labels={'goal':'Goal Category', 'pledged':'Pledged Category', 'state':'Project State'})

fig.update_layout(showlegend=False)

fig.show()
diff_count_succ = df_succ[(df_succ.goal_diff >= 0)]

diff_count_fail = df_fail[(df_fail.goal_diff < 0)]



fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]])



trace1 = go.Box(y=diff_count_fail['goal_diff'], name='failed', boxpoints=False, marker_color = color1)

trace2 = go.Box(y=diff_count_succ['goal_diff'], name='successful', boxpoints=False, marker_color = color2)



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)



fig['layout']['yaxis1'].update(range=[-30000, 0])

fig['layout']['yaxis2'].update(range=[0, 3000])

fig.update_layout(title='Goal Difference per Successful/Failed Projects')

fig.show()



print('Successful goal differences:')

display(diff_count_succ['goal_diff'].value_counts())

print()

print('Failed goal differences:')

display(diff_count_fail['goal_diff'].value_counts())
backers_fail = df_fail['backers'].value_counts()[:50].sort_index()

backers_succ = df_succ['backers'].value_counts()[:100].sort_index()



backers_fail_mean = df_fail['backers'].value_counts().mean

backers_succ_mean = df_succ['backers'].value_counts().mean



fig = make_subplots(rows=2, cols=1)



trace1 = go.Scatter(

    x=list(backers_fail.index), y=list(backers_fail.values), marker = dict(color = color1), name='failed')

trace2 = go.Scatter(

    x=list(backers_succ.index), y=list(backers_succ.values), marker = dict(color = color2), name='successful')



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 2, 1)

fig.update_layout(

            title_text='Number of Backers',

            yaxis1={'title':'projects'},

            yaxis2={'title':'projects'},

            xaxis2={'title':'backers'})

fig.show()
df.head(5)
def encode_features(df):

    features = ['country', 'main_category', 'category', 'goal', 'launch_year', 'launch_month', 'launch_day', 'deadline_year', 'deadline_month', 'deadline_day']



    for feature in features:

        lab = preprocessing.LabelEncoder()

        lab = lab.fit(df[feature])

        df[feature] = lab.transform(df[feature])

    return df



def encode_date(df):

    features = ['launched', 'deadline']



    for feature in features:

        if feature is 'launched':

            year = df.launched.dt.strftime('%Y')

            df['launch_year'] = year

            month = df.launched.dt.strftime('%B')

            df['launch_month'] = month

            day = df.launched.dt.day_name()

            df['launch_day'] = day

            df = df.drop(columns='launched')

            

        if feature is 'deadline':

            year = df.deadline.dt.strftime('%Y')

            df['deadline_year'] = year

            month = df.deadline.dt.strftime('%B')

            df['deadline_month'] = month

            day = df.deadline.dt.day_name()

            df['deadline_day'] = day

            df = df.drop(columns='deadline')

            

    df = df[df.launch_year != '1970']

    return df
X = df.set_index("name")

X = X.drop(columns=["pledged", "goal_diff", "backers"])

X = encode_date(X)



Y = X.state

X = X.drop(columns="state")



#one hot encoding labels

X_h = pd.get_dummies(X)

#normal labels

X_l = encode_features(X)



X_l.head(5)
Y.head(5)
Y=LabelEncoder().fit_transform(Y.values)

scal = StandardScaler()

X_l = scal.fit_transform(X_l)
def test_classifier(classifier, class_name, dict):

    y_pred = []

    if class_name == "Gradient Boosting":

        y_pred = classifier.fit(X_train, y_train, early_stopping_rounds=5,

             eval_set=[(X_test, y_test)], verbose=False).predict(X_test)

        y_pred = [round(value) for value in y_pred]

    else:

        classifier = classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

    

    # Compute AUC ROC

    roc = roc_auc_score(y_test, y_pred)

    

    # Compute confusion matrix

    matrix = confusion_matrix(y_test, y_pred)



    TN = matrix[0][0]

    FN = matrix[1][0]

    TP = matrix[1][1]

    FP = matrix[0][1]



    # Sensitivity, hit rate, recall, or true positive rate

    TPR = TP/(TP+FN)

    # Specificity or true negative rate

    TNR = TN/(TN+FP)

    # Precision or positive predictive value

    PPV = TP/(TP+FP)

    # Negative predictive value

    NPV = TN/(TN+FN)

    # Fall out or false positive rate

    FPR = FP/(FP+TN)

    # False negative rate

    FNR = FN/(FN+TP)

    # False discovery rate

    FDR = FP/(FP+TP)

    # Overall accuracy

    ACC = (TP+TN)/(TP+FP+FN+TN)



    print("{} Scores:\n".format(class_name))

    print("AUC (ROC): {0:.2f} %\nAccuracy: {1:.2f} %\nPrecision (PPV): {2:.2f} %\nFall-out (FPR): {3:.2f} %\nSensitivity (TPR): {4:.2f} %\nFalse Negative Rate (FNR): {5:.2f} %\n\n"

        .format(roc.round(4)*100.0,ACC.round(4)*100.0,PPV.round(4)*100.0,FPR.round(4)*100.0,TPR.round(4)*100.0,FNR.round(4)*100.0))



    dict["classifier"].append(class_name)

    dict["accuracy"].append(ACC.round(4)*100.0)

    dict["auc"].append(roc.round(4)*100.0)

    dict["fallout"].append(FPR.round(4)*100.0)

    dict["fnr"].append(FNR.round(4)*100.0)

    return classifier
X_train, X_test, y_train, y_test = train_test_split(X_l, Y, test_size=0.2, random_state=1)

mc_l = defaultdict(list)
test_classifier(LogisticRegression(max_iter=10000, class_weight='balanced'), "Logistic Regression", mc_l)
#Too much computational effort with this large dataset

"""

C_list = np.logspace(start=-2, stop=2, num=5)

gamma_l = np.logspace(start=-2, stop=2, num=5)



param_grid = {'C': C_list, 'gamma': gamma_l, 'kernel': ['rbf']}



clf = GridSearchCV(SVC(), param_grid, verbose=5, cv=2, n_jobs=-1).fit(X_train, y_train)

C = clf.best_params_['C']

gamma = clf.best_params_['gamma']

test_classifier(SVC(kernel='linear', C=1.0), "Optimized RBF SVM", mc_l)

"""
grid = {

    'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], # learning rate

    'penalty': ["l2", "l1", "elasticnet"] # penalty

}



clf = SGDClassifier(loss='hinge', max_iter=1000, class_weight='balanced')

model = GridSearchCV(clf, grid, verbose=1, cv=5, scoring='roc_auc').fit(X_train, y_train)

best_alpha = model.best_params_['alpha']

best_penalty = model.best_params_['penalty']
test_classifier(SGDClassifier(loss="hinge", max_iter=1000, alpha=best_alpha, penalty=best_penalty, 

                              class_weight='balanced'), "SGD Classifier", mc_l)
test_classifier(KNeighborsClassifier(), "K-Nearest Kneighbors", mc_l)
test_classifier(DecisionTreeClassifier(class_weight='balanced'), "Decision Tree", mc_l)
parameters = {'n_estimators': [10],

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10]

              }



clf = RandomForestClassifier(class_weight='balanced')



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, verbose=1, scoring='roc_auc').fit(X_train, y_train)

# Set the clf to the best combination of parameters

best_estimator = grid_obj.best_params_['n_estimators']

best_criterion = grid_obj.best_params_['criterion']

best_depth = grid_obj.best_params_['max_depth']



clf = test_classifier(RandomForestClassifier(n_estimators=best_estimator, criterion=best_criterion,

                                             max_depth=best_depth, class_weight='balanced'), "Random Forest", mc_l)



feature_importances = pd.DataFrame(clf.feature_importances_, index = X.columns, columns=['importance']).sort_values('importance',ascending=False)
display(feature_importances)
top_features = feature_importances['importance'].head(10)



trace1 = go.Bar(x=top_features.index, y=top_features.values, marker = dict(color = color2), name='successful')



data = [trace1]



layout = go.Layout(title='Features Importance',

    xaxis={'title':'feature','categoryorder':'total descending'},

    yaxis={'title':'feature importance'})

fig = go.Figure(data=data, layout=layout)

fig.show()
test_classifier(XGBRegressor(n_estimators=1000, learning_rate=0.05), "Gradient Boosting", mc_l)
comparison_l = pd.DataFrame(mc_l, columns=['classifier','auc','fallout', 'fnr'])
pca = PCA(.85) # retain 85% of the variance

X_r = pca.fit_transform(X_l)
X_train, X_test, y_train, y_test = train_test_split(X_r, Y, test_size=0.2, random_state=1)

mc_r = defaultdict(list)



### SGD

test_classifier(SGDClassifier(loss="hinge", max_iter=1000, alpha=best_alpha, penalty=best_penalty, 

                              class_weight='balanced'), "SGD Classifier", mc_r)



### Random Forest

clf = RandomForestClassifier(class_weight='balanced')

acc_scorer = make_scorer(accuracy_score)

grid_obj = GridSearchCV(clf, parameters, scoring='roc_auc').fit(X_train, y_train)

test_classifier(grid_obj.best_estimator_, "Random Forest", mc_r)



### XGBR

test_classifier(XGBRegressor(n_estimators=1000, learning_rate=0.05), "Gradient Boosting", mc_r)



comparison_r = pd.DataFrame(mc_r, columns=['classifier','auc','fallout', 'fnr'])
X_train, X_test, y_train, y_test = train_test_split(X_h, Y, test_size=0.2, random_state=1)

mc_h = defaultdict(list)

print(X_h.shape)



### SGD

test_classifier(SGDClassifier(loss="hinge", max_iter=1000, alpha=best_alpha, penalty=best_penalty, 

                              class_weight='balanced'), "SGD Classifier", mc_h)



### Random Forest

clf = RandomForestClassifier(class_weight='balanced')

acc_scorer = make_scorer(accuracy_score)

grid_obj = GridSearchCV(clf, parameters, scoring='roc_auc').fit(X_train, y_train)

test_classifier(grid_obj.best_estimator_, "Random Forest", mc_h)



### XGBR

test_classifier(XGBRegressor(n_estimators=100, learning_rate=0.05), "Gradient Boosting", mc_h)



comparison_h = pd.DataFrame(mc_h, columns=['classifier','auc','fallout','fnr'])
comparison_h = comparison_h.set_index("classifier")

comparison_r = comparison_r.set_index("classifier")

comparison_l = comparison_l.set_index("classifier")

comparison = comparison_h.join(comparison_r,rsuffix = "_pca",lsuffix="_1hot")

comparison = comparison.join(comparison_l,rsuffix = "_label")
display(comparison)

fig = px.bar(comparison, color_discrete_sequence=colors, barmode='group')

fig.update_layout(

            title_text='Classification Comparison',

            yaxis={'title':'auc/fallout/fnr'},

            xaxis={'title':'classifier'})

fig.show()