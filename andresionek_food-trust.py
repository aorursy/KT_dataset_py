# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
fhrs = pd.read_csv('/kaggle/input/pivigo-fsa-hackaton-2019/trust_FHRS_All_en-GB_2019-08-20.csv')
for col in fhrs.columns:

    print(col)
import datashader as ds

from datashader import transfer_functions as tf

from functools import partial

from datashader.utils import export_image

from IPython.core.display import HTML, display

from datashader.colors import colormap_select, Greys9

from colorcet import fire, rainbow, bgy, bjy, bkr, kb, kr, kbc



background = "black"

cm = partial(colormap_select, reverse=(background!="black"))

export = partial(export_image, background = background, export_path="export")

display(HTML("<style>.container { width:100% !important; }</style>"))



def create_map_cmap(data, cmap, data_agg, how, export_name='img', W=1500):

    pad = (data.x.max() - data.x.min())/50

    x_range, y_range = ((data.x.min() - pad, data.x.max() + pad), 

                             (data.y.min() - pad, data.y.max() + pad))



    ratio = (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])



    plot_width  = int(W)

    plot_height = int(plot_width * ratio)

    if ratio > 1.5:

        plot_height = 550

        plot_width = int(plot_height / ratio)

        

    cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height, x_range=x_range, y_range=y_range)



    agg = cvs.points(data, 'x', 'y', data_agg)

    img = tf.shade(agg, cmap=cmap, how=how)

    return export(img, export_name)



def create_map_ckey(data, color_key, data_agg, how, export_name='img', W=1500):

    pad = (data.x.max() - data.x.min())/50

    x_range, y_range = ((data.x.min() - pad, data.x.max() + pad), 

                             (data.y.min() - pad, data.y.max() + pad))



    ratio = (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])



    plot_width  = int(W)

    plot_height = int(plot_width * ratio)

    if ratio > 1.5:

        plot_height = 550

        plot_width = int(plot_height / ratio)

        

    cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height, x_range=x_range, y_range=y_range)



    agg = cvs.points(data, 'x', 'y', data_agg)

    img = tf.shade(agg, color_key=color_key, how=how)

    return export(img, export_name)
from datashader.utils import lnglat_to_meters as webm



fhrs_map = fhrs[['PostCode', 'LocalAuthorityName', 'BusinessType', 'Latitude', 'Longitude', 'RatingValue']].copy()

fhrs_map.RatingValue = fhrs_map.RatingValue.str.replace('[^0-9]', '', regex=True)

fhrs_map.RatingValue = fhrs_map.RatingValue.replace('', np.nan)

fhrs_map.dropna(inplace=True)

fhrs_map.RatingValue = fhrs_map.RatingValue.astype(int)

# filtrando somente manhattan

y_range = (49, 60)

fhrs_map = fhrs_map[(fhrs_map.Latitude > y_range[0]) & (fhrs_map.Latitude < y_range[1])]



x, y = webm(fhrs_map.Longitude, fhrs_map.Latitude)

fhrs_map['x'] = pd.Series(x)

fhrs_map['y'] = pd.Series(y)



fhrs_map.RatingValue.value_counts()
fhrs_map_num = fhrs_map.copy()

create_map_cmap(fhrs_map_num, fire, ds.mean('RatingValue'), 'linear', 'RatingValue')
london_la = [

'City of London',

'City of Westminster',

'Kensington and Chelsea',

'Hammersmith and Fulham',

'Wandsworth',

'Lambeth',

'Southwark',

'Tower Hamlets',

'Hackney',

'Islington',

'Camden',

'Brent',

'Ealing',

'Hounslow',

'Richmond',

'Kingston',

'Merton',

'Sutton',

'Croydon',

'Bromley',

'Lewisham',

'Greenwich',

'Bexley',

'Havering',

'Barking and Dagenham',

'Redbridge',

'Newham',

'Waltham Forest',

'Haringey',

'Enfield',

'Barnet',

'Harrow',

'Hillingdon']   

fhrs_map_no45 = fhrs_map[fhrs_map.RatingValue < 7].copy()

#fhrs_map_no45.RatingValue = fhrs_map_no45.RatingValue.astype('category')

color_key = {0:'fuchsia', 1:'fuchsia', 2:'yellow',  3:'yellow', 4:'cyan', 5:'lime'}

create_map_cmap(fhrs_map_no45[fhrs_map_no45.LocalAuthorityName.isin(london_la)], fire, ds.mean('RatingValue'), 'log', 'RatingValue', 800)

for c in fu.columns:

    print(c)


fu = pd.read_csv('/kaggle/input/pivigo-fsa-hackaton-2019/trust_food_and_you_wave_5.csv', low_memory=False)

fu_filtered = fu[['age_dv', 'RespSex', 'marstat2', 'famtype', 'bhhsize', 'below16', 'below6',

         'hhdinc', 'workstat', 'bethnicity', 'religion_dv', 'region_dv', 'country_dv', 'ProvFood3', 'Label', 'dietary', 'q4_27_5_slice'

]].copy()

fu_filtered.columns = ['Age', 'Sex', 'MaritalStatus', 'FamilyType', 'HouseholdSize', 'ChildrenBelow16', 'ChildrenBelow6',

                       'HouseholdIncome', 'WorkStatus', 'Ethnicity', 'Religion', 'Region', 'Country', 'TrustUKFood', 'Label', 'dietary', 'Hygiene']



cats = ['16-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']

cat_dtype = pd.api.types.CategoricalDtype(categories=cats, ordered=True)

fu_filtered.Age = fu_filtered.Age.astype(cat_dtype)



cats = ['One', 'Two', 'Three', 'Four', 'Five or More']

cat_dtype = pd.api.types.CategoricalDtype(categories=cats, ordered=True)

fu_filtered.HouseholdSize = fu_filtered.HouseholdSize.astype(cat_dtype)





cats = ['<£10,399', '£10,400 - £25,999', '£26,000 - £51,999', '>£52,000']

cat_dtype = pd.api.types.CategoricalDtype(categories=cats, ordered=True)

fu_filtered.HouseholdIncome = fu_filtered.HouseholdIncome.astype(cat_dtype)





cats = ['Definitely disagree', 'Tend to disagree', 'Neither agree nor disagree', 'Tend to agree', 'Definitely agree']

cat_dtype = pd.api.types.CategoricalDtype(categories=cats, ordered=True)

fu_filtered.TrustUKFood = fu_filtered.TrustUKFood.astype(cat_dtype)





cats = ['Definitely disagree', 'Tend to disagree', 'Neither agree nor disagree', 'Tend to agree', 'Definitely agree']

cat_dtype = pd.api.types.CategoricalDtype(categories=cats, ordered=True)

fu_filtered.Hygiene = fu_filtered.Hygiene.astype(cat_dtype)



fu_filtered['Count'] = 1
import plotly.graph_objects as go

import plotly.graph_objects as go

import plotly.offline as pyo

import plotly.express as px

from plotly.subplots import make_subplots



pyo.init_notebook_mode()



def plot_hbar(xdata, ydata):

    top_labels = ['Definitely<br>disagree', 'Tend to<br>disagree', 'Neither agree<br>nor disagree',

                  'Tend to<br>agree', 'Definitely<br>agree']



    colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',

              'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)',

              'rgba(190, 192, 213, 1)']



    x_data = xdata



    y_data = ydata



    fig = go.Figure()



    for i in range(0, len(x_data[0])):

        for xd, yd in zip(x_data, y_data):

            if yd in ['75+', '65-74'] and i > 3:

                calc_color = 'lightblue' 

            elif  i > 3:

                calc_color = 'darkgrey'

            else:

                calc_color = 'lightgrey'

            fig.add_trace(go.Bar(

                x=[xd[i]], y=[yd],

                orientation='h',

                marker=dict(

                    color=calc_color,

                    line=dict(color='rgb(248, 248, 249)', width=1)

                )

            ))



    fig.update_layout(

        xaxis=dict(

            showgrid=False,

            showline=False,

            showticklabels=False,

            zeroline=False,

            domain=[0.15, 1]

        ),

        yaxis=dict(

            showgrid=False,

            showline=False,

            showticklabels=False,

            zeroline=False,

        ),

        barmode='stack',

        paper_bgcolor='white',

        plot_bgcolor='white',

        margin=dict(l=10, r=10, t=140, b=80),

        showlegend=False,

    )



    annotations = []



    for yd, xd in zip(y_data, x_data):

        # labeling the y-axis

        annotations.append(dict(xref='paper', yref='y',

                                x=0.14, y=yd,

                                xanchor='right',

                                text=str(yd),

                                font=dict(family='Arial', size=14,

                                          color='grey'),

                                showarrow=False, align='right'))

        # labeling the first percentage of each bar (x_axis)

        annotations.append(dict(xref='x', yref='y',

                                x=xd[0] / 2, y=yd,

                                text=str(xd[0].round()) + '%',

                                font=dict(family='Arial', size=10,

                                          color='rgb(248, 248, 255)'),

                                showarrow=False))

        # labeling the first Likert scale (on the top)

        if yd == y_data[-1]:

            annotations.append(dict(xref='x', yref='paper',

                                    x=xd[0], y=1.1,

                                    text=top_labels[0],

                                    font=dict(family='Arial', size=14,

                                              color='gray'),

                                    showarrow=False))

        space = xd[0]

        for i in range(1, len(xd)):

                # labeling the rest of percentages for each bar (x_axis)

                annotations.append(dict(xref='x', yref='y',

                                        x=space + (xd[i]/2), y=yd,

                                        text=str(xd[i].round(0)) + '%',

                                        font=dict(family='Arial', size=10,

                                                  color='rgb(248, 248, 255)'),

                                        showarrow=False))

                # labeling the Likert scale

                if yd == y_data[-1]:

                    annotations.append(dict(xref='x', yref='paper',

                                            x=20 * i, y=1.1,

                                            text=top_labels[i],

                                            font=dict(family='Arial', size=14,

                                                      color='lightblue' if i > 3 else 'gray'),

                                            showarrow=False))

                space += xd[i]



    fig.update_layout(annotations=annotations, 

                      title='Older People have Better Trust in foo produced in UK, compared to food imported from overseas',

                      title_font_color='black')



    fig.show()
fu_age = fu_filtered.groupby(['Age', 'TrustUKFood'], as_index=False).Count.count()

ages = ['16-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']

data = []



for age in ages:

    age_data = fu_age[fu_age.Age == age].Count.tolist()

    age_data = (np.array(age_data) / np.array(age_data).sum() * 100)

    data.append(age_data)

plot_hbar(data, ages)
import plotly.graph_objects as go



def plot_hbar(xdata, ydata):

    top_labels = ['Definitely<br>disagree', 'Tend to<br>disagree', 'Neither agree<br>nor disagree',

                  'Tend to<br>agree', 'Definitely<br>agree']



    colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',

              'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)',

              'rgba(190, 192, 213, 1)']



    x_data = xdata



    y_data = ydata



    fig = go.Figure()



    for i in range(0, len(x_data[0])):

        for xd, yd in zip(x_data, y_data): 

            if  i < 2 and yd in ['£26,000 - £51,999', '>£52,000']:

                calc_color = 'lightblue'

            elif i < 2:

                calc_color = 'darkgrey'

            else:

                calc_color = 'lightgrey'

            fig.add_trace(go.Bar(

                x=[xd[i]], y=[yd],

                orientation='h',

                marker=dict(

                    color=calc_color,

                    line=dict(color='rgb(248, 248, 249)', width=1)

                )

            ))



    fig.update_layout(

        xaxis=dict(

            showgrid=False,

            showline=False,

            showticklabels=False,

            zeroline=False,

            domain=[0.15, 1]

        ),

        yaxis=dict(

            showgrid=False,

            showline=False,

            showticklabels=False,

            zeroline=False,

        ),

        barmode='stack',

        paper_bgcolor='white',

        plot_bgcolor='white',

        margin=dict(l=10, r=10, t=140, b=80),

        showlegend=False,

    )



    annotations = []



    for yd, xd in zip(y_data, x_data):

        # labeling the y-axis

        annotations.append(dict(xref='paper', yref='y',

                                x=0.14, y=yd,

                                xanchor='right',

                                text=str(yd),

                                font=dict(family='Arial', size=14,

                                          color='grey'),

                                showarrow=False, align='right'))

        # labeling the first percentage of each bar (x_axis)

        annotations.append(dict(xref='x', yref='y',

                                x=xd[0] / 2, y=yd,

                                text=str(xd[0].round()) + '%',

                                font=dict(family='Arial', size=10,

                                          color='rgb(248, 248, 255)'),

                                showarrow=False))

        # labeling the first Likert scale (on the top)

        if yd == y_data[-1]:

            annotations.append(dict(xref='x', yref='paper',

                                    x=xd[0], y=1.1,

                                    text=top_labels[0],

                                    font=dict(family='Arial', size=14,

                                              color='lightblue'),

                                    showarrow=False))

        space = xd[0]

        for i in range(1, len(xd)):

                # labeling the rest of percentages for each bar (x_axis)

                annotations.append(dict(xref='x', yref='y',

                                        x=space + (xd[i]/2), y=yd,

                                        text=str(xd[i].round(0)) + '%',

                                        font=dict(family='Arial', size=10,

                                                  color='rgb(248, 248, 255)'),

                                        showarrow=False))

                # labeling the Likert scale

                if yd == y_data[-1]:

                    annotations.append(dict(xref='x', yref='paper',

                                            x=20 * i, y=1.1,

                                            text=top_labels[i],

                                            font=dict(family='Arial', size=14,

                                                      color='lightblue' if i < 2  else 'gray'),

                                            showarrow=False))

                space += xd[i]



    fig.update_layout(annotations=annotations, 

                      title='People with higher income tend to distrust food produced in UK compared to food imported from overseas',

                      title_font_color='black')



    fig.show()
fu_income = fu_filtered.groupby(['HouseholdIncome', 'TrustUKFood'], as_index=False).Count.count()

incomes = ['<£10,399', '£10,400 - £25,999', '£26,000 - £51,999', '>£52,000']

data = []



for income in incomes:

    agg_data = fu_income[fu_income.HouseholdIncome == income].Count.tolist()

    agg_data = (np.array(agg_data) / np.array(agg_data).sum() * 100)

    data.append(agg_data)

plot_hbar(data, incomes)
import plotly.graph_objects as go



def plot_hbar(xdata, ydata):

    top_labels = ['Definitely<br>disagree', 'Tend to<br>disagree', 'Neither agree<br>nor disagree',

                  'Tend to<br>agree', 'Definitely<br>agree']



    colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',

              'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)',

              'rgba(190, 192, 213, 1)']



    x_data = xdata



    y_data = ydata



    fig = go.Figure()



    for i in range(0, len(x_data[0])):

        for xd, yd in zip(x_data, y_data): 

            if  i > 3 and yd == 'Yes':

                calc_color = 'darkgrey'

            elif i > 3:

                calc_color = 'lightblue'

            else:

                calc_color = 'lightgrey'

            fig.add_trace(go.Bar(

                x=[xd[i]], y=[yd],

                orientation='h',

                marker=dict(

                    color=calc_color,

                    line=dict(color='rgb(248, 248, 249)', width=1)

                )

            ))



    fig.update_layout(

        xaxis=dict(

            showgrid=False,

            showline=False,

            showticklabels=False,

            zeroline=False,

            domain=[0.2, 1]

        ),

        yaxis=dict(

            showgrid=False,

            showline=False,

            showticklabels=False,

            zeroline=False,

        ),

        barmode='stack',

        paper_bgcolor='white',

        plot_bgcolor='white',

        margin=dict(l=10, r=10, t=140, b=80),

        showlegend=False,

    )



    annotations = []



    for yd, xd in zip(y_data, x_data):

        # labeling the y-axis

        annotations.append(dict(xref='paper', yref='y',

                                x=0.14, y=yd,

                                xanchor='right',

                                text=str(yd),

                                font=dict(family='Arial', size=14,

                                          color='grey'),

                                showarrow=False, align='right'))

        # labeling the first percentage of each bar (x_axis)

        annotations.append(dict(xref='x', yref='y',

                                x=xd[0] / 2, y=yd,

                                text=str(xd[0].round()) + '%',

                                font=dict(family='Arial', size=10,

                                          color='rgb(248, 248, 255)'),

                                showarrow=False))

        # labeling the first Likert scale (on the top)

        if yd == y_data[-1]:

            annotations.append(dict(xref='x', yref='paper',

                                    x=xd[0], y=1.1,

                                    text=top_labels[0],

                                    font=dict(family='Arial', size=14,

                                              color='grey'),

                                    showarrow=False))

        space = xd[0]

        for i in range(1, len(xd)):

                # labeling the rest of percentages for each bar (x_axis)

                annotations.append(dict(xref='x', yref='y',

                                        x=space + (xd[i]/2), y=yd,

                                        text=str(xd[i].round(0)) + '%',

                                        font=dict(family='Arial', size=10,

                                                  color='rgb(248, 248, 255)'),

                                        showarrow=False))

                # labeling the Likert scale

                if yd == y_data[-1]:

                    annotations.append(dict(xref='x', yref='paper',

                                            x=20 * i, y=1.1,

                                            text=top_labels[i],

                                            font=dict(family='Arial', size=14,

                                                      color='lightblue' if i > 3  else 'gray'),

                                            showarrow=False))

                space += xd[i]



    fig.update_layout(annotations=annotations, 

                      title='Families without children tend to trust more UK food compared to overseas',

                      title_font_color='black',

                     yaxis_title_text='Has Children Bellow 6 years old',

                     yaxis_title_font_color='grey')



    fig.show()
fu_children = fu_filtered.groupby(['ChildrenBelow6', 'TrustUKFood'], as_index=False).Count.count()

labels = ['No', 'Yes']

data = []

for l in labels:

    agg_data = fu_children[fu_children.ChildrenBelow6 == l].Count.tolist()

    agg_data = (np.array(agg_data) / np.array(agg_data).sum() * 100)

    data.append(agg_data)

plot_hbar(data, labels)
import plotly.graph_objects as go



def plot_hbar(xdata, ydata):

    top_labels = ['Never', 'Rarely', 'Some of the time', 'Most of the time', 'Always', 'Don\'t know']



    colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',

              'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)',

              'rgba(190, 192, 213, 1)']



    x_data = xdata



    y_data = ydata



    fig = go.Figure()



    for i in range(0, len(x_data[0])):

        for xd, yd in zip(x_data, y_data): 

            if  i == 5:

                calc_color = 'grey'

            elif  i < 2 and yd in ['16-24', '25-34', '75+']:

                calc_color = 'lightblue'

            elif i < 2 :

                calc_color = 'grey'

            else:

                calc_color = 'lightgrey'

            fig.add_trace(go.Bar(

                x=[xd[i]], y=[yd],

                orientation='h',

                marker=dict(

                    color=calc_color,

                    line=dict(color='rgb(248, 248, 249)', width=1)

                )

            ))



    fig.update_layout(

        xaxis=dict(

            showgrid=False,

            showline=False,

            showticklabels=False,

            zeroline=False,

            domain=[0.2, 1]

        ),

        yaxis=dict(

            showgrid=False,

            showline=False,

            showticklabels=False,

            zeroline=False,

        ),

        barmode='stack',

        paper_bgcolor='white',

        plot_bgcolor='white',

        margin=dict(l=10, r=10, t=140, b=80),

        showlegend=False,

    )



    annotations = []



    for yd, xd in zip(y_data, x_data):

        # labeling the y-axis

        annotations.append(dict(xref='paper', yref='y',

                                x=0.14, y=yd,

                                xanchor='right',

                                text=str(yd),

                                font=dict(family='Arial', size=14,

                                          color='grey'),

                                showarrow=False, align='right'))

        # labeling the first percentage of each bar (x_axis)

        annotations.append(dict(xref='x', yref='y',

                                x=xd[0] / 2, y=yd,

                                text=str(xd[0].round()) + '%' if xd[0] > 5 else '' ,

                                font=dict(family='Arial', size=10,

                                          color='rgb(248, 248, 255)'),

                                showarrow=False))

        # labeling the first Likert scale (on the top)

        if yd == y_data[-1]:

            annotations.append(dict(xref='x', yref='paper',

                                    x=xd[0], y=1.1,

                                    text=top_labels[0],

                                    font=dict(family='Arial', size=14,

                                              color='grey'),

                                    showarrow=False))

        space = xd[0]

        for i in range(1, len(xd)):

                # labeling the rest of percentages for each bar (x_axis)

                annotations.append(dict(xref='x', yref='y',

                                        x=space + (xd[i]/2), y=yd,

                                        text=str(xd[i].round(0)) + '%' if xd[i] > 5 else '',

                                        font=dict(family='Arial', size=10,

                                                  color='rgb(248, 248, 255)'),

                                        showarrow=False))

                # labeling the Likert scale

                if yd == y_data[-1]:

                    annotations.append(dict(xref='x', yref='paper',

                                            x=20 * i, y=1.1,

                                            text=top_labels[i],

                                            font=dict(family='Arial', size=14,

                                                      color='grey' if i < 2 or i ==5   else 'lightgrey'),

                                            showarrow=False))

                space += xd[i]



    fig.update_layout(annotations=annotations, 

                      title='Younger people tend to distrust labels or menus as much as the elders' \

                      '<br><span style="font-size:12px">In general, when buying or eating food, how often do you feel confident that it is what it says it is on the label or the menu?</span>',

                      title_font_color='black',

                     yaxis_title_font_color='grey')



    fig.show()
cats = ['Never', 'Rarely', 'Some of the time', 'Most of the time', 'Always', 'Don\'t know']

cat_dtype = pd.api.types.CategoricalDtype(categories=cats, ordered=True)

fu_filtered.Label = fu_filtered.Label.astype(cat_dtype)



fu_age = fu_filtered.groupby(['Age', 'Label'], as_index=False).Count.count()

ages = ['16-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']

data = []



for age in ages:

    age_data = fu_age[fu_age.Age == age].Count.tolist()

    age_data = (np.array(age_data) / np.array(age_data).sum() * 100)

    data.append(age_data)

plot_hbar(data, ages)
import plotly.graph_objects as go



def plot_hbar(xdata, ydata):

    top_labels = ['Don\'t know', 'Never', 'Rarely', 'Some of the time', 'Most of the time', 'Always']



    colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',

              'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)',

              'rgba(190, 192, 213, 1)']



    x_data = xdata



    y_data = ydata



    fig = go.Figure()



    for i in range(0, len(x_data[0])):

        for xd, yd in zip(x_data, y_data): 

            if  i == 0:

                calc_color = 'lightgrey'

            elif i > 3 :

                calc_color = 'grey'

            else:

                calc_color = 'lightgrey'

            fig.add_trace(go.Bar(

                x=[xd[i]], y=[yd],

                orientation='h',

                marker=dict(

                    color=calc_color,

                    line=dict(color='rgb(248, 248, 249)', width=1)

                )

            ))



    fig.update_layout(

        xaxis=dict(

            showgrid=False,

            showline=False,

            showticklabels=False,

            zeroline=False,

            domain=[0.2, 1]

        ),

        yaxis=dict(

            showgrid=False,

            showline=False,

            showticklabels=False,

            zeroline=False,

        ),

        barmode='stack',

        paper_bgcolor='white',

        plot_bgcolor='white',

        margin=dict(l=10, r=10, t=140, b=80),

        showlegend=False,

    )



    annotations = []



    for yd, xd in zip(y_data, x_data):

        # labeling the y-axis

        annotations.append(dict(xref='paper', yref='y',

                                x=0.14, y=yd,

                                xanchor='right',

                                text=str(yd),

                                font=dict(family='Arial', size=14,

                                          color='grey'),

                                showarrow=False, align='right'))

        # labeling the first percentage of each bar (x_axis)

        annotations.append(dict(xref='x', yref='y',

                                x=xd[0] / 2, y=yd,

                                text=str(xd[0].round()) + '%' if xd[0] > 5 else '' ,

                                font=dict(family='Arial', size=10,

                                          color='rgb(248, 248, 255)'),

                                showarrow=False))

        # labeling the first Likert scale (on the top)

        if yd == y_data[-1]:

            annotations.append(dict(xref='x', yref='paper',

                                    x=xd[0], y=1.1,

                                    text=top_labels[0],

                                    font=dict(family='Arial', size=14,

                                              color='lightgrey'),

                                    showarrow=False))

        space = xd[0]

        for i in range(1, len(xd)):

                # labeling the rest of percentages for each bar (x_axis)

                annotations.append(dict(xref='x', yref='y',

                                        x=space + (xd[i]/2), y=yd,

                                        text=str(xd[i].round(0)) + '%' if xd[i] > 5 else '',

                                        font=dict(family='Arial', size=10,

                                                  color='rgb(248, 248, 255)'),

                                        showarrow=False))

                # labeling the Likert scale

                if yd == y_data[-1]:

                    annotations.append(dict(xref='x', yref='paper',

                                            x=20 * i, y=1.1,

                                            text=top_labels[i],

                                            font=dict(family='Arial', size=14,

                                                      color='grey' if i > 3   else 'darkgrey'),

                                            showarrow=False))

                space += xd[i]



    fig.update_layout(annotations=annotations, 

                      title='The trust in labels and menus decreases with income' \

                      '<br><span style="font-size:12px">In general, when buying or eating food, how often do you feel confident that it is what it says it is on the label or the menu?</span>',

                      title_font_color='black',

                     yaxis_title_font_color='grey')



    fig.show()
cats = ['Don\'t know', 'Never', 'Rarely', 'Some of the time', 'Most of the time', 'Always']

cat_dtype = pd.api.types.CategoricalDtype(categories=cats, ordered=True)

fu_filtered.Label = fu_filtered.Label.astype(cat_dtype)



fu_age = fu_filtered.groupby(['HouseholdIncome', 'Label'], as_index=False).Count.count()

fu_age.Count = fu_age.Count.fillna(0)

ages = ['<£10,399', '£10,400 - £25,999', '£26,000 - £51,999', '>£52,000']

data = []



for age in ages:

    age_data = fu_age[fu_age.HouseholdIncome == age].Count.tolist()

    age_data = (np.array(age_data) / np.array(age_data).sum() * 100)

    data.append(age_data)

plot_hbar(data, ages)
fhrs_business_type = fhrs.groupby(['BusinessType'], as_index=False).Hygiene.mean().sort_values('Hygiene', ascending=False)
fhrs_map = fhrs[['PostCode', 'LocalAuthorityName', 'BusinessType', 'Latitude', 'Longitude', 'RatingValue']].copy()

fhrs_map.RatingValue = fhrs_map.RatingValue.str.replace('[^0-9]', '', regex=True)

fhrs_map.RatingValue = fhrs_map.RatingValue.replace('', np.nan)

fhrs_map.dropna(inplace=True)

fhrs_map.RatingValue = fhrs_map.RatingValue.astype(int)

fhrs_business_type = fhrs_map.groupby(['BusinessType'], as_index=False).RatingValue.mean().sort_values('RatingValue', ascending=False)





fig = go.Figure()



x_data=fhrs_business_type.RatingValue.tolist()

y_data=fhrs_business_type.BusinessType.tolist()





for i in range(0, len(y_data)):

    if  i == 13:

        calc_color = 'grey'

    else:

        calc_color = 'lightgrey'

    fig.add_trace(go.Bar(

        x=[x_data[i]], y=[y_data[i]],

        orientation='h',

        marker=dict(

            color=calc_color,

            line=dict(color='rgb(248, 248, 249)', width=1)

        )

    ))

fig.update_layout(title='The Hygiene rating is much lower for takeaways/sandwich shops when compared to other business types',

                  title_font_color='black',

                 xaxis=dict(

            showgrid=False,

            showline=False,

            zeroline=False,

            domain=[0.2, 1],

            tickcolor='grey',

            title_text='Food Hygiene Rating',

            title_font_color='grey'

            

        ),

        yaxis=dict(

            showgrid=False,

            showline=False,

            zeroline=False,

            tickcolor='grey'

        ),

        barmode='stack',

        paper_bgcolor='white',

        plot_bgcolor='white',

        margin=dict(l=10, r=10, t=140, b=80),

        showlegend=False,

                  

    )

fig.show()
import plotly.graph_objects as go



def plot_hbar(xdata, ydata):

    top_labels = ['Definitely<br>disagree', 'Tend to<br>disagree', 'Neither agree<br>nor disagree',

                  'Tend to<br>agree', 'Definitely<br>agree']



    colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',

              'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)',

              'rgba(190, 192, 213, 1)']



    x_data = xdata



    y_data = ydata



    fig = go.Figure()



    for i in range(0, len(x_data[0])):

        for xd, yd in zip(x_data, y_data): 

            if  i > 3 and yd in ['North East', 'North West']:

                calc_color = 'grey'

            elif i > 3:

                calc_color = 'lightblue'

            else:

                calc_color = 'lightgrey'

            fig.add_trace(go.Bar(

                x=[xd[i]], y=[yd],

                orientation='h',

                marker=dict(

                    color=calc_color,

                    line=dict(color='rgb(248, 248, 249)', width=1)

                )

            ))



    fig.update_layout(

        xaxis=dict(

            showgrid=False,

            showline=False,

            showticklabels=False,

            zeroline=False,

            domain=[0.2, 1]

        ),

        yaxis=dict(

            showgrid=False,

            showline=False,

            showticklabels=False,

            zeroline=False,

        ),

        barmode='stack',

        paper_bgcolor='white',

        plot_bgcolor='white',

        margin=dict(l=10, r=10, t=140, b=80),

        showlegend=False,

    )



    annotations = []



    for yd, xd in zip(y_data, x_data):

        # labeling the y-axis

        annotations.append(dict(xref='paper', yref='y',

                                x=0.18, y=yd,

                                xanchor='right',

                                text=str(yd),

                                font=dict(family='Arial', size=14,

                                          color='grey'),

                                showarrow=False, align='right'))

        # labeling the first percentage of each bar (x_axis)

        annotations.append(dict(xref='x', yref='y',

                                x=xd[0] / 2, y=yd,

                                text='',

                                font=dict(family='Arial', size=10,

                                          color='rgb(248, 248, 255)'),

                                showarrow=False))

        # labeling the first Likert scale (on the top)

        if yd == y_data[-1]:

            annotations.append(dict(xref='x', yref='paper',

                                    x=xd[0], y=1.1,

                                    text=top_labels[0],

                                    font=dict(family='Arial', size=14,

                                              color='lightgrey'),

                                    showarrow=False))

        space = xd[0]

        for i in range(1, len(xd)):

                # labeling the rest of percentages for each bar (x_axis)

                annotations.append(dict(xref='x', yref='y',

                                        x=space + (xd[i]/2), y=yd,

                                        text=str(xd[i].round(0)) + '%' if xd[i] > 8 else '',

                                        font=dict(family='Arial', size=10,

                                                  color='rgb(248, 248, 255)'),

                                        showarrow=False))

                # labeling the Likert scale

                if yd == y_data[-1]:

                    annotations.append(dict(xref='x', yref='paper',

                                            x=20 * i, y=1.1,

                                            text=top_labels[i],

                                            font=dict(family='Arial', size=14,

                                                      color='gray' if i > 3  else 'lightgray'),

                                            showarrow=False))

                space += xd[i]



    fig.update_layout(annotations=annotations, 

                      title='North West and North East think that restaurants should pay more attention to food safety and hygiene',

                      title_font_color='black',)



    fig.show()
fu_age.Count = fu_age.Count.fillna(0)

ages = [

'South East',

'Yorkshire and The Humber',

'Wales',

'South West',

'Northern Ireland',

'East Midlands',

'East of England',

'London',

'West Midlands',

'North West',

'North East'

]

cat_dtype = pd.api.types.CategoricalDtype(categories=ages, ordered=True)

fu_filtered.Region = fu_filtered.Region.astype(cat_dtype)

fu_age = fu_filtered.groupby(['Region', 'Hygiene'], as_index=False).Count.count()



fu_age.Count = fu_age.Count.fillna(0)





data = []

for age in ages:

    age_data = fu_age[fu_age.Region == age].Count.tolist()

    age_data = (np.array(age_data) / np.array(age_data).sum() * 100)

    data.append(age_data)

plot_hbar(data, ages)