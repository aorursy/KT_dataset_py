import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re

import plotly.plotly as py
import plotly.tools as tls
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.set_config_file(offline=True, world_readable=True)

from scipy import stats

def make_crosstab_barcharts(df, column, n, specific_title):
    crosstab = pd.crosstab(df['Higher Education'], df[column]).apply(lambda r: r/r.sum() * 100, axis=1).T

    fig = tls.make_subplots(rows=1, cols=2, shared_yaxes=True, shared_xaxes=False, print_grid=False)
    i = 1
    colorscale=cf.colors.get_scales('accent')
    for col in crosstab:
        crosstab_filter = crosstab.sort_values(col, ascending=False).head(n)
        fig.append_trace({'x': crosstab_filter.index.str.replace(r"\(.*\)",""), 'y': crosstab_filter[col], 'type': 'bar', 'name': col, 'text': crosstab_filter[col].round(2).astype(str) + '%', 'textposition': 'auto', 'marker': {'color': colorscale[i-1]}}, 1, i)
        i +=1

    fig['layout'].update(title = "Data Scientists - Distribution of " + specific_title + " by Highest level of Education in %", margin = dict(b = 150, t = 100), yaxis = dict(autorange =True, showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False),
                         paper_bgcolor = "#F5F6F9",  plot_bgcolor = "#F5F6F9", legend=dict(x=0, y=1.15, orientation="h"))
    iplot(fig)


def make_multipleresponse_barcharts(string_to_search_for, df, specific_title):
    crosstab = pd.DataFrame()
    for col in df:
        if (col.startswith(string_to_search_for)) and \
                (col.endswith('Text')) == False:
            crosstab[col] = np.where(df[col].isnull(), 0, 1)
            crosstab = crosstab.rename(columns = {col: col.replace(string_to_search_for.replace(r"\(.*\)",""), '')})
    crosstab['Higher Education'] = df['Higher Education'] 
    crosstab = crosstab.groupby('Higher Education').sum().T.apply(lambda r: r/r.sum() * 100)

    fig = tls.make_subplots(rows=1, cols=2, shared_yaxes=True, shared_xaxes=True, print_grid=False)

    i = 1
    colorscale=cf.colors.get_scales('accent')
    for col in crosstab:
        crosstab_filter = crosstab.sort_values(col, ascending=True)
        fig.append_trace({'y': crosstab_filter.index, 'x': crosstab_filter[col], 'type': 'bar', 'orientation': 'h', 'name': col, 'text': crosstab_filter[col].round(2).astype(str) + '%', 'textposition': 'auto', 'marker': {'color': colorscale[i-1]}}, 1, i)
        i +=1

    fig['layout'].update(title = "Data Scientists - Distribution of " + specific_title + " by Highest level of Education in %", margin = dict(l = 500, b = 50), yaxis = dict(automargin = True, tickfont = dict(size =10)),
                             paper_bgcolor = "#F5F6F9",  plot_bgcolor = "#F5F6F9", legend=dict(y=1.1, x=0, orientation="h"))
    iplot(fig)
    
responses = pd.read_csv('../input/multipleChoiceResponses.csv', header = 1, low_memory=False)
responses.head()
# filter the data scientists
data_scientists = responses[(responses['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] == 'Data Scientist')]
print("The survey responses consist of" , str(data_scientists.shape[0]) , "data scientists.")
#Split our data scientists into higher education (Masters, PhD) and other education (Bachelor Degree or below)
data_scientists['Higher Education'] = np.where(data_scientists['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].isin(['Master’s degree', 'Doctoral degree']), "Master's or PhD", 'Bachelor or lower')
pd.DataFrame(data_scientists['Higher Education'].value_counts().reset_index()).sort_values(by='index', ascending = True).iplot(kind='pie', labels = 'index', values = 'Higher Education',pull=.2,hole=.2, colorscale='accent', textposition='outside',textinfo='value+percent',title = "Data Scientists - split by Master's or PhD and other education", sort = False)
pd.crosstab(data_scientists['Higher Education'], data_scientists['What is your gender? - Selected Choice']).apply(lambda r: r/r.sum() * 100, axis=1).iplot(kind='bar',yTitle='Percentages', colorscale = 'dflt', title='Data Scientists - Gender Distribution by Highest level of Education in %')

print("P-value for the chi square test of independence is", stats.chi2_contingency(pd.crosstab(data_scientists['Higher Education'], data_scientists['What is your gender? - Selected Choice']))[1], "hence statistically significant.")

pd.crosstab(data_scientists['Higher Education'], data_scientists['What is your age (# years)?']).apply(lambda r: r/r.sum() * 100, axis=1).iplot(kind='bar', yTitle='Percentages', colorscale = 'spectral', title='Data Scientists - Age Distribution by Highest level of Education in %')
print("P-value for the chi square test of independence is", stats.chi2_contingency(pd.crosstab(data_scientists['Higher Education'], data_scientists['What is your age (# years)?']))[1], ",hence statistically significant.")

make_crosstab_barcharts(data_scientists, 'In which country do you currently reside?', 5, "Countries (Top 5)")
make_crosstab_barcharts(data_scientists, 'Which best describes your undergraduate major? - Selected Choice', 5, "Undergrad Degrees (Top 5)")
make_crosstab_barcharts(data_scientists, 'In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice', 5, "Industries (Top 5)")
make_crosstab_barcharts(data_scientists, 'How many years of experience do you have in your current role?', 10, "Years in the role (Top 10)")
make_crosstab_barcharts(data_scientists,'What is your current yearly compensation (approximate $USD)?', 10, "Salary Ranges (Top 10)")
make_crosstab_barcharts(data_scientists,'Does your current employer incorporate machine learning methods into their business?',10, "Machine learning Adoption")
make_multipleresponse_barcharts('Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - ', data_scientists, "Activities")

make_crosstab_barcharts(data_scientists,'What is the primary tool that you use at work or school to analyze data? (include text response) - Selected Choice', 10, "Primary Tool")
make_multipleresponse_barcharts("Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - ",data_scientists, "IDEs")
