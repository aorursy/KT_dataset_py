import numpy as np 

import pandas as pd

import os

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import seaborn as sns

import math

import warnings

from matplotlib.lines import Line2D

from bokeh.layouts import row, column

from bokeh.transform import cumsum, transform

from bokeh.transform import dodge, factor_cmap

from bokeh.plotting import figure, show, gridplot

from bokeh.io import output_notebook

from bokeh.core.properties import value

from bokeh.palettes import d3, brewer, plasma, Plasma256

from bokeh.models import LabelSet, ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker, FactorRange



warnings.filterwarnings('ignore')
output_notebook()
multiplechoice_beg = pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv', low_memory=False, encoding='ISO-8859-1')

multiplechoice_old = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv', low_memory= False)

multiplechoice_new = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv', low_memory= False)



multiplechoice_old.columns = [x.split('_')[0] for x in list(multiplechoice_old.columns)]

multiplechoice_new.columns = [x.split('_')[0] for x in list(multiplechoice_new.columns)]



multiplechoice_old.columns = multiplechoice_old.columns + '_' + multiplechoice_old.iloc[0]

multiplechoice_new.columns = multiplechoice_new.columns + '_' + multiplechoice_new.iloc[0]

multiplechoice_old = multiplechoice_old.drop([0])

multiplechoice_new = multiplechoice_new.drop([0])



multiplechoice_old['Time from Start to Finish (seconds)_Duration (in seconds)'] = multiplechoice_old['Time from Start to Finish (seconds)_Duration (in seconds)'].astype('float')

multiplechoice_new['Time from Start to Finish (seconds)_Duration (in seconds)'] = multiplechoice_new['Time from Start to Finish (seconds)_Duration (in seconds)'].astype('float')



multiplechoice_old['Time from Start to Finish (seconds)_Duration (in seconds)'] = multiplechoice_old['Time from Start to Finish (seconds)_Duration (in seconds)'].apply(lambda x:x/3600)

multiplechoice_new['Time from Start to Finish (seconds)_Duration (in seconds)'] = multiplechoice_new['Time from Start to Finish (seconds)_Duration (in seconds)'].apply(lambda x:x/3600)



time_old = str(round(multiplechoice_old['Time from Start to Finish (seconds)_Duration (in seconds)'].median()*60, 1)) + ' min'

time_new = str(round(multiplechoice_new['Time from Start to Finish (seconds)_Duration (in seconds)'].median()*60, 2)) + ' min'



TOOLS="pan,wheel_zoom,zoom_in,zoom_out,undo,redo,reset,tap,save"
survey_data = pd.DataFrame({'Number of Respondents' : [len(multiplechoice_beg), len(multiplechoice_old), len(multiplechoice_new)],

                            'Number of Questions' :  ['64', '50', '34'],

                            'Median Response Time' : ['16.4 min', time_old, time_new]},

                            index = ['2017', '2018', '2019'])

survey_data
p0 = figure(x_range = survey_data.index.values, y_range = (0,26000), plot_width = 400, plot_height = 500, tools = TOOLS, title = "Number Of Respondents in Survey")

source0 = ColumnDataSource(dict(x=survey_data.index.values, y=survey_data.iloc[:,0].values.reshape(len(survey_data))))

labels0 = LabelSet(x='x', y='y', text='y', level='glyph', x_offset=-22, y_offset=0, source=source0, render_mode='canvas')



p0.vbar(survey_data.index.values, width = 0.9, top = survey_data.iloc[:,0].values.reshape(len(survey_data)), color='mediumseagreen')

p0.xaxis.axis_label = 'Year'

p0.yaxis.axis_label = 'Number of Respondents'

p0.yaxis.axis_label_text_font = 'times'

p0.yaxis.axis_label_text_font_size = '12pt'

p0.xaxis.axis_label_text_font = 'times'

p0.xaxis.axis_label_text_font_size = '12pt'

p0.ygrid.grid_line_color = None

p0.xgrid.grid_line_color = None

p0.add_layout(labels0)



years = ['2017', '2018', '2019']

col_name = ['No of Questions', 'Median Response Time']



plot_data = [(year, analysis_type) for year in years for analysis_type in col_name]

survey_data['Median Response Time'] = survey_data['Median Response Time'].apply(lambda x:x.replace(' min', ''))

counts = list(survey_data.iloc[0,1:].values) + list(survey_data.iloc[1,1:].values) + list(survey_data.iloc[2,1:].values)



source = ColumnDataSource(data=dict(x=plot_data, counts=counts))

source1 = ColumnDataSource(dict(x=survey_data.index.values, y=survey_data.iloc[:,1].values.reshape(len(survey_data))))

source2 = ColumnDataSource(dict(x=survey_data.index.values, y=survey_data.iloc[:,2].values.reshape(len(survey_data))))



labels1 = LabelSet(x='x', y='y', text='y', level='glyph', x_offset=-30, y_offset=0, source=source1, render_mode='canvas')

labels2 = LabelSet(x='x', y='y', text='y', level='glyph', x_offset=5, y_offset=0, source=source2, render_mode='canvas')



p1 = figure(x_range = FactorRange(*plot_data), y_range = (0,70), plot_width = 400, plot_height = 500, tools = TOOLS, title = "Number Of Questions v/s Mean Response time in Survey")

p1.vbar(x='x', top='counts', width=0.9, source=source, fill_color=factor_cmap('x', palette=['mediumslateblue', 'burlywood'], factors=col_name, start=1, end=2))

p1.xaxis.axis_label = 'Year'

p1.yaxis.axis_label_text_font = 'times'

p1.yaxis.axis_label_text_font_size = '12pt'

p1.xaxis.axis_label_text_font = 'times'

p1.xaxis.axis_label_text_font_size = '12pt'

p1.ygrid.grid_line_color = None

p1.xgrid.grid_line_color = None

p1.xaxis.major_label_orientation = math.pi/2

p1.add_layout(labels1)

p1.add_layout(labels2)



show(row(p0,p1))
sns.set_style('darkgrid')

plt.figure(figsize= (16,16))



### Histogram plot

plt.subplot(421)

plt.hist(multiplechoice_old['Time from Start to Finish (seconds)_Duration (in seconds)'], bins = 50, color= 'indianred')

plt.yscale('log')

# plt.xlabel('Duration (in hrs)', fontsize = 'large')

plt.ylabel('Number of Respondents', fontsize = 'large')

plt.title('2018', fontsize = 'x-large', fontweight = 'roman')



### Density plot

plt.subplot(423)

ax = sns.kdeplot(multiplechoice_old['Time from Start to Finish (seconds)_Duration (in seconds)'], color= 'indianred')

ax.legend_.remove()

plt.xlabel('Duration (in hrs)', fontsize = 'large')

plt.ylabel('Density', fontsize = 'large')

# plt.title('2018', fontsize = 'x-large', fontweight = 'roman')



### Histogram plot

plt.subplot(422)

plt.hist(multiplechoice_new['Time from Start to Finish (seconds)_Duration (in seconds)'], bins = 50, color= 'darkslateblue')

plt.yscale('log')

# plt.xlabel('Duration (in hrs)', fontsize = 'large')

plt.ylabel('Number of Respondents', fontsize = 'large')

plt.title('2019', fontsize = 'x-large', fontweight = 'roman')



### Density plot

plt.subplot(424)

ax = sns.kdeplot(multiplechoice_new['Time from Start to Finish (seconds)_Duration (in seconds)'], color= 'darkslateblue')

ax.legend_.remove()

plt.xlabel('Duration (in hrs)', fontsize = 'large')

plt.ylabel('Density', fontsize = 'large')

# plt.title('2019', fontsize = 'x-large', fontweight = 'roman')
gender_beg = multiplechoice_beg['GenderSelect'].value_counts().to_frame()

gender_old = multiplechoice_old['Q1_What is your gender? - Selected Choice'].value_counts().to_frame()

gender_new = multiplechoice_new['Q2_What is your gender? - Selected Choice'].value_counts().to_frame()

gender_beg.index = gender_old.index.values



gender_beg = round(gender_beg/gender_beg.sum(), 2)*100

gender_old = round(gender_old/gender_old.sum(), 2)*100

gender_new = round(gender_new/gender_new.sum(), 2)*100
p0 = figure(x_range = gender_beg.index.values, y_range = (0,90), plot_width = 265, plot_height = 400, tools = TOOLS, )

source0 = ColumnDataSource(dict(x=gender_beg.index.values, y=gender_beg.values.reshape(len(gender_beg))))

labels0 = LabelSet(x='x', y='y', text='y', level='glyph', x_offset=-10, y_offset=0, source=source0, render_mode='canvas')



p0.vbar(gender_beg.index.values, width = 0.5, top = gender_beg.values.reshape(len(gender_beg)), color=d3['Category20b'][len(gender_beg)])

p0.xaxis.axis_label = 'Gender'

p0.yaxis.axis_label = 'Percentage of Respondents -2017'

p0.yaxis.axis_label_text_font = 'times'

p0.yaxis.axis_label_text_font_size = '12pt'

p0.xaxis.axis_label_text_font = 'times'

p0.xaxis.axis_label_text_font_size = '12pt'

p0.ygrid.grid_line_color = None

p0.xgrid.grid_line_color = None

p0.xaxis.major_label_orientation = math.pi/4

p0.add_layout(labels0)



p1 = figure(x_range = gender_old.index.values, y_range = (0,90), plot_width = 265, plot_height = 400, tools = TOOLS, )

source1 = ColumnDataSource(dict(x=gender_old.index.values, y=gender_old.values.reshape(len(gender_old))))

labels1 = LabelSet(x='x', y='y', text='y', level='glyph', x_offset=-10, y_offset=0, source=source1, render_mode='canvas')



p1.vbar(gender_old.index.values, width = 0.5, top = gender_old.values.reshape(len(gender_old)), color=d3['Category20b'][len(gender_old)])

p1.xaxis.axis_label = 'Gender'

p1.yaxis.axis_label = 'Percentage of Respondents -2018'

p1.yaxis.axis_label_text_font = 'times'

p1.yaxis.axis_label_text_font_size = '12pt'

p1.xaxis.axis_label_text_font = 'times'

p1.xaxis.axis_label_text_font_size = '12pt'

p1.ygrid.grid_line_color = None

p1.xgrid.grid_line_color = None

p1.xaxis.major_label_orientation = math.pi/4

p1.add_layout(labels1)



p2 = figure(x_range = gender_new.index.values, y_range = (0,90), plot_width = 265, plot_height = 400, tools = TOOLS, )

source2 = ColumnDataSource(dict(x=gender_new.index.values, y=gender_new.values.reshape(len(gender_new))))

labels2 = LabelSet(x='x', y='y', text='y', level='glyph', x_offset=-10, y_offset=0, source=source2, render_mode='canvas')



p2.vbar(gender_new.index.values, width = 0.5, top = gender_new.values.reshape(len(gender_new)), color=d3['Category20b'][len(gender_new)])

p2.xaxis.axis_label = 'Gender'

p2.yaxis.axis_label = 'Percentage of Respondents -2019'

p2.yaxis.axis_label_text_font = 'times'

p2.yaxis.axis_label_text_font_size = '12pt'

p2.xaxis.axis_label_text_font = 'times'

p2.xaxis.axis_label_text_font_size = '12pt'

p2.ygrid.grid_line_color = None

p2.xgrid.grid_line_color = None

p2.xaxis.major_label_orientation = math.pi/4

p2.add_layout(labels2)



show(row(p0,p1,p2))
country = ['France', 'Canada','UK', 'Germany', 'Brazil', 'Russia', 'China', 'India', 'USA', 'Japan']



multiplechoice_beg['Country'] = multiplechoice_beg['Country'].replace("People 's Republic of China", 'China').replace('United Kingdom', 'UK').replace('United States', 'USA')

multiplechoice_old['Q3_In which country do you currently reside?'] = multiplechoice_old['Q3_In which country do you currently reside?'].replace('United Kingdom of Great Britain and Northern Ireland', 'UK').replace('United States of America', 'USA')

multiplechoice_new['Q3_In which country do you currently reside?'] = multiplechoice_new['Q3_In which country do you currently reside?'].replace('United Kingdom of Great Britain and Northern Ireland', 'UK').replace('United States of America', 'USA')
top_countries_beg = multiplechoice_beg['Country'].value_counts().to_frame().loc[country].sort_values('Country')

top_countries_old = multiplechoice_old['Q3_In which country do you currently reside?'].value_counts().to_frame().loc[country].sort_values('Q3_In which country do you currently reside?')

top_countries_new = multiplechoice_new['Q3_In which country do you currently reside?'].value_counts().to_frame().loc[country].sort_values('Q3_In which country do you currently reside?')



top_countries_beg = round(top_countries_beg/top_countries_beg.sum(), 2)*100

top_countries_old = round(top_countries_old/top_countries_old.sum(), 2)*100

top_countries_new = round(top_countries_new/top_countries_new.sum(), 2)*100



top_countries_old = top_countries_old.reindex(list(top_countries_beg.index))

top_countries_new = top_countries_new.reindex(list(top_countries_beg.index))
country_list = list(top_countries_beg.index)

beg = list(top_countries_beg['Country'])

old = list(top_countries_old['Q3_In which country do you currently reside?'])

new = list(top_countries_new['Q3_In which country do you currently reside?'])



dot = figure(title="Participants by Country", tools=TOOLS, plot_width = 800, plot_height = 400, y_range=country_list, x_range=[0,42])



dot.segment(0, country_list, beg, country_list, line_width=2, line_color="sienna", legend='2017')

dot.circle(beg, country_list, size=15, fill_color="plum", line_color="sienna", line_width=1, legend='2017')

dot.segment(0, country_list, old, country_list, line_width=2, line_color="sienna", legend='2018')

dot.circle(old, country_list, size=15, fill_color="skyblue", line_color="sienna", line_width=1, legend='2018')

dot.segment(0, country_list, new, country_list, line_width=2, line_color="sienna", legend='2019')

dot.circle(new, country_list, size=15, fill_color="yellowgreen", line_color="sienna", line_width=1, legend='2019')



dot.xaxis.axis_label = 'Percentage of Respondents'

dot.yaxis.axis_label = 'Country'

dot.yaxis.axis_label_text_font = 'times'

dot.yaxis.axis_label_text_font_size = '12pt'

dot.xaxis.axis_label_text_font = 'times'

dot.xaxis.axis_label_text_font_size = '12pt'

dot.ygrid.grid_line_color = None

dot.xgrid.grid_line_color = None

dot.legend.location = "bottom_right"

dot.legend.click_policy="hide"

show(dot)
age_beg_dict = {

'18-21' : len(multiplechoice_beg[(multiplechoice_beg['Age'] > 18) & (multiplechoice_beg['Age'] < 21)]['Age']),

'22-24' : len(multiplechoice_beg[(multiplechoice_beg['Age'] > 21) & (multiplechoice_beg['Age'] < 25)]['Age']),

'25-29' : len(multiplechoice_beg[(multiplechoice_beg['Age'] > 24) & (multiplechoice_beg['Age'] < 30)]['Age']),

'30-34' : len(multiplechoice_beg[(multiplechoice_beg['Age'] > 29) & (multiplechoice_beg['Age'] < 35)]['Age']),

'35-39' : len(multiplechoice_beg[(multiplechoice_beg['Age'] > 34) & (multiplechoice_beg['Age'] < 40)]['Age']),

'40-44' : len(multiplechoice_beg[(multiplechoice_beg['Age'] > 39) & (multiplechoice_beg['Age'] < 45)]['Age']),

'45-49' : len(multiplechoice_beg[(multiplechoice_beg['Age'] > 44) & (multiplechoice_beg['Age'] < 50)]['Age']),

'50-54' : len(multiplechoice_beg[(multiplechoice_beg['Age'] > 49) & (multiplechoice_beg['Age'] < 55)]['Age']),

'55-59' : len(multiplechoice_beg[(multiplechoice_beg['Age'] > 54) & (multiplechoice_beg['Age'] < 60)]['Age']),

'60-69' : len(multiplechoice_beg[(multiplechoice_beg['Age'] > 59) & (multiplechoice_beg['Age'] < 70)]['Age']),

'70+' : len(multiplechoice_beg[(multiplechoice_beg['Age'] > 70)])}
ylab_old = multiplechoice_old['Q2_What is your age (# years)?'].sort_values().unique()

ylab_new = multiplechoice_new['Q1_What is your age (# years)?'].sort_values().unique()



age_df_beg = pd.DataFrame(age_beg_dict, index = range(12)).T[0]

age_df_old = multiplechoice_old['Q2_What is your age (# years)?'].value_counts().to_frame().loc[ylab_old]

age_df_new = multiplechoice_new['Q1_What is your age (# years)?'].value_counts().to_frame().loc[ylab_new]



age_df_old_last_row = age_df_old.loc['70-79'] + age_df_old.loc['80+']

age_df_old = age_df_old.drop(['70-79','80+'])

age_df_old = age_df_old.append(pd.DataFrame([age_df_old_last_row], columns=['Q2_What is your age (# years)?'], index=['70+']))



age_df_beg = round(age_df_beg/age_df_beg.sum(), 2)*100

age_df_old = round(age_df_old/age_df_old.sum(), 2)*100

age_df_new = round(age_df_new/age_df_new.sum(), 2)*100
age_list = list(age_df_beg.index)[::-1]

beg = list(age_df_beg)[::-1]

old = list(age_df_old['Q2_What is your age (# years)?'])[::-1]

new = list(age_df_new['Q1_What is your age (# years)?'])[::-1]



dot = figure(title="Age Group of Respondents", tools=TOOLS, plot_width = 800, plot_height = 400, y_range=age_list, x_range=[0,30])



dot.segment(0, age_list, beg, age_list, line_width=2, line_color="sienna", legend='2017')

dot.circle(beg, age_list, size=15, fill_color="plum", line_color="sienna", line_width=1, legend='2017')

dot.segment(0, age_list, old, age_list, line_width=2, line_color="sienna", legend='2018')

dot.circle(old, age_list, size=15, fill_color="skyblue", line_color="sienna", line_width=1, legend='2018')

dot.segment(0, age_list, new, age_list, line_width=2, line_color="sienna", legend='2019')

dot.circle(new, age_list, size=15, fill_color="yellowgreen", line_color="sienna", line_width=1, legend='2019')



dot.xaxis.axis_label = 'Percentage of Respondents'

dot.yaxis.axis_label = 'Age Group'

dot.yaxis.axis_label_text_font = 'times'

dot.yaxis.axis_label_text_font_size = '12pt'

dot.xaxis.axis_label_text_font = 'times'

dot.xaxis.axis_label_text_font_size = '12pt'

dot.ygrid.grid_line_color = None

dot.xgrid.grid_line_color = None

dot.legend.location = "bottom_right"

dot.legend.click_policy="hide"

show(dot)
educ_lvl_beg = multiplechoice_beg['FormalEducation'].value_counts().to_frame()

educ_lvl_old = multiplechoice_old['Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts().to_frame()

educ_lvl_new = multiplechoice_new['Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts().to_frame()

educ_lvl_beg.index = ['Master’s degree', 'Bachelor’s degree', 'Doctoral degree','Some college/university study without earning a bachelor’s degree',

                      'Professional degree', 'No formal education past high school', 'I prefer not to answer']



educ_lvl_beg = educ_lvl_beg.drop('I prefer not to answer')

educ_lvl_old = educ_lvl_old.drop('I prefer not to answer')

educ_lvl_new = educ_lvl_new.drop('I prefer not to answer')



educ_lvl_beg = round(educ_lvl_beg/educ_lvl_beg.sum(), 2)*100

educ_lvl_old = round(educ_lvl_old/educ_lvl_old.sum(), 2)*100

educ_lvl_new = round(educ_lvl_new/educ_lvl_new.sum(), 2)*100



educ_lvl_beg = educ_lvl_beg.reindex(list(educ_lvl_old.index))
educ_list = list(educ_lvl_beg.index)[::-1]

beg = list(educ_lvl_beg['FormalEducation'])[::-1]

old = list(educ_lvl_old['Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'])[::-1]

new = list(educ_lvl_new['Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'])[::-1]



dot = figure(title="Education Level", tools=TOOLS, plot_width = 800, plot_height = 400, y_range=educ_list, x_range=[0,65])



dot.segment(0, educ_list, beg, educ_list, line_width=2, line_color="sienna", legend='2017')

dot.circle(beg, educ_list, size=15, fill_color="plum", line_color="sienna", line_width=1, legend='2017')

dot.segment(0, educ_list, old, educ_list, line_width=2, line_color="sienna", legend='2018')

dot.circle(old, educ_list, size=15, fill_color="skyblue", line_color="sienna", line_width=1, legend='2018')

dot.segment(0, educ_list, new, educ_list, line_width=2, line_color="sienna", legend='2019')

dot.circle(new, educ_list, size=15, fill_color="yellowgreen", line_color="sienna", line_width=1, legend='2019')



dot.xaxis.axis_label = 'Percentage of Respondents'

dot.yaxis.axis_label = 'Education Level'

dot.yaxis.axis_label_text_font = 'times'

dot.yaxis.axis_label_text_font_size = '12pt'

dot.xaxis.axis_label_text_font = 'times'

dot.xaxis.axis_label_text_font_size = '12pt'

dot.ygrid.grid_line_color = None

dot.xgrid.grid_line_color = None

dot.legend.location = "bottom_right"

dot.legend.click_policy="hide"

show(dot)
yearly_comp_old = multiplechoice_old['Q9_What is your current yearly compensation (approximate $USD)?'].value_counts().to_frame()[1:]

yearly_comp_new = multiplechoice_new['Q10_What is your current yearly compensation (approximate $USD)?'].value_counts().to_frame()[1:]



yearly_comp_old = round(yearly_comp_old/yearly_comp_old.sum(), 2)*100

yearly_comp_new = round(yearly_comp_new/yearly_comp_new.sum(), 2)*100
old_idx_sort = ['0-10,000', '10-20,000', '20-30,000', '30-40,000', '40-50,000','50-60,000', '60-70,000', '70-80,000', '80-90,000', '90-100,000',\

'100-125,000', '125-150,000',  '150-200,000', '200-250,000', '250-300,000', '300-400,000', '400-500,000', '500,000+']



new_idx_sort = ['1,000-1,999', '2,000-2,999', '3,000-3,999', '4,000-4,999', '5,000-7,499', '7,500-9,999', '10,000-14,999', '15,000-19,999',

'20,000-24,999', '25,000-29,999', '30,000-39,999', '40,000-49,999','50,000-59,999', '60,000-69,999', '70,000-79,999',

'80,000-89,999', '90,000-99,999', '100,000-124,999', '125,000-149,999', '150,000-199,999', '200,000-249,999', '250,000-299,999',

'300,000-500,000', '> $500,000']



yearly_comp_old = yearly_comp_old.reindex(index = old_idx_sort)

yearly_comp_new = yearly_comp_new.reindex(index = new_idx_sort)



comb_idx = ['0-10,000', '10-20,000', '20-30,000', '30-40,000', '40-50,000','50-60,000', '60-70,000', '70-80,000', '80-90,000', '90-100,000',\

'100-125,000', '125-150,000',  '150-200,000', '200-250,000', '250-300,000', '300-500,000', '500,000+']
yearly_comp_old = pd.DataFrame({'salary': comb_idx,

                                'count': [x[0] for x in yearly_comp_old[:-3].values] + [yearly_comp_old.iloc[-3:-1].values.sum()] +\

                                [yearly_comp_old.iloc[-1][0]]}).set_index('salary')



yearly_comp_new = pd.DataFrame({'salary': comb_idx,

                                'count': [yearly_comp_new.iloc[:6].values.sum(), yearly_comp_new.iloc[6:8].values.sum(),

                                 yearly_comp_new.iloc[8:10].values.sum()] + [x[0] for x in yearly_comp_new[10:].values]}).set_index('salary')
yearly_comp_list = list(yearly_comp_old.index)[::-1]

old = list(yearly_comp_old['count'])[::-1]

new = list(yearly_comp_new['count'])[::-1]



dot = figure(title="Yearly Compensation", tools=TOOLS, plot_width = 800, plot_height = 400, y_range=yearly_comp_list, x_range=[0,31])



dot.segment(0, yearly_comp_list, old, yearly_comp_list, line_width=2, line_color="sienna", legend='2018')

dot.circle(old, yearly_comp_list, size=15, fill_color="skyblue", line_color="sienna", line_width=1, legend='2018')

dot.segment(0, yearly_comp_list, new, yearly_comp_list, line_width=2, line_color="sienna", legend='2019')

dot.circle(new, yearly_comp_list, size=15, fill_color="yellowgreen", line_color="sienna", line_width=1, legend='2019')



dot.xaxis.axis_label = 'Percentage of Respondents'

dot.yaxis.axis_label = 'Yearly Compensation (approx $USD)'

dot.yaxis.axis_label_text_font = 'times'

dot.yaxis.axis_label_text_font_size = '12pt'

dot.xaxis.axis_label_text_font = 'times'

dot.xaxis.axis_label_text_font_size = '12pt'

dot.ygrid.grid_line_color = None

dot.xgrid.grid_line_color = None

dot.legend.location = "bottom_right"

dot.legend.click_policy="hide"

show(dot)
comb_df = multiplechoice_new.groupby(['Q6_What is the size of the company where you are employed?', 'Q7_Approximately how many individuals are responsible for data science workloads at your place of business?']).count().iloc[:,0]

comb_df = comb_df.unstack().reindex(['0-49 employees', '50-249 employees', '250-999 employees', '1000-9,999 employees',

                                     '> 10,000 employees'])

comb_df.columns = ['0', '1-2', '3-4', '5-9', '10-14', '15-19', '20+']

comb_df['1-10'] = comb_df['1-2'] + comb_df['3-4'] + comb_df['5-9']

comb_df['10+'] = comb_df['10-14'] + comb_df['15-19'] + comb_df['20+']

comb_df = comb_df[['0', '1-10', '10+']].reset_index()



patch1 = mpatches.Patch(color='sienna', label='0')

patch2 = mpatches.Patch(color='olive', label='1-10')

patch3 = mpatches.Patch(color='slategrey', label='10+')



fig, ax = plt.subplots(figsize=(15,7))

sns.pointplot(x="Q6_What is the size of the company where you are employed?", y="0", data=comb_df, color= 'sienna')

sns.pointplot(x="Q6_What is the size of the company where you are employed?", y="1-10", data=comb_df, color= 'olive')

sns.pointplot(x="Q6_What is the size of the company where you are employed?", y="10+", data=comb_df, color= 'slategrey')

plt.xticks(rotation=45)

plt.ylabel('Count', fontsize = 'large')

plt.xlabel('Company Size', fontsize = 'large')

plt.legend(title = "Data Science team size", handles=[patch1, patch2, patch3])

plt.title('Company size vs Data Science team size', fontsize = 'large')
source = multiplechoice_new.iloc[:,22:32]

for col in source.columns:

    source[col] = source[col].value_counts()[0]

src_name = [col.split('Choice - ')[1].split(' (')[0] for col in source.columns]

source.columns = src_name

source = source.drop_duplicates().T

source = source.sort_values(by=1)



p1 = figure(x_range = source.index.values, y_range = (0,11500), plot_width = 400, plot_height = 400, title = 'Sources people follow to learn Data Science', tools = TOOLS)

p1.vbar(source.index.values, width = 0.5, top = source.values.reshape(len(source)), color=['peru']*10 + ['goldenrod'])

p1.yaxis.axis_label = 'Number of Respondents'

p1.xaxis.axis_label = 'Source for learning DS'

p1.yaxis.axis_label_text_font = 'times'

p1.yaxis.axis_label_text_font_size = '12pt'

p1.xaxis.axis_label_text_font = 'times'

p1.xaxis.axis_label_text_font_size = '12pt'

p1.xaxis.major_label_orientation = math.pi/4

p1.ygrid.grid_line_color = None



platform =  multiplechoice_new.iloc[:,35:45]

for col in platform.columns:

    platform[col] = platform[col].value_counts()[0]

plt_name = [col.split('Choice - ')[1].split(' (')[0] for col in platform.columns]

platform.columns = plt_name

platform = platform.drop_duplicates().T

platform = platform.sort_values(by=1)



p2 = figure(x_range = platform.index.values, y_range = (0,9500), plot_width = 400, plot_height = 400, title = 'Platforms used for learning Data Science courses', tools = TOOLS)

p2.vbar(platform.index.values, width = 0.5, top = platform.values.reshape(len(platform)), color=['slateblue']*10 + ['goldenrod', 'slategrey'])

p2.yaxis.axis_label = 'Number of Respondents'

p2.xaxis.axis_label = 'Platforms used'

p2.yaxis.axis_label_text_font = 'times'

p2.yaxis.axis_label_text_font_size = '12pt'

p2.xaxis.axis_label_text_font = 'times'

p2.xaxis.axis_label_text_font_size = '12pt'

p2.xaxis.major_label_orientation = math.pi/4

p2.ygrid.grid_line_color = None

show(row(p1,p2))
prog_lang = multiplechoice_new.iloc[:,[55] + list(range(82,92))]

prog_lang.columns = ['Coding exp'] + [x.split('Choice -')[1].split(' (')[0] for x in prog_lang.columns[1:]]

prog_lang = prog_lang.reindex(list(prog_lang['Coding exp'].dropna().index))

prog_lang = prog_lang.groupby('Coding exp').count().iloc[:-1].reindex(['< 1 years', '1-2 years', '3-5 years', '5-10 years', '10-20 years', '20+ years']).reset_index()

prog_lang = pd.melt(prog_lang, id_vars=['Coding exp'])

prog_lang['value'] = prog_lang['value']/90



p = figure(plot_width = 800, plot_height = 650, x_range = prog_lang['Coding exp'].unique(), y_range = prog_lang['variable'].unique(), title="Programming Language v/s Coding Experience", tools = TOOLS)

source = ColumnDataSource(prog_lang)

color_mapper = LinearColorMapper(palette = Plasma256[::-1], low = prog_lang['value'].min(), high = prog_lang['value'].max())

color_bar = ColorBar(color_mapper = color_mapper, location = (0, 0), ticker = BasicTicker())

p.add_layout(color_bar, 'right')

p.scatter(x = 'Coding exp', y = 'variable', size = 'value', legend = None, fill_color = transform('value', color_mapper), source = source)

p.xaxis.axis_label = 'Coding Experience'

p.yaxis.axis_label = 'Programming Language used'

p.yaxis.axis_label_text_font = 'times'

p.yaxis.axis_label_text_font_size = '12pt'

p.xaxis.axis_label_text_font = 'times'

p.xaxis.axis_label_text_font_size = '12pt'

p.xaxis.major_label_orientation = math.pi/4

p.xgrid.grid_line_color = None

show(p)
lang_recom = multiplechoice_new.iloc[:,95].value_counts().to_frame().drop(index=['Other','None'])

lang_recom.columns = ['Language Recommended']



lang_recom.plot(kind='bar', figsize=(16,8), color = 'mediumslateblue', legend=False)

plt.yscale('log')

plt.xlabel('Programming Languages', fontsize = 'large')

plt.ylabel('Number of Respondents', fontsize = 'large')

plt.title('Language Recommended', fontsize = 'x-large', fontweight = 'roman')
ide = multiplechoice_new.iloc[:,55:66]

ide.columns = ['Coding exp'] + [x.split('Choice -')[1].split(' (')[0] for x in ide.columns[1:]]

ide = ide.reindex(list(ide['Coding exp'].dropna().index))

ide = ide.groupby('Coding exp').count().iloc[:-1].reindex(['< 1 years', '1-2 years', '3-5 years', '5-10 years', '10-20 years', '20+ years'])

ide.columns = ['Jupyter', 'RStudio', 'PyCharm', 'Atom', 'MATLAB', 'Visual Studio / VS Code', 'Spyder', 'Vim / Emacs', 'Notepad++', 'Sublime Text']



fig, ax = plt.subplots(figsize=(16,8))

sns.heatmap(ide, annot= True, fmt="d", linewidths=.5, cmap='YlGnBu')

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.xlabel('IDE used', fontsize = 'large')

plt.ylabel('Coding Experience', fontsize = 'large')

plt.title('IDE v/s Coding Experience', fontsize = 'large')
vis_lib = multiplechoice_new.iloc[:,97:107]

vis_lib.columns = [x.split('Choice -  ')[1].split(' (')[0] for x in vis_lib.columns]

for col in vis_lib.columns:

    vis_lib[col] = vis_lib[col].value_counts()[0]

vis_lib = vis_lib.drop_duplicates().T

vis_lib = vis_lib.sort_values(by=1)



color_map = ['cadetblue']*4 + ['plum'] + ['rosybrown'] + ['cadetblue'] + ['rosybrown'] + ['cadetblue']*2

vis_lib[1].plot(kind='bar', color=tuple(color_map), figsize=(15,7))

custom_lines = [Line2D([0], [0], color='cadetblue', lw=4, label='Python'), Line2D([0], [0], color='plum', lw=4, label='Javascript'), Line2D([0], [0], color='rosybrown', lw=4, label='R')]

plt.legend(['Python', 'Javascript', 'R'], handles = custom_lines, title = 'Programming Language', title_fontsize = 'large')

plt.xticks(rotation=45)

plt.xlabel('Visualization Libraries', fontsize = 'large')

plt.ylabel('Count', fontsize = 'large')

plt.title('Visualization Libraries used', fontsize = 'large')

plt.show()
db = multiplechoice_new.iloc[:,233:240]

db.columns = [x.split('Choice -')[1].split(' (')[0] for x in db.columns]

db = pd.melt(db).dropna().groupby('variable')['value'].count().sort_values()[::-1]

db.rename(index={' AWS Relational Database Service':'Amazon RDS'},inplace=True)



fig, ax = plt.subplots(figsize=(15,7))

sns.barplot(x=list(db.index), y=list(db.values), palette="rocket")

plt.axhline(0, color="k", clip_on=False)

plt.xticks(rotation=45)

plt.xlabel('Database', fontsize = 'large')

plt.ylabel('Count', fontsize = 'large')

plt.title('Database Usage', fontsize = 'large')
ml_alg = multiplechoice_new.iloc[:,117:128]

ml_alg.columns = ['ML exp'] + [x.split('Choice -')[1].split(' (')[0] for x in ml_alg.columns[1:]]

ml_alg = ml_alg.reindex(list(ml_alg['ML exp'].dropna().index))

ml_alg = ml_alg.groupby('ML exp').count().iloc[:-1].reindex(['< 1 years', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5-10 years', '10-15 years', '20+ years'])

ml_alg = ml_alg.fillna(0).astype('int').iloc[1:]



fig, ax = plt.subplots(figsize=(16,8))

sns.heatmap(ml_alg, annot= True, fmt="d", linewidths=.5, cmap='YlOrBr')

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.xlabel('ML Algorithms used', fontsize = 'large')

plt.ylabel('ML Experience', fontsize = 'large')

plt.title('ML Experience v/s Algorithms used', fontsize = 'large')
ml_fw = multiplechoice_new.iloc[:,[117] + list(range(155,165))]

ml_fw.columns = ['ML exp'] + [x.split('Choice -')[1].split(' (')[0] for x in ml_fw.columns[1:]]

ml_fw = ml_fw.reindex(list(ml_fw['ML exp'].dropna().index))

ml_fw = ml_fw.groupby('ML exp').count().iloc[:-1].reindex(['< 1 years', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5-10 years', '10-15 years', '20+ years'])

ml_fw = ml_fw.fillna(0).astype('int').iloc[1:]



fig, ax = plt.subplots(figsize=(16,8))

sns.heatmap(ml_fw, annot= True, fmt="d", linewidths=.5, cmap='GnBu')

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.xlabel('ML Frameworks used', fontsize = 'large')

plt.ylabel('ML Experience', fontsize = 'large')

plt.title('ML Experience v/s Frameworks used', fontsize = 'large')
cloud_plat = multiplechoice_new.iloc[:,168:178]

cloud_plat.columns = [x.split('Choice -')[1].split(' (')[0] for x in cloud_plat.columns]

cloud_plat = pd.melt(cloud_plat).dropna().groupby('variable')['value'].count().sort_values()[::-1]



fig, ax = plt.subplots(figsize=(15,7))

sns.barplot(x=list(cloud_plat.index), y=list(cloud_plat.values), palette="rocket")

plt.axhline(0, color="k", clip_on=False)

plt.xticks(rotation=45)

plt.xlabel('Cloud Platforms', fontsize = 'large')

plt.ylabel('Count', fontsize = 'large')

plt.title('Cloud Platform Usage', fontsize = 'large')