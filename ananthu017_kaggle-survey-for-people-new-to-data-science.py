import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
from bokeh.plotting import figure, show, gridplot
from bokeh.io import output_notebook
from bokeh.palettes import d3, brewer
from bokeh.models import LabelSet, ColumnDataSource

warnings.filterwarnings('ignore')
print(os.listdir("../input"))
os.chdir('../input/')
output_notebook()
multiplechoice = pd.read_csv('multipleChoiceResponses.csv', low_memory= False)
multiplechoice.info()
multiplechoice.columns = [x.split('_')[0] for x in list(multiplechoice.columns)]
multiplechoice.head(3)
multiplechoice.columns = multiplechoice.columns + '_' + multiplechoice.iloc[0]
multiplechoice = multiplechoice.drop([0])
multiplechoice.head(3)
multiplechoice['Time from Start to Finish (seconds)_Duration (in seconds)'] = multiplechoice['Time from Start to Finish (seconds)_Duration (in seconds)'].astype('float')
multiplechoice['Time from Start to Finish (seconds)_Duration (in seconds)'] = multiplechoice['Time from Start to Finish (seconds)_Duration (in seconds)'].apply(lambda x:x/3600)
print(str(round(multiplechoice['Time from Start to Finish (seconds)_Duration (in seconds)'].min()*60, 2)) + ' min')
print(str(round(multiplechoice['Time from Start to Finish (seconds)_Duration (in seconds)'].max(), 2)) + ' hrs')
print(str(round(multiplechoice['Time from Start to Finish (seconds)_Duration (in seconds)'].median(), 2)) + ' hrs')
sns.set_style('darkgrid')
plt.figure(figsize= (16,12))

### Histogram plot
plt.subplot(221)
plt.hist(multiplechoice['Time from Start to Finish (seconds)_Duration (in seconds)'], bins = 25)
plt.yscale('log')
plt.xlabel('Duration (in hrs)', fontsize = 'large')
plt.ylabel('Number of Respondents', fontsize = 'large')
plt.title('Histogram', fontsize = 'x-large', fontweight = 'roman')

### Density plot
plt.subplot(222)
ax = sns.kdeplot(multiplechoice['Time from Start to Finish (seconds)_Duration (in seconds)'])
ax.legend_.remove()
plt.xlabel('Duration (in hrs)', fontsize = 'large')
plt.ylabel('Density', fontsize = 'large')
plt.title('KdePlot', fontsize = 'x-large', fontweight = 'roman')
gender = multiplechoice['Q1_What is your gender? - Selected Choice'].value_counts().to_frame()
TOOLS="pan,wheel_zoom,zoom_in,zoom_out,undo,redo,reset,tap,save"
p = figure(x_range = gender.index.values,plot_height = 450, tools = TOOLS, )
source = ColumnDataSource(dict(x=gender.index.values, y=gender.values.reshape(len(gender))))
labels = LabelSet(x='x', y='y', text='y', level='glyph', x_offset=-15, y_offset=0, source=source, render_mode='canvas')

p.vbar(gender.index.values, width = 0.5, top = gender.values.reshape(len(gender)), color=d3['Category20b'][len(gender)])
p.xaxis.axis_label = 'Gender'
p.yaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
p.xgrid.grid_line_color = None
p.add_layout(labels)
show(p)
top_countries = multiplechoice['Q3_In which country do you currently reside?'].value_counts().to_frame().iloc[:10].sort_values('Q3_In which country do you currently reside?')
p = figure(y_range = top_countries.index.values, plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(top_countries.index.values, height = 0.5, right = top_countries.values.reshape(len(top_countries)), color=d3['Category20b'][len(top_countries)])
p.yaxis.axis_label = 'Country'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)
df = multiplechoice.iloc[:,[1,3,4]].set_index('Q3_In which country do you currently reside?', drop = True)
df = df.loc[top_countries.index.values, :]
male, female =[],[]
for val in top_countries.index.values[::-1]:
    male.append(len(df[df['Q1_What is your gender? - Selected Choice'] == 'Male'].loc[val]))
    female.append(len(df[df['Q1_What is your gender? - Selected Choice'] == 'Female'].loc[val]))
p1 = figure(x_range = top_countries.index.values[::-1], plot_width = 400, plot_height = 500, tools = TOOLS, title = 'Male')
p1.vbar(top_countries.index.values[::-1], width = 0.5, top = male, color=d3['Category20b'][len(male)])
p1.xaxis.axis_label = 'Country'
p1.yaxis.axis_label = 'Number of Respondents'
p1.yaxis.axis_label_text_font = 'times'
p1.yaxis.axis_label_text_font_size = '12pt'
p1.xaxis.axis_label_text_font = 'times'
p1.xaxis.axis_label_text_font_size = '12pt'
p1.xaxis.major_label_orientation = math.pi/2
p1.xgrid.grid_line_color = None

p2 = figure(x_range = top_countries.index.values[::-1], plot_width = 400, plot_height = 500, tools = TOOLS, title = 'Female')
p2.vbar(top_countries.index.values[::-1], width = 0.5, top = female, color=d3['Category20b'][len(female)])
p2.xaxis.axis_label = 'Country'
p2.yaxis.axis_label = 'Number of Respondents'
p2.yaxis.axis_label_text_font = 'times'
p2.yaxis.axis_label_text_font_size = '12pt'
p2.xaxis.axis_label_text_font = 'times'
p2.xaxis.axis_label_text_font_size = '12pt'
p2.xaxis.major_label_orientation = math.pi/2
p2.xgrid.grid_line_color = None

p = gridplot([[p1, p2], [None, None]])
show(p)
ylab = multiplechoice['Q2_What is your age (# years)?'].sort_values().unique()
age_df = multiplechoice['Q2_What is your age (# years)?'].value_counts().to_frame().loc[ylab]
p = figure(x_range = age_df.index.values, plot_width = 600, plot_height = 400, tools = TOOLS)
p.vbar(age_df.index.values, width = 0.5, top = age_df.values.reshape(len(age_df)), color=d3['Category20b'][len(age_df)])
p.xaxis.axis_label = 'Age Groups'
p.yaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.xaxis.major_label_orientation = math.pi/4
p.xgrid.grid_line_color = None
show(p)
usa = df.loc['United States of America'].groupby('Q2_What is your age (# years)?').count()
p = figure(x_range = usa.index.values, plot_width = 600, plot_height = 400, tools = TOOLS)
p.vbar(usa.index.values, width = 0.5, top = usa.values.reshape(len(usa)), color=brewer['Paired'][len(usa)])
p.xaxis.axis_label = 'Age Groups'
p.yaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.xaxis.major_label_orientation = math.pi/4
p.xgrid.grid_line_color = None
show(p)
india = df.loc['India'].groupby('Q2_What is your age (# years)?').count()
p = figure(x_range = india.index.values, plot_width = 600, plot_height = 400, tools = TOOLS)
p.vbar(india.index.values, width = 0.5, top = india.values.reshape(len(india)), color=brewer['Paired'][len(india)])
p.xaxis.axis_label = 'Age Groups'
p.yaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.xaxis.major_label_orientation = math.pi/4
p.xgrid.grid_line_color = None
show(p)
china = df.loc['China'].groupby('Q2_What is your age (# years)?').count()
p = figure(x_range = china.index.values, plot_width = 600, plot_height = 400, tools = TOOLS)
p.vbar(china.index.values, width = 0.5, top = china.values.reshape(len(china)), color=brewer['Paired'][len(china)])
p.xaxis.axis_label = 'Age Groups'
p.yaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.xaxis.major_label_orientation = math.pi/4
p.xgrid.grid_line_color = None
show(p)
educ_lvl = multiplechoice['Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts().to_frame()
p = figure(y_range = educ_lvl.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(educ_lvl.index.values[::-1], height = 0.5, right = educ_lvl.values.reshape(len(educ_lvl))[::-1], color=brewer['PuBuGn'][len(educ_lvl)])
p.yaxis.axis_label = 'Highest Education Level'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)
und_grad_major = multiplechoice['Q5_Which best describes your undergraduate major? - Selected Choice'].value_counts().to_frame()
p = figure(y_range = und_grad_major.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(und_grad_major.index.values[::-1], height = 0.5, right = und_grad_major.values.reshape(len(und_grad_major))[::-1], color=d3['Category20'][len(und_grad_major)])
p.yaxis.axis_label = 'Undergraduate Major'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)
comp_sc = multiplechoice[multiplechoice['Q5_Which best describes your undergraduate major? - Selected Choice'] == 'Computer science (software engineering, etc.)'].iloc[:,5].value_counts().to_frame()
p = figure(y_range = comp_sc.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(comp_sc.index.values[::-1], height = 0.5, right = comp_sc.values.reshape(len(comp_sc))[::-1], color=brewer['PuBuGn'][len(comp_sc)])
p.yaxis.axis_label = 'Highest Education Level'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)
engg = multiplechoice[multiplechoice['Q5_Which best describes your undergraduate major? - Selected Choice'] == 'Engineering (non-computer focused)'].iloc[:,5].value_counts().to_frame()
p = figure(y_range = engg.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(engg.index.values[::-1], height = 0.5, right = engg.values.reshape(len(engg))[::-1], color=brewer['PuBuGn'][len(engg)])
p.yaxis.axis_label = 'Highest Education Level'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)
math = multiplechoice[multiplechoice['Q5_Which best describes your undergraduate major? - Selected Choice'] == 'Mathematics or statistics'].iloc[:,5].value_counts().to_frame()
p = figure(y_range = math.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(math.index.values[::-1], height = 0.5, right = math.values.reshape(len(math))[::-1], color=brewer['PuBuGn'][len(math)])
p.yaxis.axis_label = 'Highest Education Level'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)
job_title = multiplechoice['Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].value_counts().to_frame()
p = figure(y_range = job_title.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(job_title.index.values[::-1], height = 0.5, right = job_title.values.reshape(len(job_title))[::-1], color=d3['Category20b'][len(job_title)-1]+['#636363'])
p.yaxis.axis_label = 'Job Title'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)
industry = multiplechoice['Q7_In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice'].value_counts().to_frame()
p = figure(y_range = industry.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(industry.index.values[::-1], height = 0.5, right = industry.values.reshape(len(industry))[::-1], color=d3['Category20b'][len(industry)])
p.yaxis.axis_label = 'Industry Type'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)
work_exp = multiplechoice['Q8_How many years of experience do you have in your current role?'].value_counts().to_frame()
p = figure(y_range = work_exp.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(work_exp.index.values[::-1], height = 0.5, right = work_exp.values.reshape(len(work_exp))[::-1], color=brewer['PuOr'][len(work_exp)])
p.yaxis.axis_label = 'Work Experience'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)
pgm_lang = multiplechoice['Q17_What specific programming language do you use most often? - Selected Choice'].value_counts().to_frame()
p = figure(y_range = pgm_lang.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(pgm_lang.index.values[::-1], height = 0.5, right = pgm_lang.values.reshape(len(pgm_lang))[::-1], color=d3['Category20b'][len(pgm_lang)])
p.yaxis.axis_label = 'Programming Language'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)
recom_lang = multiplechoice['Q18_What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice'].value_counts().to_frame()
p = figure(y_range = recom_lang.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(recom_lang.index.values[::-1], height = 0.5, right = recom_lang.values.reshape(len(recom_lang))[::-1], color=d3['Category20b'][len(recom_lang)])
p.yaxis.axis_label = 'Programming Language'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)
visual_lib = multiplechoice['Q22_Of the choices that you selected in the previous question, which specific data visualization library or tool have you used the most? - Selected Choice'].value_counts().to_frame()
p = figure(y_range = visual_lib.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(visual_lib.index.values[::-1], height = 0.5, right = visual_lib.values.reshape(len(visual_lib))[::-1], color=d3['Category20'][len(visual_lib)])
p.yaxis.axis_label = 'Visualization Libraries'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)
datatypes = multiplechoice['Q32_What is the type of data that you currently interact with most often at work or school? - Selected Choice'].value_counts().to_frame()
p = figure(y_range = datatypes.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(datatypes.index.values[::-1], height = 0.5, right = datatypes.values.reshape(len(datatypes))[::-1], color=d3['Category20b'][len(datatypes)])
p.yaxis.axis_label = 'Common DataTypes People Interact With'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)
online_plat = multiplechoice['Q37_On which online platform have you spent the most amount of time? - Selected Choice'].value_counts().to_frame()
p = figure(y_range = online_plat.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(online_plat.index.values[::-1], height = 0.5, right = online_plat.values.reshape(len(online_plat))[::-1], color=d3['Category20b'][len(online_plat)])
p.yaxis.axis_label = 'Online Platforms Kagglers Use'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)
yearly_comp = multiplechoice['Q9_What is your current yearly compensation (approximate $USD)?'].value_counts().to_frame()[1:]
p = figure(y_range = yearly_comp.index.values[::-1], plot_width = 800, plot_height = 400, tools = TOOLS)
p.hbar(yearly_comp.index.values[::-1], height = 0.5, right = yearly_comp.values.reshape(len(yearly_comp))[::-1], color=d3['Category20b'][len(yearly_comp)])
p.yaxis.axis_label = 'Yearly Compensation (approx $USD)'
p.xaxis.axis_label = 'Number of Respondents'
p.yaxis.axis_label_text_font = 'times'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font = 'times'
p.xaxis.axis_label_text_font_size = '12pt'
p.ygrid.grid_line_color = None
show(p)