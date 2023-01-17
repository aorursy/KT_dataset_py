#Import libraries

import numpy as np # linear algebra

import pandas as pd # data processing

import seaborn as sns # charts

import matplotlib.pyplot as plt  #charts

import matplotlib.patches as patches

import matplotlib as mpl

from matplotlib.gridspec import GridSpec

import math as math

import graphviz as gv

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



#Magic function to display plots below the code cell that produced it.

%matplotlib inline

#Generator for figures numbering

def figcount(start=1):

    num = start

    while True:

        yield str(num)

        num += 1

fig_n = figcount()
sns.set_style('whitegrid')

#color coding the roles

roles = ['Business Analyst','Data Analyst','Data Scientist']

roles_c = {'Business Analyst':'#CAD49D', 'Data Analyst': '#8D89C0', 'Data Scientist':'#1F78B4'}

single_color = '#1F78B4'





google_trends = pd.read_csv("../input/google-trends-data-scientist-comparison/GoogleTrends-DS-DA-BA.csv")

google_trends = google_trends.set_index("Month")

google_trends = google_trends.rename(columns={

    'business analyst: (Worldwide)':'Business Analyst', 'data analyst: (Worldwide)' :'Data Analyst',

       'data scientist: (Worldwide)':'Data Scientist'})



google_trends.loc[google_trends['Data Scientist']=='<1']

google_trends['Data Scientist'] = google_trends['Data Scientist'].replace('<1',0.9)

google_trends = google_trends.astype(float)

google_trends.index = pd.to_datetime(google_trends.index,format='%Y-%m')



colors = list(roles_c.values())

c_palette = sns.color_palette(colors)



fig, ax = plt.subplots(figsize=(12, 5))

ax = sns.lineplot(data=google_trends, palette=c_palette, dashes = False)

ax.set(xticks=pd.date_range(start=google_trends.index.min(), end=google_trends.index.max(), freq='12M'),xlabel = 'Year')

ax.grid(False)

ax.xaxis.set_label_text("")

ax.tick_params(labelsize=12)

annotation_text = ("Figure "+next(fig_n)+": Worldwide interest in selected search terms from 2004 until present"

                  +"\nSource: Google Trends\n100 is the peak popularity for the term. 50 means that the term is half as popular. 0 means not enough data")

sns.despine(left=True, bottom=True)

plt.annotate(annotation_text, (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', ha='left', fontsize=11)

plt.tight_layout()
#Data import and pre-processing:

responses = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv", skiprows = [1], low_memory = False) 

other_text_responses = pd.read_csv("../input/kaggle-survey-2019/other_text_responses.csv", dtype='object')

questions_only = pd.read_csv("../input/kaggle-survey-2019/questions_only.csv", dtype='object').T.reset_index().rename(columns = {'index':'q_num', 0:'q_text'})

survey_schema = pd.read_csv("../input/kaggle-survey-2019/survey_schema.csv", dtype='object')

big_mac_data = pd.read_csv("../input/worldwide-big-mac-prices/big-mac-source-data.csv")

#add indicator whether the question is single choice

questions_only['single_choice'] = questions_only.apply(lambda row: False if 'select all' in str(row['q_text']).lower() else True, axis=1)

questions_only = questions_only.drop([0])



#columns renaming and casting

responses = responses.rename(columns={"Time from Start to Finish (seconds)":"Survey_time"})

responses['Survey_time'] = responses['Survey_time'].astype(int)

responses = responses.rename(columns=lambda x: x.replace('Part_','A') if '_Part' in x else x)



#add metrics whether respondent answered each of the multiple choice questions

for q in questions_only.loc[questions_only['single_choice']==False, 'q_num']:

    #count non empty values in columns containing answers to our question excluding _OTHER_TEXT_COLUMN

    q_regex = '^'+q+'_.*\d$'

    responses.insert(responses.columns.get_loc(q+'_OTHER_TEXT'),q+'_choices_cnt',responses.filter(regex=q_regex).count(axis=1))

#responses.head(2)



#According to survey schema there were 3 possible exit points for different respondents: Q15, Q28 or Q38

#Exit_question - question which was supposed to be the last question of the survey for this individual

#conditions & outputs

conditions = [

    responses['Q15']=='I have never written code',

    (responses['Q15']!='I have never written code')&

        ((responses['Q5']=='Student')|(responses['Q5']=='Not employed')|(responses['Q11']=='$0 (USD)'))

    ]

outputs = ['Q15','Q28']



responses['Exit_question'] = np.select(conditions, outputs, 'Q34')



#Adding info about whether respondent finished the whole survey

responses['Finished'] = (

        ((responses['Exit_question']=='Q34')&(responses['Q34_choices_cnt']!=0))|

        (responses['Exit_question']=='Q15')|

        ((responses['Exit_question']=='Q28')&(responses['Q28_choices_cnt']!=0))

                        )

#Creating separate data frame for those who finished the survey (26% respondents were removed)

responses_f=responses.loc[responses['Finished']==True].copy()



#Calculating quantiles on time it took respondents to finish the survey 

#(separately for those who exit after Q15, Q28 or Q34)

responses_f['rank_by_q'] = responses_f[['Exit_question','Survey_time']].groupby('Exit_question').rank(method='first')

responses_f['size_by_q'] = responses_f.groupby('Exit_question')['Survey_time'].transform('size')

responses_f['quantile'] = responses_f['rank_by_q'] / responses_f['size_by_q']

#Creating a separate data frame, where 1% were removed 

responses_ff = responses_f.loc[(responses_f['quantile']>=0.01)].copy()
#data preparation

#calculate how many respondents finished survey by job title

chart1_d = responses.groupby(['Q5','Finished']).size()

#express this number as percentage of the group

chart1_d = chart1_d.div(chart1_d.sum(level='Q5'), axis=0)*100

chart1_d = chart1_d.unstack('Finished').rename(columns={False:'Did not finish',True:'Finished'}).reset_index().sort_values('Finished')



#plotting

fig, ax = plt.subplots(figsize=(12, 5))

#plt.barh(y=chart1_d['Q5'].str.upper(), width=chart1_d["Finished"], height=0.8, color=single_color)

plt.hlines(y=chart1_d['Q5'].str.upper(), xmin=0, xmax=chart1_d["Finished"], color=single_color, alpha=0.7, linewidth=2)

plt.scatter(y=chart1_d['Q5'].str.upper(), x=chart1_d["Finished"], s=75, color=single_color, alpha=0.7)

#plot decoration

ax.set(xlim=(0, 100))

ax.xaxis.set_ticks(np.arange(0, 101, 10))

ax.yaxis.grid(False)

ax.set_title(("Only people who are not employed were more dedicated to\nfinishing the survey than data scientists"), fontsize=14)

ax.xaxis.set_label_text("% respondents", fontsize=14)

ax.tick_params(labelsize=10)

annotation_text = "Figure "+next(fig_n)+": Percentage of respondents who finished the survey by job title"

sns.despine(left=True, bottom=True)

plt.annotate(annotation_text, (0,0), (0, -50), xycoords='axes fraction', textcoords='offset points', va='top', ha='left', fontsize=11)

plt.tight_layout()

plt.show()
#data preparation - calculate for each repported job role total number of respondents and those who were excluded from analysis

chart2_t = responses.loc[responses['Q5'].isin(roles)].groupby('Q5').size()

chart2_d = responses_ff.loc[responses_ff['Q5'].isin(roles)].groupby('Q5').size()

chart2_dd = pd.concat([chart2_d, chart2_t], axis=1).reset_index().rename(columns={0:'Not excluded',1:'All'})

chart2_dd['Excluded'] = chart2_dd['All']-chart2_dd['Not excluded']

chart2_d = chart2_dd[['Q5','Not excluded','Excluded']]



#plotting

fig, ax = plt.subplots(figsize=(12, 5))

for r in roles:

    #choose color

    bc = roles_c[r] if r in roles else '#bbbbbb'

    #plot stacked bars:

    plt.bar(x=r, height=chart2_d.loc[chart2_d['Q5']==r]['Not excluded'], color=bc, alpha=0.8)

    plt.bar(x=r, height=chart2_d.loc[chart2_d['Q5']==r]['Excluded'], bottom=chart2_d.loc[chart2_d['Q5']==r]['Not excluded'], color='#eeeeee')

    total_height = (chart2_d.loc[chart2_d['Q5']==r]['Not excluded']+chart2_d.loc[chart2_d['Q5']==r]['Excluded']).iloc[0]

    included_height = (chart2_d.loc[chart2_d['Q5']==r]['Not excluded']).iloc[0]

    ax.text(r, total_height+5, str(total_height)+"\n(total)", ha='center', va='bottom', fontsize=12)

    ax.text(r, included_height/2, str(included_height)+"\n(analyzed)", ha='center', va='center', fontsize=12)

#plot decoration

ax.set_title(("Almost twice as many data scientists participated in the survey\ncompared to business analysts and data analysts combined"), fontsize=14)

ax.yaxis.set_label_text("# respondents", fontsize=12)

ax.xaxis.set_label_text("")

ax.set_xticklabels(chart2_d['Q5'], rotation = 0, fontsize=12)

ax.set(ylim=(0, 4600))

ax.tick_params(labelsize=12)

ax.xaxis.grid(False)

sns.despine(left=True, bottom=True)

#create a special legend, for "excluded" 

handles = [plt.bar(x=0, height=0, color='#eeeeee')]

plt.legend(handles, ['Excluded*'], loc = 'upper left', bbox_to_anchor=(0, 1), ncol=1, labelspacing=0.5, fontsize=12)



annotation_text = ("Figure "+next(fig_n)+": Number of respondents in selected groups"+

                    "\nSource: ML&DL survey 2019, question 5"+

                    "\n*those who dropped out in the middle (~23%) or were too quick (~1%)")

plt.annotate(annotation_text, (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', ha='left', fontsize=11)



    



plt.tight_layout()

plt.show()
#age distribution by roles

age_raw = responses_ff.loc[responses_ff['Q5'].isin(roles),['Q1','Q5']]                                   

age_dist = age_raw.groupby(['Q5','Q1']).size()

age_dist = age_dist.div(age_dist.sum(level='Q5')).reset_index().rename(columns={0:'Percentage'})

age_dist = age_dist.pivot(index='Q1', columns='Q5', values='Percentage').reset_index()



fig, ax = plt.subplots(figsize=(12,5))



#line plot for age

for role_name, role_color in roles_c.items():

    ax.plot(age_dist['Q1'].str.upper(), age_dist[role_name]*100, label=age_dist[role_name].name,

         color = role_color, linestyle = '--', marker = 'o', markersize=12, linewidth=1.5, alpha = 0.9)



#plots decoration

ax.set_xticklabels(age_dist['Q1'].str.lower(), rotation=0, fontsize=12)

ax.set_title(("Age distributions within the groups are similar, with a slightly\nhigher share of older respondents among business analysts"), fontsize=14)

ax.set_ylim(-1,35)

ax.tick_params(labelsize=12)

ax.yaxis.set_label_text("% respondents", fontsize=12)

annotation_text = ("Figure "+next(fig_n)+": Age distribution of selected groups"+

                    "\nSource: ML&DL survey 2019, question 1")

ax.annotate(annotation_text,(0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', ha='left',fontsize=11)

ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=1, labelspacing=0.5, fontsize=11)



sns.despine(left=True, bottom=True)

plt.tight_layout()

plt.show()
#education by roles mapping to make labels shorter

ed_levels_map = {

    'No formal education past high school' : 'No\ndegree',

    'Some college/university study without earning a bachelor’s degree' : 'No\ndegree', 

    'Bachelor’s degree' : 'Bachelor’s\ndegree',  

    'Professional degree' : 'Professional\ndegree',

    'Master’s degree': 'Master’s\ndegree', 

    'Doctoral degree': 'Doctoral degree\n(PhD)',

    'I prefer not to answer':'I prefer not to answer'}

ed_levels = ['No\ndegree','Bachelor’s\ndegree','Professional\ndegree', 'Master’s\ndegree','Doctoral degree\n(PhD)','I prefer not to answer']

ed_raw = responses_ff.loc[responses_ff['Q5'].isin(roles)][['Q4','Q5']]

#map original levels to the new ones desribe in the dictionary ed_levels_map

ed_raw['Education'] = ed_raw['Q4'].map(ed_levels_map) 

#calculate data for plot

ed_dist = ed_raw.groupby(['Q5','Education']).size()

ed_dist = ed_dist.div(ed_dist.sum(level='Q5')).reset_index().rename(columns={0:'Percentage'})

ed_dist = ed_dist.pivot(index='Education', columns='Q5', values='Percentage').reset_index()

ed_dist = ed_dist.loc[(ed_dist['Education']!='I prefer not to answer')] #do not plot because this group is small,  &(ed_dist['Education']!='No degree')

ed_dist['Education'] = pd.Categorical(ed_dist['Education'],ed_levels)

ed_dist = ed_dist.sort_values('Education')

ed_dist = ed_dist.set_index('Education').multiply(100)



fig, ax = plt.subplots(figsize=(12,6))



#education bar chart

ed_dist.plot(kind='bar',  ax=ax, color=list(roles_c.values()))    



#plot decoration

ax.set_xticklabels(ed_dist.index, rotation=0, fontsize=12)

ax.set_title(("A data scientist is 3 times more likely to have a doctoral degree\nthan a business analyst or data analyst"), fontsize=14)

ax.tick_params(labelsize=12)

ax.xaxis.set_label_text("")

ax.set_ylim(0,63)

ax.yaxis.set_label_text("% respondents", fontsize=12)

annotation_text = ("Figure "+next(fig_n)+": Education level of selected groups"+

                   "\nSource: ML&DL survey 2019, question 4")

ax.annotate(annotation_text,(0,0), (0, -50), xycoords='axes fraction', textcoords='offset points', va='top', ha='left',fontsize=11)

ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), ncol=1, labelspacing=0.5, fontsize=12)



sns.despine(left=True, bottom=True)

plt.tight_layout()

plt.show()
ml_states_map = {

    'I do not know' : 'Unaware of ML state in the company', 

    'No (we do not use ML methods)' : "Company don't use ML",

    'We are exploring ML methods (and may one day put a model into production)': 'Company explore/use ML',

    'We use ML methods for generating insights (but do not put working models into production)' : 'Company explore/use ML',

    'We recently started using ML methods (i.e., models in production for less than 2 years)' : 'Company explore/use ML',

    'We have well established ML methods (i.e., models in production for more than 2 years)': 'Company explore/use ML'}



ml_states_map_detailed = {

    'We are exploring ML methods (and may one day put a model into production)': 'Exploring ML',

    'We use ML methods for generating insights (but do not put working models into production)' : 'Generate insights with ML',

    'We recently started using ML methods (i.e., models in production for less than 2 years)' : 'Models in production less than 2 years',

    'We have well established ML methods (i.e., models in production for more than 2 years)': 'Models in prodution more than 2 years'}



q9_answers_map = {

    'Q9_A1': 'Data analysis',

    'Q9_A2': 'Build or run data infrastructure',

    'Q9_A3': 'Build ML\nprototypes',

    'Q9_A4': 'Build or run\nML services',

    'Q9_A5': 'Improve\nML models',

    'Q9_A6': 'Research to\nadvance ML',

    'Q9_A7': 'None',

    'Q9_A8': 'Other'}



#data for chart a and b

company_ML_state = responses_ff.loc[responses['Q5'].isin(roles),['Q5','Q8']]

company_ML_state['Q8'] = company_ML_state['Q8'].map(ml_states_map)

company_ML_state = company_ML_state.groupby(['Q5','Q8']).size().unstack('Q8')

company_ML_state['Unaware']=company_ML_state['Unaware of ML state in the company']*100/(company_ML_state.sum(axis=1))

company_ML_state['Informed'] = 100 - company_ML_state['Unaware']

company_ML_state['no ML']=company_ML_state["Company don't use ML"]*100/(company_ML_state[["Company don't use ML","Company explore/use ML"]].sum(axis=1))

company_ML_state['ML is used']=company_ML_state["Company explore/use ML"]*100/(company_ML_state[["Company don't use ML","Company explore/use ML"]].sum(axis=1))

company_ML_state.index = company_ML_state.index.str.wrap(9)



fig, ax = plt.subplots(1,2, figsize=(12,5))



yaxis_labels = {

    0:"% respondents informed on ML state",

    1:"% respondents"}



titles = {

    0:("Data scientists are best informed on\nwhether ML methods are used\nin the company"),

    1:("94% of data scientists work at\ncompanies which either use or explore\nML methods")}



axis_colors = {

    0:[single_color],

    1:['#8A716A','#B0BAB8']}



annotations = {

    0:"Figure "+next(fig_n)+": ML usage in companies of employment"

    +"\nSource: ML&DS survey 2019, question 8.",

    1:""}



 

#plot a - anaware

company_ML_state['Informed'].plot(kind='bar', ax=ax[0], label='Informed on whether company uses ML',color=axis_colors[0], alpha=0.7, width=0.8)

for i, val in enumerate(company_ML_state['Informed'].values):

    ax[0].text(i, val/2, '{:.0f}%'.format(val), ha='center', va='bottom', fontsize=12)



#plot b - company uses/does not use ML

company_ML_state[['ML is used','no ML']].plot(kind='bar', stacked='True', ax=ax[1], color = axis_colors[1], alpha=0.8,  width=0.8)

for i, val in enumerate(company_ML_state['ML is used'].values):

    ax[1].text(i, val/2, '{:.0f}%'.format(val), ha='center', va='bottom', fontsize=12)



#plots decoration

for i in [0,1]:

    c_ax = ax[i]

    

    c_ax.xaxis.grid(False)

    c_ax.set_xticklabels(c_ax.get_xticklabels(), rotation=0, fontsize=12)

    c_ax.xaxis.set_label_text("")

    c_ax.yaxis.set_label_text(yaxis_labels[i], fontsize=12)

    c_ax.tick_params(labelsize=12)

    c_ax.set_title(titles[i], fontsize=14)

    c_ax.annotate(annotations[i],(0,0), (0, -80), xycoords='axes fraction', textcoords='offset points', 

                  va='top', ha='left',fontsize=11)



ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, labelspacing=0.5, fontsize=12)    

sns.despine(left=True, bottom=True)

plt.show()
d = responses_ff.loc[(responses_ff['Q5'].isin(roles))&responses['Q8'].isin(list(ml_states_map_detailed.keys())),['Q5','Q8']+list(q9_answers_map.keys())]

d['Q8'] = d['Q8'].map(ml_states_map_detailed)

d['Q8'] = pd.Categorical(d['Q8'],list(ml_states_map_detailed.values()))

d['ML is part of the role'] = (d['Q9_A3'].notna())| (d['Q9_A4'].notna())|(d['Q9_A5'].notna())|(d['Q9_A6'].notna())

activities_raw = d



ml_agg = activities_raw.groupby(['Q5','ML is part of the role']).size().unstack('ML is part of the role')

ml_agg = ml_agg.rename(columns={True:'Involved in ML',False:'Not involved in ML'})

ml_agg = ml_agg.div(ml_agg.sum(axis=1),axis=0)

ml = ml_agg['Involved in ML']*100



#NON ML regarless company

nonml = activities_raw

nonml_bool = nonml[['Q9_A1','Q9_A2','Q9_A7','Q9_A8']].notna().multiply(1).add_suffix('_bool')

#decode activities combination with a sequence of ones and zeros

nonml_bool['nonml_set']=(nonml_bool['Q9_A1_bool']*100+

                      nonml_bool['Q9_A2_bool']*10)#+

                      #((nonml_bool['Q9_A7_bool'])|(nonml_bool['Q9_A8_bool']))*1)

                   

nonml_data = pd.concat([nonml[['Q5','Q8']], nonml_bool['nonml_set']],axis=1)

nonml_data_grouped = nonml_data.groupby(['Q5','nonml_set']).size().unstack('Q5')

d = nonml_data_grouped.sort_values(by=['Data Scientist'],ascending=False)

d = d.div(d.sum(axis=0),axis=1)*100

a = d.loc[[100,110],:].sum(axis=0) #analysis

i = d.loc[[10,110],:].sum(axis=0) #infrastructure





nonml_data_to_plot = pd.DataFrame({ 'Data Analysis': a, 

                                  'Data Infrastructure\n(run or build)': i,

                                   'ML related activities': ml

                                  })





q9_answers_ml_map = {

    'Q9_A3' : 'Build ML\nprototypes',

    'Q9_A4': 'Build or run\nML services',

    'Q9_A5': 'Improve\nML models',

    'Q9_A6': 'Research to\nadvance ML',

}





#select those for whom ML is part of the role

ml = activities_raw.loc[activities_raw['ML is part of the role']]

ml_bool = ml[['Q9_A3','Q9_A4','Q9_A5','Q9_A6']].notna().multiply(1).add_suffix('_bool')

#decode activities combination with a sequence of ones and zeros

ml_bool['ml_set']=(ml_bool['Q9_A3_bool']*1000+

                   ml_bool['Q9_A4_bool']*100+

                   ml_bool['Q9_A5_bool']*10+

                   ml_bool['Q9_A6_bool'])

ml_data_raw = pd.concat([ml[['Q5','Q8']], ml_bool['ml_set']],axis=1)

ml_data_grouped = ml_data_raw.groupby(['Q5','ml_set']).size().unstack('Q5')





dd = ml_data_grouped.sort_values(by=['Data Scientist'],ascending=False)

dd = dd.div(dd.sum(axis=0),axis=1)*100



p = dd.loc[dd.index//1000%2 == 1].sum(axis=0) #prototyping

s = dd.loc[dd.index//100%2 == 1].sum(axis=0) #services

m = dd.loc[dd.index//10%2 == 1].sum(axis=0) #models improvement

r = dd.loc[dd.index//1%2 == 1].sum(axis=0) #research



ml_data_to_plot = pd.DataFrame({'Build ML\nprototypes': p, 

                                'Build or run\nML services': s,

                                'Improve\nML models': m,

                                'Research to\nadvance ML': r})





  

plot_colors = [roles_c[x] for x in ['Business Analyst','Data Analyst','Data Scientist']]



fig, ax = plt.subplots(2,1, figsize=(12, 12))

#plots

ax[0].axvspan(1.5, 2.5, color='grey', alpha = 0.2)

nonml_data_to_plot.T.plot(kind='bar', color = plot_colors, ax=ax[0])

ml_data_to_plot.T.plot(kind='bar', color = plot_colors, ax=ax[1])





yaxis_labels = {

    0:"% respondents",

    1:"% respondents"}



titles = {

    0:("Data Analysis is an important part of one's job regardless of their title,"+

    "\nwhereas ML tasks are more often performed by data scientists"),

    1:("Building prototypes was the top chosen activity among those"+

    "\nwho are involved in ML")}



yaxis_limits = {

    0:100,

    1:100}

fignum = next(fig_n)

annotations = {

    0:"Figure "+fignum+"a: Activities that make up an important part of one's role at work"+

    "\nSource: ML&DS survey 2019, question 9.",

    1:"Figure "+fignum+"b: ML related activities for those involved* in the ML"+

    "\nSource: ML&DS survey 2019, question 9."+

    "\n*respondent selected at least one option that mentioned ML"}



#plots decoration

for i in [0,1]:

    ax[i].xaxis.grid(False)

    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=0, fontsize=12)

    ax[i].yaxis.set_label_text(yaxis_labels[i], fontsize=12)

    ax[i].set_title(titles[i], fontsize=14)

    ax[i].set_ylim(0,yaxis_limits[i])

    ax[i].annotate(annotations[i],

                  (0,0), (0, -50), xycoords='axes fraction', textcoords='offset points', 

                  va='top', ha='left',fontsize=11)

    ax[i].tick_params(labelsize=12)



ax[0].legend().set_visible(False)

ax[1].legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=1, labelspacing=0.5, fontsize=12)



#ax[1].set_facecolor('#0E3A70')



ax[1].patch.set_facecolor('grey')

ax[1].patch.set_alpha(0.2)



sns.despine(left=True, bottom=True)

plt.tight_layout()

plt.show()
tools_map = {

    'Local development environments (RStudio, JupyterLab, etc.)':'Local IDE',

    'Basic statistical software (Microsoft Excel, Google Sheets, etc.)':'Spreadsheets',

    'Cloud-based data software & APIs (AWS, GCP, Azure, etc.)':'Cloud APIs',

    'Business intelligence software (Salesforce, Tableau, Spotfire, etc.)': 'BI software',

    'Advanced statistical software (SPSS, SAS, etc.)': 'Advanced\nstat. software',

    'Other':'Other'

}

tools_raw = responses_ff.loc[responses_ff['Q5'].isin(roles),['Q14','Q5']].groupby(['Q14','Q5']).size().unstack('Q5')

tools_agg = tools_raw.div(tools_raw.sum(axis=0),axis=1)

tools_agg.index = tools_agg.index.map(tools_map)

tools_agg.index = pd.Categorical(tools_agg.index, list(tools_map.values()), ordered=True)

tools_agg = tools_agg.sort_index() #data for bar chart

tools_agg.columns.name=None

ce_buckets_map = {

    'I have never written code':'0', 

    '< 1 years':'< 1 year',

    '1-2 years':'1-2',

    '3-5 years':'3-5',

    '5-10 years':'5-10',

    '10-20 years':'10-20',

    '20+ years':'20+ years'}



tools_ce_raw = responses_ff.loc[responses_ff['Q5'].isin(roles),['Q5','Q14','Q15']].copy()

#tools_ce_raw['Q5'] = tools_ce_raw['Q5'].str.wrap(10)

tools_ce_raw = tools_ce_raw.groupby(['Q5','Q14','Q15']).size().unstack('Q15')

tools_ce_raw = tools_ce_raw.rename(columns=ce_buckets_map)

tools_ce_raw = tools_ce_raw[list(ce_buckets_map.values())].unstack('Q14').stack('Q15').fillna(0)

tools_ce_agg = tools_ce_raw.div(tools_ce_raw.sum(axis=1),axis=0)

tools_ce_agg = tools_ce_agg.rename(columns=tools_map) #data for heatmaps





fig, ax = plt.subplots(figsize=(12,5))

#highlight the area of interest 

ax.axvspan(-0.5, 0.5, color='#586BA4', alpha = 0.2)

ax.axvspan(0.5, 1.5, color='#CAE7B9', alpha = 0.2)

#plot

(tools_agg*100).plot.bar(color = list(roles_c.values()) , ax=ax)

#plot decoration

ax.yaxis.set_label_text("% of respondents", fontsize=12)

ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)

ax.set_title(("The majority of data scientists choose IDEs as their primary tool,\nwhereas 38% of business analysts choose Spreadsheets"), fontsize=14)

annotation_text=("Figure "+next(fig_n)+": Primary tool to analyze data\nSource: ML&DS survey 2019, question 14")

ax.annotate(annotation_text,(0,0), (0, -50), xycoords='axes fraction', textcoords='offset points', va='top', ha='left',fontsize=11)

ax.tick_params(labelsize=12)

ax.set_ylim(0,75)

ax.xaxis.grid(False)

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels)

sns.despine(left=True, bottom=True)

plt.tight_layout()



plt.show()

#how long have you been written code to analyze data?

q15_raw = responses_ff.loc[(responses_ff['Q5'].isin(roles)),['Q15','Q5']]                                   

q15_dist = q15_raw.groupby(['Q5','Q15']).size()

q15_dist = q15_dist.div(q15_dist.sum(level='Q5')).reset_index().rename(columns={0:'Percentage'})

q15_dist = q15_dist.pivot(index='Q15', columns='Q5', values='Percentage').reset_index()

q15_answers_map = {

    'I have never written code':'0', 

    '< 1 years':'< 1 year',

    '1-2 years':'1-2',

    '3-5 years':'3-5',

    '5-10 years':'5-10',

    '10-20 years':'10-20',

    '20+ years':'20+ years'

}

q15_labels_ordered = list(q15_answers_map.values())

q15_dist['Q15'] = q15_dist['Q15'].map(q15_answers_map)

q15_dist['Q15'] = pd.Categorical(q15_dist['Q15'],q15_labels_ordered)

q15_dist = q15_dist.sort_values(by=['Q15'])





fig, ax = plt.subplots(figsize=(12,5))





#line plot for age

for role_name, role_color in roles_c.items():

    ax.plot(q15_dist['Q15'].astype('str'), q15_dist[role_name]*100, 

               color = role_color, linestyle = '--', marker = 'o', markersize=12, linewidth=1.5, alpha = 0.9)





ax.yaxis.set_label_text('% of respondents', fontsize=12)

ax.set_title(("Data Scientists have more years of experience in writing code to analyze data"), fontsize=14)

ax.set_xticklabels(q15_dist['Q15'].astype('str'), rotation=0, fontsize=12)

annotation_text = "Figure "+next(fig_n)+": Distribution by years of experience in writing code to analyze data\nSource: ML&DL Survey 2019, question 15"

ax.annotate(annotation_text,

                  (0,0), (0, -50), xycoords='axes fraction', textcoords='offset points', 

                  va='top', ha='left',size=11)

ax.tick_params(labelsize=12)

ax.set_ylim(-1,55)

ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=1, labelspacing=0.5, fontsize=11)



sns.despine(left=True, bottom=True)

plt.tight_layout()

plt.show()
tools_colors ={

    'Spreadsheets':'Greens',

    'Local IDE':'Purples'}

fignum = next(fig_n)

anotations = {

        0: "Figure "+fignum+"a: Spreadsheet usage within groups by job title and years of writing code for data analysis",

        1: "Figure "+fignum+"b: Local IDE usage within groups by job title and years of writing code for data analysis"+

            "\n\nSource: ML&DS Survey 2019, questions 14&15"+

            "\n*an empty square on the heatmap means that there were fewer than 30 respondents in this group"}



titles = {

    0:("Although business analysts turn away from Spreadsheets\nas they gain experience in writing code,"+

       " data scientists do it faster"),

    1:("The majority of data scientists use local IDEs as their primary tool for data analysis"+

        "\nas soon as they learn how to write code to analyze data")}



fig, ax = plt.subplots(2,1,figsize=(9,12) )





for i, tool in enumerate(tools_colors):

    

    #heatmaps plotting

    heatmap_data = tools_ce_agg[tool].unstack('Q15')[list(ce_buckets_map.values())]

    heatmap_data[tools_ce_raw.sum(axis=1).unstack() < 30] = np.nan #mask values where were less than 30 respondents

    

    sns.heatmap(heatmap_data, vmin = 0, vmax = 1, 

                annot=True, annot_kws={"size": 12}, fmt='.0%', square=True,

            cmap=tools_colors[tool], linewidths=.5, ax=ax[i], cbar=False )

    #heatmap decoration

    ax[i].set_yticklabels(ax[i].get_yticklabels(), rotation=0,ha='right',va='center')

    ax[i].set(xlabel="", ylabel="")

    ax[i].tick_params(labelsize=12)

    ax[i].set_title(titles[i], fontsize=13)

    ax[i].annotate(anotations[i],

                  (0,0), (0, -50), xycoords='axes fraction', textcoords='offset points', va='top', ha='left',size=11)

    

sns.despine(left=True, bottom=True) 



plt.show()

# years ML experience

q23_raw = responses_ff.loc[(responses_ff['Q5'].isin(roles))&(responses_ff['Q15']!='I have never written code'),['Q23','Q5']]                                   

q23_dist = q23_raw.groupby(['Q5','Q23']).size()

q23_dist = q23_dist.div(q23_dist.sum(level='Q5')).reset_index().rename(columns={0:'Percentage'})

q23_dist = q23_dist.pivot(index='Q23', columns='Q5', values='Percentage').reset_index()

q23_answers_map = {

    '< 1 years':'< 1 year',

    '1-2 years':'1-2',

    '2-3 years':'2-3',

    '3-4 years':'3-4',

    '4-5 years':'4-5',

    '5-10 years':'5-10',

    '10-15 years':'10-15',

    '20+ years':'20+ years'

}

q23_labels_ordered = list(q23_answers_map.values())

q23_dist['Q23'] = q23_dist['Q23'].map(q23_answers_map)

q23_dist['Q23'] = pd.Categorical(q23_dist['Q23'],q23_labels_ordered)

q23_dist = q23_dist.sort_values(by=['Q23'])

q23_dist_cumsum = q23_dist.set_index('Q23').cumsum(axis=0).reset_index()





d2 = responses_ff.loc[(responses_ff['Q5'].isin(roles))&responses['Q8'].isin(list(ml_states_map_detailed.keys())),

                         ['Q5','Q8','Q23']+list(q9_answers_map.keys())]

d2['Q8'] = d2['Q8'].map(ml_states_map_detailed)

d2['Q8'] = pd.Categorical(d2['Q8'],list(ml_states_map_detailed.values()))



q23_labels_ordered = list(q23_answers_map.values())

d2['Q23'] = d2['Q23'].map(q23_answers_map)

d2['Q23'] = pd.Categorical(d2['Q23'],q23_labels_ordered)

d2['ML is part of the role'] = (d2['Q9_A3'].notna())| (d2['Q9_A4'].notna())|(d2['Q9_A5'].notna())|(d2['Q9_A6'].notna())

activities_raw = d2



fig, ax = plt.subplots(figsize=(12,5))





#line plot for age

for role_name, role_color in roles_c.items():

    ax.plot(q23_dist['Q23'].astype('str'), q23_dist[role_name]*100, 

               color = role_color, linestyle = '--', marker = 'o', markersize=12, linewidth=1.5, alpha = 0.9)





ax.yaxis.set_label_text('% of respondents', fontsize=12)

ax.set_title(("Overall, data scientists have more years of ML experience"), fontsize=14)

ax.set_xticklabels(q23_dist['Q23'].astype('str'), rotation=0, fontsize=12)

annotation_text = "Figure "+next(fig_n)+": Distribution by years of ML experience\nSource: ML&DL Survey 2019, question 23"

ax.annotate(annotation_text,

                  (0,0), (0, -50), xycoords='axes fraction', textcoords='offset points', 

                  va='top', ha='left',size=11)

ax.tick_params(labelsize=12)

ax.set_ylim(-1,55)

ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=1, labelspacing=0.5, fontsize=11)



sns.despine(left=True, bottom=True)

plt.tight_layout()

plt.show()
ml = activities_raw.loc[activities_raw['ML is part of the role']]

ml_bool = ml[['Q9_A3','Q9_A4','Q9_A5','Q9_A6']].notna().multiply(1).add_suffix('_bool')

#decode activities combination with a sequence of ones and zeros

ml_bool['ml_set']=(ml_bool['Q9_A3_bool']*1000+

                   ml_bool['Q9_A4_bool']*100+

                   ml_bool['Q9_A5_bool']*10+

                   ml_bool['Q9_A6_bool'])

ml_data_raw = pd.concat([ml[['Q23','Q8']], ml_bool['ml_set']],axis=1)

ml_data_g = ml_data_raw.groupby(['Q23','ml_set']).size().unstack('Q23')

dd = ml_data_g.sort_values(by=['20+ years'],ascending=False)



p = dd.loc[dd.index//1000%2 == 1].sum(axis=0) #prototyping

s = dd.loc[dd.index//100%2 == 1].sum(axis=0) #services

m = dd.loc[dd.index//10%2 == 1].sum(axis=0) #models improvement

r = dd.loc[dd.index//1%2 == 1].sum(axis=0) #research

t = ml_data_g.sum(axis=0)#total



ml_e = pd.DataFrame({'Build ML\nprototypes': p, 

                                'Build or run\nML services': s,

                                'Improve\nML models': m,

                                'Research to\nadvance ML': r,

                                'Total': t})



ml_e = ml_e.div(ml_e['Total'],axis=0)*100

ml_e = ml_e.drop(['Total'],axis=1)

ml_e.index.name = 'ML experience'

ml_e = ml_e.sort_values(by=['ML experience'],ascending= False)



fig, ax = plt.subplots(1,4,figsize=(12,3), sharex=True, sharey=True)



for i,c in enumerate(ml_e.columns.to_list()):

    ax[i].hlines(y=ml_e.index, xmin=0, xmax=ml_e[c], color='#1F78B4', alpha=0.7, linewidth=2)

    ax[i].scatter(y=ml_e.index, x=ml_e[c], s=75, color='#1F78B4', alpha=0.7)

    ax[i].set_title(c, fontsize=12)

    ax[i].set_xticks((np.arange(0, 100, 20)))

    ax[i].set_xlabel('% respondents', fontsize=12)

    ax[i].tick_params(labelsize=11)

    



sns.despine(left=True, bottom=True)



annotation_text = "Figure "+next(fig_n)+": Involvment in ML activities for groups with different amount of ML experience\nSource: ML&DL Survey 2019, questions 9&23"

ax[0].annotate(annotation_text, (0,0), (0, -60), xycoords='axes fraction', textcoords='offset points', va='top', ha='left', fontsize=11)





sns.despine(left=True, bottom=True)

plt.show()
fig, ax = plt.subplots(3, 1, figsize=(14,18))



for i, role_name in enumerate(reversed(roles)):



    role_color = roles_c[role_name] #bar colors

    #prepare data

    ml_by_role = ml_data_grouped[role_name].sort_values(ascending=False)

    ml_by_role = ml_by_role.div(ml_by_role.sum(axis=0),axis=0).to_frame(name='per') #calculate percent from the group

    

    #dict to translates ml combinations into checkboxes

    x_coordinates = {-4:1000,

                     -3:100,

                     -2:10,

                     -1:1}

    #what is the maximum value?

    max_val = float(ml_by_role.iloc[0])*100 #maxium x value (as data frame is sorted in descending order)

    # round it to the nearest number devisible by 5

    max_x = math.ceil(max_val/5)*5

    if max_x-max_val < 1:

        max_x = max_x + 5 #increase by 5 if value was already close to the devisible by 5

    

    # scale factor - so that y-axis will be aligned for all 3 charts

    scale_factor = max_x/6

    #how many y points are in data frame?

    y_points = ml_by_role.index.shape[0]

    #add horizontal line

    ax[i].hlines(y=y_points+0.5, xmin=(-4)*scale_factor, xmax=max_x, color='grey', alpha=0.8, linewidth=1.5)

        

    for j, ml_set_code in enumerate(ml_by_role.index):

        #print(i, ml_set_code)

        xs, ys = [],[]

        for x in list(x_coordinates.keys()):

            if (ml_set_code//x_coordinates[x])%2==1:

                xs.append((x+0.5)*scale_factor)

                ys.append(y_points-j) 

                

        ax[i].scatter(y=ys, x=xs, s=12*12, color='grey', marker='$✔︎$')   

        val_x = float(ml_by_role.iloc[j])*100 #value_x

        val_y = y_points-j #value_y

        #plot the horizontal bar

        ax[i].barh(y=val_y, left=0, width=val_x, color=role_color, height = 0.8, alpha=0.8)

        #annotate bar

        ax[i].text(x=val_x+0.1, y=val_y, s='{:.1f}%'.format(val_x), color='grey', ha= 'left', va='center', fontsize=12)

        #highlight particular combinations

        if ml_set_code in [1110,1111]: ax[i].barh(y=val_y, left=0, width=-4*scale_factor, color='#FFE714', height = 0.8, alpha=0.3)

        if ml_set_code in [1000,100,10,1]: ax[i].barh(y=val_y, left=0, width=-4*scale_factor, color='#FF9A47', height = 0.8, alpha=0.3)



    

    # Plots decoraation

    x_major_ticks = np.arange(0, max_x, 5)

    x_minor_ticks = np.arange(list(x_coordinates.keys())[0]*scale_factor, 0, scale_factor)

    ax[i].set_xticks(x_major_ticks)

    ax[i].set_xticks(x_minor_ticks, minor=True)

    ax[i].set_xlim(list(x_coordinates.keys())[0]*scale_factor, max_x) 



    y_minor_ticks = np.arange(0.5, ml_by_role.shape[0]+0.6, 1)

    y_major_ticks = np.arange(y_points+0.5, y_points+3, 3)

    ax[i].set_yticks(y_major_ticks)

    ax[i].set_yticks(y_minor_ticks,minor=True)

    ax[i].set_yticklabels([])

    ax[i].set_ylim(0.5,y_points+3.5)



    ax[i].grid(which='minor', alpha=0.5)

    ax[i].grid(which='major', alpha=0.5)

    ax[i].xaxis.set_label_text("")

    ax[i].tick_params(labelsize=12)



    #put a white bar on the part with text (to hide grid lines)

    for j in [1,2,3]: ax[i].barh(y=y_points+j, left=0.1, width=max_x, height=0.9, color='white')



    #anotating and labeling

    for j,ml_label in enumerate(list(q9_answers_ml_map.values())):

        ax[i].text(((-4+j+0.5)*scale_factor),y_points+2, s=ml_label, ha= 'center', va='center', fontsize=12)



    ax[i].text(max_x/2,y_points+2, s='% of all '+role_name +'s involved in ML*', ha='center', va='bottom', fontsize=12) 



#figure annotation    

annotation_text = ("Figure "+next(fig_n)+": Detailed look at combinations of Machine Learning activities for each job title"+

                "\nSource: ML&DL Survey 2019,question 9"+

                "\n\n*selected at least one option that mentioned ML in question 9")

ax[2].annotate(annotation_text, (0,0), (2, -50), xycoords='axes fraction', textcoords='offset points', va='top', ha='left') 



plt.show()



#ML algorithms

q24_c = responses_ff.filter(like='Q24_A', axis=1).columns.to_list()



ML_methods_map = {

    'Q24_A1': 'Linear\nor\nLogistic\nRegression',

    'Q24_A2': 'Decision\nTrees\nor\nRandom\nForests',

    'Q24_A3': 'Gradient\nBoosting\nMachines',

    'Q24_A4': 'Bayesian\nApproaches',

    'Q24_A5': 'Evolu-\ntionary\nApproaches',

    'Q24_A6': 'Dense\nNeural\nNetworks',

    'Q24_A7': 'Convolu- \ntional\nNeural\nNetworks',

    'Q24_A8': 'Generative\nAdver-\nsarial\nNetworks',

    'Q24_A9': 'Recurrent\nNeural\nNetworks',

    'Q24_A10': 'Transformer\nNetworks',

    'Q24_A11': 'None',

    'Q24_A12': 'Other'}



q24_raw = responses_ff.loc[(responses_ff['Q5'].isin(roles))&(responses_ff['Q24_choices_cnt']>0)][['Survey_time','Q5']+q24_c].groupby(['Q5']).count()

q24_raw = q24_raw.div(q24_raw['Survey_time'], axis=0)

q24_raw = q24_raw.rename(columns=ML_methods_map)

q24_plot = q24_raw.T.drop(['Survey_time'],axis=0).sort_values(['Data Scientist'],ascending=False)*100

q24_plot.columns.name = ''

#plot

fig, ax = plt.subplots(figsize=(12,4))

q24_plot.plot(color = list(roles_c.values()), ax=ax, kind='bar')

ax.set_ylim(0,100)

ax.yaxis.set_label_text("% repsondents",fontsize=12)

ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)

ax.set_title(("Overall, data scientist use more ML algorithms"), fontsize=14)

ax.tick_params(labelsize=10)

annotation_text = ("Figure "+next(fig_n)

            +": Penetration of ML agorithms for different job titles (only people with coding experience were asked)\nSource: ML&DL Survey 2019, question 24")

plt.annotate(annotation_text, (0,0), (0, -100), xycoords='axes fraction', textcoords='offset points', va='top', ha='left', fontsize=11)

  

sns.despine(left=True, bottom=True)

plt.show() 

#ML tools

q25_c = responses_ff.filter(like='Q25_A', axis=1).columns.to_list()



ML_tools_cat_map = {

    'Q25_A1': 'Automated\ndata\naugmentation',

    'Q25_A2': 'Automated\nfeature\nengineering',

    'Q25_A3': 'Automated\nmodel\nselection',

    'Q25_A4': 'Automated\nmodel\narchitecture\nsearches',

    'Q25_A5': 'Automated\nhyperparameter\ntuning',

    'Q25_A6': 'Automation\nof full ML\npipelines',

    'Q25_A7': 'None',

    'Q25_A8': 'Other'}



q25_raw = responses_ff.loc[(responses_ff['Q5'].isin(roles))&(responses_ff['Q25_choices_cnt']>0)][['Survey_time','Q5']+q25_c].groupby(['Q5']).count()

q25_raw = q25_raw.div(q25_raw['Survey_time'], axis=0)

q25_raw = q25_raw.rename(columns=ML_tools_cat_map)

q25_plot = q25_raw.T.drop(['Survey_time'],axis=0).sort_values(['Data Scientist'],ascending=False)*100

q25_plot.columns.name = ''

#plot

fig, ax = plt.subplots(figsize=(12,4))

q25_plot.plot(color = list(roles_c.values()), kind ='bar', ax=ax)

#plot decoration

ax.set_ylim(0,80)

ax.yaxis.set_label_text("% repsondents",fontsize=12)

ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)

ax.set_title(("Less than half of data analysts and business analysts use ML tools.\nUsage among data scientists is a bit higher"), fontsize=14)

ax.tick_params(labelsize=11)

annotation_text = "Figure "+next(fig_n)+": Penetration of ML tools for different job titles (only people with coding experience were asked)\nSource: ML&DL Survey 2019, question 25"

plt.annotate(annotation_text, (0,0), (0, -60), xycoords='axes fraction', textcoords='offset points', va='top', ha='left', fontsize=11)

  

sns.despine(left=True, bottom=True)

plt.show()
q18_c = responses_ff.filter(like='Q18_A', axis=1).columns.to_list()

languages_map = responses_ff.filter(regex='Q18_.*\d$').fillna('').max().to_dict()

#data preparation

pl_raw = responses_ff.loc[(responses_ff['Q5'].isin(roles))&(responses_ff['Q15']!='I have never written code')][['Q5']+q18_c+['Q18_choices_cnt']]

pl_agg = pl_raw.groupby(['Q5']).count().rename(columns={'Q18_choices_cnt':'cnt'})

pl_agg = pl_agg.rename(columns=languages_map)

pl_agg = (pl_agg.div(pl_agg['cnt'],axis=0)*100).drop(['cnt'],axis=1)

pl_agg.index.name=""

#plot

fig, ax = plt.subplots(figsize=(12,4))

pl_agg.T.plot(color = list(roles_c.values()), ax=ax, kind='bar')

#plot decoration

ax.set_xticklabels(pl_agg.columns, rotation=90, fontsize=12)

ax.set_ylim(0,100)

ax.yaxis.set_label_text("% repsondents",fontsize=12)

ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)

ax.set_title(("Python, R, and SQL are the most popular languages among all job titles,\nwith 94% of data scientists using Python on a regular basis"), fontsize=14)

ax.tick_params(labelsize=11)

annotation_text = "Figure "+next(fig_n)+": Language popularity (only people with coding experience were asked)\nSource: ML&DL Survey 2019, question 18"

plt.annotate(annotation_text, (0,0), (0, -30), xycoords='axes fraction', textcoords='offset points', va='top', ha='left', fontsize=11) 

sns.despine(left=True, bottom=True)

plt.tight_layout()

plt.show()
#combinations

pl_bool = pl_raw[['Q18_A1','Q18_A2','Q18_A3']].notna().multiply(1).add_suffix('_bool')

pl_bool['pl_set']=(pl_bool['Q18_A1_bool']*100+

                   pl_bool['Q18_A2_bool']*10+

                   pl_bool['Q18_A3_bool'])

pl_combinations = pd.concat([pl_raw['Q5'], pl_bool['pl_set']],axis=1)

pl_combinations_agg = pl_combinations.groupby(['Q5','pl_set']).size().unstack('Q5')

pl_comb_plot = (pl_combinations_agg.div(pl_combinations_agg.sum(axis=0),axis=1)*100)

pl_comb_plot.columns.name, pl_comb_plot.index.name = "",""

combinations_map = {

    111:'ALL THREE',

    101:'Python + SQL',

    110:'Python + R',

    100:'ONLY Python',

    10: 'ONLY R',

    11: 'R + SQL',

    1: 'ONLY SQL',

    0: 'NONE of THREE'}

pl_comb_plot.index = pl_comb_plot.index.map(combinations_map)

pl_comb_plot.index = pd.Categorical(pl_comb_plot.index, list(combinations_map.values()))

pl_comb_plot = pl_comb_plot.sort_index()



#plot

fig, ax = plt.subplots(figsize=(12,4))

ax.axvspan(-0.5, 3.5, color='grey', alpha = 0.2)

pl_comb_plot.plot(kind='bar', color=list(roles_c.values()),ax=ax)

#plot decoration

ax.set_xticklabels(pl_comb_plot.index, rotation=90, fontsize=12)

ax.set_ylim(0,40)

ax.yaxis.set_label_text("% repsondents",fontsize=12)

ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)

ax.set_title(("A quarter of data scientists use all 3 languages on a regular basis,\nwith another 29% using Python and SQL"), fontsize=14)

ax.tick_params(labelsize=11)

annotation_text = "Figure "+next(fig_n)+": Which combinations of the 3 most popular languages are used\nSource: ML&DL Survey 2019, question 18"

plt.annotate(annotation_text, (0,0), (0, -30), xycoords='axes fraction', textcoords='offset points', va='top', ha='left', fontsize=11) 

sns.despine(left=True, bottom=True)

plt.tight_layout()

plt.show()



#colors

lang_colors = [single_color,'#C3E4E5','#E8A87D','#AAAAAA']

#which language recommend those who know all 3

pl_q19 = pd.concat([pl_combinations, responses_ff.loc[responses_ff['Q5'].isin(roles)]['Q19']], axis=1)

pl_q19_agg = pl_q19.loc[pl_q19['pl_set']==111].groupby(['Q5','Q19']).size().unstack('Q19').fillna(0)

pl_q19_agg = pl_q19_agg.div(pl_q19_agg.sum(axis=1),axis=0)

pl_q19_agg['Other '] = 1 - pl_q19_agg[['Python','R','SQL']].sum(axis=1) 

pl_q19_agg.index.name = ""

#plot

fig, ax = plt.subplots(figsize=(12,4))

(pl_q19_agg[['Python','R','SQL','Other ']]*100).plot(kind='barh', stacked='True', ax=ax, width=0.8, color=lang_colors, alpha=0.7)

#plot decoration

ax.set_title(("The majority of those who know the 3 most popular languages recommend to learn Python first"), fontsize=14)

ax.set_xlim(0,100)

ax.tick_params(labelsize=12)

ax.legend(loc='center', bbox_to_anchor=(0.5, -0.05), ncol=4, labelspacing=0.5, fontsize=14)

annotation_text = "Figure "+next(fig_n)+": Language recommendation to an aspiring data scientist\nfrom those who know all 3 most popular languages\nSource: ML&DL Survey 2019, question 18&19"

plt.annotate(annotation_text, (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', ha='left', fontsize=11) 

sns.despine(left=True, bottom=True)

ax.get_xaxis().set_visible(False)

# add annotations!

for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    x, y = p.get_xy() 

    if width > 5: 

        ax.text(x+width/2, y+height/2, '{:.0f} %'.format(width), ha='center', va='center', fontsize=12)



plt.show()

#Learning platforms

q13_c = responses_ff.filter(like='Q13_A', axis=1).columns.to_list()



platforms_map = {

    'Q13_A1':  'Udacity',

    'Q13_A2':  'Coursera',

    'Q13_A3':  'edX',

    'Q13_A4':  'DataCamp',

    'Q13_A5':  'DataQuest',

    'Q13_A6':  'Kaggle Courses',

    'Q13_A7':  'Fast.ai',

    'Q13_A8':  'Udemy',

    'Q13_A9':  'LinkedIn Learning',

    'Q13_A10':  'University Courses*',

    'Q13_A11':  'None',

    'Q13_A12':  'Other'

}



q13_raw = responses_ff.loc[responses_ff['Q5'].isin(roles)][['Survey_time','Q5']+q13_c].groupby(['Q5']).count()

q13_raw = q13_raw.div(q13_raw['Survey_time'], axis=0)

q13_raw = q13_raw.rename(columns=platforms_map)

#condition = q13_raw > 0.05  #subjective treshhold - less than 15% mentioned particular platform

q13_reduced = q13_raw.drop(['Survey_time','None','Other'],axis=1)

q13_ranks = q13_reduced.rank(method='first',ascending=False,axis=1)



#decide which one to highlight:

q13_ranks_min_max = pd.DataFrame({'diff': q13_ranks.max(axis=0)-q13_ranks.min(axis=0)})

condition = (q13_ranks_min_max['diff'] > 2)

list_highlights = (q13_ranks_min_max.sort_values(by=['diff'],ascending=False)

                       .where(condition, np.nan).dropna(axis=0,how='all').index.tolist())

highlight_colors = ['#FFE8A5','#CAE7B9','#EB9486','#586BA4','#8A716A','#F2F3AE','#1F78B4']



fig, ax = plt.subplots(figsize=(12,5))

#tools_agg.plot(kind='bar', color = list(roles_c.values()), ax=ax)



xs = {'Data Scientist':0,

      'Data Analyst':0.4,

      'Business Analyst':0.8}



for role, role_c in roles_c.items(): 

    row = q13_ranks.loc[role,:]

    interval = 0.8/(row.shape[0])

    ax.text(xs[role], 0.92, role.upper(), 

            ha='left', va='top', fontdict={'fontweight':600, 'size':12},

            bbox={'facecolor':'grey', 'boxstyle':'round',  'alpha':0.5}) 

    for label, value in row.items(): #iterating over series

        y = 0.8 - value*interval

        t = ('#'+str(int(value))+' '+label).upper()

        

        #ax.text(xs[role], y+0.1, t, ha='left', va='center', fontdict={'fontweight':540, 'size':12})

        if label in list_highlights:

            ii = (list_highlights.index(label))%7

            highlight_color = highlight_colors[ii]

            ax.text(xs[role], y+0.1, t, ha='left', va='center', fontdict={'fontweight':540, 'size':12},

                   bbox={'facecolor':highlight_color, 'boxstyle':'round', 'edgecolor':'none', 'alpha':0.5})

        else:

            ax.text(xs[role], y+0.1, t, ha='left', va='center', fontdict={'fontweight':540, 'size':12})



annotation_text = ("Figure "+next(fig_n)+":Top 10 platforms with data science courses for each job title\nSource:ML&DL Survey 2019, question 13"+

                  "\n*resulting in a university degree")

plt.annotate(annotation_text, (0,0), (0, -5), xycoords='axes fraction', textcoords='offset points', va='top', ha='left', fontsize=11)

                             

ax.grid(False)

ax.axis('off')

plt.show()
#Media resources

q12_c = responses_ff.filter(like='Q12_A', axis=1).columns.to_list()



media_map = {

    'Q12_A1':  'Twitter',

    'Q12_A2':  'Hacker News',

    'Q12_A3':  'Reddit',

    'Q12_A4':  'Kaggle',

    'Q12_A5':  'Course Forums',

    'Q12_A6':  'YouTube',

    'Q12_A7':  'Podcasts',

    'Q12_A8':  'Blogs', 

    'Q12_A9':  'Journal Publications',

    'Q12_A10':  'Slack Communities',

    'Q12_A11':  'None',

    'Q12_A12':  'Other'

}



q12_raw = responses_ff.loc[responses_ff['Q5'].isin(roles)][['Survey_time','Q5']+q12_c].groupby(['Q5']).count()

q12_raw = q12_raw.div(q12_raw['Survey_time'], axis=0)

q12_raw = q12_raw.rename(columns=media_map)

q12_reduced = q12_raw.dropna(axis=1, how='all').drop(['Survey_time','Other','None'],axis=1)

q12_ranks = q12_reduced.rank(method='first', ascending=False,axis=1)



#decide which ones to highlight:

q12_ranks_min_max = pd.DataFrame({'diff': q12_ranks.max(axis=0)-q12_ranks.min(axis=0)})

condition = (q12_ranks_min_max['diff'] >= 2)

list_highlights = (q12_ranks_min_max.sort_values(by=['diff'],ascending=False)

                       .where(condition, np.nan).dropna(axis=0,how='all').index.tolist())

highlight_colors = ['#FFE8A5','#CAE7B9','#EB9486','#586BA4','#8A716A','#F2F3AE','#1F78B4']



fig, ax = plt.subplots(figsize=(12,5))

#tools_agg.plot(kind='bar', color = list(roles_c.values()), ax=ax)



xs = {'Data Scientist':0,

      'Data Analyst':0.4,

      'Business Analyst':0.8}



for role, role_c in roles_c.items(): 

    row = q12_ranks.loc[role,:]

    interval = 0.8/(row.shape[0])

    ax.text(xs[role], 0.92, role.upper(), 

            ha='left', va='top', fontdict={'fontweight':600, 'size':12},

            bbox={'facecolor':'grey', 'boxstyle':'round',  'alpha':0.5}) 

    for label, value in row.items(): #iterating over series

        y = 0.8 - value*interval

        t = ('#'+str(int(value))+' '+label).upper()

        

        #ax.text(xs[role], y+0.1, t, ha='left', va='center', fontdict={'fontweight':540, 'size':12})

        if label in list_highlights:

            ii = (list_highlights.index(label))%7

            highlight_color = highlight_colors[ii]

            ax.text(xs[role], y+0.1, t, ha='left', va='center', fontdict={'fontweight':540, 'size':12},

                   bbox={'facecolor':highlight_color, 'boxstyle':'round', 'edgecolor':'none', 'alpha':0.5})

        else:

            ax.text(xs[role], y+0.1, t, ha='left', va='center', fontdict={'fontweight':540, 'size':12})



annotation_text = "Figure "+next(fig_n)+":Top 10 media resourses reporting on data science by job titles\nSource:ML&DL Survey 2019, question 12"

plt.annotate(annotation_text, (0,0), (0, -5), xycoords='axes fraction', textcoords='offset points', va='top', ha='left', fontsize=11)

                



ax.grid(False)

ax.axis('off')

plt.show()
responses_ff=responses_ff.replace({'Q3':'United States of America'},'USA')

#Take only 6 countries (same as in executive summary,as other countries have too few responses for this analysis) 

responses_by_country_cnt = responses_ff['Q3'].value_counts()

countries_list = responses_by_country_cnt.where(responses_by_country_cnt > 390).dropna().index.tolist()

countries_list.remove('Other')



#take latest big mac prices

big_mac_prices = big_mac_data.loc[(big_mac_data['date'] == big_mac_data['date'].max())&

                 big_mac_data['name'].isin(countries_list+['United States'])].replace('United States', 'USA')

big_mac_prices = big_mac_prices.rename(columns={'name':'Q3'})

big_mac_prices['burger_price_USD'] = big_mac_prices['local_price']/big_mac_prices['dollar_ex']



#medians for salary bins:

salary_bins_map = {

    '$0-999' : 500,'1,000-1,999': 1500,'2,000-2,999': 2500, '3,000-3,999': 3500,'4,000-4,999': 4500,

    '5,000-7,499': 6250,'7,500-9,999': 8750,

    '10,000-14,999': 12500,'15,000-19,999': 17500,'20,000-24,999': 22500,'25,000-29,999': 27500, 

    '30,000-39,999': 35000,'40,000-49,999': 45000,'50,000-59,999': 55000,'60,000-69,999': 65000,

    '70,000-79,999': 75000,'80,000-89,999': 85000,'90,000-99,999': 95000,

    '100,000-124,999': 112500,'125,000-149,999': 137500,

    '150,000-199,999': 175000,'200,000-249,999': 212500,'250,000-299,999': 275000, 

    '300,000-500,000': 400000,'> $500,000': 500000

}



salary = responses_ff.loc[responses_ff['Q3'].isin(countries_list),['Q3','Q5','Q10']].copy()



salary['Q10'] = salary['Q10'].map(salary_bins_map) #put a numeric value for the salary for each respondent (median of the bin interval)



#create table with salary medians with countries in rows and job roles in columns

salary_medians = salary.groupby(['Q3','Q5'])['Q10'].mean().unstack('Q5')

salary_medians = salary_medians['Data Scientist'] #yearly compensation 

salary_medians = pd.merge(salary_medians, big_mac_prices[['Q3','burger_price_USD']], how = 'inner', on='Q3').set_index('Q3')

salary_medians = salary_medians.rename(columns={'Data Scientist':'DS yearly salary, USD'})

salary_medians['DS yearly salary, burgers'] = salary_medians['DS yearly salary, USD']/salary_medians['burger_price_USD']

salary_medians = salary_medians.sort_values(['DS yearly salary, USD'])



# plotting

fig, ax = plt.subplots(1,2, figsize=(12,5))



axis_colors = {

    1:'#8A716A',

    0:'#018E42'}

axis_handels = {

    0:'US dollars',

    1:"McDonald's Big Macs"}    

yaxis_labels = {

    0:'yearly compensation, USD',

    1:'yearly compensation, big macs'}



fignum = next(fig_n)

annotations = {

    0:"Figure "+fignum+"a: Average yearly compensation in US dollars.\nSource: ML&DS survey 2019, question 10.",

    1:"Figure "+fignum+"b: Average yearly compensation in Big Macs equivalents.\nSource: The Economist - The Big Mac Index"}

titles = {

    0:("Data Scientists residing in the USA\non average earn 5 times more than\nthose who live in Brazil and India"),

    1:("The disparity in salaries\nlooks less shocking\nwhen they are measured in Big Macs")}

yaxis_limits = {

    0:salary_medians['DS yearly salary, USD'].max()*1.1,

    1:salary_medians['DS yearly salary, burgers'].max()*1.1}







salary_medians['DS yearly salary, USD'].plot(kind='bar',  color = axis_colors[0], alpha = 0.8 , width=0.8, ax=ax[0])

salary_medians['DS yearly salary, burgers'].plot(kind='bar',  color = axis_colors[1], alpha = 0.8 ,  width=0.8, ax=ax[1])



#plots decoration

for i in [0,1]:

    handels = [plt.bar(x=0, height=0, color=axis_colors[i])]

    ax[i].legend(handels, [axis_handels[i]], loc = 'upper left', bbox_to_anchor=(0.01, 0.98), ncol=1, labelspacing=0.5, fontsize=12)

    ax[i].xaxis.grid(False)

    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=0, fontsize=12)

    ax[i].xaxis.set_label_text("")

    ax[i].yaxis.set_label_text(yaxis_labels[i], fontsize=12)

    ax[i].set_title(titles[i], fontsize=14)

    ax[i].set_ylim(0,yaxis_limits[i])

    ax[i].annotate(annotations[i],

                  (0,0), (0, -30), xycoords='axes fraction', textcoords='offset points', 

                  va='top', ha='left',size=11)

    ax[i].tick_params(labelsize=12)



sns.despine(left=True, bottom=True)

plt.tight_layout()

plt.show()