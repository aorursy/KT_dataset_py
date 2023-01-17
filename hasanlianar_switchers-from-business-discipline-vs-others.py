# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wordcloud
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
pd.set_option('display.max_columns', None)
%matplotlib inline
# read data
path = '../input/'
mcr_df = pd.read_csv(path + 'multipleChoiceResponses.csv', low_memory=False, header=[0,1]) #multiple-choice responses
ffr_df = pd.read_csv(path + 'freeFormResponses.csv', low_memory=False, header=[0,1]) # free form responses

# adjust columns
mcr_df.columns = mcr_df.columns.map('_'.join)
ffr_df.columns = ffr_df.columns.map('_'.join)
# rename columns that we are primarily interested.
mcr_df = mcr_df.rename({'Time from Start to Finish (seconds)_Duration (in seconds)' : 'duration', 
                 'Q1_What is your gender? - Selected Choice' : 'gender', 
                 'Q2_What is your age (# years)?' : 'age', 
                 'Q3_In which country do you currently reside?' : 'country', 
                 'Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?' : 'education', 
                 'Q5_Which best describes your undergraduate major? - Selected Choice' : 'major', 
                 'Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice' : 'title', 
                 'Q7_In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice' : 'industry', 
                'Q8_How many years of experience do you have in your current role?' : 'experience', 
                'Q9_What is your current yearly compensation (approximate $USD)?' : 'compensation', 
                'Q10_Does your current employer incorporate machine learning methods into their business?' : 'employerML?', 
                'Q12_MULTIPLE_CHOICE_What is the primary tool that you use at work or school to analyze data? (include text response) - Selected Choice' : 'primary_tool', 
                'Q17_What specific programming language do you use most often? - Selected Choice' : 'language_often', 
                'Q18_What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice' : 'language_recommend', 
                'Q20_Of the choices that you selected in the previous question, which ML library have you used the most? - Selected Choice' : 'MLlibrary_most',
                'Q22_Of the choices that you selected in the previous question, which specific data visualization library or tool have you used the most? - Selected Choice' :'dataviz_most',       
                'Q23_Approximately what percent of your time at work or school is spent actively coding?' : 'time_coding',
                'Q24_How long have you been writing code to analyze data?': 'coding_experience',
                'Q25_For how many years have you used machine learning methods (at work or in school)?': 'years_ML_used',
                'Q26_Do you consider yourself to be a data scientist?': 'consider_DS',
                'Q32_What is the type of data that you currently interact with most often at work or school? - Selected Choice': 'datatype_most',
                'Q37_On which online platform have you spent the most amount of time? - Selected Choice': 'learning_platform_most',
                'Q40_Which better demonstrates expertise in data science: academic achievements or independent projects? - Your views:': 'academicsVSprojects',
                'Q48_Do you consider ML models to be "black boxes" with outputs that are difficult or impossible to explain?': 'ML_blackbox'}, axis='columns')

# rename columns that we are interested from free form response data.
ffr_df = ffr_df.rename({'Q6_OTHER_TEXT_Select the title most similar to your current role (or most recent title if retired): - Other - Text' : 'current_title',
                                  'Q7_OTHER_TEXT_In what industry is your current employer/contract (or your most recent employer if retired)? - Other - Text':'industry',
                                   "Q13_OTHER_TEXT_Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Other - Text":'ide',
                                  'Q18_OTHER_TEXT_What programming language would you recommend an aspiring data scientist to learn first? - Other - Text':'language_recommend',  
                                  'Q11_OTHER_TEXT_Select any activities that make up an important part of your role at work: (Select all that apply) - Other - Text':'main_activity',
                                  'Q12_OTHER_TEXT_What is the primary tool that you use at work or school to analyze data? (include text response) - Other - Text':'primary_tool',
                                  'Q17_OTHER_TEXT_What specific programming language do you use most often? - Other - Text':'language_often',
                                  'Q22_OTHER_TEXT_Of the choices that you selected in the previous question, which specific data visualization library or tool have you used the most? - Other - Text':'dataviz_most',
                                  'Q30_OTHER_TEXT_Which of the following big data and analytics products have you used at work or school in the last 5 years? (Select all that apply) - Other - Text':'bigdata_tools',
                                  'Q32_OTHER_What is the type of data that you currently interact with most often at work or school? - Other Data - Text':'datatype_most',
                                  'Q37_OTHER_TEXT_On which online platform have you spent the most amount of time? - Other - Text':'learning_platform_most',
                                  'Q21_OTHER_TEXT_What data visualization libraries or tools have you used in the past 5 years? (Select all that apply) - Other - Text' : 'viz'}, axis = 'columns')
# separete switchers from others
mcr_switchers_df = mcr_df[mcr_df.major=='A business discipline (accounting, economics, finance, etc.)']
mcr_exc_switchers_df = mcr_df[mcr_df.major!='A business discipline (accounting, economics, finance, etc.)']
print('Respondents with business major constitute {:.2%} of all respondents'.format(len(mcr_switchers_df)/len(mcr_df)))
print('Respondents with business major constitute {:.2%} of all respondents who answered the question about undergraduate degree'.format(mcr_switchers_df.major.value_counts().sum()/mcr_df.major.value_counts().sum()))
countries = mcr_df.country.value_counts()
data =  dict( type = 'choropleth',
                locations = countries.index,
                z = countries.values,
                text = countries.index,
                locationmode = 'country names',
                colorscale = 'Viridis',
                autocolorscale = False,
                reversescale = True,
                marker = dict( line = dict (
                        color = 'rgb(180,0,180)',width = 0.3
                    ) ),
                colorbar = dict( autotick = False,
                    title = '# of respondents'),
          ) 

layout = dict(
        #title = '2018 Survey Respondents',
        geo = dict(showframe = False,
            showcoastlines = True,
            projection = dict( type = 'Mercator')
        ))

fig = dict( data=[data], layout=layout )

py.iplot( fig, validate=False, filename='world-map' )
mcr_country_df = mcr_df.country.\
replace({'United States of America':'US','United Kingdom of Great Britain and Northern Ireland':'UK'}).value_counts().head(10)
ax = mcr_country_df.plot(kind='bar', rot=0, color='#66ff66',figsize=(10,5), width=.6, title='All respondents')
for p in ax.patches:
             ax.annotate(p.get_height(), (p.get_x() + p.get_width()/2., p.get_height()),
             ha='center', va='center', fontsize=12, color='black', xytext=(0, -10),
             textcoords='offset points')
plt.box(on=None)
mcr_swithers_country_df = mcr_switchers_df.country.replace({'United States of America':'US','United Kingdom of Great Britain and Northern Ireland':'UK'}).value_counts().head(10)
ax = mcr_swithers_country_df.plot(kind='bar', rot=0, color='#66ff66',figsize=(10,5), width=.6, title='switchers')
for p in ax.patches:
             ax.annotate(p.get_height(), (p.get_x() + p.get_width()/2., p.get_height()),
             ha='center', va='center', fontsize=12, color='black', xytext=(0, -10),
             textcoords='offset points')
plt.box(on=None)
mcr_exc_switchers_country_df = mcr_exc_switchers_df.country.\
replace({'United States of America':'US','United Kingdom of Great Britain and Northern Ireland':'UK'}).value_counts().head(10)
ax = mcr_exc_switchers_country_df.plot(kind='bar', rot=0, color='#66ff66',figsize=(10,5), width=.6, title='others')
for p in ax.patches:
             ax.annotate(p.get_height(), (p.get_x() + p.get_width()/2., p.get_height()),
             ha='center', va='center', fontsize=12, color='black', xytext=(0, -10),
             textcoords='offset points')
plt.box(on=None)
data = pd.DataFrame({'others':mcr_exc_switchers_df.consider_DS.dropna().value_counts(normalize=True),'switchers':mcr_switchers_df.consider_DS.dropna().value_counts(normalize=True)})
data['consider_DS'] = data.index
data.reset_index(drop=True, inplace=True)
data.sort_values(by=['others'], ascending=False, inplace=True)
data.dropna(inplace=True)
Y = np.arange(len(data))
X1 = data.others
X2 = data.switchers


plt.axes([0.05,0.05,0.95,0.95])
plt.barh(Y, -X1, facecolor='#df80ff', edgecolor='white')
plt.barh(Y, +X2, facecolor='#66ff66', edgecolor='white')

for y,x in zip(Y,X2):
    plt.text(x+.01, y, '{:.1%}'.format(x), ha='left', va= 'center')

for y,x in zip(Y,X1):
    plt.text(-x-.01, y, '{:.1%}'.format(x), ha='right', va= 'center')

plt.ylim(-.3,len(Y)) 
plt.yticks([])
plt.xlim(-.50,+.50) 
plt.xticks([])


plt.yticks(Y,data.consider_DS)
plt.box(on=None)
plt.legend(data.columns[:-1])
plt.show()
labels = 'Male','Prefer not to say', 'Female', 'Prefer to self-describe'
sizes = [mcr_switchers_df.gender.value_counts()[0],mcr_switchers_df.gender.value_counts()[2],\
        mcr_switchers_df.gender.value_counts()[1],mcr_switchers_df.gender.value_counts()[3]]
explode = (0.1, 0.1, 0.1, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
colors = ['#66b3ff','#ffcc29' ,'#66ff66','#ffcc99']
fig, ax = plt.subplots()
ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=30)
ax.axis('equal')
plt.title('switchers')
labels = 'Male','Prefer not to say', 'Female', 'Prefer to self-describe'
sizes = [mcr_exc_switchers_df.gender.value_counts()[0],mcr_exc_switchers_df.gender.value_counts()[2],\
        mcr_exc_switchers_df.gender.value_counts()[1],mcr_exc_switchers_df.gender.value_counts()[3]]
explode = (0.1, 0.1, 0.1, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
colors = ['#66b3ff','#ffcc29' ,'#66ff66','#ffcc99']
fig, ax = plt.subplots()
ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=30)
ax.axis('equal')  
plt.title('others')
age_swt = mcr_switchers_df.age.value_counts(normalize=True).reindex(['18-21','22-24',
                                                   '25-29','30-34',
                                                   '35-39','40-44',
                                                   '45-49','50-54',
                                                   '55-59','60-69',
                                                   '70-79','80+'])

age_oth = mcr_exc_switchers_df.age.value_counts(normalize=True).reindex(['18-21','22-24',
                                                   '25-29','30-34',
                                                   '35-39','40-44',
                                                   '45-49','50-54',
                                                   '55-59','60-69',
                                                   '70-79','80+'])

data = pd.DataFrame({'others':age_oth,'switchers':age_swt})
data['age'] = data.index
data.reset_index(drop=True, inplace=True)
# data.sort_values(by=['others'], ascending=False, inplace=True)
data.dropna(inplace=True)
Y = np.arange(len(data))
X1 = data.others
X2 = data.switchers


plt.axes([0.05,0.05,0.95,0.95])
plt.barh(Y, -X1, facecolor='#66d9ff', edgecolor='white')
plt.barh(Y, +X2, facecolor='#ff668c', edgecolor='white')

for y,x in zip(Y,X2):
    plt.text(x+.01, y, '{:.1%}'.format(x), ha='left', va= 'center')

for y,x in zip(Y,X1):
    plt.text(-x-.01, y, '{:.1%}'.format(x), ha='right', va= 'center')

plt.ylim(-.3,len(Y)) 
plt.yticks([])
plt.xlim(-.50,+.50) 
plt.xticks([])


plt.yticks(Y,data.age)
plt.box(on=None)
plt.legend(data.columns[:-1])
plt.show()
data = pd.DataFrame({'others':mcr_exc_switchers_df.education.dropna().value_counts(normalize=True),\
                     'switchers':mcr_switchers_df.education.dropna().value_counts(normalize=True)})
data['education'] = data.index
data.reset_index(drop=True, inplace=True)
data.sort_values(by=['others'], ascending=False, inplace=True)
data.dropna(inplace=True)
Y = np.arange(len(data))
X1 = data.others
X2 = data.switchers


plt.axes([0.05,0.05,0.95,0.95])
plt.barh(Y, -X1, facecolor='#ffff66', edgecolor='white')
plt.barh(Y, +X2, facecolor='#ff8000', edgecolor='white')

for y,x in zip(Y,X2):
    plt.text(x+.01, y, '{:.1%}'.format(x), ha='left', va= 'center')

for y,x in zip(Y,X1):
    plt.text(-x-.01, y, '{:.1%}'.format(x), ha='right', va= 'center')

plt.ylim(-.3,len(Y)) 
plt.yticks([])
plt.xlim(-.65,+.65) 
plt.xticks([])


plt.yticks(Y,data.education)
plt.box(on=None)
plt.legend(data.columns[:-1])
plt.show()
ax=mcr_df.major.value_counts().plot(kind='barh', rot=0, color='#ffb366',\
                                    figsize=(10,5), width=.6, title='All - Undergraduate major')
ax.title.set_fontsize(15)
for p in ax.patches:
    width = p.get_width()
    percent = '{:.1%}'.format(width/mcr_df.major.value_counts().sum())
    plt.text(350+p.get_width(), p.get_y()+0.5*p.get_height(),percent, ha='center', va='center')

plt.box(on=None)
q78s = mcr_switchers_df[['industry','experience']].dropna()
q78s['one']=1
q78s_p = pd.pivot_table(q78s, values='one',index='industry',columns='experience',aggfunc='sum', margins=True)
q78s_p = q78s_p[['0-1','1-2','2-3','3-4','4-5','5-10','10-15','15-20','20-25','25-30','30 +','All']].sort_values('All', ascending=False)
ax = q78s_p.iloc[1:,:-1].head(10).plot.barh(stacked=True, figsize=(15,7), cmap='Paired', title='Top10 Industry - switchers')
ax.title.set_fontsize(15)
# ax.legend(fontsize=15, title='experience', title_fontsize=15)
plt.box(on=None)
q78 = mcr_exc_switchers_df[['industry','experience']].dropna()
q78['one']=1
q78_p = pd.pivot_table(q78,values='one',index='industry',columns='experience',aggfunc='sum', margins=True)
q78_p = q78_p[['0-1','1-2','2-3','3-4','4-5','5-10','10-15','15-20','20-25','25-30','30 +','All']].sort_values('All', ascending=False)

ax = q78_p.iloc[1:,:-1].head(10).plot.barh(stacked=True, figsize=(15,7), cmap='Paired', title='Top10 Industry - others')
ax.title.set_fontsize(15)
# ax.legend(fontsize=15, title='experience', title_fontsize=15)
plt.box(on=None)
wc = wordcloud.WordCloud(width=1000, height=500, colormap="Paired", background_color="white")
wc.generate_from_frequencies(ffr_df.industry.value_counts())
plt.figure(figsize=(15,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
ax=mcr_exc_switchers_df.title.value_counts().head(10).plot(kind='barh', rot=0, color='#df80ff',\
                                    figsize=(10,5), width=.6, title='Top10 Titles - others')
ax.title.set_fontsize(15)
for p in ax.patches:
    width = p.get_width()
    percent = '{:.1%}'.format(width/mcr_exc_switchers_df.industry.value_counts().sum())
    plt.text(250+p.get_width(), p.get_y()+0.5*p.get_height(),percent, ha='center', va='center')

plt.box(on=None)
top10_swtitle= mcr_switchers_df.title.value_counts().head(10).to_frame()
top10_swtitle['%'] = top10_swtitle/top10_swtitle.sum()
top10_swtitle = top10_swtitle.rename(columns = {'title':'count'})
top10_swtitle.style.format({'%': '{:.1%}'.format})
ax=mcr_exc_switchers_df.compensation.value_counts().head(10).plot(kind='barh', rot=0, color='#66d9ff',\
                                    figsize=(10,5), width=.6, title='Top10 Compensation - others')
ax.title.set_fontsize(15)
# ax.legend(fontsize=15, title='experience', title_fontsize=15)
for p in ax.patches:
    width = p.get_width()
    percent = '{:.1%}'.format(width/mcr_exc_switchers_df.compensation.value_counts().sum())
    plt.text(250+p.get_width(), p.get_y()+0.5*p.get_height(),percent, ha='center', va='center')

plt.box(on=None)
# mcr_switchers_df.compensation.value_counts().head(10)
top5_swcomp= mcr_switchers_df.compensation.value_counts().head(5).to_frame()
top5_swcomp['%'] = top5_swcomp/top5_swcomp.sum()
top5_swcomp = top5_swcomp.rename(columns = {'compensation':'count'})
top5_swcomp.style.format({'%': '{:.1%}'.format})
Q11sw = mcr_switchers_df.filter(like=("Q11"))
Q11sw_unpivot = pd.melt(Q11sw.iloc[:,:-1]).dropna()
Q11sw_unpivot = Q11sw_unpivot['value'].value_counts()
Q11sw_unpivot = Q11sw_unpivot.to_frame().reset_index()
Q11sw_unpivot.columns = ['Activity type - switchers', 'Count']
Q11sw_unpivot['%'] = Q11sw_unpivot['Count']/Q11sw_unpivot['Count'].sum()
Q11sw_unpivot.style.set_properties(subset=['Activity type'], **{'width': '600px'})
Q11sw_unpivot.style.format({'%': '{:.1%}'.format})
Q11 = mcr_exc_switchers_df.filter(like=("Q11"))
Q11_unpivot = pd.melt(Q11.iloc[:,:-1]).dropna()
Q11_unpivot = Q11_unpivot['value'].value_counts()
Q11_unpivot = Q11_unpivot.to_frame().reset_index()
Q11_unpivot.columns = ['Activity type - others', 'Count']
Q11_unpivot['%'] = Q11_unpivot['Count']/Q11_unpivot['Count'].sum()
Q11_unpivot.style.set_properties(subset=['Activity type'], **{'width': '600px'})
Q11_unpivot.style.format({'%': '{:.1%}'.format})
data = pd.DataFrame({'others':mcr_exc_switchers_df.primary_tool.value_counts(normalize=True),'switchers':mcr_switchers_df.primary_tool.value_counts(normalize=True)})
data['primary_tool'] = data.index
data.reset_index(drop=True, inplace=True)
data.sort_values(by=['others'], ascending=False, inplace=True)
Y = np.arange(len(data))
X1 = data.others
X2 = data.switchers


plt.axes([0.05,0.05,0.95,0.95])
plt.barh(Y, -X1, facecolor='#a3adc2', edgecolor='white')
plt.barh(Y, +X2, facecolor='#d9ff66', edgecolor='white')

for y,x in zip(Y,X2):
    plt.text(x, y, '{:.1%}'.format(x), ha='left', va= 'center')

for y,x in zip(Y,X1):
    plt.text(-x,y, '{:.1%}'.format(x), ha='right', va= 'center')

plt.ylim(-.3,len(Y)) 
plt.yticks([])
plt.xlim(-.65,+.51) 
plt.xticks([])


plt.yticks(Y,data.primary_tool)
plt.box(on=None)
plt.legend(data.columns[:-1])

plt.show()
data = pd.DataFrame({'others':mcr_exc_switchers_df.language_often.dropna().value_counts(normalize=True),\
                     'switchers':mcr_switchers_df.language_often.dropna().value_counts(normalize=True)})
data['language_often'] = data.index
data.reset_index(drop=True, inplace=True)
data.sort_values(by=['others'], ascending=False, inplace=True)
data.dropna(inplace=True)
Y = np.arange(len(data))
X1 = data.others
X2 = data.switchers


plt.axes([0.05,0.05,0.95,0.95])
plt.barh(Y, -X1, facecolor='#66d9ff', edgecolor='white')
plt.barh(Y, +X2, facecolor='#ff668c', edgecolor='white')

for y,x in zip(Y,X2):
    plt.text(x, y, '{:.1%}'.format(x), ha='left', va= 'center')

for y,x in zip(Y,X1):
    plt.text(-x,y, '{:.1%}'.format(x), ha='right', va= 'center')

plt.ylim(-.3,len(Y)) 
plt.yticks([])
plt.xlim(-.65,+.51) 
plt.xticks([])


plt.yticks(Y,data.language_often)
plt.box(on=None)
plt.legend(data.columns[:-1])

plt.show()
wc = wordcloud.WordCloud(width=1000, height=500, colormap="Paired", background_color="white")
wc.generate_from_frequencies(mcr_df.language_often.value_counts())
plt.figure(figsize=(15,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
wc = wordcloud.WordCloud(width=1000, height=500, colormap="Paired", background_color="white")
wc.generate_from_frequencies(ffr_df.language_often.value_counts())
plt.figure(figsize=(15,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
wc = wordcloud.WordCloud(width=1000, height=500, colormap="Paired", background_color="white")
wc.generate_from_frequencies(ffr_df.language_recommend.value_counts())
plt.figure(figsize=(15,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
Q13 = mcr_exc_switchers_df.filter(like=("Q13"))
Q13_unpivot = pd.melt(Q13.iloc[:,:-2]).dropna()
Q13sw = mcr_switchers_df.filter(like=("Q13"))
Q13sw_unpivot = pd.melt(Q13sw.iloc[:,:-2]).dropna()
data = pd.DataFrame({'others':Q13_unpivot['value'].value_counts(normalize=True),'switchers':Q13sw_unpivot['value'].\
                     value_counts(normalize=True)})
data['IDEtype'] = data.index
data.reset_index(drop=True, inplace=True)
data.sort_values(by=['others'], ascending=False, inplace=True)
Y = np.arange(len(data))
X1 = data.others
X2 = data.switchers


plt.axes([0.05,0.05,0.95,0.95])
plt.barh(Y, -X1, facecolor='#df80ff', edgecolor='white')
plt.barh(Y, +X2, facecolor='#66ff66', edgecolor='white')

for y,x in zip(Y,X2):
    plt.text(x, y, '{:.1%}'.format(x), ha='left', va= 'center')

for y,x in zip(Y,X1):
    plt.text(-x,y, '{:.1%}'.format(x), ha='right', va= 'center')

plt.ylim(-.3,len(Y)) 
plt.yticks([])
plt.xlim(-.25,+.25) 
plt.xticks([])


plt.yticks(Y,data.IDEtype)
plt.box(on=None)
plt.legend(data.columns[:-1])

plt.show()
wc = wordcloud.WordCloud(width=1000, height=500, colormap="Paired", background_color="white")
wc.generate_from_frequencies(ffr_df.ide.value_counts())
plt.figure(figsize=(15,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
Q14 = mcr_exc_switchers_df.filter(like=("Q14"))
Q14_unpivot = pd.melt(Q14.iloc[:,:-1]).dropna()
Q14sw = mcr_switchers_df.filter(like=("Q14"))
Q14sw_unpivot = pd.melt(Q14sw.iloc[:,:-1]).dropna()
data = pd.DataFrame({'others':Q14_unpivot['value'].value_counts(normalize=True),'switchers':Q14sw_unpivot['value'].\
                     value_counts(normalize=True)})
data['notebook'] = data.index
data.reset_index(drop=True, inplace=True)
data.sort_values(by=['others'], ascending=False, inplace=True)
Y = np.arange(len(data))
X1 = data.others
X2 = data.switchers


plt.axes([0.05,0.05,0.95,0.95])
plt.barh(Y, -X1, facecolor='#6699ff', edgecolor='white')
plt.barh(Y, +X2, facecolor='#ff8c66', edgecolor='white')

for y,x in zip(Y,X2):
    plt.text(x+.01, y, '{:.1%}'.format(x), ha='left', va= 'center')

for y,x in zip(Y,X1):
    plt.text(-x-.01,y, '{:.1%}'.format(x), ha='right', va= 'center')

plt.ylim(-.3,len(Y)) 
plt.yticks([])
plt.xlim(-.40,+.40) 
plt.xticks([])


plt.yticks(Y,data.notebook)
plt.box(on=None)
plt.legend(data.columns[:-1])
# plt.grid(True, which='both', linestyle='--')

plt.show()
data = pd.DataFrame({'others':mcr_exc_switchers_df.time_coding.value_counts(normalize=True),\
                     'switchers':mcr_switchers_df.time_coding.value_counts(normalize=True)})
data['time_coding'] = data.index
data.reset_index(drop=True, inplace=True)
data.sort_values(by=['others'], ascending=False, inplace=True)
data.dropna(inplace=True)
Y = np.arange(len(data))
X1 = data.others
X2 = data.switchers


plt.axes([0.05,0.05,0.95,0.95])
plt.barh(Y, -X1, facecolor='#ff00ff', edgecolor='white')
plt.barh(Y, +X2, facecolor='#00ff80', edgecolor='white')

for y,x in zip(Y,X2):
    plt.text(x, y, '{:.1%}'.format(x), ha='left', va= 'center')

for y,x in zip(Y,X1):
    plt.text(-x, y, '{:.1%}'.format(x), ha='right', va= 'center')

plt.ylim(-.3,len(Y)) 
plt.yticks([])
plt.xlim(-.6,+.6) 
plt.xticks([])
plt.yticks(Y,data.time_coding)
plt.box(on=None)
plt.legend(data.columns[:-1])
plt.savefig("image.png")
plt.show()
ax=mcr_switchers_df.MLlibrary_most.value_counts().head(5).\
plot(kind='bar', rot=0, color='#d9ff66',figsize=(10,5), width=.6, title='Top5 ML - switchers')

for p in ax.patches:
             percent = '{:.1%}'.format(p.get_height()/mcr_switchers_df.MLlibrary_most.value_counts().sum())
             ax.annotate(percent, (p.get_x() + p.get_width()/2., p.get_height()),
             ha='center', va='center', fontsize=12, color='black', xytext=(0, 10),
             textcoords='offset points')

plt.box(on=None)
ax=mcr_exc_switchers_df.MLlibrary_most.value_counts().head(5).plot(kind='bar', rot=0, color='#66ffb3',figsize=(10,5), width=.6, title='Top5 ML - others')

for p in ax.patches:
             percent = '{:.1%}'.format(p.get_height()/mcr_exc_switchers_df.MLlibrary_most.value_counts().sum())
             ax.annotate(percent, (p.get_x() + p.get_width()/2., p.get_height()),
             ha='center', va='center', fontsize=12, color='black', xytext=(0, 10),
             textcoords='offset points')
plt.box(on=None)
data = pd.DataFrame({'others':mcr_exc_switchers_df.ML_blackbox.dropna().value_counts(normalize=True),'switchers':mcr_switchers_df.ML_blackbox.dropna().value_counts(normalize=True)})
data['ML_blackbox'] = data.index
data.reset_index(drop=True, inplace=True)
data.sort_values(by=['others'], ascending=False, inplace=True)
data.dropna(inplace=True)
Y = np.arange(len(data))
X1 = data.others
X2 = data.switchers


plt.axes([0.05,0.05,0.95,0.95])
plt.barh(Y, -X1, facecolor='#8000ff', edgecolor='white')
plt.barh(Y, +X2, facecolor='#ffff00', edgecolor='white')

for y,x in zip(Y,X2):
    plt.text(x+.01, y, '{:.1%}'.format(x), ha='left', va= 'center')

for y,x in zip(Y,X1):
    plt.text(-x-.01, y, '{:.1%}'.format(x), ha='right', va= 'center')

plt.ylim(-.3,len(Y)) 
plt.yticks([])
plt.xlim(-.65,+.65) 
plt.xticks([])


plt.yticks(Y,data.ML_blackbox)
plt.box(on=None)
plt.legend(data.columns[:-1])
# plt.grid(True, which='both', linestyle='--')
# plt.savefig("image.png")
plt.show()
data = pd.DataFrame({'others':mcr_exc_switchers_df.dataviz_most.dropna().value_counts(normalize=True),'switchers':mcr_switchers_df.dataviz_most.dropna().value_counts(normalize=True)})
data['dataviz_most'] = data.index
data.reset_index(drop=True, inplace=True)
data.sort_values(by=['others'], ascending=False, inplace=True)
data.dropna(inplace=True)
Y = np.arange(len(data))
X1 = data.others
X2 = data.switchers


plt.axes([0.05,0.05,0.95,0.95])
plt.barh(Y, -X1, facecolor='#ffff66', edgecolor='white')
plt.barh(Y, +X2, facecolor='#ff8000', edgecolor='white')

for y,x in zip(Y,X2):
    plt.text(x+.01, y, '{:.1%}'.format(x), ha='left', va= 'center')

for y,x in zip(Y,X1):
    plt.text(-x-.01, y, '{:.1%}'.format(x), ha='right', va= 'center')

plt.ylim(-.3,len(Y)) 
plt.yticks([])
plt.xlim(-.75,+.75) 
plt.xticks([])


plt.yticks(Y,data.dataviz_most)
plt.box(on=None)
plt.legend(data.columns[:-1])

plt.show()
wc = wordcloud.WordCloud(width=1000, height=500, colormap="Paired", background_color="white")
wc.generate_from_frequencies(ffr_df.dataviz_most.value_counts())
plt.figure(figsize=(15,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
data = pd.DataFrame({'others':mcr_exc_switchers_df.datatype_most.dropna().value_counts(normalize=True),'switchers':mcr_switchers_df.datatype_most.dropna().value_counts(normalize=True)})
data['datatype_most'] = data.index
data.reset_index(drop=True, inplace=True)
data.sort_values(by=['others'], ascending=False, inplace=True)
data.dropna(inplace=True)
Y = np.arange(len(data))
X1 = data.others
X2 = data.switchers


plt.axes([0.05,0.05,0.95,0.95])
plt.barh(Y, -X1, facecolor='#ff6666', edgecolor='white')
plt.barh(Y, +X2, facecolor='#80b3ff', edgecolor='white')

for y,x in zip(Y,X2):
    plt.text(x+.01, y, '{:.1%}'.format(x), ha='left', va= 'center')

for y,x in zip(Y,X1):
    plt.text(-x-.01, y, '{:.1%}'.format(x), ha='right', va= 'center')

plt.ylim(-.3,len(Y)) 
plt.yticks([])
plt.xlim(-.40,+.40) 
plt.xticks([])


plt.yticks(Y,data.datatype_most)
plt.box(on=None)
plt.legend(data.columns[:-1])
# plt.grid(True, which='both', linestyle='--')
# plt.savefig("image.png")
plt.show()
data = mcr_df[['title','datatype_most']].dropna()
data = data[data.title.isin(['Student', 'Data Scientist', 'Software Engineer', 'Data Analyst', 'Research Scientist'])]
data = data[data.datatype_most.isin(['Numerical Data', 'Tabular Data', 'Text Data', 'Time Series Data', 'Image Data'])]
data = data.rename(columns = {'datatype_most':'data type'})
colors = ['#66b3ff','#66d9ff','#66ff66','#ffcc99', '#ff668c']

plt.figure(figsize=(10,6))
g = sns.countplot(y=data.title, hue=data['data type'], palette=colors)
g.set(xlabel='respondents')
plt.title('Title by Data type used')
plt.box(on=None)
q37_others = mcr_exc_switchers_df.learning_platform_most.dropna().value_counts()
index_others = ['Coursera', 'Udemy', 'DataCamp', 'Udacity', 'edX','Other','Other','Other','Other','Other', 'Other', 'Other']
q37_others.index = index_others

colors = ['#66b3ff','#66d9ff','#66ff66','#ffcc99', '#ff668c', '#ffff66']
labels= ['Coursera', 'Other', 'Udemy', 'DataCamp','Udacity',  'edX']
fig, ax = plt.subplots()
ax.pie(q37_others.groupby(level=0).sum().sort_values(ascending=False),colors=colors, autopct='%1.1f%%', startangle=50,\
       labels=labels, textprops={'fontsize': 14})
ax.axis('equal')
fig.set_size_inches(6, 6)
plt.title('Others', fontsize = 20)
q37_switchers = mcr_switchers_df.learning_platform_most.dropna().value_counts()
index_switchers = ['Coursera', 'DataCamp', 'Udemy', 'edX', 'Udacity', 'Other','Other','Other','Other','Other', 'Other', 'Other']
q37_switchers.index = index_switchers

colors = ['#66b3ff','#66d9ff','#66ff66','#ffcc99', '#ff668c', '#ffff66']
labels= ['Coursera', 'DataCamp','Udemy', 'edX', 'Udacity', 'Other']
fig, ax = plt.subplots()
ax.pie(q37_switchers.groupby(level=0).sum().sort_values(ascending=False),colors=colors, autopct='%1.1f%%', startangle=70,\
       labels=labels, textprops={'fontsize': 14})
ax.axis('equal')
fig.set_size_inches(6, 6)
plt.title('Switchers', fontsize = 20)
wc = wordcloud.WordCloud(width=1000, height=500, colormap="Paired", background_color="white")
wc.generate_from_frequencies(ffr_df.learning_platform_most.value_counts())
plt.figure(figsize=(15,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")