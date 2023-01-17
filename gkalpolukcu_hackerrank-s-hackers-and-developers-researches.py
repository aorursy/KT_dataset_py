import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

import plotly

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.tools as tls

import plotly.graph_objs as go

import plotly.tools as tls
df=pd.read_csv('../input/HackerRank-Developer-Survey-2018-Values.csv')
df.head()
df.tail()
df.info()
df.columns
df.describe()
import missingno as msno

msno.matrix(df)
print('The Total Number of Respondents are:',df.shape[0])
df['q3Gender'].value_counts()[:3].plot.barh(width=0.9)

plt.show()
a=df.q3Gender.value_counts()[0:3]

sizes=a.values

labels=a.index

explode=[0,0]

colors=["blue","red","green"]

plt.figure(figsize=(7,7))

plt.pie(sizes,explode=[0.1]*3,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('Gender',color = 'blue',fontsize = 15)
countries=df['CountryNumeric2'].value_counts().to_frame()

data = [ dict(

        type = 'choropleth',

        locations = countries.index,

        locationmode = 'country names',

        z = countries['CountryNumeric2'],

        text = countries['CountryNumeric2'],

        colorscale ='Viridis',

        autocolorscale = False,

        reversescale = False,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            tickprefix = '',

            title = 'Survey Respondents'),

      ) ]



layout = dict(

    title = 'Survey Respondents by Nationality',

    geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = dict(

            type = 'Mercator'

        )

    )

)



fig = dict( data=data, layout=layout )

py.iplot( fig, validate=False, filename='survey-world-map')
f,ax=plt.subplots(1,2,figsize=(25,20))

ax1=df[df['q1AgeBeginCoding']!='#NULL!'].q1AgeBeginCoding.value_counts().sort_values(ascending=True).plot.barh(width=0.9,ax=ax[0],color='y')

for i, v in enumerate(df[df['q1AgeBeginCoding']!='#NULL!'].q1AgeBeginCoding.value_counts().sort_values(ascending=True)): 

    ax1.text(.8, i, v,fontsize=18,color='b',weight='bold')

ax[0].set_title('Age Began Coding',size=30)

ax2=df[df['q2Age']!='#NULL!'].q2Age.value_counts().sort_values(ascending=True).plot.barh(width=0.9,ax=ax[1],color='y')

for i, v in enumerate(df[df['q2Age']!='#NULL!'].q2Age.value_counts().sort_values(ascending=True)): 

    ax2.text(.8, i, v,fontsize=18,color='b',weight='bold')

ax[1].set_title('Present Age',size=30)

plt.show()
f,ax=plt.subplots(1,2,figsize=(25,12))

curr_age=df[df.q3Gender.isin(['Male','Female'])].groupby(['q2Age','q3Gender'])['StartDate'].count().reset_index()

curr_age=curr_age[curr_age['q2Age']!='#NULL!']

curr_age.pivot('q2Age','q3Gender','StartDate').plot.barh(ax=ax[0],width=0.9)

ax[0].set_title('Current Age')

code_age=df[df.q3Gender.isin(['Male','Female'])].groupby(['q1AgeBeginCoding','q3Gender'])['StartDate'].count().reset_index()

code_age=code_age[code_age['q1AgeBeginCoding']!='#NULL!']

plt.figure(figsize=(15,15))

code_age.pivot('q1AgeBeginCoding','q3Gender','StartDate').plot.barh(ax=ax[1],width=0.9)

ax[1].set_title('Age Started Coding')

plt.subplots_adjust(hspace=0.8)

plt.show()
t1=pd.to_datetime(df['StartDate'])

t2=pd.to_datetime(df['EndDate'])

d = {'col1': t1, 'col2': t2}

trying=pd.DataFrame(d)

list1=[]

for i,j in zip(trying['col2'],trying['col1']):

    list1.append(pd.Timedelta(i-j).seconds / 60.0)
print('Shortest Survey Time:',pd.Series(list1).min(),'minutes')

print('Longest Survey Time:',pd.Series(list1).max(),'minutes')

print('Mean Survey Time:',pd.Series(list1).mean(),'minutes')
(pd.Series(list1)>50).value_counts()
plt.figure(figsize=(10,6))

pd.Series(list1).hist(bins=500,edgecolor='black', linewidth=1.2)

plt.xlim([0,50])

plt.xticks(range(0,50,5))

plt.title('Response Time Distribution')

plt.show()
f,ax=plt.subplots(2,1,figsize=(15,12))

sns.countplot('q5DegreeFocus',hue='q3Gender',data=df[df['q5DegreeFocus']!='#NULL!'],ax=ax[0])

ax[0].set_title('STEM Courses')

sns.countplot('q0005_other',hue='q3Gender',data=df[df['q0005_other'].isin(df['q0005_other'].value_counts().index[:5])],ax=ax[1])

ax[1].set_title('Other Courses')

plt.show()
plt.figure(figsize=(10,8))

df[(df['q10Industry']!='#NULL!')&(df['q8Student']!='Students')].q10Industry.value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('winter_r'))

plt.title('Working Industry')

plt.show()
plt.figure(figsize=(8,8))

def absolute_value(val):

    a  = np.round(val/100.*sizes.sum(), 0)

    return a

sizes=df[(df['q3Gender']=='Female')&(df['q10Industry']!='#NULL!')&(df['q8Student']!='Students')].q10Industry.value_counts()[:10]

labels=df[(df['q3Gender']=='Female')&(df['q10Industry']!='#NULL!')&(df['q8Student']!='Students')].q10Industry.value_counts()[:10].index

plt.pie(sizes,autopct=absolute_value,labels=labels,colors=sns.color_palette('Set3',10))

my_circle=plt.Circle( (0,0), 0.7, color='white')

plt.gcf().gca().add_artist(my_circle)

plt.show()
plt.figure(figsize=(8,8))

def absolute_value(val):

    a  = np.round(val/100.*sizes.sum(), 0)

    return a

sizes=df[(df['q3Gender']=='Male')&(df['q10Industry']!='#NULL!')&(df['q8Student']!='Students')].q10Industry.value_counts()[:10]

labels=df[(df['q3Gender']=='Male')&(df['q10Industry']!='#NULL!')&(df['q8Student']!='Students')].q10Industry.value_counts()[:10].index

plt.pie(sizes,autopct=absolute_value,labels=labels,colors=sns.color_palette('Set3',10))

my_circle=plt.Circle( (0,0), 0.7, color='white')

plt.gcf().gca().add_artist(my_circle)

plt.show()
plt.figure(figsize=(8,12))

women_indus=df[(df['q3Gender']=='Female')&(df['q10Industry']!='#NULL!')&(df['q8Student']!='Students')]

ax=women_indus.q9CurrentRole.value_counts().plot.barh(width=0.9,color=sns.color_palette('winter_r',45))

for i, v in enumerate(women_indus.q9CurrentRole.value_counts().values): 

    ax.text(.8, i, v,fontsize=12,color='black',weight='bold')

plt.gca().invert_yaxis()

plt.title('Current Job Description for Women')

ax.patches[0].set_facecolor('r')

plt.show()
plt.figure(figsize=(8,12))

man_indus=df[(df['q3Gender']=='Male')&(df['q10Industry']!='#NULL!')&(df['q8Student']!='Students')]

ax=man_indus.q9CurrentRole.value_counts().plot.barh(width=0.9,color=sns.color_palette('winter_r',45))

for i, v in enumerate(man_indus.q9CurrentRole.value_counts().values): 

    ax.text(.8, i, v,fontsize=12,color='black',weight='bold')

plt.gca().invert_yaxis()

plt.title('Current Job Description for Man')

ax.patches[0].set_facecolor('r')

plt.show()
plt.figure(figsize=(8,10))

ax=women_indus.q8JobLevel.value_counts().plot.barh(width=0.9,color=sns.color_palette('winter_r',15))

for i, v in enumerate(women_indus.q8JobLevel.value_counts().values): 

    ax.text(.8, i, v,fontsize=12,color='blue',weight='bold')

plt.gca().invert_yaxis()

ax.patches[0].set_facecolor('r')

plt.title('Positions Held By Women')

plt.show()
plt.figure(figsize=(8,10))

ax=man_indus.q8JobLevel.value_counts().plot.barh(width=0.9,color=sns.color_palette('winter_r',15))

for i, v in enumerate(man_indus.q8JobLevel.value_counts().values): 

    ax.text(.8, i, v,fontsize=12,color='blue',weight='bold')

plt.gca().invert_yaxis()

ax.patches[0].set_facecolor('r')

plt.title('Positions Held By Man')

plt.show()
plt.figure(figsize=(12,10))

sns.countplot(y='q27EmergingTechSkill',hue='q3Gender',data=df[(df['q3Gender'].isin(['Male','Female']))&(df['q27EmergingTechSkill']!='#NULL!')])

plt.xticks(rotation=90)

plt.show()
from wordcloud import WordCloud, STOPWORDS

plt.figure(figsize=(16,8))

wc = WordCloud(background_color="white", max_words=1000, 

               stopwords=STOPWORDS,width=1000,height=1000)

wc.generate(" ".join(df['q0027_other'].dropna()))

plt.imshow(wc)

plt.axis('off')

plt.show()

df1=df[df['q16HiringManager']=='Yes']

f,ax=plt.subplots(1,3,figsize=(25,10))

hire1=df1[df1.columns[df1.columns.str.contains('HirCha')]].apply(pd.Series.value_counts).melt().dropna().set_index('variable').sort_values(by='value',ascending=True)

hire1.index=hire1.index.str.replace('q17HirCha','')

hire1.plot.barh(width=0.9,ax=ax[0])

ax[0].set_title('Hiring Challenges')

hire2=df1[df1.columns[df1.columns.str.contains('TalTool')]].apply(pd.Series.value_counts).melt().dropna().set_index('variable').sort_values(by='value',ascending=True)

hire2.index=hire2.index.str.replace('q19TalTool','')

hire2.plot.barh(width=0.9,ax=ax[1])

ax[1].set_title('Talent Assessment Tools')

hire3=df1[df1.columns[df1.columns.str.contains('Cand')]].apply(pd.Series.value_counts).melt().dropna().set_index('variable').sort_values(by='value',ascending=True)

hire3.index=hire3.index.str.replace('q20Cand','')

hire3.plot.barh(width=0.9,ax=ax[2])

ax[2].set_title('Prefered Qualifications')

plt.show()
df1['q0020_other'].value_counts().to_frame()[:10].style.background_gradient(cmap='summer_r')
lang_prof=df1[df1.columns[df1.columns.str.contains('LangProf')]]

lang_prof=lang_prof.apply(pd.Series.value_counts)

lang_prof=lang_prof.melt()

lang_prof.dropna(inplace=True)

lang_prof['variable']=lang_prof['variable'].str.replace('q22LangProf','')

lang_prof.set_index('variable',inplace=True)

frame_prof=df1[df1.columns[df1.columns.str.contains('q23Frame')]]

frame_prof=frame_prof.apply(pd.Series.value_counts)

frame_prof=frame_prof.melt()

frame_prof.dropna(inplace=True)

frame_prof['variable']=frame_prof['variable'].str.replace('q23Frame','')

frame_prof.set_index('variable',inplace=True)

core_comp=df1[df1.columns[df1.columns.str.contains('CoreComp')]]

core_comp=core_comp.apply(pd.Series.value_counts)

core_comp=core_comp.melt()

core_comp.dropna(inplace=True)

core_comp['variable']=core_comp['variable'].str.replace('q21CoreComp','')

core_comp.set_index('variable',inplace=True)

f,ax=plt.subplots(1,3,figsize=(25,15))

lang_prof.sort_values(ascending=True,by='value').plot.barh(width=0.9,ax=ax[0],color=sns.color_palette('inferno_r',10))

ax[0].set_ylabel('Language')

ax[0].set_title('Programming Competancy in Developer Candidates')

frame_prof.sort_values(ascending=True,by='value').plot.barh(width=0.9,ax=ax[1],color=sns.color_palette('inferno_r',10))

ax[1].set_ylabel('Frameworks')

ax[1].set_title('FrameWorks Competancy in Developer Candidates')

core_comp.sort_values(ascending=True,by='value').plot.barh(width=0.9,ax=ax[2],color=sns.color_palette('inferno_r',10))

ax[2].set_ylabel('Skills')

ax[2].set_title('Other Skills in Developer Candidates')

plt.show()
df['q0022_other'].value_counts().to_frame()[:10].style.background_gradient(cmap='summer_r')
plt.figure(figsize=(6,6))

df1[df1['q18NumDevelopHireWithinNextYear']!='#NULL!'].q18NumDevelopHireWithinNextYear.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('winter_r',15))

plt.title('Hires Within Next Year')

plt.show()
plt.figure(figsize=(8,8))

coun=df1.groupby(['CountryNumeric2','q18NumDevelopHireWithinNextYear'])['q3Gender'].count().reset_index()

coun=coun[coun['q18NumDevelopHireWithinNextYear']!='#NULL!']

coun=coun.pivot('CountryNumeric2','q18NumDevelopHireWithinNextYear','q3Gender').dropna(thresh=6)

sns.heatmap(coun,cmap='RdYlGn',fmt='2.0f',annot=True)

plt.show()
import itertools

df2=df.copy()

df2['q8Student'].fillna('Not Student',inplace=True)

columns=['q25LangC','q25LangCPlusPlus','q25LangJava','q25LangPython','q25LangJavascript','q25LangCSharp','q25LangGo','q25Scala','q25LangPHP','q25LangR']

plt.subplots(figsize=(30,30))

length=len(columns)

for i,j in itertools.zip_longest(columns,range(length)):

    plt.subplot((length/2+1),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    sns.countplot(i,hue='q8Student',data=df2,order=df2[i].value_counts().index)

    plt.title(i,size=20)

    plt.ylabel('')

    plt.xlabel('')

plt.show()
df['q25LangOther'].value_counts().to_frame()[:10].style.background_gradient(cmap='summer_r')
columns=['q28LoveC','q28LoveCPlusPlus','q28LoveJava','q28LovePython','q28LoveJavascript','q28LoveCSharp','q28LoveGo','q28LoveScala','q28LovePHP','q28LoveR']

plt.subplots(figsize=(30,30))

length=len(columns)

for i,j in itertools.zip_longest(columns,range(length)):

    plt.subplot((length/2+1),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    sns.countplot(i,hue='q8Student',data=df2,order=df2[i].value_counts().index)

    plt.title(i,size=20)

    plt.ylabel('')

    plt.xlabel('')

plt.show()
working=df2[df2['q8Student']=='Not Student']

working=working[working.columns[working.columns.str.contains('q12JobCrit')]]

working=working.apply(pd.Series.value_counts).melt().dropna().set_index('variable')

students=df2[df2['q8Student']=='Students']

students=students[students.columns[students.columns.str.contains('q12JobCrit')]]

students=students.apply(pd.Series.value_counts).melt().dropna().set_index('variable')

working=working[working.index!='q12JobCritOther']

stu_work=working.merge(students,left_index=True,right_index=True,how='left')

stu_work['total']=stu_work['value_x']+stu_work['value_y']

stu_work['%work']=stu_work['value_x']/stu_work['total']

stu_work['%student']=stu_work['value_y']/stu_work['total']

stu_work.drop(['value_x','value_y','total'],axis=1,inplace=True)

stu_work.index=stu_work.index.str.replace('q12JobCrit','')

stu_work.plot.barh(stacked=True,width=0.9)

fig=plt.gcf()

fig.set_size_inches(8,8)

plt.title('Important Things for Job(Professionals vs Students)')

plt.show()
working=df2[df2['q8Student']=='Not Student']

students=df2[df2['q8Student']=='Students']

working=working[working.columns[working.columns.str.contains('q6LearnCode')]]

working=working.apply(pd.Series.value_counts).melt().dropna().set_index('variable')

students=df2[df2['q8Student']=='Students']

students=students[students.columns[students.columns.str.contains('q6LearnCode')]]

students=students.apply(pd.Series.value_counts).melt().dropna().set_index('variable')

working=working[working.index!='q6LearnCode']

stu_work=working.merge(students,left_index=True,right_index=True,how='left')

stu_work['total']=stu_work['value_x']+stu_work['value_y']

stu_work['%work']=stu_work['value_x']/stu_work['total']

stu_work['%student']=stu_work['value_y']/stu_work['total']

stu_work.drop(['value_x','value_y','total'],axis=1,inplace=True)

stu_work.index=stu_work.index.str.replace('q6LearnCode','')

stu_work.plot.barh(stacked=True,width=0.9)

fig=plt.gcf()

fig.set_size_inches(8,4)

plt.title('Learnt Coding(Professionals vs Students)')

plt.show()
wc = WordCloud(background_color="white", max_words=1000, 

               stopwords=STOPWORDS,width=1000,height=1000)

wc.generate(" ".join(df['q0006_other'].dropna()))

plt.imshow(wc)

plt.axis('off')

fig=plt.gcf()

fig.set_size_inches(10,10)

plt.show()

def correct_answer(ans,col):

    correct=ans

    for i,j in zip(df[col],df.index):

        if i==correct:

            df.loc[j,col]='Correct Answer'

        else:

            df.loc[j,col]='Wrong Answer'

correct_answer('C','q7Level1')

correct_answer('prints "Hello, World!" n times','q15Level2')

correct_answer('num%2 == 0','q31Level3')

correct_answer('Queue','q36Level4')
ques=['q7Level1','q15Level2','q31Level3','q36Level4']

length=len(ques)

plt.figure(figsize=(10,10))

for i,j in itertools.zip_longest(ques,range(length)):

    plt.subplot((length/2),2,j+1)

    plt.subplots_adjust(wspace=0.8)

    df[i].value_counts().plot.pie(autopct='%1.1f%%',colors=['g','r'],wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' })

    plt.title(i)

    my_circle=plt.Circle( (0,0), 0.7, color='white')

    p=plt.gcf()

    p.gca().add_artist(my_circle)

    plt.xlabel('')

    plt.ylabel('')

plt.show()
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12,8))

gridspec.GridSpec(3,3)



plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)

plt.locator_params(axis='x', nbins=5)

plt.locator_params(axis='y', nbins=5)

plt.title('Recommend HackerRank')

df[df['q32RecommendHackerRank']!='#NULL!'].q32RecommendHackerRank.value_counts().plot.pie(autopct='%1.1f%%',shadow=True,colors=['g','r'])

my_circle=plt.Circle( (0,0), 0.7, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)



plt.subplot2grid((3,3), (0,2))

plt.locator_params(axis='x', nbins=5)

plt.locator_params(axis='y', nbins=5)

plt.title('Positive Experience')

df['q34PositiveExp'].value_counts().sort_values().plot.barh(width=0.9,color=sns.color_palette('inferno_r'))



plt.subplot2grid((3,3), (1,2))

plt.locator_params(axis='x', nbins=5)

plt.locator_params(axis='y', nbins=5)

plt.title('Ideal Test Length')

df[df['q34IdealLengHackerRankTest']!='#NULL!'].q34IdealLengHackerRankTest.value_counts().sort_values().plot.barh(width=0.9,color=sns.color_palette('cubehelix_r'))



plt.subplot2grid((3,3), (2,2))

plt.locator_params(axis='x', nbins=5)

plt.locator_params(axis='y', nbins=5)

plt.title('HackerRank Challenge for Job?')

df[df['q33HackerRankChallforJob']!='#NULL!'].q33HackerRankChallforJob.value_counts().plot.barh(width=0.9,color=sns.color_palette('viridis'))

fig.tight_layout()

plt.subplots_adjust(wspace=0.8,hspace=0.4)

plt.show()
df[df.columns[df.columns.str.contains('q30LearnCode')]].apply(pd.Series.value_counts).melt().set_index('variable').dropna().sort_values(by='value').plot.barh(width=0.9,color=sns.color_palette('winter_r'))

fig=plt.gcf()

fig.set_size_inches(6,6)

plt.title('Other Learning Sources')

plt.show()
wc = WordCloud(background_color="white", max_words=1000, 

               stopwords=STOPWORDS,width=1000,height=1000)

wc.generate(" ".join(df['q0030_other'].dropna()))

plt.imshow(wc)

plt.axis('off')

fig=plt.gcf()

fig.set_size_inches(10,10)

plt.show()
