#Render Matplotlib Plots Inline
%matplotlib inline

#Import the standard Python Scientific Libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

#Suppress Deprecation and Incorrect Usage Warnings 
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
# Load the dataset into pandas dataframe
result = pd.read_csv('../input/survey_results_public.csv',low_memory=False)
result.shape
result.head()
plt.figure(figsize=(8,8))
g=sns.countplot(x='SurveyTooLong',data=result,palette=sns.color_palette(palette="viridis"),order=result['SurveyTooLong'].dropna().value_counts().index)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel("Response")
g.set_ylabel("Count")
g.set_title("How do the respondents feel about the survey ?")
gend=result.set_index('SurveyTooLong').Gender.str.split(";",expand=True).stack().reset_index('SurveyTooLong').dropna()
gend.columns=['SurveyTooLong','Gender']
plt.figure(figsize=(12,10))
g=sns.countplot(x='SurveyTooLong',data=gend,hue=gend['Gender'],palette=sns.color_palette(palette="Set3"),order=gend['SurveyTooLong'].dropna().value_counts().index)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel("Response")
g.set_ylabel("Count")
g.set_title("How do the respondents feel about the survey ? - Gender SplitUp")
plt.figure(figsize=(8,8))
g=sns.countplot(result['SurveyEasy'],data=result,palette=sns.color_palette(palette="PiYG"),order=result['SurveyEasy'].value_counts().index)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel('Response')
g.set_ylabel('Count')
g.set_title('How did you find the survey ?')
print('Total Number of Countries with respondents:',result.Country.nunique())
print('Country with highest respondents:',result.Country.value_counts().index[0],'number of respondents:',result.Country.value_counts().values[0])
gender=result['Gender'].str.split(';')
gend=[]
for i in gender.dropna():
    gend.extend(i)
pd.Series(gend).value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('gnuplot2'))
plt.figure(figsize=(12,8))
result['Country'].value_counts()[:10].plot.barh(width=0.9,color=sns.color_palette('viridis_r',10))
plt.title('Respondents Country',size=15)
for i,v in enumerate(result['Country'].value_counts()[:10].values):
    plt.text(0.5,i,v,fontsize=10,color='white',weight='bold')
    
    

print(" % of respondents who prefer not to share their salary:",round((result['Salary'].isnull().sum()/result.shape[0])*100,2),"%")
result['Salary']=result['Salary'].astype(str)
result['Salary']=result['Salary'].apply(lambda x: np.nan  if (pd.isnull(x)) or (x=='-') or (x==0) else float(x.replace(",","")))
result['ConvertedSalary']=result['ConvertedSalary'].astype(str)
result['ConvertedSalary']=result['ConvertedSalary'].apply(lambda x: np.nan  if (pd.isnull(x)) or (x=='-') or (x==0) else float(x.replace(",","")))

salary=result[(result['Salary'].notnull())]
salary[salary['Country']=='United States']['Salary'].describe()
salary[salary['Country']=='India']['Salary'].describe()
salary[salary['Country']=='Germany']['Salary'].describe()
plt.figure(figsize=(10,10))

g=sns.distplot(np.log(result['Salary'].dropna()+1))
g.set_xlabel('Log of Salary',fontsize=16)
g.set_ylabel('Frequency',fontsize=16)
g.set_title('Salary Vs Frequency',fontsize=18)

country=result['Country'].value_counts().sort_values(ascending=False).head(10).reset_index()
country.columns=['Country','Count']
temp=result[result.Country.isin(country['Country'])]
temp.head()
plt.figure(figsize=(10,10))
g=sns.boxplot(x='ConvertedSalary',y='Country',data=temp,palette=sns.color_palette(palette='Set1'),linewidth=1.2,saturation=0.8)
g.set_xlabel('Salary',fontsize=10)
g.set_ylabel('Country',fontsize=10)
g.set_title('Country and Salary',fontsize=16)
import statsmodels.api as sm
from statsmodels.formula.api import ols
 
mod = ols('ConvertedSalary ~ Country',
                data=temp).fit()
                
aov_table = sm.stats.anova_lm(mod, typ=2)
print (aov_table)

#https://www.marsja.se/four-ways-to-conduct-one-way-anovas-using-python/
#list(result)
cols = ['Employment','CareerSatisfaction']
hob_yes = result[result['Hobby']=='Yes']
col = sns.light_palette(color='green',as_cmap=True)
pd.crosstab(hob_yes[cols[0]],hob_yes[cols[1]]).style.background_gradient(cmap=col,axis=0,high=0.8)
hob_no = result[result['Hobby']=='No']
col = sns.light_palette(color='green',as_cmap=True)
pd.crosstab(hob_no[cols[0]],hob_no[cols[1]]).style.background_gradient(cmap=col,axis=0,high=0.8)

#result.YearsCoding.astype('category')
#pd.Categorical(result[cols[0]],categories=['0-2 years','3-5 years','6-8 years','9-11 years','12-14 years','15-17 years','18-20 years'
                                         # '21-23 years','24-26 years','27-29 years','30 or more years'],ordered=True)
#cols = ['YearsCoding','CareerSatisfaction']
#relation =pd.crosstab(result[cols[0]],result[cols[1]])
#relation.plot(kind="bar",figsize=(15,8),width=0.7,title="Career Satisfaction Vs Years of Coding")

plt.figure(figsize=(15,8))
g=sns.countplot(x=result['YearsCoding'],hue=result['CareerSatisfaction'],order=['0-2 years','3-5 years','6-8 years','9-11 years','12-14 years','15-17 years','18-20 years','21-23 years','24-26 years','27-29 years','30 or more years'])
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Exploring the Genders by Repayment Interval", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count Distribuition", fontsize=12)

plt.show()
plt.figure(figsize=(8,8))
g=sns.countplot(x='Student',data=result)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel('Student Type')
g.set_ylabel('Count')
g.set_title('Distribution of Students in the Survey',fontsize=16)
students = result.loc[result['Student']!='No']
plt.figure(figsize=(8,8))
g=sns.countplot(x='Country',data=students,order=students.Country.value_counts().iloc[:10].index)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel('Country')
g.set_ylabel('Count')
g.set_title('Top 10 Countries where Students are from',fontsize=16)
cols =['AssessJob1','AssessJob2','AssessJob3','AssessJob4','AssessJob5','AssessJob6','AssessJob7','AssessJob8','AssessJob9','AssessJob10']
assessjob = result[cols].dropna(how='all')
colname=["The industry that I'd be working in","The financial performance or funding status of the company or organization",
         "The specific department or team I'd be working on","The languages, frameworks, and other technologies I'd be working with","The compensation and benefits offered","The office environment or company culture","The opportunity to work from home/remotely","Opportunities for professional development","The diversity of the company or organization","How widely used or impactful the product or service I'd be working on is"]
assessjob.columns=colname
assessjobmlt=pd.melt(assessjob,value_vars=colname)
assessjobmlt.columns = ['Question','Value']
plt.figure(figsize=(8,8))
g=sns.boxplot(x=assessjobmlt['Question'],y=assessjobmlt['Value'],order=colname,palette='Set3',linewidth=1.8)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel('')
g.set_yticks([1,2,3,4,5,6,7,8,9,10])
g.set_ylabel('Order of Importance')
g.set_title('Survey Response for Job Type Assessment',fontsize=16)
cols = [result.AssessBenefits1.values,result.AssessBenefits2.values,result.AssessBenefits3.values,result.AssessBenefits4.values,result.AssessBenefits5.values,result.AssessBenefits6.values,result.AssessBenefits7.values,result.AssessBenefits8.values,result.AssessBenefits9.values,result.AssessBenefits10.values,result.AssessBenefits11.values]
colnames=[" Salary and/or bonuses","Stock options or shares","Health insurance","Parental leave","Fitness or wellness benefit (ex. gym membership, nutritionist)",
         "Retirement or pension savings matching"," Company-provided meals or snacks","Computer/office equipment allowance","Childcare benefit","Transportation benefit (ex. company-provided transportation, public transit allowance)"," Conference or education budget"]
trace =[]

for i in range(11):
    trace.append(
        go.Box(
        y=cols[i],
        name=colnames[i],
        
        )
                )
    layout=go.Layout(title="Assessing the job benefits",hovermode='closest',showlegend=False,xaxis=dict(showticklabels=True,tickangle=90))
fig = go.Figure(data=trace,layout=layout)
py.iplot(fig,filename='jobbenefits')
    
result['CommunicationTools'].head(10)
commtools=result['CommunicationTools'].str.split(';')
tools =[]
for i in commtools.dropna():
    tools.extend(i)
    
tools_series =pd.Series(tools)

plts=tools_series.value_counts().sort_values(ascending=False).to_frame()
#plt.figure(figsize=(8,8))
g=sns.barplot(plts[0],plts.index,palette=sns.color_palette('inferno_r',10),order=plts.iloc[:10].index)
g.set_xlabel('Communication Tool')
g.set_ylabel('Number of respondents')
g.set_title('Top 10 Most used communication tools',fontsize=16)
result['EducationTypes'].head(10)
edu=result['EducationTypes'].str.split(';')
edutype=[]
for i in edu.dropna():
    edutype.extend(i)
plt2=pd.Series(edutype).value_counts()[1:10].sort_values(ascending=False).to_frame()
plt2.reset_index(level=0,inplace=True)
plt2.columns=['EduType','Count']
plt2['Percent']=round(plt2['Count']/sum(plt2['Count']) *100,2)

plt.figure(figsize=(8,10))
g=sns.barplot(x='EduType',y='Percent',data=plt2,palette=sns.color_palette('Set2',10))
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel('Type of Education')
g.set_ylabel('%')
g.set_title('Top 10 Non Degree Education types',fontsize=16)
plt.figure(figsize=(7,7))
g=sns.countplot(y=result['UndergradMajor'],order=result['UndergradMajor'].value_counts().index,palette=sns.color_palette('Set2'))
g.set_title('Formal Undergraduate Education',fontsize=16)
g.set_xlabel('Education')
g.set_ylabel('Count')


plt.figure(figsize=(12,7))
g=sns.boxplot(y=result['ConvertedSalary'],x=result['UndergradMajor'],data=result.dropna(),order=result['UndergradMajor'].value_counts().index,palette=sns.color_palette('Set1'))
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title('Formal Undergraduate Education vs Salary',fontsize=16)
g.set_xlabel('Education')
g.set_ylabel('Salary')
cs=result[result['UndergradMajor']=='Computer science, computer engineering, or software engineering']
plt.figure(figsize=(7,7))
g=sns.countplot(y=cs['EducationParents'],order=cs['EducationParents'].value_counts().index,palette=sns.color_palette('Set3'))
g.set_title('Formal Education of parents',fontsize=16)
g.set_xlabel('Education')
g.set_ylabel('Count')

dev=result['DevType'].str.split(";")
devtype=[]
for i in dev.dropna():
    devtype.extend(i)
plt2=pd.Series(devtype).value_counts()[1:10].sort_values(ascending=False).to_frame()
plt2.reset_index(level=0,inplace=True)
plt2.columns=['devType','Count']
plt2
plt.figure(figsize=(8,8))
sns.barplot(x='devType',y='Count',data=plt2[~plt2['devType'].str.contains('Student')],palette=sns.color_palette(palette="dark"))
plt.title("Dev Type identified through the survey")
plt.xlabel('Dev Type')
plt.xticks(rotation=90)
plt.ylabel("Count")
dev_sat=result.set_index('CareerSatisfaction').DevType.str.split(';',expand=True).stack().reset_index('CareerSatisfaction').dropna()
dev_sat.columns=['CareerSatisfaction','Job']
dev_sat.head()
cm = sns.light_palette("yellow", as_cmap=True)
pd.crosstab(dev_sat['Job'], dev_sat['CareerSatisfaction'],normalize='index').style.background_gradient(cmap = cm)
dev_asp=result.set_index('HopeFiveYears').DevType.str.split(';',expand=True).stack().reset_index('HopeFiveYears').dropna()
dev_asp.columns=['HopeFiveYears','Job']
dev_asp.head()
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(dev_asp['Job'], dev_asp['HopeFiveYears'],normalize='index').style.background_gradient(cmap = cm,axis=1)
