import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
%matplotlib inline

df = pd.read_csv('../input/stack-overflow-2018-developer-survey/survey_results_public.csv')
pd.options.display.max_columns = None
df.head()
print ("Number of rows    :",df.shape[0])
print ("Number of columns :",df.shape[1])
df.describe()
sns.heatmap(df.corr(), annot=True, fmt=".2f");
miss = df.isnull().sum().reset_index()
miss[0] = miss[0]*100/miss[0].sum()

plt.figure(figsize=(13,6))
ax = sns.barplot("index",0,data=miss,color="orange")
plt.xticks(rotation = 90,fontsize=6)
plt.title("percentage of missing values")
ax.set_facecolor("k")
ax.set_ylabel("percentage of missing values")
ax.set_xlabel("variables")
plt.show()
print(df.isnull().sum())
 #how many total missing values do we have?
missing_values_count = df.isnull().sum()
total_cells = np.product(df.shape)
total_missing = missing_values_count.sum()
print(total_missing, 'have missing values')

# percent of data that is missing
print((total_missing/total_cells) * 100, '% of Missing Values in Survey Results Public')
# checking missing data in each survey results public column
total_missing = df.isnull().sum().sort_values(ascending = False)
percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
missing_survey_results_public = pd.concat([total_missing, percentage], axis=1,
                                          keys=['Total Missing (Column-wise)', 'Percentage (%)'])
missing_survey_results_public.head()
print ("Number of respondents :",df["Respondent"].nunique())
#Hobby Do you code as a hobby?
plt.figure(figsize=(8,8))
df["Hobby"].value_counts().plot.pie(autopct = "%1.1f%%",colors = sns.color_palette("prism",3),fontsize=20,
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
plt.title("Coding as a hobby?")
plt.show()
#OpenSource	Do you contribute to open source projects?
plt.figure(figsize=(8,8))
df["OpenSource"].value_counts().plot.pie(autopct = "%1.1f%%",colors = ["r","b"],fontsize=20,
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
plt.title("Contribution to open source projects")
plt.show()
fig = pd.crosstab(df["Hobby"],df["OpenSource"]).plot(kind="bar",
                                                             figsize = (12,5),
                                                             linewidth = 1,
                                                             edgecolor = "w"*4)
fig.set_facecolor("k")
fig.set_title("Respondents who code as hobby and contribute to open dource projects")
plt.show()
#Country - In which country do you currently reside?
plt.figure(figsize=(10,10))
country = df["Country"].value_counts().reset_index()
ax = sns.barplot("Country","index",data=country[:20],linewidth=2,edgecolor="k"*20)
plt.xlabel("number of reponders")
plt.ylabel("country")
plt.title("countries with highest reponders")
plt.grid(True,alpha=.3)

for i,j in enumerate(country["Country"][:20]):
    ax.text(.7,i,j,weight = "bold")
from wordcloud import WordCloud,STOPWORDS
wrds = df[df["Country"].notnull()]["Country"].str.replace(" ","")
wc = WordCloud(background_color="black",colormap="rainbow",scale=5).generate(" ".join(wrds))
plt.figure(figsize=(14,10))
plt.imshow(wc,interpolation="bilinear")
plt.axis("on")
plt.title("word cloud for countries")
plt.show()
#Student - Are you currently enrolled in a formal, degree-granting college or university program?
plt.figure(figsize=(8,8))
df["Student"].value_counts().plot.pie(autopct = "%1.1f%%",colors = sns.color_palette("Set1"),fontsize=20,
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
plt.title("Current students in respondents")
plt.show()
import pycountry

from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected = True)
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def plotlypie(labels, values, hole, pull, colors, title):
    data = [go.Pie(
        labels = labels,
        values = values,
        hole = hole,
        pull=pull,
        marker=dict(colors=colors)
    )]
    
    layout = go.Layout(title = title)
    fig = go.Figure(data = data, layout = layout)
    
    iplot(fig)
    
def plotlybar(labels1 = None, values1 = None, name1 = None,labels2 = None, values2 = None, 
              name2 =None, markercolor1 = '#9ECAE1', markercolor2 = '#ff99ff', title= None, 
              mode = 'group', orientation = 'v'):
    trace1 = go.Bar(
    x = labels1,
    y = values1,
    orientation = orientation, 
    text = values1,
    name = name1,
    textposition = 'auto',
    marker=dict(
        color=markercolor1#'rgb(58,200,225)',
        )
    )    
    
    if labels2 is not None:
        trace2 = go.Bar(
            x = labels2,
            y = values2,
            name = name2,
            text = values2,
            textposition = 'auto',
            marker=dict(
                color=markercolor2#'rgb(58,200,225)',
            )
        )
        data = [trace1, trace2]
    
    else:
        data = [trace1]
    layout = go.Layout(title = title,xaxis=dict(tickangle=-25),
    barmode=mode)
    fig = go.Figure(data = data, layout = layout)
    iplot(fig)
    
def snslvplot(x,y,title):
    plt.figure(figsize = (16,8))
    sns.lvplot(x = x , y = y , palette = "cool" )
    plt.xticks(rotation = 75)
    plt.title(title)
    plt.show()
    
def snsstripplot(x,y,title):
    plt.figure(figsize = (16,8))
    sns.stripplot(x = x , y = y , palette = "cool" )
    plt.xticks(rotation = 75)
    plt.title(title)
    plt.show()
    
def snspointplot(x,y,data, hue, title):
    plt.figure(figsize=(16,8))
    ax = sns.pointplot(x = x, y = y ,data=data,hue=hue,palette="Set2")
    plt.xticks(rotation=90)
    plt.title(title)
    plt.show()
    
    
hobbies = df['Hobby'].value_counts()

colors = ['#FEBFB3', '#E1396C']
plotlypie(hobbies.index, hobbies.values, 0.6 ,0.05, colors, '% of developers having Coding as Hobby' )
opensource = df['OpenSource'].value_counts()
colors = ['#FEBFB3', '#E1396C']
plotlypie(opensource.index, opensource.values, 0.6 , 0.02,colors, '% of developers contribute to open source' )
age = df['Age'].value_counts()
plotlybar(age.index, age.values, 'Age', title = 'Distribution of developers based on Age')
snslvplot( df["Age"] , np.log(df['ConvertedSalary'] + 1), "Salary Based on Age")
oscontribyes = df[df["OpenSource"] == "Yes"]
oscontribNo = df[df["OpenSource"] == "No"]
Agewisecontribyes = oscontribyes["Age"].value_counts()
Agewisecontribno = oscontribNo["Age"].value_counts()
plotlybar(Agewisecontribyes.index, Agewisecontribyes.values, 'Contribute to open Source',Agewisecontribno.index, Agewisecontribno.values,
          'Not Contribute to open Source','#9ECAE1','#ff99ff', title = 'Contribution to Open Source based on Developer`s age')
yearscodingyes = oscontribyes["YearsCoding"].value_counts()
yearscodingno = oscontribNo["YearsCoding"].value_counts()
plotlybar(yearscodingyes.index, yearscodingyes.values, 'Contribute to open Source',yearscodingno.index, yearscodingno.values,
          'Not Contribute to open Source','#9ECAF0','#ff99ff', title = 'Developers contribution to Open Source based on Years of Coding')
snsstripplot( df["YearsCoding"] , np.log(df['ConvertedSalary'] + 1), "Salary Based on Years of Coding")
Student = df['Student'].value_counts()
colors = ['#FEBFB3', '#E1396C','#ff9933']
plotlypie(Student.index, Student.values, 0.4 , 0.05,colors, 'is the respondent student ?' )
studentsyes = oscontribyes["Student"].value_counts()
studentsno = oscontribNo["Student"].value_counts()

plotlybar(studentsyes.index, studentsyes.values, 'Contribute to open Source',studentsno.index, studentsno.values,
          'Not Contribute to open Source','#9ECAE1','#ff99ff', title = 'How much students contribute to Open Source?')
oscontrib = oscontribyes["Country"].value_counts()[:20]
plotlybar(oscontrib.index[:20], oscontrib.values[:20], title =  'Top 20 countries which contribute more to open source')
plotlybar(oscontrib.index[-20:], oscontrib.values[-20:], markercolor1 = '#ff99ff',
          title = 'Least 20 countries which contribute less to open source')
temp1 = []
temp2 = []
for val in oscontrib.index:
    temp1.append(np.sum(df["Gender"][df['Country'] == val] == 'Male'))
    temp2.append(np.sum(df["Gender"][df['Country'] == val] == 'Female'))
    
plotlybar(oscontrib.index, temp1, 'Male',oscontrib.index, temp2,
          'female','#9ECAE1','#ff99ff', title = 'Gender based Open source contribution among top countries ')
CareerSatisfaction = df['CareerSatisfaction'].value_counts()
colors = ['#FEBFB3', '#E1396C','#ff9933', '#ffd480', '#ff6699', '#ffa366', '#ff3300']
plotlypie(CareerSatisfaction.index, CareerSatisfaction.values, 0.4 , 0.05,colors, 
          'Are developer`s satisfied with current career ?' )

temp1 = []
temp2 = []
for val in CareerSatisfaction.index:
    temp1.append(np.sum(df["Gender"][df['CareerSatisfaction'] == val] == 'Male'))
    temp2.append(np.sum(df["Gender"][df['CareerSatisfaction'] == val] == 'Female'))
    
plotlybar(CareerSatisfaction.index, temp1, 'Male',CareerSatisfaction.index, temp2,
          'female','#9ECAE1','#ff99ff', title = 'CareerSatisfaction among Male and Female')
snslvplot( df["CareerSatisfaction"] , np.log(df['ConvertedSalary'] + 1), "Salary Based vs Career Satisfaction")
CompanySize = df['CompanySize'].value_counts()
plotlybar(CompanySize.index, CompanySize.values, title = 'Distribution of Company Size')
temp1 = []
temp2 = []
for val in CompanySize.index:
    temp1.append(np.sum(df["Gender"][df['CompanySize'] == val] == 'Male'))
    temp2.append(np.sum(df["Gender"][df['CompanySize'] == val] == 'Female'))

plotlybar(CompanySize.index, temp1, 'Male',CompanySize.index, temp2,
          'female','#9ECAE1','#ff99ff', title = 'Gender distribution in Companies')
JobSearchStatus = df['JobSearchStatus'].value_counts()
colors = ['#FEBFB3', '#E1396C','#ff9933']
plotlypie(JobSearchStatus.index, JobSearchStatus.values, 0.4 , 0.05,colors, 'Distribution of job search status' )
LastNewJob = df['LastNewJob'].value_counts()
colors = ['#FEBFB3', '#E1396C','#ff9933', '#ffd480', '#ff3300']
plotlypie(LastNewJob.index, LastNewJob.values, 0.4 , 0.05,colors, 'Distribution of current job' )
StackOverflowHasAccount = df['StackOverflowHasAccount'].value_counts()
colors = ['#FEBFB3', '#E1396C','#ff9933', '#ffd480', '#ff6699', '#ffa366', '#ff3300']
plotlypie(StackOverflowHasAccount.index, StackOverflowHasAccount.values, 0.4 , 0.05,colors,
          'How many developers has stackoverflow account ?' )
for val in StackOverflowHasAccount.index:
    temp1.append(np.sum(df["Gender"][df['StackOverflowHasAccount'] == val] == 'Male'))
    temp2.append(np.sum(df["Gender"][df['StackOverflowHasAccount'] == val] == 'Female'))
plotlybar(StackOverflowHasAccount.index, temp1, 'Male',StackOverflowHasAccount.index, temp2,
          'Female','#9ECAE1','#ff99ff', title = 'How many Male and Female developers has stackoverflow account ?')
hackathonReasons = pd.DataFrame(df["HackathonReasons"].dropna().str.split(';').tolist()).stack()
hackathonReasons = hackathonReasons.value_counts()
plotlybar(hackathonReasons.index, hackathonReasons.values, title = 'Why do developers participate in Hackathons?')
text = " ".join((df['HackathonReasons']).astype(str)).lower()
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='cool', background_color='Black').generate(text)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title('Reason for Participating in Hackathons');
plt.figure(figsize = (14,8))
plt.subplot(1,2,1)
ax1 = sns.distplot(np.log(df["ConvertedSalary"].dropna() + 1))
ax1.set_title('Histogram of Converted salary')

plt.subplot(1,2,2)
plt.scatter(range(df.shape[0]), np.sort(df['ConvertedSalary'].values))
plt.title("Distribution of Converted Salary")
plt.show()
text = " ".join((df['LanguageWorkedWith']).astype(str)).lower()
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='cool', background_color='Black').generate(text)
plt.figure(figsize=(13,13))
plt.imshow(wc)
plt.axis('off')
plt.title('Which programming language is popular?');
