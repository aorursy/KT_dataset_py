import numpy as np # linear algebra

import pandas as pd # data processing

from math import pi





#Data Visualization libraries 

import matplotlib.pyplot as plt 

import seaborn as sns 

# Bokeh

from bokeh.io import output_notebook,output_file, show

from bokeh.plotting import figure, show

from bokeh.io import save, output_file

from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider , LabelSet

from bokeh.layouts import column

from bokeh.palettes import all_palettes

from bokeh.io import output_file, show

from bokeh.palettes import Category20c

from bokeh.transform import cumsum

output_notebook()











import os

print(os.listdir("../input"))



# Print all rows and columns

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

df = pd.read_csv('../input/survey_results_public.csv')

schema = pd.read_csv('../input/survey_results_schema.csv')
df.info()
df.head()
df.tail()
df.shape
pd.options.display.max_colwidth = 500 #This particular statement sets max width to 500px, per column.

schema 
#First Step 

#Discover different jobs "unique" in DevType Column

print(df.DevType.unique())

#Second Step 

#ds is DataFrame for the Data Science industry

ds = df[df['DevType'] == 'Data scientist or machine learning specialist']



print( " THe Data Science DataFrame shape %s and The Stack Overflow DataFrame shape %s"%(ds.shape,df.shape))
country= ds.Country.value_counts().head(10)

x= country.index

y= country/country.sum() * 100

f, ax1 = plt.subplots(figsize=(20,10))

base_color = sns.color_palette()[0]

sns.barplot(x=x, y=y,ax=ax1,color = base_color)

ax1.set(ylabel="Percentage % ",xlabel="Countries " ,title = 'The Top Countries having Data Scientists and Machine Learning Speciaslists')

plt.rc('axes', titlesize=25)     # fontsize of the axes title

plt.rc('axes', labelsize=20)    # fontsize of the x and y labels

plt.rc('xtick', labelsize=15)    # fontsize of the tick labels

plt.rc('ytick', labelsize=15)    # fontsize of the tick labels

plt.show()



gen = pd.DataFrame(ds['Gender'].dropna().str.split(';').tolist()).stack()

gen=  gen.value_counts().sort_values(ascending=False)

labels = gen.index

labels= 'Male', 'Female', 'Non-binary or Transgender'



f, ax1 = plt.subplots(figsize=(15,7))





sizes = gen/gen.sum() * 100

sizes = [85.594640,12.897822 , 1.5075379]



# Pie chart, where the slices will be ordered and plotted counter-clockwise:



#explsion

explode = (0.05,0.05,0.05)



colors= ['#66b3ff','#c2c2f0', '#ff9999']

    

ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)



#draw circle

centre_circle = plt.Circle((0,0),0.50,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)



ax1.axis('equal')   # Equal aspect ratio ensures that pie is drawn as a circle

plt.tight_layout()

plt.show()

age= pd.DataFrame(ds['Age'].dropna().str.split(';').tolist()).stack()

age =  age.value_counts().sort_values(ascending=False)

x= age.index

y= age/age.sum() * 100

f, ax1 = plt.subplots(figsize=(20,10))

base_color = sns.color_palette()[0]

sns.barplot(x=x, y=y,ax=ax1,color=base_color)

ax1.set(ylabel="",xlabel="Percentage % " , title = 'The Age of Data Scientists and Machine Learning Speciaslists')

plt.rc('axes', titlesize=25)     # fontsize of the axes title

plt.rc('axes', labelsize=20)    # fontsize of the x and y labels

plt.rc('xtick', labelsize=15)    # fontsize of the tick labels

plt.rc('ytick', labelsize=15)    # fontsize of the tick labels

plt.show()

formal_edu = ds.FormalEducation.value_counts()

y= formal_edu.index

x= formal_edu/formal_edu.sum() * 100

f, ax1 = plt.subplots(figsize=(12,10))

base_color = sns.color_palette("muted")[0]

sns.barplot(x=x, y=y,ax=ax1,color=base_color)

ax1.set(ylabel="",xlabel="Percentage % " , title = 'The Highest Level of Formal Education for the Data Scientists')

plt.rc('xtick', labelsize=15)    # fontsize of the tick labels

plt.rc('ytick', labelsize=20)    # fontsize of the tick labels



plt.show()

under_grad = ds.UndergradMajor.value_counts()

y= under_grad.index

x=(under_grad/under_grad.sum() * 100)

f, ax1 = plt.subplots(figsize=(10,9))

base_color = sns.color_palette("muted")[0]

sns.barplot(x=x, y=y,ax=ax1,color=base_color)

ax1.set(ylabel="",xlabel="Percentage % " , title = 'The Major Field of Study for The Data Scientists')



plt.show()

lw = pd.DataFrame(ds['EducationTypes'].dropna().str.split(';').tolist()).stack()

lw = lw.value_counts().sort_values(ascending=False).head(10)

y1=lw.index

x1=lw/lw.sum()*100



# Set up the matplotlib figure

f, ax1= plt.subplots(figsize=(20, 8))

base_color = sns.color_palette("muted")[0]

sns.barplot(x=x1, y=y1 ,color=base_color,ax=ax1)



ax1.set(xlabel="Percentage % ",ylabel="", title = "Top Types of Non-degree Eduction ")

plt.tight_layout()

plt.show()

years= ds.YearsCodingProf.value_counts()

x= years.index



y=np.array ([1739 , 4952 , 4472 ,  2870 ,  2004 , 1600 , 1400 ,  1175 , 754 ,  589, 322])

y1= y/y.sum() * 100



# Set up the matplotlib figure

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

base_color = sns.color_palette()[0]

sns.barplot(x=x, y=y1 ,color=base_color,ax=ax1)



y2 = years/years.sum() * 100

base_color = sns.color_palette()[0]

sns.barplot(x=x, y=y2,ax=ax2,color=base_color)



ax1.set(ylabel="Percentage % ",xlabel="", title = "No.Years Coding (Including any Education)")



ax2.set(ylabel="Percentage % ",xlabel="" , title = "No.Years Coding Professionally (As a Part of your Work)")



plt.rc('xtick', labelsize=15)    # fontsize of the tick labels

plt.tight_layout()

plt.show()

employment = ds.Employment.value_counts()

y= employment.index

x= employment/employment.sum() * 100

f, ax1 = plt.subplots(figsize=(10,8))

base_color = sns.color_palette("muted")[0]

sns.barplot(x=x, y=y,ax=ax1, color=base_color)

ax1.set(ylabel="",xlabel="Percentage % " ,title = 'Employment Status for the Data scientists')

plt.show()

cs = ds.CompanySize.value_counts()

y= cs.index

x= cs/cs.sum() * 100

f, ax1 = plt.subplots(figsize=(20,10))

base_color = sns.color_palette("muted")[0]

sns.barplot(x=x, y=y,ax=ax1,color=base_color)

ax1.set(ylabel="",xlabel="Percentage % " , title = 'Which Companies Size do Data Scientists and Machine Learning Working for?')

plt.rc('axes', titlesize=25)     # fontsize of the axes title

plt.rc('axes', labelsize=20)    # fontsize of the x and y labels

plt.rc('xtick', labelsize=15)    # fontsize of the tick labels

plt.rc('ytick', labelsize=20)    # fontsize of the tick labels

plt.show()



#Change Type of Salary Series from objet to numeric 

ds.Salary = pd.to_numeric(ds.Salary,errors='coerce')

ds.Salary = ds.Salary.astype('float')

#Change Salray Type Weekly to Yearly (One Year=52 Weeks) + Filter the outlier by making maximum Salary in the Week 4.000 $  

w = ds.Salary[(ds.SalaryType == 'Weekly') & (ds.Salary <= 4000) & (ds.Salary >= 20)] * 52

#Change Salray Type Monthly to Yearly (One Year= 12 Month) + Filter the outlier by making maximum Salary in the Month = 30.000 $  

m = ds.Salary[(ds.SalaryType == 'Monthly') & (ds.Salary <= 30000)] * 12

#Filter the outlier by making maximum Salary in the year = 500.000 $  

ye = ds.Salary[(ds.SalaryType == 'Yearly') & (ds.Salary <= 500000)]

ds.new_salary = pd.concat([ye,m,w])

#Drop missing Value "NAN"

ds.new_salary.dropna(how='any',axis=0)

#Sort Index for new Salary 

ds.new_salary.sort_index(ascending=True,axis=0)

#Make new DataFrame with Join Campany Size and Year Coding Professional 

heat = pd.concat([ds.new_salary,ds.CompanySize, ds.YearsCodingProf], axis=1, join='inner')



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



heat_map= pd.pivot_table(heat,values='Salary',index=['CompanySize'],columns=["YearsCodingProf"])



# Draw a heatmap with the numeric values in each cell

f, ax = plt.subplots(figsize=(18, 12))

sns.heatmap(heat_map ,cmap='YlGnBu',annot=False, linewidths=.5, ax=ax)



plt.show()

js = ds["JobSatisfaction"].value_counts()

# Data to plot

labels = list(js.index)

sizes = list(js/gen.sum() * 100)



f, ax1 = plt.subplots(figsize=(15,10))



# Pie chart, where the slices will be ordered and plotted counter-clockwise:



#explsion

explode = (0.05,0.05,0.05,0.1,0.1,0.1,0.1)



colors= ['#99ff99','#4dae02','#b7ffb7','#ffc1c1','#FF9999','#9999ff','#e58989']



ax1.pie(sizes, labels=labels,colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.60,explode=explode)



ax1.axis('equal')   # Equal aspect ratio ensures that pie is drawn as a circle



plt.tight_layout()

plt.show()

lw = pd.DataFrame(ds['LanguageWorkedWith'].dropna().str.split(';').tolist()).stack()

lw = lw.value_counts().sort_values(ascending=False).head(10)

y1=lw.index

x1=lw/lw.sum()*100

ln= pd.DataFrame(ds['LanguageDesireNextYear'].dropna().str.split(';').tolist()).stack()

ln = ln.value_counts().sort_values(ascending=False).head(10)

y2=ln.index

x2=ln/ln.sum()*100



# Set up the matplotlib figure

f, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 8))

base_color = sns.color_palette("muted")[0]



sns.barplot(x=x1, y=y1 ,color = base_color,ax=ax1)



sns.barplot(x=x2, y=y2,ax=ax2,color=base_color)



ax1.set(xlabel="Percentage % ",ylabel="", title = "Top 10 Programming Languages used in The Data Science Indutry")



ax2.set(xlabel="Percentage % ",ylabel="" , title = " Top 10 languages desired in Next Year")



plt.rc('xtick', labelsize=12)    # fontsize of the tick labels

plt.rc('ytick', labelsize=12)    # fontsize of the tick labels

plt.rc('axes', titlesize=15)     # fontsize of the axes title

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels



plt.tight_layout()

plt.show()

lw = pd.DataFrame(ds['DatabaseWorkedWith'].dropna().str.split(';').tolist()).stack()

lw = lw.value_counts().sort_values(ascending=False).head(10)

y1=lw.index

x1=lw/lw.sum()*100

ln= pd.DataFrame(ds['DatabaseDesireNextYear'].dropna().str.split(';').tolist()).stack()

ln = ln.value_counts().sort_values(ascending=False).head(10)

y2=ln.index

x2=ln/ln.sum()*100



# Set up the matplotlib figure

f, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 8))



base_color = sns.color_palette("muted")[0]

sns.barplot(x=x1, y=y1 ,color=base_color,ax=ax1)

sns.barplot(x=x2, y=y2,ax=ax2,color=base_color)



ax1.set(xlabel="Percentage % ",ylabel="", title = "Top 10 Database used in The Data Science Indutry")

ax2.set(xlabel="Percentage % ",ylabel="" , title = " Top 10 Database desired in Next Year")



plt.rc('xtick', labelsize=12)    # fontsize of the tick labels

plt.rc('ytick', labelsize=12)    # fontsize of the tick labels

plt.rc('axes', titlesize=15)     # fontsize of the axes title

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels



plt.tight_layout()

plt.show()

lw = pd.DataFrame(ds['PlatformWorkedWith'].dropna().str.split(';').tolist()).stack()

lw = lw.value_counts().sort_values(ascending=False).head(8)

y1=lw.index

x1=lw/lw.sum()*100



# Set up the matplotlib figure

f, ax1= plt.subplots(figsize=(15, 8))

base_color = sns.color_palette("muted")[0]

sns.barplot(x=x1, y=y1 ,color=base_color,ax=ax1)



ax1.set(xlabel="Percentage % ",ylabel="", title = "Top Platform Data Scientists used it")

plt.rc('xtick', labelsize=15)    # fontsize of the tick labels

plt.rc('ytick', labelsize=15)    # fontsize of the tick labels

plt.rc('axes', titlesize=25)     # fontsize of the axes title

plt.rc('axes', labelsize=20)    # fontsize of the x and y labels



plt.tight_layout()

plt.show()

lw = pd.DataFrame(ds['FrameworkWorkedWith'].dropna().str.split(';').tolist()).stack()

lw = lw.value_counts().sort_values(ascending=False).head(5)

y1=lw.index

x1=lw/lw.sum()*100

ln= pd.DataFrame(ds['FrameworkDesireNextYear'].dropna().str.split(';').tolist()).stack()

ln = ln.value_counts().sort_values(ascending=False).head(5)

y2=ln.index

x2=ln/ln.sum()*100



# Set up the matplotlib figure

f, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 8))

base_color = sns.color_palette("muted")[0]

sns.barplot(x=x1, y=y1 ,color=base_color,ax=ax1)

sns.barplot(x=x2, y=y2,ax=ax2,color=base_color)



ax1.set(xlabel="Percentage % ",ylabel="", title = "Top Framework used in The Data Science Indutry")



ax2.set(xlabel="Percentage % ",ylabel="" , title = " Top Framewrokdesired in Next Year")



plt.rc('xtick', labelsize=12)    # fontsize of the tick labels

plt.rc('ytick', labelsize=12)    # fontsize of the tick labels

plt.rc('axes', titlesize=15)     # fontsize of the axes title

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels



plt.tight_layout()

plt.show()





lw = pd.DataFrame(ds['CommunicationTools'].dropna().str.split(';').tolist()).stack()

lw = lw.value_counts().sort_values(ascending=False).head(10)

y1=lw.index

x1=lw/lw.sum()*100



# Set up the matplotlib figure

f, ax1= plt.subplots(figsize=(15, 8))

base_color = sns.color_palette("muted")[0]

sns.barplot(x=x1, y=y1 ,color=base_color,ax=ax1)

ax1.set(xlabel="Percentage % ",ylabel="", title = "Top Communication Tools ")

plt.rc('xtick', labelsize=15)    # fontsize of the tick labels

plt.rc('ytick', labelsize=15)    # fontsize of the tick labels

plt.rc('axes', titlesize=25)     # fontsize of the axes title

plt.rc('axes', labelsize=20)    # fontsize of the x and y labels



plt.tight_layout()

plt.show()



lw = pd.DataFrame(ds['IDE'].dropna().str.split(';').tolist()).stack()

lw = lw.value_counts().sort_values(ascending=False).head(7)

y1=lw.index

x1=lw/lw.sum()*100



# Set up the matplotlib figure

f, ax1= plt.subplots(figsize=(15, 8))

base_color = sns.color_palette("muted")[0]

sns.barplot(x=x1, y=y1 ,color=base_color,ax=ax1)



ax1.set(xlabel="Percentage % ",ylabel="", title = "Top Communication Tools ")

plt.rc('xtick', labelsize=15)    # fontsize of the tick labels

plt.rc('ytick', labelsize=15)    # fontsize of the tick labels

plt.rc('axes', titlesize=25)     # fontsize of the axes title

plt.rc('axes', labelsize=20)    # fontsize of the x and y labels



plt.tight_layout()

plt.show()



lw = pd.DataFrame(ds['VersionControl'].dropna().str.split(';').tolist()).stack()

lw = lw.value_counts().sort_values(ascending=False).head(7)

y1=lw.index

x1=lw/lw.sum()*100



# Set up the matplotlib figure

f, ax1= plt.subplots(figsize=(15, 8))

base_color = sns.color_palette("muted")[0]

sns.barplot(x=x1, y=y1 ,color=base_color,ax=ax1)

ax1.set(xlabel="Percentage % ",ylabel="", title = "Top Version Control System Used")

plt.rc('xtick', labelsize=15)    # fontsize of the tick labels

plt.rc('ytick', labelsize=15)    # fontsize of the tick labels

plt.rc('axes', titlesize=25)     # fontsize of the axes title

plt.rc('axes', labelsize=20)    # fontsize of the x and y labels



plt.tight_layout()

plt.show()

lw = pd.DataFrame(ds['Methodology'].dropna().str.split(';').tolist()).stack()

lw = lw.value_counts().sort_values(ascending=False).head(4)

y1=lw.index

x1=lw/lw.sum()*100



# Set up the matplotlib figure

f, ax1= plt.subplots(figsize=(15, 8))

base_color = sns.color_palette("muted")[0]

sns.barplot(x=x1, y=y1 ,color=base_color,ax=ax1)



ax1.set(xlabel="Percentage % ",ylabel="", title = "Top Methodology Used")

plt.rc('xtick', labelsize=15)    # fontsize of the tick labels

plt.rc('ytick', labelsize=15)    # fontsize of the tick labels

plt.rc('axes', titlesize=25)     # fontsize of the axes title

plt.rc('axes', labelsize=20)    # fontsize of the x and y labels





plt.tight_layout()

plt.show()



import plotly.plotly as py1

import plotly.offline as py

py.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.offline as offline

offline.init_notebook_mode()



yes= ds.Hobby[ds.Hobby=='Yes'].count()

no = ds.Hobby[ds.Hobby=='No'].count()



fig = {"data": [{"values": [yes/(yes+no)*100,no/(yes+no)*100],"labels": ['Yes','No'],

                 'marker': {'colors':['rgb(28, 160, 101)','rgb(210, 50, 50)']},'hoverinfo':'label+percent',

                 "hole": 0.5,"type": "pie"}],"layout": {"title":"Coding In The Data Science Industry as a Hobby", 

            "annotations":[ {"font": {"size": 22 },"showarrow": False,"text": "Hobby", "x": 0.50, "y": 0.5}]}}

                

py.iplot(fig, filename='donut')

yes= ds.OpenSource[ds.OpenSource =='Yes'].count()

no = ds.OpenSource[ds.OpenSource =='No'].count()



fig = {"data": [{"values": [yes/(yes+no)*100,no/(yes+no)*100],"labels": ['Yes','No'],

                 'marker': {'colors':['rgb(28, 160, 101)','rgb(210, 50, 50)']},'hoverinfo':'label+percent',

                 "hole": 0.5,"type": "pie"}],"layout": {"title":"Contribution Data Scientist in open source projects", 

            "annotations":[ {"font": {"size": 22 },"showarrow": False,"text": "Open Source", "x": 0.50, "y": 0.5}]}}

                

py.iplot(fig, filename='donut')

lw = pd.DataFrame(ds['HackathonReasons'].dropna().str.split(';').tolist()).stack()

lw = lw.value_counts().sort_values(ascending=False).head(10)

y1=lw.index

x1=lw/lw.sum()*100



# Set up the matplotlib figure

f, ax1= plt.subplots(figsize=(25,8))

base_color = sns.color_palette("muted")[0]

sns.barplot(x=x1, y=y1 ,color= base_color,ax=ax1)



ax1.set(xlabel="Percentage % ",ylabel="",

        title = "Top Reasons for Data Scientists about \ntheir Contributions in Coding Competition or Hackathon" )



plt.rc('ytick', labelsize=22)    # fontsize of the tick labels

plt.tight_layout()

plt.show()

aid = ds.AIDangerous.value_counts()

y= aid.index

x= aid/aid.sum() * 100

f, ax1 = plt.subplots(figsize=(15,7))

base_color = sns.color_palette("muted")[0]

sns.barplot(x=x, y=y,ax=ax1,color=base_color)

ax1.set(ylabel="",xlabel="Percentage % " , title = 'Opinions about dangerous aspects of increasingly advanced AI')

plt.show()
aii = ds.AIInteresting.value_counts()

y= aii.index

x= aii/aii.sum() * 100

f, ax1 = plt.subplots(figsize=(15,8))

base_color = sns.color_palette("muted")[0]

sns.barplot(x=x, y=y,ax=ax1,color=base_color)

ax1.set(ylabel="",xlabel="Percentage % " , title = 'Opinions about exciting aspects of increasingly advanced AI')

plt.show()
air = ds.AIResponsible.value_counts()

y= air.index

x= air/air.sum() * 100

f, ax1 = plt.subplots(figsize=(17,8))

base_color = sns.color_palette("muted")[0]

sns.barplot(x=x, y=y,ax=ax1,color=base_color)

ax1.set(ylabel="",xlabel="Percentage % " , title = 'Responsibility in Increasingly Advanced AI')

plt.show()
aif = ds.AIFuture.value_counts()

y= aif.index

x= aif/aif.sum() * 100

f, ax1 = plt.subplots(figsize=(13,8))

base_color = sns.color_palette("muted")[0]

sns.barplot(x=x, y=y,ax=ax1,color=base_color)

ax1.set(ylabel="",xlabel="Percentage % " , title = 'Opinions Data Scientists & Machine Learning Specialist about AI in the Future')

plt.show()