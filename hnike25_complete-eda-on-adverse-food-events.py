import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('ticks')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../input/CAERS_ASCII_2004_2017Q2.csv')
df.head(2)
# Check the number of missing data per columns in percentage
(df.isnull().sum()/df.shape[0])*100
#Split the columns name at (_) and lower case all the word in columns
df.columns = [df.columns[x].split('_')[1].lower() for x in range(len(df.columns))]
df.columns = df.columns.str.replace (' #', '_id')
df.columns = ['report_id', 'caers created date', 'event start date', 'product role',
       'product name', 'industry code', 'industry name',
       'age at adverse event', 'age unit', 'gender', 'outcomes',
       'symptoms']
df_no_duplicate = df
#Dropping all duplicate rows but keeping the first row
df_no_duplicate.drop_duplicates(subset='report_id',inplace=True)
## Convert df_no_duplicate['caers created date'] and df_no_duplicate ['event start date'] to datetime format
df_no_duplicate['caers created date'] = pd.to_datetime(df_no_duplicate['caers created date'])
df_no_duplicate['event start date'] = pd.to_datetime(df_no_duplicate['event start date'])

# Compute the time Difference Between the Event Start Date and the CAERS Created Date
df_no_duplicate['time_diff'] = (df_no_duplicate['caers created date'] - df_no_duplicate['event start date']).dt.days

# Group the Time difference by different periods
time_list = (0,180,366,1098,1830,3660)
time_cat = ['less than 6 Months','less than 1 year','1 to 3 years','3 to 5 years','5 to 10 Years']
df_no_duplicate['time_group'] = pd.cut(df_no_duplicate['time_diff'],time_list,labels= time_cat)

df_no_duplicate['time_group'] = df_no_duplicate['time_group'].astype('str')

df_no_duplicate['time_group'] = df_no_duplicate['time_group'].replace('nan','Unknown')
## Let Split the data into two dataset...one that has age unit and one that don't('Not Available' as value)
data_age = df_no_duplicate[df_no_duplicate['age unit']!= 'Not Available']

# Convert the age unit to week, which means year = 52 week, month= 4 weeks, day = 1/7 ~= 0.143 week, decade = 520 weeks
age_change = {'Year(s)':52,'Month(s)':4,'Week(s)':1,'Day(s)':0.143,'Decade(s)':520}
data_age['age_by_week'] = df['age unit'].map(age_change)

#Multiply the age at adverse by age_by_week to have Age in a single unit which is weeks
data_age['age_in_week'] = data_age['age at adverse event'] * data_age['age_by_week']
# Generate a new column that will break down the age_in_week by category
age_list = (0,6,52,156,260,624,936,2080,3380,5200)
#The numbers in age_list are in weeks. We divide a value from age_list by 52, we will have the result in years
age_cat = ['Newborn','Infant','Toddler','Preschooler','Child','Adolescent','Young Adult','Adult','Elderly']
data_age['age_group'] = pd.cut(data_age['age_in_week'],age_list,labels=age_cat)
data_time = data_age[data_age['time_group'] != 'Unknown']
age_index = data_age['age_group'].value_counts().index.tolist()
def value_data (input):
    value_input = data_time[data_time['age_group']== input]['time_group'].value_counts().values.tolist()
    return value_input

def index_data (input):
    index_input = data_time[data_time['age_group']== input]['time_group'].value_counts().index.tolist()
    return index_input
def pie_chart (a,b,c):    
    fig = {
      "data": [
        {
          "values":value_data(a),
          "labels":index_data(a),
          "text":a,
          "textposition":"inside",
          "domain": {"x": [0, .33]},
          "textfont": {'size':12,'color':'white'},  
          "name": a,
          "hoverinfo":"label+percent+name",
          "hole": .4,
          "type": "pie"
        },
        {
          "values": value_data(b),
          "labels": index_data(b),
          "text":b,
          "textposition":"inside",
          "domain": {"x": [.34, .66]},
          "textfont": {'size':12,'color':'white'},
          "name": b,
          "hoverinfo":"label+percent+name",
          "hole": .4,
          "type": "pie"
        },  
        {
          "values": value_data(c),
          "labels": index_data(c),
          "text":c,
          "textposition":"inside",
          "domain": {"x": [.67, 1]},
          "textfont": {'size':12,'color':'white'},
          "name": c,
          "hoverinfo":"label+percent+name",
          "hole": .4,
          "type": "pie"
        }],
      "layout": {
            "title":"Duration Between Adverse Event Start Date & CAERS Created Date",
            "annotations": [
                {
                    "font": {
                        "size": 20
                    },
                    "showarrow": False,
                    "text": a,
                    "x": 0.12,
                    "y": 0.5
                },
                {
                    "font": {
                        "size": 20
                    },
                    "showarrow": False,
                    "text": b,
                    "x": 0.5,
                    "y": 0.5
                },
                {
                    "font": {
                        "size": 20
                    },
                    "showarrow": False,
                    "text": c,
                    "x": 0.9,
                    "y": 0.5
                }
            ]
        }
    }
    return py.iplot(fig)
pie_chart(age_index[0],age_index[1],age_index[2]) # Pie chart for ['Adult','Elderly','Young Adult']
pie_chart(age_index[3],age_index[4],age_index[5]) # Pie chart for ['Child','Adolescent','Infant]
pie_chart(age_index[6],age_index[7],age_index[8]) # Pie chart for [Toddler','Preschooler','Newborn']
data_age['year_start'] = data_age['event start date'].apply(lambda x: x.year)
data_age['year_reported'] = data_age['caers created date'].apply(lambda x: x.year)
x1= data_age['year_start'].value_counts().sort_index().index 
y1= data_age['year_start'].value_counts().sort_index().values 

x2= data_age['year_reported'].value_counts().sort_index().index 
y2= data_age['year_reported'].value_counts().sort_index().values 

trace1 = go.Scatter(
    x=x1,
    y=y1,
    mode = 'lines+markers',
    marker=dict(
    color='red'),
    name = 'Event start',
    
)
trace2 = go.Scatter(
    x=x2,
    y=y2,
    mode = 'lines+markers',
    marker=dict(
    color='blue'),
    name = 'Event Reported',
    
)
data = [trace1,trace2]
layout = go.Layout(
    
    xaxis=dict(
        title = 'Years',
        range = [1999.9,2018],
        autotick=True,
        
    ),
    
    xaxis2=dict(
        range = [1999.9,2018],
        autotick=True
    ),
    yaxis = dict(
    title = 'Frequency'
    ),
    
    title = 'Number of Adverse Events & CAERS Reports Created From [2000 - 2016]'
    )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

gender = ['Male','Female']
data_gender = data_age[data_age['gender'].isin (gender)]
def value_data_2 (input):
    value_input = data_gender[data_gender['age_group']== input]['gender'].value_counts().values.tolist()
    return value_input

def index_data_2 (input):
    index_input = data_gender[data_gender['age_group']== input]['gender'].value_counts().index.tolist()
    return index_input
def pie_chart (a,b,c):    
    fig = {
      "data": [
        {
          "values":value_data_2(a),
          "labels":index_data_2(a),
          "text":a,
          "textposition":"inside",
          "domain": {"x": [0, .33]},
          "textfont": {'size':12,'color':'white'},  
          "name": a,
          "hoverinfo":"label+percent+name",
          "hole": .4,
          "type": "pie"
        },
        {
          "values": value_data_2(b),
          "labels": index_data_2(b),
          "text":b,
          "textposition":"inside",
          "domain": {"x": [.34, .66]},
          "textfont": {'size':12,'color':'white'},
          "name": b,
          "hoverinfo":"label+percent+name",
          "hole": .4,
          "type": "pie"
        },  
        {
          "values": value_data_2(c),
          "labels": index_data_2(c),
          "text":c,
          "textposition":"inside",
          "domain": {"x": [.67, 1]},
          "textfont": {'size':12,'color':'white'},
          "name": c,
          "hoverinfo":"label+percent+name",
          "hole": .4,
          "type": "pie"
        }],
      "layout": {
            "title":"Gender Distribution by Age Group",
            "annotations": [
                {
                    "font": {
                        "size": 20
                    },
                    "showarrow": False,
                    "text": a,
                    "x": 0.12,
                    "y": 0.5
                },
                {
                    "font": {
                        "size": 20
                    },
                    "showarrow": False,
                    "text": b,
                    "x": 0.5,
                    "y": 0.5
                },
                {
                    "font": {
                        "size": 20
                    },
                    "showarrow": False,
                    "text": c,
                    "x": 0.9,
                    "y": 0.5
                }
            ]
        }
    }
    return py.iplot(fig)
pie_chart(age_index[0],age_index[1],age_index[2]) # Pie chart for ['Adult','Elderly','Young Adult']
pie_chart(age_index[3],age_index[4],age_index[5]) # Pie chart for ['Child','Adolescent','Infant]
pie_chart(age_index[6],age_index[7],age_index[8]) # Pie chart for [Toddler','Preschooler','Newborn']
from pandas.api.types import CategoricalDtype
def date_extract (input,cmap):
    data_age['month']= data_age[input].apply(lambda x: x.month) #this will extract Month from the event_date format
    data_age['day of week']= data_age[input].apply(lambda x: x.dayofweek)

    # Extracting Month from the event start date
    dmonth = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',
             9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    data_age['month'] = data_age['month'].map(dmonth)
    #Categorizing the month from the normal order
    cat_type = CategoricalDtype(categories=['Jan', 'Feb', 'Mar', 'Apr','May','Jun', 'Jul', 
                                            'Aug','Sep', 'Oct', 'Nov', 'Dec'],ordered=True)
    data_age['month'] = data_age['month'].astype(cat_type)

    # Extracting Day of Week from the event start date
    dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
    cat_type_2 = CategoricalDtype(categories=['Mon', 'Tue', 'Wed', 'Thu','Fri','Sat', 'Sun'],ordered=True)

    data_age['day of week']= data_age['day of week'].map(dmap)
    data_age['day of week'] = data_age['day of week'].astype(cat_type_2)
    
    ## Plot the heatmap of day of week and month
    event_timeline = data_age.groupby(['day of week','month']).count()['report_id'].unstack()

    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}

    f, (ax, cbar_ax) = plt.subplots(2,gridspec_kw=grid_kws, figsize=(15,10))

    ax = sns.heatmap(event_timeline, cmap=cmap,linecolor='white',linewidths=1,annot=True,fmt='.1f',
                    ax=ax, cbar_ax=cbar_ax,cbar_kws={"orientation": "horizontal"}, annot_kws={'fontsize':13})
    ax.tick_params(length=3, width=1, colors='black',labelsize='large')
    kwargs= {'fontsize':13, 'color':'black'}
    ax.set_xlabel('Months',**kwargs)
    ax.set_ylabel('Day of Week',**kwargs)
    ax.set_title('Weekly '+ input + ' per Month',fontdict={'fontsize':20, 'color':'black'})
    
    return plt.show()
# Event Start Date
date_extract('event start date','Reds')
# caers created date
date_extract('caers created date','Blues')

def death_num (input):
    death_data = data_age.dropna(subset=[input])
    ##death outcomes
    dd = death_data[death_data[input].str.contains('DEATH')]['age_group']
    da = pd.DataFrame(dd.value_counts()).reset_index()
    da.columns = ['age_group','values']
    return da
# Age Group data
dag = pd.DataFrame(data_age['age_group'].value_counts()).reset_index()
dag.columns = ['age_group','values']
f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,8))
color1 = ["darkgray","darkgray","darkgray",'DarkOrange',"darkgray",'dodgerblue',"darkgray","darkgray","DarkCyan"]

#Color
width = 0.8
#Graph formula
dag.plot(kind = 'barh',ax=ax1,color='gray',legend=None,width=width)
death_num('symptoms').plot(kind = 'barh',ax=ax2,color='slategray',legend=None,width=width)
death_num('outcomes').plot(kind = 'barh',ax=ax3,color='darkred',legend=None,width=width)
#title
title = ["Age Group Distribution by Adverse","Age Group Distribution by (Potential) Death Symptoms",
      "Age Group Distribution by Death Outcomes"]
# set individual bar lables using above list
axes = [ax1,ax2,ax3]
labels = [dag,death_num('symptoms'),death_num('outcomes')]
for j in range(len(axes)):
    for i in axes[j].patches:
    # get_width pulls left or right; get_y pushes up or down
        axes[j].text(i.get_width()+.1, i.get_y()+.40, \
            str(round((i.get_width()), 2)), fontsize=12, color='black',alpha=0.8)  
    #Set ylabel
    axes[j].set_yticklabels(labels[j]['age_group'])
    # invert for largest on top 
    axes[j].invert_yaxis()
    # Set Title
    axes[j].set_title(title[j], fontsize=18)
    
    #Set Axis
    kwargs= {'length':3, 'width':1, 'colors':'black','labelsize':'x-large'}
    axes[j].tick_params(**kwargs)
    x_axis = axes[j].axes.get_xaxis().set_visible(False)
    
    f.subplots_adjust(wspace=0.22,right=2,hspace=0.4)
    sns.despine(bottom=True)
def symptom_count (age_group):
    ex = data_age[data_age['age_group']== age_group]#['symptoms'].astype(object).str.split(",").iteritems()
    word_count = []
    for i, word in ex['symptoms'].replace(' ',"").astype(object).str.split(",").iteritems():
        word_count += [str(w).strip().title().lower() for w in pd.Series(word).astype(object)]
    ds = pd.DataFrame(data = pd.Series(word_count).value_counts().head(10)).reset_index()
    ds.columns = ['symptoms','values']
    return ds
def func_age (a,b,c):    
    f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,8))
    
    #Color
    color = ['slategray']
    width = 0.8
    #Graph formula
    symptom_count(a).plot(kind = 'barh',ax=ax1,color=color,legend=None,width=width)
    symptom_count(b).plot(kind = 'barh',ax=ax2,color=color,legend=None,width=width)
    symptom_count(c).plot(kind = 'barh',ax=ax3,color=color,legend=None,width=width)

    # set individual bar lables using above list
    axes = [ax1,ax2,ax3]
    labels = [a,b,c]
    for j in range(len(axes)):
        for i in axes[j].patches:
        # get_width pulls left or right; get_y pushes up or down
            axes[j].text(i.get_width()+.12, i.get_y()+.38, \
                str(round((i.get_width()), 2)), fontsize=12, color='black',alpha=0.8)  
    # Place the y label on all the axis
        kwargs = {'fontsize':12,'color': 'black','alpha':0.8}
        axes[j].set_yticklabels(symptom_count(labels[j]).symptoms,**kwargs)
    # invert for largest on top 
        axes[j].invert_yaxis()
        #Set Axis
        kwargs= {'length':3, 'width':1, 'colors':'black','labelsize':'x-large'}
        axes[j].tick_params(**kwargs)
        x_axis = axes[j].axes.get_xaxis().set_visible(False)
        f.subplots_adjust(wspace=0.22,right=2,hspace=0.4)
        sns.despine(bottom=True)
        # Set Title
        k = [a,b,c]
        axes[j].set_title("Top 10 Symptoms on "+ k[j], fontsize=25, color ='black')
    return plt.show()
func_age(age_index[0],age_index[1],age_index[2]) # bar Plot for ['Adult','Elderly','Young Adult']

func_age(age_index[3],age_index[4],age_index[5]) # bar plot for ['Child','Adolescent','Infant]

func_age(age_index[6],age_index[7],age_index[8]) # bar plot for [Toddler','Preschooler','Newborn']
def outcomes_count (age_group):
    ex = data_age[data_age['age_group']== age_group]#['symptoms'].astype(object).str.split(",").iteritems()
    word_count = []
    for i, word in ex['outcomes'].replace(' ',"").astype(object).str.split(",").iteritems():
        word_count += [str(w).strip().title().lower() for w in pd.Series(word).astype(object)]
    ds = pd.DataFrame(data = pd.Series(word_count).value_counts().head(10)).reset_index()
    ds.columns = ['outcome','values']
    return ds
def func_age_2 (a,b,c):    
    f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,8))
    #Color
    color = ['dodgerblue']
    width = 0.8
    #Graph formula
    outcomes_count(a).plot(kind = 'barh',ax=ax1,color=color,legend=None,width=width)
    outcomes_count(b).plot(kind = 'barh',ax=ax2,color=color,legend=None,width=width)
    outcomes_count(c).plot(kind = 'barh',ax=ax3,color=color,legend=None,width=width)

    # set individual bar lables using above list
    axes = [ax1,ax2,ax3]
    labels = [a,b,c]
    for j in range(len(axes)):
        for i in axes[j].patches:
        # get_width pulls left or right; get_y pushes up or down
            axes[j].text(i.get_width()+.12, i.get_y()+.38, \
                str(round((i.get_width()), 2)), fontsize=12, color='black',alpha=0.8)  
        
        # Place the y label on all the axis
        axes[j].set_yticklabels(outcomes_count(labels[j]).outcome)
    # invert for largest on top 
        axes[j].invert_yaxis()
        #Set Axis
        kwargs= {'length':3, 'width':1, 'colors':'black','labelsize':'x-large'}
        axes[j].tick_params(**kwargs)
        x_axis = axes[j].axes.get_xaxis().set_visible(False)
        f.subplots_adjust(wspace=0.3,right=2,hspace=0.4)
        sns.despine(bottom=True)
        # Set Title
        k = [a,b,c]
        axes[j].set_title("Top 10 Outcomes on "+ k[j], fontsize=25, color ='black')
        
    return plt.show()
func_age_2(age_index[0],age_index[1],age_index[2]) # bar Plot for ['Adult','Elderly','Young Adult']

func_age_2(age_index[3],age_index[4],age_index[5]) # bar plot for ['Child','Adolescent','Infant]

func_age_2(age_index[6],age_index[7],age_index[8]) # bar plot for [Toddler','Preschooler','Newborn']
