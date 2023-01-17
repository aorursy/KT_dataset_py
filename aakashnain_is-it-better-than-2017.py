import warnings
warnings.filterwarnings('ignore')
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import re
import math
import glob
import itertools
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff
init_notebook_mode(connected=True)
print(os.listdir("../input"))
# For plotting within the notebook
%matplotlib inline

# Graphics in SVG format are more sharp and legible
%config InlineBackend.figure_format = 'svg'

color = sns.color_palette()

# For REPRODUCIBILITY
seed = 111
np.random.seed(seed)
# Define input path
input_dir = Path('../input/')

# Read the csvs
survey_schema = pd.read_csv(input_dir/ 'SurveySchema.csv')
freeFormResp = pd.read_csv(input_dir/ 'freeFormResponses.csv')
multiChoiceResp = pd.read_csv(input_dir/'multipleChoiceResponses.csv')
print(f"Total number of responses: {len(multiChoiceResp)}")
# Check the schema first
survey_schema.head()
multiChoiceResp.head()
# A handy dandy function for making a bar plot. You can make it as flexible as much as you want!!
def do_barplot(df, 
               figsize=(20,8), 
               plt_title=None, 
               xlabel=None, 
               ylabel=None, 
               title_fontsize=20, 
               fontsize=16, 
               orient='v', 
               clr_code=None, 
               max_counts=None,
               print_pct=True,
               normalize=False,
               rotation=None):
    """
    This function can be used to make a barplot from a pandas dataframe very quickly. It counts the number of instances
    per category and plot all the values on a barchart. The barchart can be made to represent count or in terms of 
    percentages. 
    
    Arguments:
    df: pandas dataframe used for this plot
    figsize: size of the plot
    plt_title: title of the plot
    xlabel: label on X-axis
    ylabel: label on Y-axis
    title_fontsize = fontsize for title
    fontsize: fontsize for x and y labels
    orient: orientation of the plot 'h' or 'v'
    clr_code: color code for seaborn color paelette
    max_counts: limit the number of labels to de displayed
    print_pct: whether to print the count values for each category
    normalize: whether to print percentage instead of raw counts
    rotation: rotation value for ticks
    
    """
    
    # Get the value counts 
    if normalize:
        df_counts = round(df.value_counts(normalize=normalize)*100,2)
    else:
        df_counts = df.value_counts()
        
    total = df.shape[0]
    
    # If there are too many values, limit the amount of information for display purpose
    if max_counts:
        df_counts = df_counts[:max_counts]
    
    # Print the values along with their counts and overall %age
    if print_pct and not normalize:
        for i, idx in enumerate(df_counts.index):
            val = df_counts.values[i]
            percentage = round((val/total)*100, 2)
            print(f"{str(idx).ljust(25)}  {val} or roughly {percentage}%")
    
    # Plot the results 
    plt.figure(figsize=figsize)
    
    if clr_code is None:
        clr_code = np.random.randint(6)

    if orient=='h':
        sns.barplot(y=df_counts.index, x=df_counts.values, orient='h', color=color[clr_code])
    else:
        sns.barplot(x=df_counts.index, y=df_counts.values, orient='v', color=color[clr_code])
            
    plt.title(plt_title, fontsize=title_fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    
    if orient=='h':
        plt.yticks(range(len(df_counts.index)), df_counts.index)
    else:
        plt.xticks(range(len(df_counts.index)), df_counts.index, rotation=rotation)
    plt.show()
    del df_counts
def pct_on_bars(axs, df, offset=50, orientation='v', adjustment=2, pos='center', prec=1, fontsize=10):
    """
    This function can be used to plot percentage on each bar in a barplot. The function assumes
    that for each value on an axis, there is only one corresponding bar. So, if you have plotted something with
    hue, then you should consider using something else
    
    Arguments:
    axs: Matplotlib axis
    df: pandas dataframe used for this plot
    offset: Relative position of the text w.r.t the bar
    orientation: 'h' or 'v'
    adjustment: If the text overflows the bar on either side, you can adjust it by passing some value
    prec: How much precision is to be used for displaying percentage?
    fontsize: size of the font used in percentage text
    
    """
    
    # Get all the bars
    bars = axs.patches
    
    # Size of dataframe
    items = len(df)
    
    assert round(prec)>-1, "Precision value passed is wrong "
    
    # Iterate over each bar and plot the percentage
    for bar in bars:
        width = bar.get_width()
        height = bar.get_height()
        precision = '{0:.' + str(prec) + '%}'
        
        if math.isnan(width):
            width=0
        if math.isnan(height):
            height=0
        
        # Check orientation of the bars
        if orientation=='h':
            val_to_sub = height/adjustment
            axs.text(width + offset, bar.get_y()+bar.get_height()-val_to_sub, 
                    precision.format(width/items), ha=pos, fontsize=fontsize)
        
        elif orientation=='v':
            val_to_sub = width/adjustment
            axs.text(bar.get_x()+width-val_to_sub, height + offset, 
            precision.format(height/items), ha=pos, fontsize=fontsize)
        
        else:
            print("The orientation value you passed is wrong. It can either be horizontal 'h' or vertical 'v'")
def pct_on_stacked_bars(axs, values, orientation='v', pos='center', prec=0, fontsize=10, adjustment=0.05):
    """
    This function can be used to plot percentage on each bar in a stacked barplot. 
    
    Arguments:
    axs: Matplotlib axis
    values: percentage values corresponding to each rectangle in a stacked plot
    offset: Relative position of the text w.r.t the bar
    orientation: 'h' or 'v'
    prec: How much precision is to be used for displaying percentage?
    fontsize: size of the font used in percentage text
    adjustment: decimal value in case the values flow out of the rectange
    
    """
    
    # Get all the bars
    bars = axs.patches
    
    
    # Get all the percentages
    assert round(prec)>-1, "Precision value passed is wrong "
    values = np.round(values, decimals=prec).T.flatten()
    
    if adjustment is not None:
        adjustment=0.05
    
    # Iterate over each bar and plot the percentage
    for i, bar in enumerate(bars):
        width = bar.get_width()
        height = bar.get_height()
        x = bar.get_x()
        y = bar.get_y()
        label = str(values[i]) + "%"
        
        if math.isnan(width):
            width=0
        if math.isnan(height):
            height=0
        
        axs.text(x+width/2+adjustment, y+height/2+adjustment, label, ha=pos, fontsize=fontsize) 
def get_traces(df, 
                freq_df, 
                text=True, 
                textposition='auto', 
                opacity=0.5, 
                orientation='v', 
                prec=0,
                stacked=False):
    """
    This function can be used to generate the traces for a plotly plot. 
    
    Arguments:
    df: unstacked value_counts of a grouped pandas dataframe
    freq_df: similar to freq_df but contains percentages
    text: Whether to put text on bar or not
    textposition: Where to put the text. Look for plotly docs for more info
    opacity: opacity in the bars
    orientation: horizontal or vertical bars
    prec: how much precision to be considered for displaying text
    stacked: whether the traces are for stacked plots or not
    
    """
    
    # An empty list to collect traces for plotly
    data = []

    # Iterate for each group in df
    for i, j in zip(range(freq_df.shape[0]), range(freq_df.shape[1])):
        x = df.index
        if stacked:
            y = freq_df.values[:,j]
        else:
            y = df.values[:,j]
        z = freq_df.values[:,j]
        
        if orientation=='v': 
            # define a trace for the current index
            trace = go.Bar(
            x=x,
            y=y,
            text=[str(np.round(i, decimals=prec)) + "%" for i in z],
            textposition = textposition,
            opacity=opacity,
            orientation='v',
            name = df.columns[j])
        
        elif orientation=='h':
            # define a trace for the current index
            trace = go.Bar(
            y=x,
            x=y,
            text=[str(np.round(i, decimals=prec)) + "%" for i in z],
            textposition = textposition,
            opacity=opacity,
            orientation='h',
            name = df.columns[j])
        
        else:
            print("Wrong orientation value provided")
            return
        
        # add it to the list
        data.append(trace)
    return data
# A handy-dany function for plottinf funnel charts
def draw_funnel_chart(values, 
                      phases, 
                      colors=None, 
                      plot_width=400,
                      section_h=100,
                      section_d=10):
    """
    A function that can be used to generate funnel charts in plotly.
    
    """
    n_phase = len(phases)
    plot_width = plot_width

    # height of a section and difference between sections 
    section_h = section_h
    section_d = section_d

    # Check if the color values are given or not
    if colors is None:
        colors = ['rgb' + str(tuple(np.random.randint(255, size=(3)))) for i in range(n_phase)]
    elif len(colors)!=n_phase:
        assert len(colors)==n_phase, "Number of color values didn't match the number of values"
    else:
        colors = colors

    # multiplication factor to calculate the width of other sections
    unit_width = plot_width / max(values)

    # width of each funnel section relative to the plot width
    phase_w = [int(value * unit_width) for value in values]

    # plot height based on the number of sections and the gap in between them
    height = section_h * n_phase + section_d * (n_phase - 1)

    # list containing all the plot shapes
    shapes = []

    # list containing the Y-axis location for each section's name and value text
    label_y = []

    for i in range(n_phase):
            if (i == n_phase-1):
                    points = [phase_w[i] / 2, height, phase_w[i] / 2, height - section_h]
            else:
                    points = [phase_w[i] / 2, height, phase_w[i+1] / 2, height - section_h]

            path = 'M {0} {1} L {2} {3} L -{2} {3} L -{0} {1} Z'.format(*points)

            shape = {
                    'type': 'path',
                    'path': path,
                    'fillcolor': colors[i],
                    'line': {
                        'width': 1,
                        'color': colors[i]
                    }
            }
            shapes.append(shape)

            # Y-axis location for this section's details (text)
            label_y.append(height - (section_h / 2))

            height = height - (section_h + section_d)

    # For phase names
    label_trace = go.Scatter(
        x=[-100]*n_phase,
        y=label_y,
        mode='text',
        hoverinfo='text',
        text=phases,
        textfont=dict(
            color='rgb(200,200,200)',
            size=12
        )
    )

    # For phase values
    value_trace = go.Scatter(
        x=[70]*n_phase,
        y=label_y,
        mode='text',
        hoverinfo='text',
        text=values,
        textfont=dict(
            color='rgb(200,200,200)',
            size=12
        )
    )
    
    return label_trace, value_trace, shapes
# Select the column Q1 "What is your geneder?"
gender_df = multiChoiceResp['Q1'][1:].dropna()

f,ax=plt.subplots(figsize=(10,5))

# Do countplot 
ax=sns.countplot(gender_df, orient='v', color=color[2])

# Plot percentage on the bars
pct_on_bars(ax, gender_df, orientation='v', offset=50,adjustment=2)

    
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution')
plt.show()
# Select the column for the corresponding question
age_df = multiChoiceResp['Q2'][1:].dropna()
order= ['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-69', '70-79', '80+']

f,ax=plt.subplots(figsize=(10,5))

# Do countplot 
ax=sns.countplot(age_df,order=order, orient='v')

# plot the percentage on bars
pct_on_bars(ax, age_df, orientation='v', offset=50, adjustment=2)

    
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Age Group Distribution')
plt.show()
# Do countplot on age df
age_df = multiChoiceResp[['Q1', 'Q2']][1:].dropna()

# We will consider only three categries for brevity: Female, Male and Others. This isn't done to hurt someone
# I am really sorry if I did hurt someone by doing this.
age_df['Q1'] = age_df['Q1'].replace(['Prefer not to say', 'Prefer to self-describe'], 'Others')

f,ax = plt.subplots(figsize=(20,5))
ax=sns.countplot(x='Q2', data=age_df, order=order, orient='v', hue='Q1')

# Get all the bars and plot the percentage also
bars = ax.patches

sections = len(bars)//3
first_bar = bars[:sections]
second_bar = bars[sections:len(first_bar)+sections]
third_bar = bars[len(second_bar)+sections:]


# Loop over the bars and put text on each bar
for left, middle, right in zip(first_bar, second_bar, third_bar):
        height_l = left.get_height()
        height_m = middle.get_height()
        height_r = right.get_height()
    
        
        if math.isnan(height_l):
            height_l=0.0001
        if math.isnan(height_m):
            height_m=0.0001
        if math.isnan(height_r):
            height_r=0.0001
        
        total = height_l + height_m + height_r

        ax.text(left.get_x() + left.get_width()/3., height_l + 40, '{0:.1%}'.format(height_l/total), ha="center", fontsize=8)
        ax.text(middle.get_x() + middle.get_width()/3., height_m + 40, '{0:.1%}'.format(height_m/total), ha="center",fontsize=8)


    
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Gender Distribution in different age groups')
plt.show()
# Select the column for the corresponding question
country_df = multiChoiceResp['Q3'][1:].dropna()

# Drop columns where the country is mentioned as 'others' or where the respondent declined to share the name of the country
index_to_drop = country_df.index[(country_df=='Other') | (country_df=='I do not wish to disclose my location')]
country_df = country_df.drop(index_to_drop)

# Get the counts for each country
country_value_counts = country_df.value_counts()

# Define data for plotly interactive plot
data = [dict(
        type = 'choropleth',
        autocolorscale = True,
        showscale = True,
        locations = country_value_counts.index,
        z = country_value_counts.values,
        locationmode = 'country names',
        reversescale = False,
        marker = dict(
            line = dict(color = 'rgb(180,180,180)',width = 0.5) 
            ),
        colorbar = dict(
            title = "Count"),
            autotick = False,    
        )]

# Layout of th plot
layout = dict(
    title = 'Where do data science people live?',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(type = 'Mercator')
        ),
    autosize=False,
    width=1000,
    height=600,
    )

# Plot
fig = dict( data=data, layout=layout )
iplot( fig, validate=False, show_link=False)
# Select the column for the corresponding question
country_df = multiChoiceResp['Q3'][1:].dropna()

# Drop columns where the country is mentioned as 'others' or where the respondent declined to share the name of the country
index_to_drop = country_df.index[(country_df=='Other') | (country_df=='I do not wish to disclose my location')]
country_df = country_df.drop(index_to_drop)
country_df = country_df.replace(['United States of America', 'United Kingdom of Great Britain and Northern Ireland'], ['USA', 'UK & Northern Ireland'])

# Check the counts and plot the values
# We will only consider top 10 counries only
do_barplot(country_df, plt_title='Country wise distribution of Data Science people(Top 10 only)', 
           xlabel='Country', ylabel='Percent', 
           figsize=(12,5), orient='v',
           max_counts=10,
           clr_code=5, 
           normalize=True, 
           rotation=20)
# Select the columns you are insterested in
df = multiChoiceResp[['Q1', 'Q3']].dropna()

# Select only those indices where the top 10 countries are there
df = df.iloc[country_df.index]

# Select countries to consider
top10_countries = ['United States of America', 'United Kingdom of Great Britain and Northern Ireland', 
                   'India', 'China', 'Russia','Brazil', 'Germany', 'France', 'Canada', 'Japan']

df = df[df['Q3'].isin(top10_countries)]
df['Q3'] = df['Q3'].replace(['United States of America', 'United Kingdom of Great Britain and Northern Ireland'], 
                            ['USA', 'UK & Northern Ireland'])
df['Q1'] = df['Q1'].replace(['Prefer not to say', 'Prefer to self-describe'], 'Others')


# Group by roles and get the count for experience 
freq_df = df.groupby(['Q3'])['Q1'].value_counts().unstack()

# Convert the frequencies to percentage
pct_df = freq_df.divide(freq_df.sum(axis=1), axis=0)*100

# Get traces
data = get_traces(freq_df, pct_df, stacked=True, orientation='v', textposition='auto', opacity=0.9)

# Define the layout for plotly figure
layout = go.Layout(
     autosize=False,
     width=900,
     height=600,
     barmode='stack',   
     margin=go.layout.Margin(
                            l=50,
                            r=0,
                            b=100,
                            t=50,),
    title='Country-wise gender distribution',
    xaxis=dict(title='Country'),
    yaxis=dict(title='Percentage'),
)

# Visualize
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)
# Select the column corresponding to Q4, the education level
edu_df = multiChoiceResp['Q4'][1:].dropna()

# Get the value counts
edu_df_counts = edu_df.value_counts()

# Get the labels and the corresponding counts 
counts = edu_df_counts.values
labels = edu_df_counts.index


# Function to show percentage in pie plot
def show_autopct(values):
    def my_autopct(pct):
        total = len(edu_df)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%'.format(p=pct)
    return my_autopct


# Plot a pie chart showing level of formal education
plt.figure(figsize=(10,8))
patches, text, autotext = plt.pie(counts, labels=labels, autopct=show_autopct(counts), )
plt.title("Highest level of formal education", fontsize=16)
plt.show()

# Delete variables that aren't going to be used further in order to save memory
del counts,labels, patches, text, autotext, edu_df_counts
# Select column Q6
role_df = multiChoiceResp['Q6'][1:].dropna()

# Get the value counts
role_df_counts = role_df.value_counts()

# Perecntage values
role_df_pct = (round(role_df.value_counts(normalize=True)*100,1)).values
role_df_pct = [str(x)+'%' for x in role_df_pct]

# Visualize
trace0 = go.Bar(
                x=role_df_counts.values,
                y=role_df_counts.index,
                orientation='h',
                text = role_df_pct,
                textposition='outside',
                marker=dict(
                color='rgb(200,2,55)',
                line=dict(color='rgb(125,75,55)',
                         )),
                opacity=0.7
               )

layout = go.Layout(title='<b>Job titles</b>',
                  autosize=False,
                  width=1000,
                  height=600,
                  xaxis=dict(title='Count'),
                  margin=go.layout.Margin(
                                        l=200,
                                        r=50,
                                        b=100,
                                        t=100, pad=10)
                   )

fig = go.Figure(data=[trace0], 
                layout=layout)
iplot(fig, show_link=False)
# Select the column for the corresponding question
exp_df = multiChoiceResp['Q8'][1:].dropna()

f,ax=plt.subplots(figsize=(10,5))

# Order in which we want the plot to ensure readability 
order= ['0-1', '1-2', '2-3', '3-4', '4-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30+']


# Do countplot 
ax=sns.countplot(exp_df,order=order, orient='v', color=color[4])

# Plot percentage on the bars
pct_on_bars(ax, exp_df, orientation='v', offset=50,adjustment=2)

    
plt.xlabel('Experience (in years)')
plt.ylabel('Respondents count')
plt.title('Years of experience in current role')
plt.show()
# Select the columns we are insterested in
df = multiChoiceResp[['Q6', 'Q8']][1:].dropna()

# Select top roles to consider
roles_to_consider = ['Data Scientist', 'Data Analyst', 'Software Engineer', 'Research Scientist'] 
                   
# Select only those rows that are of our interest
df = df[df['Q6'].isin(roles_to_consider)]

# Map the column values to numeric values because sorting the columsn then would become easy
col_dict = dict([(x,i)  for i,x in enumerate(order)])
df['Q8'] = df['Q8'].replace(col_dict)

# Group by roles and get the count for experience 
freq_df = df.groupby(['Q8'])['Q6'].value_counts().unstack()

# Convert the frequencies to percentage
pct_df = freq_df.divide(freq_df.sum(axis=1), axis=0)*100

# Do a percentage plot
f,ax=plt.subplots(figsize=(15,6))
ax=pct_df.plot(kind='bar', stacked=True, ax=ax, rot=30, use_index=True);

# Use our handy dandy function to plot percentage on the stacked bars
pct_on_stacked_bars(axs=ax,orientation='v',values=pct_df.values)

# Map the xticks back to original column values 
ax.set_xticklabels(order)

plt.xlabel('Experience in years');
plt.ylabel('Percentage')
plt.title('How much of experience respondents are having in top roles?')
plt.legend(loc=(1.02,0.5))
plt.show()
# Define a function to clean data and convert the salaries to desired type
def cleanup_salary(salary):
    '''The salaries are represented as 0-10,000 10-20,000...We will clean up this column 
       and convert the salary to numeric. Also for a range we will just take the upper value 
       as the representative salary. For example, if the salary in range 0-10,000 we will consider 
       the salary to be 10,000 as the representative. 
     '''
    
    # Replace the unwanted characters. PAY ATTENTION TO THE CHAINING
    salary = salary.str.replace(',', '').str.replace('+','')
    
    # Split the salaries on '-' and choose the last value
    # P.S: See the pandas usage here. This is the most elegant way to do such thigs in pandas
    salary = salary.str.split('-').str[-1]
    
    # Convert to numeric type
    salary = salary.astype(np.float64)
    
    return salary
# Select the desired columns
salary_df = multiChoiceResp[['Q6', 'Q8', 'Q9']][1:].dropna()

# Remove all rows where the respondent declined to disclose the salary
salary_df = salary_df[~(salary_df['Q9']=='I do not wish to disclose my approximate yearly compensation')]

# Clean the salary column
salary_df['Q9'] = cleanup_salary(salary_df['Q9'])

# Order in which we want the plot to ensure readability 
order= ['0-1', '1-2', '2-3', '3-4', '4-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30+']

# A list to store box plot data for each category
salary_data= []

# Create a box plot trace
for exp in reversed(order):
    salary_data.append(go.Box(x=salary_df[salary_df['Q8'] == exp]['Q9'], name=exp, orientation='h'))

layout = go.Layout(title='Salary distribution w.r.t years of experience',
                   autosize=False,
                   height=500,
                   width=1000,
                   xaxis=dict(title='Salary'),
                   yaxis=dict(title='Experience(in years)',),)

fig = go.Figure(data=salary_data, layout=layout)
# Visualize
iplot(fig, show_link=False)

del salary_data
# Select only those data points that are of interest
salary_df = salary_df[salary_df['Q6'].isin(roles_to_consider)] # roles_to_consider is defined in previous section

# A list to store box plot data for each category
salary_data= []

# Create a box plot trace
for role in roles_to_consider:
    salary_data.append(go.Box(y=salary_df[salary_df['Q6']==role]['Q9'], name=role, orientation='v'))

layout = go.Layout(title='Salary distribution w.r.t current role',
                   xaxis=dict(title='Role'),
                   yaxis=dict(title='Salary',),)

fig = go.Figure(data=salary_data, layout=layout)
# Visualize
iplot(fig, show_link=False)
# Select the question 
bus_df = multiChoiceResp['Q10'][1:].dropna()

# Check percentage
bus_df_counts = round(bus_df.value_counts(normalize=True)*100)
bus_df_counts = bus_df_counts.to_dict()

for k,v in bus_df_counts.items():
    print(k.ljust(100), v,"%")
# Select the desired columns
prog_df = multiChoiceResp['Q17'][1:].dropna()
order = prog_df.value_counts().index

f,ax=plt.subplots(figsize=(15,7))

# Do countplot 
ax=sns.countplot(y=prog_df, orient='h', order=order)

# plot the percentage also
pct_on_bars(ax, prog_df, orientation='h', offset=170, adjustment=3)
    
plt.ylabel('Programming language')
plt.xlabel('Count')
plt.title('Which programming language is preferred how much?')
plt.show()
# Select the columns corresponding to current role(Q6) and programming language(Q17)
prog_df = multiChoiceResp[['Q6', 'Q17']][1:].dropna()

# Filter only top four roles and top 10 programming languages for brevity 
prog_df = prog_df[prog_df['Q6'].isin(roles_to_consider)]
prog_to_consider = prog_df['Q17'].value_counts().index[:10]
prog_df = prog_df[prog_df['Q17'].isin(prog_to_consider)]


#col_dict = dict([(x,i)  for i,x in enumerate(prog_to_consider)])
#prog_df['Q17'] = prog_df['Q17'].replace(col_dict)

# Group by roles and get the count for experience 
freq_df = prog_df.groupby(['Q17'])['Q6'].value_counts().unstack()

# Convert the frequencies to percentage
pct_df = freq_df.divide(freq_df.sum(axis=1), axis=0)*100

# Get traces
data = get_traces(freq_df, pct_df, stacked=True, orientation='v', textposition='auto', opacity=0.7)

# Define the layout for plotly figure
layout = go.Layout(
     autosize=False,
     width=1000,
     height=600,
     barmode='stack',   
     margin=go.layout.Margin(
                            l=50,
                            r=0,
                            b=100,
                            t=50,),
    title='<b>Programming language preference in top roles</b>',
    xaxis=dict(title='Programming language'),
    yaxis=dict(title='Percentage'),
)

# Visualize
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)
# Select the desired columns
prog_df = multiChoiceResp['Q18'][1:].dropna()

# Consider only top 5 languages
order = prog_df.value_counts().index[:5]

f,ax=plt.subplots(figsize=(10,5))

# Do countplot 
ax=sns.countplot(x=prog_df, order=order)

# Plot percentage on the bars
pct_on_bars(ax, prog_df, orientation='v', offset=50, adjustment=2)
    
plt.ylabel('Programming language')
plt.xlabel('Count')
plt.title('Which programming language should you learn first?')
plt.show()
# Pick the data we are interested in
libraries_df = multiChoiceResp['Q20'][1:].dropna()

f,ax=plt.subplots(figsize=(15,5))

# Order in which we want the plot to ensure readability 
order= libraries_df.value_counts().index


# Do countplot 
ax=sns.countplot(y=libraries_df,order=order, orient='h', color=color[2])

# Plot percentage on the bars
pct_on_bars(ax, libraries_df, orientation='h', offset=150, adjustment=3)
    
plt.ylabel('Library/Framework')
plt.xlabel('Count')
plt.title('Which library is used by how much?')
plt.show()
# Select the column corresponding to Q22
vis_df = multiChoiceResp['Q22'][1:].dropna()

f,ax=plt.subplots(figsize=(15,5))

# Order in which we want the plot to ensure readability 
order= vis_df.value_counts().index


# Do countplot 
ax=sns.countplot(y=vis_df,order=order)

# Plot percentage on the bars
pct_on_bars(ax, vis_df, orientation='h', offset=150, adjustment=3)
    
plt.ylabel('Visualization Library/tool')
plt.xlabel('Count')
plt.title('Which libraries/tools are most popular for visualizations?')
plt.show()
# Select all the columns that we are interested in
coding_df = multiChoiceResp[['Q6','Q23']][1:].dropna()

# Some cleanup
coding_df['Q23'] = coding_df['Q23'].str.replace("of my time", '').str.replace(" to ", '-').str.strip()

f,axes=plt.subplots(2,1, figsize=(10,10))

# Order in which we want the plot to ensure readability 
order= ['0%', '1%-25%', '25%-49%', '50%-74%', '75%-99%', '100%']


# Do countplot 
ax=sns.countplot(x='Q23', data=coding_df,order=order, color=color[1], ax=axes[0])

# Plot percentage on the bars
pct_on_bars(ax, vis_df, orientation='v', offset=50, adjustment=3)
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Time spent coding')

# Another interesting thing to do
roles = ['Data Scientist', 'Data Analyst', 'Research Scientist']
coding_df = coding_df[coding_df['Q6'].isin(roles)]
ax=sns.countplot(x='Q23', data=coding_df,order=order, ax=axes[1], hue='Q6')

# Get all the bars and plot the percentage also
bars = ax.patches
sections = len(bars)//3
first_bar = bars[:sections]
second_bar = bars[sections:len(first_bar)+sections]
third_bar = bars[len(second_bar)+sections:]

# Loop over the bars and put text on each bar
for left, middle, right in zip(first_bar, second_bar, third_bar):
        height_l = left.get_height()
        height_m = middle.get_height()
        height_r = right.get_height()
    
        
        if math.isnan(height_l):
            height_l=0
        if math.isnan(height_m):
            height_m=0
        if math.isnan(height_r):
            height_r=0
        
        total = height_l + height_m + height_r

        ax.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.1%}'.format(height_l/total), ha="center", fontsize=7)
        ax.text(middle.get_x() + middle.get_width()/2., height_m + 40, '{0:.1%}'.format(height_m/total), ha="center",fontsize=7)
        ax.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.1%}'.format(height_r/total), ha="center",fontsize=7)


axes[1].set_ylabel('Count')
axes[1].set_xlabel('Time spent coding')

plt.suptitle('How much do data science people code?')
plt.show()
# In order to support my above argument, we will plot profiles and their education level
# Select rows from educational dataframe corresponding to the current indices
edu_level = edu_df[edu_df.index.isin(coding_df.index)]

# Create a new column
coding_df['Q4'] = edu_level.values

# Group by roles and get the count for experience 
freq_df = coding_df.groupby(['Q4'])['Q6'].value_counts().unstack()

# Convert the frequencies to percentage
pct_df = freq_df.divide(freq_df.sum(axis=1), axis=0)*100

# Get traces
data = get_traces(freq_df, pct_df, stacked=True, orientation='h', textposition='outside', opacity=0.7)

# Define the layout for plotly figure
layout = go.Layout(
     autosize=False,
     width=1300,
     height=500,
     barmode='stack',   
     margin=go.layout.Margin(
                            l=400,
                            r=0,
                            b=100,
                            t=50,),
    title='How much do data science people code?',
    yaxis=dict(title='Level of Education'),
    xaxis=dict(title='Percentage'),
    
)

# Visualize
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)
# Selcting the columns of interest
eda_df = multiChoiceResp[['Q6', 'Q23', 'Q24']][1:].dropna()

# Get the value counts
eda_df_counts = eda_df['Q24'].value_counts()

# Perecntage values
eda_df_pct = (round(eda_df['Q24'].value_counts(normalize=True)*100,1)).values
eda_df_pct = [str(x)+'%' for x in eda_df_pct]

# Visualize
trace0 = go.Bar(
                y=eda_df_counts.values,
                x=eda_df_counts.index,
                orientation='v',
                text = eda_df_pct,
                textposition = 'outside',
                marker=dict(
                color='rgb(100,223,225)',
                line=dict(color='rgb(5,4,150)',width=2.0,
                         )),
                opacity=0.5
               )

layout = go.Layout(title='<b>How long have people been writing code to analyze data?</b>',
                  autosize=False,
                  width=800,
                  height=500,
                  margin=go.layout.Margin(
                                        l=50,
                                        r=200,
                                        b=200,
                                        t=50,),
                  yaxis=dict(title='Count'),
                
                )

fig = go.Figure(data=[trace0], 
                layout=layout)
iplot(fig, filename='eda_df-hover-bar', show_link=False)
# Selecting only top profiles only
roles = ['Data Scientist', 'Data Analyst', 'Research Scientist']
years_to_consider = ['< 1 year', '1-2 years', '3-5 years', '5-10 years', '10-20 years']
eda_df = eda_df[eda_df['Q6'].isin(roles) & eda_df['Q24'].isin(years_to_consider)]

# Groupby and get the normalize counts
df = eda_df.groupby('Q24')['Q6'].value_counts().unstack()

# Reindex in the order we want
df = df.reindex(reversed(years_to_consider))

# Get the percentage values for each bar in each group on X-axis
freq_df = np.round(df.divide(df.sum(axis=0), axis=1)*100)

# Get traces
data = get_traces(df, freq_df, stacked=False, orientation='h', textposition='outside', opacity=0.6)

# Define the layout for plotly figure
layout = go.Layout(
    title='Who among DS,DA and RS have been writing code to analyze data for how long?',
    xaxis=dict(
        title='Count',
        titlefont=dict(
            family='Times New Roman',
            size=18
        )
    ),
    yaxis=dict(
        title='Years',
        titlefont=dict(
            family='Times New Roman',
            size=18,
            )
    )
)

# Visualize
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)
# Select the question column
ml_df = multiChoiceResp['Q25'][1:].dropna()

# Some cleansing
ml_df = ml_df.str.replace('I have never studied machine learning but plan to learn in the future', 
                          'Never studied but planned')
ml_df = ml_df.str.replace('I have never studied machine learning and I do not plan to', 
                          'Never studied and no plan')

# order in which we want to display the categoricals
order = ['< 1 year', '1-2 years', '2-3 years', '3-4 years', 
         '4-5 years', '5-10 years', '10-15 years', '20+years',
         'Never studied but planned', 'Never studied and no plans']

# visualize
f,ax=plt.subplots(figsize=(12,5))
ax=sns.countplot(y=ml_df,order=order, color=color[3])

# Plot percentage on the bars
pct_on_bars(ax, ml_df, orientation='h', offset=170, adjustment=2)
    
plt.ylabel('')
plt.xlabel('Count')
plt.title('For how many years people have used ML methods at work/school?')
plt.show()
# Select the question column
ds_df = multiChoiceResp['Q26'][1:].dropna()

# Get the counts
ds_df_counts = ds_df.value_counts()

# visualize
f,ax=plt.subplots(figsize=(10,5))
ax=sns.countplot(x=ds_df, color=color[6], order=ds_df_counts.index)

# Plot percentage on the bars
pct_on_bars(ax, ds_df, orientation='v', offset=70, adjustment=2, prec=1)
    
plt.ylabel('Count')
plt.xlabel('Response')
plt.title('Do you consider yourself to be a data scientist?')
plt.xticks(rotation=30)
plt.show()
# Select the question column
ds_df = multiChoiceResp[['Q3', 'Q26']][1:].dropna()

# select top 5 countries
countries_to_consider = top10_countries[:5]

# Some cleansing
ds_df = ds_df[ds_df['Q3'].isin(countries_to_consider)]
ds_df['Q3'] = ds_df['Q3'].replace(['United States of America', 'United Kingdom of Great Britain and Northern Ireland'], 
                            ['USA', 'UK & Northern Ireland'])

# Groupby and get the counts
df = ds_df.groupby('Q26')['Q3'].value_counts().unstack()

# Get the percentage values for each bar in each group on X-axis
freq_df = np.round(df.divide(df.sum(axis=0), axis=1)*100)

# Get traces
data = get_traces(df, freq_df, stacked=False, orientation='v', textposition='outside', opacity=0.8)

# Define the layout for plotly figure
layout = go.Layout(
    title='<b>Do you consider yourself to be a Data Scientist?</b>',
    xaxis=dict(
        title='Response',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Count',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

# Visualize
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='data-scientist-in-top-countries', show_link=False)
# Select the question column
dataset_df = multiChoiceResp['Q32'][1:].dropna()

# Get the counts
dataset_df_counts = dataset_df.value_counts()

# visualize
f,ax=plt.subplots(figsize=(10,5))
ax=sns.countplot(x=dataset_df, order=dataset_df_counts.index)

# Plot percentage on the bars
pct_on_bars(ax, dataset_df, orientation='v', offset=50, adjustment=2, prec=1)
    
plt.ylabel('Count')
plt.xlabel('Dataset type')
plt.title('What is the type of data that you currently interact with most often at work or school?')
plt.xticks(rotation=60)
plt.show()
# Select the question column
platform_df = multiChoiceResp['Q37'][1:].dropna()

# Get the counts
platform_df_counts = platform_df.value_counts()

# visualize
f,ax=plt.subplots(figsize=(15,5))
ax=sns.countplot(y=platform_df, order=platform_df_counts.index, color=color[2])

# Plot percentage on the bars
pct_on_bars(ax, platform_df, orientation='h', offset=80, adjustment=2, prec=1)
    
plt.xlabel('Count')
plt.ylabel('Platform')
plt.title('On which online platform have you spent the most amount of time?')
plt.show()
# Selcting the columns of interest
expertise_df = multiChoiceResp[['Q4', 'Q40']][1:].dropna()

# Get the value counts
expertise_df_counts = expertise_df['Q40'].value_counts()

# Perecntage values
expertise_df_pct = (round(expertise_df['Q40'].value_counts(normalize=True)*100,1)).values
expertise_df_pct = [str(np.round(x).astype(int))+'%' for x in expertise_df_pct]

# Visualize
trace0 = go.Bar(
                x=expertise_df_counts.values,
                y=expertise_df_counts.index,
                orientation='h',
                text = expertise_df_pct,
                textposition = 'outside',
                marker=dict(
                color='rgb(300,100,25)',
                line=dict(color='rgb(5,4,150)',width=2,
                         )),
                opacity=0.6
               )

layout = go.Layout(title='<b>Which better demonstrates expertise in Data Science?</b>',
                  autosize=False,
                  width=900,
                  height=500,
                  margin=go.layout.Margin(
                                        l=450,
                                        r=0,
                                        b=100,
                                        t=100,),
                   xaxis=dict(title='Count')
                  )

fig = go.Figure(data=[trace0], 
                layout=layout)
iplot(fig, show_link=False)
# We will consider only three levels of education here
edu_level_to_consider=["Doctoral degree", "Master’s degree", "Bachelor’s degree"]
expertise_df = expertise_df[expertise_df['Q4'].isin(edu_level_to_consider)]

# Groupby and get the value counts
df = expertise_df.groupby('Q40')['Q4'].value_counts().unstack()

# Get the percentage values for each bar in each group on X-axis
freq_df = np.round(df.divide(df.sum(axis=1), axis=0)*100)

# Get traces
data = get_traces(df, freq_df, stacked=False, orientation='h', textposition='outside', opacity=0.6)

# Define the layout for plotly figure
layout = go.Layout(
    title='Which better demonstrates expertise in Data Science?',
    xaxis=dict(title='Response'),
    margin=go.layout.Margin(
                            l=450,
                            r=0,
                            b=100,
                            t=50,),
    
)

# Visualize
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)
# Get the percentage values for each bar in each group on X-axis
freq_df = np.round(df.divide(df.sum(axis=0), axis=1)*100)

# Get traces
data = get_traces(df, freq_df, stacked=False, orientation='h', textposition='outside', opacity=0.6)

# Define the layout for plotly figure
layout = go.Layout(
    title='Which better demonstrates expertise in Data Science?',
    xaxis=dict(title='Response'),
    margin=go.layout.Margin(
                            l=450,
                            r=0,
                            b=100,
                            t=50,),
    
)

# Visualize
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)
# Select the question column
bias_df = multiChoiceResp['Q43'][1:].dropna()

# Get the counts
bias_df_counts = bias_df.value_counts()

# visualize
f,ax=plt.subplots(figsize=(10,7))
ax=sns.countplot(x=bias_df, order=sorted(bias_df_counts.index), color=color[4])

# Plot percentage on the bars
pct_on_bars(ax, bias_df, orientation='v', offset=50, adjustment=2, prec=0)
    
plt.ylabel('Count')
plt.xlabel('Time spent(%)')
plt.title('What percent of your data projects involved exploring unfair bias in the dataset and/or algorithm?')
plt.show()
# Select the question column
bias_df = multiChoiceResp[['Q6','Q43']][1:].dropna()

# We will consider only three roles here : Data Analyst, Data Scientist, Research Scientist
bias_df = bias_df[bias_df['Q6'].isin(roles)]

# Get the frequency counts for each category
freq_df = bias_df.groupby(['Q43'])['Q6'].value_counts().unstack()

# Get the percentages across each category for each role
pct_df = freq_df.divide(freq_df.sum(axis=1), axis=0)*100

# Get traces
data = get_traces(freq_df, pct_df, stacked=True, orientation='v', textposition='auto', opacity=0.7)

# Define the layout for plotly figure
layout = go.Layout(
     autosize=False,
     width=900,
     height=500,
     barmode='stack',   
     margin=go.layout.Margin(
                            l=50,
                            r=0,
                            b=50,
                            t=50,),
    title='What % of your data projects involved exploring unfair bias in the dataset and/or algorithm?',
    xaxis=dict(title='Time(%)'),
    yaxis=dict(title='Percentage'),
)

# Visualize
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)
# Select the question column
model_insights_df = multiChoiceResp['Q46'][1:].dropna()

# Get the counts
model_insights_df_counts = model_insights_df.value_counts()

# visualize
f,ax=plt.subplots(figsize=(10,7))
ax=sns.countplot(x=model_insights_df, order=sorted(model_insights_df_counts.index), color=color[2])

# Plot percentage on the bars
pct_on_bars(ax, model_insights_df, orientation='v', offset=30, adjustment=2, prec=0)
    
plt.ylabel('Count')
plt.xlabel('Time spent(%)')
plt.title('Approximately what percent of your data projects involve exploring model insights?')
plt.show()
# Select the column of interest
df = multiChoiceResp['Q48'][1:].dropna()

# Get a frequency count
freq_counts = df.value_counts()

# Define a trace for the plotly pie chart
trace = go.Pie(labels=freq_counts.index, values=freq_counts.values,
               hoverinfo='label+value', 
               textfont=dict(size=20),
               marker=dict( line=dict(color='#000000', width=2)))

# Define a layout for the figure on which you are going to plot your data
layout = go.Layout(
            autosize=True,
            margin=go.layout.Margin(
                            l=10,
                            r=100,
                            b=10,
                            t=50,),
            legend=dict(
                x=1,
                y=1,
                traceorder='normal',
                font=dict(
                    family='sans-serif',
                    size=12,
                    color='#000'
                ),
                bgcolor='#E2E2E2',
                bordercolor='#FFFFFF',
                borderwidth=2
            )
)    

# Define the figure object
fig = go.Figure(data=[trace], layout=layout)

# Plot
iplot(fig, show_link=False)
def divide_categories(x):
    if x in ['I am confident that I can understand and explain the outputs of many but not all ML models',
            'I am confident that I can explain the outputs of most if not all ML models']:
        x = 'Not black boxes'
    else:
        x = 'Black boxes'
    return x
# Select the question column
df = multiChoiceResp[['Q6','Q48']][1:].dropna()

# We will consider only three roles here : Data Analyst, Data Scientist, Research Scientist
df = df[df['Q6'].isin(roles)]

# Drop the rows where people didn't provide opinion on explainable ML models
df = df[~df['Q48'].isin(['I do not know; I have no opinion on the matter'])]

# divide into two categories
df['Q48'] = df['Q48'].apply(divide_categories)

# Get the frequency counts for each category
freq_df = df.groupby(['Q6'])['Q48'].value_counts().unstack()

# Get the percentages across each category for each role
pct_df = freq_df.divide(freq_df.sum(axis=1), axis=0)*100

# Get traces
data = get_traces(freq_df, pct_df, stacked=False, orientation='v', textposition='auto', opacity=0.8)

# Define the layout for plotly figure
layout = go.Layout(
     autosize=False,
     width=900,
     height=500,   
     margin=go.layout.Margin(
                            l=70,
                            r=0,
                            b=50,
                            t=50,),
    title='Do you consider ML models as black boxes?',
    xaxis=dict(title='Role'),
    yaxis=dict(title='Count'),
)

# Visualize
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)
## Checking all the columns that are split into multiple parts
cols = multiChoiceResp.columns.tolist()
ques = set()
#cols = [x for x in cols if 'Part' in x]
for col in cols:
    col = col.split('_')[0]
    ques.add(col)
print("Number of questions split into different parts ", len(ques))
# A handy danyd function to rename columns which are parts of a single question
def rename_columns(df, column_names, custom_name=None, index=None, exclude_others=True):
    column_names = column_names.apply(lambda x: x.split("-", 2)[-1]).tolist()
    column_names = [x.strip() for x in column_names]
    if custom_name:
        if index!= None:
            column_names[index] = custom_name
        else:
            return "Index value must be passed to use a custom name for a column"
    
    # Rename columns
    df.columns = column_names 
    
    # Whether to remove "None" and "Other" categories
    if exclude_others:
        columns_to_consider = [x for x in column_names if x not in ['None', 'Other']]
    else:
        columns_to_consider = column_names
    
    return columns_to_consider
# We will select the columns corresponding to roles and the question in which IDE pref was asked in the survey
df = multiChoiceResp[[x for x in cols if x=='Q6' or 'Q13_Part' in x]]

# Rename the columns and remove "None" and "Other"
column_names = df.iloc[0, :]
columns_to_consider = rename_columns(df, column_names, custom_name="Current Role", index=0, exclude_others=True)


df = df[columns_to_consider][1:] 

# Do a groupby on current role and get count for each column corresponding to each group
freq_df = df.groupby('Current Role')[columns_to_consider[1:]].count()
pct_df = np.round(freq_df.divide(freq_df.sum(axis=1), axis=0)*100)


# Sort the dataframe values
#pct_df.values.sort()

# values for annotations
pct_df_text = pct_df.applymap(lambda x: str(int(x)) + "%") 


# Define a annotated figure object
fig = ff.create_annotated_heatmap(z=pct_df.values.tolist(), 
                                  y=pct_df.index.tolist(), 
                                  x=pct_df.columns.tolist(), 
                                  annotation_text=pct_df_text.values.tolist(), 
                                  colorscale='Viridis')


# Define the layout for the fig created above
fig.layout.title = "Which IDE people prefer to work with?"
fig.layout.height = 800
fig.layout.width = 1000
fig.layout.margin.l = 200
fig.layout.margin.t = 150

# Visualize
iplot(fig,show_link=False)
df = multiChoiceResp[[x for x in cols  if 'Q14_Part' in x]]

# Rename the columns and remove "None" and "Other"
column_names = df.iloc[0, :]
columns_to_consider = rename_columns(df, column_names, exclude_others=True)

# select only relevant columns
df = df[columns_to_consider][1:]

# Get the count and plot them
df = df.count().sort_values()

# Get percentages
pct = np.round(df.divide(df.sum())*100,2).values
pct = [str(x) + "%" for x in pct]


# Visualize
trace = go.Bar(
                x=df.values,
                y=df.index,
                orientation='h',
                text = pct,
                textposition = 'outside',
                marker=dict(
                color='rgb(100,225,25)',
                line=dict(color='rgb(5,4,150)',width=2,
                         )),
                opacity=0.6
               )

layout = go.Layout(title='<b>Popular hosted notebooks</b>',
                  autosize=False,
                  width=900,
                  height=500,
                  margin=go.layout.Margin(
                                        l=150,
                                        r=0,
                                        b=100,
                                        t=50,),
                   xaxis=dict(title='Count')
                  )

fig = go.Figure(data=[trace], layout=layout)
iplot(fig, show_link=False)
df = multiChoiceResp[[x for x in cols  if 'Q33_Part' in x]]

# Rename the columns and remove "None" and "Other"
column_names = df.iloc[0, :]
columns_to_consider = rename_columns(df, column_names, exclude_others=True)

# select only relevant columns
df = df[columns_to_consider][1:]

# Get the count and plot them
df = df.count().sort_values()

# Define trace for plotly chart
trace = go.Pie(labels=df.index, values=df.values, showlegend=True)

# Layout of the fig
layout = go.Layout(title='Where do you find public datasets?',
        margin=go.layout.Margin(
                                l=10,
                                r=200,
                                b=10,
                                t=50,),
                legend=dict(
                    x=1,
                    y=1,
                    traceorder='normal',
                    font=dict(
                        family='sans-serif',
                        size=12,
                        color='#000'
                    ),
                    bgcolor='#E2E2E2',
                    bordercolor='#FFFFFF',
                    borderwidth=2
                )
    )

# Plot
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
df = multiChoiceResp[[x for x in cols  if 'Q34_Part' in x]]

# Rename the columns and remove "None" and "Other"
column_names = df.iloc[0, :]
columns_to_consider = rename_columns(df, column_names, exclude_others=False)

# select only relevant columns
df = df[columns_to_consider][1:].dropna().astype(np.float32)

# Get the median values for each column. The median value will be our representative value for that particular stage
mean_df = np.round(df.mean().sort_values(ascending=False))

# Get the phases and values(time spent) corresponding to each phase
values = mean_df.values.tolist()
phases = mean_df.index.tolist()

# Get the traces for plotly funnel chart
label_trace, value_trace, shapes = draw_funnel_chart(values, phases, colors=None, 
                                                     plot_width=50, section_d=10, section_h=70)


data = [label_trace, value_trace]

# Define layout
layout = go.Layout(
    title="Average time (%) spent on different stages in a typical Data Science project",
    titlefont=dict(
        size=15,
        color='rgb(203,203,203)'),
    margin=go.layout.Margin(l=50,
                            r=0,
                            b=10,
                            t=50,),
    shapes=shapes,
    height=500,
    width=1000,
    showlegend=False,
    hovermode='closest',
    paper_bgcolor='rgba(44,58,71,1)',
    plot_bgcolor='rgba(44,58,71,1)',
    xaxis=dict(
        showticklabels=False,
        zeroline=False,
    ),
    yaxis=dict(
        showticklabels=False,
        zeroline=False)
)
 
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)
df = multiChoiceResp[[x for x in cols  if 'Q29_Part' in x]]

# Rename the columns and remove "None" and "Other"
column_names = df.iloc[0, :]
columns_to_consider = rename_columns(df, column_names, exclude_others=True)

# select only relevant columns
df = df[columns_to_consider][1:]

# Get the count and plot them
df = df.count().sort_values()

# Get percentages
pct = np.round(df.divide(df.sum())*100,2).values
pct = [str(x) + "%" for x in pct]


# Visualize
trace = go.Bar(
                x=df.values,
                y=df.index,
                orientation='h',
                text = pct,
                textposition = 'auto',
                marker=dict(
                color='rgb(35,35,65)',
                line=dict(color='rgb(0,0,0)',width=2,
                         )),
                opacity=0.6
               )

layout = go.Layout(title='Databases used by people over last 5 years',
                  autosize=True,
                  width=1500,
                  height=800,
                  margin=go.layout.Margin(
                                        l=200,
                                        r=0,
                                        b=100,
                                        t=50, pad=10),
                   xaxis=dict(title='Count'),
                  )

fig = go.Figure(data=[trace], layout=layout)
iplot(fig, show_link=False)
df = multiChoiceResp[[x for x in cols  if 'Q28_Part' in x]]

# Rename the columns and remove "None" and "Other"
column_names = df.iloc[0, :]
columns_to_consider = rename_columns(df, column_names, exclude_others=True)

# select only relevant columns
df = df[columns_to_consider][1:]

# Get the count and plot them
df = df.count().sort_values()

# Get percentages
pct = np.round(df.divide(df.sum())*100,2).values
pct = [str(x) + "%" for x in pct]


# Visualize
trace = go.Bar(
                x=df.values,
                y=df.index,
                orientation='h',
                text = pct,
                textposition = 'outside',
                marker=dict(
                color='rgb(10,20,225)',
                line=dict(color='rgb(0,0,0)',width=2,
                         )),
                opacity=0.6
               )

layout = go.Layout(title='Machine Learning Products',
                  autosize=False,
                  width=1500,
                  height=800,
                  margin=go.layout.Margin(
                                        l=300,
                                        r=0,
                                        b=10,
                                        t=50,
                                        pad=10),
                   xaxis=dict(title='Count'),
                  )

fig = go.Figure(data=[trace], layout=layout)
iplot(fig, show_link=False)
df = multiChoiceResp[[x for x in cols  if 'Q50_Part' in x]]

# Rename the columns and remove "None" and "Other"
column_names = df.iloc[0, :]
columns_to_consider = rename_columns(df, column_names, exclude_others=True)

# select only relevant columns
df = df[columns_to_consider][1:]

# Get the count and plot them
df = df.count().sort_values()

# Define trace for plotly chart
trace = go.Pie(labels=df.index, 
               values=df.values, 
               showlegend=True,
               marker=dict(
                           line=dict(color='#000000', width=2))
              )

# Layout of the fig
layout = go.Layout(title='<b>What barriers prevent you from making your work even easier to reuse and reproduce?</b>',
                   margin=go.layout.Margin(
                                l=10,
                                r=200,
                                b=10,
                                t=100,),
                   legend=dict(
                                x=1,
                                y=1,
                                traceorder='normal',
                                borderwidth=5
                            )
                )

# Plot
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, show_link=False)
