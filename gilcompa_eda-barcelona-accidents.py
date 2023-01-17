# Import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from pyproj import Proj, transform
from datetime import datetime
from datetime import date, time
from dateutil.parser import parse

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Load dataframes with special encoding
df_2010 = pd.read_csv('../input/2010_accidents.csv', encoding='cp1252')
df_2011 = pd.read_csv('../input/2011_accidents.csv', encoding='cp1252')
df_2012 = pd.read_csv('../input/2012_accidents.csv', encoding='cp1252')
df_2013 = pd.read_csv('../input/2013_accidents.csv', encoding='cp1252')
#df_2014 = pd.read_csv('../input/2014_accidents.csv', encoding='latin1')
df_2015 = pd.read_csv('../input/2015_accidents.csv', encoding='cp1252')
#df_2016 = pd.read_csv('../input/2016_accidents.csv', encoding='iso-8859-1')

# Concatenate dataframes
frames = [df_2010, df_2011, df_2012, df_2013, df_2015]
df = pd.concat(frames)
# Shape&Info 
print('Shape:',df.shape)
print('Columns:', df.columns)

# Plot helper functions
def value_barplot_label(plot_name):
    # Inserts the value label on the top of each bar.

    for p in plot_name.patches:
        height = p.get_height()
        plot_name.text(p.get_x()+p.get_width()/2., height,'{:0.0f}'.format(height), ha="center").set_weight('bold')

def tick_format(plot,tick_type):
    if tick_type == 1:
        for label in plot.get_xticklabels():
            label.set_fontsize(16)
            label.set_rotation(90) 
        for label in plot.get_yticklabels():
            label.set_fontsize(14)
    
    if tick_type == 2:
        for label in plot.get_xticklabels():
            label.set_fontsize(16)
        for label in plot.get_yticklabels():
            label.set_fontsize(14)
    
    if tick_type == 3:
        for label in plot.get_xticklabels():
            label.set_fontsize(16)
        for label in plot.get_yticklabels():
            label.set_fontsize(14)
            label.set_rotation(20)
typeofvehicle = df['Desc. Tipus vehicle implicat'].value_counts()
pedestriancause = df['Descripció causa vianant'].value_counts()
weekday = df['Descripció dia setmana'].value_counts()
daytype = df['Descripció tipus dia'].value_counts()
shifttype = df['Descripció torn'].value_counts()
severity = df['Descripció victimització'].value_counts()
sex = df['Descripció sexe'].value_counts()
persontype = df['Descripció tipus persona'].value_counts()

# Visualize 
fig = plt.figure(figsize=(30,40))
ax1 = fig.add_subplot(421)
ax1 = sns.barplot(x=typeofvehicle, 
                 y=typeofvehicle.index, 
                 palette=sns.cubehelix_palette(reverse=True, n_colors=len(typeofvehicle.index)),
                 ax=ax1)
plt.title("Type of vechicle", fontsize=30).set_weight('bold')
tick_format(ax1, 3)
ax1.set_ylabel('Type', fontsize=16).set_weight('bold')
ax1.set_xlabel('# of accidents', fontsize=16).set_weight('bold')

ax2 = fig.add_subplot(422)
ax2 = sns.barplot(x=pedestriancause, 
                 y=pedestriancause.index, 
                 palette=sns.cubehelix_palette(reverse=True, n_colors=len(pedestriancause.index)),
                 ax=ax2)
plt.title("Pedestrian cause", fontsize=30).set_weight('bold')
tick_format(ax2, 3)
ax2.set_xlabel('# of accidents', fontsize=16).set_weight('bold')

# Visualize the accident group
ax3 = fig.add_subplot(423)
ax3 = sns.barplot(x=weekday, 
                 y=weekday.index, 
                 palette=sns.cubehelix_palette(reverse=True, n_colors=len(weekday.index)),
                 ax=ax3)
plt.title("Week day", fontsize=30).set_weight('bold')
tick_format(ax3, 3)
ax3.set_ylabel('Condition', fontsize=16).set_weight('bold')
ax3.set_xlabel('# of accidents', fontsize=16).set_weight('bold')

# Visualize the accident group
ax4 = fig.add_subplot(424)
ax4 = sns.barplot(x=daytype, 
                 y=daytype.index, 
                 palette=sns.cubehelix_palette(reverse=True, n_colors=len(daytype.index)),
                 ax=ax4)
plt.title("Type of road", fontsize=30).set_weight('bold')
tick_format(ax4, 3)
ax4.set_ylabel('Condition', fontsize=16).set_weight('bold')
ax4.set_xlabel('# of accidents', fontsize=16).set_weight('bold')

ax5 = fig.add_subplot(425)
ax5 = sns.barplot(x=shifttype, 
                 y=shifttype.index, 
                 palette=sns.cubehelix_palette(reverse=True, n_colors=len(shifttype.index)),
                 ax=ax5)
plt.title("Type of vechicle", fontsize=30).set_weight('bold')
tick_format(ax5, 3)
ax5.set_ylabel('Type', fontsize=16).set_weight('bold')
ax5.set_xlabel('# of accidents', fontsize=16).set_weight('bold')

ax6 = fig.add_subplot(426)
ax6 = sns.barplot(x=severity, 
                 y=severity.index, 
                 palette=sns.cubehelix_palette(reverse=True, n_colors=len(severity.index)),
                 ax=ax6)
plt.title("Pedestrian cause", fontsize=30).set_weight('bold')
tick_format(ax6, 3)
ax6.set_xlabel('# of accidents', fontsize=16).set_weight('bold')

# Visualize the accident group
ax7 = fig.add_subplot(427)
ax7 = sns.barplot(x=sex, 
                 y=sex.index, 
                 palette=sns.cubehelix_palette(reverse=True, n_colors=len(sex.index)),
                 ax=ax7)
plt.title("Week day", fontsize=30).set_weight('bold')
tick_format(ax7, 3)
ax7.set_ylabel('Condition', fontsize=16).set_weight('bold')
ax7.set_xlabel('# of accidents', fontsize=16).set_weight('bold')

# Visualize the accident group
ax8 = fig.add_subplot(428)
ax8 = sns.barplot(x=persontype, 
                 y=persontype.index, 
                 palette=sns.cubehelix_palette(reverse=True, n_colors=len(persontype.index)),
                 ax=ax8)
plt.title("Type of road", fontsize=30).set_weight('bold')
tick_format(ax8, 3)
ax8.set_ylabel('Condition', fontsize=16).set_weight('bold')
ax8.set_xlabel('# of accidents', fontsize=16).set_weight('bold')
# District names
district_names = df['Nom districte'].value_counts().index
district_names
# Neighbourhood names
neighbourhood_names = df['Nom barri'].value_counts().index
neighbourhood_names
# Visualize how many universitie sper country are there in the best 200 ranking
plt.figure(figsize=(10,6))
ax = sns.barplot(x=df['NK Any'].value_counts().index, y=df['NK Any'].value_counts())
plt.title("Number of accidents per year", fontsize=20).set_weight('bold')
value_barplot_label(ax)
ax.set_ylabel('# Accidents', fontsize=16).set_weight('bold')
ax.set_xlabel('Year', fontsize=16).set_weight('bold')
tick_format(ax,2)
# Visualization
plt.figure(figsize=(13,6))
x = df['Nom districte'].value_counts().index
y = df['Nom districte'].value_counts()
ax = sns.barplot(x=x, y=y)
plt.title("The 10 districts with more accidents", fontsize=20).set_weight('bold')
value_barplot_label(ax)
ax.set_ylabel('# Accidents', fontsize=16).set_weight('bold')
ax.set_xlabel('District', fontsize=16).set_weight('bold')
tick_format(ax,1)
# Visualization
plt.figure(figsize=(10,6))
x = df['Nom barri'].value_counts().head(10).index
y = df['Nom barri'].value_counts().head(10)
ax = sns.barplot(x=x, y=y)
plt.title("The 10 neighbourhoods with more accidents", fontsize=20).set_weight('bold')
value_barplot_label(ax)
ax.set_ylabel('# Accidents', fontsize=16).set_weight('bold')
ax.set_xlabel('Neighbourhood', fontsize=16).set_weight('bold')
tick_format(ax,1)
# Change the unknowns values to -5
#df['Edat'] = df['Edat'].apply(pd.to_numeric).astype(int)
df.Edat.replace('Desconegut', '-5', inplace=True)
df.Edat = df.Edat.apply(pd.to_numeric).astype(int)

# Visualization
plt.figure(figsize=(10,6))
ax = sns.distplot(df.Edat, kde=False, color="g")
plt.title("Age distribution", fontsize=20).set_weight('bold')
ax.set_ylabel('# Accidents', fontsize=16).set_weight('bold')
ax.set_xlabel('Age', fontsize=16).set_weight('bold')
tick_format(ax,1)
df.columns
time_features = ['Descripció dia setmana','Descripció tipus dia', 'Descripció torn',
                 'Dia de mes', 'Dia setmana','Hora de dia', 'Mes de any', 'NK Any']
df[time_features].head(3)
df['Mes de any'].value_counts().index
sns.set(font_scale=1.4)
heat_data=df.groupby(['Descripció dia setmana', 'Hora de dia'])['NK Any'].count().to_frame().unstack()
heat_data.columns = heat_data.columns.droplevel()
heat_data = heat_data.reindex(index = ['Dilluns', 'Dimarts', 'Dimecres', 'Dijous',
                                        'Divendres', 'Dissabte', 'Diumenge'])
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
plt.figure(figsize=(15,5))
sns.heatmap(heat_data,linewidths=.2,cmap=cmap)
plt.title('Accidents per dia de la setmana i hora', fontsize=25)
sns.set(font_scale=1.4)
heat_data=df.groupby(['Mes de any', 'Hora de dia'])['NK Any'].count().to_frame().unstack()
heat_data.columns = heat_data.columns.droplevel()
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
plt.figure(figsize=(15,5))
sns.heatmap(heat_data,linewidths=.2,cmap=cmap)
plt.title('Accidents per mes de l any i hora', fontsize=25)

