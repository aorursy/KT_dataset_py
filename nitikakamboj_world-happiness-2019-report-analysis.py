# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
from matplotlib.patches import Rectangle

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_2019=pd.read_csv('/kaggle/input/world-happiness/2019.csv')
df_2019.info()
df_2019.head()
df_2019['Country or region'].nunique()
def plot_hist_2019(column):
    perc_25_colour = 'gold'
    perc_50_colour = 'mediumaquamarine'
    perc_75_colour = 'deepskyblue'
    perc_95_colour = 'peachpuff'
    
    fig, ax = plt.subplots(figsize=(8,8))
    counts,bins,patches=ax.hist(df_2019[column],facecolor=perc_50_colour,edgecolor='gray')
    ax.set_xticks(bins.round(2))
    plt.xticks(rotation=70)
    twentyfifth, seventyfifth, ninetyfifth = np.percentile(df_2019[column], [25, 75, 95])
    for patch, leftside, rightside in zip(patches, bins[:-1], bins[1:]):
        if rightside < twentyfifth:
            patch.set_facecolor(perc_25_colour)
        elif leftside > ninetyfifth:
            patch.set_facecolor(perc_95_colour)
        elif leftside > seventyfifth:
            patch.set_facecolor(perc_75_colour)
    bin_x_centers = 0.5 * np.diff(bins) + bins[:-1]
    bin_y_centers = ax.get_yticks()[1] * 0.25
    for i in range(len(bins)-1):
        bin_label = "{0:,}".format(counts[i]) + "  ({0:,.2f}%)".format((counts[i]/counts.sum())*100)
        plt.text(bin_x_centers[i], bin_y_centers, bin_label, rotation=90, rotation_mode='anchor')
        
    ax.annotate('Each bar shows count and percentage of total',
            xy=(.85,.30), xycoords='figure fraction',
            horizontalalignment='center', verticalalignment='bottom',
            fontsize=10, bbox=dict(boxstyle="round", fc="white"),
            rotation=-90)

    handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [perc_25_colour, perc_50_colour, perc_75_colour, perc_95_colour]]
    labels= ["0-25 Percentile","25-50 Percentile", "50-75 Percentile", ">95 Percentile"]
    plt.legend(handles, labels, bbox_to_anchor=(0.5, 0., 0.80, 0.99))
    plt.axvline(df_2019[column].mean(), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(df_2019[column].mean()*1.2, max_ylim*0.9, 'Mean: {:.2f}'.format(df_2019[column].mean()))
    plt.xlabel(column,fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
plot_hist_2019('Score')
df_2019[df_2019['Score'] < 4.33]['Country or region']
df_2019[df_2019['Score'] > 7.28]['Country or region']
plot_hist_2019('GDP per capita')
df_2019[df_2019['GDP per capita']>1.52]['Country or region']
df_2019[df_2019['GDP per capita']<0.17]['Country or region']
plot_hist_2019('Social support')
df_2019[df_2019['Social support']<0.16]['Country or region']
plot_hist_2019('Healthy life expectancy')
df_2019[df_2019['Healthy life expectancy']<0.23]['Country or region']
plot_hist_2019('Freedom to make life choices')
df_2019[df_2019['Freedom to make life choices']>0.57]['Country or region']
plot_hist_2019('Generosity')
df_2019[df_2019['Generosity']>0.40]['Country or region']
df_2019[df_2019['Generosity']<0.06]['Country or region']
plot_hist_2019('Perceptions of corruption')
df_2019[df_2019['Perceptions of corruption']>0.36]['Country or region']
df_2019[df_2019['Perceptions of corruption']<0.05]['Country or region']
selected_data=df_2019.iloc[:,2:]
selected_data.head()
corr=selected_data.corr()
sns.heatmap(corr,annot=True)
sns.pairplot(df_2019,kind='reg')
fig,axes= plt.subplots(nrows=3,ncols=2,figsize=(15,15))
fig.subplots_adjust(wspace=0.5,hspace=0.5)
sns.barplot(x='GDP per capita',y='Country or region',data=df_2019.nlargest(10,'GDP per capita'),ax=axes[0,0])
sns.barplot(x='Social support',y='Country or region',data=df_2019.nlargest(10,'Social support'),ax=axes[0,1])
sns.barplot(x='Healthy life expectancy',y='Country or region',data=df_2019.nlargest(10,'Healthy life expectancy'),ax=axes[1,0])
sns.barplot(x='Generosity',y='Country or region',data=df_2019.nlargest(10,'Generosity'),ax=axes[1,1])
sns.barplot(x='Freedom to make life choices',y='Country or region',data=df_2019.nlargest(10,'Freedom to make life choices'),ax=axes[2,0])
sns.barplot(x='Perceptions of corruption',y='Country or region',data=df_2019.nlargest(10,'Perceptions of corruption'),ax=axes[2,1])
data = dict(type='choropleth',
locations = df_2019['Country or region'],
locationmode = 'country names', z = df_2019['Score'],
text = df_2019['Country or region'], colorbar = {'title':'Happiness Score'},
colorscale=[[0, 'rgb(224,255,255)'],
            [0.1, 'rgb(166,206,227)'], [0.2, 'rgb(31,120,180)'],
           ],    
reversescale = False)
geo = dict(showframe = True, projection={'type':'Mercator'})
choromap = go.Figure(data = [data])
iplot(choromap, validate=False)
df_2015=pd.read_csv("/kaggle/input/world-happiness/2015.csv")
df_2016=pd.read_csv("/kaggle/input/world-happiness/2016.csv")
df_2017=pd.read_csv("/kaggle/input/world-happiness/2017.csv")
df_2018=pd.read_csv("/kaggle/input/world-happiness/2018.csv")
df_2015.columns
df_2019.columns
df_2015=df_2015.rename(columns={"Country": "Country or region",
                        "Happiness Rank": "Overall rank_15",
                        "Happiness Score": "Score_15",
                        "Economy (GDP per Capita)": "GDP per capita_15",
                        "Health (Life Expectancy)": "Healthy life expectancy_15",
                        "Freedom": "Freedom to make life choices_15",
                        "Trust (Government Corruption)": "Perceptions of corruption_15",
                        "Generosity": "Generosity_15",
                        "Family": "Social support_15"
                       })
df_2015.drop(['Standard Error','Dystopia Residual'],axis=1,inplace=True)
df_2015.head()
data=pd.merge(left=df_2019, right=df_2015, how='left', left_on='Country or region', right_on='Country or region')
data.head()
df_2016.columns
df_2016=df_2016.rename(columns={"Country": "Country or region",
                        "Happiness Rank": "Overall rank_16",
                        "Happiness Score": "Score_16",
                        "Economy (GDP per Capita)": "GDP per capita_16",
                        "Health (Life Expectancy)": "Healthy life expectancy_16",
                        "Freedom": "Freedom to make life choices_16",
                        "Trust (Government Corruption)": "Perceptions of corruption_16",
                        "Generosity": "Generosity_16",
                        "Family": "Social support_16"
                       })
df_2016.drop(['Region','Lower Confidence Interval','Upper Confidence Interval','Dystopia Residual'],axis=1,inplace=True)
df_2016.head()
data=pd.merge(left=data, right=df_2016, how='left', left_on='Country or region', right_on='Country or region')
data.head()
df_2017.columns
df_2017=df_2017.rename(columns={"Country": "Country or region",
                        "Happiness.Rank": "Overall rank_17",
                        "Happiness.Score": "Score_17",
                        "Economy..GDP.per.Capita.": "GDP per capita_17",
                        "Health..Life.Expectancy.": "Healthy life expectancy_17",
                        "Freedom": "Freedom to make life choices_17",
                        "Trust..Government.Corruption.": "Perceptions of corruption_17",
                        "Generosity": "Generosity_17",
                        "Family": "Social support_17"
                       })
df_2017.drop(['Whisker.high','Whisker.low','Dystopia.Residual'],axis=1,inplace=True)
data=pd.merge(left=data, right=df_2017, how='left', left_on='Country or region', right_on='Country or region')
data.head()
df_2018.columns
df_2018.columns = [str(col) + '_18' for col in df_2018.columns]
data=pd.merge(left=data, right=df_2018, how='left', left_on='Country or region', right_on='Country or region_18')
data.head()
data.drop(['Country or region_18'],axis=1,inplace=True)
def label(x, color, label):
    ax = plt.gca()
    ax.text(-0.1, .2, label, fontweight="bold", color="black",
            ha="left", va="center", transform=ax.transAxes)
def ridgeplot(df,column):
    ridge_plot = sns.FacetGrid(df, row="year", hue="year", aspect=5, height=1.25)
    # Draw the densities in a few steps
    ridge_plot.map(sns.kdeplot,column, clip_on=False, shade=True, alpha=0.7, lw=4, bw=.2)
    #g.map(sns.kdeplot, "co2_emission", clip_on=False, color="b", lw=4, bw=.2)
    ridge_plot.map(plt.axhline, y=0, lw=4, clip_on=False)
    ridge_plot.map(label,column)
    # Set the subplots to overlap
    ridge_plot.fig.subplots_adjust(hspace=-0.01)
    # Remove axes details that don't play well with overlap
    ridge_plot.set_titles("")
    ridge_plot.set(yticks=[])
    ridge_plot.despine(bottom=True, left=True)
series_year=pd.Series(['2019']).repeat(156).append(pd.Series(['2018']).repeat(156)).append(pd.Series(['2017']).repeat(156)).append(pd.Series(['2016']).repeat(156)).append(pd.Series(['2015']).repeat(156))
series_year=series_year.reset_index(drop=True)
health_data=data['Healthy life expectancy'].append(data['Healthy life expectancy_18']).append(data['Healthy life expectancy_17']).append(data['Healthy life expectancy_16']).append(data['Healthy life expectancy_15'])
health_data=health_data.reset_index(drop=True)
df=pd.DataFrame()
df['Healthy Life Expectancy']=health_data
df['year']=series_year
ridgeplot(df,'Healthy Life Expectancy')
economy_data=data['GDP per capita'].append(data['GDP per capita_18']).append(data['GDP per capita_17']).append(data['GDP per capita_16']).append(data['GDP per capita_15'])
economy_data=economy_data.reset_index(drop=True)
df=pd.DataFrame()
df['GDP per capita']=economy_data
df['year']=series_year
ridgeplot(df,'GDP per capita')
social_data=data['Social support'].append(data['Social support_18']).append(data['Social support_17']).append(data['Social support_16']).append(data['Social support_15'])
social_data=social_data.reset_index(drop=True)
df=pd.DataFrame()
df['Social Support']=social_data
df['year']=series_year
ridgeplot(df,'Social Support')
fig,ax=plt.subplots(2,3)
fig.set_figheight(12)
fig.set_figwidth(22)
fig.tight_layout(pad=7.0)

df_2019['Healthy life expectancy'].plot(kind='line', color='blue',linewidth=1,grid=True,linestyle="-",ax=ax[0,0])
df_2019['GDP per capita'].plot(kind='line', color='limegreen',linewidth=1,grid=True,linestyle="-",ax=ax[0,0])
ax[0,0].set_xlabel('Healthy life expectancy',fontsize=15)
ax[0,0].set_ylabel('GDP per capita',fontsize=15)
ax[0,0].set_title('2019',fontsize=20)
ax[0,0].legend(loc='upper right',fontsize=10)
ax[0,0].tick_params(axis='both', which='major', labelsize=15)

df_2018['Healthy life expectancy_18'].plot(kind='line', color='blue',linewidth=1,grid=True,linestyle="-",ax=ax[0,1])
df_2018['GDP per capita_18'].plot(kind='line', color='limegreen',linewidth=1,grid=True,linestyle="-",ax=ax[0,1])
ax[0,1].set_xlabel('Healthy life expectancy',fontsize=15)
ax[0,1].set_ylabel('GDP per capita',fontsize=15)
ax[0,1].set_title('2018',fontsize=20)
ax[0,1].legend(loc='upper right',fontsize=10)
ax[0,1].tick_params(axis='both', which='major', labelsize=15)

df_2017['Healthy life expectancy_17'].plot(kind='line', color='blue',linewidth=1,grid=True,linestyle="-",ax=ax[0,2])
df_2017['GDP per capita_17'].plot(kind='line', color='limegreen',linewidth=1,grid=True,linestyle="-",ax=ax[0,2])
ax[0,2].set_xlabel('Healthy life expectancy',fontsize=15)
ax[0,2].set_ylabel('GDP per capita',fontsize=15)
ax[0,2].set_title('2017',fontsize=20)
ax[0,2].legend(loc='upper right',fontsize=10)
ax[0,2].tick_params(axis='both', which='major', labelsize=15)

df_2016['Healthy life expectancy_16'].plot(kind='line', color='blue',linewidth=1,grid=True,linestyle="-",ax=ax[1,0])
df_2016['GDP per capita_16'].plot(kind='line', color='limegreen',linewidth=1,grid=True,linestyle="-",ax=ax[1,0])
ax[1,0].set_xlabel('Healthy life expectancy',fontsize=15)
ax[1,0].set_ylabel('GDP per capita',fontsize=15)
ax[1,0].set_title('2016',fontsize=20)
ax[1,0].legend(loc='upper right',fontsize=10)
ax[1,0].tick_params(axis='both', which='major', labelsize=15)

df_2015['Healthy life expectancy_15'].plot(kind='line', color='blue',linewidth=1,grid=True,linestyle="-",ax=ax[1,1])
df_2015['GDP per capita_15'].plot(kind='line', color='limegreen',linewidth=1,grid=True,linestyle="-",ax=ax[1,1])
ax[1,1].set_xlabel('Healthy life expectancy',fontsize=15)
ax[1,1].set_ylabel('GDP per capita',fontsize=15)
ax[1,1].set_title('2015',fontsize=20)
ax[1,1].legend(loc='upper right',fontsize=10)
ax[1,1].tick_params(axis='both', which='major', labelsize=15)

fig.delaxes(ax[1][2])
fig,ax=plt.subplots(2,3)
fig.set_figheight(12)
fig.set_figwidth(22)
fig.tight_layout(pad=7.0)

df_2019['Healthy life expectancy'].plot(kind='line', color='blue',linewidth=1,grid=True,linestyle="-",ax=ax[0,0])
df_2019['Social support'].plot(kind='line', color='red',linewidth=1,grid=True,linestyle="-",ax=ax[0,0])
ax[0,0].set_xlabel('Healthy life expectancy',fontsize=15)
ax[0,0].set_ylabel('Social support',fontsize=15)
ax[0,0].set_title('2019',fontsize=20)
ax[0,0].legend(loc='upper right',fontsize=10)
ax[0,0].tick_params(axis='both', which='major', labelsize=15)

df_2018['Healthy life expectancy_18'].plot(kind='line', color='blue',linewidth=1,grid=True,linestyle="-",ax=ax[0,1])
df_2018['Social support_18'].plot(kind='line', color='red',linewidth=1,grid=True,linestyle="-",ax=ax[0,1])
ax[0,1].set_xlabel('Healthy life expectancy',fontsize=15)
ax[0,1].set_ylabel('Social support',fontsize=15)
ax[0,1].set_title('2018',fontsize=20)
ax[0,1].legend(loc='upper right',fontsize=10)
ax[0,1].tick_params(axis='both', which='major', labelsize=15)

df_2017['Healthy life expectancy_17'].plot(kind='line', color='blue',linewidth=1,grid=True,linestyle="-",ax=ax[0,2])
df_2017['Social support_17'].plot(kind='line', color='red',linewidth=1,grid=True,linestyle="-",ax=ax[0,2])
ax[0,2].set_xlabel('Healthy life expectancy',fontsize=15)
ax[0,2].set_ylabel('Social support',fontsize=15)
ax[0,2].set_title('2017',fontsize=20)
ax[0,2].legend(loc='upper right',fontsize=10)
ax[0,2].tick_params(axis='both', which='major', labelsize=15)

df_2016['Healthy life expectancy_16'].plot(kind='line', color='blue',linewidth=1,grid=True,linestyle="-",ax=ax[1,0])
df_2016['Social support_16'].plot(kind='line', color='red',linewidth=1,grid=True,linestyle="-",ax=ax[1,0])
ax[1,0].set_xlabel('Healthy life expectancy',fontsize=15)
ax[1,0].set_ylabel('Social support',fontsize=15)
ax[1,0].set_title('2016',fontsize=20)
ax[1,0].legend(loc='upper right',fontsize=10)
ax[1,0].tick_params(axis='both', which='major', labelsize=15)

df_2015['Healthy life expectancy_15'].plot(kind='line', color='blue',linewidth=1,grid=True,linestyle="-",ax=ax[1,1])
df_2015['Social support_15'].plot(kind='line', color='red',linewidth=1,grid=True,linestyle="-",ax=ax[1,1])
ax[1,1].set_xlabel('Healthy life expectancy',fontsize=15)
ax[1,1].set_ylabel('Social support',fontsize=15)
ax[1,1].set_title('2015',fontsize=20)
ax[1,1].legend(loc='upper right',fontsize=10)
ax[1,1].tick_params(axis='both', which='major', labelsize=15)

fig.delaxes(ax[1][2])
fig,ax=plt.subplots(2,3)
fig.set_figheight(12)
fig.set_figwidth(22)
fig.tight_layout(pad=7.0)

df_2019['GDP per capita'].plot(kind='line', color='limegreen',linewidth=1,grid=True,linestyle="-",ax=ax[0,0])
df_2019['Social support'].plot(kind='line', color='red',linewidth=1,grid=True,linestyle="-",ax=ax[0,0])
ax[0,0].set_xlabel('GDP per capita',fontsize=15)
ax[0,0].set_ylabel('Social support',fontsize=15)
ax[0,0].set_title('2019',fontsize=20)
ax[0,0].legend(loc='upper right',fontsize=10)
ax[0,0].tick_params(axis='both', which='major', labelsize=15)

df_2018['GDP per capita_18'].plot(kind='line', color='limegreen',linewidth=1,grid=True,linestyle="-",ax=ax[0,1])
df_2018['Social support_18'].plot(kind='line', color='red',linewidth=1,grid=True,linestyle="-",ax=ax[0,1])
ax[0,1].set_xlabel('GDP per capita',fontsize=15)
ax[0,1].set_ylabel('Social support',fontsize=15)
ax[0,1].set_title('2018',fontsize=20)
ax[0,1].legend(loc='upper right',fontsize=10)
ax[0,1].tick_params(axis='both', which='major', labelsize=15)

df_2017['GDP per capita_17'].plot(kind='line', color='limegreen',linewidth=1,grid=True,linestyle="-",ax=ax[0,2])
df_2017['Social support_17'].plot(kind='line', color='red',linewidth=1,grid=True,linestyle="-",ax=ax[0,2])
ax[0,2].set_xlabel('GDP per capita',fontsize=15)
ax[0,2].set_ylabel('Social support',fontsize=15)
ax[0,2].set_title('2017',fontsize=20)
ax[0,2].legend(loc='upper right',fontsize=10)
ax[0,2].tick_params(axis='both', which='major', labelsize=15)

df_2016['GDP per capita_16'].plot(kind='line', color='limegreen',linewidth=1,grid=True,linestyle="-",ax=ax[1,0])
df_2016['Social support_16'].plot(kind='line', color='red',linewidth=1,grid=True,linestyle="-",ax=ax[1,0])
ax[1,0].set_xlabel('GDP per capita',fontsize=15)
ax[1,0].set_ylabel('Social support',fontsize=15)
ax[1,0].set_title('2016',fontsize=20)
ax[1,0].legend(loc='upper right',fontsize=10)
ax[1,0].tick_params(axis='both', which='major', labelsize=15)

df_2015['GDP per capita_15'].plot(kind='line', color='limegreen',linewidth=1,grid=True,linestyle="-",ax=ax[1,1])
df_2015['Social support_15'].plot(kind='line', color='red',linewidth=1,grid=True,linestyle="-",ax=ax[1,1])
ax[1,1].set_xlabel('GDP per capita',fontsize=15)
ax[1,1].set_ylabel('Social support',fontsize=15)
ax[1,1].set_title('2015',fontsize=20)
ax[1,1].legend(loc='upper right',fontsize=10)
ax[1,1].tick_params(axis='both', which='major', labelsize=15)

fig.delaxes(ax[1][2])
fig,ax=plt.subplots(2,3)
fig.set_figheight(12)
fig.set_figwidth(22)
fig.tight_layout(pad=7.0)

df_2019['Perceptions of corruption'].plot(kind='line', color='black',linewidth=1,grid=True,linestyle="-",ax=ax[0,0])
df_2019['Freedom to make life choices'].plot(kind='line', color='magenta',linewidth=1,grid=True,linestyle="-",ax=ax[0,0])
ax[0,0].set_xlabel('Perceptions of corruption',fontsize=15)
ax[0,0].set_ylabel('Freedom to make life choices',fontsize=15)
ax[0,0].set_title('2019',fontsize=20)
ax[0,0].legend(loc='upper right',fontsize=10)
ax[0,0].tick_params(axis='both', which='major', labelsize=15)

df_2018['Perceptions of corruption_18'].plot(kind='line', color='black',linewidth=1,grid=True,linestyle="-",ax=ax[0,1])
df_2018['Freedom to make life choices_18'].plot(kind='line', color='magenta',linewidth=1,grid=True,linestyle="-",ax=ax[0,1])
ax[0,1].set_xlabel('Perceptions of corruption',fontsize=15)
ax[0,1].set_ylabel('Freedom to make life choices',fontsize=15)
ax[0,1].set_title('2018',fontsize=20)
ax[0,1].legend(loc='upper right',fontsize=10)
ax[0,1].tick_params(axis='both', which='major', labelsize=15)

df_2017['Perceptions of corruption_17'].plot(kind='line', color='black',linewidth=1,grid=True,linestyle="-",ax=ax[0,2])
df_2017['Freedom to make life choices_17'].plot(kind='line', color='magenta',linewidth=1,grid=True,linestyle="-",ax=ax[0,2])
ax[0,2].set_xlabel('Perceptions of corruption',fontsize=15)
ax[0,2].set_ylabel('Freedom to make life choices',fontsize=15)
ax[0,2].set_title('2017',fontsize=20)
ax[0,2].legend(loc='upper right',fontsize=10)
ax[0,2].tick_params(axis='both', which='major', labelsize=15)

df_2016['Perceptions of corruption_16'].plot(kind='line', color='black',linewidth=1,grid=True,linestyle="-",ax=ax[1,0])
df_2016['Freedom to make life choices_16'].plot(kind='line', color='magenta',linewidth=1,grid=True,linestyle="-",ax=ax[1,0])
ax[1,0].set_xlabel('Perceptions of corruption',fontsize=15)
ax[1,0].set_ylabel('Freedom to make life choices',fontsize=15)
ax[1,0].set_title('2016',fontsize=20)
ax[1,0].legend(loc='upper right',fontsize=10)
ax[1,0].tick_params(axis='both', which='major', labelsize=15)

df_2015['Perceptions of corruption_15'].plot(kind='line', color='black',linewidth=1,grid=True,linestyle="-",ax=ax[1,1])
df_2015['Freedom to make life choices_15'].plot(kind='line', color='magenta',linewidth=1,grid=True,linestyle="-",ax=ax[1,1])
ax[1,1].set_xlabel('Perceptions of corruption',fontsize=15)
ax[1,1].set_ylabel('Freedom to make life choices',fontsize=15)
ax[1,1].set_title('2015',fontsize=20)
ax[1,1].legend(loc='upper right',fontsize=10)
ax[1,1].tick_params(axis='both', which='major', labelsize=15)

fig.delaxes(ax[1][2])
score_19=df_2019[df_2019['Country or region']=='India']['Score']
score_18=df_2018[df_2018['Country or region_18']=='India']['Score_18']
score_17=df_2017[df_2017['Country or region']=='India']['Score_17']
score_16=df_2016[df_2016['Country or region']=='India']['Score_16']
score_15=df_2015[df_2015['Country or region']=='India']['Score_15']

scores=score_19.append(score_18).append(score_17).append(score_16).append(score_15)
scores.reset_index(drop=True,inplace=True)

df=pd.DataFrame({'Year':['2019','2018','2017','2016','2015'],'Score':scores})
sns.lineplot(x="Year", y="Score",data=df)
plt.title("Happiness Score Trend for India",fontsize=15)
gdp_19=df_2019[df_2019['Country or region']=='India']['GDP per capita']
gdp_18=df_2018[df_2018['Country or region_18']=='India']['GDP per capita_18']
gdp_17=df_2017[df_2017['Country or region']=='India']['GDP per capita_17']
gdp_16=df_2016[df_2016['Country or region']=='India']['GDP per capita_16']
gdp_15=df_2015[df_2015['Country or region']=='India']['GDP per capita_15']

gdp=gdp_19.append(gdp_18).append(gdp_17).append(gdp_16).append(gdp_15)
gdp.reset_index(drop=True,inplace=True)

df=pd.DataFrame({'Year':['2019','2018','2017','2016','2015'],'GDP per capita':gdp})
sns.lineplot(x="Year", y="GDP per capita",data=df)
plt.title("GDP per capita Trend for India",fontsize=15)
health_19=df_2019[df_2019['Country or region']=='India']['Healthy life expectancy']
health_18=df_2018[df_2018['Country or region_18']=='India']['Healthy life expectancy_18']
health_17=df_2017[df_2017['Country or region']=='India']['Healthy life expectancy_17']
health_16=df_2016[df_2016['Country or region']=='India']['Healthy life expectancy_16']
health_15=df_2015[df_2015['Country or region']=='India']['Healthy life expectancy_15']

health=health_19.append(health_18).append(health_17).append(health_16).append(health_15)
health.reset_index(drop=True,inplace=True)

df=pd.DataFrame({'Year':['2019','2018','2017','2016','2015'],'Healthy life expectancy':health})
sns.lineplot(x="Year", y="Healthy life expectancy",data=df)
plt.title("Healthy life expectancy Trend for India",fontsize=15)
social_19=df_2019[df_2019['Country or region']=='India']['Social support']
social_18=df_2018[df_2018['Country or region_18']=='India']['Social support_18']
social_17=df_2017[df_2017['Country or region']=='India']['Social support_17']
social_16=df_2016[df_2016['Country or region']=='India']['Social support_16']
social_15=df_2015[df_2015['Country or region']=='India']['Social support_15']

social=social_19.append(social_18).append(social_17).append(social_16).append(social_15)
social.reset_index(drop=True,inplace=True)

df=pd.DataFrame({'Year':['2019','2018','2017','2016','2015'],'Social support':social})
sns.lineplot(x="Year", y="Social support",data=df)
plt.title("Social support Trend for India",fontsize=15)
corruption_19=df_2019[df_2019['Country or region']=='India']['Perceptions of corruption']
corruption_18=df_2018[df_2018['Country or region_18']=='India']['Perceptions of corruption_18']
corruption_17=df_2017[df_2017['Country or region']=='India']['Perceptions of corruption_17']
corruption_16=df_2016[df_2016['Country or region']=='India']['Perceptions of corruption_16']
corruption_15=df_2015[df_2015['Country or region']=='India']['Perceptions of corruption_15']

corruption=corruption_19.append(corruption_18).append(corruption_17).append(corruption_16).append(corruption_15)
corruption.reset_index(drop=True,inplace=True)

df=pd.DataFrame({'Year':['2019','2018','2017','2016','2015'],'Perceptions of corruption':corruption})
sns.lineplot(x="Year", y="Perceptions of corruption",data=df)
plt.title("Perceptions of corruption Trend for India",fontsize=15)
freedom_19=df_2019[df_2019['Country or region']=='India']['Freedom to make life choices']
freedom_18=df_2018[df_2018['Country or region_18']=='India']['Freedom to make life choices_18']
freedom_17=df_2017[df_2017['Country or region']=='India']['Freedom to make life choices_17']
freedom_16=df_2016[df_2016['Country or region']=='India']['Freedom to make life choices_16']
freedom_15=df_2015[df_2015['Country or region']=='India']['Freedom to make life choices_15']

freedom=freedom_19.append(freedom_18).append(freedom_17).append(freedom_16).append(freedom_15)
freedom.reset_index(drop=True,inplace=True)

df=pd.DataFrame({'Year':['2019','2018','2017','2016','2015'],'Freedom to make life choices':freedom})
sns.lineplot(x="Year", y="Freedom to make life choices",data=df)
plt.title("Freedom to make life choices Trend for India",fontsize=15)
generosity_19=df_2019[df_2019['Country or region']=='India']['Generosity']
generosity_18=df_2018[df_2018['Country or region_18']=='India']['Generosity_18']
generosity_17=df_2017[df_2017['Country or region']=='India']['Generosity_17']
generosity_16=df_2016[df_2016['Country or region']=='India']['Generosity_16']
generosity_15=df_2015[df_2015['Country or region']=='India']['Generosity_15']

generosity=generosity_19.append(generosity_18).append(generosity_17).append(generosity_16).append(generosity_15)
generosity.reset_index(drop=True,inplace=True)

df=pd.DataFrame({'Year':['2019','2018','2017','2016','2015'],'Generosity':generosity})
sns.lineplot(x="Year", y="Generosity",data=df)
plt.title("Generosity Trend for India",fontsize=15)
