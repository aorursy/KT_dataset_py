import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2,venn2_circles
import seaborn as sns


df=pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

#Spliting DataFrame
Iran=df[df['Q3']=='Iran, Islamic Republic of...']
Iran['Q3']='Iran'
Israel=df[df['Q3']=='Israel']
Saudi=df[df['Q3']=='Saudi Arabia']

#Merging
Countries=[Iran,Israel,Saudi]
Country_Names=['Iran','Israel','Saudi Arabia']
n_participants={Country_Names[i]:len(Countries[i]) for i in range(3)}
mydf=pd.concat(Countries)

#Styling
colors=[['#ffb1ab','#ff7d7d','#fa2616','#850000'],['#c4d3ff','#60c6fc','#3666ff','#000485'],['#bdffbf','#8cff93', '#00ff0f','#008507']]
palette=[c[2] for c in colors]
plt.figure(figsize=(10,6))
serie=pd.Series(n_participants)
mybar=plt.bar(serie.index,serie.values,color=palette)
sns.despine(top=True, right=True, left=False, bottom=False)
_=plt.title('Number of Participants')
gender=mydf.groupby(['Q3','Q2']).count()['Q1'].unstack()
gen_pen=np.round(100*gender['Female']/gender.sum(axis=1),2)
gen_pen=pd.DataFrame(gen_pen).reset_index().reindex([2,1,0])
gen_pen.columns=['country','percentage']


# © https://towardsdatascience.com/donut-plot-with-matplotlib-python-be3451f22704
r=.7
rd=.3
startingRadius = r + (rd* (len(gen_pen)-1))
plt.figure(figsize=(15,8))


for index, row in gen_pen.iterrows():
    country = row["country"]
    percentage = row["percentage"]
    textLabel = country + ' ' + str(percentage)+'%'
    remainingPie = 100 - percentage

    donut_sizes = [remainingPie, percentage]

    plt.text(0.01, startingRadius - 0.18, textLabel, horizontalalignment='center', verticalalignment='center')
    plt.pie(donut_sizes, radius=startingRadius, startangle=90, colors=colors[index][::2],
            wedgeprops={"edgecolor": "white", 'linewidth': 1})

    startingRadius-=rd
    

plt.axis('equal')
plt.title('Women\'s Paritipation Rate')

circle = plt.Circle(xy=(0, 0), radius=0.40, facecolor='white')
_=plt.gca().add_artist(circle)
def pie_plotter(data,title,col_dir=1):

    plt.figure(figsize=(16,5))
    plt.suptitle(title)
    for i,country_name in enumerate(Country_Names):
        plt.subplot(1,3,i+1)
        plt.pie(data.loc[country_name],labels=data.columns,colors=colors[i][col_dir-1::col_dir],
            wedgeprops={"edgecolor": "white", 'linewidth': 1},shadow=True)
         #,autopct='%1.0f%%', pctdistance=.8, labeldistance=1.1)
        # Uncomment the styles abovefor getting percentages
        
        circle = plt.Circle(xy=(0, 0), radius=0.60, facecolor='white')
        plt.xlabel(country_name)
        _=plt.gca().add_artist(circle)
def age_categorizer(age):
    if age<3: 
        return '0-29'
    elif age>3:
        return '30-39'
    else:
        return '40+'

mydf['Age_Cat']=mydf['Q1'].apply(lambda x:age_categorizer (int(x[0])))
age_df=mydf.groupby(['Q3','Age_Cat']).count()['Q4'].unstack()
pie_plotter(age_df,'Age Decomposition',-1)
temp=mydf.groupby(['Q3','Q4']).count()['Q1'].unstack()
temp['Other']=temp.drop(['Doctoral degree','Master’s degree'],axis=1).fillna(0).sum(axis=1)
degree_df=temp[['Other','Master’s degree','Doctoral degree']]
degree_df.columns=['B.Sc./Other','M.Sc.','PhD']
pie_plotter(degree_df,'Education Decomposition')
temp=mydf.groupby(['Q3','Q6']).count()['Q1'].unstack()

temp['50-999']=temp['50-249 employees']+temp['250-999 employees']
emp_df=temp[['0-49 employees','50-999','1000-9,999 employees','> 10,000 employees']]
emp_df.columns=['0-49','50-1k','1k-10k','10k+']
pie_plotter(emp_df,'Number of Employees Decomposition')
s1=['$0-999']
s2=['1,000-1,999','2,000-2,999','3,000-3,999','4,000-4,999','5,000-7,499','7,500-9,999','10,000-14,999','15,000-19,999','20,000-24,999','25,000-29,999']
s3=['30,000-39,999','40,000-49,999','50,000-59,999','60,000-69,999','70,000-79,999','80,000-89,999', '90,000-99,999']
s4=['100,000-124,999','125,000-149,999','150,000-199,999','200,000-249,999','> $500,000']

S=[s1,s2,s3,s4]
S_n=['s1','s2','s3','s4']

temp=mydf.groupby(['Q3','Q10']).count()['Q1'].unstack()

for i,s in enumerate(S):
    temp[S_n[i]]=temp[s].sum(axis=1)
    
sal_df=temp[S_n]
sal_df.columns=['1k-','1k-30k','30k-100k','100k+']
pie_plotter(sal_df,'Salary Decomposition')
def preprocess(question):
    cols=[col for col in mydf.columns if ('{}_Part'.format(question) in col)]
    temp=mydf[cols]
    lables=temp.describe().loc['top']
    notnull=temp.notnull()
    notnull.columns=lables
    notnull['Country']=mydf['Q3']
    return notnull
    

def multi_handler(question):
    notnull=preprocess(question)
    return notnull.groupby('Country').sum().transpose()

#  If you don't mind about styling, you may summerize the code section below in just one line!
# _=multi_handler('Q13').plot(kind='barh',colors=palette)
def percent_handler(table):
    for c in Country_Names:
        table[c]=100*table[c]/n_participants[c]
    return table
        


def bar_handler(table,is_percentage=0):
    table['sum']=table.sum(axis=1)
    table=table.reset_index().sort_values('sum',ascending=False)
    n=len(table)

    cats=list(table['top'])*3
    hue=['Iran']*n+['Israel']*n+['Saudi Arabia']*n
    vals=list(table['Iran'])+list(table['Israel'])+list(table['Saudi Arabia'])
    
    last_label='Percentage of Participants' if is_percentage else 'Number of Participants'
    bardata=pd.DataFrame(zip(cats,hue,vals),columns=['Categories','hue',last_label])
    bardata['Categories']=bardata['Categories'].apply(lambda x: x.split('(')[0])

    return bardata

def bar_plotter(ax,data):
    sns.barplot(ax=ax,data=data,y='Categories',x=data.columns[-1],hue='hue',palette=palette)
    
def multi_plotter(question,title,rotation=False):
    multi_output=multi_handler(question)
    
    _,ax=plt.subplots(1,2,figsize = (16, 5), dpi=300)
    
    data=bar_handler(multi_output)
    bar_plotter(ax[0],data)
    
    data=bar_handler(percent_handler(multi_output),1)
    bar_plotter(ax[1],data)
    ax[0].set_ylabel('')
    ax[1].set_ylabel('')
    ax[0].legend().set_visible(False)
    if rotation:
        
        ax[0].set_yticklabels(ax[0].get_yticklabels(),rotation=15)
        ax[1].set_yticklabels(ax[1].get_yticklabels(),rotation=15)
    
    plt.suptitle(title)
    #plt.subplots_adjust(wspace=wspace)
    sns.despine(top=True, right=True, left=False, bottom=False)
    
    
def single_plotter(question,title):
    data=bar_handler(multi_handler(question))
    plt.figure(figsize=(14,8))
    bar_plotter(None,data)
    plt.title(title)
    plt.ylabel('')
    sns.despine(top=True, right=True, left=False, bottom=False)
multi_plotter('Q13','Source of Learning')
multi_plotter('Q18','Favorite Programming Language')
single_plotter('Q24','Favorite Machine Learning Algorithm')
single_plotter('Q34','Favorite Database Product')
def num_of_chioce(question,title):

    notnull= preprocess(question)
    plt.figure(figsize=(16,4))
    for i,country_name in enumerate(Country_Names):
        plt.suptitle(title)
        plt.subplot(1,3,i+1)
        data=notnull[notnull['Country']==country_name].drop('Country',axis=1).sum(axis=1)
        ax=sns.countplot(data,color=palette[i])    
        ax.set_xlabel(country_name)
        ax.set_ylabel('')
        sns.despine(top=True, right=True, left=False, bottom=False)
num_of_chioce('Q34','Number of DB, Selected by each Participants')
plt.figure(figsize=(15,6))
for i,country in enumerate([Iran,Israel,Saudi]):

    time=country['Time from Start to Finish (seconds)'].apply(int)
    time=time[time<3600]
    sns.distplot(time,hist=False,color=colors[i][2])

_=plt.legend(Country_Names)
def venn_data_creator(question,cat1,cat2):

    notnull=preprocess(question)
    notnull['Both']=notnull[cat1]+notnull[cat2]
    return notnull[['Country',cat1,cat2,'Both']]
    
def venn_plotter(question,cat1,cat2):
    plt.figure(figsize=(16,8))
    venn_data=venn_data_creator(question,cat1,cat2)
    for i,country_name in enumerate(Country_Names):
        data=venn_data[venn_data['Country']==country_name][[cat1,cat2,'Both']].sum(axis=0)
        values=tuple(data.values)
        ax=plt.subplot(1,3,i+1)
        ax.set_title(country_name)
        venn2(subsets = values,set_labels=data.index,set_colors=['#ff6f00','#e417ff'])
        venn2_circles(subsets = values, linewidth=1,color=palette[i])

venn_plotter(question='Q34',cat1='MySQL',cat2='Microsoft SQL Server')