import numpy as np

import pandas as pd

import os

import json

import re

import matplotlib

import matplotlib.pyplot as plt

from matplotlib.patches import ConnectionPatch

import plotly.express as px

import plotly.figure_factory as ff

import pycountry

import copy

import seaborn as sns

from pandas_profiling import ProfileReport

from collections import Counter

from matplotlib import style

from tqdm import tqdm
def get_target_dict(targets, text, paper_id, original_word_dist, word_distribution_in_paper):

    """"

    Summary:This function finds the words in the targets list in the text variable and returns 

    their count in a dictionary 

    

    Parameters:

    targets: The list of words which we are going to search for in a given text 

    text: the text in which we are searching for the words in targets list

    paper_id: The id of the research papers in which we are searching. This field is considered as the key

    original_word_dist: A dictionary which contains the count of the words in targets list as found in 

    the text variable

    word_distribution_in_paper: A dictionary which contains the count of all words in the targets list.

    A count of 0 is added to the ones not found in the text variable.Useful for plotting stacked bar graph

    """

    word_count_new = {}

    word_count = {}

    for sentence in text.split('.'):

        for word in targets:

            if word in sentence:

                if word in word_count: 

                    word_count[word] += sentence.count(word)

                else:

                    word_count[word] = sentence.count(word)

        word_count_new = copy.deepcopy(word_count)

        if bool(word_count_new):

            for word in targets:

                if word not in word_count_new:

                    word_count_new[word] = 0

    word_distribution_in_paper[paper_id] = word_count_new

    original_word_dist[paper_id] = word_count #dictionary without the 0 appends

    return original_word_dist, word_distribution_in_paper
def get_word_count(targets, dataf, col1_text, col2_key):

    """

    Summary: This function creates a dataframe containing sentences which has 

    any or all of the words from the targets list. This dataframe is then passed into the function

    get_target_dict to get the count of the words in the targets list as found in the sentences.

    

    Parameters:

    targets:The list of words which we are going to search for in a given text

    dataf:Dataframe containing the original full text from which we can get the sentences 

    containing words in the targets list

    col1_text:The field of the dataframe dataf in which we will search for the words

    col2_key:The key which we will use to identify a paper uniquely(paper_id in our example)

    """

    df_targets = dataf[dataf[col1_text].apply(lambda sentence: any(word in sentence for word in targets))] 

        

    original_word_dist = {}

    word_distribution_in_paper = {}

    for index, row in df_targets.iterrows():

        original_word_dist, word_distribution_in_paper = get_target_dict(targets, row[col1_text], row[col2_key], original_word_dist, word_distribution_in_paper)



    return original_word_dist, word_distribution_in_paper
#max word count should be more than a threshold value(denoted by the varibale 'limit')

def get_word_distribution(word_dictionary,limit):

    """

    Summary:This function accepts a word dictionary containing words and their respective counts and 

    only keeps words whose count is more than a certain limit.

    Parameters:

    word_dictionary:Dictionary containing words and their count

    limit:The threshold value. If the count of a word is more than this threshold value it is 

    kept in the dictionary

    """

    temp_dictionary={}

    for paperid, lists in word_dictionary.items():

        keep=False

        for word, wordcount in lists.items():

            if wordcount > limit:

                keep=True

                break

        if keep:

            temp_dictionary[paperid]=lists    

    word_dictionary = temp_dictionary

    return word_dictionary
def draw_plot(paper_word_distribution):

    """

    This fuction accepts a dictionary and uses it to draw a stacked bar graph of 

    each research paper containing the distribution of the target list of words found

    """

    labels = paper_word_distribution.keys()

    word_count_list={}

    count=0

    for eachvalue in paper_word_distribution.values():

        for key,value in eachvalue.items():

            if key in word_count_list:

                word_count_list[key].append(value)

            else:

                word_count_list[key]=[value]

    width = 0.35

    fig = plt.figure(figsize=(20,8))

    ax = fig.add_subplot(111)



    for key,value in word_count_list.items():

        ax.bar(labels, value, width, label=key)



    ax.set_ylabel('Word Distribution')

    ax.set_title('Paper wise distribution of keywords')

    ax.legend()

    plt.show()
def draw_plot_horizontal(paper_word_distribution):

    """

    This fuction accepts a dictionary and uses it to draw a horizontal stacked bar graph of 

    each paper containing the distribution of the target list of words found

    """



    matplotlib.rcParams.update({'font.size': 16})



    labels = paper_word_distribution.keys()

    y_pos = np.arange(len(labels))

    word_count_list={}

    count=0

    for eachvalue in paper_word_distribution.values():

        for key,value in eachvalue.items():

            if key in word_count_list:

                word_count_list[key].append(value)

            else:

                word_count_list[key]=[value]

    fig = plt.figure(figsize=(18,9))

    ax = fig.add_subplot(111)



    for key,value in word_count_list.items():

        h = ax.barh(y_pos, value, align='center',label=key)



    ax.set_yticks(y_pos)

    ax.set_yticklabels(labels)

    ax.invert_yaxis()  # labels read top-to-bottom

    ax.set_xlabel('Word Distribution')

    ax.set_title('Paper wise distribution of keywords')

    ax.legend()

    plt.show()
def plot_clustered_stacked(dfall, labels=None, title="Comparing occurence of target words between abstract and full text in COVID19 papers",  H="/", **kwargs):

    n_df = len(dfall)

    n_col = len(dfall[0].columns) 

    n_ind = len(dfall[0].index)

    axe = plt.subplot(111)



    for df in dfall : 

        axe = df.plot(kind="bar",

                      linewidth=2,

                      stacked=True,

                      ax=axe,

                      legend=False,

                      grid=False,

                      **kwargs,

                      figsize=(20,5)) 



    h,l = axe.get_legend_handles_labels()

    for i in range(0, n_df * n_col, n_col): 

        for j, pa in enumerate(h[i:i+n_col]):

            for rect in pa.patches: 

                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))

                rect.set_hatch(H * int(i / n_col))      

                rect.set_width(1 / float(n_df + 1))



    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)

    axe.set_xticklabels(df.index, rotation = 0)

    axe.set_title(title)



    #Add invisible data to add another legend

    n=[]        

    for i in range(n_df):

        n.append(axe.bar(0, 0, color="gray", hatch=H * i))



    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])

    if labels is not None:

        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 

    axe.add_artist(l1)

    return axe
style.use("ggplot")

dirs=["pmc_json","pdf_json"]

docs=[]

counts=0

for d in dirs:

    print(d)

    counts = 0

    for file in tqdm(os.listdir(f"../input/CORD-19-research-challenge/document_parses/{d}")):#What is an f string?

        file_path = f"../input/CORD-19-research-challenge/document_parses/{d}/{file}"

        j = json.load(open(file_path,"rb"))

        #Taking last 7 characters. it removes the 'PMC' appended to the beginning

        #also paperid in pdf_json are guids and hard to plot in the graphs hence the substring

        paper_id = j['paper_id']

        paper_id = paper_id[-7:]

        title = j['metadata']['title']



        try:#sometimes there are no abstracts

            abstract = j['abstract'][0]['text']

        except:

            abstract = ""

            

        full_text = ""

        bib_entries = []

        for text in j['body_text']:

            full_text += text['text']

            for csp in text['cite_spans']:

                try:

                    title = j['bib_entries'][csp['ref_id']]['title']

                    bib_entries.append(title)

                except:

                    pass

                

        docs.append([paper_id, title, abstract, full_text, bib_entries])

        #comment this below block if you want to consider all files

        #comment block start

        counts = counts + 1

        if(counts >= 10000):

            break

        #comment block end    

df=pd.DataFrame(docs,columns=['paper_id','title','abstract','full_text','bib_entries']) 
profile = ProfileReport(df, title='Pandas Profiling Report',html={'style':{'full_width': True}}, progress_bar=False )

profile.to_widgets()
incubation=df[df['full_text'].str.contains('incubation')]    

texts=incubation['full_text'].values

incubation_times = []

for t in texts:

    for sentence in t.split('. '):

        if "incubation" in sentence:

            num_day = 0.0

            num_week = 0.0

            single_day=re.findall(r" \d{1,2} day",sentence)

            single_week=re.findall(r" \d{1,2} week",sentence)

            if len(single_day) == 1 : #picked up one string

                num_day = float(single_day[0].split(" ")[1]) # 6 days; only extracting the no.

            if len(single_week) == 1 :

                num_week = float(single_week[0].split(" ")[1])

            if num_day or num_week:

                incubation_times.append([sentence, num_day, num_week])    
#Renaming the columns in incubation_df.

incubation_df = pd.DataFrame(incubation_times,columns=['sentence','days','weeks']) 

display(incubation_df.loc[incubation_df['days'] != 0.0].head(5))

display(incubation_df.loc[incubation_df['weeks'] != 0.0].head(5))

print(f"The mean projected incubation time in days is ", incubation_df['days'].mean()," days")

print(f"The mean projected incubation time in weeks is ", incubation_df['weeks'].mean()," weeks")

#Datatypes of the various columns in the incubation_df dataframe

incubation_df.dtypes
#Displaying days and their count in the dataset

days_df = pd.DataFrame(incubation_df['days'].value_counts()).reset_index()

days_df.columns = ['Days', 'count']

days_df = days_df.sort_values(by='count', ascending=False)#you can also sort by 'Days'

display(days_df.head(20))#Just wanted to see the ones with double digit values
result_df = days_df.groupby(pd.cut(days_df["Days"], np.arange(0, 90, 7))).sum()

display(result_df)

#Groups and their counts

#0-7 : 401

#8-14 : 239

#15-21 : 65

#Over 21 :We sum up the values more than 21

count_21 = days_df.loc[days_df['Days'] > 21, 'count'].sum()

#Total count of all days mentioned in the dataframe

total = result_df['count'].sum()
font = {'family' : 'normal',

        'weight' : 'bold',

        'size'   : 16}



matplotlib.rc('font', **font)







fig = plt.figure(figsize=(20, 9))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)

fig.subplots_adjust(wspace=0)





#Finding no. of incubation period which is only in days

only_non_zero_days = len(incubation_df[(incubation_df['days'] > 0) & (incubation_df['weeks'] == 0)])



#Finding no. of incubation period which is only in weeks

only_non_zero_weeks = len(incubation_df[(incubation_df['days'] == 0) & (incubation_df['weeks'] > 0)])



#Finding no. of incubation period which ranges from days to weeks

both_days_weeks = len(incubation_df[(incubation_df['days'] > 0) & (incubation_df['weeks'] > 0)])



#Total no. of incubation periods either in days or weeks or both

total_rows = len(incubation_df[(incubation_df['days'] > 0) | (incubation_df['weeks'] > 0)])



#Pie chart parameters

ratios = [only_non_zero_days/total_rows, only_non_zero_weeks/total_rows, both_days_weeks/total_rows]

labels = ['Incubation period in days', 'Incubation period in weeks', 'Incubation period ranging from days to weeks']

explode = [0.3, 0, 0]



#Rotate so that first wedge is split by the x-axis

angle = -180 * ratios[0]

cmap = plt.get_cmap("tab20c")

outer_colors = cmap(np.array([9, 5, 1]))

patches, texts, autotexts = ax1.pie(ratios, autopct='%1.2f%%', startangle=angle,colors=outer_colors,shadow=True,

        labels=labels, explode=explode)

for text in texts:

    text.set_fontsize(15)

for text in autotexts:

    text.set_fontsize(14)



#Bar chart parameters

xpos = 0

bottom = 0



#0-7 days, 8-14 days,15-21 days, Over 21 days(4 categories)

ratios = [401/total, 239/total, 65/total, count_21/total]

width = .2

colors = [[.1, .3, .3], [.3, .5, .5], [.5, .7, .7], [.7, .9, .9]]#colors based on values



for j in range(len(ratios)):

    height = ratios[j]

    ax2.bar(xpos, height, width, bottom=bottom, color=colors[j])

    ypos = bottom + ax2.patches[j].get_height() / 2

    bottom += height

    ax2.text(xpos, ypos, "%d%%" % (ax2.patches[j].get_height() * 100),

             ha='center')



ax2.set_title('Distribution of Incubation Period(In Days)')

ax2.legend(('0-7 days', '8-14 days', '15-21 days', 'Over 21 days'))

ax2.axis('off')

ax2.set_xlim(- 2.5 * width, 2.5 * width)



#Use ConnectionPatch to draw lines between the two plots

#Get the wedge data

theta1, theta2 = ax1.patches[0].theta1, ax1.patches[0].theta2

center, r = ax1.patches[0].center, ax1.patches[0].r

bar_height = sum([item.get_height() for item in ax2.patches])



#Draw top connecting line

x = r * np.cos(np.pi / 180 * theta2) + center[0]

y = r * np.sin(np.pi / 180 * theta2) + center[1]

con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,

                      xyB=(x, y), coordsB=ax1.transData)

con.set_color([0, 0, 0])

con.set_linewidth(4)

ax2.add_artist(con)



#Draw bottom connecting line

x = r * np.cos(np.pi / 180 * theta1) + center[0]

y = r * np.sin(np.pi / 180 * theta1) + center[1]

con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,

                      xyB=(x, y), coordsB=ax1.transData)

con.set_color([0, 0, 0])

ax2.add_artist(con)

con.set_linewidth(4)



plt.show()
df_abstract = df.loc[df['abstract'] != '']

with pd.option_context('display.max_colwidth', -1):

    display(df_abstract[['paper_id','title','abstract','bib_entries']].head(1))
targets = ['vaccine','vaccination','vaccines','vaccinations'] 

limit=10

a,b=get_word_count(targets,df,'abstract','paper_id')

paper_word_distribution=get_word_distribution(b,limit)

draw_plot(paper_word_distribution)
targets = ['vaccine','vaccination','vaccines','vaccinations'] 
def get_wordcount_in_dataframe(df1, df2):

    """

    Getting index from df1 and finding wordcount with same index in df2.

    """

    l=[]

    for index, row in df1.iterrows():

        try:

            l.append(df2.loc[index,:])

        except:

            data = {'vaccine':0,'vaccination':0,'vaccines':0,'vaccinations':0}

            s = pd.Series(data,dtype='int64',name=index)

            l.append(s)

    df3 = pd.DataFrame(l)

    return df3
DF_list = list()



limit = 1

a,b = get_word_count(targets,df_abstract,'abstract','paper_id')

paper_word_distribution1 = get_word_distribution(b,limit)

df1 = pd.DataFrame.from_dict(paper_word_distribution1, orient='index')



#vaccine had the highest count among the target words hence sorted in descending order by vaccine

df1 = df1.sort_values(by='vaccine', ascending=False)

display('Table 1 :Count of target words in abstracts')

display(df1[0:10])

#considering only first 5 values of this list for better plotting

DF_list.append(df1[0:10])



c,d = get_word_count(targets,df_abstract,'full_text','paper_id')

paper_word_distribution2 = get_word_distribution(d,limit)

df2 = pd.DataFrame.from_dict(paper_word_distribution2, orient='index')

df2 = df2.sort_values(by='vaccine', ascending=False)

display('Table 2: Count of target words in full_text')

display(df2[0:10])



display('Table 3: Count of target words in full text for papers in Table 1')

df3 = get_wordcount_in_dataframe(df1[0:10], df2)

display(df3)

DF_list.append(df3)



plot_clustered_stacked(DF_list, cmap=plt.cm.Set2)
DF_list2 = list()

targets = ['vaccine','vaccination','vaccines','vaccinations'] 

display('Table 1 :Count of target words in abstracts')

display(df1[0:10])

display('Table 2: Count of target words in full_text')

display(df2[0:10])

DF_list2.append(df2[0:10])

df3 = get_wordcount_in_dataframe(df2[0:10], df1)

display('Table 3: Count of target words in full text for papers in Table 1')

display(df3)

DF_list2.append(df3)



plot_clustered_stacked(DF_list2, cmap=plt.cm.Set2)
targets = ['vaccine','vaccination','vaccines','vaccinations'] 

limit=200

a,b=get_word_count(targets,df,'full_text','paper_id')

paper_word_distribution=get_word_distribution(b,limit)
draw_plot(paper_word_distribution)
targets = ['anti-viral','anti viral','genome','genome data','genome','strain'] 

limit=200

a,b=get_word_count(targets,df,'full_text','paper_id')

paper_word_distribution=get_word_distribution(b,limit)
draw_plot_horizontal(paper_word_distribution)
targets=['livestock','reservoir','farmer','wildlife','host range','hosts','spillover','animal']

limit=150

a,b=get_word_count(targets,df,'full_text','paper_id')

paper_word_distribution=get_word_distribution(b,limit)
draw_plot(paper_word_distribution)
country_name=[]

for country in pycountry.countries:

    country_name.append(country.name)

targets=country_name

limit=40

countries_mentioned,b = get_word_count(targets,df,'full_text','paper_id')
Total_count = {}

for eachvalue in countries_mentioned.values():

    Total_count = {key: Total_count.get(key, 0) + eachvalue.get(key, 0)

          for key in set(Total_count) | set(eachvalue)}

for country,counts in Total_count.items():

    Total_count[country]=[counts]
print ("{:<35} {:<7}".format('Country','Total no. of times mentioned'))

for k, v in Total_count.items():

     print ("{:<35} {:<7}".format(k, v[0]))
country_count_df= pd.DataFrame.from_dict(Total_count)

country_count_df = country_count_df.T

country_count_df.columns = ['Total no. of times mentioned']

country_count_df['Country'] = country_count_df.index

#Rearranging columns

cols = country_count_df.columns.tolist()

cols = cols[-1:] + cols[:-1]

country_count_df=country_count_df[cols]

sorted_country_count_df = country_count_df.sort_values(by='Total no. of times mentioned', ascending=False)



df_sample = sorted_country_count_df

colorscale = [[0, '#4d004c'],[.5, '#f2e5ff'],[1, '#fffffe']]



fig =  ff.create_table(df_sample, height_constant=20)

fig.show()
np.random.seed(12)

gapminder = px.data.gapminder().query("year==2007")



d = Total_count



data_country = pd.DataFrame(d).T.reset_index()

data_country.columns=['country', 'count']



df_merge=pd.merge(gapminder, data_country, how='left', on='country')



fig = px.choropleth(df_merge, locations="iso_alpha",

                    color="count", 

                    hover_name="country",

                    color_continuous_scale=px.colors.sequential.Plasma)



fig.show()
px.scatter(df_merge, x="country", y="count", color="continent", size="pop", size_max=60,

          hover_name="country")
#Add all the references in a single list and find the count of the repeats

bibs=[]

for item in df['bib_entries']:

    for eachbib in item:

        bibs.append(eachbib)

a = dict(Counter(bibs))

del a['']

df_a=pd.DataFrame.from_dict(a, orient='index',columns=['no. of times cited'])

df_a['no. of times cited'] = df_a['no. of times cited'].astype(str).astype(int)

sorted_df_a=df_a.sort_values(by='no. of times cited', ascending=False)

new_df = sorted_df_a.loc[sorted_df_a['no. of times cited'] >= 50] #50 here is the minimum no. of times the paper has been cited

new_df['title'] = new_df.index.str.slice(0,30)#truncated the title
#Getting Seaborn Style for Pandas Plots

top_n=25

sns.set()       

new_df[0:top_n].reset_index().plot(

    x = 'title', 

    y = 'no. of times cited', 

    kind='bar', 

    legend = False,

    width=0.8

)

plt.ylabel("Number of times cited")

plt.xlabel("Paper")

plt.title("No. of citations of top "+format(top_n)+" papers")

plt.gca().yaxis.grid(linestyle=':')