# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import matplotlib.lines as mlines

import seaborn as sns

import warnings

warnings.simplefilter('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
mu_df = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

qu_df = pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')

su_df = pd.read_csv('/kaggle/input/kaggle-survey-2019/survey_schema.csv')

ot_df = pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')

mu19_df = mu_df.drop(index=0, axis=0)

mu18_df1 = pd.read_csv('/kaggle/input/kaggle-survey-2018/multipleChoiceResponses.csv')

mu18_df = mu18_df1.drop(index=0, axis=0)

mu17_df = pd.read_csv('/kaggle/input/kaggle-survey-2017/multipleChoiceResponses.csv', encoding="ISO-8859-1")
def make_df(df, val, col, year, s ):

# Count by country

# df: input DataFrame

# val: count value_name

# col: count column_name

# year: survey year

# s: age or gen

    wdf = pd.DataFrame(df[val].groupby(df['country_w']).value_counts())

    wdf.columns=['count']

    wdf.reset_index(inplace=True)

    wdf.columns=['country', col, s+'_count']

    wdf['year'] = year

    

    return wdf

#Eliminate the shaking

mu19_df['country_w'] = mu19_df['Q3']

mu19_df.loc[mu19_df[mu19_df['country_w']=='Republic of Korea'].index, 

            ['country_w']]='South Korea'



mu18_df['age_class'] = mu18_df['Q2']

mu18_df.loc[mu18_df[mu18_df['age_class']=='70-79'].index, 

            ['age_class']]='70+'

mu18_df.loc[mu18_df[mu18_df['age_class']=='80+'].index, 

            ['age_class']]='70+'



mu18_df['country_w'] = mu18_df['Q3']

mu18_df.loc[mu18_df[mu18_df['country_w']=='Republic of Korea'].index, 

            ['country_w']]='South Korea'



mu17_df['age_class'] = pd.cut(mu17_df['Age'], [18, 22, 25, 30, 35, 40, 45, 50, 55, 60, 70, 100], 

           right=False, labels=['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-69', '70+'])



mu17_df['country_w'] = mu17_df['Country']

mu17_df.loc[mu17_df[mu17_df['country_w']=="People 's Republic of China"].index, 

            ['country_w']]='China'

mu17_df.loc[mu17_df[mu17_df['country_w']=="Republic of China"].index, 

            ['country_w']]='China'

mu17_df.loc[mu17_df[mu17_df['country_w']=="United Kingdom"].index, 

            ['country_w']]='United Kingdom of Great Britain and Northern Ireland'

mu17_df.loc[mu17_df[mu17_df['country_w']=="United States"].index, 

            ['country_w']]='United States of America'

mu17_df.loc[mu17_df[mu17_df['country_w']=="Vietnam"].index, 

            ['country_w']]='Viet Nam'

mu17_df.loc[mu17_df[mu17_df['country_w']=="Hong Kong"].index, 

            ['country_w']]='Hong Kong (S.A.R.)'



country_age_19 = make_df(mu19_df, 'Q1', 'age_class', '2019', 'age')

country_gen_19 = make_df(mu19_df, 'Q2', 'gender', '2019', 'gen')

country_age_18 = make_df(mu18_df, 'Q2', 'age_class', '2018', 'age')

country_gen_18 = make_df(mu18_df, 'Q1', 'gender', '2018', 'gen')

country_age_17 = make_df(mu17_df, 'age_class', 'age_class', '2017', 'age')

country_gen_17 = make_df(mu17_df, 'GenderSelect', 'gender', '2017', 'gen')



country_age_all = pd.concat([country_age_19, country_age_18, country_age_17], axis=0)

country_gen_all = pd.concat([country_gen_19, country_gen_18, country_gen_17], axis=0)

country_age_all.reset_index(drop=True, inplace=True)

country_gen_all.reset_index(drop=True, inplace=True)



elder_age_lst = ['50-54', '55-59', '60-69', '70+']

country_age_all['s_age']='youth'

country_age_all.loc[country_age_all[country_age_all['age_class'].isin(elder_age_lst)].index, 

                    ['s_age']]='elderly'



country_age_all.replace({'United States of America':'USA', 

             'United Kingdom of Great Britain and Northern Ireland':'UK'}, inplace=True)

country_gen_all.replace({'United States of America':'USA', 

             'United Kingdom of Great Britain and Northern Ireland':'UK'}, inplace=True)
def calculate_ratio(df, val):

# Calculate the ratio

# df: input DataFrame

# val: count value_name

    

    wdf = pd.DataFrame(df.groupby(['year', 'country', val]).sum()).reset_index()

    wdf_a = pd.DataFrame(df.groupby(['year', 'country']).sum()).reset_index()

    wdf = pd.merge(wdf, wdf_a, on=['year', 'country'], how='left')

    wdf.columns = ['year', 'country', val, 'count', 'total']

    wdf['rate'] = wdf['count'] / wdf['total'] *100

    

    return wdf
age_df = calculate_ratio(country_age_all, 's_age')

gen_df = calculate_ratio(country_gen_all, 'gender')

ranking_2019 = age_df[(age_df['s_age']=='elderly')&(age_df['year']=='2019')].sort_values('total', ascending=False)

total_df = age_df[(age_df['s_age']=='elderly') & (age_df['country'].isin(ranking_2019.iloc[:10, 1]))

                 ].sort_values(['year','total'], ascending=False)

gender_df = gen_df[(gen_df['gender']=='Female') & (gen_df['country'].isin(ranking_2019.iloc[:10, 1]))

                 ].sort_values(['year','total'], ascending=False)
fig = plt.figure(figsize=(12,4))

ax = fig.add_subplot(1, 1, 1)

g = sns.barplot(x='country', y='total', 

            data=total_df, hue='year', palette='inferno_r', alpha=0.7)

 

plt.ylabel('Total count of respondents')

plt.title('Top 10 respondents by country',fontsize=14 )

plt.show()
plt.figure(figsize=(12,4))

ax = fig.add_subplot(1, 1, 1)



sns.scatterplot(x='country', y='rate', data=total_df[total_df['year']=='2019'], 

                s=350, marker='*', color='cyan', label='Elderly')



sns.scatterplot(x='country', y='rate', data=gender_df[gender_df['year']=='2019'], 

                s=150, marker='o', color='magenta', label='Female')



plt.legend(loc='upper center')

plt.ylabel('Parcentage(%)')

plt.title('Parcentage of Elderly or Female in Top10 country',fontsize=14 )

plt.show()
def make_1819df(wdf, val, val1):

# Calculate the Growth rate

# wdf: input DataFrame

# val: 

# val1:



    df_19 = wdf[wdf['year']=='2019'].rename(columns={'rate': '19rate'}).loc[:, ['country', val, '19rate']]

    df_18 = wdf[wdf['year']=='2018'].rename(columns={'rate': '18rate'}).loc[:, ['country', val, '18rate']]

    df_1819 = pd.merge(df_19, df_18, on=['country', val], how='inner')

    df_1819['up1819'] = (df_1819['19rate']-df_1819['18rate'])/df_1819['18rate']

    up_df = df_1819[df_1819[val]==val1].sort_values('19rate', ascending=False)

    return up_df
age_up_df = make_1819df(age_df, 's_age', 'elderly')

gen_up_df = make_1819df(gen_df, 'gender', 'Female')
fig = plt.figure(figsize=(12,4))



ax1 = fig.add_subplot(1, 1, 1)

ax2 = ax1.twinx()



sns.barplot(ax=ax1, data=age_up_df, x='country', y='19rate', palette='spring')

sns.scatterplot(ax=ax2, data=age_up_df, x='country', y='up1819', color='blue', 

                label='Growth rate of Elderly(2018->2019)')



ax1.set_xticklabels(age_up_df['country'], rotation='vertical')

ax1.set_ylabel('The Parcentage of Elderly(%) 2019')

ax2.set_ylabel('Growth rate(%)')

plt.legend(loc='upper left')

plt.title('Change in the proportion of elderly people',fontsize=14 )

plt.show()
age_growth = age_up_df.sort_values('19rate', ascending=False).iloc[:10,:]

age_growth['rank'] = np.arange(10,0,-1)

age_growth.reset_index(drop=True, inplace=True)



gen_growth = gen_up_df.sort_values('19rate', ascending=False).iloc[:10,:]

gen_growth['rank'] = np.arange(10,0,-1)

gen_growth.reset_index(drop=True, inplace=True)
def newline(p1, p2, color='black'):

    ax = plt.gca()

    l = mlines.Line2D([p1[0],p2[0]], [p1[1],p2[1]], color='red' if p1[1]-p2[1] > 0 else 'black', alpha=0.3)

    ax.add_line(l)

    return l
def draw_growth(wdf, title_text, ylabel_text, y_min, y_max):

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(7,7))

    pos = np.arange(y_max-1,y_max-11,-1)



    ax.vlines(x=1, ymin=y_min, ymax=y_max, color='black', alpha=0.7, linewidth=1, linestyles='dotted')

    ax.vlines(x=3, ymin=y_min, ymax=y_max, color='black', alpha=0.7, linewidth=1, linestyles='dotted')

    ax.scatter(y=wdf['18rate'], x=np.repeat(1, wdf.shape[0]), s=300, alpha=0.7, vmin=1, vmax=10, c=wdf['rank'], cmap=cm.gist_rainbow)

    ax.scatter(y=wdf['19rate'], x=np.repeat(3, wdf.shape[0]), s=300, alpha=0.7, vmin=1, vmax=10, c=wdf['rank'], cmap=cm.gist_rainbow)

    ax.scatter(y=pos, x=np.repeat(4.2, wdf.shape[0]), s=300, alpha=0.7, vmin=1, vmax=10, c=wdf['rank'], cmap=cm.gist_rainbow)



    for p1, p2, p3, c in zip(wdf['18rate'], wdf['19rate'], pos, wdf['country']):

        newline([1,p1], [3,p2])

        ax.text(1-0.1, p1, str(round(p1)), horizontalalignment='right', verticalalignment='center', fontdict={'size':8})

        ax.text(3+0.1, p2, str(round(p2)), horizontalalignment='left', verticalalignment='center', fontdict={'size':8})

        ax.text(4.2, p3, c, horizontalalignment='left', verticalalignment='center', fontdict={'size':9})

    

    ax.text(1, y_max+0.1, '2018', horizontalalignment='center', verticalalignment='center', fontdict={'size':11, 'weight':500})

    ax.text(3, y_max+0.1, '2019', horizontalalignment='center', verticalalignment='center', fontdict={'size':11, 'weight':500})



    ax.set_title(title_text, fontdict={'size':14})

    ax.set(xlim=(0,5), ylim=(y_min-2,y_max+2), ylabel=ylabel_text)

    ax.set_xticks([])

    #ax.set_xticklabels(["2018", "2019"])

    plt.yticks(np.arange(y_min, y_max, 5), fontsize=9)



    plt.gca().spines["top"].set_alpha(.0)

    plt.gca().spines["bottom"].set_alpha(.0)

    plt.gca().spines["right"].set_alpha(.0)

    plt.gca().spines["left"].set_alpha(.0)

    plt.show()
title_text='Change in ratio of Elderly 2018 -> 2019'

ylabel_text = 'Elderly Ratio'

draw_growth(age_growth, title_text, ylabel_text, 4, 25)
fig = plt.figure(figsize=(12,4))



ax1 = fig.add_subplot(1, 1, 1)

ax2 = ax1.twinx()



sns.barplot(ax=ax1, data=gen_up_df, x='country', y='19rate', palette='summer')

sns.scatterplot(ax=ax2, data=gen_up_df, x='country', y='up1819', color='red',

               label='Growth rate of Female(2018->2019)')



ax1.set_xticklabels(gen_up_df['country'], rotation='vertical')

ax1.set_ylabel('The Parcentage of Female(%) 2019')

ax2.set_ylabel('Growth rate(%)')

plt.legend(loc='upper left')

plt.title('Change in the proportion of female',fontsize=14 )

plt.show()
title_text='Change in ratio of Female 2018 -> 2019'

ylabel_text = 'Female Ratio'

draw_growth(gen_growth, title_text, ylabel_text, 13, 50)
age_pickup_lst = ['Canada', 'UK', 'Japan', 'Germany', 'Brazil', 

                  'Singapore', 'Egypt', 

                  'Belgium', 'New Zealand', 'Italy', 'Israel', 'Netherlands', 'Australia']

gen_pickup_lst = ['Canada', 'UK', 'Germany', 'Russia', 'Brazil',

                  'Czech Republic', 'Ireland', 

                  'Tunisia', 'Phillipines', 'Iran, Islamic Republic of...', 'Malaysia', 'Kenya']

gen_pickup_lst.extend(age_pickup_lst)

country_list = list(set(gen_pickup_lst))



sample_df = mu19_df[(mu19_df['Q3'].isin(country_list)) &

                    (mu19_df['Q2']=='Female') & (mu19_df['Q1'].isin(elder_age_lst))]

usa_df = mu19_df[(mu19_df['Q3']=='United States of America') & 

                 (mu19_df['Q2']=='Female') & (mu19_df['Q1'].isin(elder_age_lst))]



target_df = pd.DataFrame(sample_df['Q3'].value_counts())

target_df = pd.concat([target_df,pd.DataFrame(target_df.sum(axis=0),columns=['Grand Total']).T])

target_df = pd.concat([target_df, pd.DataFrame(usa_df['Q3'].value_counts())])

target_df.columns=['Count of Elderly Female Kaggler']

target_df.index.name = 'Country'



xy = [[0.25, 0.55], [0.35, 0.6], [0.25, 0.7], [0.25, 0.8], [0.3, 0.4], [0.4, 0.5], [0.45, 0.6], [0.45, 0.7],

[0.55, 0.55],[0.6, 0.65], [0.65, 0.75], [0.45, 0.8], [0.75, 0.4]]



xy = pd.DataFrame(xy, columns=['x', 'y'])

country_df = pd.concat([target_df.drop(index='Grand Total').reset_index(), xy], axis=1)
plt.figure(figsize=(5,5))

cmap = sns.cubehelix_palette(dark=.4, light=.7, as_cmap=True)

g = sns.scatterplot(country_df['x'], country_df['y'], size=country_df['Count of Elderly Female Kaggler'], 

                hue=country_df['Count of Elderly Female Kaggler'], palette=cmap, sizes=(10,5000),

               legend=False)

import matplotlib.patches as patches

patches.Circle(xy=(0.0, 0.0), radius=0.2, fc='g', ec='r')



for p1, p2, c, n in zip(country_df['x'], country_df['y'], country_df['Country'], country_df['Count of Elderly Female Kaggler']):

    plt.text(p1, p2+0.03, c, horizontalalignment='center', verticalalignment='center', fontdict={'size':9})

    plt.text(p1, p2, str(n), horizontalalignment='left', verticalalignment='center', fontdict={'size':9})

plt.ylim(0.2, 1.0)

plt.xlim(0.1, 1.05)



plt.xlabel('')

plt.ylabel('')



g.set(xticklabels=[])

g.set(yticklabels=[])



plt.title('Sample Countries & USA')

    

#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=11)



plt.show()
deg_level={'No formal education past high school':0,

           'Some college/university study without earning a bachelor’s degree':1,

           'Bachelor’s degree':2,

           'Master’s degree':3,

           'Doctoral degree':4,

           'Professional degree':5}
def compare_item(val, v_name, val_dic, v_level):

    s = pd.DataFrame(sample_df[val].value_counts(normalize=True).sort_values(ascending=True)).reset_index()

    s.columns=[v_name, 'sample']

    u = pd.DataFrame(usa_df[val].value_counts(normalize=True).sort_values(ascending=True)).reset_index()

    u.columns=[v_name, 'usa']

    s_u = pd.merge(s, u, on=v_name, how='outer')

    s_u[v_level] = 0

    for k in val_dic.keys():

        s_u.loc[s_u[s_u[v_name]==k].index, [v_level]] = val_dic[k]

    s_u_df = s_u.sort_values(v_level, ascending=False)

    s_u_df.fillna(0, inplace=True)

                             

    return s_u_df
def draw_pie(wdf, labels, titletext, v):

    

    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,5))



    ax.set_aspect('equal')

    

    col=cm.Spectral(np.arange(len(wdf))/float(len(wdf)))

    (_, textsL, autotextsL) = axL.pie(wdf['sample'], 

        autopct=lambda p: '{:.1f}%'.format(p) if p >= v else '', colors=col,

        #autopct='%1.1f%%', 

        startangle=90, pctdistance=0.7,radius=1.25, counterclock=False,

        #textprops={'rotation':0, 'color': "b", 'weight': "bold"},

        wedgeprops={'linewidth': 1, 'edgecolor':"white"})



    (_, textsR, autotextsR)  = axR.pie(wdf['usa'], 

        autopct=lambda p: '{:.1f}%'.format(p) if p >= v else '', colors=col,

        #autopct='%1.1f%%', 

        startangle=90, pctdistance=0.7,radius=1.25, counterclock=False,

        #textprops={'rotation':0, 'color': "black", 'weight': "bold"},

        wedgeprops={'linewidth': 1, 'edgecolor':"white"})



    box = axR.get_position()

    axR.set_position([0.5, box.y0, box.width, box.height])

    axR.legend(labels=labels, loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 12})



    axL.set_title('Sample_Countries')

    axR.set_title('USA')

    fig.suptitle(titletext, fontsize=20)



    plt.show()
s_u_df_Q4 = compare_item('Q4', 'degree', deg_level, 'd_no')

labels = s_u_df_Q4['degree']

titletext = 'Degree comparison'

draw_pie(s_u_df_Q4, labels, titletext, 2)
role_level={'Statistician':11,'Research Scientist':10,

'Data Scientist':9,'Data Analyst':8,'Data Engineer':7,'Business Analyst':6,'Product/Project Manager':5,

'Software Engineer':4,'DBA/Database Engineer':3, 'Student':2,'Other':1,'Not employed':0}

s_u_df_Q5 = compare_item('Q5', 'role', role_level, 'r_no')

labels = s_u_df_Q5['role']

titletext = 'Role comparison'

draw_pie(s_u_df_Q5, labels, titletext, 2)
def multiple_ans(wdf, val_n, k, flg):

    tate_df = pd.DataFrame()

    for i in range(1,k+1):

        n = val_n + str(i)

        tate_df = pd.concat([tate_df, wdf[n]])

    tate_df.columns = [flg]

    m_ans = pd.DataFrame(tate_df[flg].value_counts())

    m_ans.reset_index(inplace=True)

    m_ans.columns = ['part', flg]

    

    return m_ans
s_Q9 = multiple_ans(sample_df, 'Q9_Part_', 8, 'sample')

u_Q9 = multiple_ans(usa_df, 'Q9_Part_', 8, 'usa')

s_u_df_Q9 = pd.merge(s_Q9, u_Q9, on='part', how='outer')

labels=['Build prototypes to explore applying ML to new areas',

       'Analyze and understand data to influence product or business decisions',

       'Build/run a ML service that operationally improvement',

       'Build/run the data infrastructure for storing, analyzing, and operationalizing',

       'Do research that advances the state of the art of ML',

       'Experimentation and iteration to improve existing ML models',

       'None of these activities are an important part of my role at work',

       'Other']

titletext='Activities comparison'

s_u_df_Q9.fillna(0, inplace=True)
draw_pie(s_u_df_Q9, labels, titletext,2)
s_Q10 = pd.DataFrame(sample_df['Q10'].value_counts(normalize=True).sort_values(ascending=True)).reset_index()

s_Q10.columns=['money', 'count']

s_Q10['type'] = 'sample'

u_Q10 = pd.DataFrame(usa_df['Q10'].value_counts(normalize=True).sort_values(ascending=True)).reset_index()

u_Q10.columns=['money', 'count']

u_Q10['type'] = 'usa'

s_u_Q10 = pd.concat([s_Q10, u_Q10]).reset_index(drop=True)



import re

s_u_Q10['level'] = 0

for i in range(len(s_u_Q10)):

    content = s_u_Q10['money'][i]

    pattern = '[0-9]+,[0-9]{3}'

    if content == "$0-999":

        s_u_Q10.loc[i, ['level']] = 0

    else:

        result = re.match(pattern, content)

        ss = result.group().replace(",", "")

        s_u_Q10.loc[i, ['level']]= ss

s_u_Q10['level'] = s_u_Q10['level'].astype(int)

s_u_Q10.sort_values('level', inplace=True)

s_u_Q10['rate'] = s_u_Q10['count']*100
fig = plt.figure(figsize=(15,3))



ax = fig.add_subplot(1, 1, 1)



sns.barplot(ax=ax, x='level', y='rate', data=s_u_Q10, hue='type', palette='hot')

#ax.set_xticklabels(s_u_Q10['level'], rotation='vertical')

ax.set_xlabel('Yearly compensation')

ax.set_ylabel('Parcentage(%)')

ax.set_title('Compensation comparison')

plt.show()
spent_level = {'$0 (USD)':0,

               '$1-$99':1,

               '$100-$999':2,

               '$1000-$9,999':3,

               '$10,000-$99,999':4,

               '> $100,000 ($USD)':5}



s_u_df_Q11 = compare_item('Q11', 'spend', spent_level, 'sp_no')

labels = s_u_df_Q11['spend']

titletext = 'Money spent comparison'

draw_pie(s_u_df_Q11, labels, titletext,2)
write_level = {'I have never written code':0, '< 1 years':1, '1-2 years':2, '3-5 years':3,

'5-10 years':4, '10-20 years':5, '20+ years':6 }

s_u_df_Q15 = compare_item('Q15', 'time', write_level, 'ti_no')

labels = s_u_df_Q15['time']

titletext = 'Writing code time comparison'

draw_pie(s_u_df_Q15, labels, titletext, 2)
s_Q13 = multiple_ans(sample_df, 'Q13_Part_', 12, 'sample')

u_Q13 = multiple_ans(usa_df, 'Q13_Part_', 12, 'usa')

s_u_df_Q13 = pd.merge(s_Q13, u_Q13, on='part', how='outer')



labels=s_u_df_Q13['part']

titletext='Learning course comparison'

s_u_df_Q13.fillna(0, inplace=True)

draw_pie(s_u_df_Q13, labels, titletext, 2)
lange_level = {'Python':8, 'R':7, 'C++':6, 'SQL':5, 'MATLAB':4, 'Bash':3, 'Java':2, 'Javascript':1, 'Other':0}

s_u_df_Q19 = compare_item('Q19', 'lange', lange_level, 'la_no')

labels = s_u_df_Q19['lange']

titletext = 'Recommended language comparison'

draw_pie(s_u_df_Q19, labels, titletext, 3)
s_Q12 = multiple_ans(sample_df, 'Q12_Part_', 12, 'sample')

u_Q12 = multiple_ans(usa_df, 'Q12_Part_', 12, 'usa')

s_u_df_Q12 = pd.merge(s_Q12, u_Q12, on='part', how='outer')



labels=s_u_df_Q12['part']

titletext='Media sources comparison'

s_u_df_Q12.fillna(0, inplace=True)

draw_pie(s_u_df_Q12, labels, titletext, 2)
s_Q18 = multiple_ans(sample_df, 'Q18_Part_', 12, 'sample')

u_Q18 = multiple_ans(usa_df, 'Q18_Part_', 12, 'usa')

s_u_df_Q18 = pd.merge(s_Q18, u_Q18, on='part', how='outer')

labels=s_u_df_Q18['part']



titletext='Programming languages comparison'

s_u_df_Q18.fillna(0, inplace=True)

draw_pie(s_u_df_Q18, labels, titletext, 3)
s_Q20 = multiple_ans(sample_df, 'Q20_Part_', 12, 'sample')

u_Q20 = multiple_ans(usa_df, 'Q20_Part_', 12, 'usa')

s_u_df_Q20 = pd.merge(s_Q20, u_Q20, on='part', how='outer')



labels=s_u_df_Q20['part']

titletext='Visualization libraries or tools'

s_u_df_Q20.fillna(0, inplace=True)

draw_pie(s_u_df_Q20, labels, titletext, 2)
s_Q24 = multiple_ans(sample_df, 'Q24_Part_', 12, 'sample')

u_Q24 = multiple_ans(usa_df, 'Q24_Part_', 12, 'usa')

s_u_df_Q24 = pd.merge(s_Q24, u_Q24, on='part', how='outer')

labels=s_u_df_Q24['part']

titletext=' ML algorithms  comparison'

s_u_df_Q24.fillna(0, inplace=True)

draw_pie(s_u_df_Q24, labels, titletext, 3)
s_Q28 = multiple_ans(sample_df, 'Q28_Part_', 12, 'sample')

u_Q28 = multiple_ans(usa_df, 'Q28_Part_', 12, 'usa')

s_u_df_Q28 = pd.merge(s_Q28, u_Q28, on='part', how='outer')



labels=[s_u_df_Q28['part'][i].strip() for i in range(len(s_u_df_Q28))]

titletext='ML frameworks  comparison'

s_u_df_Q28.fillna(0, inplace=True)

draw_pie(s_u_df_Q28, labels, titletext, 3)