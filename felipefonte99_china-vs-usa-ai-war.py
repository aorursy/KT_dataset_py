import pandas as pd

import plotly.offline as py

py.init_notebook_mode(connected=True)



gdp_2019 = pd.read_csv('../input/gdpimf/WEO_Data_Clean.csv', encoding='latin-1', sep='|')

worldmap = [dict(type = 'choropleth', locations = gdp_2019['Country'], locationmode = 'country names',

                 z = gdp_2019['2019'], colorscale = "Blues", reversescale = True, 

                 marker = dict(line = dict( width = 0.4)),

                 colorbar = dict(autotick = False, title = 'GDP<br>Billions USD'))]

layout = dict(title = 'Global GDP 2019 (IMF Estimates)', geo = dict(showframe = False, showcoastlines = False))

fig = dict(data=worldmap, layout=layout)

py.iplot(fig, validate=False)
# Import libraries

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import textwrap

        

# Import 2019 Dataset

df_2019 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv', low_memory=False)

df_2019.columns = df_2019.iloc[0]

df_2019.drop([0], inplace=True)



# Import 2018 Dataset

df_2018 = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv', low_memory=False)

df_2018.columns = df_2018.iloc[0]

df_2018=df_2018.drop([0])



# Import 2017 Dataset

df_2017=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1', low_memory=False)



# Helper Variables

col_country = 'In which country do you currently reside?'

df_2019_cu = df_2019[(df_2019[col_country] == 'United States of America') | (df_2019[col_country] == 'China')]

df_2019_china = df_2019[df_2019[col_country] == 'China']

df_2019_usa = df_2019[df_2019[col_country] == 'United States of America']



# Helper Functions

def create_pie(col, explode_usa='', explode_china='', limit=1000):

    """

    Function designed to create pie plots comparing China and USA.

    col = Column of the dataframe. Type: str.

    explode_usa = Tuple with values to explode in the USA's pie plot. Type: tuple.

    explode_china = Tuple with values to explode in the China's pie plot. Type: tuple.

    limit = The top values to show. Type: int.

    """

    labels_usa = df_2019_usa[col].value_counts()[:limit].index

    sizes_usa = df_2019_usa[col].value_counts()[:limit].values

    

    labels_china = df_2019_china[col].value_counts()[:limit].index

    sizes_china = df_2019_china[col].value_counts()[:limit].values

    

    fig = plt.figure(figsize=(12,7))



    ax1 = fig.add_axes([0, 0, 1, 1], aspect=1)

    if explode_usa != '':

        ax1.pie(sizes_usa, explode=explode_usa, labels=labels_usa, autopct='%1.1f%%', startangle=45)

    else:

        ax1.pie(sizes_usa, labels=labels_usa, autopct='%1.1f%%', startangle=45)

    ax2 = fig.add_axes([0.7, 0, 1, 1], aspect=1)

    if explode_china != '':

        ax2.pie(sizes_china, explode=explode_china, labels=labels_china, autopct='%1.1f%%', startangle=45)

    else:

        ax2.pie(sizes_china, labels=labels_china, autopct='%1.1f%%', startangle=45)

    ax1.set_title('USA')

    ax2.set_title('China')

    plt.show()



def create_countplot(col, x_label, text_wrap=1000):

    """

    Function designed to create countplots.

    col = Column of the dataframe. Type: str.

    x_label = Label of the X axis. Type: str.

    text_wrap = Maximum number of letters of the words. Type: int.

    """

    plt.figure(figsize=(14,6))

    ax = sns.countplot(x=col, data=df_2019_cu, palette='rainbow', hue=col_country,

                      order=np.sort(df_2019_cu[col].unique()[~pd.isnull(df_2019_cu[col].unique())]))

    ax.set_xlabel(x_label)

    ax.legend(loc=1)

    _, labels = plt.xticks()

    x_axis=range(len(labels))

    plt.xticks(x_axis, [textwrap.fill(label.get_text(), text_wrap) for label in labels], rotation = 0, horizontalalignment="center")



    for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()+5), ha='center')

    plt.show()



def create_barplot(col_start, col_end, x_label, text_warp=1000):

    """

    Function designed to create countplots.

    col_start = Start column index of the dataframe. Type: int.

    col_end = End column index of the dataframe. Type: int.

    x_label = Label of the X axis. Type: str.

    text_wrap = Maximum number of letters of the words. Type: int.

    """

    names = []

    values = []

    countries = []

    for col_index in range (col_start, col_end):

        try:

            names.append(df_2019_usa.iloc[:,col_index].value_counts().index[0])

            values.append(df_2019_usa.iloc[:,col_index].value_counts().values[0])

            countries.append('United Stades')

        except:

            pass

        try:

            names.append(df_2019_china.iloc[:,col_index].value_counts().index[0])

            values.append(df_2019_china.iloc[:,col_index].value_counts().values[0])

            countries.append('China')

        except:

            pass

    temp_df = pd.DataFrame()

    temp_df['Names'] = names

    temp_df['Values'] = values

    temp_df['In which country do you currently reside?'] = countries



    plt.figure(figsize=(14,6))

    ax = sns.barplot(x='Names', y='Values', data=temp_df, palette='rainbow', hue=col_country,

                      order=np.sort(temp_df['Names'].unique()[~pd.isnull(temp_df['Names'].unique())]))

    ax.set_xlabel(x_label)

    ax.legend(loc=1)

    _, labels = plt.xticks()

    x_axis=range(len(labels))

    plt.xticks(x_axis, [textwrap.fill(label.get_text(), text_warp) for label in labels], rotation = 0, horizontalalignment="center")

    for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()+8), ha='center')

    plt.show()
plt.figure(figsize=(9,5))

ax = sns.countplot(x=col_country, data=df_2019_cu, palette='rainbow')

ax.set_title('Country')

ax.set_ylabel('Respondents')

ax.set_xlabel('')

for p in ax.patches:

        ax.annotate(str(p.get_height()) + ' (' + '{:.1f}%'.format(p.get_height()/len(df_2019_cu)*100) + ')', (p.get_x()+p.get_width()/2, p.get_height()-250), ha='center')

plt.show()
countries_2019 = df_2019[col_country]



labels = countries_2019.value_counts()[:10].index

sizes = countries_2019.value_counts()[:10].values

explode = (0, 0.07, 0, 0, 0, 0, 0.07, 0, 0, 0)



fig1, ax1 = plt.subplots(figsize=(11,8))

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)

ax1.axis('equal')

ax1.set_title('Top 10 Respondents', y=1.1)

plt.show()
print(f'{countries_2019.value_counts().values[1]/len(countries_2019)*100:.2f}% of the respondents are from USA')

print(f'{countries_2019.value_counts().values[6]/len(countries_2019)*100:.2f}% of the respondents are from China')
replace_dict = {"United States of America":"United States", "People 's Republic of China":"China"}

df_2017['Country'].replace(replace_dict,inplace=True)

df_2018['In which country do you currently reside?'].replace(replace_dict,inplace=True)



respondents = []

respondents.append(len(df_2017[df_2017['Country'] == 'United States']))

respondents.append(len(df_2017[df_2017['Country'] == 'China']))

respondents.append(len(df_2018[df_2018[col_country] == 'United States']))

respondents.append(len(df_2018[df_2018[col_country] == 'China']))

respondents.append(len(df_2019_usa))

respondents.append(len(df_2019_china))



temp_df = pd.DataFrame()

temp_df['Year'] = [2017, 2017, 2018, 2018, 2019, 2019]

temp_df['Respondents'] = respondents

temp_df['Country'] = ['United States', 'China', 'United States', 'China', 'United States', 'China']



plt.figure(figsize=(9,5))

ax = sns.barplot(x='Year', y='Respondents', data=temp_df, palette='rainbow', hue='Country',

                  order=np.sort(temp_df['Year'].unique()))

ax.set_xlabel('Year')

ax.legend(loc=1)



for p in ax.patches:

    ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()+40), ha='center')

plt.show()
ax = sns.lmplot(x='Year', y='Respondents', data=temp_df, col='Country', size=5, aspect=1)
create_pie('What is your gender? - Selected Choice',(0, 0, 0.2, 0.9), (0, 0, 0))
col = 'What is your age (# years)?'

create_countplot(col, 'Age')
below_40 = ['18-21','22-24','25-29','30-34','35-39']

labels = ['Below 40','Above 40']

sizes_usa = [len(df_2019_usa[df_2019_usa[col].isin(below_40)]), len(df_2019_usa[df_2019_usa[col].isin(below_40) == False])]

sizes_china = [len(df_2019_china[df_2019_china[col].isin(below_40)]), len(df_2019_china[df_2019_china[col].isin(below_40) == False])]



fig = plt.figure(figsize=(10,6))



ax1 = fig.add_axes([0, 0, 1, 1], aspect=1)

ax1.pie(sizes_usa, labels=labels, autopct='%1.1f%%', startangle=45)

ax2 = fig.add_axes([0.7, 0, 1, 1], aspect=1)

ax2.pie(sizes_china, labels=labels, autopct='%1.1f%%', startangle=45)

ax1.set_title('USA')

ax2.set_title('China')

plt.show()
fig, ax = plt.subplots(1,2, figsize=(14, 6))

sns.countplot(x=col, data=df_2019_usa, palette='rainbow', hue='What is your gender? - Selected Choice',

                hue_order = ['Male', 'Female', 'Prefer not to say', 'Prefer to self-describe'],

                order=np.sort(df_2019_usa[col].unique()), ax=ax[0])

sns.countplot(x=col, data=df_2019_china, palette='rainbow', hue='What is your gender? - Selected Choice',

                hue_order = ['Male', 'Female', 'Prefer not to say'],

                order=np.sort(df_2019_china[col].unique()), ax=ax[1])

ax[0].set_title('USA')

ax[0].legend(loc=1)

ax[0].set_xlabel('Age')

ax[1].set_title('China')

ax[1].legend(loc=1)

ax[1].set_xlabel('Age')

for p in ax[0].patches:

    ax[0].annotate('{:.0f}'.format(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()+3), ha='center')

for p in ax[1].patches:

    ax[1].annotate('{:.0f}'.format(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()+3), ha='center')

fig.show()
create_countplot('What is the highest level of formal education that you have attained or plan to attain within the next 2 years?', 'Education', text_wrap=12)
create_barplot(35, 47, 'Platform', text_warp=10)
create_pie('Select the title most similar to your current role (or most recent title if retired): - Selected Choice', limit=10)
create_pie('What is the size of the company where you are employed?')
create_countplot('Does your current employer incorporate machine learning methods into their business?', 'Machine Learning into Business', text_wrap=20)
create_pie('What is your current yearly compensation (approximate $USD)?', limit=10)
create_pie('Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?')
col = 'How long have you been writing code to analyze data (at work or at school)?'

create_pie(col, explode_china=(0,0,0,0,0,0,0.3))
below_2 = ['I have never written code','< 1 years','1-2 years']

labels = ['Below 2 years','Above 2 years']

sizes_usa = [len(df_2019_usa[df_2019_usa[col].isin(below_2)]), len(df_2019_usa[df_2019_usa[col].isin(below_2) == False])]

sizes_china = [len(df_2019_china[df_2019_china[col].isin(below_2)]), len(df_2019_china[df_2019_china[col].isin(below_2) == False])]



fig = plt.figure(figsize=(10,6))



ax1 = fig.add_axes([0, 0, 1, 1], aspect=1)

ax1.pie(sizes_usa, labels=labels, autopct='%1.1f%%', startangle=45)

ax2 = fig.add_axes([0.7, 0, 1, 1], aspect=1)

ax2.pie(sizes_china, labels=labels, autopct='%1.1f%%', startangle=45)

ax1.set_title('USA')

ax2.set_title('China')

plt.show()
create_pie('For how many years have you used machine learning methods?')
below_2 = ['< 1 years','1-2 years']

labels = ['Below 2 years','Above 2 years']

sizes_usa = [len(df_2019_usa[df_2019_usa[col].isin(below_2)]), len(df_2019_usa[df_2019_usa[col].isin(below_2) == False])]

sizes_china = [len(df_2019_china[df_2019_china[col].isin(below_2)]), len(df_2019_china[df_2019_china[col].isin(below_2) == False])]



fig = plt.figure(figsize=(10,6))



ax1 = fig.add_axes([0, 0, 1, 1], aspect=1)

ax1.pie(sizes_usa, labels=labels, autopct='%1.1f%%', startangle=45)

ax2 = fig.add_axes([0.7, 0, 1, 1], aspect=1)

ax2.pie(sizes_china, labels=labels, autopct='%1.1f%%', startangle=45)

ax1.set_title('USA')

ax2.set_title('China')

plt.show()
create_barplot(110, 115, 'Hardware Used on a Regular Basis')
create_pie('Have you ever used a TPU (tensor processing unit)?', (0,0,0,0,0.3), (0,0,0,0,0.3))
create_barplot(82, 94, 'Programming Language Used on a Regular Basis')
create_barplot(56, 68, 'IDE Used on a Regular Basis', 12)
create_barplot(233, 245, 'Database Used on a Regular Basis', 11)
create_barplot(168, 180, 'Cloud Used on a Regular Basis', 11)