from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

import missingno as msno 
data = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv', index_col='Unnamed: 0')

print(data.shape)

data.head()
data.columns
msno.matrix(data, figsize=(15, 3), fontsize=10)

plt.title('Missing Value?')

plt.show()
data['Job Title'] = data['Job Title'].str.lower()



banned = ['data', 'analyst']





def filter(x):

    x = x.replace('/', " ").replace("-", " ")

    return ' '.join([item for item in x.split() if item not in banned])





data["filter1"] = data["Job Title"].apply(filter)
data.filter1.value_counts()[1:15].sort_values(ascending=True).plot(

    kind='barh', title='Title after replacing Data Analyst')



plt.show()
junior = ['jr.', 'junior', 'entry', 'intern', 'jr']

senior = ['sr.', 'senior', 'lead', 'sr']





def check(x):

    for item in x.split():

        if item in junior:

            return 'Junior'

        elif item in senior:

            return 'Senior'

    return 'Not Specified'





data['experience'] = data["filter1"].apply(check)

data.experience.value_counts().plot(kind='bar', title='Jobs For?')

plt.show()
data.query('experience == "Junior"')['Job Title'].value_counts()[:5]
data[['Salary_lowerlimit', 'Salary_upperlimit']] = data['Salary Estimate'].str.split().str.get(0).str.split("-", expand = True)



data.Salary_lowerlimit = data.Salary_lowerlimit.apply(lambda x: x[1:-1])

data.Salary_upperlimit = data.Salary_upperlimit.apply(lambda x: x[1:-1])



oddoneout = data.query('Salary_lowerlimit == ""').index[0] ## this data have salary estimate -1

lowermean = data[data.index != oddoneout].Salary_lowerlimit.astype('int').mean()

uppermean = data[data.index != oddoneout].Salary_upperlimit.astype('int').mean()



data.at[oddoneout, 'Salary_lowerlimit'] = lowermean

data.at[oddoneout, 'Salary_upperlimit'] = uppermean



data.Salary_lowerlimit = data.Salary_lowerlimit.astype('int')

data.Salary_upperlimit = data.Salary_upperlimit.astype('int')





bins = np.linspace(20, 200, 20)

plt.hist(data.Salary_lowerlimit, bins, alpha=0.5, label='Lower Limit', rwidth=0.93)

plt.hist(data.Salary_upperlimit, bins, alpha=0.5, label='Upper Limit', rwidth=0.93)

plt.legend(loc='upper right')

plt.title('Distribution of Salary')

plt.xlabel('Salary(in K)')

plt.show()
data.groupby('experience').agg({'Salary_lowerlimit': ['mean', 'median', 'max', 'min'], 'Salary_upperlimit' : ['mean', 'median', 'max', 'min']})
data['Company_name_strip'] = data['Company Name'].str.split('\n').str.get(0)





print("Total No. of Companies - " , (data['Company Name'].nunique()))

print("Total No. of Companies after strip - " , (data['Company_name_strip'].nunique()))
df = data.groupby(['Company_name_strip']).agg({'Company Name' : 'nunique'})

duplicate_name = df[df["Company Name"] > 1].index

print(duplicate_name)
data[data.Company_name_strip.isin(duplicate_name)].groupby(

    ['Company_name_strip', "Company Name", 'Location', "Headquarters"]).size()
data['Company_name_strip'].value_counts()[:10].plot(kind = 'bar', title = "Company having opening").set(ylabel = 'No. of Jobs')

plt.show()
data.query('experience == "Junior"').groupby(['Company Name']).agg({'Salary_lowerlimit': 'mean', 'Salary_upperlimit':'mean'}

                                    ).sort_values(('Salary_upperlimit'), ascending=False)[:20].plot(kind='bar',

                                    title='Company searching Young Data Analyst', figsize=(20, 5)).set(ylabel="Salary(K)")

plt.show()
data.groupby(['Company_name_strip']).get_group('Cognoa')
df = data.Location.str.split(',', expand=True)

df2 = data.Headquarters.str.split(',', expand=True)



df.loc[df[2] == ' CO', 0] = "Greenwood Village, Arapahoe"

df.loc[df[2] == ' CO', 1] = " CO"





df2.loc[df2[2] == " NY", 1] = " NY"

df2.loc[df2[1] == " 061", 1] = " NY"





data[['Place', 'State']] = df.loc[:, [0, 1]]

data[['Headquarters_place', 'Headquarters_state_or_country']] = df2.loc[:, [0, 1]]





data.State = data.State.str.split(" ").str.get(1)
foreign = df2[df2[1].str.len() > 3][1].unique()

data['Company_origin'] = data.Headquarters_state_or_country.apply(

    lambda x: "Foreign" if x in foreign else 'Domestic')



data.groupby('Company_origin').agg({'Company Name': 'nunique'})
#Lets findout

oddoneout = data.groupby('Company_name_strip').agg({'Company_origin': 'nunique'}).query('Company_origin == 2').index

data[data['Company_name_strip'].isin(oddoneout)]
data.query('Company_origin == "Foreign"').groupby('Headquarters_state_or_country').agg({'Company_name_strip': "nunique"}).sort_values(

    by='Company_name_strip', ascending=False).plot(kind='bar', title="Foreign Company Count in US", legend = False).set(xlabel = 'Country')

plt.show()
# Let's see some Companies from India

data.query('Headquarters_state_or_country == " India"')['Company_name_strip'].unique()
print('Mean Salary Lower limit of Indian Company in Us - ', data.query('Headquarters_state_or_country == " India"')['Salary_lowerlimit'].mean())

print('Mean Salary Upper limit of Indian Company in Us - ', data.query('Headquarters_state_or_country == " India"')['Salary_upperlimit'].mean())
! pip install chart_studio
import chart_studio.plotly as py 

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



df = data.State.value_counts().reset_index()



df2 = dict(type='choropleth',

            locations = df['index'],

            locationmode = 'USA-states',

            colorscale = 'reds',

            z = df['State'],

            colorbar = {'title':"Job Count"}

            )

layout = dict(title = 'Data Analytic Job',

              geo = dict(scope='usa')

             )

choromap = go.Figure(data = [df2],layout = layout)

iplot(choromap)
df = data.groupby('State').agg({'Salary_lowerlimit':'mean', 'Salary_upperlimit':"mean"}).reset_index()

df2 = dict(type='choropleth',

            locations = df['State'],

            locationmode = 'USA-states',

            colorscale = "reds",

            z = df['Salary_upperlimit'],

            colorbar = {'title':"Salary Mean"},)



layout = dict(title = 'Data Analytic Mean Salary',

              geo = dict(scope='usa')

             )

choromap = go.Figure(data = [df2],layout = layout)

iplot(choromap)
df = data.groupby(['State']).agg({'Rating': 'mean'}).reset_index()

df2 = dict(type='choropleth',

            locations = df['State'],

            locationmode = 'USA-states',

            colorscale = "reds",

            z = df['Rating'],

            colorbar = {'title':"Rating"},)



layout = dict(title = 'Data Analytic Companies Mean Rating',

              geo = dict(scope='usa')

             )

choromap = go.Figure(data = [df2],layout = layout)

iplot(choromap)
data['Max_size'] = data.Size.str.split(" employees").str.get(0).str.split(' to ').apply(lambda x: x[-1] if len(x) > 1 else x[0])





oddoneout = data.query('Max_size == "-1"')['Company Name'].unique()

data[data['Company Name'].isin(oddoneout)].Max_size.value_counts()  ## one of the company has 50 employees





## updating max size

company = data[data['Company Name'].isin(oddoneout)].query('Max_size == "50"')['Company Name'].iloc[0]

oddoneout = data.query('Max_size == "-1"').groupby('Company Name').get_group(company).index[0]

data.loc[oddoneout, "Max_size"] ="50"





df = data.Max_size.value_counts(normalize=True)*100



df.drop(df.index[[6, 8]], inplace=True) # some company employee records are unknown and some bymistake -1





df.plot.pie(title='Max Employee in a Company',

            autopct=lambda x: f"{round(x)}%", figsize=(5, 5)).set(ylabel="")

plt.show()
data.query('Founded != "-1"').Founded.hist(bins=20, grid=False, figsize=(20,5), color='#86bf91', zorder=2, rwidth=0.9)

plt.xlabel("Foundation of Company(Year)", labelpad=20, weight='bold', size=12)

plt.ylabel("No. of Company", labelpad=20, weight='bold', size=12)

plt.show()
age_bins = [0,1900, 1950,2000 , 2010]

labels = ["1900 and below", "1901 - 1950", " 1951 - 2000", "2000-2010"]

data['company_founded'] = pd.cut(data['Founded'], age_bins, labels=labels)





df = data.groupby(['State', 'company_founded']).size().groupby('State').cumsum().reset_index()

df.head()  # cummulative sum of no. of companies in a state
data_slider = []

for year_category in df['company_founded'].unique():

    df_year = df[df['company_founded'] == year_category]

    data_one_year = dict(

                        type='choropleth',

                        locations = df_year['State'],

                        z=df_year[0].astype(float),

                        locationmode = 'USA-states',

                        colorscale = "reds",

                        colorbar = {'title':'Count'}

                        )

    data_slider.append(data_one_year)

    

steps = []



for i in range(len(data_slider)):

    step = dict(method='restyle',

                args=['visible', [False] * len(data_slider)],

                label='Year {}'.format(labels[i]))

    step['args'][1][i] = True

    steps.append(step)



sliders = [dict(active=0, pad={"t": 1}, steps=steps)] 



layout = dict(title = 'Data Analytic Companies Founded', 

              geo=dict(scope='usa'), sliders=sliders)



fig = dict(data=data_slider, layout=layout)

iplot(fig, show_link = True)

data.groupby(['Company_origin', 'Type of ownership'])['Location'].nunique().plot(kind = 'barh')

plt.show()
data.groupby('Sector').size()[1:].sort_values().plot(kind = 'bar', title= 'Job Opening in sectors')

plt.show()
data.query('Sector != "-1"').groupby('Sector').agg({'Salary_lowerlimit': ['size', 'mean', 'median', 'max', 'min'], 'Salary_upperlimit': [

    'mean', 'median', 'max', 'min']}).sort_values(by=('Salary_lowerlimit', 'size'), ascending=False)
revenue_high = {

    'Unknown / Non-Applicable': 0,

     '$100 to $500 million (USD)' : 500,

     '$50 to $100 million (USD)' : 100,

     '$10+ billion (USD)' : 10000,

     '-1' : 0,

     '$10 to $25 million (USD)' : 25,

     '$2 to $5 billion (USD)' : 5000,

     '$1 to $5 million (USD)' : 5,

     '$25 to $50 million (USD)' : 50,

     'Less than $1 million (USD)' : 1,

     '$1 to $2 billion (USD)' : 2000,

     '$500 million to $1 billion (USD)' : 1000,

     '$5 to $10 million (USD)' : 10,

     '$5 to $10 billion (USD)' : 10000   

}



data["Revenue_millions"] = data.Revenue.map(revenue_high)
# correlation - revenue and salary

import seaborn as sns

sns.heatmap(data.query('Revenue_millions != 0').loc[:, [

            'Revenue_millions', 'Salary_lowerlimit', 'Salary_upperlimit']].corr(), annot=True)

plt.show()
data.query('Revenue_millions != 0').groupby('Revenue_millions').agg({'Salary_lowerlimit': ['size', 'mean', 'median', 'max', 'min'], 'Salary_upperlimit': [

    'mean', 'median', 'max', 'min']}).sort_index()
from wordcloud import WordCloud, STOPWORDS



def wordcloudplot(category):

    stopwords = set(STOPWORDS)

    text = " ".join(review for review in data[data.experience == category]['Job Description'].str.lower())

    text = " ".join([i.strip() for i in text.split(" ") if i.lower().strip() not in ['job', 'description', 'data', 'analyst', 'analysis', 'analytic']+ list(stopwords)])

    wordcloud = WordCloud(width=1600, height=800, background_color='white',max_words=150,prefer_horizontal=1,

                          stopwords=stopwords, min_font_size=20).generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.title(f'WordCloud :- Job Description for  {category}')

    plt.tight_layout(pad=0)

    plt.show()





category = 'Junior'

wordcloudplot(category)
category = 'Senior'

wordcloudplot(category)
category = 'Not Specified'

wordcloudplot(category)
stopwords = set(STOPWORDS)

text = " ".join(review for review in data['filter1'].str.lower())

wordcloud = WordCloud(width=1600, height=800, background_color='white',max_words=300,prefer_horizontal=1,

                          stopwords=stopwords, min_font_size=20).generate(text)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.title(f'WordCloud :- Job Title')

plt.tight_layout(pad=0)

plt.show()