#You know what this step is for

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from datetime import datetime

import seaborn as sns

import plotly.plotly as py

import re

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from PIL import Image

from wordcloud import WordCloud, STOPWORDS

init_notebook_mode(connected=True)



import os

reviews = pd.read_csv("../input/google-amazon-facebook-employee-reviews/employee_reviews.csv")

#Getting shape of the dataset

print("Shape is",reviews.shape)
reviews.info()
# Preprocessing steps

reviews.columns = reviews.columns.str.replace('-','_')

reviews.head(10)
#Since column Unnamed:0 is nothing but the index, therefore dropping it.

reviews = reviews.drop("Unnamed: 0",axis=1)



#Plotting the number of reviews present for each company

counts = reviews['company'].value_counts()

counts = pd.DataFrame(counts)

trace1 = go.Bar(

                x = counts.index,

                y = counts.company,

                name = "Number of reviews for each company",

                marker = dict(color='rgb(26, 118, 255)',

                             line=dict(color='black',width=1.5)),

                text = counts.company)

data = [trace1]

layout = go.Layout(barmode = "group",title='Number of reviews of each company')

fig = go.Figure(data = data, layout = layout)

iplot(fig)
#Plotting number of reviews of each company by country

cmp_lc= reviews[["company","location"]]

cmp_lc= cmp_lc[cmp_lc.location != 'none']

cmp_lc= cmp_lc.reset_index(drop=True)



def find_country(s):

    try:

        x = re.search(r'\((.*?)\)',s).group(1)

        return str(x)

    except:

        return str("United States")

    

cmp_lc['country'] = cmp_lc['location'].apply(find_country)

cmp_lc = cmp_lc.drop('location',axis=1)

cmp_lc = cmp_lc.groupby(["company", "country"]).size().reset_index()

cmp_lc = cmp_lc.rename(columns={0: 'total_reviews'})

companies = list(cmp_lc.company.unique())

company_data = []

v = True





for i in companies:

    if i!='amazon':

        v=False

    data_upd = [dict(type='choropleth',

                     locations = cmp_lc[cmp_lc['company']==i]['country'],

                     z = cmp_lc[cmp_lc['company']==i]['total_reviews'],

                     locationmode='country names',

                     text = cmp_lc[cmp_lc['company']==i]['country'],

                     visible = v,

                     colorbar = dict(title = "Numbers of Reviews"))]

    

    company_data.extend(data_upd)



companies = [x.capitalize() for x in companies]



# set menues inside the plot

steps = []

cp = 0

for i in range(0,len(company_data)):

    step = dict(method = "restyle",

                args = ["visible", [False]*len(company_data)],

                label = companies[cp]) 

    step['args'][1][i] = True

    steps.append(step)

    cp += 1

    



sliders = [dict(active = 6,

                currentvalue = {"prefix": "Company: "},

                pad = {"t": 50},

                steps = steps)]



# Set the layout

layout = dict(title = 'Number of reviews country specific for each company',

              sliders = sliders)



fig = dict(data=company_data, layout=layout)

iplot( fig, filename='companies-cloropleth-map')
#Getting wordclouds for each company

stopwords = set(STOPWORDS)

extras = ["great","work","company","place","good"]

stopwords.update(extras)

companies = list(reviews.company.unique())

for company in companies:

    stopwords.add(company)



def wordclouds(df,companies):

    for company in companies:

        temp = df.loc[df["company"]==company]

        text = " ".join(str(review) for review in temp.summary)

        # Create and generate a word cloud image:

        wordcloud = WordCloud(stopwords=stopwords,collocations = False).generate(text)

        # Display the generated image:

        plt.imshow(wordcloud, interpolation='bilinear')

        plt.axis("off")

        plt.title(company.upper())

        plt.show()

        

wordclouds(reviews,companies)
apple_mask = np.array(Image.open("../input/mask-images/apple.png"))

amazon_mask = np.array(Image.open("../input/mask-images/amazon.png"))

netflix_mask = np.array(Image.open("../input/mask-images/netflix.png.png"))



def transform_format(val):

    if val == 0:

        return 255

    elif val == 26:

        return 255

    else:

        return val



# Transforming mask into a new one that will work with the function:

t_apple_mask = np.ndarray((apple_mask.shape[0],apple_mask.shape[1]), np.int32)

t_amazon_mask = np.ndarray((amazon_mask.shape[0],amazon_mask.shape[1]), np.int32)

t_netflix_mask = np.ndarray((netflix_mask.shape[0],netflix_mask.shape[1]), np.int32)



for i in range(len(apple_mask)):

    t_apple_mask[i] = list(map(transform_format,apple_mask[i]))



for i in range(len(amazon_mask)):

    t_amazon_mask[i] = list(map(transform_format,amazon_mask[i]))



for i in range(len(netflix_mask)):

    t_netflix_mask[i] = list(map(transform_format,netflix_mask[i]))



stopwords = set(STOPWORDS)

extras = ["great","work","company","place","good"]

stopwords.update(extras)

companies = ["apple","amazon","microsoft","netflix"]

for company in companies:

    stopwords.add(company)



temp = reviews.loc[reviews["company"]=="apple"]

text = " ".join(str(review) for review in temp.summary)

# Create and generate a word cloud image:

wc1 = WordCloud(background_color="white", max_words=500, mask=t_apple_mask,

               stopwords=stopwords, contour_width=5)

wc1.generate(text)

temp = reviews.loc[reviews["company"]=="amazon"]

text = " ".join(str(review) for review in temp.summary)

wc2 = WordCloud(background_color="white", max_words=500, mask=t_amazon_mask,

               stopwords=stopwords, contour_width=5)

wc2.generate(text)



temp = reviews.loc[reviews["company"]=="netflix"]

text = " ".join(str(review) for review in temp.summary)

wc3 = WordCloud(background_color="white", max_words=1000, mask=t_netflix_mask,

               stopwords=stopwords, contour_width=5)

wc3.generate(text)





# fig, ((ax1, ax2),ax3)= plt.subplots(2, 2, sharex=True, sharey=True,figsize=(15,15))

# ax1.imshow(wc1, interpolation='bilinear')

# ax1.axis("off")

# ax2.imshow(wc2, interpolation='bilinear')

# ax2.axis("off")

# ax3.imshow(wc3, interpolation='bilinear')

# ax3.axis("off")



gs = gridspec.GridSpec(2, 2)



fig = plt.figure(figsize=(40,40))

ax1 = fig.add_subplot(gs[0, 0]) # row 0, col 0

ax1.imshow(wc1, interpolation='bilinear')

ax1.axis("off")



ax2 = fig.add_subplot(gs[0, 1]) # row 0, col 1

ax2.imshow(wc2, interpolation='bilinear')

ax2.axis("off")



ax3 = fig.add_subplot(gs[1, :]) # row 1, span all columns

ax3.imshow(wc3, interpolation='bilinear')

ax3.axis("off")
# Preprocessing done for datetime data

reviews.dates = reviews.dates.str.strip()

reviews = reviews[reviews.dates != 'None']

reviews = reviews[reviews.dates != 'Jan 0, 0000']

reviews = reviews[reviews.dates != 'Nov 0, 0000']

reviews.dates = [datetime.strptime(x, '%b %d, %Y') for x in reviews.dates]

reviews['year'] = pd.DatetimeIndex(reviews.dates).year  
cmp_lc = reviews[["company","dates"]]

cmp_lc = cmp_lc.reset_index(drop=True)

cmp_lc.sort_values(by='dates')

cmp_lc = cmp_lc.groupby(["company", "dates"]).size().reset_index()

cmp_lc = cmp_lc.rename(columns={0: 'total_reviews'})



amazon = go.Scatter(

    x=cmp_lc[cmp_lc['company']=='amazon']['dates'],

    y=cmp_lc[cmp_lc['company']=='amazon']['total_reviews'],

    name = "Amazon",

    line = dict(color = '#17BECF'),

    opacity = 0.8)



microsoft = go.Scatter(

    x=cmp_lc[cmp_lc['company']=='microsoft']['dates'],

    y=cmp_lc[cmp_lc['company']=='microsoft']['total_reviews'],

    name = "Microsoft",

    line = dict(color = '#7F7F7F'),

    opacity = 0.8)



apple = go.Scatter(

    x=cmp_lc[cmp_lc['company']=='apple']['dates'],

    y=cmp_lc[cmp_lc['company']=='apple']['total_reviews'],

    name = "Apple",

    line = dict(color = '#0fac1f'),

    opacity = 0.8)



google = go.Scatter(

    x=cmp_lc[cmp_lc['company']=='google']['dates'],

    y=cmp_lc[cmp_lc['company']=='google']['total_reviews'],

    name = "Google",

    line = dict(color = '#6335c3'),

    opacity = 0.8)



facebook = go.Scatter(

    x=cmp_lc[cmp_lc['company']=='facebook']['dates'],

    y=cmp_lc[cmp_lc['company']=='facebook']['total_reviews'],

    name = "Facebook",

    line = dict(color = '#808000'),

    opacity = 0.8)





data = [amazon,microsoft,apple,google,facebook]



layout = dict(

    title='Analysing any trend between number of reviews and dates',

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label='1m',

                     step='month',

                     stepmode='backward'),

                dict(count=6,

                     label='6m',

                     step='month',

                     stepmode='backward'),

                dict(step='all')

            ])

        ),

        rangeslider=dict(

            visible = True

        ),

        type='date'

    )

)



fig = dict(data=data, layout=layout)

iplot(fig)
def find_employee_type(s):

    x = re.search(r'([^-]*)-',s).group(1)

    return str(x)

    

reviews['employee_type'] = reviews['job_title'].apply(find_employee_type)

cmp_lc= reviews[["company","employee_type"]] 

cmp_lc = cmp_lc.groupby(["company", "employee_type"]).size().reset_index()

cmp_lc = cmp_lc.rename(columns={0: 'total_reviews_by_title'})



years = list(cmp_lc.company.unique())

company_data = []

v = True



for i in years:

    if i!='amazon':

        v=False

    data_upd = [dict(type='bar',

                     visible = v,

                     x = cmp_lc[cmp_lc['company']==i]['employee_type'],

                     y = cmp_lc[cmp_lc['company']==i]['total_reviews_by_title'],

                     textposition = 'auto',

                     marker=dict(

                     color='rgb(158,202,225)',

                     line=dict(

                         color='rgb(8,48,107)',

                         width=1.5),

                     ),

                 opacity=0.6)]

    

    company_data.extend(data_upd)



years = [x.capitalize() for x in years]



# set menus inside the plot

steps = []

yr = 0

for i in range(0,len(company_data)):

    step = dict(method = "restyle",

                args = ["visible", [False]*len(company_data)],

                label = years[yr]) 

    step['args'][1][i] = True

    steps.append(step)

    yr += 1

    



sliders = [dict(active = 6,

                currentvalue = {"prefix": "Company: "},

                pad = {"t": 50},

                steps = steps)]



# Set the layout

layout = dict(title = 'Do current employees review the comapnies more than the ex-employees?',

              sliders = sliders)



fig = dict(data=company_data, layout=layout)

iplot(fig)
#Function to plot

def plotit(df,company):

    cmp_lc = df[["year","company","overall_ratings","work_balance_stars","culture_values_stars",

                      "carrer_opportunities_stars","comp_benefit_stars","senior_mangemnet_stars"]]

    cmp_lc = cmp_lc.loc[cmp_lc["company"]==company]

    cmp_lc = cmp_lc.drop('company',axis=1)

    cmp_lc = cmp_lc[cmp_lc.work_balance_stars != 'none']

    cmp_lc = cmp_lc[cmp_lc.culture_values_stars != 'none']

    cmp_lc = cmp_lc[cmp_lc.carrer_opportunities_stars != 'none']

    cmp_lc = cmp_lc[cmp_lc.comp_benefit_stars != 'none']

    cmp_lc = cmp_lc[cmp_lc.senior_mangemnet_stars != 'none']

    cmp_lc[["overall_ratings","work_balance_stars","culture_values_stars","carrer_opportunities_stars",

            "comp_benefit_stars","senior_mangemnet_stars"]] = cmp_lc[["overall_ratings","work_balance_stars","culture_values_stars",

            "carrer_opportunities_stars","comp_benefit_stars","senior_mangemnet_stars"]].apply(pd.to_numeric)

    cmp_lc = cmp_lc.groupby('year').agg({'overall_ratings':np.median,'work_balance_stars':np.median,

                                        'culture_values_stars':np.median,'carrer_opportunities_stars':np.median,

                                        'comp_benefit_stars':np.median,'senior_mangemnet_stars':np.median}).reset_index()



    t1 = go.Scatter(

        x = cmp_lc.year,

        y = cmp_lc.overall_ratings,

        mode = "lines+markers",

        name = "Overall Ratings",

        marker = dict(color = 'rgba(240,230,140 0.8)'),

    )



    t2 = go.Scatter(

        x = cmp_lc.year,

        y = cmp_lc.work_balance_stars,

        mode = "lines+markers",

        name = "Work Balance Stars",

        marker = dict(color = 'rgba(211,211,211, 0.8)'),

    )

    t3 = go.Scatter(

        x = cmp_lc.year,

        y = cmp_lc.culture_values_stars,

        mode = "lines+markers",

        name = "Culture Value Stars",

        marker = dict(color = 'rgba(220,165,112)'),

    )

    t4 = go.Scatter(

        x = cmp_lc.year,

        y = cmp_lc.carrer_opportunities_stars,

        mode = "lines+markers",

        name = "Career Opportunities Stars",

        marker = dict(color = 'rgba(218,165,32, 0.8)'),

    )

    t5 = go.Scatter(

        x = cmp_lc.year,

        y = cmp_lc.comp_benefit_stars,

        mode = "lines+markers",

        name = "Compensation and Benefits Stars",

        marker = dict(color = 'rgba(128,128,128, 0.8)'),

    )

    t6 = go.Scatter(

        x = cmp_lc.year,

        y = cmp_lc.senior_mangemnet_stars,

        mode = "lines+markers",

        name = "Senior Management Stars",

        marker = dict(color = 'rgba(144,89,35, 0.8)'),

    )



    data = [t1,t2,t3,t4,t5,t6]

    layout = dict(title = 'How did ratings change over the years for %s?'%company,

                  xaxis= dict(title= 'Years',ticklen= 5,zeroline= False)

                 )

    fig = dict(data = data, layout = layout)

    iplot(fig)

plotit(reviews,'amazon')
plotit(reviews,'microsoft')
plotit(reviews,'apple')
plotit(reviews,'google')
plotit(reviews,'facebook')
plotit(reviews,'netflix')