import pandas as pd

import numpy as np

import datetime

### Plots

import matplotlib.pyplot as plt

import squarify

import seaborn as sns

plt.style.use('fivethirtyeight')

#

from wordcloud import WordCloud, STOPWORDS 

# map

from shapely.geometry import Point

import geopandas as gpd

from geopandas import GeoDataFrame
df_covid = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')
df_covid.shape
df_covid.info()
df_covid.head(2)
print(df_covid.columns)
df_covid = df_covid.drop(['Unnamed: 33','Unnamed: 34','Unnamed: 35','Unnamed: 36','Unnamed: 37','Unnamed: 38','Unnamed: 39','Unnamed: 40','Unnamed: 41','Unnamed: 42','Unnamed: 43','Unnamed: 44'], axis = 1)

df_covid.shape
df_covid.isnull().sum()
df_covid.age.unique()
###### Drop Record where Age = Belgium

df_covid = df_covid[df_covid.age != "Belgium"]

# Clean records where Age is decimal

df_covid['age'] = np.where(((df_covid.age == '0.08333') | 

                            (df_covid.age == '0.58333') | 

                            (df_covid.age == '0.25')| 

                             (df_covid.age == '0.5')| 

                            (df_covid.age == '1.75')), '0',

                           df_covid.age )

# Clean records where Age is a range

df_covid['age'] = np.where(df_covid.age == '0-10', '5',

                  np.where(df_covid.age == '10-19', '15',

                  np.where(df_covid.age == '20-29', '25',

                  np.where(df_covid.age == '30-39', '35',

                  np.where(df_covid.age == '40-49', '45',

                  np.where(df_covid.age == '50-59', '55',

                  np.where(df_covid.age == '60-69', '65',

                  np.where(df_covid.age == '70-79', '75',

                  np.where(df_covid.age == '80-89', '85',

                  np.where(df_covid.age == '36-45', '40',

                  np.where(df_covid.age == '80-80', '80',

                  np.where(df_covid.age == '13-19', '16',

                  np.where(df_covid.age == '27-40', '33',

                  np.where(df_covid.age == '22-80', '51',

                  np.where(df_covid.age == '19-77', '48',

                  np.where(df_covid.age == '8-68', '38',

                  np.where(df_covid.age == '0-6', '3',

                  np.where(df_covid.age == '18-65', '41',

                  np.where(df_covid.age == '16-80', '48',

                  np.where(df_covid.age == '38-68', '53',

                  np.where(df_covid.age == '40-89', '64',

                  np.where(df_covid.age == '21-72', '46',

                  np.where(df_covid.age == '60-60', '60',

                  np.where(df_covid.age == '0-18', '9',  

                  np.where(df_covid.age == '23-72', '47',  

                                        df_covid.age)))))))))))))))))))))))))
df_covid.sex.unique()
# Drop record where sex = "4000"

#df_covid = df_covid[df_covid.sex != "4000"]

# Change record for unanimous values

df_covid['sex'] = np.where(df_covid.sex == 'Male', 'male',

                                      np.where(df_covid.sex == 'Female', 'female',

                                        df_covid.sex       ))

df_covid.sex.unique()
df_covid['date_confirmation'] .unique()
df_covid['date_confirmation'] = np.where(df_covid.date_confirmation == '25.02.2020-26.02.2020', '25.02.2020',

                                      df_covid.date_confirmation       )

df_covid.date_confirmation =pd.to_datetime(df_covid.date_confirmation, format = '%d.%m.%Y')
df_dates = pd.DataFrame(df_covid.groupby("date_confirmation")["ID"].count())

df_dates =  df_dates.reset_index()

df_dates.columns = ['date_confirmation','count']

df_dates['date_confirmation'] = pd.to_datetime(df_dates['date_confirmation'])
df_dates['month'] = np.where(df_dates.date_confirmation.dt.month == 1, 'jan',

                                      'feb' )
df_covid.lives_in_Wuhan.value_counts()
df_covid['wuhan_connection'] =  df_covid.lives_in_Wuhan.apply(lambda x: "No" if x in ['no',0,'No','live in Hangzhou','Xiantao City resident','thai national'] else "Yes")
geometry = [Point(xy) for xy in zip(df_covid['longitude'], df_covid['latitude'])]

gdf = GeoDataFrame(df_covid, geometry=geometry)  

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

gdf.plot(ax=world.plot(figsize=(15, 6)), marker='o', color='red', markersize=15);
pl = df_covid.country.value_counts().sort_values(ascending = False)[:10].plot(kind = "bar", figsize = (15,6), title = "Covid-19 - Most Affected Countries")

pl.grid(True)

pl.set_ylim(10,12000)

pl.set_xlabel("Country")

pl.set_ylabel("CasesCount")
pl = df_covid[(df_covid.country != "China") & (df_covid.country != "South Korea")]["country"].value_counts().sort_values()[:10].plot(kind = "bar", figsize = (15,6), title = "Covid-19 - Least Affected Countries", color = "orange")

pl.grid(True)

pl.set_ylim(0.8,2)

pl.set_xlabel("Country")

pl.set_ylabel("CasesCount")
pl = df_covid.sex.value_counts().sort_values(ascending = False).plot(kind = "bar", figsize = (15,6), title = "Covid-19 - GenderWise Statistics ", color = "aqua")

pl.grid(True)

pl.set_xlabel("Gender")

pl.set_ylabel("CasesCount")
plt.figure(figsize=(15,6))

sns.distplot(df_covid['age'], hist=True, kde=False,  color = 'blue',

             hist_kws={'edgecolor':'black'})

# Add labels

plt.title('Covid-19 - Agewise Affected people')

plt.xlabel('Age')

plt.ylabel('CasesCount')
pl = df_covid.city.value_counts().sort_values(ascending = False)[:20].plot(kind = "bar", figsize = (15,6), title = "Covid-19 - CityWise Statistics ",color = "green")

pl.grid(True)

pl.set_xlabel("City")

pl.set_ylabel("CasesCount")
comment_words = ' '

stopwords = set(STOPWORDS) 

# iterate through the csv file 

for val in df_covid[df_covid.symptoms.notnull()].symptoms: 

    # typecaste each val to string 

    val = str(val) 

    #print(val)

    # split the value 

    tokens = val.split()

    #print(tokens)

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower()

        for words in tokens: 

            comment_words = comment_words + words + ' '

        

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10,

                collocations=False).generate(comment_words)   

   

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
df_dates.groupby("month").sum().plot(kind = "bar",figsize = (15,6), title = "Covid- 19 - Monthwise Cases", color = "teal")

plt.grid(True)

plt.xlabel("Month")

plt.ylabel("CasesCount")
df_covid.wuhan_connection.value_counts().plot("bar",figsize = (15,6), title = "Covid- 19 - Wuhan Connection?", color = "yellow")

plt.grid(True)

plt.xlabel("Connection with Wuhan")

plt.ylabel("CasesCount")
stopwords = set(STOPWORDS)

addl_info = " ".join(info for info in df_covid[df_covid.additional_information.notnull()].additional_information)

wordcloud = WordCloud(width = 800, 

                      height = 800,

                      stopwords=stopwords, 

                      background_color="white" ,

                      max_words=1000,

                      min_font_size = 10,

                      collocations=False).generate(addl_info)



# Display the generated image:

# the matplotlib way:                   

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.tight_layout(pad = 0) 

plt.show()