import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

import matplotlib

matplotlib.rcParams.update({'font.size': 15})



np.set_printoptions(suppress=True)



sns.set_palette("muted")

sns.set_style("whitegrid")



from IPython.display import display



def display_dataframe(df):

    with pd.option_context("display.max_columns", 1000):

        with pd.option_context("display.max_rows",5):

            display(df)
dataset = pd.read_csv("../input/ted_main.csv")

display_dataframe(dataset)
dataset.describe()
dataset['main_speaker'].value_counts().sort_values().tail(10).plot(kind = "bar", title = "Most TED talks by a speaker",

                                                                   color = "black");
dataset['event'].unique()
# creating a new feature, is_subevent. If the event is TED then is_subevent is 0, else 1. 

dataset['is_subevent'] = dataset['event'].apply(lambda x: 1 if "TEDx" in x else 0)

dataset['is_subevent'].value_counts().plot(kind = "bar", color = "black", title = "How many are TED / Non-TED events");
dataset['sub_event'] = dataset['event'].apply(lambda x: x[4:] if "TEDx" in x else "Nan")



inspect_df = dataset['sub_event'].value_counts().sort_values().tail(10).to_frame().reset_index()

inspect_df.columns = ['name', 'n']

inspect_df = inspect_df.drop(inspect_df.index[9])



fig = plt.figure(figsize = (15,8))

sns.barplot(x = "name", y = "n", data = inspect_df).set_title("Where were the most Non-TED events hosted at")

plt.ylim(0,50);
# create dates from timestamps. 

import datetime as dt

def create_date_fields():

    global dataset

    film_date = dataset['film_date'] = dataset['film_date'].apply(lambda x: dt.datetime.fromtimestamp(int(x)).strftime('%d-%m-%Y'))

    film_date = pd.to_datetime(film_date)

    

    published_date = dataset['published_date'] = dataset['published_date'].apply(lambda x: dt.datetime.fromtimestamp(int(x)).strftime('%d-%m-%Y'))

    published_date = pd.to_datetime(published_date)



    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek'):

        dataset['filmdate'+n] = getattr(film_date.dt,n.lower())

        dataset['published_date'+n] = getattr(published_date.dt, n.lower())

    temp = pd.DataFrame({'film_date': film_date, 

                         'published_date': published_date})

    dataset['time_difference'] = temp.apply(lambda x: (x['published_date'] - x['film_date']).days , axis = 1)

create_date_fields()

display_dataframe(dataset)
#views and counts by year, month, date, day etc. 

inspect_df = dataset['published_dateYear'].value_counts().sort_values().tail(10).to_frame().reset_index()

inspect_df.columns = ['published_dateYear', 'n']

inspect_df.sort_values(['published_dateYear'], inplace = True)



fig = plt.figure(figsize = (18,8))

fig.add_subplot(121)

sns.barplot(x = 'published_dateYear', y = 'n', data = inspect_df).set_title("Number of talks by Year")



fig.add_subplot(122)

dataset.groupby(['published_dateYear'])['views'].median().plot(kind = "bar", title = "Median of views by Year released");

from pandas.api.types import is_string_dtype

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence



selected_features = ['comments', 'duration', 'event', 'languages','main_speaker','num_speaker','ratings','speaker_occupation',

                    'views','is_subevent','filmdateYear', 'published_dateYear', 'filmdateMonth', 

                     'published_dateMonth','filmdateWeek', 'published_dateWeek', 'filmdateDay',

       'published_dateDay', 'filmdateDayofweek', 'published_dateDayofweek', 'time_difference']

model_data_x = dataset[selected_features]

model_data_y = dataset['views']

for n,c in model_data_x.items():

    if is_string_dtype(c):

        model_data_x[n] = c.astype('category').cat.as_ordered()

        model_data_x[n] = model_data_x[n].cat.codes + 1



# Not doing an hyper parameter tuning. We are only interested in feature interaction. 

m = GradientBoostingRegressor(random_state = 123)

m.fit(model_data_x, model_data_y);



my_plots = plot_partial_dependence(m,       

                                   features=[11],

                                   X = model_data_x,          

                                   grid_resolution=20) 
#views and counts by year, month, date, day etc.

Month_index = {1:'Jan', 2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

dataset['published_dateMonth'] = dataset['published_dateMonth'].apply(lambda x: Month_index[x])



inspect_df = dataset['published_dateMonth'].value_counts().sort_values().tail(10).to_frame().reset_index()

inspect_df.columns = ['published_dateMonth', 'n']

inspect_df.sort_values(['published_dateMonth'], inplace = True)



fig = plt.figure(figsize = (18,8))

fig.add_subplot(121)

sns.barplot(x = 'published_dateMonth', y = 'n', data = inspect_df).set_title("Number of talks by Month")



fig.add_subplot(122)

dataset.groupby(['published_dateMonth'])['views'].median().plot(kind = "bar", title = "Median of views by Month released");

plt.figure(figsize = (20,20))

my_plots = plot_partial_dependence(m,       

                                   features=[13],

                                   X = model_data_x) 

inspect_df = dataset['published_dateWeek'].value_counts().sort_values().to_frame().reset_index()

inspect_df.columns = ['published_dateWeek', 'n']

inspect_df.sort_values(['published_dateWeek'], inplace = True)



fig = plt.figure(figsize = (35,15))

fig.add_subplot(121)

sns.barplot(x = 'published_dateWeek', y = 'n', data = inspect_df).set_title("Number of talks by Week-of-Year")



fig.add_subplot(122)

dataset.groupby(['published_dateWeek'])['views'].median().plot(kind = "bar", title = "Median of views by Week-of-Year released");

#views and counts by year, month, date, day etc.

Day_index = {0:"Sun",1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat"}

dataset['published_dateDayofweek'] = dataset['published_dateDayofweek'].apply(lambda x: Day_index[x])

inspect_df = dataset['published_dateDayofweek'].value_counts().sort_values().tail(10).to_frame().reset_index()

inspect_df.columns = ['published_dateDayofweek', 'n']

inspect_df.sort_values(['published_dateDayofweek'], inplace = True)



fig = plt.figure(figsize = (18,8))

fig.add_subplot(121)

sns.barplot(x = 'published_dateDayofweek', y = 'n', data = inspect_df).set_title("Number of talks by Day of Week")



fig.add_subplot(122)

dataset.groupby(['published_dateDayofweek'])['views'].median().plot(kind = "bar", title = "Median of views by Day released");

plt.figure(figsize = (20,20))

my_plots = plot_partial_dependence(m,       

                                   features=[19],

                                   X = model_data_x) 



fig = plt.figure(figsize = (15,10))

fig.add_subplot(221)

dataset.groupby(['is_subevent'])['comments'].median().plot(kind = "bar", title = "Median of Comments", color = "black")



fig.add_subplot(222)

dataset.groupby(['is_subevent'])['duration'].median().plot(kind = "bar", title = "Median of Duration")



fig.add_subplot(223)

dataset.groupby(['is_subevent'])['views'].median().plot(kind = "bar", title = "Median of Views", color = "black")



fig.add_subplot(224)

dataset.groupby(['is_subevent'])['time_difference'].median().plot(kind = "bar", title = "Median of Time_Difference");
# How big is the speaker name? Any interesting trends there?

fig = plt.figure(figsize = (20,10))

fig.add_subplot(121)

dataset['main_speaker_namelength'] = dataset['main_speaker'].apply(lambda x: len(x.split(" ")))

dataset['main_speaker_namelength'].value_counts().plot(kind = "bar", title = "Any trend of speaker name length?", 

                                                      color = "black")



fig.add_subplot(122)

dataset.groupby(['main_speaker_namelength'])['views'].median().plot(kind = "bar",title = "Length of Speaker name vs Median Views");

from string import punctuation



def extract_talkemotion(string):

    start_index = string.find(":", 10)

    end_index = string.find(",", start_index + 1)

    def strip_punctuation(s):

        return "".join(c for c in s if c not in punctuation)

    return strip_punctuation(string[start_index + 1 : end_index].strip())



dataset['talk_emotion'] = dataset['ratings'].apply(extract_talkemotion)



fig = plt.figure(figsize = (20,10))

fig.add_subplot(121)

dataset['talk_emotion'].value_counts().sort_values().tail(10).plot(kind = "bar", color = "black",

                                                                   title = "What did listeners think of talks?")



fig.add_subplot(122)

dataset.groupby(['talk_emotion'])['views'].median().sort_values().tail(10).plot(kind = "bar",title = "What talks garnered most views?");

model_data_x['talk_emotion'] = dataset['ratings'].apply(extract_talkemotion)

emotion_type = model_data_x['talk_emotion'] = model_data_x['talk_emotion'].astype('category').cat.as_ordered()

emotion_code = model_data_x['talk_emotion'] = model_data_x['talk_emotion'].cat.codes + 1

    

m = GradientBoostingRegressor(random_state = 1234)

m.fit(model_data_x, model_data_y);



my_plots = plot_partial_dependence(m,       

                                   features=[21],

                                   X = model_data_x)

# This is the type of emotion that is getting most views. 

emotion_type.cat.categories[-5:]
def extract_talkgenre(string):

    start_index = string.find("'")

    end_index = string.find("'", start_index + 1)

    def strip_punctuation(s):

        return "".join(c for c in s if c not in punctuation)

    return strip_punctuation(string[start_index + 1 : end_index].strip())

dataset['talk_genre'] = dataset['tags'].apply(extract_talkgenre)



fig = plt.figure(figsize = (20,10))

fig.add_subplot(121)

dataset['talk_genre'].value_counts().sort_values().tail(10).plot(kind = "bar", color = "black",

                                                                 title = "What were the talks mostly about?")



fig.add_subplot(122)

dataset.groupby(['talk_genre'])['views'].median().sort_values().tail(10).plot(kind = "bar", 

                                                                              title = "Which genre had more views?");

    
model_data_x['talk_genre'] = dataset['tags'].apply(extract_talkgenre)

genre_names = model_data_x['talk_genre'] = model_data_x['talk_genre'].astype('category').cat.as_ordered()

genre_codes = model_data_x['talk_genre'] = model_data_x['talk_genre'].cat.codes + 1

    

m = GradientBoostingRegressor(random_state = 12345)

m.fit(model_data_x, model_data_y);



my_plots = plot_partial_dependence(m,       

                                   features=[22],

                                   X = model_data_x) 

genre_names.cat.categories[100:]