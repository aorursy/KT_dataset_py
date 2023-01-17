import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import missingno as msno

import seaborn as sns

from fbprophet import Prophet

from wordcloud import WordCloud, STOPWORDS

%matplotlib inline
! pip install bar_chart_race

import bar_chart_race as bcr
data = pd.read_csv('../input/windows-store/msft.csv', parse_dates=['Date'])

data.dropna(inplace=True)

data.tail()
data.columns
msno.matrix(data, figsize=(15, 3), fontsize=10)

plt.title('Missing Value Check')

plt.show()
df = data.groupby([data.Date.dt.strftime('%Y-%m'), 'Category']).size().unstack().fillna(0).cumsum()

df.set_index(pd.to_datetime(df.index), inplace=True)



def summary(values, ranks):

    total_apps = values.sum()

    s = f'Total APPS - {total_apps:,.0f}'

    return {'x': .92, 'y': .05, 's': s, 'ha': 'right', 'size': 8}





bcr.bar_chart_race(df,

                   title='Category Count Race',

                   steps_per_period=10,

                   period_length=800,

                   period_summary_func=summary,

                   period_fmt='%b, %Y',

                   interpolate_period=True,

                   )
data.groupby('Category').size().plot(kind='bar', title='Count Plot')

plt.show()
data['Price_type'] = data.Price.apply(

    lambda x: "Free" if x == "Free" else 'Paid')

df = data.Price_type.value_counts(normalize=True)*100

df.plot.pie(title='App Type',

            autopct=lambda x: f"{round(x, 1)}%", figsize=(5, 5)).set(ylabel="")

plt.show()
data.groupby(['Category', 'Price_type']).size().unstack().plot(

    kind='bar', stacked=True, title="Count Plot")

plt.show()
df = data.query('Price_type == "Paid"').reset_index()

df.Price = df.Price.str.split().apply(

    lambda x: x[1].replace(",", "")).astype('float')



df.boxplot('Price', 'Category', rot=90, grid=True,  return_type='axes')

plt.suptitle('')

plt.show()
sns.heatmap(df.loc[:, ['Rating', 'No of people Rated',

                       'Price']].corr(),  cmap="YlGnBu",  annot=True)

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(20, 5))

data.boxplot('Rating', 'Category', ax=ax[0], grid=False, rot=90)

data.boxplot('No of people Rated', 'Category', ax=ax[1], grid=False, rot=90)

plt.show()
data['months'] = data.Date.dt.strftime("%m")



data['date_num'] = data.Date.dt.strftime('%d')



data['day'] = data.Date.dt.strftime('%A')



data['year'] = data.Date.dt.strftime('%Y')
data.year = data.year.astype('int')

data.months = data.months.astype('int')

data.date_num = data.date_num.astype('int')
data.query('year == 2020 and months > 6').months.value_counts().sort_index().cumsum().plot(title = 'Future Data').set(xlabel = "2020's months")

plt.show()
sns.heatmap(data.loc[:, ['Rating', 'No of people Rated', 'year',

                         'day', 'date_num', 'months']].corr(),  cmap="YlGnBu",   annot=True)

plt.show()
data.groupby(['year', 'Price_type']).size().unstack().plot(

    kind='line', figsize=(10, 5), title='Count ')

plt.show()
data.query('Price_type == "Free"').groupby(['year', 'Category']).size(

).unstack().plot(kind='line', figsize=(20, 30), subplots=True)

plt.show()
data.query('Price_type == "Paid"').groupby(['year', 'Category']).size(

).unstack().plot(kind='line', figsize=(20, 6), subplots=True)

plt.show()
df = data.sort_values(['Category', 'Date']).reset_index()

df["date_diff"] = df.Date.diff() / np.timedelta64(1, 'D')

df.drop(df.query('date_diff < 0').index, inplace=True)
fig, ax = plt.subplots(1, 2, figsize=(20, 5))

df.query('year > 2010').boxplot(

    'date_diff', 'Category', ax=ax[0], rot=90)



df.query('year > 2010 and date_diff <40').boxplot(

    'date_diff', 'Category', ax=ax[1], grid=False, rot=90)



plt.title('Zoomed')

plt.suptitle('Difference in days for new App')

plt.show()
data.day.value_counts().plot.bar(title='App Launch Day')

plt.show()
df = data.groupby(data.Date.dt.strftime('%Y-%m')).size()

df.plot(title = 'Time series of No. of app develop in a month')

plt.show()



df = df.reset_index()

df.columns=["ds","y"]
model=Prophet()

model.fit(df)

future_dates=model.make_future_dataframe(periods=30,freq = 'M' )

prediction=model.predict(future_dates)

prediction.tail()
fig1 = model.plot(prediction)
fig2 = model.plot_components(prediction)
data.groupby(['Price_type']).agg({'No of people Rated': ['sum','count','mean', 'median']})
data.groupby(['Price_type', 'Rating']).agg({'No of people Rated': ['sum','count','mean', 'median']})
def wordcloudplot(category):

    stopwords = set(STOPWORDS)

    text = " ".join(review for review in data[data.Category == category].Name.str.lower())

    text = text.replace('by', "").replace('and', "")

    print(f"There are {len(text)} words in the combination of all {category} App Name.")



    wordcloud = WordCloud(width=800, height=800, background_color='white',max_words=150,prefer_horizontal=1,

                          stopwords=stopwords, min_font_size=20).generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.title(f'WordCloud for {category} APP Name')

    plt.tight_layout(pad=0)

    plt.show()





category = 'Music'

wordcloudplot(category)
category = 'Books'

wordcloudplot(category)
category = 'Social'

wordcloudplot(category)