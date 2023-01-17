import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, plot, iplot

init_notebook_mode(connected=False)

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)

import plotly_express as px

from wordcloud import WordCloud

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from gensim.models import Word2Vec

from keras.utils import to_categorical

from keras.layers import Dense, Dropout, LSTM, GlobalMaxPool1D, Activation

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.snowball import PorterStemmer

from sklearn import preprocessing
zomato_data = pd.read_csv('/kaggle/input/zomato-bangalore-restaurants/zomato.csv')

zomato_data.head()
zomato_data.shape
total_of_all = zomato_data.isnull().sum().sort_values(ascending=False)

percent_of_all = (zomato_data.isnull().sum()/zomato_data.isnull().count()).sort_values(ascending=False)

missing_data_test = pd.concat([total_of_all, percent_of_all], axis=1, keys=['Total', 'Percent'])

missing_data_test.head(10)
# Deleting Unnnecessary Columns



zomato_data = zomato_data.drop(['url','dish_liked','phone'], axis=1)
# Replace New by NaN



zomato_data["rate"] = zomato_data["rate"].replace("NEW", np.nan)

zomato_data.dropna(how="any", inplace=True)
# Changing the data type from approx_cost columns



zomato_data['approx_cost'] = zomato_data['approx_cost(for two people)'].astype(str).apply(lambda x: x.replace(',', ''))

zomato_data['approx_cost'] = zomato_data['approx_cost'].astype(float)
all_ratings = []



for name,ratings in tqdm(zip(zomato_data['name'],zomato_data['reviews_list'])):

    ratings = eval(ratings)

    for score, doc in ratings:

        if score:

            score = score.strip("Rated").strip()

            doc = doc.strip('RATED').strip()

            score = float(score)

            all_ratings.append([name,score, doc])
zomato_rating_data = pd.DataFrame(all_ratings,columns=['name','rating','review'])

zomato_rating_data['review'] = zomato_rating_data['review'].apply(lambda x : re.sub('[^a-zA-Z0-9\s]',"",x))
# most common type of restaurants 



restaurants_type_analysis = pd.DataFrame(zomato_data['rest_type'].value_counts().sort_values(ascending=False))

restaurants_type_analysis = restaurants_type_analysis.rename(columns={'rest_type':'count'})



trace = go.Bar(x = restaurants_type_analysis.index[:15],

              y = restaurants_type_analysis['count'][:15],

              marker = dict(color='rgba(125, 215, 180, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top 15 restaurants Type",

                  xaxis=dict(title='Type of restaurant',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Counts',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
franchises_analysis = pd.DataFrame(zomato_data['name'].value_counts().sort_values(ascending=False))

franchises_analysis = franchises_analysis.rename(columns={'name':'count'})



trace = go.Bar(x = franchises_analysis.index[:15],

              y = franchises_analysis['count'][:15],

              marker = dict(color='rgba(150, 200, 100, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top Franchises of Bangluru",

                  xaxis=dict(title='Franchises Name',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Counts',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
location_analysis = pd.DataFrame(zomato_data['location'].value_counts().sort_values(ascending=False))

location_analysis = location_analysis.rename(columns={'location':'count'})



trace = go.Bar(x = location_analysis.index[:15],

              y = location_analysis['count'][:15],

              marker = dict(color='rgba(125, 115, 180, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top Locations",

                  xaxis=dict(title='Location Name',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Frequency of Restaurants',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
trace = go.Pie(labels=['yes', 'No',], values=zomato_data['online_order'].value_counts())

data = [trace]

layout = go.Layout(title='Accepting vs Not Accepting online orders')

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace = go.Pie(labels=['yes', 'No',], values=zomato_data['book_table'].value_counts())

data = [trace]

layout = go.Layout(title='Booking of Table vs No Booking of Table')

fig = go.Figure(data=data, layout=layout)

iplot(fig)
zomato_data['rate'] = zomato_data['rate'].astype(str).apply(lambda x: x.split('/')[0])

while True:

    try:

        zomato_data['rate'] = zomato_data['rate'].astype(float)

        break

    except ValueError as e1:

        noise_entry = str(e1).split(":")[-1].strip().replace("'", "")

        print(f'Threating noisy entrance on rate: {noise_entry}')

        zomato_data['rate'] = zomato_data['rate'].apply(lambda x: x.replace(noise_entry, str(np.nan)))
high_rating_yes = zomato_data[(zomato_data["rate"] >= 4.5) & (zomato_data["online_order"] == "Yes")]

high_rating_no = zomato_data[(zomato_data["rate"] >= 4.5) & (zomato_data["online_order"] == "No")]



medium_rating_yes = zomato_data[(zomato_data["rate"] >= 3.5) & (zomato_data["rate"] <= 4.4) & (zomato_data["online_order"] == "Yes")]

medium_rating_no = zomato_data[(zomato_data["rate"] >= 3.5) & (zomato_data["rate"] <= 4.4) & (zomato_data["online_order"] == "No")]



low_rating_yes = zomato_data[(zomato_data["rate"] < 3.5) & (zomato_data["online_order"] == "Yes")]

low_rating_no = zomato_data[(zomato_data["rate"] < 3.5) & (zomato_data["online_order"] == "No")]
top_rating_yes = pd.DataFrame(high_rating_yes['name'].value_counts().sort_values(ascending=False))

top_rating_yes = top_rating_yes.rename(columns={'name':'count'})



trace = go.Bar(x = top_rating_yes.index[:15],

              y = top_rating_yes['count'][:15],

              marker = dict(color='rgba(250, 200, 150, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top Rating Frenchises with online order",

                  xaxis=dict(title='Frenchises Name',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Frequency',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
top_rating_no = pd.DataFrame(high_rating_no['name'].value_counts().sort_values(ascending=False))

top_rating_no = top_rating_no.rename(columns={'name':'count'})



trace = go.Bar(x = top_rating_no.index[:15],

              y = top_rating_no['count'][:15],

              marker = dict(color='rgba(200, 180, 250, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top Rating Frenchises without online order",

                  xaxis=dict(title='Frenchises Name',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Frequency',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
lower_rating_no = pd.DataFrame(low_rating_no['name'].value_counts().sort_values(ascending=False))

lower_rating_no = lower_rating_no.rename(columns={'name':'count'})



trace = go.Bar(x = lower_rating_no.index[:15],

              y = lower_rating_no['count'][:15],

              marker = dict(color='rgba(200, 50, 70, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top Low Rating Frenchises without online order",

                  xaxis=dict(title='Frenchises Name',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Frequency',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
lower_rating_yes = pd.DataFrame(low_rating_yes['name'].value_counts().sort_values(ascending=False))

lower_rating_yes = lower_rating_yes.rename(columns={'name':'count'})



trace = go.Bar(x = lower_rating_yes.index[:15],

              y = lower_rating_yes['count'][:15],

              marker = dict(color='rgba(100, 50, 270, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top Low Rating Frenchises with online order",

                  xaxis=dict(title='Frenchises Name',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Frequency',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# low budget restaurent

low_budget = zomato_data.groupby(['name','rest_type','cuisines', 'rate', 'reviews_list'])['approx_cost'].sum().sort_values(ascending=True).reset_index()

low_budget = low_budget[low_budget["approx_cost"] <= 1000]



# mid budget restaurent

mid_budget = zomato_data.groupby(['name','rest_type','cuisines', 'rate', 'reviews_list'])['approx_cost'].sum().sort_values(ascending=True).reset_index()

mid_budget = mid_budget[(mid_budget["approx_cost"] > 1000) & (mid_budget["approx_cost"] <= 3000)]



# High budget restaurent

high_budget = zomato_data.groupby(['name','rest_type','cuisines',  'rate', 'reviews_list'])['approx_cost'].sum().sort_values(ascending=True).reset_index()

high_budget = high_budget[(high_budget["approx_cost"] > 3000) & (high_budget["approx_cost"] <= 6000)]
low_budget_cuisines = pd.DataFrame(low_budget['cuisines'].value_counts().sort_values(ascending=False))

low_budget_cuisines = low_budget_cuisines.rename(columns={'cuisines':'count'})





trace = go.Bar(x = low_budget_cuisines.index[:15],

              y = low_budget_cuisines['count'][:15],

              marker = dict(color='rgba(200, 150, 270, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top Low Budget Cuisines",

                  xaxis=dict(title='Cuisines Name',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Frequency',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
low_budget_rest_type = pd.DataFrame(low_budget['rest_type'].value_counts().sort_values(ascending=False))

low_budget_rest_type = low_budget_rest_type.rename(columns={'rest_type':'count'})





trace = go.Bar(x = low_budget_rest_type.index[:15],

              y = low_budget_rest_type['count'][:15],

              marker = dict(color='rgba(120, 150, 120, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top Low Budget Restaurant type",

                  xaxis=dict(title='Restaurant type',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Frequency',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
low_budget_rest_name = pd.DataFrame(low_budget['name'].value_counts().sort_values(ascending=False))

low_budget_rest_name = low_budget_rest_name.rename(columns={'name':'count'})





trace = go.Bar(x = low_budget_rest_name.index[:15],

              y = low_budget_rest_name['count'][:15],

              marker = dict(color='rgba(50, 130, 90, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top Low Budget Restaurant",

                  xaxis=dict(title='Restaurant Name',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Frequency',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
medium_budget_cuisines = pd.DataFrame(mid_budget['cuisines'].value_counts().sort_values(ascending=False))

medium_budget_cuisines = medium_budget_cuisines.rename(columns={'cuisines':'count'})





trace = go.Bar(x = medium_budget_cuisines.index[:15],

              y = medium_budget_cuisines['count'][:15],

              marker = dict(color='rgba(150, 130, 90, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top Medium Budget Cuisines",

                  xaxis=dict(title='Cuisines Name',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Frequency',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

fig.update_layout(

    autosize=False,

    width=1200,

    height=700,)

iplot(fig)
medium_budget_rest_type = pd.DataFrame(mid_budget['rest_type'].value_counts().sort_values(ascending=False))

medium_budget_rest_type = medium_budget_rest_type.rename(columns={'rest_type':'count'})





trace = go.Bar(x = medium_budget_rest_type.index[:15],

              y = medium_budget_rest_type['count'][:15],

              marker = dict(color='rgba(120, 145, 170, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top Medium Budget Restaurant types",

                  xaxis=dict(title='Restaurant Type',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Frequency',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
medium_budget_rest_name = pd.DataFrame(mid_budget['name'].value_counts().sort_values(ascending=False))

medium_budget_rest_name = medium_budget_rest_name.rename(columns={'name':'count'})





trace = go.Bar(x = medium_budget_rest_name.index[:15],

              y = medium_budget_rest_name['count'][:15],

              marker = dict(color='rgba(190, 15, 70, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top Medium Budget Restaurant Names",

                  xaxis=dict(title='Restaurant Names',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Frequency',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
high_budget_cuisines = pd.DataFrame(high_budget['cuisines'].value_counts().sort_values(ascending=False))

high_budget_cuisines = high_budget_cuisines.rename(columns={'cuisines':'count'})





trace = go.Bar(x = high_budget_cuisines.index[:15],

              y = high_budget_cuisines['count'][:15],

              marker = dict(color='rgba(90, 135, 160, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top High Budget Cuisines Names",

                  xaxis=dict(title='Cuisines Names',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Frequency',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
high_budget_rest_type = pd.DataFrame(high_budget['rest_type'].value_counts().sort_values(ascending=False))

high_budget_rest_type = high_budget_rest_type.rename(columns={'rest_type':'count'})





trace = go.Bar(x = high_budget_rest_type.index[:15],

              y = high_budget_rest_type['count'][:15],

              marker = dict(color='rgba(90, 185, 10, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top High Budget Restaurant Type",

                  xaxis=dict(title='Restaurant Type',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Frequency',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
high_budget_rest_name = pd.DataFrame(high_budget['name'].value_counts().sort_values(ascending=False))

high_budget_rest_name = high_budget_rest_name.rename(columns={'name':'count'})





trace = go.Bar(x = high_budget_rest_name.index[:15],

              y = high_budget_rest_name['count'][:15],

              marker = dict(color='rgba(90, 185, 140, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top High Budget Restaurant Name",

                  xaxis=dict(title='Restaurant Names',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Frequency',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

fig.update_layout(

    autosize=False,

    width=1200,

    height=700,)

iplot(fig)
# sentimental analysis if review is greater than 3 then positive else negative



zomato_rating_data['sentiment'] = zomato_rating_data['rating'].apply(lambda x: 'positive' if int(x)>3 else 'negative')
zomato_rating_data.head()
review_text = []

for i in zomato_rating_data['review']:

    review_text.append(i.split())

print(review_text[:2])
w2vz_model = Word2Vec(review_text, size=50, workers=32, window=3, min_count=1)

print(w2vz_model)
zomato_token = Tokenizer(24512)

zomato_token.fit_on_texts(zomato_rating_data['review'])

zomato_review = zomato_token.texts_to_sequences(zomato_rating_data['review'])

zomato_review = pad_sequences(zomato_review)
label = preprocessing.LabelEncoder()

y = label.fit_transform(zomato_rating_data['sentiment'])

y = to_categorical(y)

print(y[:5])
X_train, X_test, y_train, y_test = train_test_split(np.array(zomato_review), y, test_size=0.2, stratify=y)
model = Sequential()

model.add(w2vz_model.wv.get_keras_embedding(True))

model.add(Dropout(0.2))

model.add(LSTM(50, return_sequences=True))

model.add(GlobalMaxPool1D())

model.add(Dropout(0.2))

model.add(Dense(50))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(2))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

model.summary()
model.fit(X_train, y_train, batch_size=1024, epochs=2, validation_data=(X_test, y_test))
labels = label.classes_

print(labels)
# check prediction



predicted = model.predict(X_test)
for i in range(40,60,2):

    print(zomato_rating_data['review'].iloc[i][:50], "...")

    print("Actual category: ", labels[np.argmax(y_test[i])])

    print("predicted category: ", labels[np.argmax(predicted[i])])