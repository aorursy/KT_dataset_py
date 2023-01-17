import pandas as pd

import numpy as np



import plotly_express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go



import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS



import tensorflow as tf

import tensorflow_hub as hub

from sklearn.decomposition import PCA



from nltk.sentiment.vader import SentimentIntensityAnalyzer



from plotly.offline import init_notebook_mode

init_notebook_mode()
data = pd.read_csv('../input/avatar-the-last-air-bender/avatar_data.csv')

series = pd.read_csv('../input/avatar-the-last-air-bender/series_names.csv')

avatar = pd.read_csv('../input/avatar-the-last-air-bender/avatar.csv', encoding = 'latin-1')
avatar['imdb_rating'] = avatar['imdb_rating'].fillna(9.7)
fig = px.bar(series, x = 'book', y = 'series_rating', template = 'simple_white', color_discrete_sequence=['#f18930'] * 3 ,

             opacity = 0.6, text = 'series_rating', category_orders={'book':['Water','Earth','Fire']}, 

            title = 'IMDB Rating Across Seasons')

fig.add_layout_image(

        dict(

            source="https://i.imgur.com/QWoqOZd.jpg",

            xref="x",

            yref="y",

            x=-0.5,

            y=10,

            sizex=3,

            sizey=10,

            opacity = 0.7,

            sizing="stretch",

            layer="below")

)

fig.show()
fig = px.bar(data, x = 'Unnamed: 0', y = 'imdb_rating',color = 'book',hover_name='book_chapt',template = 'simple_white',

             color_discrete_map={'Fire':'#cd0000', 'Water':'#3399ff', 'Earth':'#663307'},labels={'imdb_rating':'IMDB Ratig','Unnamed: 0':'Episode'})

fig.show()
director_counts = pd.DataFrame(data['director'].value_counts()).reset_index()

director_counts.columns = ['Director Name', 'Number of Episdoes']



fig = make_subplots(rows=1, cols=2,specs=[[{'type':'bar'}, {'type':'pie'}]], horizontal_spacing=0.2)



directorColors = ['#adbce6'] * 7

directorColors[5] = ['#ba72d4']

director_rating = pd.DataFrame(data.groupby('director')['imdb_rating'].mean()).reset_index().sort_values(by = 'imdb_rating')

trace0 = go.Bar(y = director_rating['director'], x = director_rating['imdb_rating'], orientation='h',

                hovertext=director_rating['imdb_rating'],name = 'Director Average Ratings',

               marker_color = directorColors )

fig.add_trace(trace0, row = 1, col = 1)



trace1 = go.Pie(values= director_counts['Number of Episdoes'],labels = director_counts['Director Name'],name = 'Director Number of Episodes')

fig.add_trace(trace1, row = 1, col = 2)



fig.update_layout(showlegend = False, title = {'text':'Directors and Their Average Rating', 'x':0.5}, template = 'plotly_white')

fig.show()
character_dialogues = pd.DataFrame({'Character':[], 'Number of Dialogues':[],'Book' : []})

for book in ['Water', 'Earth', 'Fire']:

    temp = pd.DataFrame(avatar[avatar['book'] == book]['character'].value_counts()).reset_index()

    temp.columns = ['Character', 'Number of Dialogues']

    temp['Book'] = book

    temp = temp.sort_values(by = 'Number of Dialogues', ascending = False)

    character_dialogues = pd.concat([character_dialogues, temp])
important_characters = ['Aang', 'Katara', 'Zuko', 'Sokka','Toph','Iroh','Azula']
bookColor = {

    'Fire':'#cd0000', 

    'Water':'#3399ff', 

    'Earth':'#663307'

}

fig = make_subplots(rows = 1, cols = 3, subplot_titles=['Water','Earth','Fire'])

for i, book in enumerate(['Water','Earth', 'Fire']):

    temp = character_dialogues[(character_dialogues['Character'] != 'Scene Description') & (character_dialogues['Book'] == book)]

    trace = go.Bar(x = temp.iloc[:10][::-1]['Number of Dialogues'].values, y = temp.iloc[:10][::-1]['Character'].values,

                   orientation = 'h', marker_color = bookColor[book], name = book,opacity=0.8)

    fig.add_trace(trace, row = 1, col = i+1)

fig.update_layout(showlegend = False, template = 'plotly_white', title = 'Characters with Most Dialogues in Each Book')

fig.show()
fig = px.bar(character_dialogues[character_dialogues['Character'].isin(important_characters)],template = 'gridon',title = 'Important Characters Number of Dialogues each season',

             x = 'Number of Dialogues', y = 'Character', orientation = 'h', color='Book',barmode = 'group',

             color_discrete_map={'Fire':'#cd0000', 'Water':'#3399ff', 'Earth':'#663307'})

fig.add_layout_image(

    dict(

        source="https://vignette.wikia.nocookie.net/avatar/images/1/12/Azula.png",

        x=0.25,

        y=0.9,

    ))

fig.add_layout_image(

    dict(

        source="https://vignette.wikia.nocookie.net/avatar/images/4/46/Toph_Beifong.png",

        x=0.42,

        y=0.77,

    ))

fig.add_layout_image(

    dict(

        source="https://vignette.wikia.nocookie.net/avatar/images/c/c1/Iroh_smiling.png",

        x=0.35,

        y=0.6,

    ))

fig.add_layout_image(

    dict(

        source="https://vignette.wikia.nocookie.net/avatar/images/4/4b/Zuko.png",

        x=0.62,

        y=0.47,

    ))



fig.add_layout_image(

    dict(

        source="https://vignette.wikia.nocookie.net/avatar/images/c/cc/Sokka.png",

        x=0.85,

        y=0.32,

    ))

fig.add_layout_image(

    dict(

        source="https://static.wikia.nocookie.net/loveinterest/images/c/cb/Avatar_Last_Airbender_Book_1_Screenshot_0047.jpg",

        x=0.85,

        y=0.18,

    ))

fig.add_layout_image(

    dict(

        source="https://comicvine1.cbsistatic.com/uploads/scale_small/11138/111385676/7212562-5667359844-41703.jpg",

        x=1.05,

        y=0.052,

    ))

fig.update_layout_images(dict(

        xref="paper",

        yref="paper",

        sizex=0.09,

        sizey=0.09,

        xanchor="right",

        yanchor="bottom"

))



fig.show()
chapter_dialogues = pd.DataFrame({'Chapter':[], 'Number of Dialogues':[],'Book' : []})

dialogue_df = avatar[avatar['character']!='Scene Description']

for book in ['Water', 'Earth', 'Fire']:

    temp = pd.DataFrame(dialogue_df[(dialogue_df['book'] == book)]['chapter'].value_counts()).reset_index()

    temp.columns = ['Chapter', 'Number of Dialogues']

    temp['Book'] = book

    chapter_dialogues = pd.concat([chapter_dialogues, temp])

chapter_dialogues = chapter_dialogues.sort_values(by = 'Number of Dialogues')
colors = []

for i in range(20):

    if(chapter_dialogues.iloc[i]['Book'] == 'Fire'):

        colors.append('#cd0000')

    elif(chapter_dialogues.iloc[i]['Book'] == 'Water'):

        colors.append('#3399ff')

    else:

        colors.append('#663307')

trace = go.Bar(x = chapter_dialogues.iloc[:20]['Number of Dialogues'], y = chapter_dialogues.iloc[:20]['Chapter'], 

               orientation = 'h', marker_color = colors)

fig = go.Figure([trace])

fig.update_layout(title = {'text':'Top 20 Episodes with the Most Number of Dialogues', 'x':0.5},

                 xaxis_title="Number of Dialogues",

                 yaxis_title="Chapter Name",

                 template = 'plotly_white')

fig.show()
ratings = []

for i in range(len(chapter_dialogues)):

    chapter = chapter_dialogues.iloc[i]['Chapter']

    imdb_rating = avatar[avatar['chapter'] == chapter]['imdb_rating'].mean()

    ratings.append(imdb_rating)

chapter_dialogues['IMDB Rating'] = ratings

chapter_dialogues['IMDB Rating'].fillna(9.7, inplace = True)
chapter_dialogues['Dialogues Per Rating'] = chapter_dialogues['Number of Dialogues'] / chapter_dialogues['IMDB Rating']

chapter_dialogues = chapter_dialogues.sort_values(by = 'Dialogues Per Rating')
colors = []

for i in range(20):

    if(chapter_dialogues.iloc[i]['Book'] == 'Fire'):

        colors.append('#cd0000')

    elif(chapter_dialogues.iloc[i]['Book'] == 'Water'):

        colors.append('#3399ff')

    else:

        colors.append('#663307')

trace = go.Bar(x = chapter_dialogues.iloc[:20][::-1]['Dialogues Per Rating'], y = chapter_dialogues.iloc[:20][::-1]['Chapter'],

              text = chapter_dialogues.iloc[:20][::-1]['IMDB Rating'], orientation = 'h', marker_color = colors, 

              textposition="outside",texttemplate='%{text:.2s}',

              textfont=dict(

              family="sans serif",

              size=18,

              color="Black")

)

fig = go.Figure([trace])

fig.update_layout(title = {'text':'Top 20 Episodes with the Least Dialogues Per Rating', 'x':0.5},

                 xaxis_title="Num of Dialogues / IMDB Rating",

                 yaxis_title="Chapter Name",

                 template = 'plotly_white')

fig.show()
fig  = px.scatter(chapter_dialogues, x = 'Number of Dialogues', y = 'IMDB Rating', trendline = 'ols', color = 'Book',

                 color_discrete_map={'Fire':'#cd0000', 'Water':'#3399ff', 'Earth':'#663307'},hover_name='Chapter' ,template = 'plotly_white',

                 title = 'Relation Between Number of Dialogues and IMDB Rating')

fig.show()
stopwords = set(STOPWORDS)

def createCorpus(character_name):

    df = avatar[avatar['character'] == character_name]

    corpus = ""

    for des in df['character_words'].to_list():

        corpus += des

    return corpus



def generateWordCloud(character_name, background_color):

    plt.subplots(figsize=(12,8))

    corpus = createCorpus(character_name)

    wordcloud = WordCloud(background_color=background_color,

                          contour_color='black', contour_width=4, 

                          stopwords=stopwords,

                          width=1500, margin=10,

                          height=1080

                         ).generate(corpus)

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()
generateWordCloud('Aang', 'White')
generateWordCloud('Katara', 'LightBlue')
generateWordCloud('Sokka','Blue')
generateWordCloud('Toph','Brown')
generateWordCloud('Zuko', 'Red')
generateWordCloud('Iroh', 'Pink')
generateWordCloud('Azula','Red')
sentInt = SentimentIntensityAnalyzer()

def get_vader_score(character_name, key = 'pos'):

    corpus = createCorpus(character_name)

    sentimentScore = sentInt.polarity_scores(corpus)

    return sentimentScore[key]





character_sent_dict = {}

for sentiment in ['pos', 'neg', 'neu']:

    char_sents = []

    for character in important_characters:

        char_sents.append(get_vader_score(character, key = sentiment))

    character_sent_dict[sentiment] = char_sents

character_sent_dict['Character Name'] = important_characters

character_sentiments = pd.DataFrame(character_sent_dict)
fig = px.bar(character_sentiments, x = ['pos', 'neg','neu'], y = 'Character Name',barmode='group',

             labels = {'pos':'Positive', 'neg':'Negative','neu':'Neutral', 'value':'Sentiment Score'},

             title = 'Sentiment Analysis of Characters',

             template = 'presentation')

fig.show()
chapterCorpus = pd.DataFrame({'Chapter Name' : [], 'Full Text': [], 'Book' : []})

chapters = []

chapterTexts = []

books = []

for book in ['Water', 'Earth', 'Fire']:

    subBook = avatar[(avatar['book'] == book) & (avatar['character']!='Scene Description')]

    for chapter_name, df in subBook.groupby('chapter'):

        full_text = df['character_words'].values

        chapters.append(chapter_name)

        chapterTexts.append(" ".join(full_text).lower())

        books.append(book)

chapterCorpus['Chapter Name'] = chapters

chapterCorpus['Full Text'] = chapterTexts

chapterCorpus['Book'] = books
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 

model = hub.load(module_url)



features = model(chapterCorpus['Full Text'].values)

pca = PCA(n_components=2, random_state=42)

reduced_features = pca.fit_transform(features)



chapterCorpus['Dimension 1'] = reduced_features[:,0]

chapterCorpus['Dimension 2'] = reduced_features[:,1]

fig = px.scatter(chapterCorpus, x = 'Dimension 1', y = 'Dimension 2', color = 'Book', hover_name='Chapter Name',

                color_discrete_map={'Fire':'#cd0000', 'Water':'#3399ff', 'Earth':'#663307'},

                title = 'Finding Similar Episodes',

                template = 'plotly_white')

fig.update_traces(marker=dict(size=12))

fig.show()
chapterwise_dialogues = pd.DataFrame({})

for character in important_characters:

    character_df = avatar[avatar['character'] == character]

    chapter_counts = character_df.groupby('chapter').size().reset_index()

    chapter_counts.columns = ['chapter','Num of Dialogues']

    imdb_ratings = character_df.groupby('chapter')['imdb_rating'].mean().reset_index()

    dialogues_and_rating = pd.merge(chapter_counts, imdb_ratings)

    dialogues_and_rating['Character'] = character

    chapterwise_dialogues = pd.concat([chapterwise_dialogues, dialogues_and_rating])
fig = px.scatter(chapterwise_dialogues, 

                 x = 'chapter',y='imdb_rating', size='Num of Dialogues',

                 facet_col='Character',facet_col_wrap=2, 

                template = 'plotly_white')

fig.update_xaxes(matches = None,visible = False)

fig.show()