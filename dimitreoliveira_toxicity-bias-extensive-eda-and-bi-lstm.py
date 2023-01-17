import os

import json

import warnings

import numpy as np

import pandas as pd

import altair as alt

import seaborn as sns

import matplotlib.pyplot as plt

from PIL import Image

from altair.vega import v3

from IPython.display import HTML

from nltk import FreqDist

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS



from keras.models import Model

from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional, GlobalMaxPool1D, Dropout, GRU

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from keras.callbacks import EarlyStopping



%matplotlib inline

sns.set(style="darkgrid")

warnings.filterwarnings("ignore")

pd.set_option('display.float_format', lambda x: '%.4f' % x)

alt.data_transformers.enable('default', max_rows=None)



# Set seeds to make the experiment more reproducible.

from tensorflow import set_random_seed

from numpy.random import seed

set_random_seed(0)

seed(0)
# Preparing altair. I use code from this great kernel: https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey

vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3.SCHEMA_VERSION

vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'

vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION

vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'

noext = "?noext"



paths = {

    'vega': vega_url + noext,

    'vega-lib': vega_lib_url + noext,

    'vega-lite': vega_lite_url + noext,

    'vega-embed': vega_embed_url + noext

}



workaround = """

requirejs.config({{

    baseUrl: 'https://cdn.jsdelivr.net/npm/',

    paths: {}

}});

"""



#------------------------------------------------ Defs for future rendering

def add_autoincrement(render_func):

    # Keep track of unique <div/> IDs

    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):

        if autoincrement:

            if id in cache:

                counter = 1 + cache[id]

                cache[id] = counter

            else:

                cache[id] = 0

            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])

        else:

            if id not in cache:

                cache[id] = 0

            actual_id = id

        return render_func(chart, id=actual_id)

    # Cache will stay outside and 

    return wrapped

            

@add_autoincrement

def render(chart, id="vega-chart"):

    chart_str = """

    <div id="{id}"></div><script>

    require(["vega-embed"], function(vg_embed) {{

        const spec = {chart};     

        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);

        console.log("anything?");

    }});

    console.log("really...anything?");

    </script>

    """

    return HTML(

        chart_str.format(

            id=id,

            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)

        )

    )



HTML("".join((

    "<script>",

    workaround.format(json.dumps(paths)),

    "</script>",

)))
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
total_comments = train.shape[0]

n_unique_comments = train['comment_text'].nunique()

n_comments_both = len(set(train['comment_text'].unique()) & set(test['comment_text'].unique()))

print('Train set: %d rows and %d columns.' % (total_comments, train.shape[1]))

display(train.head())

display(train.describe())

print('Test set: %d rows and %d columns.' % (test.shape[0], test.shape[1]))

display(test.head())

print('Number of unique comments: %d or %.2f%%'% (n_unique_comments, (n_unique_comments / total_comments * 100)))

print('Number of comments that are both in train and test sets: %d'% n_comments_both)
train['comment_length'] = train['comment_text'].apply(lambda x : len(x))

test['comment_length'] = test['comment_text'].apply(lambda x : len(x))

train['word_count'] = train['comment_text'].apply(lambda x : len(x.split(' ')))

test['word_count'] = test['comment_text'].apply(lambda x : len(x.split(' ')))

bin_size = max(train['comment_length'].max(), test['comment_length'].max())//10



plt.figure(figsize=(20, 6))

sns.distplot(train['comment_length'], bins=bin_size)

sns.distplot(test['comment_length'], bins=bin_size)

plt.show()
bin_size = max(train['word_count'].max(), test['word_count'].max())//10

plt.figure(figsize=(20, 6))

sns.distplot(train['word_count'], bins=bin_size)

sns.distplot(test['word_count'], bins=bin_size)

plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

sns.distplot(train['target'], bins=20, ax=ax1).set_title("Complete set")

sns.distplot(train[train['target'] > 0]['target'].values, bins=20, ax=ax2).set_title("Only toxic comments")

plt.show()
train['is_toxic'] = train['target'].apply(lambda x : 1 if (x > 0.5) else 0)

plt.figure(figsize=(8, 6))

sns.countplot(train['is_toxic'])

plt.show()
HEIGHT = 300

WIDTH = 600

train['created_date'] = pd.to_datetime(train['created_date']).values.astype('datetime64[M]')

target_df = train.sort_values('created_date').groupby('created_date', as_index=False).agg({'id':['count'], 'target':['mean']})

target_df.columns = ['Date', 'Count', 'Toxicity rate']



# Create a selection that chooses the nearest point & selects based on x-value

nearest = alt.selection(type='single', nearest=True, on='mouseover',

                        fields=['created_date'], empty='none')



rate_line = alt.Chart().mark_line(interpolate='basis', color='red').encode(

    x='Date:T',

    y='Toxicity rate:Q'

).interactive()



# The basic line

count_line = alt.Chart().mark_line(interpolate='basis').encode(

    x='Date:T',

    y='Count:Q'

).properties(title="Counts and rate of toxicity comments").interactive()



render(alt.layer(rate_line, count_line, data=target_df, width=WIDTH, height=HEIGHT).resolve_scale(

    y='independent'

))
disability_gp = ['intellectual_or_learning_disability', 'other_disability', 'physical_disability', 'psychiatric_or_mental_illness']

disability_df = train[train[disability_gp[0]] > 0].sort_values('created_date').groupby('created_date', as_index=False).agg({'id':['count']})

disability_df.columns = ['Date', disability_gp[0]]



for x in disability_gp[1:]:

    df = train[train[x] > 0].sort_values('created_date').groupby('created_date', as_index=False).agg({'id':['count']})

    df.columns = ['Date', x]

    disability_df = pd.merge(disability_df, df)



disability_df = disability_df.melt('Date', var_name='category', value_name='Count').sort_values('Date')



# Create a selection that chooses the nearest point & selects based on x-value

nearest = alt.selection(type='single', nearest=True, on='mouseover',

                        fields=['Date'], empty='none')



# The basic line

line = alt.Chart().mark_line(interpolate='basis').encode(

    x='Date:T',

    y='Count:Q',

    color='category:N'

).properties(title="Counts of disability related toxic comments")



# Transparent selectors across the chart. This is what tells us

# the x-value of the cursor

selectors = alt.Chart().mark_point().encode(

    x='Date:T',

    opacity=alt.value(0),

).add_selection(

    nearest

)



# Draw points on the line, and highlight based on selection

points = line.mark_point().encode(

    opacity=alt.condition(nearest, alt.value(1), alt.value(0))

)



# Draw text labels near the points, and highlight based on selection

text = line.mark_text(align='left', dx=5, dy=-5).encode(

    text=alt.condition(nearest, 'Count:Q', alt.value(' '))

)



# Draw a rule at the location of the selection

rules = alt.Chart().mark_rule(color='gray').encode(

    x='Date:T',

).transform_filter(

    nearest

)



# Put the five layers into a chart and bind the data

render(alt.layer(line, selectors, points, rules, text, data=disability_df, width=WIDTH, height=HEIGHT))
race_gp = ['asian', 'black', 'latino', 'other_race_or_ethnicity', 'white']

race_df = train[train[race_gp[0]] > 0].sort_values('created_date').groupby('created_date', as_index=False).agg({'id':['count']})

race_df.columns = ['Date', race_gp[0]]



for x in race_gp[1:]:

    df = train[train[x] > 0].sort_values('created_date').groupby('created_date', as_index=False).agg({'id':['count']})

    df.columns = ['Date', x]

    race_df = pd.merge(race_df, df)



race_df = race_df.melt('Date', var_name='category', value_name='Count').sort_values('Date')



# Create a selection that chooses the nearest point & selects based on x-value

nearest = alt.selection(type='single', nearest=True, on='mouseover',

                        fields=['Date'], empty='none')



# The basic line

line = alt.Chart().mark_line(interpolate='basis').encode(

    x='Date:T',

    y='Count:Q',

    color='category:N'

).properties(title="Counts of race related toxic comments")



# Transparent selectors across the chart. This is what tells us

# the x-value of the cursor

selectors = alt.Chart().mark_point().encode(

    x='Date:T',

    opacity=alt.value(0),

).add_selection(

    nearest

)



# Draw points on the line, and highlight based on selection

points = line.mark_point().encode(

    opacity=alt.condition(nearest, alt.value(1), alt.value(0))

)



# Draw text labels near the points, and highlight based on selection

text = line.mark_text(align='left', dx=5, dy=-5).encode(

    text=alt.condition(nearest, 'Count:Q', alt.value(' '))

)



# Draw a rule at the location of the selection

rules = alt.Chart().mark_rule(color='gray').encode(

    x='Date:T',

).transform_filter(

    nearest

)



# Put the five layers into a chart and bind the data

render(alt.layer(line, selectors, points, rules, text, data=race_df, width=WIDTH, height=HEIGHT))
religion_gp = ['atheist', 'buddhist', 'christian', 'hindu', 'jewish', 'muslim', 'other_religion']

religion_df = train[train[religion_gp[0]] > 0].sort_values('created_date').groupby('created_date', as_index=False).agg({'id':['count']})

religion_df.columns = ['Date', religion_gp[0]]



for x in religion_gp[1:]:

    df = train[train[x] > 0].sort_values('created_date').groupby('created_date', as_index=False).agg({'id':['count']})

    df.columns = ['Date', x]

    religion_df = pd.merge(religion_df, df)



religion_df = religion_df.melt('Date', var_name='category', value_name='Count').sort_values('Date')



# Create a selection that chooses the nearest point & selects based on x-value

nearest = alt.selection(type='single', nearest=True, on='mouseover',

                        fields=['Date'], empty='none')



# The basic line

line = alt.Chart().mark_line(interpolate='basis').encode(

    x='Date:T',

    y='Count:Q',

    color='category:N'

).properties(title="Counts of religion related toxic comments")



# Transparent selectors across the chart. This is what tells us

# the x-value of the cursor

selectors = alt.Chart().mark_point().encode(

    x='Date:T',

    opacity=alt.value(0),

).add_selection(

    nearest

)



# Draw points on the line, and highlight based on selection

points = line.mark_point().encode(

    opacity=alt.condition(nearest, alt.value(1), alt.value(0))

)



# Draw text labels near the points, and highlight based on selection

text = line.mark_text(align='left', dx=5, dy=-5).encode(

    text=alt.condition(nearest, 'Count:Q', alt.value(' '))

)



# Draw a rule at the location of the selection

rules = alt.Chart().mark_rule(color='gray').encode(

    x='Date:T',

).transform_filter(

    nearest

)



# Put the five layers into a chart and bind the data

render(alt.layer(line, selectors, points, rules, text, data=religion_df, width=WIDTH, height=HEIGHT))
gender_gp = ['bisexual', 'female', 'heterosexual',  'homosexual_gay_or_lesbian', 'male', 'other_gender', 'other_sexual_orientation', 'transgender']

gender_df = train[train[gender_gp[0]] > 0].sort_values('created_date').groupby('created_date', as_index=False).agg({'id':['count']})

gender_df.columns = ['Date', gender_gp[0]]



for x in gender_gp[1:]:

    df = train[train[x] > 0].sort_values('created_date').groupby('created_date', as_index=False).agg({'id':['count']})

    df.columns = ['Date', x]

    gender_df = pd.merge(gender_df, df)



gender_df = gender_df.melt('Date', var_name='category', value_name='Count').sort_values('Date')



# Create a selection that chooses the nearest point & selects based on x-value

nearest = alt.selection(type='single', nearest=True, on='mouseover',

                        fields=['Date'], empty='none')



# The basic line

line = alt.Chart().mark_line(interpolate='basis').encode(

    x='Date:T',

    y='Count:Q',

    color='category:N'

).properties(title="Counts of gender & orientation related toxic comments")



# Transparent selectors across the chart. This is what tells us

# the x-value of the cursor

selectors = alt.Chart().mark_point().encode(

    x='Date:T',

    opacity=alt.value(0),

).add_selection(

    nearest

)



# Draw points on the line, and highlight based on selection

points = line.mark_point().encode(

    opacity=alt.condition(nearest, alt.value(1), alt.value(0))

)



# Draw text labels near the points, and highlight based on selection

text = line.mark_text(align='left', dx=5, dy=-5).encode(

    text=alt.condition(nearest, 'Count:Q', alt.value(' '))

)



# Draw a rule at the location of the selection

rules = alt.Chart().mark_rule(color='gray').encode(

    x='Date:T',

).transform_filter(

    nearest

)



# Put the five layers into a chart and bind the data

render(alt.layer(line, selectors, points, rules, text, data=gender_df, width=WIDTH, height=HEIGHT))
toxicity_gp = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat' , 'sexual_explicit']

toxicity_df = train[train[toxicity_gp[0]] > 0].sort_values('created_date').groupby('created_date', as_index=False).agg({'id':['count']})

toxicity_df.columns = ['Date', toxicity_gp[0]]



for x in toxicity_gp[1:]:

    df = train[train[x] > 0].sort_values('created_date').groupby('created_date', as_index=False).agg({'id':['count']})

    df.columns = ['Date', x]

    toxicity_df = pd.merge(toxicity_df, df)



toxicity_df = toxicity_df.melt('Date', var_name='category', value_name='Count').sort_values('Date')



# Create a selection that chooses the nearest point & selects based on x-value

nearest = alt.selection(type='single', nearest=True, on='mouseover',

                        fields=['Date'], empty='none')



# The basic line

line = alt.Chart().mark_line(interpolate='basis').encode(

    x='Date:T',

    y='Count:Q',

    color='category:N'

).properties(title="Counts of toxic type comments")



# Transparent selectors across the chart. This is what tells us

# the x-value of the cursor

selectors = alt.Chart().mark_point().encode(

    x='Date:T',

    opacity=alt.value(0),

).add_selection(

    nearest

)



# Draw points on the line, and highlight based on selection

points = line.mark_point().encode(

    opacity=alt.condition(nearest, alt.value(1), alt.value(0))

)



# Draw text labels near the points, and highlight based on selection

text = line.mark_text(align='left', dx=5, dy=-5).encode(

    text=alt.condition(nearest, 'Count:Q', alt.value(' '))

)



# Draw a rule at the location of the selection

rules = alt.Chart().mark_rule(color='gray').encode(

    x='Date:T',

).transform_filter(

    nearest

)



# Put the five layers into a chart and bind the data

render(alt.layer(line, selectors, points, rules, text, data=toxicity_df, width=WIDTH, height=HEIGHT))
# Got this from here https://www.kaggle.com/artgor/toxicity-eda-model-interpretation-and-more/data

train['disability'] = train['intellectual_or_learning_disability'] + train['other_disability'] + train['physical_disability'] + train['psychiatric_or_mental_illness']

train['race'] = train['asian'] + train['black'] + train['latino'] + train['other_race_or_ethnicity'] + train['white']

train['religion'] = train['atheist'] + train['buddhist'] + train['christian'] + train['hindu'] + train['jewish'] + train['muslim'] + train['other_religion']

train['gender'] = train['bisexual'] + train['female'] + train['heterosexual'] + train['homosexual_gay_or_lesbian'] + train['male'] + train['other_gender'] + train['other_sexual_orientation'] + train['transgender']

train['type'] = train['severe_toxicity'] + train['obscene'] + train['identity_attack'] + train['insult'] + train['threat'] + train['sexual_explicit']

super_groups_gp = ['disability', 'race', 'religion', 'gender', 'type']

plot_dict = {}

bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 0.9, 0.95, 1]



for col in super_groups_gp:

    df_ = train.loc[train[col] > 0]

    hist_df = pd.cut(df_[col], bins).value_counts().sort_index().reset_index().rename(columns={'index': 'bins'})

    hist_df['bins'] = hist_df['bins'].astype(str)

    plot_dict[col] = alt.Chart(hist_df).mark_bar().encode(

        x=alt.X("bins:O", axis=alt.Axis(title='Target bins')),

        y=alt.Y(f'{col}:Q', axis=alt.Axis(title='Count')),

        tooltip=[col, 'bins']

    ).properties(title=f"Distribution of {col}", width=300, height=200).interactive()

    

render((plot_dict['disability'] | plot_dict['race']) & (plot_dict['religion'] | plot_dict['gender']) & (plot_dict['type']))
eng_stopwords = stopwords.words('english')

train['comment_text'] = train['comment_text'].apply(lambda x: x.lower())

# train['comment_text'] = train['comment_text'].str.replace('[^\w\s]','')

train['comment_text'] = train['comment_text'].str.replace('[^a-z ]','')

train['comment_text'] = train['comment_text'].apply(lambda x: " ".join(x for x in x.split() if x not in eng_stopwords))



freq_dist = FreqDist([word for comment in train[train['target'] > 0.8]['comment_text'] for word in comment.split()])



plt.figure(figsize=(20, 6))

plt.title('Word frequency on toxic comments').set_fontsize(20)

freq_dist.plot(60, marker='.', markersize=10)

plt.show()
wc_stopwords = set(STOPWORDS)

def plot_WordCloud(group, mask_path):

    mask = np.array(Image.open(mask_path))[:,:,1]

    text = train[train[group] > 0.8 ]['comment_text'].values

    wc = WordCloud(background_color="white", max_words=2000, mask=mask,

                   stopwords=wc_stopwords, contour_width=3, contour_color='steelblue')



    wc.generate(" ".join(text))



    plt.figure(figsize=(6, 6))

    plt.imshow(wc, interpolation='bilinear')

    plt.axis("off")



plot_WordCloud(group='type', mask_path="../input/toxicity-comments-images/toxic1.png")
plot_WordCloud(group='race', mask_path="../input/toxicity-comments-images/race.png")
plot_WordCloud(group='disability', mask_path="../input/toxicity-comments-images/disability3.png")
plot_WordCloud(group='gender', mask_path="../input/toxicity-comments-images/gender2.png")
plot_WordCloud(group='religion', mask_path="../input/toxicity-comments-images/religion1.png")
annotators_len_df = train[train['identity_annotator_count'] > 0].groupby('word_count', as_index=False).agg({'identity_annotator_count': 'mean'})

annotators_len_df.columns = ['word_count', 'mean_annotators']



# Create a selection that chooses the nearest point & selects based on x-value

nearest = alt.selection(type='single', nearest=True, on='mouseover',

                        fields=['word_count'], empty='none')



# The basic line

line = alt.Chart().mark_line().encode(

    y='mean_annotators:Q',

    x='word_count:Q'

).properties(title="Counts of toxic type comments")



# Transparent selectors across the chart. This is what tells us

# the x-value of the cursor

selectors = alt.Chart().mark_point().encode(

    x='word_count:Q',

    opacity=alt.value(0),

).add_selection(

    nearest

)



# Draw points on the line, and highlight based on selection

points = line.mark_point().encode(

    opacity=alt.condition(nearest, alt.value(1), alt.value(0))

)



# Draw text labels near the points, and highlight based on selection

text = line.mark_text(align='left', dx=5, dy=-5).encode(

    text=alt.condition(nearest, 'mean_annotators:Q', alt.value(' '))

)



# Draw a rule at the location of the selection

rules = alt.Chart().mark_rule(color='gray').encode(

    x='word_count:Q',

).transform_filter(

    nearest

)



# Put the five layers into a chart and bind the data

render(alt.layer(line, selectors, points, rules, text, data=annotators_len_df, width=WIDTH, height=HEIGHT))
X_train = train['comment_text'].astype(str)

X_test = test['comment_text'].astype(str)

y = np.where(train['target'] >= 0.5, True, False) * 1
num_words = 20000

max_len = 150

emb_size = 128
tok = Tokenizer(num_words = num_words)

tok.fit_on_texts(list(X_train))
X = tok.texts_to_sequences(X_train)

test = tok.texts_to_sequences(X_test)
X = sequence.pad_sequences(X, maxlen=max_len)

X_test = sequence.pad_sequences(test, maxlen=max_len)
# Got this model here: https://www.kaggle.com/lsjsj92/toxic-simple-eda-and-modeling-with-lstm

inp = Input(shape = (max_len, ))

layer = Embedding(num_words, emb_size)(inp)

layer = Bidirectional(LSTM(50, return_sequences=True, recurrent_dropout=0.15))(layer)

layer = GlobalMaxPool1D()(layer)

layer = Dropout(0.5)(layer)

layer = Dense(100, activation='relu')(layer)

layer = Dropout(0.5)(layer)

layer = Dense(1, activation='sigmoid')(layer)



model = Model(inputs=inp, outputs=layer)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

history = model.fit(X, y, batch_size=1024, epochs=20, validation_split=0.2, callbacks=[es])
sns.set_style("whitegrid")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8))



ax1.plot(history.history['acc'], label='Train Accuracy')

ax1.plot(history.history['val_acc'], label='Validation accuracy')

ax1.legend(loc='best')

ax1.set_title('Accuracy')



ax2.plot(history.history['loss'], label='Train loss')

ax2.plot(history.history['val_loss'], label='Validation loss')

ax2.legend(loc='best')

ax2.set_title('Loss')



plt.xlabel('Epochs')

sns.despine()

plt.show()
Y_test = model.predict(X_test)
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

submission['prediction'] = Y_test

submission.to_csv('submission.csv', index=False)

submission.head(10)