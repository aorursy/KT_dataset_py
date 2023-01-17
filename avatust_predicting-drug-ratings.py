VOCABULARY_SIZE = 4095

VALIDATION_SPLIT = 0.25

SPACY_MODEL = 'en_core_web_lg'
# basic imports

import os



import IPython



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import keras

from keras import layers

from keras.regularizers import l1, l2



from sklearn.model_selection import train_test_split
# an NLP library

# get a model with pretrained embeddings and load it



# if GPU is on, Internet connection is needed in Kaggle kernels!

!python -m spacy download $SPACY_MODEL



import spacy

nlp = spacy.load(SPACY_MODEL)
# convenience settings



### plotting ###

# prettier and larger basic graphs

sns.set(rc={

    'figure.figsize':(18,8),

    'axes.titlesize':14,

})



### pandas ###

# make the tables more compact vertically, too

pd.options.display.max_rows = 20
print(os.listdir("../input"))
# load and preview the training set

reviews_df = pd.read_csv('../input/drugsComTrain_raw.csv')

reviews_df
# once more for the test set

reviews_test_df = pd.read_csv('../input/drugsComTest_raw.csv')

reviews_test_df
# sanitizing review text



from html import unescape



def clean_review(text):

    """Replace HTML escaped characters, byte order mark

    and strip outer quotes and spaces"""

    return unescape(text.strip(' "\'')).replace('\ufeff1', '')



reviews_df.review = reviews_df.review.apply(clean_review)

reviews_test_df.review = reviews_test_df.review.apply(clean_review)
from wordcloud import WordCloud



wordcloud = WordCloud(

    width=1600,

    height=900,

    random_state=2019,

    background_color='white',

    max_words=400,

    colormap='bone'

)



wordcloud.generate('\n'.join(reviews_df.review))

plt.imshow(wordcloud, interpolation='lanczos')

plt.axis('off')

plt.show()
markdown = '### A peek at some reviews\n'

for idx, row in reviews_df.sample(10).iterrows():

    markdown += f'\n\nReview {row.uniqueID}\n\n> '

    markdown += '\n> '.join(row.review.splitlines())

    

IPython.display.Markdown(markdown)
IPython.display.Markdown(

    f"I will use spaCy's `{SPACY_MODEL}` model for tokenizing reviews and the embeddings, "

    f"the model has {len(nlp.vocab)} lexemes in its vocabulary "

    f"and {len(nlp.vocab.vectors)} have embeddings of length {nlp.vocab.vectors_length}."

)
# sequencing review text



def text2tokens(text):

    """text -> tokenize -> lemmatize/normalize"""

    

    tokens = nlp(

        # also split on "/"

        text.replace('/', ' / '),

        

        # we only need tokenizer and lemmas, so disable the rest

        disable=['tagger', 'parser', 'ner']

    )

    

    lexemes = []

    for token in tokens:

        

        # sometimes whitespace gets recognized as a token...

        if token.text.isspace():

            continue

            

        # prefer more general representations

        # but only if they have an embedding

        if nlp.vocab[token.lemma_.lower()].has_vector:

            lexeme = token.lemma_.lower()

        elif nlp.vocab[token.norm_.lower()].has_vector:

            lexeme = token.norm_.lower()

        else:

            lexeme = token.lower_

        

        lexemes.append(lexeme)

        

    return lexemes



reviews_word_seq = [text2tokens(review) for review in reviews_df.review]

reviews_test_word_seq = [text2tokens(review) for review in reviews_test_df.review]
# count occurences of each word/token



word_count = {}

for lemmas in reviews_word_seq:

    for lemma in lemmas:

        word_count[lemma] = word_count.get(lemma, 0) + 1

word_count = dict(sorted(word_count.items(), key=lambda pair: pair[1], reverse=True))
# display a table of the most frequent words



# keep it within our vocabulary

show_words = min(1000, VOCABULARY_SIZE)

columns = 5



markdown_rows = []

markdown_rows.append(f'#### Top {show_words} most frequent tokens\n')

markdown_rows.append('| Token | Count '*columns + '|')

markdown_rows.append('| ---: | :---- '*columns + '|')



row = ''

for index, (word, count) in enumerate(word_count.items()):

    if index >= show_words: break

    if index%columns == 0 and row:

        markdown_rows.append(row + '|')

        row = ''

    row += f'| {word} | {count} '

if row:

    markdown_rows.append(row + '|')

    

IPython.display.Markdown('\n'.join(markdown_rows))
plt.title('Vocabulary Distribution')



wc_vals = list(word_count.values())



plt.plot(wc_vals, 'o', label='Word count')

plt.plot(sum(wc_vals)-np.cumsum(wc_vals), '.', label='Uncaptured words')

plt.axvline(x=VOCABULARY_SIZE, color='yellowgreen', label=f'Current vocabulary size ({VOCABULARY_SIZE})')



plt.gca().set_yscale('log')

# plt.gca().set_xscale('log')



plt.xlabel('Vocabulary Size')

plt.ylabel('Amount')

plt.legend()



plt.show()
# word sequences to indexed sequences



vocab = list(word_count)[:VOCABULARY_SIZE]

word2index = {word:index for index, word in enumerate(vocab, start=1)}



word_seq2indexed = lambda reviews: [

    [

        word2index.get(word, 0)

        for word in review

    ]

    for review in reviews

]



reviews_seq = word_seq2indexed(reviews_word_seq)

reviews_test_seq = word_seq2indexed(reviews_test_word_seq)



# no need for these anymore

del reviews_word_seq, reviews_test_word_seq
# set max sequence length so that it captures 99% full reviews

review_lengths = np.array(list(map(len, reviews_seq)))

sequence_cutoff_legth = int(np.quantile(review_lengths, 0.99))
plt.title('Distribution of Review Lengths')



sns.distplot(

    review_lengths,

    hist_kws=dict(label='Normalized histogram'),

    kde=True,

    kde_kws=dict(label='Kernel density'),

    rug=True,

    norm_hist=False,

    rug_kws=dict(color='orangered', label='Review'),

    axlabel='Sequence Length',

)

plt.axvline(

    x=sequence_cutoff_legth,

    color='yellowgreen',

    label=f'Sequence cutoff length ({sequence_cutoff_legth})'

)



plt.gca().set_xscale('log')



plt.xlabel('Review Length')

plt.ylabel('Density')

plt.legend()



plt.show()
# pad the sequences



reviews_seq = keras.preprocessing.sequence.pad_sequences(

    reviews_seq,

    maxlen=sequence_cutoff_legth

)



reviews_test_seq = keras.preprocessing.sequence.pad_sequences(

    reviews_test_seq,

    maxlen=sequence_cutoff_legth

)
# extract embeddings for our vocabulary



embedding_weights = np.zeros((

    VOCABULARY_SIZE+1, # indices/hashes

    nlp.vocab.vectors_length # embedding dimmension

))



for word, index in word2index.items():

    embedding_weights[index] = nlp.vocab[word].vector
# here it goes :-)



input_reviews = layers.Input(shape=(sequence_cutoff_legth,), dtype='int32')





embedding = layers.Embedding(

    *embedding_weights.shape,

    weights=[embedding_weights],

    input_length=sequence_cutoff_legth,

    trainable=True,

)(input_reviews)

embedding = layers.GaussianNoise(0.15)(embedding)





branch_a = layers.CuDNNLSTM(128)(embedding)

branch_a = layers.BatchNormalization()(branch_a)

branch_a = layers.Dropout(0.3)(branch_a)

branch_a = layers.GaussianNoise(0.3)(branch_a)



branch_a = layers.Dense(10, activation='relu')(branch_a)

branch_a = layers.BatchNormalization()(branch_a)

branch_a = layers.GaussianNoise(0.3)(branch_a)





branch_b = layers.Conv1D(16, 3, padding='same', activation='relu')(embedding)

branch_b = layers.BatchNormalization()(branch_b)

branch_b = layers.MaxPool1D(2)(branch_b)



branch_b = layers.Conv1D(32, 3, padding='same', activation='relu')(branch_b)

branch_b = layers.BatchNormalization()(branch_b)

branch_b = layers.MaxPool1D(2)(branch_b)



branch_b = layers.Conv1D(64, 3, padding='same', activation='relu')(branch_b)

branch_b = layers.BatchNormalization()(branch_b)

branch_b = layers.MaxPool1D(2)(branch_b)



branch_b = layers.Conv1D(128, 3, padding='same', activation='relu')(branch_b)

branch_b = layers.Dropout(0.3)(branch_b)



branch_b = layers.GlobalAvgPool1D()(branch_b)

branch_b = layers.BatchNormalization()(branch_b)

branch_b = layers.Dropout(0.3)(branch_b)

branch_b = layers.GaussianNoise(0.3)(branch_b)



branch_b = layers.Dense(10, activation='relu')(branch_b)

branch_b = layers.BatchNormalization()(branch_b)

branch_b = layers.GaussianNoise(0.3)(branch_b)





input_useful_count = layers.Input(shape=(1,))





intermediate = layers.concatenate([branch_a, branch_b, input_useful_count])

intermediate = layers.Dropout(0.3)(intermediate)



intermediate = layers.Dense(32, activation='relu')(intermediate)

intermediate = layers.BatchNormalization()(intermediate)

intermediate = layers.Dropout(0.3)(intermediate)

intermediate = layers.GaussianNoise(0.3)(intermediate)





regression_top = layers.Dense(10, activation='relu')(intermediate)

regression_top = layers.BatchNormalization()(regression_top)

regression_top = layers.GaussianNoise(0.1)(regression_top)



regression_top = layers.Dense(1, activation='sigmoid', name='regressor')(regression_top)





classification_top = layers.Dense(10, activation='relu', activity_regularizer=l2(2e-4))(intermediate)

classification_top = layers.BatchNormalization()(classification_top)

classification_top = layers.Dropout(0.2)(classification_top)

classification_top = layers.GaussianNoise(0.2)(classification_top)



classification_top = layers.Dense(10, activation='softmax', name='classifier')(classification_top)





model = keras.models.Model(

    inputs=[input_reviews, input_useful_count],

    outputs=[regression_top, classification_top],

)

model.summary()
# visualize the model graph



model_viz = keras.utils.vis_utils.model_to_dot(model)

IPython.display.SVG(model_viz.create(prog='dot', format='svg'))
# standard scaling of the "useful count" attribute



useful_count_mean = reviews_df.usefulCount.mean()

useful_count_std = reviews_df.usefulCount.std()



reviews_df.usefulCount += useful_count_mean

reviews_df.usefulCount /= useful_count_std



reviews_test_df.usefulCount += useful_count_mean

reviews_test_df.usefulCount /= useful_count_std
# training/validation split and creation



rating = reviews_df.rating.values



(

    y_regr_train, y_regr_valid,

    y_clas_train, y_clas_valid,

    x1_train, x1_valid,

    x2_train, x2_valid,

) = train_test_split(

    # reggression output between 0.1 and 1.0

    rating / 10,

    # classification between 0 and 9 and to categorical

    keras.utils.to_categorical(rating-1, num_classes=10),

    

    reviews_seq, # main input

    reviews_df.usefulCount.values, # aux. input

    

    # options

    test_size=VALIDATION_SPLIT,

    stratify=reviews_df.rating.values

)



x_train = [x1_train, x2_train]

y_train = [y_regr_train, y_clas_train]



x_valid = [x1_valid, x2_valid]

y_valid = [y_regr_valid, y_clas_valid]
model.compile(

    optimizer=keras.optimizers.Adam(),

    loss=['mae', 'categorical_crossentropy'],

    metrics={'classifier':'accuracy', 'regressor':'mae'},

    # put more weight to the regressor

    loss_weights=[2,1],

)
# let's give it some work out!



history = model.fit(

    x=x_train,

    y=y_train,

    validation_data=(x_valid, y_valid),

    batch_size=256,

    epochs=300,

    verbose=0,

    callbacks=[

        keras.callbacks.ModelCheckpoint(

            'model'

            '-epoch_{epoch:02d}'

            '-regr_mae_{val_regressor_loss:.2f}'

            '-clas_acc_{val_classifier_acc:.2f}.hdf5',

            monitor='val_regressor_loss',

            verbose=0,

            save_best_only=True,

            save_weights_only=False,

            mode='auto',

            period=1,

        ),

    ],

)





model.save('model-last_epoch.hdf5')
def plot_history(history, skip_first_n_epochs=0):

    """Show information about the training"""

    

    # plot every train-valid metric pair separately

    for metric in history:

        if not metric.startswith('val_'):

            x = np.arange(len(history[metric]))+1



            y_train = history[metric][skip_first_n_epochs:]

            y_valid = history['val_'+metric][skip_first_n_epochs:]



            plt.plot(x, y_train)

            plt.plot(x, y_valid)



            plt.legend([metric, 'val_'+metric], fontsize='large')



            plt.title(

                f'{metric.upper()} - '

                f'min/max [train: {min(y_train):.3f}/{max(y_train):.3f}, '

                f'valid: {min(y_valid):.3f}/{max(y_valid):.3f}]'

            )

            

            plt.xlabel('epoch')

            plt.show()

            

plot_history(history.history)
# evaluate the model



def y2original(y):

    """Convert Y result to the original score scale"""

    y_regr, y_clas = y

    y_regr = 10*y_regr

    y_clas = np.argmax(y_clas, axis=1) + 1

    return y_regr, y_clas



x_test = [reviews_test_seq, reviews_test_df.usefulCount.values]



y_test = reviews_test_df.rating.values

y_train = y2original(y_train)[0]

y_valid = y2original(y_valid)[0]



# evaluate model and convert output

yx_train_regr, yx_train_clas = y2original(model.predict(x_train))

yx_valid_regr, yx_valid_clas = y2original(model.predict(x_valid))

yx_test_regr,  yx_test_clas  = y2original(model.predict(x_test))
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, accuracy_score



def show_metrics(title, y_true, y_regr, y_clas):

    y_regr_closest = np.round(y_regr)

    

    fmt = '{:<16} | {:>8} | {:>8} | {:>8}'.format

    nums2str = lambda *nums: (f'{n:.3f}' for n in nums)

    

    print(fmt(title, 'MAE', 'KAPPA', 'ACCURACY'))

    print(fmt(' regression', *nums2str( 

            mean_absolute_error(y_true, y_regr),

            cohen_kappa_score(y_true, y_regr_closest),

            accuracy_score(y_true, y_regr_closest)

    )))

    

    print(fmt(' classification', *nums2str( 

            mean_absolute_error(y_true, y_clas),

            cohen_kappa_score(y_true, y_clas),

            accuracy_score(y_true, y_clas)

    )))

    print()

    
show_metrics('TRAINING SET', y_train, yx_train_regr, yx_train_clas)

show_metrics('VALIDATION SET', y_valid, yx_valid_regr, yx_valid_clas)

show_metrics('TEST SET', y_test, yx_test_regr, yx_test_clas)
# rearange so that true labels lump together

# (sort is stable so within a rating level train-valid-test order holds)



Y = np.concatenate([y_train, y_valid, y_test])

YX_REGR = np.concatenate([yx_train_regr, yx_valid_regr, yx_test_regr])

YX_CLAS = np.concatenate([yx_train_clas, yx_valid_clas, yx_test_clas])

_INDEX = np.arange(len(Y))



quadratuplets = zip(Y, YX_REGR, YX_CLAS, _INDEX)

rearanged = sorted(quadratuplets, key=lambda q: q[0])



y, yx_regr, yx_clas, index = zip(*rearanged)
# separate them again, so that we can add colors



# alternative would be to make an array with colors already earlier

# and add it to the sorting above, then having it as an argument

# for scatter plot, it was, however, slow and not so pretty :-)



# for axis indices on the plot

axis = np.arange(len(y))



index_train = []

index_valid = []

index_test = []

t1, t2 = len(y_train), len(y_train) + len(y_valid)

for i, a in zip(index, axis):

    if i < t1:

        index_train.append(a)

    elif i < t2:

        index_valid.append(a)

    else:

        index_test.append(a)



axis_train = axis[index_train]

axis_valid = axis[index_valid]

axis_test = axis[index_test]



yx_train_regr = np.asarray(yx_regr)[index_train]

yx_valid_regr = np.asarray(yx_regr)[index_valid]

yx_test_regr  = np.asarray(yx_regr)[index_test]



yx_train_clas = np.asarray(yx_clas)[index_train]

yx_valid_clas = np.asarray(yx_clas)[index_valid]

yx_test_clas  = np.asarray(yx_clas)[index_test]
# the plot finally

cm = plt.cm.viridis



plt.plot(axis_train, yx_train_regr, '.', alpha=0.1, color=cm(0.75))

plt.plot(axis_valid, yx_valid_regr, '.', alpha=0.1, color=cm(0.8))

plt.plot(axis_test,  yx_test_regr , '.', alpha=0.1, color=cm(0.9))



plt.plot(axis_train, yx_train_clas, 'o', alpha=0.1, color=cm(0.1))

plt.plot(axis_valid, yx_valid_clas, 'o', alpha=0.1, color=cm(0.17))

plt.plot(axis_test,  yx_test_clas , 'o', alpha=0.1, color=cm(0.3))



plt.plot(axis, y, 'o', c='#88888801', markersize=12)



legend_label = lambda c, l: plt.Line2D(

    [0], [0], linewidth=0, marker='o', color=c, label=l)



legend_elements = [

    legend_label('#888888', 'Actual score'),

    legend_label(cm(0.1), 'Classification (training)'),

    legend_label(cm(0.17), 'Classification (validation)'),

    legend_label(cm(0.3), 'Classification (test)'),

    legend_label(cm(0.75), 'Regression (training)'),

    legend_label(cm(0.8), 'Regression (validation)'),

    legend_label(cm(0.9), 'Regression (test)'),

]



plt.legend(handles=legend_elements)

plt.xticks([])

plt.xlabel('Reviews')

plt.ylabel('Rating')

plt.show()