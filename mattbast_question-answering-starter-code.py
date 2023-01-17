import json

from tqdm.notebook import tqdm



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, SpatialDropout1D, Dense, Dropout, Input, concatenate, Conv1D, Activation, Flatten



from nltk.corpus import stopwords

import re
# data to load

NUM_OF_TRAIN_QUESTIONS = 1000

NUM_OF_VAL_QUESTIONS = 1050

SAMPLE_RATE = 15

TRAIN_PATH = '../input/tensorflow2-question-answering/simplified-nq-train.jsonl'



# TOKENIZATION

FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

LOWER_CASE = True

MAX_LEN = 300



# long answer model parameters

EPOCHS = 40

BATCH_SIZE = 64

EMBED_SIZE = 100

CLASS_WEIGHTS = {0: 0.5, 1: 5.}



# short answer model parameters

SHORT_EPOCHS = 80

SHORT_BATCH_SIZE = 32

SHORT_EMBED_SIZE = 200
def get_line_of_data(file):

    line = file.readline()

    line = json.loads(line)

    

    return line





def get_question_and_document(line):

    question = line['question_text']

    text = line['document_text'].split(' ')

    annotations = line['annotations'][0]

    

    return question, text, annotations

                

                

def get_long_candidate(i, annotations, candidate):

    # check if this candidate is the correct answer

    if i == annotations['long_answer']['candidate_index']:

        label = True

    else:

        label = False



    # get place where long answer starts and ends in the document text

    long_start = candidate['start_token']

    long_end = candidate['end_token']

    

    return label, long_start, long_end





def form_data_row(question, label, text, long_start, long_end):

    row = {

        'question': question,

        'long_answer': ' '.join(text[long_start:long_end]),

        'is_long_answer': label,

    }

    

    return row





def load_data(file_path, questions_start, questions_end):

    rows = []

    

    with open(file_path) as file:



        for i in tqdm(range(questions_start, questions_end)):

            line = get_line_of_data(file)

            question, text, annotations = get_question_and_document(line)



            for i, candidate in enumerate(line['long_answer_candidates']):

                label, long_start, long_end = get_long_candidate(i, annotations, candidate)



                if label == True or (i % SAMPLE_RATE == 0):

                    rows.append(

                        form_data_row(question, label, text, long_start, long_end)

                    )

        

    return pd.DataFrame(rows)
train_df = load_data(TRAIN_PATH, 0, NUM_OF_TRAIN_QUESTIONS)

val_df = load_data(TRAIN_PATH, NUM_OF_TRAIN_QUESTIONS, NUM_OF_VAL_QUESTIONS)
train_df.head(5)
def remove_stopwords(sentence):

    words = sentence.split()

    words = [word for word in words if word not in stopwords.words('english')]

    

    return ' '.join(words)





def remove_html(sentence):

    html = re.compile(r'<.*?>')

    return html.sub(r'', sentence)





def clean_df(df):

    df['long_answer'] = df['long_answer'].apply(lambda x : remove_stopwords(x))

    df['long_answer'] = df['long_answer'].apply(lambda x : remove_html(x))



    df['question'] = df['question'].apply(lambda x : remove_stopwords(x))

    df['question'] = df['question'].apply(lambda x : remove_html(x))

    

    return df
train_df = clean_df(train_df)

val_df = clean_df(val_df)
train_df.head(5)
def define_tokenizer(df_series):

    sentences = pd.concat(df_series)

    

    tokenizer = tf.keras.preprocessing.text.Tokenizer(

        filters=FILTERS, 

        lower=LOWER_CASE

    )

    tokenizer.fit_on_texts(sentences)

    

    return tokenizer



    

def encode(sentences, tokenizer):

    encoded_sentences = tokenizer.texts_to_sequences(sentences)

    

    encoded_sentences = tf.keras.preprocessing.sequence.pad_sequences(

        encoded_sentences, 

        padding='post',

        maxlen=MAX_LEN

    )

    

    return encoded_sentences
tokenizer = define_tokenizer([

    train_df.long_answer, 

    train_df.question,

    val_df.long_answer, 

    val_df.question

])
tokenizer.word_index['tracy']
train_long_answers = encode(train_df['long_answer'].values, tokenizer)

train_questions = encode(train_df['question'].values, tokenizer)



val_long_answers = encode(val_df['long_answer'].values, tokenizer)

val_questions = encode(val_df['question'].values, tokenizer)
train_long_answers[0]
train_labels = train_df.is_long_answer.astype(int).values

val_labels = val_df.is_long_answer.astype(int).values
train_labels
embedding_dict = {}



with open('../input/glove-global-vectors-for-word-representation/glove.6B.' + str(EMBED_SIZE) + 'd.txt','r') as f:

    for line in f:

        values = line.split()

        word = values[0]

        vectors = np.asarray(values[1:],'float32')

        embedding_dict[word] = vectors

        

f.close()
num_words = len(tokenizer.word_index) + 1

embedding_matrix = np.zeros((num_words, EMBED_SIZE))



for word, i in tokenizer.word_index.items():

    if i > num_words:

        continue

    

    emb_vec = embedding_dict.get(word)

    

    if emb_vec is not None:

        embedding_matrix[i] = emb_vec
embedding = tf.keras.layers.Embedding(

    len(tokenizer.word_index) + 1,

    EMBED_SIZE,

    embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix),

    trainable = False

)
# question encoding

question_input = Input(shape=(None,))

question_x = embedding(question_input)

question_x = SpatialDropout1D(0.2)(question_x)

question_x = Bidirectional(LSTM(100, return_sequences=True))(question_x)

question_x = GlobalMaxPooling1D()(question_x)



# answer encoding

answer_input = Input(shape=(None,))

answer_x = embedding(answer_input)

answer_x = SpatialDropout1D(0.2)(answer_x)

answer_x = Bidirectional(LSTM(150, return_sequences=True))(answer_x)

answer_x = GlobalMaxPooling1D()(answer_x)



# classification

combined_x = concatenate([question_x, answer_x])

combined_x = Dense(300, activation='relu')(combined_x)

combined_x = Dropout(0.5)(combined_x)

combined_x = Dense(300, activation='relu')(combined_x)

combined_x = Dropout(0.5)(combined_x)

output = Dense(1, activation='sigmoid')(combined_x)



# combine model parts into one

model = tf.keras.models.Model(inputs=[answer_input, question_input], outputs=output)
model.compile(

    loss='binary_crossentropy', 

    optimizer='adam',

    metrics=['BinaryAccuracy', 'Recall', 'Precision']

)
callbacks = [

    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1),

    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1),

]
history = model.fit(

    x = [train_long_answers, train_questions], 

    y = train_labels,

    validation_data = (

        [val_long_answers, val_questions], 

        val_labels

    ),

    epochs = EPOCHS,

    callbacks = callbacks,

    class_weight = CLASS_WEIGHTS,

    batch_size = BATCH_SIZE,

    shuffle = True

)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))



ax[0].set_title('Training Loss')

ax[0].plot(history.history['loss'])



ax[1].set_title('Validation Loss')

ax[1].plot(history.history['val_loss'])
fig, ax = plt.subplots(3, 2, figsize=(15, 10))



ax[0,0].set_title('Training Accuracy')

ax[0,0].plot(history.history['binary_accuracy'])



ax[0,1].set_title('Validation Accuracy')

ax[0,1].plot(history.history['val_binary_accuracy'])



ax[1,0].set_title('Training Recall')

ax[1,0].plot(history.history['recall'])



ax[1,1].set_title('Validation Recall')

ax[1,1].plot(history.history['val_recall'])



ax[2,0].set_title('Training Precision')

ax[2,0].plot(history.history['precision'])



ax[2,1].set_title('Validation Precision')

ax[2,1].plot(history.history['val_precision'])
print('Epochs: {0}'.format(

    len(history.history['loss'])

))
recall = history.history['recall'][-1]

precision = history.history['precision'][-1]



print('Train F1 score: {0:.4f}'.format(

    2 * (precision * recall) / (precision + recall)

))
recall = history.history['val_recall'][-1]

precision = history.history['val_precision'][-1]



print('Validation F1 score: {0:.4f}'.format(

    2 * (precision * recall) / (precision + recall)

))
def test_question(question, positive, negative):

    sentences = [question, positive, negative]

    

    for i in range(3):

        sentences[i] = remove_stopwords(sentences[i])

        sentences[i] = remove_html(sentences[i])

    

    sentences = encode(sentences, tokenizer)

    

    predictions = model.predict(

        [np.expand_dims(sentences[1], axis=0), np.expand_dims(sentences[0], axis=0)]

    )



    print('Positive: {0:.2f}'.format(predictions[0][0]))



    predictions = model.predict(

        [np.expand_dims(sentences[2], axis=0), np.expand_dims(sentences[0], axis=0)]

    )



    print('Negative: {0:.2f}'.format(predictions[0][0]))
question = 'which is the most common use of opt-in e-mail marketing'



positive = "<P> A common example of permission marketing is a newsletter sent to an advertising firm 's customers . Such newsletters inform customers of upcoming events or promotions , or new products . In this type of advertising , a company that wants to send a newsletter to their customers may ask them at the point of purchase if they would like to receive the newsletter . </P>"



negative = '<P> Email marketing has evolved rapidly alongside the technological growth of the 21st century . Prior to this growth , when emails were novelties to the majority of customers , email marketing was not as effective . In 1978 , Gary Thuerk of Digital Equipment Corporation ( DEC ) sent out the first mass email to approximately 400 potential clients via the Advanced Research Projects Agency Network ( ARPANET ) . This email resulted in $13 million worth of sales in DEC products , and highlighted the potential of marketing through mass emails . However , as email marketing developed as an effective means of direct communication , users began blocking out content from emails with filters and blocking programs . In order to effectively communicate a message through email , marketers had to develop a way of pushing content through to the end user , without being cut out by automatic filters and spam removing software . This resulted in the birth of triggered marketing emails , which are sent to specific users based on their tracked online browsing patterns . </P>'
test_question(question, positive, negative)
question = 'how i.met your mother who is the mother'



positive = "<P> Tracy McConnell , better known as `` The Mother '' , is the title character from the CBS television sitcom How I Met Your Mother . The show , narrated by Future Ted , tells the story of how Ted Mosby met The Mother . Tracy McConnell appears in 8 episodes from `` Lucky Penny '' to `` The Time Travelers '' as an unseen character ; she was first seen fully in `` Something New '' and was promoted to a main character in season 9 . The Mother is played by Cristin Milioti . </P>"



negative = "<P> In `` Bass Player Wanted '' , the Mother picks up a hitchhiking Marshall , carrying his son Marvin , on her way to Farhampton Inn . On their way , it is revealed that the Mother is a bass player in the band , that is scheduled to play at the wedding reception . But the band 's leader , Darren , forced her to quit . The Mother ultimately decides to confront Darren and retake the band . She ends up alone at the bar , and while practicing a speech to give Darren , Darren walks up to her furious the groom 's best man punched him for `` no reason . '' Amused by this , the Mother laughs , and Darren quits the band in anger . </P>"
test_question(question, positive, negative)
question = 'how i met your mother who is the mother'

test_question(question, positive, negative)
question = 'who is tracy mcconnell'

test_question(question, positive, negative)
def get_short_answer(annotations, long_start, long_end):

    if len(annotations['short_answers']) > 0:

        short_start = annotations['short_answers'][0]['start_token']

        short_end = annotations['short_answers'][0]['end_token']

        

        short_start = short_start - long_start

        short_end = short_end - long_start

        

        return short_start, short_end

    else:

        return 0, 0

    



def form_short_data_row(question, text, long_start, long_end, short_start, short_end):

    long_answer = ' '.join(text[long_start:long_end])

    short_answer = ' '.join(long_answer.split(' ')[short_start:short_end])

    

    row = {

        'question': question,

        'long_answer': long_answer,

        'short_answer': short_answer,

        'short_start': short_start,

        'short_end': short_end

    }

    

    return row





def load_short_data(file_path, questions_start, questions_end):

    rows = []

    

    with open(file_path) as file:



        for i in tqdm(range(questions_start, questions_end)):

            line = get_line_of_data(file)

            question, text, annotations = get_question_and_document(line)



            for i, candidate in enumerate(line['long_answer_candidates']):

                label, long_start, long_end = get_long_candidate(i, annotations, candidate)



                if label == True:

                    short_start, short_end = get_short_answer(annotations, long_start, long_end)

                    

                    rows.append(

                        form_short_data_row(question, text, long_start, long_end, short_start, short_end)

                    )

        

    return pd.DataFrame(rows)
train_short_df = load_short_data(TRAIN_PATH, 0, NUM_OF_TRAIN_QUESTIONS)

val_short_df = load_short_data(TRAIN_PATH, NUM_OF_TRAIN_QUESTIONS, NUM_OF_VAL_QUESTIONS)
train_short_df.head()
train_long_answers = encode(train_short_df['long_answer'].values, tokenizer)

train_questions = encode(train_short_df['question'].values, tokenizer)



val_long_answers = encode(val_short_df['long_answer'].values, tokenizer)

val_questions = encode(val_short_df['question'].values, tokenizer)
def form_short_labels(df, sentence_length):

    start_labels = np.zeros((len(df), sentence_length))

    end_labels = np.zeros((len(df), sentence_length))



    for i in range(len(df)):

        start = df.loc[i].short_start

        end = df.loc[i].short_end



        if start < 300 and end < 300:

            start_labels[i, start] = 1

            end_labels[i, end] = 1

        else:

            continue

    

    return start_labels, end_labels





train_start_labels, train_end_labels = form_short_labels(train_short_df, MAX_LEN)

val_start_labels, val_end_labels = form_short_labels(val_short_df, MAX_LEN)
print(train_short_df.loc[0].long_answer)

print('Start index: {0}'.format(train_start_labels[0]))

print('End index: {0}'.format(train_end_labels[0]))
# load from file

embedding_dict = {}



with open('../input/glove-global-vectors-for-word-representation/glove.6B.' + str(SHORT_EMBED_SIZE) + 'd.txt','r') as f:

    for line in f:

        values = line.split()

        word = values[0]

        vectors = np.asarray(values[1:],'float32')

        embedding_dict[word] = vectors

        

f.close()



# write to matrix

num_words = len(tokenizer.word_index) + 1

embedding_matrix = np.zeros((num_words, SHORT_EMBED_SIZE))



for word, i in tokenizer.word_index.items():

    if i > num_words:

        continue

    

    emb_vec = embedding_dict.get(word)

    

    if emb_vec is not None:

        embedding_matrix[i] = emb_vec

        

# load as tensorflow embedding

embedding = tf.keras.layers.Embedding(

    len(tokenizer.word_index) + 1,

    SHORT_EMBED_SIZE,

    embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix),

    trainable = False

)
# encode question

question_input = Input(shape=(None,))

question_x = embedding(question_input)

question_x = SpatialDropout1D(0.2)(question_x)

question_x = Bidirectional(LSTM(200, return_sequences=True))(question_x)

question_x = Bidirectional(LSTM(100, return_sequences=True))(question_x)



# encode answer

answer_input = Input(shape=(None,))

answer_x = embedding(answer_input)

answer_x = SpatialDropout1D(0.2)(answer_x)

answer_x = Bidirectional(LSTM(250, return_sequences=True))(answer_x)

answer_x = Bidirectional(LSTM(150, return_sequences=True))(answer_x)



# merge the encodings

combined_x = concatenate([question_x, answer_x])



# predict start index

start_x = Dropout(0.1)(combined_x) 

start_x = Conv1D(1,1)(start_x)

start_x = Flatten()(start_x)

start_x = Activation('softmax', name='start_token_out')(start_x)



# predict end index

end_x = Dropout(0.1)(combined_x) 

end_x = Conv1D(1,1)(end_x)

end_x = Flatten()(end_x)

end_x = Activation('softmax', name='end_token_out')(end_x)



# merge the parts into one model

short_model = tf.keras.models.Model(inputs=[answer_input, question_input], outputs=[start_x, end_x])
short_model.compile(

    loss='categorical_crossentropy', 

    optimizer='adam',

    metrics=['categorical_accuracy', 'Recall', 'Precision']

)
callbacks = [

    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=3, verbose=1),

    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1),

]
history = short_model.fit(

    x = [train_long_answers, train_questions], 

    y = [train_start_labels, train_end_labels],

    validation_data = (

        [val_long_answers, val_questions], 

        [val_start_labels, val_end_labels]

    ),

    epochs = SHORT_EPOCHS,

    callbacks = callbacks,

    batch_size = SHORT_BATCH_SIZE,

    shuffle = True

)
print('Epoch: {0}'.format(len(history.history['loss'])))

print('Loss: {0}'.format(history.history['loss'][-1]))
fig, ax = plt.subplots(1, 2, figsize=(15, 5))



ax[0].set_title('Training Loss')

ax[0].plot(history.history['loss'])



ax[1].set_title('Validation Loss')

ax[1].plot(history.history['val_loss'])
fig, ax = plt.subplots(3, 2, figsize=(15, 10))



fig.suptitle('Start Token')



ax[0,0].set_title('Training Accuracy')

ax[0,0].plot(history.history['start_token_out_categorical_accuracy'])



ax[0,1].set_title('Validation Accuracy')

ax[0,1].plot(history.history['val_start_token_out_categorical_accuracy'])



ax[1,0].set_title('Training Recall')

ax[1,0].plot(history.history['start_token_out_recall'])



ax[1,1].set_title('Validation Recall')

ax[1,1].plot(history.history['val_start_token_out_recall'])



ax[2,0].set_title('Training Precision')

ax[2,0].plot(history.history['start_token_out_precision'])



ax[2,1].set_title('Validation Precision')

ax[2,1].plot(history.history['val_start_token_out_precision'])
fig, ax = plt.subplots(3, 2, figsize=(15, 10))



fig.suptitle('End Token')



ax[0,0].set_title('Training Accuracy')

ax[0,0].plot(history.history['end_token_out_categorical_accuracy'])



ax[0,1].set_title('Validation Accuracy')

ax[0,1].plot(history.history['val_end_token_out_categorical_accuracy'])



ax[1,0].set_title('Training Recall')

ax[1,0].plot(history.history['end_token_out_recall_1'])



ax[1,1].set_title('Validation Recall')

ax[1,1].plot(history.history['val_end_token_out_recall_1'])



ax[2,0].set_title('Training Precision')

ax[2,0].plot(history.history['end_token_out_precision_1'])



ax[2,1].set_title('Validation Precision')

ax[2,1].plot(history.history['val_end_token_out_precision_1'])
accuracy = history.history['start_token_out_categorical_accuracy'][-1]

recall = history.history['start_token_out_recall'][-1]

precision = history.history['start_token_out_precision'][-1]





print('Training')

print('Start token accuracy: {0}'.format(accuracy))

print('Start token recall: {0}'.format(recall))

print('Start token precision: {0}'.format(precision))

print('Start token F1 score: {0:.4f}'.format(

    2 * (precision * recall) / (precision + recall)

))



accuracy = history.history['end_token_out_categorical_accuracy'][-1]

recall = history.history['end_token_out_recall_1'][-1]

precision = history.history['end_token_out_precision_1'][-1]



print('End token accuracy: {0}'.format(accuracy))

print('End token recall: {0}'.format(recall))

print('End token precision: {0}'.format(precision))

print('End token F1 score: {0:.4f}'.format(

    2 * (precision * recall) / (precision + recall)

))
accuracy = history.history['val_start_token_out_categorical_accuracy'][-1]

recall = history.history['val_start_token_out_recall'][-1]

precision = history.history['val_start_token_out_precision'][-1]





print('Validation')

print('Start token accuracy: {0}'.format(accuracy))

print('Start token recall: {0}'.format(recall))

print('Start token precision: {0}'.format(precision))

print('Start token F1 score: {0:.4f}'.format(

    2 * (precision * recall) / (precision + recall)

))



accuracy = history.history['val_end_token_out_categorical_accuracy'][-1]

recall = history.history['val_end_token_out_recall_1'][-1]

precision = history.history['val_end_token_out_precision_1'][-1]



print('End token accuracy: {0}'.format(accuracy))

print('End token recall: {0}'.format(recall))

print('End token precision: {0}'.format(precision))

print('End token F1 score: {0:.4f}'.format(

    2 * (precision * recall) / (precision + recall)

))
def test_short_answer(question, long_answer):

    sentences = [long_answer, question]

    

    sentences = encode(sentences, tokenizer)

    

    predictions = short_model.predict(

        [np.expand_dims(sentences[0], axis=0), np.expand_dims(sentences[1], axis=0)]

    )

    

    predictions = np.array(predictions)

    

    pred_start = np.argmax(predictions[0,0])

    pred_end = np.argmax(predictions[1,0])

    pred_string = ' '.join(long_answer.split(' ')[pred_start:pred_end])



    return pred_start, pred_end, pred_string
question = 'which is the most common use of opt-in e-mail marketing'

long_answer = "<P> A common example of permission marketing is a newsletter sent to an advertising firm 's customers . Such newsletters inform customers of upcoming events or promotions , or new products . In this type of advertising , a company that wants to send a newsletter to their customers may ask them at the point of purchase if they would like to receive the newsletter . </P>"
start, end, short_answer = test_short_answer(question, long_answer)



print('Start token: ' + str(start))

print('End token: ' + str(end))

print('Answer: ' + short_answer)
question = 'how i.met your mother who is the mother'

long_answer = "<P> Tracy McConnell , better known as `` The Mother '' , is the title character from the CBS television sitcom How I Met Your Mother . The show , narrated by Future Ted , tells the story of how Ted Mosby met The Mother . Tracy McConnell appears in 8 episodes from `` Lucky Penny '' to `` The Time Travelers '' as an unseen character ; she was first seen fully in `` Something New '' and was promoted to a main character in season 9 . The Mother is played by Cristin Milioti . </P>"
start, end, short_answer = test_short_answer(question, long_answer)



print('Start token: ' + str(start))

print('End token: ' + str(end))

print('Answer: ' + short_answer)