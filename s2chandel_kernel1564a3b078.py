# Libs

import pandas as pd

import numpy as np

from sklearn_pandas import DataFrameMapper, cross_val_score

from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

import tensorflow_hub as hub

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from keras import Sequential, Model

from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Embedding

from scipy.stats import spearmanr





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')

data.head()
features = data[['qa_id', 'question_title', 'question_body', 'question_user_name',

        'question_user_page', 'answer', 

        'answer_user_name', 'answer_user_page',

        'url', 'category', 'host']]



# features.head(20)
def encoder(data):



    encoder = LabelEncoder()



    df = data[[

        'qa_id', 'question_title', 'question_body', 'question_user_name',

        'question_user_page', 'answer', 

        'answer_user_name', 'answer_user_page',

        'url', 'category', 'host'

    ]]

    mapper = DataFrameMapper([

        ('qa_id', None),

        ('question_title', encoder),

        ('question_body', None),

        ('question_user_name', encoder),

        ('question_user_page', encoder),

        ('answer', None),

        ('answer_user_name', encoder),

        ('answer_user_page', encoder),

        ('url', encoder),

        ('category', encoder),

        ('host', encoder),

    ])

    x = pd.DataFrame(mapper.fit_transform(data),columns=[

        'qa_id','question_title',

        'question_body','question_user_name',

        'question_user_page','answer','answer_user_name',

        'answer_user_page','url','category','host'

    ])





    return x

def scale(x):



    scaler = MinMaxScaler()



    df = x[[

        'question_title','question_user_name',

        'question_user_page','answer_user_name',

        'answer_user_page','url','category','host'

    ]]





    df = pd.DataFrame(scaler.fit_transform(df),columns=[

        'question_title','question_user_name',

        'question_user_page','answer_user_name',

        'answer_user_page','url','category','host'

    ])



    x = x.drop(columns=[

        'question_title','question_user_name',

        'question_user_page','answer_user_name',

        'answer_user_page','url','category','host'

    ])



    x = pd.concat([x,df],axis=1)

    x = x.drop(columns='qa_id')





    return x

def word2vec(x):

    tfidf = TfidfVectorizer()



    mapper = DataFrameMapper([

            ('question_title', None),

            ('question_body', tfidf),

            ('question_user_name', None),

            ('question_user_page', None),

            ('answer', tfidf),

            ('answer_user_name', None),

            ('answer_user_page', None),

            ('url', None),

            ('category', None),

            ('host', None),

        ])



    vectors = mapper.fit(x)

    return vectors
x = encoder(data)

x = scale(x)

word_vectors = word2vec(x)

x = pd.DataFrame(word_vectors.transform(x))
x.head()
x.head()

len(x.columns)
y = data[[

    'question_asker_intent_understanding',

    'question_body_critical', 'question_conversational',

    'question_expect_short_answer', 'question_fact_seeking',

    'question_has_commonly_accepted_answer',

    'question_interestingness_others', 'question_interestingness_self',

    'question_multi_intent', 'question_not_really_a_question',

    'question_opinion_seeking', 'question_type_choice',

    'question_type_compare', 'question_type_consequence',

    'question_type_definition', 'question_type_entity',

    'question_type_instructions', 'question_type_procedure',

    'question_type_reason_explanation', 'question_type_spelling',

    'question_well_written', 'answer_helpful',

    'answer_level_of_information', 'answer_plausible', 'answer_relevance',

    'answer_satisfaction', 'answer_type_instructions',

    'answer_type_procedure', 'answer_type_reason_explanation',

    'answer_well_written'

]]
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=1)


def model():



    model = Sequential()

#     model.add(activation)

    model.add(Dense(30, input_dim=82044)) #batch size 30

    model.add(Dense(30, activation = 'sigmoid')) #shape 1

    model.compile(loss='mse', optimizer='sgd', metrics=['mse'])

    # model.summary()

    return model



model = model()





history = model.fit(x_train,y_train,

                         epochs = 40,

                         batch_size=50,

                         validation_data = (x_test,y_test),

                         verbose=1,)


# model evaluation

loss, mse = model.evaluate(x_train,y_train, verbose=0)

print("Training MSE: {:.4f}".format(mse))



loss, mse = model.evaluate(x_test,y_test, verbose=0)

print("Testing MSE:  {:.4f}".format(mse))



acc = history.history



# model history plot(train_val_acc and train_val_loss)



import matplotlib.pyplot as plt

plt.style.use('ggplot')



def plot_history(history):

    acc = history.history['mse']

    val_acc = history.history['val_mse']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training mse')

    plt.xlabel('epochs')

    plt.ylabel('mse')

    plt.plot(x, val_acc, 'r', label='Validation mse')

    plt.title('Training and validation mse')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.xlabel('epochs')

    plt.ylabel('loss')



    plt.legend()



plot_history(history)
y_pred = model.predict(x_test)
spearmanr(y_test, y_pred, axis=None)
test_data = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')

test_ids = test_data['qa_id']

test_ids.columns = 'qa_id'
test_data.head()
xtest= pd.get_dummies(test_data['question_user_name'])

xtest= encoder(data=test_data)

xtest= scale(xtest)

# vectors = word2vec(x)

xtest = pd.DataFrame(word_vectors.transform(xtest))
xtest.head()
pred = model.predict(xtest)
predictions = pd.DataFrame(pred,columns=['question_asker_intent_understanding',

       'question_body_critical', 'question_conversational',

       'question_expect_short_answer', 'question_fact_seeking',

       'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written', 'answer_helpful',

       'answer_level_of_information', 'answer_plausible', 'answer_relevance',

       'answer_satisfaction', 'answer_type_instructions',

       'answer_type_procedure', 'answer_type_reason_explanation',

       'answer_well_written'],index=test_ids)

predictions = predictions.reset_index()

predictions.head()
predictions.describe()
predictions.to_csv('submission.csv',index=False)