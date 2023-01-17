import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt
test_corpus = np.load('../input/preprocessing/test_corp.npy')

train_corpus = np.load('../input/preprocessing/train_corp.npy')



glove_table = pd.read_csv('../input/preprocessing/filled_glove_table.csv', index_col=0)

glove_table.describe()



train_data = pd.read_csv('../input/spooky-author-identification/train.csv')
glove_table.loc[['man','woman','man']].as_matrix().shape



train_data.iloc[477]
train_sequence = []

train_sequence_lengths   = []

index = 0

for sentence_chunk in train_corpus:

    if (len(sentence_chunk) > 0):

        words = [word for (word, tag) in sentence_chunk]

        features = glove_table.loc[words].as_matrix()

        train_sequence.append(features)

        train_sequence_lengths.append(len(words))

    else:

        # If sentence is empty put (1,ndims) zeros as data

        print(train_data['text'][index])

        train_sequence.append(np.zeros((1,300)))

        train_sequence_lengths.append(1)

    index += 1

    

train_sequence_lengths = np.array(train_sequence_lengths)
from hmmlearn import hmm

def hmm_train_model(train_sequence, train_sequence_lengths):

    num_hidden_states = 10

    num_dims = train_sequence[0].shape[1]

    

    model = hmm.GaussianHMM(n_components=num_hidden_states, covariance_type="full",

                            n_iter=20, verbose=True, init_params = "")



    start_prob = np.random.rand(num_hidden_states)

    start_prob = start_prob / np.sum(start_prob)

    model.startprob_ = start_prob



    transmat = np.random.rand(num_hidden_states,num_hidden_states)

    transmat = transmat / np.sum(transmat, axis=1)[:, np.newaxis]

    model.transmat_ = transmat



    model.means_ = np.random.rand(num_hidden_states, num_dims)

    model.covars_ = np.tile(np.identity(num_dims), (num_hidden_states, 1, 1))

    print('Training started.')

    return model.fit(np.concatenate(train_sequence), train_sequence_lengths) 
import warnings



# List of authors

authors = np.unique(train_data['author'])

author_models = {} # Stores 1 Model for each author



for author in authors:

    print('Training for {}:'.format(author))

    

    # Filter sentences of the same author :D

    is_author_sentence = np.array(train_data['author'] == author)

    author_sentences = [train_sequence[i] if is_author_sentence[i] else None for i in range(len(train_sequence))]

    author_sentences = list(filter(lambda x: x is not None,author_sentences))

    

    author_sentence_lengths = train_sequence_lengths[is_author_sentence]

    

    # Train for author

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        author_models[author] = hmm_train_model(author_sentences, author_sentence_lengths)
def predict(author_models, sentence):

    scores = {}

    predicted_author = list(author_models.keys())[0]

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        for author in author_models.keys():

            scores[author] = author_models[author].score(sentence)

            if (scores[author] > scores[predicted_author]):

                predicted_author = author

    return predicted_author, scores



correct_prediction = 0

total_prediction = 0

for index in range(1000):

    prediction, scores = predict(author_models, train_sequence[index]) 

    true_data = train_data.iloc[index]

    if (true_data['author'] == prediction):

        correct_prediction += 1

    total_prediction += 1

    #print("{}\nPrediction: {}\tTrue: {}\n".format(true_data['text'], prediction, true_data['author']))

    

acc = float(correct_prediction) / total_prediction

print(acc)