import numpy as np
from tqdm import tqdm
import re
from string import punctuation
import pickle
import gc

from sklearn.model_selection import train_test_split

from keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy
from keras import utils
pairs = pickle.load(open('/kaggle/input/chatbot-training-conversations/chatbot_training_conversations.pkl','rb'))
pairs = pairs[:12000]
print(pairs[:1])
questions, answers = [],[]

for pair in pairs:
    questions.append(pair['request'])
    answers.append('<start> '+pair['reply']+' <eos>')
    
print(len(questions), len(answers))
del(pairs)
gc.collect()
#Create Vocabulary

vocab = {}

for question, answer in zip(questions, answers):
    text = question + ' ' + answer
    for word in text.split(' '):
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

print(len(vocab))
#word2idx and idx2word

word2idx = {}
idx2word = {}

word2idx['<pad>'] = 0
idx2word[0] = '<pad>'

for i, (word, count) in enumerate(vocab.items()):
    word2idx[word] = i+1
    idx2word[i+1] = word
vocab['<pad>'] = 1

print(len(word2idx), len(idx2word), len(vocab))
#del(vocab)
gc.collect()
questions_to_seq = []
answers_to_seq = []

for question, answer in zip(questions, answers):
    questions_to_seq.append([word2idx[word] for word in question.split(' ')])
    answers_to_seq.append([word2idx[word] for word in answer.split(' ')])

print(len(questions_to_seq), len(answers_to_seq))
del(answers, questions)
gc.collect()
max_question_len = max(len(q) for q in questions_to_seq)
max_answer_len = max(len(a) for a in answers_to_seq)

max_len = max(max_question_len, max_answer_len)

vocab_size = len(vocab)

print(max_len,max_question_len, max_answer_len, vocab_size)
questions_padded = pad_sequences(questions_to_seq, maxlen=max_len, padding='post')
answers_padded = pad_sequences(answers_to_seq, maxlen=max_len, padding='post')
answers_to_seq_1 = answers_to_seq

for i in range(len(answers_to_seq_1)):
    answers_to_seq_1[i] = answers_to_seq_1[i][1:]
answers_padded_1 = pad_sequences(answers_to_seq_1, maxlen=max_len, padding='post')
onehot_answers = utils.to_categorical(answers_padded_1, vocab_size)
del([answers_to_seq, answers_to_seq_1, questions_to_seq, answers_padded_1])
gc.collect()
decoder_output_data = np.array(onehot_answers)
del(onehot_answers)
gc.collect()
encoder_input_data = np.array(questions_padded)
del(questions_padded)
gc.collect()
decoder_input_data = np.array(answers_padded)
del(answers_padded)
gc.collect()
glove_dict = pickle.load(open('/kaggle/input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl','rb'))
print(len(glove_dict))
'''def read_glove(file):
    with open(file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            word = line[0]
            words.add(word)
            word_to_vec_map[word] = np.array(line[1:], dtype = np.float64)
    return words, word_to_vec_map

glove_words, glove_dict = read_glove('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.50d.txt')
del(glove_words)
gc.collect() '''
embeddings_dim = 300
embedding_matrix = np.zeros((vocab_size, embeddings_dim))

for word, index in word2idx.items():
    try:
        embedding_matrix[index,:] = glove_dict[word]
    except:
        continue

del(glove_dict)
print(embedding_matrix.shape)
gc.collect()
pickle.dump(embedding_matrix,open('embedding_matrix.pkl','wb'))
gc.collect()
embedding_layer = Embedding(input_dim = vocab_size, output_dim = embeddings_dim, input_length = max_len, weights= [embedding_matrix], trainable= False)
del(embedding_matrix)
gc.collect()
encoder_input = Input(shape=(max_len,), dtype='int32',)
encoder_embedding = embedding_layer(encoder_input)
encoder_LSTM = LSTM(embeddings_dim, return_state= True)
encoder_output, state_h, state_c = encoder_LSTM(encoder_embedding)
encoder_state = [state_h, state_c]

decoder_input = Input(shape=(max_len, ), dtype='int32')
decoder_embedding = embedding_layer(decoder_input)
decoder_LSTM = LSTM(embeddings_dim, return_state=True, return_sequences= True)
decoder_output, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])
decoder_dense = Dense(vocab_size, activation = 'softmax')
output = decoder_dense(decoder_output)

model = Model([encoder_input, decoder_input], output)
model.compile(optimizer = RMSprop(), loss= 'categorical_crossentropy')
model.summary()
'''model_stuff={'encoder_input':encoder_input,
           'encoder_state':encoder_state,
           'decoder_LSTM':decoder_LSTM,
           'decoder_embedding':decoder_embedding,
           'decoder_dense':decoder_dense,
           } '''
#pickle.dump(model_stuff, open('model_stuff.pkl','wb'))
history = model.fit([encoder_input_data, decoder_input_data], decoder_output_data, batch_size=50, epochs=200)
#model.save('seq2seq_kaggle.h5')
model.save_weights('model_weights.h5', overwrite=True)
gc.collect()
encoder_model = Model(encoder_input, encoder_state)

decoder_input_state_h = Input(shape=(embeddings_dim, ))
decoder_input_state_c = Input(shape=(embeddings_dim, ))

decoder_state_inputs = [decoder_input_state_h, decoder_input_state_c]

decoder_output, state_h, state_c = decoder_LSTM(decoder_embedding, initial_state=decoder_state_inputs)
decoder_states = [state_h, state_c]
decoder_output = decoder_dense(decoder_output)

decoder_model = Model([decoder_input] + decoder_state_inputs, [decoder_output]+decoder_states)
def preprocess_text(line):
    GOOD_SYMBOLS_RE = re.compile('[^0-9a-z ]')
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;#+_]')
    REPLACE_SEVERAL_SPACES = re.compile('\s+')
    line = line.lower()
    line = re.sub('\n','', line)
    line = re.sub('can\'t','cannot', line)
    line = re.sub('won\'t','will not', line)
    line = re.sub('\'ll',' will', line)
    line = re.sub('n\' t',' not',line)
    line = re.sub('\'m',' am', line)
    line = re.sub('\'d',' would', line)
    line = re.sub('\'re',' are', line)
    line = re.sub('\'ve',' have', line)
    line = re.sub('\'s',' is', line)
    line = REPLACE_BY_SPACE_RE.sub(' ', line)
    line = GOOD_SYMBOLS_RE.sub('', line)
    line = REPLACE_SEVERAL_SPACES.sub(' ', line)
    return line.strip()
def tokenize_sentence(text, max_len=max_len):
    seq = []
    text = preprocess_text(text)
    for word in text.split(' '):
        seq.append(word2idx[word])
    seq = pad_sequences([seq], maxlen=max_len, padding='post')
    return seq
x = tokenize_sentence('hello')
y = encoder_model.predict(x)
target = np.zeros((1,1))
target[0,0] = word2idx['<start>']
target
o, h, c = decoder_model.predict([target] + y)
np.argmax(o[0,-1,:])

pickle.dump(idx2word,open('index_to_word.pkl','wb'))
pickle.dump(word2idx, open('word_to_index.pkl','wb'))
def make_talk():
    states_values = encoder_model.predict(tokenize_sentence(input( 'Enter question: ' )))
    #print('input taken')
    empty_target_seq = np.zeros((1,1))
    empty_target_seq[0,0] = word2idx['<start>']
    stop_condition = False
    decoded_translation = ''
    #print('while loop')
    while not stop_condition:
        #print(stop_condition)
        dec_outputs, h,c = decoder_model.predict([empty_target_seq] + states_values)
        sampled_word_index = np.argmax(dec_outputs[0,-1,:])
        #print(sampled_word_index)
        decoded_translation += ' ' + idx2word[sampled_word_index]
        sampled_word = idx2word[sampled_word_index]
        #print(sampled_word)
        
        if sampled_word == '<eos>' or len(decoded_translation.split(' ')) > max_len:
            stop_condition = True
        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ]
    return decoded_translation
make_talk()
