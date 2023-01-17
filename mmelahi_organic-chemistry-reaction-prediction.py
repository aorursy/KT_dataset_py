from keras.layers import Bidirectional, Concatenate, Dot, Input, Permute, Reshape

from keras.layers import RepeatVector, Dense, Activation, GRU, Lambda, Add

from keras import optimizers, regularizers, initializers

from keras.engine.topology import Layer

from keras.utils import to_categorical

from keras.models import load_model, Model

from keras.utils import Sequence

from keras.callbacks import ModelCheckpoint, Callback

import keras.backend as K

import keras

import numpy as np

from nltk.translate.bleu_score import sentence_bleu

import random

from keras.utils.vis_utils import plot_model

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt





random.seed(40)



def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):

    

    X, Y = zip(*dataset)

    

    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])

    Y = [string_to_int(t, Ty, machine_vocab) for t in Y]

    

    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X)))

    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y)))



    return X, np.array(Y), Xoh, Yoh

        

        

def string_to_int(string, length, vocab):

    """

    Converts all strings in the vocabulary into a list of integers representing the positions of the

    input string's characters in the "vocab"

    

    Arguments:

    string -- input string

    length -- the number of time steps you'd like, determines if the output will be padded or cut

    vocab -- vocabulary, dictionary used to index every character of your "string"

    

    Returns:

    rep -- list of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary

    """



    u = vocab["<unk>"]   

    if len(string) > length:

        string = string[:length]

        

    rep = list(map(lambda x: vocab.get(x, u), string))

    

    if len(string) < length:

        rep += [vocab['<pad>']] * (length - len(string))

    

    #print (rep)

    return rep





def int_to_string(ints, inv_vocab):

    """

    Output a machine readable list of characters based on a list of indexes in the machine's vocabulary

    

    Arguments:

    ints -- list of integers representing indexes in the machine's vocabulary

    inv_vocab -- dictionary mapping machine readable indexes to machine readable characters 

    

    Returns:

    l -- list of characters corresponding to the indexes of ints thanks to the inv_vocab mapping

    """

    

    l = [inv_vocab[i] for i in ints]

    return l





def softmax(x, axis=-1):

    """Softmax activation function.

    # Arguments

        x : Tensor.

        axis: Integer, axis along which the softmax normalization is applied.

    # Returns

        Tensor, output of softmax transformation.

    # Raises

        ValueError: In case `dim(x) == 1`.

    """

    ndim = K.ndim(x)

    if ndim == 2:

        return K.softmax(x)

    elif ndim > 2:

        e = K.exp(x - K.max(x, axis=axis, keepdims=True))

        s = K.sum(e, axis=axis, keepdims=True)

        return e / s

    else:

        raise ValueError('Cannot apply softmax to a tensor that is 1D')
K.set_floatx('float32')

#K.set_epsilon(1e-05)
n_a = 256

n_s = 256

BATCH_SIZE = 512

data_path = '../input/ocrtrain.csv'

lines = open(data_path).read().split('\n')
input_characters = set()

target_characters = set()

data_path = '../input/ocrtrain.csv'

lines = open(data_path).read().split('\n')



max_x = 5

max_y = 5

max_len = 50

all_idx = []

#for line in lines[0: len(lines) - 1]:

for idx, line in enumerate(lines[0: len(lines) - 1]):

    input_text, target_text = line.split(',')

    input_text = input_text[1:-2]

    input_text = input_text.split(' ')

    target_text = target_text.split(' ')

    if len(input_text) <= max_len and len(target_text) <= max_len:

        all_idx.append(idx)

        

        if len(input_text) > max_x:

            max_x = len(input_text)



        if len(target_text) > max_y:

            max_y = len(target_text)



        for char in input_text:

            if char not in input_characters:

                input_characters.add(char)

        for char in target_text:

            if char not in target_characters:

                target_characters.add(char)

                

Tx = max_x

Ty = max_y

#all_idx = list(range(0,len(lines) - 1))

train_idx, valid_idx = train_test_split(all_idx, test_size=0.05, random_state = 43)



input_characters = sorted(list(input_characters)) + ['<unk>', '<pad>']

target_characters = sorted(list(target_characters)) + ['<unk>', '<pad>']

reactants_vocab = {v:k for k,v in enumerate(input_characters)}

products_vocab = {v:k for k,v in enumerate(target_characters)}

inv_products_vocab = {v:k for k,v in products_vocab.items()}
def load_data(idx):

    dataset = []

    #for line in lines[idx]:

    line = lines[idx]

    input_text, target_text = line.split(',')

    input_text = input_text[1:-2]

    input_text = input_text.split(' ')

    target_text = target_text.split(' ')

    ds = (input_text,target_text)

    dataset.append(ds)

    return dataset    
class SMILESDataGenerator(keras.utils.Sequence):

            

    def __init__(self, all_idx, batch_size, shuffle = False):

        self.all_idx = all_idx

        self.batch_size = batch_size

        self.shuffle = shuffle

    

    def __len__(self):

        return int(np.ceil(len(self.all_idx) / float(self.batch_size)))

    

    def __getitem__(self, idx):

        #indexes = [idx * self.batch_size : (idx+1) * self.batch_size]

        batch_indexes = self.all_idx[idx * self.batch_size:(idx + 1) * self.batch_size]



        #all_idx = self.all_idx[indexes]

        X = np.zeros((self.batch_size, Tx, len(reactants_vocab)))

        y = np.zeros((self.batch_size, Ty, len(products_vocab)))

        # Generate data

        for i, idx_ in enumerate(batch_indexes):

            X[i,:,:], y[i,:,:] = self.__load_data(idx_)

        s0 = np.zeros((self.batch_size, n_s))

        y = list(y.swapaxes(0,1))

        X = [X, s0]

        return X,y

        

    def on_epoch_end(self):

        

        # Updates indexes after each epoch

        #self.indexes = np.arange(len(self.all_idx))

        if self.shuffle == True:

            np.random.shuffle(self.all_idx)



    def __iter__(self):

        """Create a generator that iterate over the Sequence."""

        for item in (self[i] for i in range(len(self))):

            yield item

            

    def __load_data(self, idx_):

        dataset = load_data(idx_)

        X, Y, Xoh, Yoh = preprocess_data(dataset, reactants_vocab, products_vocab, Tx, Ty)

        return Xoh, Yoh
repeator = RepeatVector(1)

permutor = Permute((2,1))

dotor1 = Lambda(lambda x: K.batch_dot(x[0],x[1]))

activator = Activation('softmax')

dotor2 = Lambda(lambda x: K.batch_dot(x[0],x[1]))



def one_step_attention(a, s_prev):

    """

    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights

    "alphas" and the hidden states "a" of the Bi-GRU.

    

    Arguments:

    a -- hidden state output of the Bi-GRU, numpy-array of shape (m, Tx, n_a)

    s_prev -- previous hidden state of the (post-attention) GRU, numpy-array of shape (m, n_s)

    

    Returns:

    context -- context vector, input of the next (post-attetion) GRU cell

    """

    

    s_prev = repeator(s_prev)

    a_trans = permutor(a)

    alphas = dotor1([s_prev,a_trans])

    alphas = activator(alphas)

    c = dotor2([alphas,a])

    

    return c
repeator_rev = RepeatVector(1)

permutor_rev = Permute((2,1))

dotor1_rev = Lambda(lambda x: K.batch_dot(x[0],x[1]))

activator_rev = Activation('softmax')

dotor2_rev = Lambda(lambda x: K.batch_dot(x[0],x[1]))



def one_step_attention_rev(a, s_prev):

    """

    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights

    "alphas" and the hidden states "a" of the Bi-GRU.

    

    Arguments:

    a -- hidden state output of the Bi-GRU, numpy-array of shape (m, Tx, n_a)

    s_prev -- previous hidden state of the (post-attention) GRU, numpy-array of shape (m, n_a)

    

    Returns:

    context -- context vector, input of the next (post-attetion) GRU cell

    """

    

    s_prev = repeator_rev(s_prev)

    a_trans = permutor_rev(a)

    alphas = dotor1_rev([s_prev,a_trans])

    alphas = activator_rev(alphas)

    c = dotor2_rev([alphas,a])

    

    return c
post_activation_GRU_cell = GRU(n_s,dropout=0.1,recurrent_dropout=0.1)

post_activation_rev_GRU_cell = GRU(n_s,dropout=0.1,recurrent_dropout=0.1)

add_states1 = Add()

add_states2 = Add()

reshaper1 = Reshape((n_s,))

reshaper2 = Reshape((n_s,))

concatenator1 = Concatenate(axis=-1)

concatenator2 = Concatenate(axis=-1)

densor1 = Dense(n_s,activation='tanh')

densor2 = Dense(n_s,activation='tanh')

output_layer = Dense(len(products_vocab), activation='softmax',

                     bias_initializer=initializers.Constant(value=0.025),

                     activity_regularizer=regularizers.l2(0.05))





def model(Tx, Ty, n_a,n_s,reactants_vocab_size, products_vocab_size):

    """

    Arguments:

    Tx -- length of the input sequence

    Ty -- length of the output sequence

    n_a -- hidden state size of the Bi-LSTM

    n_s -- hidden state size of the post-attention LSTM

    human_vocab_size -- size of the python dictionary "human_vocab"

    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:

    model -- Keras model instance

    """

    

    X = Input(shape=(Tx, reactants_vocab_size))

    s0 = Input(shape=(n_s,), name='s0')

    s = s0

    context_rev_seq = []

    av2_seq = []

    outputs = []

    a = Bidirectional(GRU(n_a,return_sequences=True,

                          recurrent_dropout=0.1),merge_mode='sum')(X)

    a_rev = Lambda(lambda x: K.reverse(x,axes=1),output_shape=(Tx,n_a,))(a)

    

    for t in range(Ty):

        context_rev = one_step_attention_rev(a_rev, s)

        s = post_activation_rev_GRU_cell(context_rev, initial_state = s)

        contextr = reshaper2(context_rev)

        c2 = concatenator2([contextr,s])

        av2 = densor2(c2)

        av2_seq.append(av2)

        context_rev_seq.append(context_rev)

    

    for t in range(Ty):

        context = one_step_attention(a, s)

        context_add = add_states1([context,context_rev_seq[Ty-1-t]])

        s = post_activation_GRU_cell(context_add, initial_state = s)

        context = reshaper1(context)

        c1 = concatenator1([context,s])

        av1 = densor1(c1)

        av = add_states2([av1,av2_seq[Ty-1-t]])

        out = output_layer(av)

        outputs.append(out)

        

    model = Model(inputs = [X,s0], outputs = outputs)

    return model





model = model(Tx, Ty, n_a,n_s, len(reactants_vocab), len(products_vocab))

model.summary()
#plot_model(model, to_file='OCR.png', show_shapes=True)
tg = SMILESDataGenerator(train_idx, BATCH_SIZE, shuffle = True)

vg = SMILESDataGenerator(valid_idx, BATCH_SIZE, shuffle = True)

checkpoint1 = ModelCheckpoint('model-CRP_best.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')

checkpoint2 = ModelCheckpoint('model-CRP.h5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='min',period=1) 

#model.fit([Xoh,s0], outputs, batch_size = batch_size, epochs=1)

#model.save_weights('weights256.h5')
class SGDRScheduler(Callback):

    '''Cosine annealing learning rate scheduler with periodic restarts.

    # Usage

        ```python

            schedule = SGDRScheduler(min_lr=1e-5,

                                     max_lr=1e-2,

                                     steps_per_epoch=np.ceil(epoch_size/batch_size),

                                     lr_decay=0.9,

                                     cycle_length=5,

                                     mult_factor=1.5)

            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])

        ```

    # Arguments

        min_lr: The lower bound of the learning rate range for the experiment.

        max_lr: The upper bound of the learning rate range for the experiment.

        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 

        lr_decay: Reduce the max_lr after the completion of each cycle.

                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.

        cycle_length: Initial number of epochs in a cycle.

        mult_factor: Scale epochs_to_restart after each full cycle completion.

    # References

        Blog post: jeremyjordan.me/nn-learning-rate

        Original paper: http://arxiv.org/abs/1608.03983

    '''

    def __init__(self,

                 min_lr,

                 max_lr,

                 steps_per_epoch,

                 lr_decay=1,

                 cycle_length=10,

                 mult_factor=2):



        self.min_lr = min_lr

        self.max_lr = max_lr

        self.lr_decay = lr_decay



        self.batch_since_restart = 0

        self.next_restart = cycle_length



        self.steps_per_epoch = steps_per_epoch



        self.cycle_length = cycle_length

        self.mult_factor = mult_factor



        self.history = {}



    def clr(self):

        '''Calculate the learning rate.'''

        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)

        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))

        return lr



    def on_train_begin(self, logs={}):

        '''Initialize the learning rate to the minimum value at the start of training.'''

        logs = logs or {}

        K.set_value(self.model.optimizer.lr, self.max_lr)



    def on_batch_end(self, batch, logs={}):

        '''Record previous batch statistics and update the learning rate.'''

        logs = logs or {}

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))

        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)



        self.batch_since_restart += 1

        K.set_value(self.model.optimizer.lr, self.clr())



    def on_epoch_end(self, epoch, logs={}):

        '''Check for end of current cycle, apply restarts when necessary.'''

        if epoch + 1 == self.next_restart:

            self.batch_since_restart = 0

            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)

            self.next_restart += self.cycle_length

            self.max_lr *= self.lr_decay

            self.best_weights = self.model.get_weights()



    def on_train_end(self, logs={}):

        '''Set weights to the values from the end of the most recent cycle for best performance.'''

        self.model.set_weights(self.best_weights)

        

schedule = SGDRScheduler(min_lr=1e-6,

                         max_lr=1e-3,

                         steps_per_epoch=len(tg),

                         lr_decay=1,

                         cycle_length=10,

                         mult_factor=1)
opt = optimizers.Adam(lr = 0.005, beta_1=0.9, beta_2=0.999, decay=0.0005)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_name = 'model-CRP-1.h5'

weigh_path = 'https://www.kaggleusercontent.com/kf/12327992/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..zIlibl9Qa0Pcug8AeifFOA.RXz6fPMGmmw9s6QEsIDDuhYpttFCFmmH-Ac2GYAo_Jasdgl0EOLyfIQLeeVYU-3Ng2BZ7-yUqrpPXHL99uc5AGDYpMKXI5BxCnf5XRSZdLLdFalhWjYSCWhyzLV2u2epUnREdi5ymC316CFZZu8LZxqJEejamoiTmCJTch5crt4.xf9ut4ls0iePtHldkbaDpQ/model-CRP.h5' 

weights_path = get_file(model_name, weigh_path,cache_subdir='models')

model.load_weights(weights_path)
hist = model.fit_generator(

    tg,

    steps_per_epoch=len(tg),

    validation_data=vg,

    #validation_steps=8,

    epochs=30,

    use_multiprocessing=False,

    workers=1,

    verbose=1,

    callbacks=[checkpoint1,checkpoint2,schedule])
fig, ax = plt.subplots(1, 2, figsize=(15,5))

ax[0].set_title('loss')

ax[0].plot(hist.epoch, hist.history["loss"], label="Train loss")

ax[0].plot(hist.epoch, hist.history["val_loss"], label="Validation loss")

ax[1].set_title('acc')

ax[1].plot(hist.epoch, hist.history["acc"], label="Train acc")

ax[1].plot(hist.epoch, hist.history["val_acc"], label="Validation acc")

ax[0].legend()

ax[1].legend()
def beam_search_decoder(data, k=3):

    sequences = [[list(), 0.0]]

    # walk over each step in sequence

    for row in data:

        all_candidates = list()

        # expand each current candidate

        for i in range(len(sequences)):

            seq, score = sequences[i]

            for j in range(len(row)):

                candidate = [seq + [j], score-np.log(row[j])]

                all_candidates.append(candidate)

        # order all candidates by score

        ordered = sorted(all_candidates, key=lambda tup:tup[1])

        # select k best

        sequences = ordered[:k]

    return sequences
def predict_result(model,x_test): # predict both orginal and reflect x

    preds_test = model.predict_generator(x_test)

    return preds_test
output = []

pad = '<pad>'

m = len(valid_idx)

bleu_score = np.zeros((m,1))

#prediction = model.predict([Xoh, s0])

prediction = predict_result(model,vg)

count = 0

bw = 5

for i in range(m):

    p0 = np.array(prediction)[:,i,:]

    p0 = beam_search_decoder(p0,bw)

    for j in reversed(range(bw)):

        p = p0[j][0]

        p = int_to_string(p,inv_products_vocab)

        o2 = []

        for x in p:

            if x != pad:

                o2.append(x)

        o1 = o2

        o2 = ''.join(o2)

        

        if o2 == ''.join(dataset[i][1]):

            count += 1

            bleu_score[i] = sentence_bleu([dataset[i][1]], o1)

            output.append(''.join(dataset[i][0])+','+''.join(dataset[i][1])+','+o2+','+str(o2 == ''.join(dataset[i][1])))

            break;

        elif j==0:

            bleu_score[i] = sentence_bleu([dataset[i][1]], o1)

            output.append(''.join(dataset[i][0])+','+''.join(dataset[i][1])+','+o2+','+str(o2 == ''.join(dataset[i][1])))



f = open('accuracy_bw.txt','w')

f.write(str(count/m))

f.close()



f = open('bleu_score_bw.txt','w')

f.write(str(sum(bleu_score)/m))

f.close()



print(count/m)

print(sum(bleu_score)/m)



with open('predicted_bw.csv','w') as file:

    for line in output:

        file.write(line)

        file.write('\n')
output = []

pad = '<pad>'

bleu_score = np.zeros((m,1))

#prediction = model.predict([Xoh, s0, c0])

#prediction = model.predict([Xoh, s0])

count = 0

for i in range(m):

    p = np.argmax(np.array(prediction)[:,i,:], axis = 1)

    p = int_to_string(p,inv_products_vocab)

    o2 = []

    for x in p:

        if x != pad:

            o2.append(x)

    bleu_score[i] = sentence_bleu([dataset[i][1]], o2)

    o2 = ''.join(o2)

    if o2 == ''.join(dataset[i][1]):

        count += 1

    output.append(''.join(dataset[i][0])+','+''.join(dataset[i][1])+','+o2+','+str(o2 == ''.join(dataset[i][1])))



print(count/m)

print(sum(bleu_score)/m)



f = open('accuracy.txt','w')

f.write(str(count/m))

f.close()



f = open('bleu_score.txt','w')

f.write(str(sum(bleu_score)/m))

f.close()



with open('predicted.csv','w') as file:

    for line in output:

        file.write(line)

        file.write('\n')