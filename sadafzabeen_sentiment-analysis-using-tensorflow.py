from tensorflow.python.keras.datasets import imdb
#We will create 4 arrays which will be populated by load_data function.

(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=10000)   #This means that only the top 10000 from the Bag of words will be used and the rest will be ignored.
x_train[0] #We see that we have a review at 0'th index already converted to its numeric representation
y_train[0] #Its an output variable which is positive or negative review
word_index=imdb.get_word_index()

print(word_index["love"])
reverse_word_index=dict((value,key)for key,value in word_index.items())



def decode(review):

    text=""

    for i in review:

        text += reverse_word_index[i]

        text +=" "

    return text

        

    
decode (x_train[0])
def show_lengths():

    print('Length of 1st training example: ', len(x_train[0]))

    print('Length of 2nd training example: ',  len(x_train[1]))

    print('Length of 1st test example: ', len(x_test[0]))

    print('Length of 2nd test example: ',  len(x_test[1]))

    

show_lengths()
from tensorflow.python.keras.preprocessing.sequence import pad_sequences



x_train = pad_sequences(x_train, value = word_index['the'], padding = 'post', maxlen = 256)

x_test = pad_sequences(x_test, value = word_index['the'], padding = 'post', maxlen = 256)
show_lengths()
decode(x_train[0])
from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Embedding, Dense, GlobalAveragePooling1D



model = Sequential([

    Embedding(10000, 16),             #First layer, vocab size=10,000 & 16 is no. of features

    GlobalAveragePooling1D(),         #16 dimension vector for each batch

    Dense(16, activation = 'relu'),

    Dense(1, activation = 'sigmoid')  #output layer, sigmoid is for binary classification

])



model.compile(

    optimizer = 'adam',

    loss = 'binary_crossentropy',

    metrics = ['acc']

)



model.summary()
from tensorflow.python.keras.callbacks import LambdaCallback



simple_logging = LambdaCallback(on_epoch_end = lambda e, l: print(e, end='.'))



E = 20



h = model.fit(

    x_train, y_train,

    validation_split = 0.2,

    epochs = E,

    callbacks = [simple_logging],

    verbose = False

)
import matplotlib.pyplot as plt

%matplotlib inline
plt.plot(range(E),h.history['acc'],label='Training')

plt.plot(range(E),h.history['val_acc'],label='Validation')

plt.legend()

plt.show()
loss,acc=model.evaluate(x_test,y_test)

print(acc*100)