# Got the code from the link - https://stackabuse.com/text-generation-with-python-and-tensorflow-keras/
# Post from Dan Nelson
# I will be altering the code later for my own neural net

# Layered code which adds more functionality to your code 
# Still early days yet still comes out with gibberish im going to alter the layer types over
# time to get the thing to work not sure whats wrong will try other models 
import numpy
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.layers import ELU
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
file = open("../input/gothic-literature/frankenstein.txt").read() # Copy the file path from using add data on the right
# Create a function that makes the text easier to manipulate by the computer

def tokenize_words(input):
    # lowercase everything to standardize it
    input = input.lower()

    # instantiate the tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    # if the created token isn't in the stop words, make it part of "filtered"
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)
# preprocess the input data, make tokens
processed_inputs = tokenize_words(file)
# Tokenize make the text easier to process for the computer 

# Preprocess the input data, make tokens

processed_inputs = tokenize_words(file)
# We need to work out vocabulary length and character length

chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c, i) for i, c in enumerate(chars))
input_len = len(processed_inputs)
vocab_len = len(chars)
print ("Total number of characters:", input_len)
print ("Total vocab:", vocab_len)
# Numbers array store - I/O arrays list - size seq_length of 100

seq_length = 100
x_data = []
y_data = []
# Now let's covert the text into numbers

# loop through inputs, start at the beginning and go until we hit
# the final character we can create a sequence out of
for i in range(0, input_len - seq_length, 1):
    # Define input and output sequences
    # Input is the current character plus desired sequence length
    in_seq = processed_inputs[i:i + seq_length]

    # Out sequence is the initial character plus total sequence length
    out_seq = processed_inputs[i + seq_length]

    # We now convert list of characters to integers based on
    # previously and add the values to our lists
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])
# Now we have our input sequences of characters and our output, which is the character 
# that should come after the sequence ends. We now have our training data features and labels, 
# stored as x_data and y_data. Let's save our total number of sequences and check to see how many 
# total input sequences we have:

n_patterns = len(x_data)
print ("Total Patterns:", n_patterns)

# Now we'll go ahead and convert our input sequences into a processed numpy array that our 
# network can use. We'll also need to convert the numpy array values into floats so that the 
# sigmoid activation function our network uses can interpret them and output probabilities from 0 to 1:
# The neural net uses sigmoid activations - lets reshape the data and turn them into floats 
# so the activation function sigmoid can process them 

X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
X = X/float(vocab_len)
# Use - one hot encoding - to convert them to our output labels to train the neural network like this
#
#                                                   10000000
#                                                   01000000
#                                                   00000001
# It makes the neural net easier to train

y = np_utils.to_categorical(y_data)
# Create our Neuralnet that learns - dont worry about what all that means below for now
# just remind yourself its a little blackbox you dont need to understand how it does 
# what it does as long as after training it gives the desired outcome
# as AI gets better you will be able to describe what you want from your
# blackbox and the AI will create it for you

# Currently AI researchers code at such a low level inorder to get the maths to marry up
# with the function the neuralnet needs to learn. If you want to do research yes you need a 
# deeper understanding of the below code, but remember what is said above. 

# You don't have to understand how to make a pair of trainers inorder to wear them and
# understand what they do. Treat neuralnets the same way ...

model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True)) # Think of it as a type of complex switch or gate
model.add(Dropout(0.2)) # Stops the Neuralnet over fitting
model.add(LSTM(256))
 

# Drop layer is stop over fitting you want your neuralnet to generalise well with novel inputs
# the dropout layers helps the neuralnet generalise 
# For more on this type into your search engine neuralnetworks what is overfitting?
# ----------------------------------------------------------------------------------------------
model.add(Dropout(0.1)) 
# ----------------------------------------------------------------------------------------------
model.add(Dense(y.shape[1], activation='softmax'))
# Compile the Neural Net calulate how well its working by its loss & optimize it's learning using adam

model.compile(loss='categorical_crossentropy', optimizer='adam')
# Create a check point that stores how the neural net our litlle brain is wired up - its neuralnet
# pathways

filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]

# Lets start the training of the neural net feed in the input data in batches 'x' 
# also tell the neural net what output we want it to learn with the labels 'y'
# The amount of cycles the neuralnet trains with is set with epochs if its set lower than 20 
# the worse the text and the output turns into gibberish

model.fit(X, y, epochs=1, batch_size=64, callbacks=desired_callbacks)
# Load the saved neuralnet pathways
filename = "model_weights_saved.hdf5"
model.load_weights(filename)
# Now we have loaded the pathways again recompile into the neuralnet
model.compile(loss='categorical_crossentropy', optimizer='adam')
# Convert the numbers back to characters
num_to_char = dict((i, c) for i, c in enumerate(chars))
# Generate some random text to get the neuralnet to produce more
start = numpy.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
print("Random Seed:")
print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")
# Now to finally generate text, we're going to iterate through our chosen number of 
# characters and convert our input (the random seed) into float values.

# We'll ask the model to predict what comes next based off of the random seed, 
# convert the output numbers to characters and then append it to the pattern, 
# which is our list of generated characters plus the initial seed:
for i in range(10):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = num_to_char[index]
    seq_in = [num_to_char[value] for value in pattern]

    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
