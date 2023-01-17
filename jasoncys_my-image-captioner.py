#IMPORTS 



import numpy as np

import pandas as pd



import keras

from keras.applications import vgg16

from keras.preprocessing import image, text, sequence



from keras.models import Model, Sequential

from keras.layers import Input, Dense, GRU, Embedding

from keras.callbacks import EarlyStopping, ModelCheckpoint

import keras.backend as K



from sklearn.model_selection import train_test_split



from tqdm.notebook import tqdm



import matplotlib.pyplot as plt



import cv2
results_df = pd.read_csv('/kaggle/input/flickr-image-dataset/flickr30k_images/results.csv', delimiter='|')

results_df.head()#Extract list of all of the image names

image_names = results_df['image_name'][::5].values
#Extract list of all of the image names

image_names = results_df['image_name'][::5].values
#load the vgg16 model and create the encoder which encludes all bar the final dense and softmax layers

#resulting in a 4096 length vector representation of each image

vgg = vgg16.VGG16(weights='imagenet', include_top=True)

encoder = Model(vgg.input, vgg.layers[-2].output)



#encoder.summary()
#preload and process each image with the encoder model. so that we don't have ot repeat the process for each epoch

#in this implementation we will not be training the encoder so makes sense to precompute vectors

root_path = '/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images/'



def load_process_image(root_path, image_name):

    """

    load and process an image ready to be fed into the pre_build vgg16 encoder.

    """

    img = image.load_img(root_path + image_name)

    img = image.img_to_array(img)

    img = cv2.resize(img, (224,224))

    img = vgg16.preprocess_input(img)

    return img



def vectorize_images(root_path, image_names):

    image_vectors = []

    for image_name in tqdm(image_names):

        img = load_process_image(root_path, image_name)

        image_vectors.append(encoder.predict(np.expand_dims(img, axis=0)))



    image_vectors = np.array(image_vectors)

    image_vectors = image_vectors.squeeze()    

    return image_vectors

    

image_vectors = vectorize_images(root_path, image_names)                         
#process all of the text



VOCAB_SIZE = 10000



tokenizer = text.Tokenizer(num_words=VOCAB_SIZE)



sequenced_comments = ['ssss ' + str(t) + ' eeee' for t in results_df[' comment']]  # add start and end markers to the sentences

tokenizer.fit_on_texts(sequenced_comments)

sequenced_comments = tokenizer.texts_to_sequences(sequenced_comments)

sequenced_comments = np.array(sequenced_comments)



# reshape into an array of the same length of images but with 5 comments per image. 

sequenced_comments = sequenced_comments.reshape(-1,5)
#sanity check, make sure the sequences have been encoded and decoded as expected, and reshaped into a matrix which is easy or us to use

comment_index = 124

print(results_df[' comment'][comment_index])

' '.join([tokenizer.index_word[i] for i in sequenced_comments[divmod(comment_index, 5)] if i != 0])
K.clear_session()

np.random.seed(20)



HIDDEN_LAYER_SIZE = 512   # experiment with the number of nodes in the hidden layer



text_input = Input(shape=(None,))

text_embedding = Embedding(VOCAB_SIZE, 64)(text_input)



image_vector_input = Input(shape=(4096,))

inital_state = Dense(HIDDEN_LAYER_SIZE, activation='tanh')(image_vector_input)



recurrent_layer_1 = GRU(HIDDEN_LAYER_SIZE, return_sequences=True)(text_embedding, initial_state=inital_state)

recurrent_layer_2 = GRU(HIDDEN_LAYER_SIZE, return_sequences=True)(recurrent_layer_1, initial_state=inital_state)

recurrent_layer_3 = GRU(HIDDEN_LAYER_SIZE, return_sequences=True)(recurrent_layer_2, initial_state=inital_state)



text_output = Dense(VOCAB_SIZE, activation='softmax')(recurrent_layer_3)



decoder = Model([text_input, image_vector_input], text_output)



decoder.summary()
decoder.compile(optimizer='RMSprop', loss='sparse_categorical_crossentropy')
class Text_Generator(keras.utils.Sequence):

    

    def __init__(self, image_vectors, sequenced_comments, batch_size=128, shuffle=True):

        self.image_vectors = image_vectors

        self.sequenced_comments = sequenced_comments

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.indexes = np.arange(image_vectors.shape[0])

        self.on_epoch_end()

        

        

    def __len__(self):

        """

        total number of batches per epoch

        """

        return self.image_vectors.shape[0]//self.batch_size

    

    

    def __getitem__(self, index):

        """

        generate a batch of inputs and outputs

        """

        batch_indexes = self.indexes[(index*self.batch_size): (index+1)*self.batch_size]

        

        batch_comments  = [self.sequenced_comments[i, np.random.randint(5)] for i in batch_indexes]

        batch_comments = sequence.pad_sequences(batch_comments, padding='post', truncating='post')

        batch_comments = np.array(batch_comments)

        

        batch_image_vectors = self.image_vectors[batch_indexes]

        

        

        text_input = batch_comments[:,:-1]

        text_output = batch_comments[:,1:].reshape(self.batch_size,-1,1)

        

        X = [text_input, batch_image_vectors]

        y = text_output

    

        return X, y



    

    def on_epoch_end(self):

        'Updates indexes after each epoch'

        if self.shuffle == True:

            np.random.shuffle(self.indexes)    

        

    
#split the image_names, image_vectors, sequenced_comments and original comments for training and validation

image_names_train, image_names_val, image_vectors_train, image_vectors_val, sequenced_comments_train, sequenced_comments_val, original_comments_train, original_comments_val = train_test_split(image_names, image_vectors, sequenced_comments, results_df[' comment'].values.reshape(-1,5), test_size=0.05, random_state=1)
#generators

train_generator = Text_Generator(image_vectors_train, sequenced_comments_train)

val_generator = Text_Generator(image_vectors_val, sequenced_comments_val)
# early_stopping = EarlyStopping(monitor='val_loss', patience=20)

# model_checkpoint = ModelCheckpoint('model_weights.h5',monitor='val_loss', save_best_only=True)

# call_backs_list = [early_stopping, model_checkpoint]

# fit the model 



#decided not to use the call backs as the validation data is too small. the validation data is purely so I have some unseen data to demonstate on.

# and I do not think the model is overfitting after only 20 epochs. 



decoder.fit_generator(generator=train_generator, epochs=20, validation_data=val_generator)
#decoder.load_weights('model_weights.h5')
def generate_caption(image_vector):

    """

    Generate an english sentence given an image_vector

    """

    word = 'ssss'

    token = tokenizer.word_index[word]

    sentence = [word]

    sequence = [token]

    

    while word != 'eeee':

        pred = decoder.predict([[sequence], [image_vector]]).reshape(-1,VOCAB_SIZE)[-1]

        token = np.argmax(pred)

        word = tokenizer.index_word[token]

        sentence.append(word)

        sequence.append(token)

        

    print('generated: ', ' '.join(sentence[1:-1]))



def get_original_captions(original_captions):

    for i in range(5):

        print('original: ', original_captions[i])

        

def get_image(image_name):

    img = plt.imread(f"/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images/{image_name}")

    plt.imshow(img)

    plt.show()

    
for i in range(50):

    generate_caption(image_vectors_train[i])

    print()

    get_original_captions(original_comments_train[i])

    get_image(image_names_train[i])

 
for i in range(50):

    generate_caption(image_vectors_val[i])

    print()

    get_original_captions(original_comments_val[i])

    get_image(image_names_val[i])

 