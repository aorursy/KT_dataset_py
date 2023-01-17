import pandas as pd

import numpy as np

import os 

import matplotlib.pyplot as plt

import keras

import re

import nltk

from nltk.corpus import stopwords

import string

import json

from time import time

import pickle

from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

from keras.preprocessing import image

from keras.models import Model, load_model

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

from keras.layers import Input, Dense, Dropout, Embedding, LSTM

from keras.layers.merge import add

import cv2

#Code to access file in Directory:

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Function to Read Text Captions

def ReadTextFileData(path):

    with open(path) as file:

        captions=file.read()

    return captions
#Reading Text Data from Flickr8k.token file of Flick8k Datasetb

Caption_Data_Path='/kaggle/input/flickr8k-sau/Flickr_Data/Flickr_TextData/'

captions = ReadTextFileData(Caption_Data_Path+"Flickr8k.token.txt")

# print(captions)
#Splitting Captions based on new line

captions=captions.split('\n')[:-1]

Image_Descriptions={}



for caption in captions:

    

    description=caption.split('\t')

#     print(description)

    caption=description[1]

    image_name=description[0].split('.jpg')

    image_id=image_name[0]

    

    if Image_Descriptions.get(image_id) is None:

        

        Image_Descriptions[image_id]=[]

        Image_Descriptions[image_id].append(caption)

        

    else:

         Image_Descriptions[image_id].append(caption)

            

print(Image_Descriptions)
ImagePath="/kaggle/input/flickr8k-sau/flickr8k-sau/Flickr_Data/Images/"

img=cv2.imread(ImagePath+'3637013_c675de7705.jpg')

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.imshow(img)

plt.show()
# Now is the time for Data Cleaning 

# It is basically preprocessing of Text Data:

# 1.Coverting all text to lower case so that: 'Day' and 'day' are treated as same

# 2.Removing all non-alphaets
def TextCleaning(sentence):

    

    print("Sentence before Cleaning: ",sentence)

    print('\n')

    

    sentence=sentence.lower()

    #Replace anything that is numeric with a blank space

    sentence=re.sub("[^a-z]+"," ",sentence)

    sentence=sentence.split()

    

    #Basically removing words like  a  - To reduce Vocab Size

    #Example 1. a man goes --> man goes

    

    sentence= [word for word in sentence if len(word)>1]

    sentence=" ".join(sentence)  

    

    print("Sentence post Cleaning: ",sentence)

    print('\n')

    

    return sentence
TextCleaning("a man is using 2343 guns his job and his mail id is asas@gmail.com")
#Cleaning all the captions in Image_Descriptions (Dictionary)



for img_id,Description_List in Image_Descriptions.items():

    

    for caption in range(len(Description_List)):

        

        Description_List[caption]=TextCleaning(Description_List[caption])
#Writing the cleaned Text Data to a Text File:



file = open("CleanedCaptionDictionary.txt","w")

file.write(str(Image_Descriptions))

file.close()
# Creating Vocabulary



descriptions=None

with open("CleanedCaptionDictionary.txt",'r') as file:



    descriptions=file.read()





json_acceptable_string=descriptions.replace("'","\"")



#Evaluating the stored captions as a dictionary of Image Id and Captions

descriptions=json.loads(json_acceptable_string)



#Vocab:

Vocabulary=set()

for image_id in descriptions.keys():

    [Vocabulary.update(caption.split()) for caption in descriptions[image_id]]



# Size of Vocabulary 

print("Size of Vocab: %d"%len(Vocabulary))

print('Vocabulary :',Vocabulary)
#Listing total words



list_of_total_words=[]



for key in descriptions.keys():

    

   [list_of_total_words.append(word) for sentence in descriptions[key] for word in sentence.split()]



#Total no of words

print(len(list_of_total_words))
# List of Total Words

print(list_of_total_words)
# Filtering out based on the frequency of occurence 

# Keep the word only if it's frequency passes a specified threshold 



import collections



Threshold_Frequency_of_Word=10

Frequency_per_Word=collections.Counter(list_of_total_words)



#Sorting Based on Frequency

Sorted_FrequencyBased_WordDict= sorted(Frequency_per_Word.items(),reverse=False,key=lambda x:x[1])



#Considering only those words that pass the Threshold_Frequency_of_Word

Sorted_FrequencyBased_WordDict=[x for x in Sorted_FrequencyBased_WordDict if x[1]>Threshold_Frequency_of_Word]



print(Sorted_FrequencyBased_WordDict)



#Updating the Total Word List

total_words=[x[0] for x in Sorted_FrequencyBased_WordDict]



print("Total Unique words previously",len(Frequency_per_Word))



print("Total Unique Words after Processing",len(total_words))



#Now after processing we have only high priority words 
#Preparing the Train and Test Data using the text files from Flickr Dataset:



#Load Training file

Train_File_Data= ReadTextFileData(Caption_Data_Path+"Flickr_8k.trainImages.txt")

#Load Test file

Test_File_Data= ReadTextFileData(Caption_Data_Path+"Flickr_8k.testImages.txt")



Train_File_Data=[row.split('.')[0] for row in Train_File_Data.split('\n')[:-1]]

Test_File_Data=[row.split('.')[0] for row in Test_File_Data.split('\n')[:-1]]



#print(Train_File_Data)

# print(Test_File_Data)
#Making Train Descriptions



training_descriptions={}



for Image_Id in Train_File_Data:

     

    training_descriptions[Image_Id]=[]

    

    for Caption in descriptions[Image_Id]:

    

        Caption_to_append="startseq " + Caption + " endseq"

        training_descriptions[Image_Id].append(Caption_to_append)
training_descriptions['3637013_c675de7705']
#Tranfer Learning

#Extracting Features from both type of Inputs (Image and Text)

# Images -> Features

# Text  ->  Features
#Image Feature Extraction



model=ResNet50(weights="imagenet",input_shape=(224,224,3))

model.summary()
model.layers[-2]
#We are using the ResNet50 Model till the Global Average Pooling Layer

#We use the ResNet50 model as a Image Feature Extractor

#We ignore the fully connected layer in the end which is a categorizer in Resnet50

new_model=Model(model.input,model.layers[-2].output)
#Function to Pre-Process Image to be fed in the Image  Feature Extractor Model

def functiontoPreprocessImage(img):

    

    img=image.load_img(img,target_size=(224,224))

    img=image.img_to_array(img)

    

    #ResNet accepts a 4D Tensor so we do not feed an single Image but Batch of Image thus we expand :

    #(1,224,224,3)

    img=np.expand_dims(img,axis=0)

    

    #Normalizing the Data as per ResNet50: preprocess_input is Resnet's Inbuilt Function

    

    img=preprocess_input(img)

    

    return img
img=functiontoPreprocessImage(ImagePath+"3637013_c675de7705.jpg")

plt.imshow(img[0])

plt.axis('off')

print(img)

plt.show()
#Function that Preprocesses the Image in the Format required by ResNet50 and

#Extracts features from the Image

def encode_image(img):

    

    img= functiontoPreprocessImage(img)

    feature_vector=new_model.predict(img)

    

    feature_vector=feature_vector.reshape((-1,))

    print(feature_vector.shape)

    

    return feature_vector
encoded_image=encode_image(ImagePath+"3637013_c675de7705.jpg")

print(encoded_image)
encode_train={}

start=time()

#Image_Id-->Feature_Vector extracted from Resnet Image



for idx,img_id in enumerate(Train_File_Data):

    

    Input_Image=ImagePath+img_id+".jpg"

    encode_train[img_id]=encode_image(Input_Image)

    

    if idx%100==0:

        print("Encoding in Time Step %d"%idx)

        

end_time=time()

print("Total time Taken",end_time-start)
#Storing Train Data on the Disk

with open("encoded_train_features.pkl","wb") as f:

    pickle.dump(encode_train,f)
encode_test={}

start=time()

#Image_Id-->Feature_Vector extracted from Resnet Image



for idx,img_id in enumerate(Test_File_Data):

    

    Input_Image=ImagePath+img_id+".jpg"

    encode_test[img_id]=encode_image(Input_Image)

    

    if idx%100==0:

        print("Encoding in Time Step %d"%idx)

        

end_time=time()

print("Total time Taken",end_time-start)
#Storing Train Data on the Disk

with open("encoded_test_features.pkl","wb") as f:

    pickle.dump(encode_test,f)
#PreProcessing Captions 



#Making Word to Index and Index to Word Dictionaries

word_to_idx={}

idx_to_word={}



for i,word in enumerate(total_words):

    word_to_idx[word]=i+1

    idx_to_word[i+1]=word
word_to_idx['red']

idx_to_word[1824]
#Getting the total no of words

len(total_words)
word_to_idx['startseq']=1846

idx_to_word[1846]='startseq'



word_to_idx['endseq']=1847

idx_to_word[1847]='endseq'
print(word_to_idx)
print(idx_to_word)
#Finding the maximum Sentence to use in Batches



max_sentence_len=0



for id in training_descriptions.keys():

    

    for caption in training_descriptions[id]:

        #Finding the Maximum Sentence L

        max_sentence_len= max(max_sentence_len,len(caption.split()))

        

print(max_sentence_len)
#Loading the Train and Test Encoded Pickle Files:



f = open("/kaggle/working/encoded_test_features.pkl","rb")

unpickler = pickle.Unpickler(f)

        # if file is not empty scores will be equal

        # to the value unpickled

encode_test = unpickler.load()





f=open("/kaggle/working/encoded_train_features.pkl","rb")

unpickler = pickle.Unpickler(f)

        # if file is not empty scores will be equal

        # to the value unpickled

encode_train = unpickler.load()
#Calculationg the Vocab Size

vocab_size = len(idx_to_word)+1

print(vocab_size)
#Custon Data Loader / Generator



def Data_Generator(training_descriptions,encode_train,word_to_idx,max_sentence_len,batch_size):

    

    Image_Input,Text_Input,Caption_Output=[],[],[]

    

    n=0

    while True:

        

        for key,description_list in training_descriptions.items():

            n+=1

            photo=encode_train[key]

            

            for description in description_list:

                seq=[word_to_idx[word] for word in description.split() if word in word_to_idx]

                for value in range(1,len(seq)):

                    xi=seq[0:value]

                    yi=seq[value]

                    

                    xi=pad_sequences([xi],maxlen=max_sentence_len,value=0,padding='post')[0]

                    yi=to_categorical([yi],num_classes=vocab_size)[0]

                    

                    Image_Input.append(photo)

                    Text_Input.append(xi)

                    Caption_Output.append(yi)

                if n==batch_size:

                    yield [[np.array(Image_Input), np.array(Text_Input)], np.array(Caption_Output)]

                    Image_Input, Text_Input, Caption_Output = [], [], []

                    n=0

#print(encode_train)

print(encode_test)
#Word Embedding:

Base_Path='/kaggle/input/glove6b50dtxt/'

f = open(Base_Path+"glove.6B.50d.txt",encoding='utf8')
#Using Glove Embedding to create word:embedding Dictionary

embedding_index_dict={}

for line in f:

    values=line.split()

    word=values[0]

    word_embedding=np.array(values[1:],dtype='float')

    embedding_index_dict[word]=word_embedding

    

f.close()
embedding_index_dict['is']
#Function to create Embedding Matrix

def get_embedding_matrix():

    

    embedding_dimension=50

    #Creating Vocab_Size X Embedding_Dimension Matrix 

    matrix=np.zeros((vocab_size,embedding_dimension))

    

    for word,idx in word_to_idx.items():

        embedding_vector=embedding_index_dict.get(word)

        if embedding_vector is not None:

                matrix[idx]=embedding_vector

                

    return matrix
embedding_matrix=get_embedding_matrix()

print(embedding_matrix.shape)
#As the Glove does not have any embedding for keywords like startseand endseq

#so they are arrays filled with zeroes.

embedding_matrix[1847]
#Model Architecture

input_image_features=Input(shape=(2048,))

input_img_aftr_Dropout=Dropout(0.3)(input_image_features)

input_img_aftr_Dense_Layer=Dense(256,activation='relu')(input_img_aftr_Dropout)
#Captions as Input

input_captions=Input(shape=(max_sentence_len,))

input_caption_after_embedding=Embedding(input_dim=vocab_size,output_dim=50,mask_zero=True)(input_captions)

input_caption_aftr_Dropout=Dropout(0.3)(input_caption_after_embedding)

input_aftr_LSTM=LSTM(256)(input_caption_aftr_Dropout)
#Decoding

decoder=add([input_img_aftr_Dense_Layer,input_aftr_LSTM])

decoder_aftr_dense_layer=Dense(256,activation='relu')(decoder)

Output=Dense(vocab_size,activation='softmax')(decoder_aftr_dense_layer)
#Combined Model

model=Model(inputs=[input_image_features,input_captions],outputs=Output)
model.summary()
#Pre-Initializing Embedding Layer:

model.layers[2].set_weights([embedding_matrix])

# Making the Layer non Trainable as we want to use the weights of Pre-Trained Embedding Layer

model.layers[2].trainable=False
#Complie

model.compile(loss='categorical_crossentropy',optimizer='adam')
#Model Training

epochs=20

batch_size=3

number_of_pics_per_batch=batch_size

steps=len(training_descriptions)//number_of_pics_per_batch

import os

path = "/kaggle/working/model/"

os.mkdir(path)
def train_model():

    

    for epoch in range(epochs):

        

        generator=Data_Generator(training_descriptions,encode_train,word_to_idx,max_sentence_len,batch_size)

        model.fit_generator(generator,epochs=1,steps_per_epoch=steps,verbose=1)

        model.save(path+str(epoch)+'.h5')

    
train_model()
model=load_model(path+'9.h5')
#Function that takes Image and startseq as input and Predicts the Caption



def prediction_caption(Picture):

    

    input_text="startseq"

    for i in range(max_sentence_len):

        

        sequence=[word_to_idx[word] for word in input_text.split() if word in word_to_idx ]

        sequence=pad_sequences([sequence],maxlen=max_sentence_len,padding='post')

        

        prediction=model.predict([Picture,sequence])

        prediction=prediction.argmax()

        predicted_word=idx_to_word[prediction]

        input_text+=(' '+predicted_word)

        

        if predicted_word=='endseq':

            break

            

    final_caption = input_text.split()[1:-1]

    final_caption=' '.join(final_caption)

    

    return final_caption
#Testing the trained model on some random Test Images

for i in range(10):

    

    image_no=np.random.randint(0,1000)

    all_img_names=list(encode_test.keys())

    image_name=all_img_names[image_no]

    

    photo_2048=encode_test[image_name].reshape((1,2048))

    i = plt.imread(ImagePath+image_name+'.jpg')

    caption = prediction_caption(photo_2048)

    

    plt.title(caption)

    plt.imshow(i)

    plt.axis("off")

    plt.show()