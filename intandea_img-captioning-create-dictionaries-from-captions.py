import numpy as np

import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer

import string

import pickle
fn = "../input/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt"

f = open(fn, 'r')

capts = f.read()
#Group all captions by filename, for references

captions = dict()

i = 0



try:

    for line in capts.split("\n"):

        txt = line.split('\t')

        fn = txt[0].split('#')[0]

        if fn not in captions.keys():

            captions[fn] = [txt[1]]

        else:

            captions[fn].append(txt[1])

        i += 1

except:

    pass
#Now, appending startseq & endseq to training set



#fn_train is just a list of filenames

fn_train = "../input/flickr8k/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt"

f = open(fn_train, 'r')

train_capts = f.read()



train_desc = dict()



try:

    for line in train_capts.split("\n"):

        image_id = line

        image_descs = captions[image_id]

        

        for desc in image_descs:

            ws = desc.split(" ")

            w = [word for word in ws if word.isalpha()]

            desc = "startseq " + " ".join(w) + " endseq"

            if image_id not in train_desc:

                train_desc[image_id] = list()

            train_desc[image_id].append(desc)

except:

    pass
len(train_desc)
#preparing to make word-index and index-word

train_captions = []

for key, desc_list in train_desc.items():

    for i in range(len(desc_list)):

        train_captions.append(desc_list[i])
#Tokenize top 5000 words in Train Captions



tokenizer = Tokenizer(num_words=5000,

                      oov_token="<unk>",

                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

tokenizer.fit_on_texts(train_captions)

word_index = tokenizer.word_index

index_word = tokenizer.index_word

pickle.dump(word_index,open("word_index.pkl","wb"))

pickle.dump(index_word,open("index_word.pkl","wb"))

pickle.dump(train_desc,open("train_caps_startseq_endseq.pkl","wb"))

# pickle.dump(captions,open("all_captions.pkl","wb"))
# #now cleaning captions from unnecessary characters like punctuations and numbers



# table = str.maketrans('', '', string.punctuation)

# for key, desc_list in captions.items():

#     for i in range(len(desc_list)):

#         d = desc_list[i].split(' ')

#         # convert to lower case

#         d = [word.lower() for word in d]

#         # remove punctuation from each token

#         d = [w.translate(table) for w in d]

#         # remove hanging 's' and 'a'

#         d = [word for word in d if len(word)>1]

#         # remove tokens with numbers in them

#         d = [word for word in d if word.isalpha()]

#         # store as string

#         desc_list[i] =  ' '.join(d)