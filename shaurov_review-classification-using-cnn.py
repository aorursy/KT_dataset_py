from nltk.corpus import stopwords

import string, re

from collections import Counter

import wordcloud



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



import os



review_dataset_path= "/kaggle/input/movie-review/movie_reviews/movie_reviews"

print(os.listdir(review_dataset_path))
#Positive and negative reviews folder paths 

pos_review_folder_path=review_dataset_path+"/"+"pos"

neg_review_folder_path=review_dataset_path+"/"+"neg"



#Positive and negative file names

pos_review_file_names=os.listdir(pos_review_folder_path)

neg_review_file_names=os.listdir(neg_review_folder_path)
def load_text_from_textfile(path):

    file=open(path,"r")

    review=file.read()

    file.close()

    return review
def get_data_target(folder_path, file_names, review_type):

    data=list()

    target =list()

    for file_name in file_names:

        full_path = folder_path + "/" + file_name

        review =load_text_from_textfile(path=full_path)

        data.append(review)

        target.append(review_type)

    return data, target
pos_data, pos_target=get_data_target(folder_path=pos_review_folder_path,

                                     file_names=pos_review_file_names,

                                     review_type="positive")

print("Positive data ve target builded...")

print("positive data length:",len(pos_data))

print("positive target length:",len(pos_target))
neg_data, neg_target = get_data_target(folder_path = neg_review_folder_path,

                                      file_names= neg_review_file_names,

                                      review_type="negative")

print("Negative data ve target builded..")

print("negative data length :",len(neg_data))

print("negative target length :",len(neg_target))
data = pos_data + neg_data

target_ = pos_target + neg_target

print("Positive and Negative sets concatenated")

print("data length :",len(data))

print("target length :",len(target_))
le = LabelEncoder()

le.fit(target_)

target = le.transform(target_)

print("Target labels transformed to number...")
print(le.inverse_transform([0,0,0,1,1,1]))
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, stratify=target, random_state=24)

print("Dataset splited into train and test parts...")

print("train data length  :",len(X_train))

print("train target length:",len(y_train))

print()

print("test data length  :",len(X_test))

print("test target length:",len(y_test))
import seaborn as sns

fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(8,4),sharey=True)

axarr[0].set_title("Number of samples in train")

sns.countplot(x=y_train, ax=axarr[0])

axarr[1].set_title("Number of samples in test")

sns.countplot(x=y_test, ax=axarr[1])

plt.show()
class MakeString:

    def process(self, text):

        return str(text)
class ReplaceBy:

    def __init__(self, replace_by):

        #replace_by is a tuple contains pairs of replace and by characters.

        self.replace_by = replace_by

    def process(self, text):

        for replace, by in replace_by:

            text = text.replace(replace, by)

        return text
class LowerText:

    def process(self, text):

        return text.lower()
class ReduceTextLength:

    def __init__(self, limited_text_length):

        self.limited_text_length = limited_text_length

    def process(self, text):

        return text[:self.limited_text_length]
class VectorizeText:

    def __init__(self):

        pass

    def process(self, text):

        return text.split()
class FilterPunctuation:

    def __init__(self):

        print("Punctuation Filter created...")

    def process(self, words_vector):

        reg_exp_filter_rule=re.compile("[%s]"%re.escape(string.punctuation))

        words_vector=[reg_exp_filter_rule.sub("", word) for word in words_vector]

        return words_vector
class FilterNonalpha:

    def __init__(self):

        print("Nonalpha Filter created...")

    def process(self, words_vector):

        words_vector=[word for word in words_vector if word.isalpha()]

        return words_vector
class FilterStopWord:

    def __init__(self, language):

        self.language=language

        print("Stopwords Filter created...")

    def process(self, words_vector):

        stop_words=set(stopwords.words(self.language))

        words_vector=[word for word in words_vector if not word in stop_words]

        return words_vector
class FilterShortWord:

    def __init__(self, min_length):

        self.min_length=min_length

        print("Short Words Filter created...")

    def process(self, words_vector):

        words_vector=[word for word in words_vector if len(word)>=self.min_length]

        return words_vector  
class TextProcessor:

    def __init__(self, processor_list):

        self.processor_list = processor_list

    def process(self, text):

        for processor in self.processor_list:

            text = processor.process(text)

        return text
text_len = np.vectorize(len)

text_lengths = text_len(X_train)

text_lengths
mean_review_length =int(text_lengths.mean())

print("Mean length of reviews   :",mean_review_length)    

print("Minimum length of reviews:",text_lengths.min())

print("Maximum length of reviews:",text_lengths.max())
sns.distplot(a=text_lengths)
makeString = MakeString()



replace_by = [("."," "), ("?"," "), (","," "), ("!"," "),(":"," "),(";"," ")]

replaceBy =ReplaceBy(replace_by=replace_by)



lowerText = LowerText()



FACTOR=8

reduceTextLength = ReduceTextLength(limited_text_length=mean_review_length*FACTOR)



vectorizeText = VectorizeText()

filterPunctuation = FilterPunctuation()

filterNonalpha = FilterNonalpha()

filterStopWord = FilterStopWord(language = "english")



min_length = 2

filterShortWord = FilterShortWord(min_length=min_length)

processor_list_1 = [makeString,

                      replaceBy,

                      lowerText,

                      reduceTextLength,

                      vectorizeText,

                      filterPunctuation,

                      filterNonalpha,

                      filterStopWord,

                      filterShortWord]
textProcessor1 = TextProcessor(processor_list=processor_list_1)
random_number=np.random.randint(0,len(X_train))

print("Original Review:\n",X_train[random_number][:500])

print("="*100)

print("Processed Review:\n",textProcessor1.process(text=X_train[random_number][:500]))
class VocabularyHelper:

    def __init__(self, textProcessor):

        self.textProcessor=textProcessor

        self.vocabulary=Counter()

    def update(self, text):

        words_vector=self.textProcessor.process(text=text)

        #print("words_vector", words_vector)

        #print("\nself.vocabulary", self.vocabulary)

        self.vocabulary.update(words_vector)

    def get_vocabulary(self):

        return self.vocabulary
vocabularyHelper=VocabularyHelper(textProcessor=textProcessor1)

print("VocabularyHelper created...")



for text in X_train:

    vocabularyHelper.update(text)

vocabulary = vocabularyHelper.get_vocabulary()

print("Vocabulary filled...")
print("Length of vocabulary:",len(vocabulary))

n=10

print("{} most frequented words in vocabulary:{}".format(n, vocabulary.most_common(n)))
print("{} least frequented words in vocabulary:{}".format(n, vocabulary.most_common()[:-n-1:-1]))
vocabulary_list = " ".join([key for key, freq in vocabulary.most_common()])

plt.figure(figsize=(15, 35))

wordcloud_image = wordcloud.WordCloud(width = 1000, height = 1000, 

                background_color ='white', 

                #stopwords = stopwords, 

                min_font_size = 10).generate(vocabulary_list)



plt.xticks([])

plt.yticks([])

plt.imshow(wordcloud_image)
min_occurence=2

vocabulary = Counter({key:value for key, value in vocabulary.items() if value>min_occurence})
print("{} least frequented words in vocabulary:{}".format(n, vocabulary.most_common()[:-n-1:-1]))
class FilterNotInVocabulary:

    def __init__(self, vocabulary):

        self.vocabulary = vocabulary

    def process(self, words_vector):

        words_vector = [word for word in words_vector if word in self.vocabulary]

        return words_vector
class JoinWithSpace:

    def __init__(self):

        pass

    def process(self, words_vector):

        return " ".join(words_vector)
filterNotInVocabulary = FilterNotInVocabulary(vocabulary = vocabulary)

joinWithSpace = JoinWithSpace()

processor_list_2 = [makeString,

                    replaceBy,

                    lowerText,

                    reduceTextLength,

                    vectorizeText,

                    filterPunctuation,

                    filterNonalpha,

                    filterStopWord,

                    filterShortWord,

                    filterNotInVocabulary,

                    joinWithSpace

                   ]

textProcessor2=TextProcessor(processor_list = processor_list_2)
review = X_train[np.random.randint(0,len(X_train))]

print("Original Text:\n",review[:500])

processed_review = textProcessor2.process(review[:500])

print("="*100)

print("Processed Text:\n",processed_review)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, stratify=target, random_state=24)

print("Dataset splited into train and test parts...")

print("train data length  :",len(X_train))

print("train target length:",len(y_train))

print()

print("test data length  :",len(X_test))

print("test target length:",len(y_test))
def  process_text(texts, textProcessor):

    processed_texts=list()

    for text in texts:

        processed_text = textProcessor.process(text)

        processed_texts.append(processed_text)

    return processed_texts
X_train_processed = process_text(texts=X_train, textProcessor=textProcessor2)

print("X_train processed...")

X_test_processed = process_text(texts=X_test, textProcessor=textProcessor2)

print("X_test processed...")
from keras.preprocessing.text import Tokenizer

def create_and_train_tokenizer(texts):

    tokenizer=Tokenizer()

    tokenizer.fit_on_texts(texts)

    return tokenizer
from keras.preprocessing.sequence import pad_sequences

def encode_reviews(tokenizer, max_length, docs):

    encoded=tokenizer.texts_to_sequences(docs)

    

    padded=pad_sequences(encoded, maxlen=max_length, padding="post")

    

    return padded
tokenizer=create_and_train_tokenizer(texts = X_train)

vocab_size=len(tokenizer.word_index) + 1

print("Vocabulary size:", vocab_size)
max_length=max([len(row.split()) for row in X_train])

print("Maximum length:",max_length)

#for row in X_train[:3]:

#    print('1>',row)
X_train_encoded=encode_reviews(tokenizer, max_length, X_train_processed)

print("X_train encoded...")
X_test_encoded = encode_reviews(tokenizer, max_length, X_test_processed)

print("X_test encoded...")
from keras import layers, models

def create_embedding_model(vocab_size, max_length):

    model=models.Sequential()

    model.add(layers.Embedding(vocab_size, 100, input_length=max_length))

    model.add(layers.Conv1D(32, 8, activation="relu"))

    model.add(layers.MaxPooling1D(2))

    model.add(layers.Flatten())

    model.add(layers.Dense(10, activation="relu"))

    model.add(layers.Dense(1,  activation="sigmoid"))   

    return model
embedding_model = create_embedding_model(vocab_size=vocab_size, max_length=max_length)

embedding_model.summary()
embedding_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
from keras.callbacks import EarlyStopping

earlyStopping = EarlyStopping(monitor="val_accuracy", patience=1)

modelHistory = embedding_model.fit(X_train_encoded, 

                                   y_train, 

                                   validation_data=(X_test_encoded, y_test),

                                   epochs=10, 

                                   callbacks=[earlyStopping])

print("Model trained...")
_, acc = embedding_model.evaluate(X_train_encoded, y_train, verbose=0)

print("Train accuracy:{:.2f}".format(acc*100))
_,acc= embedding_model.evaluate(X_test_encoded, y_test, verbose=0)

print("Test accuracy:{:.2f}".format(acc*100))
embedding_model.save("embedding_model.h5")

print(os.listdir("/kaggle/working"))
loaded_model=models.load_model("/kaggle/working/embedding_model.h5")

print("Saved model loaded...")