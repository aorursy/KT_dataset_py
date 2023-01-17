# from nltk.corpus import stopwords,words

# from nltk.tokenize import word_tokenize

# from nltk.stem import SnowballStemmer

# from string import punctuation

# stemmer = SnowballStemmer('english')



# def process_sentence(sentence, tokens_to_remove, English_words, max_tokens=None):

#     words = word_tokenize(sentence) # Tokenize

#     if max_tokens is not None and len(words) < max_tokens:

#         return None

#     else:

#         words = [w.lower() for w in words if not w.isdigit()] # Convert to lowercase and also remove digits

#         filter_words = [stemmer.stem(w) for w in words if w not in tokens_to_remove and w in English_words] # remove tokens + check english words + stem

#         return filter_words
# %%time

# import csv

# max_docs = 30000 # test with this number of docs first. If would like to do for all docs, set this value to None

# review_outfile = 'review_text.txt'



# stop_words = set(stopwords.words('english'))

# tokens_to_remove = stop_words.union(set(punctuation))

# English_words = set(words.words())





# doc_count = 0



# with open('../input/amazon-review-testset/test.csv','rt',encoding='utf-8') as rf:

#     with open(review_outfile, 'w') as outputfile:

#         reader = csv.reader(rf, delimiter=',')

#         for row in reader:

#             score = int(row[0])-1 #fit into class for easier work later

#             review = process_sentence(row[1], tokens_to_remove, English_words, 100)     

#             if review is not None:

#                 outputfile.writelines(str(score) + ", " + " ".join(review) + '\n') # write the results

#                 doc_count += 1

#                 if  max_docs and doc_count >= max_docs: # if we do define the max_docs

#                     break

#     outputfile.close()

# rf.close()
# # View the file if needed

# from IPython.display import FileLink

# FileLink('review_text.txt')
# from sklearn.model_selection import train_test_split

# import numpy as np

# from numpy import savetxt



# with open("review_text.txt", "rt") as infile:

#     data = infile.read().split('\n')

#     data = np.array(data)



# data = np.delete(data,-1) 



# train_data,test_data = train_test_split(data,test_size=0.15,random_state = 1) #set the seed to 1 for reproducibility



# #Save data into csv files

# savetxt('train_set.csv', train_data, delimiter=',',fmt='%s')

# savetxt('test_set.csv', test_data, delimiter=',', fmt='%s')
# # View the file if needed

# FileLink('train_set.csv')
# # View the file if needed

# FileLink('test_set.csv')
# # Read data

# import pandas as pd

# from sklearn.feature_extraction.text import CountVectorizer



# train_df = pd.read_csv("train_set.csv", header=None)

# train_df.columns = ["label", "text"]

# class_label = train_df['label'].values.astype(str)

# train_document = train_df['text'].values



# feature_extraction = CountVectorizer()

# trans_docs =  feature_extraction.fit_transform(train_document)
# feature_set = pd.DataFrame(trans_docs.toarray(),columns=feature_extraction.get_feature_names())
# # Import test data

# test_df = pd.read_csv('test_set.csv', header=None)

# test_df.columns = ['label', 'text']

# test_label = test_df['label'].values.astype(str)

# test_document = test_df['text']

# trans_test =  feature_extraction.transform(test_document)
#from sklearn.model_selection import GridSearchCV

#from sklearn.metrics import classification_report

#from sklearn.svm import SVC

#from dask.distributed import Client

#import joblib

#

#client = Client(processes=False) 

#

#

#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1.0,0.1,0.01,0.001],'C': [1,10,100]},

#                    {'kernel': ['linear'], 'C': [1,10,100]},

#                   {'kernel': ['poly'], 'C': [1,10,100],'degree':[2,3,4]},

#                   {'kernel': ['sigmoid'], 'C': [1,10,100]}]

#

## scores = ['precision', 'recall']

#

#result =[]

#

#

#search = GridSearchCV(SVC(), tuned_parameters, scoring='accuracy')

#

#with joblib.parallel_backend('dask'):

#    search.fit(trans_docs, class_label)

#

#print("Best parameters set found on development set: ",search.best_params_)

#print("Grid scores on development set:")

#print()

#means = search.cv_results_['mean_test_score']

#stds = search.cv_results_['std_test_score']

#for mean, std, params in zip(means, stds, search.cv_results_['params']):

#    print("%0.3f (+/-%0.03f) for %r"

#          % (mean, std * 2, params))

#    score = {}

#    score = params.copy()

#    score.update( {'mean accuracy' : mean} )

#    result.append(score)

#result = np.array(result)

    
#from IPython.display import FileLink

#

#print(result)

#savetxt('grid_search.csv', result, delimiter=',', fmt='%s')

#FileLink('grid_search.csv')
# print(str(search.best_params_))
# # Build SVM model

# from sklearn import svm

# from sklearn.metrics import classification_report

# from dask.distributed import Client

# import joblib



# client = Client(processes=False) 



# SVM_classifier = svm.SVC(C=10, kernel = 'rbf', degree = 3, gamma=0.001,decision_function_shape='ovo')



# with joblib.parallel_backend('dask'):

#     SVM_classifier.fit(trans_docs, class_label)



# predicted_class = SVM_classifier.predict(trans_test)



# print(classification_report(predicted_class, test_label))
# import pickle



# # Save to file in the current working directory

# pkl_filename = "SVM_model.pkl"

# with open(pkl_filename, 'wb') as file:

#     pickle.dump(SVM_classifier, file)
# from IPython.display import FileLink

# FileLink('SVM_model.pkl')
import numpy as np
with open('../input/svmresults/grid_search.csv', 'r') as f:

    grid_search_lines = f.readlines()
grid_search_results = np.array([line.replace("\n", "").replace("{","").replace("}","").split(", 'mean accuracy': ") for line in grid_search_lines])
y = [float(a) for a in grid_search_results[:, 1]]
x = grid_search_results[:,0]
import matplotlib.pyplot as plt
with plt.style.context('ggplot'):

    plt.figure(figsize=(20, 10))

    plt.plot(x, y, '-o', color='steelblue')

    idx = np.argmax(y)

    plt.plot(x[idx], y[idx], 'P', color='red', ms=10)

    plt.xlabel("SVM parameters")

    plt.ylabel("Mean accuracy")

    plt.xticks(rotation=90)

    plt.show()
# Read data

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer



train_df = pd.read_csv("../input/svmresults/train_set.csv", header=None)

train_df.columns = ["label", "text"]

class_label = train_df['label'].values.astype(str)

train_document = train_df['text'].values



feature_extraction = CountVectorizer()

trans_docs =  feature_extraction.fit_transform(train_document)
feature_set = pd.DataFrame(trans_docs.toarray(),columns=feature_extraction.get_feature_names())
# Import test data

test_df = pd.read_csv('../input/svmresults/test_set.csv', header=None)

test_df.columns = ['label', 'text']

test_label = test_df['label'].values.astype(str)

test_document = test_df['text']

trans_test =  feature_extraction.transform(test_document)
import pickle



# Save to file in the current working directory

pkl_filename = "../input/svmresults/SVM_model.pkl"

with open(pkl_filename, 'rb') as file:

    SVM_classifier = pickle.load(file)
predicted_class = SVM_classifier.predict(trans_test)
set(predicted_class)
from sklearn.metrics import classification_report
print(classification_report(predicted_class, test_label))
def confusion_type_for_class(cls, true_label, predicted_label):

    result={}

    result["true_positive"] = [true_label[i] == cls and predicted_label[i] == cls for i in range(len(true_label))]

    result["false_positive"] = [true_label[i] != cls and predicted_label[i] == cls for i in range(len(true_label))]

    result["false_negative"] = [true_label[i] == cls and predicted_label[i] != cls for i in range(len(true_label))]

    result["true_negative"] = [true_label[i] != cls and predicted_label[i] != cls for i in range(len(true_label))]

    return result
def visualize_confusion_results(cls, test_df, true_label, predicted_label):

    confusion_result = confusion_type_for_class(cls, true_label, predicted_label)

    for cf in confusion_result.keys():

        texts = test_df[confusion_result[cf]]['text'].values

        if len(texts) == 0:

            import pdb

            pdb.set_trace()

        if len(texts) > 0:

            show_wordcloud(" ".join(texts), f"Test set class {cls}, type {cf}")
import wordcloud

import matplotlib.pyplot as plt

def show_wordcloud(text, title=None):

    # Create and generate a word cloud image:

    wc = wordcloud.WordCloud(background_color='white').generate(text)

    # Display the generated image:

    plt.figure(figsize=(10, 10))

    plt.imshow(wc, interpolation='bilinear')

    plt.axis("off")

    if title is not None:

        plt.title(title)

    plt.show()
for cls in set(class_label):

    visualize_confusion_results(cls, test_df, test_label, predicted_class)