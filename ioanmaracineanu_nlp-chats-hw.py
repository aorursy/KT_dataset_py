import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm, feature_extraction, linear_model, preprocessing
from sklearn.metrics import accuracy_score
# train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv", encoding='latin-1')
# test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv", encoding='latin-1')

train_df = pd.read_csv("/kaggle/input/nlp-chats-v2/train.csv", converters={'text' : str, 'target': int, 'has_ref': int})
test_df = pd.read_csv("/kaggle/input/nlp-chats-v2/test.csv", converters={'text' : str, 'target': int, 'has_ref': int})

train_df
def preproc(Corpus):
    # print(Corpus)
    print('### preproc, size is', Corpus.size)
#     print(Corpus['text'])

    # Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    Corpus['text'] = [entry.lower() for entry in Corpus['text']]

    # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
    Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]

    # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.

    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV


    for index,entry in enumerate(Corpus['text']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        
        assert Corpus.size > index
        
        # The final processed set of words for each iteration will be stored in 'text_final'
        Corpus.loc[index,'text_final'] = str(Final_words)
        
    
    print('### done, size is', Corpus.size)
    print(Corpus.head())

preproc(train_df)
preproc(test_df)


# Step - 2: Split the model into Train and Test Data set
Train_X, Train_Y = train_df['text_final'], train_df['target']
Test_X, Test_Y = test_df['text_final'], test_df['target']

print(train_df.size)
print(test_df.size)

# Step - 3: Label encode the target variable  - This is done to transform Categorical data of string type in the data set into numerical values
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

# Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the corpus
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(pd.concat([Train_X, Test_X]))

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


# Step - 5: Now we can run different algorithms to classify out data check for accuracy

# Classifier - Algorithm - Naive Bayes
# fit the training dataset on the classifier
# Naive = naive_bayes.MultinomialNB()
# clf = linear_model.RidgeClassifier()
# clf.fit(Train_X_Tfidf,Train_Y)

# clf = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
clf = linear_model.RidgeClassifier()
clf.fit(Train_X_Tfidf,Train_Y)

'''
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
'''
# predict the labels on validation dataset
predictions = clf.predict(Test_X_Tfidf)


# Use accuracy_score function to get the accuracy
print("Accuracy Score -> ",accuracy_score(predictions, Test_Y)*100)
# sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
# sample_submission["target"] = predictions
# sample_submission.to_csv("submission.csv", index=False)

train_arr = Train_X_Tfidf.toarray()
N = train_arr.shape[0]
# train_arr_2 = np.concatenate((train_arr, np.ones(N, 1)), axis=1)
refs = train_df['has_ref'].to_numpy().reshape(N, 1)
train_arr_2 = np.hstack((train_arr, refs))
# train_arr_2

clf.fit(train_arr_2,Train_Y)
test_arr = Test_X_Tfidf.toarray()
N = test_arr.shape[0]
# train_arr_2 = np.concatenate((train_arr, np.ones(N, 1)), axis=1)
refs = test_df['has_ref'].to_numpy().reshape(N, 1)
test_arr_2 = np.hstack((test_arr, refs))
# train_arr_2
test_df['predictions'] = predictions
print("Accuracy Score -> ",accuracy_score(test_df['predictions'], test_df['target'])*100)
# print(test_df[test_df['predictions'] >= 1])

test_df.to_csv("nlp_chats_result_v2.csv", index=False)


print(test_df[test_df["target"] == 2])
from sklearn.metrics import precision_score, recall_score, f1_score

def reports(train_predictions, val_predictions, train_targets, val_targets):
        train_precision = precision_score(train_targets, train_predictions, average='macro')
        train_recall = recall_score(Train_Y, train_predictions, average='macro')
        train_f1 = f1_score(train_targets, train_predictions, average='macro')
        
        val_precision = precision_score(val_targets, val_predictions, average='macro')
        val_recall = recall_score(val_targets, val_predictions, average='macro')
        val_f1 = f1_score(val_targets, val_predictions, average='macro')
       
        
        print('Training Precision: {:.6} - Training Recall: {:.6} - Training F1: {:.6}'.format(train_precision, train_recall, train_f1))
        print('Validation Precision: {:.6} - Validation Recall: {:.6} - Validation F1: {:.6}'.format(val_precision, val_recall, val_f1))  


# predict the labels on validation dataset
train_predictions = clf.predict(train_arr_2)

# Use accuracy_score function to get the accuracy
print("Accuracy Score (train) -> ",accuracy_score(train_predictions, Train_Y)*100)

# predict the labels on validation dataset
test_predictions = clf.predict(test_arr_2)

# Use accuracy_score function to get the accuracy
print("Accuracy Score (test) -> ",accuracy_score(test_predictions, Test_Y)*100)

reports(train_predictions, test_predictions, Train_Y, Test_Y)

conv_true_positives = np.logical_and(test_predictions == 1, Test_Y == 1).sum()
conv_false_positives = np.logical_and(test_predictions == 1, Test_Y == 2).sum()

conv_true_negatives = np.logical_and(test_predictions == 2, Test_Y == 2).sum()
conv_false_negatives = np.logical_and(test_predictions == 2, Test_Y == 1).sum()

print(conv_true_positives, conv_false_positives, conv_true_negatives, conv_false_negatives)

print("c precision:", conv_true_positives/(conv_true_positives + conv_false_positives))
print("c recall:", conv_true_positives/(conv_true_positives + conv_false_negatives))

print("d precision:", conv_true_negatives/(conv_true_negatives + conv_false_negatives))
print("d recall:", conv_true_negatives/(conv_true_negatives + conv_false_positives))
print(np.sum(test_predictions == 1))
print(np.sum(test_predictions == 2))
print(np.sum(test_predictions == 0))
