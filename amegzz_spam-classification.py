# i build this model  on google colab so i will comment command which i use on it
# dawnload dataset
#if you want to download
#!wget https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
#!unzip smsspamcollection.zip
import os
print(os.listdir("../input/smsspamcollection"))
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
from wordcloud import WordCloud
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score

# Pretty display for notebooks
%matplotlib inline
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

#Load dataset
pd.set_option('display.max_colwidth', 100)

data = pd.read_csv("../input/smsspamcollection/SMSSpamCollection", sep='\t',skipinitialspace=True, header=None)
data.columns = ['label', 'body_text']

display(data.head())
display(data.info())

# Total number of records
n_records = data.shape[0]


# Number of records where SMS  are ham
n_ham = data[data.label=="ham"].label.count()

# Number of records where SMS are ham
n_spam =data[data.label=="spam"].label.count()

# Percentage of SMS which are ham
spam_percent = (n_spam / n_records) * 100

# Print the results
print("Total number of records: {}".format(n_records))
print("SMS which are ham: {}".format(n_ham))
print("SMS which are spam: {}".format(n_spam))
print("Percentage of SMS which are spam : {:.3f}%".format(spam_percent))
# Remove punctuation
import string

def remove_punct(text):
  text_nopunct = "".join([char for char in text if char not in string.punctuation]) 
  return text_nopunct.lower()

data['body_text_clean'] = data['body_text'].apply(lambda x: remove_punct(x))
display(data.head())

def wc_message(df, type):
    """
     Visualization repeated words in the messages (spam or ham)!

     inputs:
      - data fram
      - label: type of messages (spam or ham) 
    """
    wc = WordCloud(width = 512, height = 512)
    try:
        if type == 'spam':
            spam_words = ''.join(list(data[data['label'] == 'spam']['body_text_clean'] ))
            spam_wc = wc.generate(spam_words)
            pl.figure(figsize= (10,8), facecolor= 'k')
            pl.imshow(spam_wc)

        elif type == 'ham':
            ham_words = ''.join(list(data[data['label'] == 'ham']['body_text_clean'] ))
            ham_wc = wc.generate(ham_words)
            pl.figure(figsize= (10,8), facecolor= 'k')
            pl.imshow(ham_wc)
        else:
          print("please input right parmeters type wc_message.__doc__ and follow the inputs")



        pl.axis('off')
        pl.tight_layout(pad = 0)
        pl.show()
    except Exception as e:
        print(e)

wc_message(data, 'spam')
wc_message(data, 'ham')
#!pip install -U spacy
import spacy

!python -m spacy info en 
#!python3 -m spacy download en
nlp = spacy.load('en', tagger=False, parser=False, matcher=False)
def toknize(text):
  tokens = nlp(text)    
  return tokens.text.split(" ")
  

data['tokens'] = data['body_text_clean'].apply(lambda x: toknize(x))
display(data.head())
# Remove English stop words 
from spacy.lang.en.stop_words import STOP_WORDS

print(STOP_WORDS) 
# you can add new stop words with this command 
#STOP_WORDS.add("your_additional_stop_word_here")
def remove_stopwords(tokenized_list):
  text = [word for word in tokenized_list if word not in STOP_WORDS]
  return text
data['text_nostop'] = data['tokens'].apply(lambda x: remove_stopwords(x))

print(data.head())
from spacy import displacy

doc = nlp(data.body_text_clean[0])
displacy.render(doc, style='dep', jupyter=True)

def lemmatizing(tokenized_text):
    doc = str(tokenized_text).replace('[','').replace(']','')
    doc = nlp(doc)
    text = [token.lemma_ for token in doc]
    text = [w for w in text if w.isalpha() or w.isdigit()]

    return text
data['text_lemmatized'] = data['text_nostop'].apply(lambda x: lemmatizing(x))

print(data.head())
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))
print(data.head())
bins = np.linspace(0, 200, 40)

pl.hist(data['body_len'], bins)
pl.title("Body Length Distribution")
pl.show()
bins = np.linspace(0, 50, 40)

pl.hist(data['punct%'], bins)
pl.title("Punctuation % Distribution")
pl.show()
display(data.head(2))
# preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['body_len', 'punct%']

data[numerical] = scaler.fit_transform(data[numerical])

# Show an example of a record with scaling applied
display(data.head(n = 5))
# save data after normlize it in pkl file to use it latter
from sklearn.externals import joblib

joblib.dump(scaler, 'scalr.pkl')

# convert label to number before aplly ML algorithms
label_map = {'spam': 1, 'ham': 0}
data['label'] = data['label'].map(label_map)
display(data.head(3))
features_final = data[['text_lemmatized', 'body_len', 'punct%']]
label = data['label']
# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    label, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))
# conver 'text_lemmatized' col from list to string to apply TfidfVectorizer on it
def tostring(text):
  doc = str(text).replace('[','').replace(']','')
  return doc

  
from sklearn.feature_extraction.text import TfidfVectorizer


tfidf_vect = TfidfVectorizer(analyzer = tostring)
tfidf_vect_fit = tfidf_vect.fit(X_train['text_lemmatized'])

tfidf_train = tfidf_vect_fit.transform(X_train['text_lemmatized'])
tfidf_test = tfidf_vect_fit.transform(X_test['text_lemmatized'])

X_train_vect = pd.concat([X_train[['body_len', 'punct%']].reset_index(drop=True), 
           pd.DataFrame(tfidf_train.toarray())], axis=1)
X_test_vect = pd.concat([X_test[['body_len', 'punct%']].reset_index(drop=True), 
           pd.DataFrame(tfidf_test.toarray())], axis=1)

display(X_train_vect.head())
from sklearn.externals import joblib

joblib.dump(tfidf_vect, 'vectroizer.pkl')
'''
TP = np.sum(label) # Counting the ones as this is the naive case. 
FP = label.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case
'''

# Calculate ones in income (True postive) and zeroes 
TP = np.sum(label)
FP = label.count() - TP 
TN = 0
FN = 0

# TODO: Calculate accuracy, precision and recall
accuracy = TP /(TP + FP)
recall = TP / (TP + FN)
precision = TP / (TP + FP)

# TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
beta = 0.5
fscore = (1 + beta**2)*((precision * recall) / ((beta**2 * precision) + recall))

# Print the results 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))
# Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score , accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: label training set
       - X_test: features testing set
       - y_test: label testing set
    '''
    
    results = {}
    
    # Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[: sample_size], y_train[: sample_size])
    end = time() # Get end time
    
    # Calculate the training time
    results['train_time'] = end - start
        
    # Get the predictions on the test set(X_test),
    # then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # Calculate the total prediction time
    results['pred_time'] = end - start
            
    # Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta = 0.5)
        
    # Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta = 0.5)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results

#visualization

def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """
  
    # Create figure
    fig, ax = pl.subplots(2, 4, figsize = (11,7))

    # Constants
    bar_width = 0.3
    colors = ['#A00000','#00A0A0','#00A000']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                
                # Creative plot code
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Set additional plots invisibles
    ax[0, 3].set_visible(False)
    ax[1, 3].axis('off')

    # Create legend
    for i, learner in enumerate(results.keys()):
        pl.bar(0, 0, color=colors[i], label=learner)
    pl.legend()
    
    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    pl.tight_layout()
    pl.show()
# Import the  supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from time import time
import matplotlib.pyplot as pl

# Initialize the three models
clf_A = GaussianNB()
clf_B = AdaBoostClassifier(random_state = 1)
clf_C = SVC(random_state = 1)

# Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_100 = len(y_train)
samples_10 = len(y_train)//10
samples_1 = len(y_train)//100

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train_vect, y_train, X_test_vect, y_test)

# Run metrics visualization for the three supervised learning models chosen
evaluate(results, accuracy, fscore)
# Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer

# Initialize the classifier
clf = AdaBoostClassifier(random_state=1)

parameters = {'n_estimators' : [50, 100, 150, 250, 500], 'learning_rate': [0.01, 0.1, 1, 1.5, 2]}

# Make an fbeta_score scoring object using make_scorer()
scorer =make_scorer(fbeta_score, beta= 0.5)

# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train_vect, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train_vect, y_train)).predict(X_test_vect)
best_predictions = best_clf.predict(X_test_vect)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
display(grid_fit.best_estimator_)
#Train the model on the training set using .fit(X_train_vect, y_train)
model = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=0.1, n_estimators=100, random_state=1)
start = time()
model.fit(X_train_vect, y_train)
end = time()
fit_time = (end - start)
start = time()
y_pred = model.predict(X_test_vect)
end = time()
pred_time = (end - start)

recision, recall, fscore, train_support = score(y_test, y_pred)
print('Fit time: {} / Predict time: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(
    np.round(fit_time, 3), np.round(pred_time, 3), np.round(precision, 3), np.round(recall, 3), np.round((y_pred==y_test).sum()/len(y_pred), 3)))

# Save to file in the current working directory
joblib_file = "joblib_model.pkl"  
joblib.dump(model, joblib_file)

joblib_file = "joblib_model.pkl"  

joblib_model = joblib.load(joblib_file)

# Calculate the accuracy and predictions
score = joblib_model.score(X_test_vect, y_test)  
print("Test score: {0:.2f} %".format(100 * score))  
Ypredict = joblib_model.predict(X_test_vect)  

def prep_text(mesg):
  text = remove_punct(mesg)
  data = pd.DataFrame({'body_text' : text}, index=[0])
  data['tokens'] = data['body_text'].apply(lambda x: toknize(x))
  data['text_nostop'] = data['tokens'].apply(lambda x: remove_stopwords(x))
  data['text_lemmatized'] = data['text_nostop'].apply(lambda x: lemmatizing(x))
  data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))
  data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))
  return data
  
  
  
# ham msg
msg = """As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune"""
data_prep = prep_text(msg)
display(data_prep)
file = 'scalr.pkl'
scaler = joblib.load(file) 
numerical = ['body_len',  'punct%']
data_prep[numerical] = scaler.transform(data_prep[numerical])

display(data_prep)
vectorizer = joblib.load('vectroizer.pkl')
tfidf_msg_pred = vectorizer.transform(data_prep['text_lemmatized'])
X_pred_vect = pd.concat([data_prep[['body_len', 'punct%']].reset_index(drop=True), 
           pd.DataFrame(tfidf_msg_pred.toarray())], axis=1)

display(X_pred_vect.head())

try:
    Ypredict = joblib_model.predict(X_pred_vect)
    if Ypredict[0] == 0:
       print("msg is ham")
    elif Ypredict[0] == 1:
        print("msg is spam")
except Exception as e:
  print(e)

