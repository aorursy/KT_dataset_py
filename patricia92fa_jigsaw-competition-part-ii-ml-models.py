import pandas as pd, numpy as np

from scipy import sparse

import gc, sys, warnings

warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
import zipfile



# unzip file to specified path

def import_zipped_data(file, output_path):

    with zipfile.ZipFile("../input/"+file+".zip","r") as z:

        z.extractall("/kaggle/working")

        

datasets = ['train.csv', 'test.csv', 'test_labels.csv', 'sample_submission.csv']



kaggle_home = '/kaggle/working'

for dataset in datasets:

    import_zipped_data(dataset, output_path = kaggle_home)
test_df = pd.read_csv('/kaggle/working/test.csv')

train_df = pd.read_csv('/kaggle/working/train.csv')

sample_input = pd.read_csv('/kaggle/working/sample_submission.csv')

test_labels = pd.read_csv('/kaggle/working/test_labels.csv')
train_df.head()
TEXT = 'comment_text'

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# add label to mark non-toxic comments

train_df['non-toxic'] = 1 - train_df[labels].max(axis=1)

# replace na values with placeholder

train_df[TEXT].fillna("unknown", inplace=True)

test_df[TEXT].fillna("unknown", inplace=True)
# tokenizing and filtering of stopwords is included in CountVectorizer

# apply sublinear tf scaling, i.e. replace tf with 1 + log(tf)

# max_df = 0.9, i.e. ignore words appearing in > 90% documents

vec_words = TfidfVectorizer(stop_words='english', analyzer='word',

                            min_df=3, max_df=0.9, strip_accents='unicode', sublinear_tf=1)

# ngram_range=(1,2) # ideally added, increases training time



# create vocabulary based on training data

vec_words.fit_transform(train_df[TEXT])

# vectorize train and test data for scoring

train_vec_words = vec_words.transform(train_df[TEXT])

test_vec_words = vec_words.transform(test_df[TEXT])
## OPTIONAL: n-grams at char level (VERY TIME CONSUMING!) ##

# vectorizer for ngrams with characters

#vec_chars = TfidfVectorizer(ngram_range=(4,5), stop_words='english', analyzer='char',

#                            min_df=3, max_df=0.9, strip_accents='unicode', sublinear_tf=1)



# create vocabulary based on training data

#vec_chars.fit_transform(train_df[TEXT])

# vectorize train and test data for scoring

#train_vec_chars = vec_chars.transform(train_df[TEXT])

#test_vec_chars = vec_chars.transform(test_df[TEXT])
# stack features in one matrix

# features = sparse.hstack([train_vec_words, train_vec_chars])

# test_features = sparse.hstack([test_vec_words, test_vec_chars])
#features.shape
# clean up vectorizations - if matrix was created

#del train_vec_words, test_vec_words, train_vec_chars, test_vec_chars

#gc.collect()
features, test_features = train_vec_words, test_vec_words

labels = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']

y = train_df[labels]
# create an array to store all the predictions

predictions = np.zeros((test_features.shape[0],y.shape[1]))

# fit a model per class

for i, label in enumerate(labels):

    lr = LogisticRegression(C=2, random_state = i, class_weight = 'balanced')

    print('Building {} model for column:{''}'.format(i, label)) 

    lr.fit(features, y[label])

    predictions[:, i] = lr.predict_proba(test_features)[:, 1]
label = 'threat'

pred =  lr.predict(features)

# show confussion matrix for toxicity classification

print(confusion_matrix(y[label], pred))

print(classification_report(y[label], pred))
pred_prob_lr = lr.predict_proba(features)[:,1]

frp, trp, threshold = roc_curve(y[label], pred_prob_lr)

auc_val = auc(frp, trp)



plt.plot([0,1], [0,1], color='b')

plt.plot(frp, trp, color='r', label= 'AUC = %.2f'%auc_val)

plt.legend(loc='lower right')

plt.xlabel('True Positive rate')

plt.ylabel('False Positive rate')

plt.title('ROC Curve')
pred = np.zeros((test_features.shape[0],y.shape[1]))

for i, label in enumerate(labels):

    sgdc = SGDClassifier(loss='squared_loss', penalty='l2',

                        alpha=1e-3, random_state=42,

                        max_iter=5, tol=None)

    print('Building {} model for column:{''}'.format(i, label)) 

    sgdc.fit(features, y[label])

    pred[:, i] = sgdc.decision_function(test_features)
pred = sgdc.predict(features)

# show confussion matrix for toxicity classification

print(confusion_matrix(y[label], pred))

print(classification_report(y[label], pred))
pred_prob_sgdc = sgdc.decision_function(features)

frp, trp, threshold = roc_curve(y[label], pred_prob_sgdc)

auc_val = auc(frp, trp)



plt.plot([0,1], [0,1], color='b')

plt.plot(frp, trp, color='r', label= 'AUC = %.2f'%auc_val)

plt.legend(loc='lower right')

plt.xlabel('True Positive rate')

plt.ylabel('False Positive rate')

plt.title('ROC Curve')
pred = np.zeros((test_features.shape[0],y.shape[1]))

for i, label in enumerate(labels):

    nb = MultinomialNB(alpha = 0.0001)

    print('Building {} model for column:{''}'.format(i, label)) 

    nb.fit(features, y[label])

    pred[:, i] = nb.predict_proba(test_features)[:, 1]
pred =  nb.predict(features)

print(confusion_matrix(y[label], pred))

print(classification_report(y[label], pred))
pred_prob_nb = nb.predict_proba(features)[:,1]

frp, trp, threshold = roc_curve(y[label], pred_prob_nb)

auc_val = auc(frp, trp)



plt.plot([0,1], [0,1], color='b')

plt.plot(frp, trp, color='r', label= 'AUC = %.2f'%auc_val)

plt.legend(loc='lower right')

plt.xlabel('True Positive rate')

plt.ylabel('False Positive rate')

plt.title('ROC Curve')
toxic_clf = Pipeline([

    ('tfidf', TfidfVectorizer(stop_words='english', analyzer='word',

                              min_df=3, max_df=0.9, strip_accents='unicode', sublinear_tf=1)),

    ('clf', MultinomialNB())

])
parameters = {

    'clf__alpha': (0, 0.25, 0.5, 0.75, 1)

}
# n_jobs = -1 detects how many cores are installed and uses them all

opt_toxic_clf = GridSearchCV(toxic_clf, parameters, cv=5, n_jobs=-1)

opt_toxic_clf = opt_toxic_clf.fit(train_df[TEXT], y[label])
print(opt_toxic_clf.best_score_)

for p in sorted(parameters.keys()):

    print("%s: %r" % (p, opt_toxic_clf.best_params_[p]))

print(opt_toxic_clf.cv_results_)
toxic_clf = Pipeline([

    ('tfidf', TfidfVectorizer(stop_words='english', analyzer='word',

                              min_df=3, max_df=0.9, strip_accents='unicode', sublinear_tf=1)),

    ('clf', SGDClassifier())

])

parameters = {

    'clf__alpha': (0.0001, 0.001, 0.01, 0.1)

}

# n_jobs = -1 detects how many cores are installed and uses them all

opt_toxic_clf = GridSearchCV(toxic_clf, parameters, cv=5, n_jobs=-1)

opt_toxic_clf = opt_toxic_clf.fit(train_df[TEXT], y[label])
print(opt_toxic_clf.best_score_)

for p in sorted(parameters.keys()):

    print("%s: %r" % (p, opt_toxic_clf.best_params_[p]))

print(opt_toxic_clf.cv_results_)
def str_to_class(classname):

    return getattr(sys.modules[__name__], classname)



def compare_clfs(label):

    plt.figure(0).clf()

    # plot reference AUC 0.5 line

    plt.plot([0,1], [0,1], color='k')

    

    # compute ROC for each classifier

    classifiers = {'lr':'g', 'sgdc':'b', 'nb':'r'}

    for c in classifiers.keys():

        model = str_to_class(c)

        if c == 'sgdc':

            pred_prob = model.decision_function(features)

        else:

            pred_prob = model.predict_proba(features)[:,1]

        frp, trp, threshold = roc_curve(y[label], pred_prob)

        auc_val = auc(frp, trp)

        plt.plot(frp, trp, color=classifiers[c], label= f'{c.upper()} AUC = %.2f'%auc_val)



    plt.legend(loc=0)

    plt.xlabel('True Positive rate')

    plt.ylabel('False Positive rate')

    plt.title(f'ROC Curve for {label.capitalize()} Class Classifiers')

    plt.show()
compare_clfs('toxic')

compare_clfs('severe_toxic')
compare_clfs('threat')
compare_clfs('identity_hate')
compare_clfs('obscene')
compare_clfs('insult')