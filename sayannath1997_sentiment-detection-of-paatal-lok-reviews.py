import warnings

warnings.filterwarnings("ignore")
import nltk

nltk.download('all')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random

import string

from nltk.tokenize import WordPunctTokenizer 

from nltk.corpus import stopwords

from nltk import WordNetLemmatizer

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve

!pip install treeinterpreter

from treeinterpreter import treeinterpreter as ti

import pickle
original_dataset = pd.read_csv("/kaggle/input/paatal-lok-imdb-reviews-dataset/Paatal_Lok_IMDB_Reviews.csv",

                              index_col = '#')

dataset = original_dataset.copy()

dataset.head()
dataset = dataset.drop(['username', 'review-date', 'actions-helpful', 'rating'], axis = 1)

dataset.head()
dataset[['title', 'review-text']] = dataset[['title', 'review-text']].astype(str)

dataset['Type'].value_counts().plot(kind = 'barh')

plt.show()
pos, neg = dataset['Type'].value_counts().values

total = neg + pos

print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(

    total, pos, 100 * pos / total))
random.seed(123)

msk = np.random.rand(len(dataset)) < 0.75

train_dataset = dataset[msk]

test_dataset = dataset[~msk]

print(len(train_dataset))

print(len(test_dataset))
train_dataset.head()
test_dataset.head()
train_dataset['Type'].value_counts().plot(kind ='barh')

plt.show()
test_dataset['Type'].value_counts().plot(kind ='barh')

plt.show()
y_train = train_dataset.pop('Type')

y_test = test_dataset.pop('Type')
labelencoder_y = LabelEncoder()

y_train = labelencoder_y.fit_transform(y_train)

y_test = labelencoder_y.transform(y_test)
def preprocess(text):

    text = "".join([character for character in text if character not in string.punctuation + '0123456789'])

    text = text.lower()

    tk = WordPunctTokenizer()

    text = tk.tokenize(text)

    stopword = stopwords.words('english')

    text = [word for word in text if word not in stopword]

    wn = WordNetLemmatizer()

    text = [wn.lemmatize(word) for word in text]

    text = " ".join(text)

    text = text.replace('propoganda', 'propaganda')

    return text
train_corpus = []

for i in range(len(train_dataset)):

    review = train_dataset.iloc[i, 0] + ' ' + train_dataset.iloc[i, 1]

    train_corpus.append(review)

    

test_corpus = []

for i in range(len(test_dataset)):

    review = test_dataset.iloc[i, 0] + ' ' + test_dataset.iloc[i, 1]

    test_corpus.append(review)
weight_for_neg = (1 / neg)*(total)

weight_for_pos = (1 / pos)*(total)



class_weight = {0: weight_for_neg, 1: weight_for_pos}



print('Weight for Negative Reviews: {:.2f}'.format(weight_for_neg))

print('Weight for Positive Reviews: {:.2f}'.format(weight_for_pos))
text_clf_gs = Pipeline([

    ('tfidf', TfidfVectorizer(preprocessor = preprocess)),

    ('clf', RandomForestClassifier(n_estimators = 500, random_state = 0, class_weight = class_weight))

])



parameters = {

    'tfidf__ngram_range' : [(1, 1), (1, 2), (1, 3)],

    'clf__criterion' : ['gini', 'entropy'],

    'clf__min_samples_leaf' : [2, 3, 4],

    'clf__max_features' : ['None', 'sqrt', 'log2']

}
text_clf_gs = GridSearchCV(text_clf_gs, param_grid = parameters,

                  cv = 5,

                  scoring = 'roc_auc')

text_clf_gs = text_clf_gs.fit(train_corpus, y_train)

print("Best CV ROC_AUC Score:", text_clf_gs.best_score_)

best_parameters = text_clf_gs.best_params_

print("Best Parameters:", best_parameters)
text_clf = Pipeline([

    ('tfidf', TfidfVectorizer(preprocessor = preprocess, ngram_range = best_parameters['tfidf__ngram_range'])),

    ('clf', RandomForestClassifier(n_estimators = 500, random_state = 0, class_weight = class_weight,

                                  criterion = best_parameters['clf__criterion'], 

                                   max_features = best_parameters['clf__max_features'], 

                                   min_samples_leaf = best_parameters['clf__min_samples_leaf']))

])



text_clf = text_clf.fit(train_corpus, y_train)
scores = text_clf.predict_proba(test_corpus)[: , 1]

fpr, tpr, thresholds = roc_curve(y_test, scores)

roc_auc = auc(fpr, tpr) 

print("ROC-AUC:", roc_auc)



plt.subplots(figsize=(10, 10))

plt.plot(fpr, tpr, 'o-', label="ROC curve")

plt.plot(np.linspace(0,1,10), np.linspace(0,1,10), label="diagonal")

for x, y, txt in zip(fpr[::2], tpr[::2], thresholds[::2]):

    plt.annotate(np.round(txt,2), (x, y-0.04))

rnd_idx = 12

plt.annotate('this point refers to the tpr and the fpr\n at a probability threshold of {}'.format(np.round(thresholds[rnd_idx], 2)), 

             xy=(fpr[rnd_idx], tpr[rnd_idx]), xytext=(fpr[rnd_idx]+0.2, tpr[rnd_idx]-0.25),

             arrowprops=dict(facecolor='black', lw=2, arrowstyle='->'),)

plt.legend(loc="upper left")

plt.xlabel("FPR")

plt.ylabel("TPR")
i = np.arange(len(tpr))

roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),

                    'tpr' : pd.Series(tpr, index = i), 

                    '1-fpr' : pd.Series(1-fpr, index = i), 

                    'tf' : pd.Series(tpr - (1-fpr), index = i), 

                    'thresholds' : pd.Series(thresholds, index = i)})

roc.iloc[(roc.tf-0).abs().argsort()[:1], :]
thres = roc.iloc[(roc.tf-0).abs().argsort()[:1], 4].values
def plot_cm(labels, predictions, p = 0.5):

  cm = confusion_matrix(labels, predictions > p)

  plt.figure(figsize = (5, 5))

  sns.heatmap(cm, annot = True, fmt = "d")

  plt.title('Confusion matrix for threshold = {}'.format(p))

  plt.ylabel('Actual label')

  plt.xlabel('Predicted label')



  print('Negative Reviews Detected Negative (True Negatives): ', cm[0][0])

  print('Negative Reviews Detected Positive (False Positives): ', cm[0][1])

  print('Positive Reviews Detected Negative (False Negatives): ', cm[1][0])

  print('Positive Reviews Detected Positive (True Positives): ', cm[1][1])

  print('Total Negative Reviews: ', np.sum(cm[0]))
plot_cm(y_test, scores, p = thres)

# Predict test set result

y_pred = (scores > thres).astype('int64')

print("Test Accuracy:", accuracy_score(y_test, y_pred))
prediction, bias, contributions = ti.predict(text_clf['clf'], text_clf['tfidf'].transform(test_corpus))
contribution_frame = pd.DataFrame(contributions[:, :, 1], columns = text_clf['tfidf'].get_feature_names())

contribution_frame.describe().T.iloc[np.argsort(-(contribution_frame.describe().T['mean'].abs()))]
sns_colors = sns.color_palette('colorblind')
# Boilerplate code for plotting :)

def _get_color(value):

    """To make positive DFCs plot green, negative DFCs plot red."""

    green, red = sns.color_palette()[2:4]

    if value >= 0: return green

    return red



def _add_feature_values(feature_values, ax):

    """Display feature's values on left of plot."""

    x_coord = ax.get_xlim()[0]

    OFFSET = 0.15

    for y_coord, (feat_name, feat_val) in enumerate(feature_values.items()):

        t = plt.text(x_coord, y_coord - OFFSET, '{}'.format(round(feat_val, 2)), size=12)

        t.set_bbox(dict(facecolor='white', alpha=0.5))

    from matplotlib.font_manager import FontProperties

    font = FontProperties()

    font.set_weight('bold')

    t = plt.text(x_coord, y_coord + 1 - OFFSET, 'feature\nvalue',

    fontproperties=font, size=12)



def plot_example(example):

  TOP_N = 8 # View top 8 features.

  sorted_ix = example.abs().sort_values()[-TOP_N:].index  # Sort by magnitude.

  example = example[sorted_ix]

  colors = example.map(_get_color).tolist()

  ax = example.to_frame().plot(kind='barh',

                          color=[colors],

                          legend=None,

                          alpha=0.75,

                          figsize=(10,6))

  ax.grid(False, axis='y')

  ax.set_yticklabels(ax.get_yticklabels(), size=14)



  # Add feature values.

  _add_feature_values(pd.DataFrame.sparse.from_spmatrix(text_clf['tfidf'].transform(test_corpus),

                                 columns = text_clf['tfidf'].get_feature_names()).iloc[ID][sorted_ix], ax)

  return ax
probs = prediction[: , 1]

labels = y_test
ID = np.where(y_test == 0)[0][0]

example = contribution_frame.iloc[ID] 

TOP_N = 8  

sorted_ix = example.abs().sort_values()[-TOP_N:].index

ax = plot_example(example)

ax.set_title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, probs[ID], labels[ID]))

ax.set_xlabel('Contribution to predicted probability', size = 14)

plt.show()
ID = np.where(y_test == 1)[0][0]

example = contribution_frame.iloc[ID]  

TOP_N = 8 

sorted_ix = example.abs().sort_values()[-TOP_N:].index

ax = plot_example(example)

ax.set_title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, probs[ID], labels[ID]))

ax.set_xlabel('Contribution to predicted probability', size = 14)

plt.show()
# Boilerplate plotting code.

def dist_violin_plot(df_dfc, ID):

  # Initialize plot.

  fig, ax = plt.subplots(1, 1, figsize=(10, 6))



  # Create example dataframe.

  TOP_N = 8  # View top 8 features.

  example = df_dfc.iloc[ID]

  ix = example.abs().sort_values()[-TOP_N:].index

  example = example[ix]

  example_df = example.to_frame(name='dfc')



  # Add contributions of entire distribution.

  parts=ax.violinplot([df_dfc[w] for w in ix],

                 vert=False,

                 showextrema=False,

                 widths=0.7,

                 positions=np.arange(len(ix)))

  face_color = sns_colors[0]

  alpha = 0.15

  for pc in parts['bodies']:

      pc.set_facecolor(face_color)

      pc.set_alpha(alpha)



  # Add feature values.

  _add_feature_values(pd.DataFrame.sparse.from_spmatrix(text_clf['tfidf'].transform(test_corpus),

                                 columns = text_clf['tfidf'].get_feature_names()).iloc[ID][sorted_ix], ax)



  # Add local contributions.

  ax.scatter(example,

              np.arange(example.shape[0]),

              color=sns.color_palette()[2],

              s=100,

              marker=".",

              label='contributions for example')



  # Legend

  # Proxy plot, to show violinplot dist on legend.

  ax.plot([0,0], [1,1], label='test set contributions\ndistributions',

          color=face_color, alpha=alpha, linewidth=10)

  legend = ax.legend(loc='lower center', shadow=True, fontsize='x-large',

                     frameon=True)

  legend.get_frame().set_facecolor('white')



  # Format plot.

  ax.set_yticks(np.arange(example.shape[0]))

  ax.set_yticklabels(example.index)

  ax.grid(False, axis='y')

  ax.set_xlabel('Contribution to predicted probability', size=14)
ID = np.where(y_test == 1)[0][0]

dist_violin_plot(contribution_frame, ID)

plt.title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, probs[ID], labels[ID]))

plt.show()
# Gain Based

importances = text_clf['clf'].feature_importances_

df_imp = pd.Series(importances, index = text_clf['tfidf'].get_feature_names())

df_imp = df_imp.sort_values(ascending = False)

N = 8

ax = (df_imp.iloc[0:N][::-1]

    .plot(kind='barh',

          color=sns_colors[2],

          title='Gain feature importances',

          figsize=(10, 6)))

ax.grid(False, axis='y')
# Aggregated Feature Importances

contribution_mean = contribution_frame.abs().mean()

N = 8

sorted_ix = contribution_mean.abs().sort_values()[-N:].index  

ax = contribution_mean[sorted_ix].plot(kind='barh',

                       color=sns_colors[1],

                       title='Mean directional feature contributions',

                       figsize=(10, 6))

ax.grid(False, axis='y')
filename = 'Paatal_Lok_Review_Classifier.pkl'

pickle.dump(text_clf, open(filename, 'wb'))
model = pickle.load(open(filename, 'rb'))
# Put the review in a list

# Put in this format [review title : review text]

a_positive_and_a_negative_review = ["Don't go by the overall ratings of the show : I don't write reviews but the ratings break my heart. The show shows exactly what is wrong with the Indian society. The bad ratings and the backlash proves how ignorant and overtly sensitive we are to the truth. Stellar performances, incredibly layered characters, amazing character building, brilliant writing & direction. This tops Sacred Games for me as an overall show. Believe a stranger, give it a watch if you haven't. Watch it with an open mind. Set aside your ego and privileges before you do, though.",

                                   "Hindu religion is being shown in Bad light. So much hatred based on lies : Showing a Hindu Pandit cooking meat and serving to a guy sitting in front of Hindu Goddess photo and eating Non-veg. Is this artistry ? And the dialogue used there is so puke worthy. Can't even say it here. It's regarding a mother. Sickest minds. Calling a Nepalese woman as prostitute. This is Racism. Sickness. And showing our Hindu symbols and Pooja as violence. All exaggerations."]

model.predict_proba(a_positive_and_a_negative_review)