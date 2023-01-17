import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
train_data = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
test_data = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
train_data.head()
train_data.tail()
data = train_data.copy()
comments = data['comment_text'].to_numpy()
labels = data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].to_numpy()
dataframe = []
for index in range(len(labels)):
    num = np.count_nonzero(labels[index])
    if(num == 0):
        dataframe.append([comments[index], 0])
    else:
        dataframe.append([comments[index], 1])
df = pd.DataFrame(dataframe, columns = ['comment', 'label'])
df.head()
df.tail()
ax = plt.subplot()

g = sns.countplot(df.label)
g.set_xticklabels(['Toxic', 'Not Toxic'])
g.set_yticklabels(['Count'])

# function to show values on bars
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.0f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
show_values_on_bars(ax)

sns.despine(left=True, bottom=True)
plt.xlabel('')
plt.ylabel('')
plt.title('Distribution of Classes', fontsize=30)
plt.tick_params(axis = 'x', which='major', labelsize=15)
plt.show()
df['label'].value_counts()
count_class_toxic, count_class_non_toxic = df.label.value_counts()

class_toxic = df[df['label'] == 0]
class_not_toxic = df[df['label'] == 1]
class_not_toxic_over = class_not_toxic.sample(count_class_toxic, replace = True)
test_over = pd.concat([class_toxic, class_not_toxic_over], axis = 0)
print(test_over.label.value_counts())
test_over.head()
test_over.tail()
new_filter = test_over["comment"] != ""
test_over = test_over[new_filter]
test_over = test_over.dropna()
def preprocessing_text(sen):
    # Remove punctuations and numbers
    sent = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sent = re.sub(r"\s+[a-zA-Z]\s+", ' ', sent)

    # Removing multiple spaces
    sent = re.sub(r'\s+', ' ', sent)

    return sent
X_new = []
new_sentences = list(test_over["comment"])
for sents in new_sentences:
    X_new.append(preprocessing_text(sents))
new_stopwords = stopwords.words('english')
vectorizer_new = TfidfVectorizer(stop_words = new_stopwords, use_idf = True)
bag_of_words_new = vectorizer_new.fit_transform(X_new)
x = bag_of_words_new
y = test_over['label']
X_train_old, X_test_old, Y_train_old, Y_test_old = train_test_split(x, y, test_size = 0.25, random_state = 27)
rf_clf_resampled = RandomForestClassifier(25)
rf_clf_resampled.fit(X_train_old, Y_train_old)
rf_predict_resampled = rf_clf_resampled.predict(X_test_old)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
rf_cm_resampled = confusion_matrix(Y_test_old, rf_predict_resampled)
plot_confusion_matrix(rf_cm_resampled, [0, 1])
plt.show()
print('Accuracy Score:', accuracy_score(Y_test_old, rf_predict_resampled))
print('Precision:', precision_score(Y_test_old, rf_predict_resampled))
print('Recall:',recall_score(Y_test_old, rf_predict_resampled))
print('F1 Score:', f1_score(Y_test_old, rf_predict_resampled))
print("Area under curve for Random Forest:", roc_auc_score(Y_test_old, rf_predict_resampled))
score_rf = cross_val_score(rf_clf_resampled, x, y, cv = 5)
print("CV score {}".format(np.mean(score_rf)))