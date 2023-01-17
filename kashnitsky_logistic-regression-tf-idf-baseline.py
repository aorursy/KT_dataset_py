import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack

from matplotlib import pyplot as plt

import seaborn as sns

import eli5
train = pd.read_csv('../input/train.csv', index_col='id').fillna(' ')

valid = pd.read_csv('../input/valid.csv', index_col='id').fillna(' ')

test = pd.read_csv('../input/test.csv', index_col='id').fillna(' ')
train.head()
train_val = pd.concat([train, valid])
sns.countplot(train_val['label']);

plt.title('Train+val: Target distribution');
plt.subplots(1, 2)

plt.subplot(1, 2, 1)

train_val['text'].apply(lambda x: len(x.split())).plot(kind='hist');

plt.yscale('log');

plt.title('Train & val');

plt.subplot(1, 2, 2)

test['text'].apply(lambda x: len(x.split())).plot(kind='hist');

plt.yscale('log');

plt.title('Test');
from wordcloud import WordCloud, STOPWORDS



# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='black',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  

    

plot_wordcloud(train_val["text"], title="Word Cloud of reviews")
text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=150000)
%%time

X_train_text = text_transformer.fit_transform(train_val['text'])

X_test_text = text_transformer.transform(test['text'])
X_train_text.shape, X_test_text.shape
logit = LogisticRegression(C=5e1, solver='lbfgs', multi_class='multinomial', random_state=17, n_jobs=4)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
%%time

cv_results = cross_val_score(logit, X_train_text, train_val['label'], cv=skf, scoring='f1_micro')
cv_results, cv_results.mean()
%%time

logit.fit(X_train_text, train_val['label'])
eli5.show_weights(estimator=logit, 

                  feature_names= list(text_transformer.get_feature_names()),

                 top=(50, 5))
test_preds = logit.predict(X_test_text)
pd.DataFrame(test_preds, columns=['label']).head()
pd.DataFrame(test_preds, columns=['label']).to_csv('logit_tf_idf_starter_submission.csv',

                                                  index_label='id')