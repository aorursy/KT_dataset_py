import warnings

warnings.filterwarnings("ignore")
import numpy as np 

import pandas as pd 



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer



from string import punctuation 



from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier

import lightgbm as lgb
df = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")

df.head()
pd.read_csv("../input/quora-insincere-questions-classification/test.csv").head()
df.info()
df.duplicated().sum()
df.target.value_counts()
df.target.describe()
plt.figure(figsize=(5, 4))

sns.countplot(x='target', data=df)

plt.title('Reparition of question by sincerity (insincere = 1)');
print(f'There are {df.target.sum() / df.shape[0] * 100 :.1f}% of insincere questions, which make the dataset highly unbalanced.')
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
print('Word cloud image generated from sincere questions')

sincere_wordcloud = WordCloud(width=600, height=400, background_color ='white', min_font_size = 10).generate(str(df[df["target"] == 0]["question_text"]))

#Positive Word cloud

plt.figure(figsize=(15,6), facecolor=None)

plt.imshow(sincere_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show();
print('Word cloud image generated from INsincere questions')

insincere_wordcloud = WordCloud(width=600, height=400, background_color ='white', min_font_size = 10).generate(str(df[df["target"] == 1]["question_text"]))

#Positive Word cloud

plt.figure(figsize=(15,6), facecolor=None)

plt.imshow(insincere_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show();
# if needed

# nltk.download('stopwords')
import nltk

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

stop_words
def create_features(df_):

    """Retrieve from the text column the nb of : words, unique words, characters, stopwords,

    punctuations, upper/lower case char, title..."""

    

    df_["nb_words"] = df_["question_text"].apply(lambda x: len(x.split()))

    df_["nb_unique_words"] = df_["question_text"].apply(lambda x: len(set(str(x).split())))

    df_["nb_chars"] = df["question_text"].apply(lambda x: len(str(x)))

    df_["nb_stopwords"] = df_["question_text"].apply(lambda x : len([nw for nw in str(x).split() if nw.lower() in stop_words]))

    df_["nb_punctuation"] = df_["question_text"].apply(lambda x : len([np for np in str(x) if np in punctuation]))

    df_["nb_uppercase"] = df_["question_text"].apply(lambda x : len([nu for nu in str(x).split() if nu.isupper()]))

    df_["nb_lowercase"] = df_["question_text"].apply(lambda x : len([nl for nl in str(x).split() if nl.islower()]))

    df_["nb_title"] = df_["question_text"].apply(lambda x : len([nl for nl in str(x).split() if nl.istitle()]))

    return df_
df = create_features(df)

df.sample(2)
num_feat = ['nb_words', 'nb_unique_words', 'nb_chars', 'nb_stopwords', \

            'nb_punctuation', 'nb_uppercase', 'nb_lowercase', 'nb_title', 'target'] 

# side note : remove target if needed later



df_sample = df[num_feat].sample(n=round(df.shape[0]/6), random_state=42)



plt.figure(figsize=(16,10))

sns.pairplot(data=df_sample, hue='target')

plt.show()
df_sample[df_sample['target'] == 0].describe()
df_sample[df_sample['target'] == 1].describe()
plt.figure(figsize=(10,10))

plt.subplot(331)



i=0

for c in num_feat:

    plt.subplot(3, 3, i+1)

    i += 1

    sns.kdeplot(df_sample[df_sample['target'] == 0][c], shade=True)

    sns.kdeplot(df_sample[df_sample['target'] == 1][c], shade=False)

    plt.title(c)



plt.show()
sns.set(style="white")



# Compute the correlation matrix

corr = df_sample[num_feat].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(7, 6))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})
class Vocabulary(object):

    # credits : Shankar G see https://www.kaggle.com/kaosmonkey/visualize-sincere-vs-insincere-words

    

    def __init__(self):

        self.vocab = {}

        self.STOPWORDS = set()

        self.STOPWORDS = set(stopwords.words('english'))

        

    def build_vocab(self, lines):

        for line in lines:

            for word in line.split(' '):

                word = word.lower()

                if (word in self.STOPWORDS):

                    continue

                if (word not in self.vocab):

                    self.vocab[word] = 0

                self.vocab[word] +=1 

    

    def generate_ngrams(text, n_gram=1):

        """arg: text, n_gram"""

        token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

        ngrams = zip(*[token[i:] for i in range(n_gram)])

        return [" ".join(ngram) for ngram in ngrams]

    

    def horizontal_bar_chart(df, color):

        trace = go.Bar(

            y=df["word"].values[::-1],

            x=df["wordcount"].values[::-1],

            showlegend=False,

            orientation = 'h',

            marker=dict(

            color=color,

            ),

        )

        return trace
sincere_vocab = Vocabulary()

sincere_vocab.build_vocab(df[df['target'] == 0]['question_text'])

sincere_vocabulary = sorted(sincere_vocab.vocab.items(), reverse=True, key=lambda kv: kv[1])

    

df_sincere_vocab = pd.DataFrame(sincere_vocabulary, columns=['word_sincere', 'frequency'])

sns.barplot(y='word_sincere', x='frequency', data=df_sincere_vocab[:20])
insincere_vocab = Vocabulary()

insincere_vocab.build_vocab(df[df['target'] == 1]['question_text'])

insincere_vocabulary = sorted(insincere_vocab.vocab.items(), reverse=True, key=lambda kv: kv[1])



df_insincere_vocab = pd.DataFrame(insincere_vocabulary, columns=['word_insincere', 'frequency'])

sns.barplot(y='word_insincere', x='frequency', data=df_insincere_vocab[:20])
def get_fscore_matrix(fitted_clf, model_name):

    print(model_name, ' :')

    

    # get classes predictions for the classification report 

    y_train_pred, y_pred = fitted_clf.predict(X_train), fitted_clf.predict(X_test)

    print(classification_report(y_test, y_pred), '\n') # target_names=y

    

    # computes probabilities keep the ones for the positive outcome only      

    print(f'F1-score = {f1_score(y_test, y_pred):.2f}')
# if needed the first time  

# import nltk

# nltk.download('punkt')
df = df[['question_text', 'target']]



def text_processing(local_df):

    """ return the dataframe with tokens stemmetized without numerical values & stopwords """

    stemmer = PorterStemmer()

    # Perform preprocessing

    local_df['txt_processed'] = local_df['question_text'].apply(lambda df: word_tokenize(df))

    local_df['txt_processed'] = local_df['txt_processed'].apply(lambda x: [item for item in x if item.isalpha()])

    local_df['txt_processed'] = local_df['txt_processed'].apply(lambda x: [item for item in x if item not in stop_words])

    local_df['txt_processed'] = local_df['txt_processed'].apply(lambda x: [stemmer.stem(item) for item in x])

    return local_df
df = text_processing(df)

df.tail(2)
vectorizer = TfidfVectorizer(lowercase=False, analyzer=lambda x: x, min_df=0.01, max_df=0.999)

# min_df & max_df param added for less memory usage



tf_idf = vectorizer.fit_transform(df['txt_processed']).toarray()

pd.DataFrame(tf_idf, columns=vectorizer.get_feature_names()).head()
# Split the data

X_train, X_test, y_train, y_test = train_test_split(tf_idf, df['target'], test_size=0.2, random_state=42)
model = XGBClassifier(objective="binary:logistic")

model.fit(X_train, y_train)

get_fscore_matrix(model, 'XGB Clf withOUT weights')
ratio = ((len(y_train) - y_train.sum()) - y_train.sum()) / y_train.sum()

ratio
model = XGBClassifier(objective="binary:logistic", scale_pos_weight=ratio)

model.fit(X_train, y_train)

get_fscore_matrix(model, 'XGB Clf WITH weights')
model = lgb.LGBMClassifier(n_jobs = -1, class_weight={0:y_train.sum(), 1:len(y_train) - y_train.sum()})

model.fit(X_train, y_train)

get_fscore_matrix(model, 'LGBM weighted')
model = LogisticRegression(class_weight={0:y_train.sum(), 1:len(y_train) - y_train.sum()}, C=0.5, max_iter=100, n_jobs=-1)

model.fit(X_train, y_train)

get_fscore_matrix(model, 'LogisticRegression')
df['str_processed'] = df['txt_processed'].apply(lambda x: " ".join(x))

df.head(2)
pipeline = Pipeline([("cv", CountVectorizer(analyzer="word", ngram_range=(1,4), max_df=0.9)),

                     ("clf", LogisticRegression(solver="saga", class_weight="balanced", C=0.45, max_iter=250, verbose=1, n_jobs=-1))])
X_train, X_test, y_train, y_test = train_test_split(df['str_processed'], df.target, test_size=0.2, stratify = df.target.values)
lr_model = pipeline.fit(X_train, y_train)

lr_model
get_fscore_matrix(lr_model, 'lr_pipe')
pd.read_csv("../input/quora-insincere-questions-classification/sample_submission.csv").head(2)
df_test = pd.read_csv("../input/quora-insincere-questions-classification/test.csv", index_col='qid')

df_test.tail(2)
df_test = text_processing(df_test)

df_test['str_processed'] = df_test['txt_processed'].apply(lambda x: " ".join(x))

df_test.head(2)
y_pred_final = lr_model.predict(df_test['str_processed'])

y_pred_final
df_submission = pd.DataFrame({"qid":df_test.index, "prediction":y_pred_final})

df_submission.head()
df_submission.to_csv('submission.csv', index=False)