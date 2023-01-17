import os

import sys

import numpy as np

import pandas as pd



pd.set_option('display.max_colwidth', 200)

pd.set_option('display.max_rows', 20)





from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm, tqdm_notebook

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline
MAX_SEQUENCE_LENGTH = 320

SEED = 1234

EPOCHS = 1

Data_dir="../input/jigsaw-unintended-bias-in-toxicity-classification"

Input_dir = "../input"

WORK_DIR = "../working/"

BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-24_h-1024_a-16/uncased_L-24_H-1024_A-16/'  

TOXICITY_COLUMN = 'target'



package_dir_a = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"

sys.path.insert(0, package_dir_a)

from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch

from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification,BertAdam

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)

train_df = pd.read_csv(os.path.join(Data_dir,"train.csv"))

train_df['comment_text'] = train_df['comment_text'].astype(str) 



test_df = pd.read_csv(os.path.join(Data_dir, "test.csv"))

test_df['comment_text'] = test_df['comment_text'].astype(str)



print('loaded %d train data' % len(train_df))

print('loaded %d test data' % len(test_df))

train_df.head()
identity_columns = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']



train_df[identity_columns] = (train_df[identity_columns]>=0.5).astype(float)

train_df['target']=(train_df['target']>=0.5).astype(float)

train_df['target'].value_counts()



train_df[['target']+identity_columns].head()
train_df['target'].value_counts()
# Number of words distribution

sns.distplot(train_df['comment_text'].apply(lambda t:len(t.split(" "))+1))
from wordcloud import WordCloud, STOPWORDS



def plot_wordcloud(text, mask=None, max_words=300, max_font_size=100, figure_size=(10,12), 

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



plot_wordcloud(train_df["comment_text"], title="All Comments")

plot_wordcloud(train_df["comment_text"][train_df['target']==1], title="Toxic Comments")

def convert_lines(example, max_seq_length,tokenizer):

    max_seq_length -=2

    all_tokens = []

    longer = 0

    for text in tqdm_notebook(example):

        tokens_a = tokenizer.tokenize(text)

        if len(tokens_a)>max_seq_length:

            tokens_a = tokens_a[:max_seq_length]

            longer += 1

        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))

        all_tokens.append(one_token)

    print(longer)

    return np.array(all_tokens)



sequences = convert_lines(train_df["comment_text"],MAX_SEQUENCE_LENGTH,tokenizer)

sequences
import pickle



skf = StratifiedKFold(n_splits=5, random_state=True, shuffle=True)

splits = list(skf.split(train_df, train_df['target']))





with open('skf_5_splits.pkl', 'wb') as f:

    pickle.dump(splits, f)

    

with open('bert_large_sequences_{}.pkl'.format(MAX_SEQUENCE_LENGTH), 'wb') as t:

    pickle.dump(sequences, t)

    