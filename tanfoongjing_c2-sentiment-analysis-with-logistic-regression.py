# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('../input/student-shopee-code-league-sentiment-analysis/train.csv')

df_train.head()
import re

def review_cleaning(text):

    

    # delete lowercase and newline

    text = text.lower()

    text = re.sub(r'\n', '', text)

    text = re.sub('([.,!?()])', r' \1 ', text)

    text = re.sub('\s{2,}', ' ', text)

    

    # change emoticon to text

    text = re.sub(r':\(', 'dislike', text)

    text = re.sub(r': \(\(', 'dislike', text)

    text = re.sub(r':, \(', 'dislike', text)

    text = re.sub(r':\)', 'smile', text)

    text = re.sub(r';\)', 'smile', text)

    text = re.sub(r':\)\)\)', 'smile', text)

    text = re.sub(r':\)\)\)\)\)\)', 'smile', text)

    text = re.sub(r'=\)\)\)\)', 'smile', text)

    

    # We decide to include punctuation in the model so we comment this line out!

#     text = re.sub('[^a-z0-9! ]', ' ', text)

    

    tokenizer = text.split()

    

    return ' '.join([text for text in tokenizer])
df_train['review'] = df_train['review'].apply(review_cleaning)

df_train.head()
repeated_rows_train = []





for idx, review in enumerate(df_train['review']):

    if re.match(r'\w*(\w)\1+', review):

        repeated_rows_train.append(idx)
def delete_repeated_char(text):

    

    text = re.sub(r'(\w)\1{2,}', r'\1', text)

    

    return text
df_train.loc[repeated_rows_train, 'review'] = df_train.loc[repeated_rows_train, 'review'].apply(delete_repeated_char)

df_train.head()
import emoji

def emoji_cleaning(text):

    

    # Change emoji to text

    text = emoji.demojize(text).replace(":", " ")

    

    # Delete repeated emoji

    tokenizer = text.split()

    repeated_list = []

    

    for word in tokenizer:

        if word not in repeated_list:

            repeated_list.append(word)

    

    text = ' '.join(text for text in repeated_list)

    text = text.replace("_", " ").replace("-", " ")

    return text
have_emoji_train_idx = []



for idx, review in enumerate(df_train['review']):

    if any(char in emoji.UNICODE_EMOJI for char in review):

        have_emoji_train_idx.append(idx)

train_emoji_percentage = round(len(have_emoji_train_idx) / df_train.shape[0] * 100, 2)

print(f'Train data has {len(have_emoji_train_idx)} rows that used emoji, that means {train_emoji_percentage} percent of the total')

train_df_original = df_train.copy()



# emoji_cleaning

df_train.loc[have_emoji_train_idx, 'review'] = df_train.loc[have_emoji_train_idx, 'review'].apply(emoji_cleaning)
#print('Before: ', train_df_original.loc[92129, 'review'])

print('After: ', df_train.loc[92129, 'review'])



#print('\nBefore: ', train_df_original.loc[56938, 'review'])

print('After: ', df_train.loc[56938, 'review'])



#print('\nBefore: ', train_df_original.loc[72677, 'review'])

print('After: ', df_train.loc[72677, 'review'])



#print('\nBefore: ', train_df_original.loc[36558, 'review'])

print('After: ', df_train.loc[36558, 'review'])
def recover_shortened_words(text):

    

    # put \b (boundary) for avoid the characters in the word to be replaced

    # I only make a few examples here, you can add if you're interested :)

    

    text = re.sub(r'\bapaa\b', 'apa', text)

    

    text = re.sub(r'\bbsk\b', 'besok', text)

    text = re.sub(r'\bbrngnya\b', 'barangnya', text)

    text = re.sub(r'\bbrp\b', 'berapa', text)

    text = re.sub(r'\bbgt\b', 'banget', text)

    text = re.sub(r'\bbngt\b', 'banget', text)

    text = re.sub(r'\bgini\b', 'begini', text)

    text = re.sub(r'\bbrg\b', 'barang', text)

    

    text = re.sub(r'\bdtg\b', 'datang', text)

    text = re.sub(r'\bd\b', 'di', text)

    text = re.sub(r'\bsdh\b', 'sudah', text)

    text = re.sub(r'\bdri\b', 'dari', text)

    text = re.sub(r'\bdsni\b', 'disini', text)

    

    text = re.sub(r'\bgk\b', 'gak', text)

    

    text = re.sub(r'\bhrs\b', 'harus', text)

    

    text = re.sub(r'\bjd\b', 'jadi', text)

    text = re.sub(r'\bjg\b', 'juga', text)

    text = re.sub(r'\bjgn\b', 'jangan', text)

    

    text = re.sub(r'\blg\b', 'lagi', text)

    text = re.sub(r'\blgi\b', 'lagi', text)

    text = re.sub(r'\blbh\b', 'lebih', text)

    text = re.sub(r'\blbih\b', 'lebih', text)

    

    text = re.sub(r'\bmksh\b', 'makasih', text)

    text = re.sub(r'\bmna\b', 'mana', text)

    

    text = re.sub(r'\borg\b', 'orang', text)

    

    text = re.sub(r'\bpjg\b', 'panjang', text)

    

    text = re.sub(r'\bka\b', 'kakak', text)

    text = re.sub(r'\bkk\b', 'kakak', text)

    text = re.sub(r'\bklo\b', 'kalau', text)

    text = re.sub(r'\bkmrn\b', 'kemarin', text)

    text = re.sub(r'\bkmrin\b', 'kemarin', text)

    text = re.sub(r'\bknp\b', 'kenapa', text)

    text = re.sub(r'\bkcil\b', 'kecil', text)

    

    text = re.sub(r'\bgmn\b', 'gimana', text)

    text = re.sub(r'\bgmna\b', 'gimana', text)

    

    text = re.sub(r'\btp\b', 'tapi', text)

    text = re.sub(r'\btq\b', 'thanks', text)

    text = re.sub(r'\btks\b', 'thanks', text)

    text = re.sub(r'\btlg\b', 'tolong', text)

    text = re.sub(r'\bgk\b', 'tidak', text)

    text = re.sub(r'\bgak\b', 'tidak', text)

    text = re.sub(r'\bgpp\b', 'tidak apa apa', text)

    text = re.sub(r'\bgapapa\b', 'tidak apa apa', text)

    text = re.sub(r'\bga\b', 'tidak', text)

    text = re.sub(r'\btgl\b', 'tanggal', text)

    text = re.sub(r'\btggl\b', 'tanggal', text)

    text = re.sub(r'\bgamau\b', 'tidak mau', text)

    

    text = re.sub(r'\bsy\b', 'saya', text)

    text = re.sub(r'\bsis\b', 'sister', text)

    text = re.sub(r'\bsdgkan\b', 'sedangkan', text)

    text = re.sub(r'\bmdh2n\b', 'semoga', text)

    text = re.sub(r'\bsmoga\b', 'semoga', text)

    text = re.sub(r'\bsmpai\b', 'sampai', text)

    text = re.sub(r'\bnympe\b', 'sampai', text)

    text = re.sub(r'\bdah\b', 'sudah', text)

    

    text = re.sub(r'\bberkali2\b', 'repeated', text)

    

    text = re.sub(r'\byg\b', 'yang', text)

    

    return text
%%time

df_train['review'] = df_train['review'].apply(recover_shortened_words)
from sklearn.model_selection import train_test_split



x = df_train['review']

y = df_train['rating']



#one_hot_encoded_label = pd.get_dummies(y)

#one_hot_encoded_label.head()



X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=42)
from sklearn.feature_extraction.text import CountVectorizer



vect = CountVectorizer(min_df=10, ngram_range=(1, 1))

X_train = vect.fit(X_train).transform(X_train) 

X_test = vect.transform(X_test)
from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression()

log_reg.fit(X_train, Y_train)
y_pred=log_reg.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))