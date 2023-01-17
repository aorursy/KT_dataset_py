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
#from tqdm import tqdm



df = pd.read_csv( '../input/clickbait-news-detection/train.csv')



dft = pd.read_csv( '../input/clickbait-news-detection/test.csv')







#tqdm.pandas(desc="my bar!")

#df.progress_apply(lambda x: x)



#pd.set_option('display.max_rows',24871)

df



dft
print(df.shape)

print(dft.shape)
df['title'].fillna('missing', inplace=True)

dft['title'].fillna('missing', inplace=True)







df

dft
import re

from string import punctuation



def process_text2(title):

    

    result = title.replace('/','').replace('\n','')

    result = re.sub(r'[0-9]+','number', result)   # we are substituting all kinds of no. with word number

    result = re.sub(r'(\w)(\1{2,})', r'\1', result)  # \w matches one word/non word character

    result = re.sub(r'(?x)\b(?=\w*\d)\w+\s*', '', result)

    

    result = ''.join(word for word in result if word not in punctuation)  # removes all characters such as "!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"

    result = re.sub(r' +', ' ', result).lower().strip()

    return result
#removing the stopwords

from nltk.corpus import stopwords



stop = stopwords.words("english")



def cnt_stopwords(title):

    

    result1 = title.split()

    num1 =  len([word for word in result1 if word in stop])

    

    return num1
contractions = ['tis', 'aint', 'amnt', 'arent', 'cant', 'couldve', 'couldnt', 'couldntve',

                'didnt', 'doesnt', 'dont', 'gonna', 'gotta', 'hadnt', 'hadntve', 'hasnt',

                'havent', 'hed', 'hednt', 'hedve', 'hell', 'hes', 'hesnt', 'howd', 'howll',

                'hows', 'id', 'idnt', 'idntve', 'idve', 'ill', 'im', 'ive', 'ivent', 'isnt',

                'itd', 'itdnt', 'itdntve', 'itdve', 'itll', 'its', 'itsnt', 'mightnt',

                'mightve', 'mustnt', 'mustntve', 'mustve', 'neednt', 'oclock', 'ol', 'oughtnt',

                'shant', 'shed', 'shednt', 'shedntve', 'shedve', 'shell', 'shes', 'shouldve',

                'shouldnt', 'shouldntve', 'somebodydve', 'somebodydntve', 'somebodys',

                'someoned', 'someonednt', 'someonedntve', 'someonedve', 'someonell', 'someones',

                'somethingd', 'somethingdnt', 'somethingdntve', 'somethingdve', 'somethingll',

                'somethings', 'thatll', 'thats', 'thatd', 'thered', 'therednt', 'theredntve',

                'theredve', 'therere', 'theres', 'theyd', 'theydnt', 'theydntve', 'theydve',

                'theydvent', 'theyll', 'theyontve', 'theyre', 'theyve', 'theyvent', 'wasnt',

                'wed', 'wedve', 'wednt', 'wedntve', 'well', 'wontve', 'were', 'weve', 'werent',

                'whatd', 'whatll', 'whatre', 'whats', 'whatve', 'whens', 'whered', 'wheres',

                'whereve', 'whod', 'whodve', 'wholl', 'whore', 'whos', 'whove', 'whyd', 'whyre',

                'whys', 'wont', 'wontve', 'wouldve', 'wouldnt', 'wouldntve', 'yall', 'yalldve',

                'yalldntve', 'yallll', 'yallont', 'yallllve', 'yallre', 'yallllvent', 'yaint',

                'youd', 'youdve', 'youll', 'youre', 'yourent', 'youve', 'youvent']



def cnt_contract(title):

    

    result2 = title.split()

    num2 = len([word for word in result2 if word in contractions])

    return num2
question_words = ['who', 'whos', 'whose', 'what', 'whats', 'whatre', 'when', 'whenre', 'whens', 'couldnt',

        'where', 'wheres', 'whered', 'why', 'whys', 'can', 'cant', 'could', 'will', 'would', 'is',

        'isnt', 'should', 'shouldnt', 'you', 'your', 'youre', 'youll', 'youd', 'here', 'heres',

        'how', 'hows', 'howd', 'this', 'are', 'arent', 'which', 'does', 'doesnt']



def question_word(title):

    

    result3 = title.lower().split()

    

    if result3[0] in question_words:

        return 1

    else:

        return 0

    #return result3    
def pos_tags(title):

    

    result4 = title.split()

    

    non_stop = [word for word in result4 if word not in stopwords.words("english")]

    pos = [part[1] for part in nltk.pos_tag(non_stop)]

    pos = " ".join(pos)

    return pos
import nltk



# progress bar

from tqdm import tqdm_notebook,tqdm



# instantiate

tqdm.pandas(tqdm_notebook)



df['processed_headline']     = df['title'].progress_apply(process_text2)

df['question'] = df['title'].progress_apply(question_word)



df['num_words']       = df['title'].progress_apply(lambda x: len(x.split()))

df['part_speech']     = df['title'].progress_apply(pos_tags)

df['num_contract']    = df['title'].progress_apply(cnt_contract)

df['num_stop_words']  = df['title'].progress_apply(cnt_stopwords)

df['stop_word_ratio'] = df['num_stop_words']/df['num_words']

df['contract_ratio']  = df['num_contract']/df['num_words']
from tqdm import tqdm_notebook,tqdm



# instantiate

tqdm.pandas(tqdm_notebook)



dft['processed_headline']     = dft['title'].progress_apply(process_text2)

dft['question'] = dft['title'].progress_apply(question_word)



dft['num_words']       = dft['title'].progress_apply(lambda x: len(x.split()))

dft['part_speech']     = dft['title'].progress_apply(pos_tags)

dft['num_contract']    = dft['title'].progress_apply(cnt_contract)

dft['num_stop_words']  = dft['title'].progress_apply(cnt_stopwords)

dft['stop_word_ratio'] = dft['num_stop_words']/dft['num_words']

dft['contract_ratio']  = dft['num_contract']/dft['num_words']
df
df = df.drop(columns = ['num_contract','num_stop_words'])



df
dft = dft.drop(columns = ['num_contract','num_stop_words'])



dft
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,5),use_idf=1, smooth_idf=1, sublinear_tf=1)



X_train_headline = tfidf.fit_transform(df['processed_headline'])#1

X_test_headline = tfidf.transform(dft['processed_headline'])











print(X_train_headline.shape)

print(X_train_headline.ndim)



print(X_test_headline.shape)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import StandardScaler

cv = CountVectorizer()

sc = StandardScaler(with_mean = False)



X_train_pos = cv.fit_transform(df['part_speech'])

X_train_pos_sc = sc.fit_transform(X_train_pos)   #2



X_test_pos = cv.transform(dft['part_speech'])

X_test_pos_sc = sc.transform(X_test_pos)



print(X_test_pos_sc.shape)



print(X_train_pos_sc.shape)

print(X_train_pos_sc.ndim)
X_train_val = df.drop( columns = ['title','label','processed_headline','part_speech','text']).values

X_test_val = dft.drop( columns = ['title','processed_headline','part_speech','text']).values





sc = StandardScaler()



X_train_val_sc = sc.fit_transform(X_train_val) #3

X_test_val_sc = sc.transform(X_test_val)





print(X_train_val_sc.shape)

print(X_train_val_sc.ndim)



print(X_test_val_sc.shape)
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder



label_encoder = LabelEncoder()

Y_train = label_encoder.fit_transform(df[['label']])



#onehot_encoder = OneHotEncoder(sparse=False)

#Y_train = onehot_encoder.fit_transform(df[['label']])



Y_train



print(Y_train.shape)
from scipy import sparse



X_train = sparse.hstack([X_train_val_sc, X_train_headline, X_train_pos_sc]).tocsr()

X_test = sparse.hstack([X_test_val_sc, X_test_headline, X_test_pos_sc]).tocsr()





print(X_train.shape)

print(X_test.shape)
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression





#param_grid  = [{'C': np.linspace(90,100,20)}]

#grid_cv = GridSearchCV(LogisticRegression(), param_grid, scoring='accuracy', cv=5, verbose=1)



#grid_cv.fit(X_train, Y_train)



#print(grid_cv.best_params_)

#print(grid_cv.best_score_)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



model = LogisticRegression(penalty='l2', C=93.684210526315795)

model = model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
Y_pred
Y_predres = []

for x in Y_pred:

    if x == 0 :

        x = 'clickbait'

        

    if x == 1 :

        x = 'news'

    if x == 2:

        x = 'other'

    Y_predres.append(x)

    

Y_predres = np.array(Y_predres)



Y_predres
Y_pred
dft.id
my_submission = pd.DataFrame({'id': dft.id, 'label': Y_predres })



my_submission.to_csv('submission.csv', index=False)