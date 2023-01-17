import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import seaborn as sns

import string

import re

import time

import gc

import itertools



from tqdm import tqdm

from nltk import FreqDist

from nltk.corpus import stopwords

from wordcloud import WordCloud

from multiprocessing import Pool

from matplotlib_venn import venn2



plt.style.use('ggplot')

tqdm.pandas()
# Note: I'm using the custom train & test file I created, which contains the original cols + additional columns of POS tags 



train = pd.read_csv('../input/spacy-pos-tagging-12-workers/jigsaw_train_w_pos_tags.csv')

test = pd.read_csv('../input/spacy-pos-tagging-12-workers/jigsaw_test_w_pos_tags.csv')
toxic_subtypes = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']

identities = ['asian', 'atheist', 'bisexual',

       'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu',

       'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability',

       'jewish', 'latino', 'male', 'muslim', 'other_disability',

       'other_gender', 'other_race_or_ethnicity', 'other_religion',

       'other_sexual_orientation', 'physical_disability',

       'psychiatric_or_mental_illness', 'transgender', 'white']



selected_identities = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
plt.figure(figsize=(12,6))

plot = train.target.plot(kind='hist',bins=10)



ax = plot.axes



for p in ax.patches:

    ax.annotate(f'{p.get_height() * 100 / train.shape[0]:.2f}%',

                (p.get_x() + p.get_width() / 2., p.get_height()), 

                ha='center', 

                va='center', 

                fontsize=8, 

                color='black',

                xytext=(0,7), 

                textcoords='offset points')

plt.title('Target Distribution (Raw)')

plt.show()
# Referene: benchmark kernel for the competition



def convert_to_bool(df, col_name):

    df[col_name] = np.where(df[col_name] >= 0.5, True, False)



def convert_dataframe_to_bool(df):

    bool_df = df.copy()

    for col in ['target'] + selected_identities:

        convert_to_bool(bool_df, col)

    return bool_df



train = convert_dataframe_to_bool(train)
plt.figure(figsize=(12,6))

plot = sns.countplot(x='target', data=pd.DataFrame(train['target'].map({True:'Toxic', False:'Non-toxic'}), columns=['target']))



ax = plot.axes



for p in ax.patches:

    ax.annotate(f'{p.get_height() * 100 / train.shape[0]:.2f}%',

                (p.get_x() + p.get_width() / 2., p.get_height()), 

                ha='center', 

                va='center', 

                fontsize=8, 

                color='black',

                xytext=(0,7), 

                textcoords='offset points')

    

plt.title('Target Distribution (Binary)')

plt.show()
def preprocessing(titles_array, return_len = False):

    

    processed_array = []

    

    for title in tqdm(titles_array):

        

        # remove other non-alphabets symbols with space (i.e. keep only alphabets and whitespaces).

        processed = re.sub('[^a-z ]', '', title.lower())

        

        words = processed.split()

        

        if return_len:

            processed_array.append(len([word for word in words if word not in eng_stopwords]))

        else:

            processed_array.append(' '.join([word for word in words if word not in eng_stopwords]))

    

    return processed_array
eng_stopwords = []



for w in stopwords.words('english'):

    processed = re.sub('[^a-z ]', '', w.lower())

    eng_stopwords.append(processed)



eng_stopwords = set(eng_stopwords)
train['comment_text_len'] = train['comment_text'].progress_apply(len)

test['comment_text_len'] = train['comment_text'].progress_apply(len)



train['preprocessed_comment_len'] = preprocessing(train['comment_text'], return_len=True)

test['preprocessed_comment_len'] = preprocessing(test['comment_text'], return_len=True)
plt.figure(figsize=(20,6))

plt.subplot(121)

sns.distplot(train['comment_text_len'], kde=False, bins=150, label='Train Set', norm_hist=True)

sns.distplot(test['comment_text_len'], kde=False, bins=150, label='Test Set', norm_hist=True)

plt.legend()

plt.ylabel('Frequency')

plt.title('Comment Text Length (char level)')



plt.subplot(122)

sns.distplot(train['preprocessed_comment_len'], kde=False, bins=150, label='Train Set', norm_hist=True)

sns.distplot(test['preprocessed_comment_len'], kde=False, bins=150, label='Test Set', norm_hist=True)

plt.legend()

plt.ylabel('Frequency')

plt.title('Comment Text Length (word level, simple preprocessing)')

plt.show()
# Just wondering if how diligent our annotators are lol



plt.figure(figsize=(20,6))

plt.subplot(121)

sns.scatterplot(x='preprocessed_comment_len', y='toxicity_annotator_count',data=train)

plt.title('No. of Toxicity Annotators vs Comment Length (word level, simple pre-processing)')



plt.subplot(122)

sns.scatterplot(x='comment_text_len', y='toxicity_annotator_count',data=train)

plt.title('No. of Toxicity Annotators vs Comment Length (char level)')

plt.show()
"""

# Preprocessing based on https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing/notebook

# Currently not used



contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}



def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text



def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    

    for p in punct:

        text = text.replace(p, f' {p} ')

    

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last

    for s in specials:

        text = text.replace(s, specials[s])

    

    return text



def correct_spelling(x, dic):

    for word in dic.keys():

        x = x.replace(word, dic[word])

    return x



def clean_text(text):

    text = text.lower()

    text = clean_contractions(text, contraction_mapping)

    text = clean_special_chars(text, punct, punct_mapping)

    text = correct_spelling(text, mispell_dict)

    return text

    

n_partitions = 24

n_workers = 8



def parallelize_dataframe(df, func):

    df_split = np.array_split(df, n_partitions)

    pool = Pool(n_workers)

    df = pd.concat(pool.map(func, df_split))

    pool.close()

    pool.join()

    return df



def text_processing(data):

    data['processed_text'] = data['comment_text'].apply(clean_text)

    gc.collect()

    return data



train = parallelize_dataframe(train, text_processing)



"""
plt.figure(figsize=(20,6))

plt.subplot(121)

sns.kdeplot(train['toxicity_annotator_count'], color='red')

plt.title('Toxicity Annotator Distribution')

plt.subplot(122)

sns.kdeplot(train['identity_annotator_count'], color='blue')

plt.title('Identity Annotator Distribution')

plt.show()
for identity in selected_identities:

    counts = train[identity].sum()

    percentage = train[identity].sum() / train[identity].count() * 100

    print(f'{identity:<30}: {percentage:.2f}% , {counts}')
%%time

train['non_zero_identity_counts'] = np.count_nonzero(train[identities], axis=1)

train.loc[train['identity_annotator_count'] == 0, 'non_zero_identity_counts'] = np.NaN



train['non_zero_selected_identity_counts'] = np.count_nonzero(train[selected_identities], axis=1)

train.loc[train['identity_annotator_count'] == 0, 'non_zero_selected_identity_counts'] = np.NaN
plt.figure(figsize=(16,6))



non_zero_selected_identity_counts = ~train['non_zero_selected_identity_counts'].isna(),'non_zero_selected_identity_counts'



plot = sns.countplot(train.loc[non_zero_selected_identity_counts])

ax = plot.axes



y_lim = 0



for p in ax.patches:

    

    if p.get_height() > y_lim:

        y_lim = p.get_height()

    

    ax.annotate(f'{p.get_height() * 100 / train.loc[non_zero_selected_identity_counts].shape[0]:.3f}%\n({p.get_height()})',

                (p.get_x() + p.get_width() / 2., p.get_height()), 

                ha='center', 

                va='center', 

                fontsize=11, 

                color='black',

                xytext=(0,15), 

                textcoords='offset points')



plt.ylim((0,round(y_lim*1.1)))

plt.title('Number of Non-Zero Selected Identities for Each Sample (identity_annotator > 0)')



plt.show()
# Check out how the selected identities are related to each other



selected_identity_corr = train.loc[~train['non_zero_selected_identity_counts'].isna(), selected_identities].corr()
mask = np.zeros_like(selected_identity_corr,dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



plt.style.use('default')



plt.figure(figsize=(6,6))

sns.heatmap(selected_identity_corr, 

            cmap = sns.diverging_palette(220, 10, as_cmap=True), 

            center=0, mask=mask, vmin=-1, vmax=1, annot=True, fmt='.2f')

plt.show()
pair_identity_dict = dict(train.loc[~train['non_zero_selected_identity_counts'].isna(), selected_identities].sum().reset_index().values)

pair_keys = list(pair_identity_dict.keys())

pair_values = list(pair_identity_dict.values())
def venn_diagram_subplot(id1, id2):

    overlap = train.loc[~train['non_zero_selected_identity_counts'].isna() & \

                                          (train[id1] == True) & \

                                          (train[id2] == True),\

                                          id1].count()

    v = venn2(subsets = (pair_identity_dict[id1], pair_identity_dict[id2], overlap), set_labels = (id1[0:10], id2[0:10]))



    return v
with plt.rc_context({"axes.edgecolor": 'red', "axes.linewidth": 3, "font.size":'14'}):

    plt.figure(figsize=(20,40))

    for i, (id1,id2) in enumerate(itertools.combinations(list(pair_identity_dict.keys()), 2)):

        plt.subplot(9,4,i+1)

        venn_diagram_subplot(id1,id2)

        if ((id1,id2) == ('male','female')) or ((id1,id2) == ('black','white')):

            plt.gca().set_axis_on()





    plt.suptitle(f'Overlap of Identities', fontsize=25, fontweight='heavy')

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    plt.show()
"""

identity = train.loc[~train['non_zero_selected_identity_counts'].isna(),'black']



plot = sns.countplot(identity)



ax = plot.axes

y_lim = 0



for p in ax.patches:

    if p.get_height() > y_lim:

        y_lim = p.get_height()



    ax.annotate(f'{p.get_height() * 100 / identity.shape[0]:.2f}%',

                (p.get_x() + p.get_width() / 2., p.get_height()), 

                ha='center', 

                va='center', 

                fontsize=11, 

                color='black',

                xytext=(0,7), 

                textcoords='offset points')

    

plt.ylim((0,round(y_lim*1.1)))

plt.show()

"""
def get_pos_neg_string(identity, pos_query_criteria, neg_query_criteria, return_cols = ['adj','noun','propn','verb']):

    

    pos_word_list = [row[0] for row in train.loc[pos_query_criteria,return_cols].values if type(row[0]) != float]

    neg_word_list = [row[0] for row in train.loc[neg_query_criteria,return_cols].values if type(row[0]) != float]

    pos_string = ' '.join(pos_word_list)

    neg_string = ' '.join(neg_word_list)

    pos_string = re.sub('[^a-zA-Z ]', '', pos_string)

    neg_string = re.sub('[^a-zA-Z ]', '', neg_string)

    

    return pos_string, neg_string
def get_word_freq_and_plot(identity, most_freq_word_count=10):



    # Set the criteria for dataframe query

    pos_query_criteria = ~train['non_zero_selected_identity_counts'].isna() & train[identity] & train['target']

    neg_query_criteria = ~train['non_zero_selected_identity_counts'].isna() & train[identity] & ~train['target']



    # get the concatenated string for both positive (toxic) and negative (non-toxic) samples for an identity

    pos_string, neg_string = get_pos_neg_string(identity, pos_query_criteria, neg_query_criteria)

    

    pos_freq_dist = FreqDist([word for word in pos_string.split()])

    neg_freq_dist = FreqDist([word for word in neg_string.split()])

    

    pos_words, pos_word_count = list(zip(*pos_freq_dist.most_common(most_freq_word_count)))



    pos_row_count = train.loc[pos_query_criteria,'target'].shape[0]

    neg_row_count = train.loc[neg_query_criteria,'target'].shape[0]



    # as negative samples are much larger in population, there is a need to normalize them to the positive sample size

    neg_word_count_normalized = [int(neg_freq_dist.get(w) * pos_row_count / neg_row_count) \

                                 if neg_freq_dist.get(w) != None else 0 \

                                 for w in pos_words]



    words_freq_df = pd.DataFrame(list(zip(pos_words,pos_word_count,neg_word_count_normalized)), 

                                 columns=['vocab','pos_freq','neg_freq_norm'])

    

    toxic_cloud = WordCloud(background_color='Black', 

                              colormap='Paired', 

                              width=600, 

                              height=700, 

                              random_state=123).generate_from_frequencies(pos_freq_dist)



    non_toxic_cloud = WordCloud(background_color='White', 

                              #colormap='Paired', 

                              width=600, 

                              height=700, 

                              random_state=123).generate_from_frequencies(neg_freq_dist)



    plt.figure(figsize=(18,8))

    # Word Cloud plot

    plt.subplot(131)

    plt.imshow(toxic_cloud,interpolation='bilinear')

    plt.axis('off')

    plt.title('Toxic', fontsize=20)

    

    plt.subplot(132)

    plt.imshow(non_toxic_cloud,interpolation='bilinear')

    plt.axis('off')

    plt.title('Non-toxic', fontsize=20)

    

    # Line plot of the term frequencies

    plt.subplot(133)

    sns.lineplot(x='vocab',y='pos_freq',data=words_freq_df, sort=False, marker='o', label='Toxic')

    sns.lineplot(x='vocab',y='neg_freq_norm',data=words_freq_df, sort=False, marker='o', label='Non-toxic\n(normalized)', alpha=0.8)

    plt.legend()

    plt.xticks(rotation=90)

    plt.grid(b=True, which='major', axis='x', linestyle='--')

    plt.ylabel('Term Frequency')

    plt.title(f'FreqDist (Top {most_freq_word_count} words)', fontsize=20)

    

    plt.suptitle(f'Identity : {str.capitalize(identity)}, {pos_row_count/(pos_row_count + neg_row_count)*100:.2f}% toxic', 

                 fontsize=25, fontweight='heavy', ha='center')

    plt.tight_layout(rect=[0,0,1,0.93])

    plt.show()
for identity in selected_identities:

    get_word_freq_and_plot(identity)