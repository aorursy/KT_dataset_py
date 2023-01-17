

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os; os.environ['OMP_NUM_THREADS'] = '1'

import scipy

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from collections import defaultdict

from multiprocessing import Pool

from gensim.models import Word2Vec



from sklearn.model_selection import StratifiedKFold

import gc

from functools import wraps

import torch.nn.functional as F

from torch.autograd import Variable

import pandas as pd

import numpy as np

from multiprocessing import Pool



from collections import OrderedDict

from collections import defaultdict

from string import digits



import nltk

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet, stopwords

from nltk.tokenize import word_tokenize



from sklearn.metrics import f1_score

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler



import torch

import torch as t

import torch.nn as nn

import torch.nn.functional as F

import torch.nn as nn

import torch.utils.checkpoint as checkpoint

import torch.utils.data

import torch.nn.functional as F

from torch.nn import Parameter

from functools import wraps

from torch.autograd import Variable



import re

import random

import os

import time

import gc

import shutil



from tqdm import tqdm



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



from multiprocessing.pool import ThreadPool

from nltk.corpus import wordnet, stopwords

from sklearn import metrics

from contextlib import contextmanager

import string

import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import csr_matrix, hstack

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation, BatchNormalization, PReLU

from keras.initializers import he_uniform

from keras.layers import Conv1D

from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.optimizers import Adam, SGD

from keras.models import Model

from tqdm import tqdm_notebook as tqdm

import time

import pandas as pd

import numpy as np

from multiprocessing import Pool

from nltk.stem import WordNetLemmatizer

from itertools import combinations

from sklearn.feature_extraction.text import CountVectorizer

LOCAL = False

from sklearn.linear_model import LogisticRegression, Ridge

from keras import backend as K

import scipy

from gensim import utils



seed =1018

if LOCAL: 

    FILE_DIR = './'

else: 

    FILE_DIR = '../input'

@contextmanager

def timer(task_name="timer"):

    # a timer cm from https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s

    print("----{} started".format(task_name))

    t0 = time.time()

    yield

    print("----{} done in {:.0f} seconds".format(task_name, time.time() - t0))

# Any results you write to the current directory are saved as output.

train = pd.read_csv(f'{FILE_DIR}/train.csv')

test = pd.read_csv(f'{FILE_DIR}/test.csv')



def do_rnns(dfs):

    if dfs[2]:

        split = -1

        train_df, test_df = dfs[0], dfs[1]



        print(train_df.shape, test_df.shape)



        test_idx = list(test_df.qid.values)



        test_df.drop('qid', axis=1, inplace=True)



        training_labels = train_df['target'].values

        if split>0:

            testing_labels = test_df['target'].values



        # PREPROCESS-----#

        def get_special_feats(df):

            df['question_text'] = df['question_text'].apply(lambda x:str(x))

            df['total_length'] = df['question_text'].apply(len)

            df['capitals'] = df['question_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))

            df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']), axis=1)

            df['num_words'] = df.question_text.str.count('\S+')

            df['num_unique_words'] = df['question_text'].apply(lambda comment: len(set(w for w in comment.split())))

            df['words_vs_unique'] = df['num_unique_words'] / df['num_words']

            features = df[['caps_vs_length', 'words_vs_unique']].fillna(0).values

            gc.collect()

            return features

        print('START SPECIAL FEATS')

        train_features = get_special_feats(train_df)

        test_features = get_special_feats(test_df)

        print('END SPECIAL FEATS')

        ss = StandardScaler()

        ss.fit(np.vstack((train_features, test_features)))

        train_features = ss.transform(train_features)

        test_features = ss.transform(test_features)

        print('SCALE SPECIAL FEATS')

        punctuations = [

            ',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#',

            '*', '+', '\\', '•',  '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™',

            '›', '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢',

            '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', 

            '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅',

            '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬',

            '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', '\xa0', '高', '端', '大', '气', '上', '档', '次', '_', '½', 'π', '#', 

        '小', '鹿', '乱', '撞', '成', '语', 'ë', 'à', 'ç', '@', 'ü', 'č', 'ć', 'ž', 'đ', '°', 'द', 'े', 'श', '्', 'र', 'ो', 'ह', 'ि', 'प', 'स', 'थ', 'त', 'न', 'व', 'ा', 'ल', 'ं', '林', '彪', '€', '\u200b', '˚', 'ö', '~', '—', '越', '人', 'च', 'म', 'क', 

        'ु', 'य', 'ी', 'ê', 'ă', 'ễ', '∞', '抗', '日', '神', '剧', '，', '\uf02d', '–', 'ご', 'め', 'な', 'さ', 'い', 'す', 'み', 'ま', 'せ', 'ん', 'ó', 'è', '£', '¡', 'ś', '≤', '¿', 'λ', '魔', '法', '师', '）', 'ğ', 'ñ', 'ř', '그', '자', '식', '멀', 

        '쩡', '다', '인', '공', '호', '흡', '데', '혀', '밀', '어', '넣', '는', '거', '보', '니', 'ǒ', 'ú', '️', 'ش', 'ه', 'ا', 'د', 'ة', 'ل', 'ت', 'َ', 'ع', 'م', 'ّ', 'ق', 'ِ', 'ف', 'ي', 'ب', 'ح', 'ْ', 'ث', '³', '饭', '可', '以', '吃', '话', '不', '讲', 

        '∈', 'ℝ', '爾', '汝', '文', '言', '∀', '禮', 'इ', 'ब', 'छ', 'ड', '़', 'ʒ', '有', '「', '寧', '錯', '殺', '一', '千', '絕', '放', '過', '」', '之', '勢', '㏒', '㏑', 'ू', 'â', 'ω', 'ą', 'ō', '精', '杯', 'í', '生', '懸', '命', 'ਨ', 'ਾ', 'ਮ', 'ੁ', 

        '₁', '₂', 'ϵ', 'ä', 'к', 'ɾ', '\ufeff', 'ã', '©', '\x9d', 'ū', '™', '＝', 'ù', 'ɪ', 'ŋ', 'خ', 'ر', 'س', 'ن', 'ḵ', 'ā']



        def clean_text( text):

            text = str(text)

            for p in punctuations:

                if p in text:

                    text = text.replace(p, f' {p} ')

            return(text)  

        print('CLEAN TEXT')

        train_df['question_text'] = train_df['question_text'].apply(lambda x: clean_text(x))

        test_df['question_text'] = test_df['question_text'].apply(lambda x: clean_text(x))

        print('FINISH CLEANING TEXT')

        contractions_dict = { 

            "ain't": "am not",

            "aren't": "are not",

            "can't": "cannot",

            "can't've": "cannot have",

            "'cause": "because",

            "could've": "could have",

            "couldn't": "could not",

            "couldn't've": "could not have",

            "didn't": "did not",

            "doesn't": "does not",

            "don't": "do not",

            "hadn't": "had not",

            "hadn't've": "had not have",

            "hasn't": "has not",

            "haven't": "have not",

            "he'd": "he had",

            "he'd've": "he would have",

            "he'll": "he shall",

            "he'll've": "he shall have",

            "he's": "he has",

            "how'd": "how did",

            "how'd'y": "how do you",

            "how'll": "how will",

            "how's": "how has",

            "I'd": "I had",

            "I'd've": "I would have",

            "I'll": "I will",

            "I'll've": "I will have",

            "I'm": "I am",

            "I've": "I have",

            "isn't": "is not",

            "it'd": "it would",

            "it'd've": "it would have",

            "it'll": "it will",

            "it'll've": "it will have",

            "it's": "it is",

            "let's": "let us",

            "ma'am": "madam",

            "mayn't": "may not",

            "might've": "might have",

            "mightn't": "might not",

            "mightn't've": "might not have",

            "must've": "must have",

            "mustn't": "must not",

            "mustn't've": "must not have",

            "needn't": "need not",

            "needn't've": "need not have",

            "o'clock": "of the clock",

            "oughtn't": "ought not",

            "oughtn't've": "ought not have",

            "shan't": "shall not",

            "sha'n't": "shall not",

            "shan't've": "shall not have",

            "she'd": "she would",

            "she'd've": "she would have",

            "she'll": "she will",

            "she'll've": "she will have",

            "she's": "she is",

            "should've": "should have",

            "shouldn't": "should not",

            "shouldn't've": "should not have",

            "so've": "so have",

            "so's": "so is",

            "that'd": "that had",

            "that'd've": "that would have",

            "that's": "that is",

            "there'd": "there would",

            "there'd've": "there would have",

            "there's": "there is",

            "they'd": "they would",

            "they'd've": "they would have",

            "they'll": "they will",

            "they'll've": "they will have",

            "they're": "they are",

            "they've": "they have",

            "to've": "to have",

            "wasn't": "was not",

            "we'd": "we would",

            "we'd've": "we would have",

            "we'll": "we will",

            "we'll've": "we will have",

            "we're": "we are",

            "we've": "we have",

            "weren't": "were not",

            "what'll": "what will",

            "what'll've": "what will have",

            "what're": "what are",

            "what's": "what is",

            "what've": "what have",

            "when's": "when is",

            "when've": "when have",

            "where'd": "where did",

            "where's": "where is",

            "where've": "where have",

            "who'll": "who will",

            "who'll've": "who will have",

            "who's": "who is",

            "who've": "who have",

            "why's": "why is",

            "why've": "why have",

            "will've": "will have",

            "won't": "will not",

            "won't've": "will not have",

            "would've": "would have",

            "wouldn't": "would not",

            "wouldn't've": "would not have",

            "y'all": "you all",

            "y'all'd": "you all would",

            "y'all'd've": "you all would have",

            "y'all're": "you all are",

            "y'all've": "you all have",

            "you'd": "you would",

            "you'd've": "you would have",

            "you'll": " you will",

            "you'll've": "you will have",

            "you're": "you are",

            "you've": "you have",

            '隶书': 'Lishu', 'तपस्': 'Tapse', 'ให้': 'give', '宗教官': 'Religious officer', '没有神一样的对手': 'No god-like opponent', 'יהודיות': 'Jewish women', '我明明是中国人': 'I am obviously Chinese.', 'お早う': 'Good morning', 'मेरे': 'my', 'পরীক্ষা': 'Test', 'נעכטן': 'Yesterday', '中华民国': 'Republic of China', 'तकनीकी': 'Technical', 'यहाँ': 'here', '奇兵隊': 'Qibing', 'かﾟ': 'Or', '知识分子': 'Intellectuals', 'ごめなさい': "I'm sorry", '하기를': 'To', 'गाढवाचे': 'Donkey', 'इबादत': 'Worship', '千金': 'Thousand gold', 'ельзи': 'elzhi', '低端人口': 'Low-end population', '맛저': 'Mazur', 'दानि': 'Danias', 'юродство': 'foolishness', 'τον': 'him', '素质': 'Quality', '王晓菲': 'Wang Xiaofei', 'माहेर': 'Maher', 'لمصارى': 'For my advice', '送客！': 'see a visitor out!', '然而': 'however', 'जिग्यासा': 'Jigyaasa', '都市': 'city', 'ஜோடி': 'Pair', 'لاني': 'See', 'пельмени': 'dumplings', '请不要误会': "Please don't misunderstand", 'люис': 'Lewis', '不送！': 'Do not send!', 'के': 'Of', 'मस्का': 'Mascara', 'вoyѕ': 'voyce', 'बुन्ना': 'Bunna', '战略支援部队': 'Strategic support force', '平埔族': 'Pingpu', 'فهمت؟': 'I see?', 'कयामत': 'Apocalypse', 'ức': 'memory', 'ᗰeᑎ': 'ᗰe ᑎ', 'सेकना': 'Sexting', 'धकेलना': 'Shove', 'পারি': 'We can', 'مقام': 'Official', 'बेसति': 'Baseless', '歪果仁研究协会': 'Hazelnut Research Association', '短信': 'SMS', 'всегда': 'is always', '修身': 'Slim fit', 'إنَّ': 'that', '不是民国': 'Not the Republic of China', '写的好': 'Written well', 'õhtupoolik': 'afternoon', 'तालीन': 'Training', 'और': 'And', 'щит': 'shield', 'ᗩtteᑎtiᐯe': 'ᗩtte ᑎ ti ᐯ e', 'اسکوپیه': 'Script', 'çomar': 'fido', '和製英語': 'Japanglish', '吊着命': 'Hanging', 'много': 'much', 'समुराय': 'Samurai', 'भटकना': 'Wander', 'مقاومة': 'resistance', '싱관없어': 'I have no one.', '修身養性': 'Self-cultivation', 'मटका': 'pot', 'θιοψʼ': 'thiós', 'ㅎㅎㅎㅎ': 'Hehe', 'تساوى': 'Equal', 'बाट': 'From', '不地道啊。': 'Not authentic.', 'контакт': 'contact', '이런것도': 'This', 'तै': 'The', 'मेल': 'similarity', 'álvarez': 'Alvarez', 'नुक्स': 'Damage', '口訣': 'Mouth', 'масло': 'butter', 'परम्परा': 'Tradition', '学会': 'learn', 'کردن': 'Make up', 'öd': 'bile', 'टशन': 'Tashan', 'つらやましす': 'I am cheating', 'чего': 'what', '为什么总有人能在shoplex里免费获取iphone等重金商品？': 'Why do people always get free access to iphone and other heavy commodities in shoplex?', 'श्री': 'Mr', 'प्रेषक': 'Sender', 'خواندن': 'Read', 'बदलेंगे': 'Will change', 'बीएड': 'Breed', 'अदा': 'Paid', 'फैलाना': 'spread', 'ബുദ്ധിമാനായ': 'Intelligent', '谢谢六佬': 'Thank you, Liu Wei', 'উপস্থাপন': 'Present', 'بكلها': 'All of them', 'जाए': 'Go', '无可挑剔': 'Impeccable', 'आना': 'come', '太阴': 'lunar', 'القط': 'The cat', '있네': 'It is.', 'कर्म': 'Karma', 'आड़': 'Shroud', 'įit': 'IIT', 'जशने': 'Jashnay', 'से': 'From', 'åkesson': 'Åkesson', '乳名': 'Milk name', '我看起来像韩国人吗': 'Do I look like a Korean?', 'тнan': 'tn', 'العظام': 'Bones', '暹罗': 'Siam', '小肥羊鍋店': 'Little fat sheep pot shop', 'করেন': 'Do it', 'हरामी': 'bastard', 'डाले': 'Cast', 'استراحة': 'Break', '射道': 'Shooting', 'επιστήμη': 'science', 'χράομαι': "I'm glad", '脚踏车': 'bicycle', 'हिलना': 'To move', 'đa': 'multi', '標點符號': 'Punctuation', 'मैं': 'I', '不能化生水谷精微': "Can't transform the water valley", 'निवाला': 'Boar', 'दर्शनाभिलाषी': 'Spectacular', 'বাছাই': 'Picking', 'どくぜんてき': 'Dokedari', 'पीवें।': 'Pieve.', 'ʻoumuamua': 'Priority', 'में': 'In', 'चलाऊ': 'Run', 'निपटाना': 'To settle', 'ごはん': 'rice', 'चार्ज': 'Charge', 'शिथिल': 'Loosely', 'ястребиная': 'yastrebina', 'ложись': 'lie down', '李银河': 'Li Yinhe', 'へも': 'Also', 'हुलिया': 'Hulia', 'ऊब': 'Bored', 'छाछ': 'Buttermilk', 'عن': 'About', 'नहीं': 'No', 'जीव': 'creatures', 'जेहाद': 'Jehad', 'νερό': 'water', '열여섯': 'sixteen', 'मार': 'Kill', '早就': 'Long ago', '《齐天大圣》for': '"Qi Tian Da Sheng" for', 'الجنس': 'Sex', 'مع': 'With', 'रंगरलियाँ': 'Color palettes', 'जोल': 'Jol', 'बुद्धी': 'Intelligence', '罗思成': 'Luo Sicheng', '独善的': 'Self-righteousness', 'धर्मः': 'Religion', '中国大陆的同胞们': 'Chinese compatriots', '愛してない': 'I do not love you', '日本語が上手ですね': 'Japanese is good', 'ओठ': 'Lips', '如果你希望表達你的觀點': 'If you want to express your opinion', 'देशवाशी': 'Countryman', 'καί': 'and', '阮铭': 'Yu Ming', '跑步跑': 'Running', 'हो': 'Ho', '「褒められる」': '"Praised"', 'నా': 'My', '\x10new': '?new', 'ゆいがどくそん': 'A graduate student', '白语': 'White language', 'वह': 'She', '白い巨塔': 'White big tower', '로리이고': 'Lori.', 'परि': 'Circumcision', 'æj': 'Aw', 'انضيع': 'Lost', '平民苗字必称義務令': 'Civilian Miaozi must be called an obligation', 'ਸਿੰਘ': 'Singh', 'уже': 'already', 'đầu': 'head', '雨热同期': 'Rain and heat', 'घुलना': 'Dissolve', 'мис': 'Miss', 'て下さいませんか': 'Could it be?', '猫头鹰': 'owl', 'चढ़': 'Climbing', '漢音': 'Hanyin', 'यही': 'This only', '只有猪一样的队友': 'Only pig-like teammates', 'übersoldaten': 'about soldiers', 'αθ': 'Ath', 'ουδεις': 'no', '党国合一': 'Party and country', 'रेड': 'Red', 'ढोलना': 'Drift', 'शाक': 'Shake', '같이': 'together', '攻克': 'capture', 'özalan': 'exclusive', 'काम': 'work', 'चाहना': 'Wish', '坚持一个中国原则': 'Adhere to the one-China principle', '배우고': 'Learn', '柴棍': 'Firewood stick', 'उसे': 'To him', 'на': 'on', '無手': 'No hands', 'čvrsnica': 'Čvrsnica', 'ज़ार': 'Tsar', 'ˆo': 'O', '後宮甄嬛傳': 'Harem of the harem', '意音文字': 'Sound character', 'बहारा': 'Bahara', 'てみる': 'Try', '老铁': 'Old iron', '野比のび太': 'Nobita Nobita', 'याद': 'remember', 'پیشی': 'Surpass', 'توقعك': 'Your expectation', 'はたち': 'Slender', 'فزت': 'I won', '伊藤': 'Ito', 'कलेजे': 'Liver', 'αγεν': 'ruin', 'ìmprovement': 'Improvement', 'では': 'Then.', 'பார்ப்பது': 'Viewing', '只留清气满乾坤': 'Only stay clear and full of energy', '이정도쯤': 'About this', 'すてん': 'Sponge', 'ما': 'What', '海南人の日本': "Hainan people's Japan", '小鹿乱撞': 'very excited', 'ῥωμαῖοι': 'ῥomaşii', 'वारी': 'Vary', 'भोज': 'Feast', '陈庚': 'Chen Geng', 'कट्टरवादि': 'Fanatic', '凌霸': 'Lingba', 'पार': 'The cross', 'कुचलना': 'To crush', 'रहा': 'Stayed', 'हम': 'We', 'бойся': 'be afraid', '大刀': 'Large sword', 'ন্ন': 'Done', '汽车': 'car', 'কে': 'Who', 'الکعبه': 'Alkaebe', '网络安全法': 'Cybersecurity law', 'नेपाली': 'Nepali', 'পদের': 'Position', '\x92t': '??t', 'रास': 'Ras', 'لكل': 'for every', 'शूद्राणां': 'Shudran', '此问题只是针对外国友人的一次统计': 'This question is only a statistic for foreign friends.', 'ឃើញឯកសារអំពីប្រវត្តិស្ត្រនៃប្រាសាទអង្គរវត្ត': 'See the history of Angkor Wat', 'спасибо': 'thank', 'رب': 'God', 'वजूद': 'Non existent', 'पकड़': 'Hold', 'बरहा': 'Baraha', '雾草': 'Fog grass', 'мощный': 'powerful', 'বৃদ্ধাশ্রম': 'Old age', 'आई': 'Mother', 'खड़ी': 'Steep', '\x1aaùõ\x8d': '? aùõ ??', 'āto': 'standard', '在冰箱裡': 'in the refrigerator', 'đời': 'life', 'लड़का': 'The boy', 'τὸ': 'the', 'مدرسان': 'Instructors', 'பறை': 'Drum', 'índia': 'India', 'दिया': 'Gave', '小篆': 'Xiao Yan', '唐樂': 'Tang Le', '米国': 'USA', '文言文': 'Classical Chinese', 'उवाच': 'Uwach', 'هدف': 'Target', 'घेरा': 'Cordon', 'झूम': 'Zoom', 'εσσα': 'all the same', '和製漢語': 'And making Chinese', 'ऐलोपैथिक': 'Allopathic', 'создала': 'has created', 'إدمان': 'addiction', '游泳游': 'Swimming tour', '狮子头': 'Lion head', 'दिये': 'Given', 'पास': 'near', 'εντjα': 'in', 'まする': 'To worship', 'हाल': 'condition', '짱깨': 'Chin', 'জল': 'Water', 'चीज': 'thing', 'îmwe': 'one', '胡江南': 'Hu Jiangnan', 'तमाम': 'All', 'कट्टर': 'Hardcore', '魔法师': 'Magician', 'బుూ\u200c': 'Buu', 'चनचल': 'Chanchal', 'सीधा': 'Straightforward', '不要人夸颜色好': "Don't exaggerate the color", 'हिमालय': 'Himalaya', 'सिंह': 'Lion', 'خراميشو': 'Wastewater', 'мoѕт': 'the moment', 'কোথায়': 'Where?', 'نصف': 'Half', 'رائع': 'Wonderful', 'добрый': 'kind', 'ज़र्रा': 'Zarra', 'يعقوب': 'Yacoub', 'सम्झौता': 'Agreement', 'ān': 'yes', '했다': 'did', 'ἀριστοκράτης': 'aristocrat', 'çan': 'Bell', '太阳': 'sun', 'सर': 'head', 'गुरु': 'Master', '嘉定': 'Jiading', '故乡': 'home', '安倍晋三': 'Shinzo Abe', 'मच्छर': 'Mosquito', 'सभी': 'All', '成語': 'idiom', '唯我独尊': 'Only me', 'นะค่ะ': 'Yes', 'นั่นคุณกำลังทำอะไร': 'What are you doing', 'ın': 's', 'الشامبيونزليج': 'Shamballons', '抗日神剧': 'Anti-Japanese drama', '小妹': 'Little sister', 'çok': 'very', 'लोड': 'Load', 'साहित्यिक': 'Literary', 'लाल।': 'Red.', 'فى': 'in a', '絕不放過一人」之勢': 'Never let go of one person', '国家主席': 'National president', '异议': 'objection', 'अनुस्वार': 'Anuswasar', 'តើបងប្អូនមានមធ្យបាយអ្វីខ្លះដើម្បីរក': 'What are some ways to find out', '允≱ၑご搢': '≱ ≱ ≱ 搢', '黒髪': 'Black hair', '自戳双目': 'Self-poke binocular', 'βροχέως': 'rainy', 'ని': 'The', '一带一路': 'Belt and Road', 'śliwińska': 'Śliwińska', 'κατὰ': 'against', '打古人名': 'Hit the ancient name', 'குறை': 'Low', 'üjin': 'Uji', 'öppning': 'opening', 'अन्जल': 'Anjal', '이와': 'And', 'आविष्कार': 'Invention', 'غنج': 'Gnostic', 'рассвет': 'dawn', 'होना': 'Happen', 'বাড়ি': 'Home', '入定': 'Enter', 'भेड़': 'The sheep', 'देगा': 'Will give', 'ब्रह्मा': 'Brahma', 'बीन': 'Bean', 'χράω': "I'm glad", 'तीन': 'three', 'ضرار': 'Drar', 'विराजमान': 'Seated', 'सैलाब': 'Salab', 'συνηθείας': 'communication', '구경하고': 'To see', 'चुल्ल': 'Chool', '红宝书': 'Red book', '羡慕和嫉妒是不一样的。': 'Envy and jealousy are not the same.', 'मेरा': 'my', 'कलटि': 'Katiti', 'हिमाती': 'Himalayan', 'ಸಾಧು': 'Sadhu', 'عطيتها': 'Her gift', 'छान': 'Nice', 'เกาหลี': 'Korea', 'íntruments': 'instruments', 'يتحمل': 'Bear', 'रुस्तम': 'Rustam', 'बरात': 'Baraat', 'रंग': 'colour', '나이': 'age', 'פרפר': 'butterfly', '老乡': 'Hometown', '謝謝': 'Thank you', 'í‰lbƒ': '‰ in lbƒ', 'हरजाई': 'Harjai', 'পেতে': 'Get to', '「勉強': '"study', 'रड़कना': 'To cry', '清真诊所': 'Halal clinic', 'أفضل': 'Best', 'استطيع': 'I can', 'नाकतोडा': 'Nag', 'बड़ि': 'Elder', 'वैद्य': 'Vaidya', 'أنَّ': 'that', 'いたロリィis': 'There was Lory', 'ìn': 'print', '本人堅持一個中國的原則': 'I adhere to the one-China principle', 'океан': 'ocean', 'बाद': 'after', '禮儀': 'etiquette', 'सिषेविरे': 'Sisavir', 'अमृत': 'Honeydew', 'بدو': 'run', 'ὅς': 'that is', 'छोटा': 'small', 'स्वभावप्रभवैर्गुणै': 'Nature is bad', 'ענג': 'Tight', 'שלמה': 'Complete', 'डोरे': 'Dore', '我害怕得毛骨悚然': 'I am afraid of creeps.', 'झोलि': 'Jolly', 'единадесет': 'eleven', 'என்ன': 'What', 'आयुरवेद': 'Ayurveda', '堵天下悠悠之口': 'Blocking the mouth of the world', 'الجمعي': 'Collective', 'ángel': 'Angel', 'रोना': 'crying', 'हमराज़': 'Humraj', 'चिलमन': 'Drape', 'औषध': 'Medicine', 'निया': 'Nia', 'תעליעוכופור': 'Go up and go', '中国队大胜': 'Chinese team wins', 'δ##': 'd ##', 'चला': 'walked', 'पर': 'On', 'ἔργον': 'work', 'रंक': 'Rank', 'सिक्ताओ': 'Sixtao', '陳太宗': 'Chen Taizong', 'لومڤور': 'لوموور', 'விளையாட்டு': 'Sports', '的？': 'of?', '안녕하세요': 'Hi', 'נבאים': 'Prophets', 'ন্য': 'N.', 'şimdi': 'now', 'भरना': 'Fill', 'धरी': 'Clog', '漢字': 'Chinese character', 'इसि': 'This', 'आवर': 'Hour', 'काटना': 'cutting', '僕だけがいない街': 'City where I alone does not exist', 'जूठा': 'Lier', 'çekerdik': 'We Are', 'čaj': 'tea', 'నందొ': 'Nando', '核对无误': 'Check is correct', '無限': 'unlimited', 'समेटना': 'Crimp', 'żurek': 'sour soup', 'আছে': 'There are', 'مرتك': 'Committed', 'आपाद': 'Exorcism', 'चोरी': 'theft', 'బవిష్కతి': 'Baviskati', 'निकालना': 'removal', 'सीमा': 'Limit', '配信開始日': 'Distribution start date', '宝血': 'Blood', 'ग्यारह': 'Eleven', 'गरीब': 'poor', '있고': 'There', 'île': 'island', 'स्मि': 'Smile', 'енгел': 'engel', 'вы': 'you', 'οπότε': 'so', 'सूनापन': 'Desolation', 'áraszt': 'exudes', 'मारि': 'Marie', '서로를': 'To each other', 'घोपना': 'To announce', 'फितूर': 'Fitur', 'あからさまに機嫌悪いオーラ出してんのに話しかけてくるし': 'I will speak to the out-putting aura outright bad', '封柏荣': 'Feng Bairong', '「褒められたものではない」': '"It is not a praise"', '傳統文化已經失存了': 'Traditional culture has lost', '水木清华': 'Shuimu Tsinghua', 'पन्नी': 'Foil', 'ਰਾਜਬੀਰ': 'Rajbir', '煮立て': 'Boiling', '外国的月亮比较远': 'Foreign moon is far away', 'شغف': 'passion', 'గురించి': 'About', 'övp': 'ÖVP', '实名买票制': 'Real name ticket system', 'बिखरना': 'To scatter', 'التَعَمّقِ': 'Persecution', 'जीवें': 'The living', 'गई': 'Has been', '部頭合約': 'Head contract', 'دقیانوس': 'Precise', 'फंकी': 'Funky', '粵拼': 'Cantonese fight', 'ĉiohelpanto': 'all supporter', 'मारना': 'killing', 'मानजा': 'Manja', 'अष्टांगिक': 'Octagonal', 'への': 'To', 'रहना': 'live', 'खाना': 'food', 'ಕೋಕಿಲ': 'Kokila', 'చిన్న': 'Small', '金髪': 'Blond hair', '籍贯': 'Birthplace', 'उखाड़ना': 'Extirpate', 'α2': "A'2", 'להשתלט': 'take over control', '露西亚': 'Lucia', '大有「寧可錯殺一千': 'There is a lot of "I would rather kill a thousand', 'संस्कृत': 'Sanskrit', 'христо': 'Christo', 'लगाना': 'to set', '선배님': 'Seniors', 'उड़ीसा': 'Orissa', 'ஆய்த': 'Ayta', 'तेरे': 'Your', 'आकांक्षी': 'Aspiring', 'बाज़ार': 'The market', 'हर्ज़': 'Herz', 'だします': 'Will do.', 'चक्कर': 'affair', '없어': 'no', 'चम्पक': 'Champak', 'ताल': 'rythm', 'āgamas': 'hobby', 'योद्धा': 'Warrior', 'αλυπος': 'chain', 'बेड़ा': 'Fleet', 'बात': 'talk', '做莫你酱紫的？': 'Do you make your sauce purple?', '見逃し': 'Miss', '鰹節': 'Bonito', 'तथैव': 'Fact', 'आध्यात्मिकता': 'Spirituality', '正弦': 'Sine', 'लिया': 'took', 'îndrăgire': 'love', '我願意傾聽': 'I am willing to listen', '家乡': 'hometown', '大败': 'Big defeat', 'मए': 'Ma', 'జ్ఞ\u200cా': 'Sign', 'օօ': 'oo', 'खेमे': 'Camps', 'الشوكولاه': 'Chocolate', 'полностью': 'completely', '商品発売日': 'Product Release Date', '金継ぎ': 'Gold piecing', '烤全羊是多少人民币呢？': 'How much is the renminbi roasted in the whole sheep?', 'γолемата': 'golem', '孫瀛枚': 'Grandson', 'özlüyorum': 'I am missing', 'خلبوص': 'Libs', 'アンダージー': 'Undergar', '蝴蝶': 'butterfly', 'প্রশ্ন': 'The question', 'जरूरी': 'Necessary', 'बूरा': 'Bura', '有毒': 'poisonous', 'सान्त्वना': 'Comfort', '눈치': 'tact', '\ufeffwhat': 'what', 'çeşme': 'fountain', 'ごっめんなさい': 'Please, sorry.', 'விளக்கம்': 'Description', '罗西亚': 'Rosia', 'गोटा': 'Knot', 'लेखावलिपुस्तके': 'Accounting Tips', 'सम्भोग': 'Sexual intercourse', '慢走': 'Slow walking', 'तुम': 'you', 'लीला': 'Leela', 'ฉันจะทำให้ดีที่สุด': 'I will do my best', 'चम्पु': 'Shampoo', '視聽華語': 'Audiovisual Chinese', '比比皆是': 'Abound', 'धावा': 'Run', '娘娘': 'Goddess', 'पहाड़': 'the mountain', 'राजा': 'King', '茹西亚': 'Rusia', 'ब्राह्मणक्षत्रियविशां': 'Brahmin Constellations', '수업하니까': "I'm in class.", 'చెల్లెలు': 'Sister', 'साजे': 'Made', '支那人': 'Zhina', '团表': 'Group table', '讲得': 'Speak', 'へからねでて': 'From to', 'über': 'over', 'քʀɛʋɛռt': 'kʀɛʋɛr t', 'นเรศวร': 'Naresuan', '方言': 'dialect', 'पना': 'Find out', '怕樱': 'Afraid of cherry', 'घुल': 'Dissolve', '米帝': 'Midi', 'طيب': 'Ok', 'प्रेम': 'Love', 'पढ़ाई': 'studies', '他穿褲子': 'He wears pants', 'मेरी': 'Mine', 'উত্তর': 'Reply', 'स्थिति': 'Event', '\x1b\xadü': '?\xadü', 'तैश': 'Tachsh', '写得好': 'Well written', 'मॉलूम': 'Known', '创意梦工坊': 'Creative Dream Workshop', 'आपकी': 'your', 'मिलना': 'Get', '배웠어요': 'I learned it', '許自東': 'Xu Zidong', 'जाऊँ': 'Go', 'अहिंसा': 'Nonviolence', 'джоу': 'Joe', '金繕い': 'Gold patting', '好心没好报': 'No return on a good deed', 'çevapsiz': 'unanswered', 'मिल': 'The mill', '日本語': 'Japanese', 'اخذ': 'Get', '水军': 'Water army', 'बिना': 'without', 'बनाना': 'Make', 'التوبه': 'Repentance', '一个灵运行在我的家': 'a spirit running in my home', '한국어를': 'Korean', 'कि': 'That', 'εισα': 'import', 'लगना': 'feel', 'गपोड़िया': 'Gopodiya', 'ārūpyadhātu': 'exhyadadate', 'आए': 'Returns', '吃好吃金': 'Eat delicious gold', 'นั่น': 'that', 'बढ़ाना': 'raise up', '보니': 'Bonnie', '爱着': 'Love', '선배': 'Elder', '刷屏': 'Brush screen', '人性化': 'Humanize', 'خرسانة': 'Concrete', '麻辣乾鍋': 'Spicy dry pot', '\x10œø': '? œø', 'सतरंगी': 'Satarangi', '磨合': 'Run-in', 'को': 'To', 'شهادة': 'Degree', 'একটি': 'A', 'へと': 'To', 'يلي': 'Following', '光本': 'Light source', '褒め殺し': 'Praise kill', 'आँचल': 'Anchal', 'এর': 'Of it', 'पिटा': 'Pita', 'بديش': 'Badish', 'गंगु': 'Gangu', '可是': 'but', 'की': 'Of', '谢谢。台灣同胞': 'Thank you. Taiwan compatriots', 'दम': 'power', 'मैकशि': 'Macshi', 'రాజ': 'King', '玉蘭花': 'Magnolia', '江ノ島盾子': 'Enoshima Junko', 'ѕтυpιd': 'ѕtυpιd', 'जिसका': 'Whose', 'எழுத்து': 'Letter', '甲骨文': 'Oracle', 'चूरमा': 'Churma', 'चूलें': 'Chulen', 'प्रविभक्तानि': 'Interpretation', 'いる': 'To have', 'مقال': 'article', 'पाय': 'Feet', 'अतीत': 'Past', 'ármin': 'Armin', '東夷': 'Dongyi', 'आदमकद': 'Life expectancy', 'किये': 'Done', 'पतवार': 'Helm', '楽曲派アイドル': 'Song musical idols', 'डगमगाना': 'Waver', '북한': 'North Korea', '禮記': 'Book of Rites', '西魏': 'Xi Wei', '過労死': 'Death from overwork', 'बेबुनियाद': 'Unbounded', '仙人跳': 'Fairy jump', '港女': 'Hong Kong girl', '虞朝': 'Sui Dynasty', 'µ0': 'μ0', '字母词': 'Letter word', 'अर्धांगिनि': 'Arghangini', '真功夫': 'real Kong Fu', '飯糰': 'Rice ball', 'علم': 'Science', 'गुड़': 'Jaggery', 'гречку': 'buckwheat', '我方记账数字与贵方一致': 'Our billing figures are consistent with yours', 'आने': 'To arrive', 'कब': 'When', '宋楚瑜': 'James Soong', 'διητ': 'filter', 'राई': 'Rye', 'விரதம்': 'Fasting', 'बदल': 'change', '成语': 'idiom', 'ĺj': 'junk', 'какая': 'which one', '文翰': 'Wen Han', 'کـ': 'K', 'ʀɛċօʍʍɛռɖɛɖ': 'ʀɛċ oʍʍɛрɖɛɖ', 'दिलों': 'Hearts', '星期七': 'Sunday', '傳送': 'Transfer', '陳云根': 'Chen Yunen', 'ʿalaʾ': 'Allah', 'गुल': 'Gul', 'ख़बर': 'The news', 'मन्च': 'Manch', '国家知识产权局': 'State Intellectual Property Office', '행복하게': 'happily', 'बबीता': 'Babita', 'юродивый': 'holy fool', 'सफाई': 'clean', 'вода': 'water', 'लाठि': 'End', 'γὰρ': 'γσρ', '在日朝鮮人／韓国人': 'Koreans/Koreans in Japan', '学过': 'Learned', 'जमाना': 'Solidification', 'इकोनॉमिक्स': 'Economics', 'क्या': 'what', 'ドア': 'door', '中国的存在本身就是原罪': 'The existence of China itself is the original sin', 'मौसमि': 'Season', 'ठाठ': 'Chic', 'ἡμετέραν': 'another', '火了mean': 'Fired mean', 'せえの': 'Set out', 'ஆயுத': 'Arms', 'कंपनी': 'company', 'अहम्': 'Ego', 'भर': 'Filled', 'फिरना': 'To revolve', '高山族': 'Gaoshan', '王飞飞是一个哑巴的婊子。': 'Wang Feifei is a dumb nephew.', '反清復明': 'Anti-clearing', '关门放狗': 'Close the dog', '中国民族伟大复兴？i': 'The great revival of the Chinese nation? i', '漢名': 'Han name', 'ín': 'into the', '이름은': 'name is', '뽀비엠퍼러': 'Pobi Emperor', '堅決反對台獨言論': 'Resolutely oppose Taiwan independence speech', 'ड़': 'D', 'ادب': 'Literary', '〖2x〗': '〖〗 2x', 'في': 'in a', 'परन्तप': 'Parantap', '不正常人類研究中心': 'Abnormal human research center', 'метью': 'by the way', 'σε': 'in', '罗马炮架': 'Roman gun mount', 'đỗ': 'parked', '二哈': 'Erha', '中國': 'China', 'मुझे': 'me', 'бджа': 'bzha', 'छुटाना': 'To leave', 'সহ': 'Including', 'δίδυμος': 'twin', 'दौरा': 'Tour', 'आया': 'He Came', 'ľor': 'Lor', 'شاف': 'balmy', 'افشلك': 'I miss you', 'पता': 'Address', 'śląska': 'Silesian', '我还有几个条件呢。': 'I still have a few conditions.', '元寇': 'Yuan', 'सहायक': 'Assistant', 'टाइम': 'Time', 'समुदाय': 'Community', 'टिपटिपवा': 'Tiptipava', '五毛党': 'Wu Mao Party', 'दे': 'Give', '屠城': 'Slaughter city', 'कहने': 'To say', 'कलमूहा': 'Kalamuha', 'בפנים': 'in', 'कर्माणि': 'Creation', 'अथवा': 'Or', 'રાજ્ય': 'State', 'घसोना': 'Shed', '今そう言う冗談で笑える気分じゃないから一人にしてって言ったら何があったの？話聞くよって言われたんだけど': 'I do not feel like being laughing with what I say now, so what happened when I told you to be alone? I was told by listening and talking', 'परमो': 'Paramo', 'υφηιρειτω': 'I suppose', 'כתובים': 'Written', '能夠': 'were able', '고등학교는': 'High School', 'बजरि': 'Breeze', 'जानेदिल': 'Knowingly', '大胜': 'Big win', 'काल': 'period', 'すみません': "I'm sorry", 'हिमाकत': 'Snow plant', 'þîfû': 'iphone', 'よね': 'right', 'هالنحس': 'Halting', 'الحجازية': 'Hijaz', 'жизнь': 'a life', 'किया': 'Did', '沸騰する': 'To boil', 'นะครับ': 'Yes', 'پیش': 'Before', 'परदाफास': 'Bustard', 'वफाएं': 'Affection', 'सैनानि': 'Sanani', 'ölü': 'dead', 'हैं': 'Are', '宋美齡': 'Song Meiling', 'கவாடம்': 'Valve', '我是一名来自大陆的高中生': 'I am a high school student from the mainland.', 'اجمل': 'The most beautiful', '가용': 'Available', 'चरम': 'Extreme', '象形文字': 'Hieroglyphics', '男女授受不亲': 'Men and women don’t kiss', 'моя': 'my', '疑义': 'Doubt', '管中閔': 'In the tube', 'çekseydin': 'pulls you were', 'عم': 'uncle', 'ルージュ': 'Rouge', '永遠': 'forever and always', 'بيت': 'a house', '「褒められた」': '"I was praised"', 'कर्णजित्': 'Karnajit', 'あたまわるい': 'Bad headache', 'ε0': 'e0', 'şafak': 'dawn', 'जलाओ': 'Burn', '観世音菩薩': '観世音', 'पीत': 'Yellow', 'दारी': 'Dari', 'የሚያየኝን': 'What i see', 'هاپو': 'Dog', '发声点': 'Vocal point', 'être': 'be', 'மாண்பாளன்': 'Manpalan', '감겨드려유': 'You can wrap it.', '知音': 'Bosom friend', 'एक': 'One', '\x01jhó': '? Jho', 'かんぜおんぼさつ': 'Punctuation', 'ødegaard': 'Ødegaard', '得理也要让人': 'It’s ok to make people', 'مرة': 'Once', 'दो': 'two', 'ने': 'has', '用乡村包围城市': 'Surround the city with the countryside', 'مكتب': 'Office', 'řepa': 'beet', 'день': 'day', '江青': 'Jiang Qing', 'ángeles': 'angels', 'çonstant': 'constant', 'टिपटिपुआ': 'Tip Tip', 'ஆன்லைன்': 'Online', '盧麗安': "Lu Li'an", 'самый': 'most', 'からかってくる男に': 'To the coming men', 'بعرف': 'I know', '知乎': 'Know almost', 'पहेलि': 'Puzzles', 'बहार': 'spring', 'भंगिमाँ': 'Bhangima', 'くわんぜおんぼさつ': 'Honey', '河殤': 'River', 'شو': 'Shaw', 'محمد': 'Mohammed', 'спереди': 'in front', '没毛病': 'No problem', 'árbenz': 'Arbenz', 'लार्वा': 'Larvae', 'चढ़ा': 'Ascend', 'तो': 'so', 'يلعب': 'Play', 'घिसा': 'Ginger', 'रेखागड़ित': 'Sketchy', 'मुरदा': 'Murada', 'चित्र': 'picture', 'தாக்கல்': 'Filing', '大败美国队': 'Big defeat to the US team', 'искусственный': 'artificial', 'चाल': 'Trick', 'घुटता': 'Kneeling', 'बिगड़ना': 'Deteriorate', '您这口味奇特也就罢了': 'Your taste is strange.', 'लायक': 'Worth', '大篆': 'Daxie', 'खिलाना': 'To feed', 'ماهو': 'What is the', 'पहुँचनेके': 'To reach', '外来語': 'Foreign language', 'चटका': 'Click', 'کوالا': 'Koala bear', '不过': 'but', 'पड़ना': 'Fall', 'पानी': 'Water', '저의': 'my', '一生懸命': 'Hard', 'مش': 'not', 'लिखामि': 'Written', 'गया': 'Gaya', 'компания': 'company', 'तेरि': 'Teri', '茶髪': 'Brown hair', '하다': 'Do', '不是共和国；中华人民共和国和朝鲜民主主义人民共和国是共和国': 'Not a republic; the People’s Republic of China and the Democratic People’s Republic of Korea are Republic', '烤全羊多少人民币呢？': 'How much is the price of roast whole sheep?', 'घर': 'Home', 'पाला': 'Frost', '左右手': 'Left and right hand', 'کوالالامپور': 'Kuala Lumpur', 'āsanas': 'asanas', 'ոո': 'e', 'сегодня': 'Today', '저는': 'I am', 'русофил': 'blossoming', '甘蓝': 'Cabbage', '流浪到淡水': 'Wandering to fresh water', 'единайсет': 'eleven', 'परात': 'Underwear', '\x7fhow': '?how', '¡que': 'what', 'ἀχιλλεύς': 'Achilles', 'так': 'So', 'آداب': 'Rituals', '\x10i': '?i', 'मट्ठा': 'Whey', '平天下悠悠之口': 'The mouth of the world', 'लुन्डा': 'Lunda', 'ся': 'camping', 'вареники': 'Vareniks', 'δο': 'gt;', 'öyle': 'so', 'पुरे': 'Enough', '看他不顺眼': 'Seeing him not pleasing to the eye', '「o」': '"O"', 'とする': 'To', 'হচ্ছে': 'Being', 'लाखों': 'Millions', 'α1': "A'1", 'نهائي': 'Final', 'ɾ̃': 'ɾ', 'तिलक': 'Tilak', 'لا': 'No', 'صور': 'photo', '怒怼': 'Roar', 'மௌன': 'Mauna', 'परिस्थिति': 'Situation', '서로가': 'Mutually', '山進': 'Shanjin', '蝴蝶蛋': 'Butterfly egg', 'ксш': 'ksş', '饭可以乱吃': 'Rice can be eaten', '渡します': 'I will hand it over.', 'österreich': 'Austria', 'øÿ\x13': 'øÿ?', 'ممتاز': 'Excellent', '蝴蝶卵': 'Butterfly egg', 'ही': 'Only', 'ठोकरे': 'Knock', '湿婆': 'Shiva', 'обожаю': 'love', 'लस्सी': 'Lassi', '操你妈': 'Fuck your mother', 'फौलादि': 'Fauladi', 'жизни': 'of life', 'đổi': 'change', '阻天下悠悠之口': 'Block the mouth of the world', 'オメエだよオメエと話してんのが苦痛なんだよ。シネ。': "It's omn it is painful to talk to Oume. Cine.", 'खाट': 'The cot', '译文': 'Translation', 'सँवरना': 'To embellish', 'дванадесет': 'twelve', '陳雲': 'Chen Yun', 'дванайсет': 'twelve', 'आँखें': 'Eyes', 'पूछते': 'Inquires', 'भंगी': 'Posture', 'सूना': 'Deserted', 'प्याला': 'Cup', 'には': 'To', 'доктора': 'the doctors', 'देना': 'give', 'बिरेन्र्द': 'Forget', 'गला': 'throat', 'रखा': 'Kept', 'हाले': 'Haley', '欢迎入坑': 'Welcome to the pit', 'डलि': 'Dallie', 'ôš': 'OS', 'يوسف': 'Yousuf', 'छोंकना': 'Strain', 'пп': 'pp', 'उसपर': 'on that', 'υρολογιστών': 'computers', '新年快乐！学业进步！身体健康！谢谢您们读我的翻译篇章': 'happy New Year! Academic progress! Healthy body! Thank you for reading my translation chapter.', '人民': 'people', 'घंटे': 'Hours', 'شباط': 'February', '食べる': 'eat', 'صلاح': 'Salah', '土澳': 'Tuao', '干嘛天天跟我说韩语': 'Why do you speak Korean with me every day?', 'ōnogi': 'climate', 'صاحب': 'owner', 'اكل': 'ate', '大唐': 'Datang', 'مطالعه': 'Study', '养生': 'Health', '车子': 'Car', 'कचरा': 'Garbage', 'महरबानि': 'Mehraban', 'शुष्टि': 'Shutti', 'интеллект': 'intelligence', '阮鏐': '阮镠', '鲁玥': 'Reckless', '입니다': 'is', 'ῥιζὤματα': 'Threads', 'люблю': 'love', 'ɛxɛʀċɨsɛ': 'it is not', 'कपड़े': 'dresses', 'उड़न': 'Flying', '广电总局': 'SARFT', '骂人': 'curse', 'የየየኝን': 'What do i say', 'बेलना': 'Crib', 'பல்லாக்குப்': 'Pallakkup', 'बराबर': 'equal', 'ظرف': 'Dish', 'होनोपैथिक': 'Homeopathic', '君子': 'Gentleman', '河和湖': 'Kawahata lake', '精一杯': 'Utmost', 'है': 'is', '非要以此为依据对人家批判一番': 'I have to criticize others on this basis.', 'வச்சன்': 'Vaccan', 'நான்': 'I', 'का': 'Of', '三味線': 'Shamisen', 'šwejk': 'švejk', 'дурак': 'fool', '风琴': 'organ', 'हिंसा': 'Violence', 'βιον': 'bio', 'नारी': 'Woman', '知らない': 'Do not know', 'मुँह': 'The mouth', 'अपरम्पार': 'Unperturbed', '秋季新款': 'Autumn new style', 'আমার': 'Me', 'عبقري': 'genius', 'आहें': 'Ah', 'ਨਾਮ': 'Name', 'महफ़िल': 'Mehfil', 'बटेर': 'Quail', '林彪': 'Lin Wei', 'जाने': 'Know', 'डोंगरिचाल': 'Mountain move', 'εἰρήνη': 'Irene', 'प्रतिलेखनम्': 'Transcript', 'дп': 'dp', 'उसने': 'He', '몇시간': 'how many hours', 'नैया': 'Naiya', 'ἐξήλλακτο': 'inexpensive', '彩蛋': 'Egg', 'उलटफेर': 'Reverse', '台湾最美的风景是人': 'The most beautiful scenery in Taiwan is people.', '馄饨': 'ravioli', 'सदके': 'Shake', '饺子': 'Dumplings', 'भूमिका': 'role', '为什么说': 'Why do you say', '요즘': 'Nowadays', 'गिरि': 'Giri', '中國話': 'Chinese words', 'द्रोहि': 'Drohi', '中庸之道': 'The doctrine of the mean', 'хочу': 'want', 'ḵarasāna': 'ḵarasana', '走gǒ': 'Go gǒ', 'जमाल': 'Jamal', 'मन': 'The mind', 'तेलि': 'Oilseed', '非诚勿扰': 'You Are the One', 'атом': 'atom', '中华民国和大韩民国是民国': 'The Republic of China and the Republic of Korea are the Republic of China', 'नलि': 'Nile', 'हाथ': 'hand', 'खाजला': 'Itching', 'ōe': 'yes', '것이다': 'will be', 'şoųl': 'şoùl', 'तश्तरि': 'Cleverness', 'χρῆσιν': 'use', '民族': 'Nationality', 'الرياضيات': 'Mathematics', 'にじゅうさい': 'Twelve months', 'ὠς': 'as', '恋に落ちないからよく悲しい': 'It is often sad because it does not fall in love', 'ਸ਼ੀਂਹ': 'Lion', '黎氏玉英': 'Li Shiyuying', 'فبراير': 'February', '白濮': 'Chalk', 'অধীনে': 'Under', 'प्रशंसा': 'appreciation', 'ขอพระเจ้าอยู่ด้วย': 'May God be with you', 'अडला': 'Bent', 'ده': 'Ten', 'पसीजना': 'Exudate', 'कोई': 'someone', 'கருக்குமட்டை': 'Karukkumattai', 'कन्नी': 'Kanni', 'यात्रा': 'journey', '白酒': 'Liquor', 'τὴν': 't', 'करना': 'do', 'उल': 'ul', '俄罗斯': 'Russia', 'உனக்கு': 'You', 'जौहर': 'Johar', 'ಸ್ವರಕ್ಷರಗಳು': 'Self-defense', '论语': 'Analects', 'šakotis': 'branching', '儿臣惶恐': 'Childier fear', '讲的？': 'Said?', 'প্রতিনিয়ত': 'Every day', 'சன்னல்': 'Sill', '蠢的像猪一样': 'Stupid like a pig', 'رح': 'Please', 'بس': 'Yes', 'εὔιδον': 'you see', 'सदबुद्धि': 'Good sense', 'भगवान': 'God', '사랑해': 'I love you', 'בתוך': 'Inside', 'čeferin': 'чеферин', '民国': 'Republic of China', '但是': 'but', 'सकता': 'can', 'घाट': 'Wharf', 'čechy': 'Bohemia', '抹黑': 'Smear', 'γλαυκῶπις': 'greyhounds', 'నీకెందుకు': 'Nikenduku', 'चमत्कार': 'Miracle', 'दुनिया': 'world', 'یہاں': 'Here', 'اللي': 'Elly', 'খামার': 'The farm', '一呼百诺': 'One call', 'വിഡ്ഢി': 'Stupid', 'दिल': 'heart', 'тeenage': 'trinage', '皇上': 'emperor', 'टटोलना': 'Grope', '犬子': 'Dog', '我希望有一天你沒有公王病': 'I hope that one day you don’t have a king', '并无分裂中国的意图': 'No intention to split China', 'óscar': 'oscar', 'ኤልሮኢ': 'Alright', 'ŷhat': 'hhat', '천사': 'Angel', 'दी।': 'Given', 'रड़क': 'Raze', 'कानी': 'Kani', '江之島盾子': 'Enokima Shiko', '老生常谈': 'Old talk', 'των': 'of', '星期日': 'on Sunday', 'पैन्डा': 'Panda', 'マリも仲直りしました': 'Mari also made up.', 'ты': 'you', 'देश': 'Country', 'ठंडक': 'Coolness', 'নামল': 'Get down', 'जर्मनी': 'Germany', 'шли': 'walked', 'たべる': 'To eat', 'लिए': 'for', '鸡汤文': 'Chicken soup', 'มวยไทย': 'Thai boxing', '簡訊': 'Newsletter', 'منزل': 'Home', 'कर': 'Tax', '生女眞': 'Daughter-in-law', 'ек': 'ek', 'સંઘ': 'Union', '\u200bsalarpuria': 'Salphary', 'ţara': 'the country', 'नही': 'No', 'मगज': 'Mercury', 'অক্ষয়': 'Akshay', 'حال': 'Now', 'चिन्दी': 'Chindee', 'τῆς': 'her', '福哒柄': 'Good fortune', '청하': 'Qinghai', '越人': 'Yueren', 'なかなかに謎だな': "It's quite a mystery.", 'रखना': 'keep', 'பரை': 'Parai', 'करके': 'By doing', 'فِي': 'F', 'गुथना': 'Knit', '话不可以乱讲': "Can't talk nonsense", 'वैकल्पिक': 'Alternative', 'ਨਾਮੁ': 'Name', '你别高兴得太早': "Don't be too happy too early", '煎餅': 'Pancake', '한다': 'do', 'सबब': 'Cause', 'বিষয়টি': 'Matter', 'कोसना': 'To crack', 'ㅜㅜ': 'ㅜ', 'অক্সয়': 'Akshay', 'الدوالي': 'Varicose veins', 'பயிர்ப்பு': 'Yields', 'अजा': 'SC', 'あの色々': 'That kind of variety', 'емеля': 'emel', 'मेवा': 'Meva', 'जलवाफरोज़': 'Jalwa Phoroz', '中庸': 'Moderate', 'उसके': 'his', 'अहम': 'Important', 'वहम': 'Vanity', 'ís': 'ice', 'कलात्मक': 'Artistic', 'ἀχιλῆος': 'Achilles', '民族罪人': 'National sinner', 'मुकाम': 'Peer', 'ão': 'to', '한국': 'Korea', 'ادهر': 'Idir', '一長': 'One long', 'てくれませんか': 'Would you please', 'ブレイク': 'break', 'शहनाई': 'the clarinet', 'तीरन्दाज़ि': 'Arrows', 'रूढ़ीवादि': 'Conservative', 'झाँकी': 'Peeping', 'सत्यवादी': 'Truthful', '郑琳': 'Zheng Lin', 'युनानी': 'Unani', 'φώνας': 'light', 'जमना': 'Solidify', '〖plg〗': '〖Plg〗', 'हरी': 'Green', 'بطولة': 'championship', '这些词怎么读？这些词怎么说？这些词怎么念？which': 'How do you read these words? What do these words say? How do you read these words? Which', 'алтерман': 'alterman', 'اَلبَحْثِ': 'البحث', 'موجود': 'Available', 'कश्ति': 'Power', 'اسم': 'Noun', 'लाज़मि': 'Shameful', 'तुमने': 'you', '공화국': 'republic', 'लुक्का': 'Lukka', '史记': 'Historical record', '포경수술': 'Circumcised', '高端大气上档次into': 'High-end atmospheric grade into', 'что': 'what', 'योध': 'Rift', 'धर्म': 'religion', 'दरदर': 'Tariff', '訓読み': 'Kun Readings', 'впереди': 'in front', '민국': 'Republic of Korea', 'εντσα': 'in', '我搜了这本小说': 'I searched this novel.', '杨皎滢？': 'Yang Wei?', 'भारतीयों': 'Indians', '巴蜀': 'Bayu', '\x02tñ\x7f': '? tñ?', 'αβtαβ': 'aba', 'जल': 'water', 'बाध्य': 'Bound', 'মহাবিশ্ব': 'Universe', 'প্রসারিত': 'Stretch', 'अन्जाम': 'Anjaam', 'जीतना': 'win', 'कड़ा': 'Hard', '刁民': 'Untouchable', 'ขอให้พระเจ้าอยู่ด้วย': 'May God live too.', '油腻': 'Greasy', 'ᗯoᗰeᑎ': 'ᗯoᗰe ᑎ', 'להתראות': 'Goodbye', 'वाले': 'Ones', 'አየሁ': 'I saw', 'ओर': 'And', 'ずand': 'Without', 'निगोड़ा': 'Nigoda', 'эй': 'Hey'



        }



        def _get_contractions(contractions_dict):

            contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

            return contractions_dict, contractions_re



        contractions, contractions_re = _get_contractions(contractions_dict)

        def replace_contractions(text):

            def replace(match):

                return contractions[match.group(0)]

            return contractions_re.sub(replace, text)

        print('REPLACE CONTRACTIONS')

        train_df['question_text'] = train_df['question_text'].apply(lambda x: replace_contractions(x))

        test_df['question_text'] = test_df['question_text'].apply(lambda x: replace_contractions(x))

        print('FINISH REPLACE CONTRACTIONS')

        list_sentences_train = train_df["question_text"].fillna("_na_").values

        list_sentences_test = test_df["question_text"].fillna("_na_").values



        # TOKENIZE-----#

        num_words=120000

        maxlen=72

        print('TOKENIZING')

        tokenizer = Tokenizer(num_words=num_words, char_level=False, lower=False)

        all_text  = list(list_sentences_train) + list(list_sentences_test)

        tokenizer.fit_on_texts(all_text)

        del all_text

        gc.collect()

        print('FINISH FITTING TOKENIZER')

        train_sequences = tokenizer.texts_to_sequences(list_sentences_train)

        train_sequences = pad_sequences(train_sequences, maxlen=maxlen)



        test_sequences = tokenizer.texts_to_sequences(list_sentences_test)

        test_sequences = pad_sequences(test_sequences, maxlen=maxlen)

        print('FINISH prep sequences')

        del list_sentences_train, list_sentences_test, contractions_dict

        gc.collect()





        # EMBEDDINGS-----#

        def load_embedding(embedding):

            print(f'Loading {embedding} embedding..')

            def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

            def load_word2vec(fname, encoding='utf8', unicode_errors='strict',datatype=np.float32):

                embedding_index = dict()

                with utils.smart_open(fname) as fin:

                    header = utils.to_unicode(fin.readline(), encoding=encoding)

                    vocab_size, vector_size = (int(x) for x in header.split())

                    binary_len = np.dtype(datatype).itemsize * vector_size

                    for _ in tqdm(range(vocab_size)):

                        # mixed text and binary: read text first, then binary

                        word = []

                        while True:

                            ch = fin.read(1)

                            if ch == b' ':

                                break

                            if ch == b'':

                                raise EOFError("unexpected end of input")

                            if ch != b'\n':

                                word.append(ch)

                        word = utils.to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)

                        weights = np.fromstring(fin.read(binary_len), dtype=datatype).astype(datatype)

                        embedding_index[word] = weights

                return embedding_index

            if embedding == 'glove':

                EMBEDDING_FILE = f'{FILE_DIR}/embeddings/glove.840B.300d/glove.840B.300d.txt'

                embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8"))

            elif embedding == 'wiki-news':

                EMBEDDING_FILE = f'{FILE_DIR}/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

                embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8") if len(o)>100)

            elif embedding == 'paragram':

                EMBEDDING_FILE = f'{FILE_DIR}/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

                embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

            elif embedding == 'google-news':

                from gensim.models import KeyedVectors

                EMBEDDING_FILE = f'{FILE_DIR}/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

                #embeddings_index = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

                embeddings_index = load_word2vec(EMBEDDING_FILE)

            return embeddings_index



        def build_embedding_matrix(embeddings_index_1, one=True, embedding_name_1='random'): 

            if one:

                embeddings_index_1 = load_embedding(embeddings_index_1)



            wl = WordNetLemmatizer().lemmatize

            word_index = tokenizer.word_index

            nb_words = min(num_words, len(word_index))

            embedding_matrix = np.zeros((nb_words, 301))



            def all_caps(word):

                return len(word) > 1 and word.isupper()



            if embedding_name_1 == 'google-news':

                something_1 = embeddings_index_1.word_vec("something")

                something = np.zeros((301,))

                something[:300,] = something_1

                something[300,] = 0



                hit = 0

                def embed_word(embedding_matrix,i,word):

                    embedding_vector_1 = embeddings_index_1.word_vec(word)

                    if embedding_vector_1 is not None: 

                        if all_caps(word):

                            last_value = np.array([1])

                        else:

                            last_value = np.array([0])

                        embedding_matrix[i,:300] = embedding_vector_1

                        embedding_matrix[i,300] = last_value



                for word, i in word_index.items():

                    if i >= nb_words: continue

                    if word not in embeddings_index_1.vocab:

                        embedding_matrix[i] = something 

                    else:

                        if embeddings_index_1.word_vec(word) is not None:

                            embed_word(embedding_matrix,i,word)

                            hit += 1

                        else:

                            if len(word) > 20:

                                embedding_matrix[i] = something

                            else:

                                word2 = wl(wl(word, pos='v'), pos='a')

                                if embeddings_index_1.word_vec(word2) is not None:

                                    embed_word(embedding_matrix,i,word2)

                                    hit += 1

                                else:                   

                                    if len(word) < 3: continue

                                    word2 = word.lower()

                                    if embeddings_index_1.word_vec(word2) is not None:

                                        embed_word(embedding_matrix,i,word2)

                                        hit += 1

                                    else:

                                        word2 = word.lower()

                                        word2 = wl(wl(word2, pos='v'), pos='a')

                                        if embeddings_index_1.word_vec(word2) is not None:

                                            embed_word(embedding_matrix,i,word2)

                                            hit += 1

                                        else:

                                            word2 = word.upper()

                                            if embeddings_index_1.get(word2) is not None:

                                                embed_word(embedding_matrix,i,word2)

                                                hit += 1

                                            else:

                                                word2 = word.upper()

                                                word2 = wl(wl(word2, pos='v'), pos='a')

                                                if embeddings_index_1.get(word2) is not None:

                                                    embed_word(embedding_matrix,i,word2)

                                                    hit += 1

                                                else:

                                                    embedding_matrix[i] = something 

                             

            else:

                something_1 = embeddings_index_1.get("something")

                something = np.zeros((301,))

                something[:300,] = something_1

                something[300,] = 0



                hit = 0

                def embed_word(embedding_matrix,i,word):

                    embedding_vector_1 = embeddings_index_1.get(word)

                    if embedding_vector_1 is not None: 

                        if all_caps(word):

                            last_value = np.array([1])

                        else:

                            last_value = np.array([0])

                        embedding_matrix[i,:300] = embedding_vector_1

                        embedding_matrix[i,300] = last_value



                for word, i in word_index.items():

                    if i >= nb_words: continue

                    if embeddings_index_1.get(word) is not None:

                        embed_word(embedding_matrix,i,word)

                        hit += 1

                    else:

                        if len(word) > 20:

                            embedding_matrix[i] = something

                        else:

                            word2 = wl(wl(word, pos='v'), pos='a')

                            if embeddings_index_1.get(word2) is not None:

                                embed_word(embedding_matrix,i,word2)

                                hit += 1

                            else:                   

                                if len(word) < 3: continue

                                word2 = word.lower()

                                if embeddings_index_1.get(word2) is not None:

                                    embed_word(embedding_matrix,i,word2)

                                    hit += 1

                                else:

                                    word2 = word.lower()

                                    word2 = wl(wl(word2, pos='v'), pos='a')

                                    if embeddings_index_1.get(word2) is not None:

                                        embed_word(embedding_matrix,i,word2)

                                        hit += 1

                                    else:

                                        word2 = word.upper()

                                        if embeddings_index_1.get(word2) is not None:

                                            embed_word(embedding_matrix,i,word2)

                                            hit += 1

                                        else:

                                            word2 = word.upper()

                                            word2 = wl(wl(word2, pos='v'), pos='a')

                                            if embeddings_index_1.get(word2) is not None:

                                                embed_word(embedding_matrix,i,word2)

                                                hit += 1

                                            else:

                                                embedding_matrix[i] = something 

            del embeddings_index_1

            gc.collect()

            print("Matched Embeddings: found {} out of total {} words at a rate of {:.2f}%".format(hit, nb_words, hit * 100.0 / nb_words))

            return embedding_matrix

        def build_concatenate_embedding_matrix_google(embeddings_index_1, embeddings_index_2, one=True, two=True): 

            if one:

                embeddings_index_1 = load_embedding(embeddings_index_1)

            if two:

                embeddings_index_2 = load_embedding(embeddings_index_2)





            wl = WordNetLemmatizer().lemmatize

            word_index = tokenizer.word_index

            nb_words = min(num_words, len(word_index))

            embedding_matrix = np.zeros((nb_words, 401))



            something_1 = embeddings_index_1.get("something")

            something = np.zeros((401,))

            something[:300,] = something_1

            something[400,] = 0



            def all_caps(word):

                return len(word) > 1 and word.isupper()



            hit = 0

            def embed_word(embedding_matrix,i,word, ori_word):

                embedding_vector_1 = embeddings_index_1.get(word)

                if embedding_vector_1 is not None: 

                    if all_caps(word):

                        last_value = np.array([1])

                    else:

                        last_value = np.array([0])

                    embedding_matrix[i,:300] = embedding_vector_1

                    embedding_matrix[i,400] = last_value

                    if word in embeddings_index_2.vocab:

                        embedding_vector_2 = embeddings_index_2.word_vec(ori_word)

                        embedding_matrix[i,300:400] = embedding_vector_2



            for word, i in word_index.items():

                if i >= nb_words: continue

                if embeddings_index_1.get(word) is not None:

                    embed_word(embedding_matrix,i,word, word)

                    hit += 1

                else:

                    if len(word) > 20:

                        embedding_matrix[i] = something

                    else:

                        word2 = wl(wl(word, pos='v'), pos='a')

                        if embeddings_index_1.get(word2) is not None:

                            embed_word(embedding_matrix,i,word2, word)

                            hit += 1

                        else:                   

                            if len(word) < 3: continue

                            word2 = word.lower()

                            if embeddings_index_1.get(word2) is not None:

                                embed_word(embedding_matrix,i,word2, word)

                                hit += 1

                            else:

                                word2 = word.lower()

                                word2 = wl(wl(word2, pos='v'), pos='a')

                                if embeddings_index_1.get(word2) is not None:

                                    embed_word(embedding_matrix,i,word2, word)

                                    hit += 1

                                else:

                                    word2 = word.upper()

                                    if embeddings_index_1.get(word2) is not None:

                                        embed_word(embedding_matrix,i,word2, word)

                                        hit += 1

                                    else:

                                        word2 = word.upper()

                                        word2 = wl(wl(word2, pos='v'), pos='a')

                                        if embeddings_index_1.get(word2) is not None:

                                            embed_word(embedding_matrix,i,word2, word)

                                            hit += 1

                                        else:

                                            embedding_matrix[i] = something 

            print("Matched Embeddings: found {} out of total {} words at a rate of {:.2f}%".format(hit, nb_words, hit * 100.0 / nb_words))

            del embeddings_index_1, embeddings_index_2

            gc.collect()

            return embedding_matrix

        def build_concatenate_embedding_matrix(embeddings_index_1, embeddings_index_2, one=True, two=True): 

            if one:

                embeddings_index_1 = load_embedding(embeddings_index_1)

            if two:

                embeddings_index_2 = load_embedding(embeddings_index_2)





            wl = WordNetLemmatizer().lemmatize

            word_index = tokenizer.word_index

            nb_words = min(num_words, len(word_index))

            embedding_matrix = np.zeros((nb_words, 601))



            something_1 = embeddings_index_1.get("something")

            something_2 = embeddings_index_2.get("something")

            something = np.zeros((601,))

            something[:300,] = something_1

            something[300:600,] = something_2

            something[600,] = 0



            def all_caps(word):

                return len(word) > 1 and word.isupper()



            hit = 0

            def embed_word(embedding_matrix,i,word):

                embedding_vector_1 = embeddings_index_1.get(word)

                if embedding_vector_1 is not None: 

                    if all_caps(word):

                        last_value = np.array([1])

                    else:

                        last_value = np.array([0])

                    embedding_matrix[i,:300] = embedding_vector_1

                    embedding_matrix[i,600] = last_value

                    embedding_vector_2 = embeddings_index_2.get(word)

                    if embedding_vector_2 is not None:

                        embedding_matrix[i,300:600] = embedding_vector_2



            for word, i in word_index.items():

                if i >= nb_words: continue

                if embeddings_index_1.get(word) is not None:

                    embed_word(embedding_matrix,i,word)

                    hit += 1

                else:

                    if len(word) > 20:

                        embedding_matrix[i] = something

                    else:

                        word2 = wl(wl(word, pos='v'), pos='a')

                        if embeddings_index_1.get(word2) is not None:

                            embed_word(embedding_matrix,i,word2)

                            hit += 1

                        else:                   

                            if len(word) < 3: continue

                            word2 = word.lower()

                            if embeddings_index_1.get(word2) is not None:

                                embed_word(embedding_matrix,i,word2)

                                hit += 1

                            else:

                                word2 = word.lower()

                                word2 = wl(wl(word2, pos='v'), pos='a')

                                if embeddings_index_1.get(word2) is not None:

                                    embed_word(embedding_matrix,i,word2)

                                    hit += 1

                                else:

                                    word2 = word.upper()

                                    if embeddings_index_1.get(word2) is not None:

                                        embed_word(embedding_matrix,i,word2)

                                        hit += 1

                                    else:

                                        word2 = word.upper()

                                        word2 = wl(wl(word2, pos='v'), pos='a')

                                        if embeddings_index_1.get(word2) is not None:

                                            embed_word(embedding_matrix,i,word2)

                                            hit += 1

                                        else:

                                            embedding_matrix[i] = something 

            print("Matched Embeddings: found {} out of total {} words at a rate of {:.2f}%".format(hit, nb_words, hit * 100.0 / nb_words))

            del embeddings_index_1, embeddings_index_2

            gc.collect()

            return embedding_matrix





        # MODELS-----#

        def f1_smart(y_true, y_pred):

            args = np.argsort(y_pred)

            tp = y_true.sum()

            fs = (tp - np.cumsum(y_true[args[:-1]])) / np.arange(y_true.shape[0] + tp - 1, tp, -1)

            res_idx = np.argmax(fs)

            return 2 * fs[res_idx], (y_pred[args[res_idx]] + y_pred[args[res_idx + 1]]) / 2



        def threshold_search(y_true, y_proba):

            best_threshold = 0

            best_score = 0

            for threshold in [i * 0.01 for i in range(100)]:

                score = f1_score(y_true=y_true, y_pred=y_proba > threshold)

                if score > best_score:

                    best_threshold = threshold

                    best_score = score

            search_result = {'threshold': best_threshold, 'f1': best_score}

            return search_result



        def dropout_mask(x, sz, dropout):

            return x.new(*sz).bernoulli_(1-dropout)/(1-dropout)



        class LockedDropout(nn.Module):

            def __init__(self, p=0.5):

                super().__init__()

                self.p=p



            def forward(self, x):

                if not self.training or not self.p: return x

                m = dropout_mask(x.data, (1, x.size(1), x.size(2)), self.p)

                return Variable(m, requires_grad=False) * x



        def sigmoid(x):

            return 1 / (1 + np.exp(-x))



        class AverageMeter(object):

            """Computes and stores the average and current value"""



            def __init__(self):

                self.reset()



            def reset(self):

                self.val = 0

                self.avg = 0

                self.sum = 0

                self.count = 0



            def update(self, val, n=1):

                self.val = val

                self.sum += val * n

                self.count += n

                self.avg = self.sum / self.count



        def accuracy(output, target, topk=(1,)):

            """Computes the accuracy over the k top predictions for the specified values of k"""

            with torch.no_grad():

                maxk = max(topk)

                batch_size = target.size(0)



                _, pred = output.topk(maxk, 1, True, True)

                pred = pred.t()

                correct = pred.eq(target.view(1, -1).expand_as(pred))



                res = []

                for k in topk:

                    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)

                    res.append(correct_k.mul_(100.0 / batch_size))

            return res



        def train_model(train_loader, model, criterion, optimizer, epoch, gpu=None, print_freq=100):

            batch_time = AverageMeter()

            data_time = AverageMeter()

            losses = AverageMeter()

            top1 = AverageMeter()



            model.train()



            epoch_time = time.time()

            end = time.time()



            for i, (x_batch, x_feats_batch, y_batch) in enumerate(train_loader):

                data_time.update(time.time() - end)

                if gpu is not None:

                    x_batch = x_batch.cuda(gpu, non_blocking=True)

                    x_feats_batch= x_feats_batch.cuda(gpu, non_blocking=True)

                y_batch = y_batch.cuda(gpu, non_blocking=True)



                output = model(x_batch, x_feats_batch)

                loss = criterion(output, y_batch)



                # measure accuracy

                acc1 = (y_batch == (output > 0.5).float()).float()

                losses.update(loss.item(), x_batch.size(0))

                top1.update(acc1.mean().item(), x_batch.size(0))



                # compute gradient and do SGD step

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()



                # measure elapsed time

                batch_time.update(time.time() - end)

                end = time.time()



                if i % print_freq == 0:

                    print('Epoch: [{0}][{1}/{2}]\t'

                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'

                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'

                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'

                          'Acc@1 {top1.avg:.3f}'

                          .format(

                              epoch, i, len(train_loader), batch_time=batch_time,

                              data_time=data_time, loss=losses, top1=top1))



            print('Epoch time: {:.4f}min'.format((time.time() - epoch_time) / 60))

            return

        def train_model_boost(train_loader, model, criterion, optimizer, epoch, gpu=None, print_freq=100):

            batch_time = AverageMeter()

            data_time = AverageMeter()

            losses = AverageMeter()

            top1 = AverageMeter()



            model.train()



            epoch_time = time.time()

            end = time.time()



            for i, (x_batch, x_feats_batch, y_batch) in enumerate(train_loader):

                data_time.update(time.time() - end)

                if gpu is not None:

                    x_batch = x_batch.cuda(gpu, non_blocking=True)

                    x_feats_batch= x_feats_batch.cuda(gpu, non_blocking=True)

                y_batch = y_batch.cuda(gpu, non_blocking=True)



                output = model(x_batch, x_feats_batch)

                output = torch.nn.Tanh()(output)

                loss = criterion(output, y_batch)



                # measure accuracy

                acc1 = (y_batch == (output > 0.5).float()).float()

                losses.update(loss.item(), x_batch.size(0))

                top1.update(acc1.mean().item(), x_batch.size(0))



                # compute gradient and do SGD step

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()



                # measure elapsed time

                batch_time.update(time.time() - end)

                end = time.time()



                if i % print_freq == 0:

                    print('Epoch: [{0}][{1}/{2}]\t'

                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'

                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'

                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'

                          'Acc@1 {top1.avg:.3f}'

                          .format(

                              epoch, i, len(train_loader), batch_time=batch_time,

                              data_time=data_time, loss=losses, top1=top1))



            print('Epoch time: {:.4f}min'.format((time.time() - epoch_time) / 60))

            return

        def test_model(loader, model, gpu=None, return_true=False):

            model.eval()

            outputs = []



            with torch.no_grad():

                end = time.time()

                for i, (x_batch, x_feats_batch) in enumerate(loader):

                    if gpu is not None:

                        x_batch = x_batch.cuda(gpu, non_blocking=True)

                        x_feats_batch = x_feats_batch.cuda(gpu, non_blocking=True)



                    output = model(x_batch, x_feats_batch)



                    y_pred = output.detach()

                    outputs.append(sigmoid(y_pred.cpu().numpy())[:, 0])



            outputs = np.concatenate(outputs)

            return outputs

        def test_model_boost(loader, model, gpu=None, return_true=False):

            model.eval()

            outputs = []



            with torch.no_grad():

                end = time.time()

                for i, (x_batch, x_feats_batch) in enumerate(loader):

                    if gpu is not None:

                        x_batch = x_batch.cuda(gpu, non_blocking=True)

                        x_feats_batch = x_feats_batch.cuda(gpu, non_blocking=True)



                    output = model(x_batch, x_feats_batch)

                    output = torch.nn.Tanh()(output)

                    y_pred = output.detach()

                    outputs.append(y_pred.cpu().numpy())



            outputs = np.concatenate(outputs)

            return outputs



        # zoo

        class ModelRNN(nn.Module):

            def __init__(self, max_features, embedding_matrix, hidden_size, linear1_in, linear1_out, lockout, dropout):

                super(ModelRNN, self).__init__()

                self.max_features = max_features

                self.embedding_matrix = embedding_matrix

                self.hidden_size = hidden_size

                self.linear1_in = linear1_in

                self.linear1_out = linear1_out

                self.locked_dropout = LockedDropout(lockout)

                self.dropout = dropout



                embed_size = self.embedding_matrix.shape[1]



                self.embedding = nn.Embedding(self.max_features, embed_size)

                self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

                self.embedding.weight.requires_grad = False



                self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)

                self.gru = nn.GRU(hidden_size*2, hidden_size*2, bidirectional=True, batch_first=True)



                self.linear = nn.Linear(self.linear1_in, self.linear1_out)

                self.relu = nn.ReLU()

                # self.dropout = nn.Dropout(self.dropout)

                self.out = nn.Linear(self.linear1_out, 1)



            def forward(self, x, x_feats):

                h_embedding = self.embedding(x)

                # h_embedding = self.locked_dropout(h_embedding)



                h_lstm, _ = self.lstm(h_embedding)

                h_gru, _ = self.gru(h_lstm)



                avg_pool = torch.mean(h_gru, 1)

                max_pool, _ = torch.max(h_gru, 1)

                last_state = h_gru[:, -1, :]



                conc = torch.cat((avg_pool, max_pool, last_state, x_feats), 1)



                conc = self.relu(self.linear(conc))

                # conc = self.dropout(conc)

                out = self.out(conc)



                return out



        class ModelRNN_fc2(nn.Module):

            def __init__(self, max_features, embedding_matrix, hidden_size, linear1_in, linear1_out, linear2_out, lockout, dropout):

                super(ModelRNN_fc2, self).__init__()

                self.max_features = max_features

                self.embedding_matrix = embedding_matrix

                self.hidden_size = hidden_size

                self.linear1_in = linear1_in

                self.linear1_out = linear1_out

                self.linear2_out = linear2_out

                self.locked_dropout = LockedDropout(lockout)

                self.dropout = dropout



                embed_size = self.embedding_matrix.shape[1]



                self.embedding = nn.Embedding(self.max_features, embed_size)

                self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

                self.embedding.weight.requires_grad = False



                self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)

                self.gru = nn.GRU(hidden_size*2, hidden_size*2, bidirectional=True, batch_first=True)



                self.linear1 = nn.Linear(self.linear1_in, self.linear1_out)

                self.relu1 = nn.ReLU()

                self.linear2 = nn.Linear(self.linear1_out, self.linear2_out)

                self.relu2 = nn.ReLU()

                self.dropout = nn.Dropout(self.dropout)



                self.out = nn.Linear(self.linear2_out, 1)



            def forward(self, x, x_feats):

                h_embedding = self.embedding(x)

                # h_embedding = self.locked_dropout(h_embedding)



                h_lstm, _ = self.lstm(h_embedding)

                h_gru, _ = self.gru(h_lstm)



                avg_pool = torch.mean(h_gru, 1)

                max_pool, _ = torch.max(h_gru, 1)

                last_state = h_gru[:, -1, :]



                conc = torch.cat((avg_pool, max_pool, last_state, x_feats), 1)



                conc = self.relu1(self.linear1(conc))

                conc = self.dropout(conc)

                conc = self.relu2(self.linear2(conc))

                conc = self.dropout(conc)

                out = self.out(conc)



                return out



        class DPCNNTextClassifier(nn.Module):

            """

            DPCNN for sentences classification.

            """

            def __init__(self, max_features, embedding_matrix, hidden_size, input_dropout_p):

                super(DPCNNTextClassifier, self).__init__()

                self.max_features = max_features

                self.embedding_matrix = embedding_matrix

                self.hidden_size = hidden_size

                self.input_dropout_p = input_dropout_p



                embed_size = self.embedding_matrix.shape[1]



                self.embedding = nn.Embedding(self.max_features, embed_size)

                self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

                self.embedding.weight.requires_grad = False



                self.input_dropout = nn.Dropout(p=input_dropout_p)



                self.channel_size = 150

                self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, embed_size), stride=1)

                self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)

                self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)

                self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))

                self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))

                self.act_fun = nn.ReLU()

                self.linear_out = nn.Linear(self.channel_size, 1)



            def forward(self, input_var, lengths=None):

                embeded = self.embedding(input_var)

                embeded = self.input_dropout(embeded)

                batch, width, height = embeded.shape

                embeded = embeded.view((batch, 1, width, height))



                # Region embedding

                x = self.conv_region_embedding(embeded)

                x = self.padding_conv(x)

                x = self.act_fun(x)

                x = self.conv3(x)

                x = self.padding_conv(x)

                x = self.act_fun(x)

                x = self.conv3(x)



                while x.size()[-2] >= 2:

                    x = self._block(x)



                x = x.view(batch, self.channel_size)

                x = self.linear_out(x)



                return x



            def _block(self, x):

                # Pooling

                x = self.padding_pool(x)

                px = self.pooling(x)



                # Convolution

                x = self.padding_conv(px)

                x = F.relu(x)

                x = self.conv3(x)



                x = self.padding_conv(x)

                x = F.relu(x)

                x = self.conv3(x)



                # Short Cut

                x = x + px



                return x



        hidden_size = 90

        gru_len = hidden_size



        Routings = 4 #5

        Num_capsule = 5

        Dim_capsule = 5#16

        dropout_p = 0.25

        rate_drop_dense = 0.28

        LR = 0.001

        T_epsilon = 1e-7

        num_classes = 30

        class Embed_Layer(nn.Module):

            def __init__(self, embedding_matrix=None, vocab_size=None, embedding_dim=601):

                super(Embed_Layer, self).__init__()

                self.encoder = nn.Embedding(vocab_size + 1, embedding_dim)

                if use_pretrained_embedding:

                    # self.encoder.weight.data.copy_(t.from_numpy(np.load(embedding_path))) # 方法一，加载np.save的npy文件

                    self.encoder.weight.data.copy_(t.from_numpy(embedding_matrix))  # 方法二



            def forward(self, x, dropout_p=0.25):

                return nn.Dropout(p=dropout_p)(self.encoder(x))





        class GRU_Layer(nn.Module):

            def __init__(self):

                super(GRU_Layer, self).__init__()

                self.gru = nn.GRU(input_size=300,

                                  hidden_size=gru_len,

                                  bidirectional=True)

                '''

                自己修改GRU里面的激活函数及加dropout和recurrent_dropout

                如果要使用，把rnn_revised import进来，但好像是使用cpu跑的，比较慢

               '''

                # # if you uncomment /*from rnn_revised import * */, uncomment following code aswell

                # self.gru = RNNHardSigmoid('GRU', input_size=300,

                #                           hidden_size=gru_len,

                #                           bidirectional=True)



            # 这步很关键，需要像keras一样用glorot_uniform和orthogonal_uniform初始化参数

            def init_weights(self):

                ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)

                hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)

                b = (param.data for name, param in self.named_parameters() if 'bias' in name)

                for k in ih:

                    nn.init.xavier_uniform_(k)

                for k in hh:

                    nn.init.orthogonal_(k)

                for k in b:

                    nn.init.constant_(k, 0)



            def forward(self, x):

                return self.gru(x)





        # core caps_layer with squash func

        class Caps_Layer(nn.Module):

            def __init__(self, input_dim_capsule=gru_len * 2, num_capsule=Num_capsule, dim_capsule=Dim_capsule, \

                         routings=Routings, kernel_size=(9, 1), share_weights=True,

                         activation='default', **kwargs):

                super(Caps_Layer, self).__init__(**kwargs)



                self.num_capsule = num_capsule

                self.dim_capsule = dim_capsule

                self.routings = routings

                self.kernel_size = kernel_size  # 暂时没用到

                self.share_weights = share_weights

                if activation == 'default':

                    self.activation = self.squash

                else:

                    self.activation = nn.ReLU(inplace=True)



                if self.share_weights:

                    self.W = nn.Parameter(

                        nn.init.xavier_normal_(t.empty(1, input_dim_capsule, self.num_capsule * self.dim_capsule)))

                else:

                    self.W = nn.Parameter(

                        t.randn(BATCH_SIZE, input_dim_capsule, self.num_capsule * self.dim_capsule))  # 64即batch_size



            def forward(self, x):



                if self.share_weights:

                    u_hat_vecs = t.matmul(x, self.W)

                else:

                    print('add later')



                batch_size = x.size(0)

                input_num_capsule = x.size(1)

                u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule,

                                              self.num_capsule, self.dim_capsule))

                u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)  # 转成(batch_size,num_capsule,input_num_capsule,dim_capsule)

                b = t.zeros_like(u_hat_vecs[:, :, :, 0])  # (batch_size,num_capsule,input_num_capsule)



                for i in range(self.routings):

                    b = b.permute(0, 2, 1)

                    c = F.softmax(b, dim=2)

                    c = c.permute(0, 2, 1)

                    b = b.permute(0, 2, 1)

                    outputs = self.activation(t.einsum('bij,bijk->bik', (c, u_hat_vecs)))  # batch matrix multiplication

                    # outputs shape (batch_size, num_capsule, dim_capsule)

                    if i < self.routings - 1:

                        b = t.einsum('bik,bijk->bij', (outputs, u_hat_vecs))  # batch matrix multiplication

                return outputs  # (batch_size, num_capsule, dim_capsule)



            # text version of squash, slight different from original one

            def squash(self, x, axis=-1):

                s_squared_norm = (x ** 2).sum(axis, keepdim=True)

                scale = t.sqrt(s_squared_norm + T_epsilon)

                return x / scale



        class Capsule_Main(nn.Module):

            def __init__(self, embedding_matrix=None, vocab_size=None):

                super(Capsule_Main, self).__init__()

                self.embed_layer = Embed_Layer(embedding_matrix, vocab_size)

                self.gru_layer = GRU_Layer()

                # 【重要】初始化GRU权重操作，这一步非常关键，acc上升到0.98，如果用默认的uniform初始化则acc一直在0.5左右

                self.gru_layer.init_weights()

                self.caps_layer = Caps_Layer()

                self.dense_layer = Dense_Layer()



            def forward(self, content):

                content1 = self.embed_layer(content)

                content2, _ = self.gru_layer(

                    content1)  # 这个输出是个tuple，一个output(seq_len, batch_size, num_directions * hidden_size)，一个hn

                content3 = self.caps_layer(content2)

                output = self.dense_layer(content3)

                return output



        class Attention(nn.Module):

            def __init__(self, feature_dim, step_dim, bias=True, **kwargs):

                super(Attention, self).__init__(**kwargs)



                self.supports_masking = True



                self.bias = bias

                self.feature_dim = feature_dim

                self.step_dim = step_dim

                self.features_dim = 0



                weight = torch.zeros(feature_dim, 1)

                nn.init.xavier_uniform_(weight)

                self.weight = nn.Parameter(weight)



                if bias:

                    self.b = nn.Parameter(torch.zeros(step_dim))



            def forward(self, x, mask=None):

                feature_dim = self.feature_dim

                step_dim = self.step_dim



                eij = torch.mm(

                    x.contiguous().view(-1, feature_dim), 

                    self.weight

                ).view(-1, step_dim)



                if self.bias:

                    eij = eij + self.b



                eij = torch.tanh(eij)

                a = torch.exp(eij)



                if mask is not None:

                    a = a * mask



                a = a / torch.sum(a, 1, keepdim=True) + 1e-10



                weighted_input = x * torch.unsqueeze(a, -1)

                return torch.sum(weighted_input, 1)





        class CapsNet(nn.Module):

            def __init__(self, max_features, embedding_matrix, hidden_size, linear1_in):

                super(CapsNet, self).__init__()

                self.max_features = max_features

                self.embedding_matrix = embedding_matrix

                self.hidden_size = hidden_size

                self.linear1_in = linear1_in



                fc_layer = 16

                fc_layer1 = 16



                embed_size = self.embedding_matrix.shape[1]



                self.embedding = nn.Embedding(self.max_features, embed_size)

                self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

                self.embedding.weight.requires_grad = False



                self.embedding_dropout = nn.Dropout2d(0.1)

                self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)

                self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)



                self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

                self.bn = nn.BatchNorm1d(16, momentum=0.5)

                self.linear = nn.Linear(self.linear1_in, fc_layer1) #643:80 - 483:60 - 323:40

                self.relu = nn.ReLU()

                self.dropout = nn.Dropout(0.1)

                self.fc = nn.Linear(fc_layer**2,fc_layer)

                self.out = nn.Linear(fc_layer, 1)

                self.lincaps = nn.Linear(Num_capsule * Dim_capsule, 1)

                self.caps_layer = Caps_Layer()



            def forward(self, x, x_feats):

                h_embedding = self.embedding(x)



                h_lstm, _ = self.lstm(h_embedding)

                h_gru, _ = self.gru(h_lstm)



                ##Capsule Layer        

                content3 = self.caps_layer(h_gru)

                content3 = self.dropout(content3)

                batch_size = content3.size(0)

                content3 = content3.view(batch_size, -1)

                content3 = self.relu(self.lincaps(content3))



                # global average pooling

                avg_pool = torch.mean(h_gru, 1)

                # global max pooling

                max_pool, _ = torch.max(h_gru, 1)

                last_state = h_gru[:, -1, :]



                conc = torch.cat((content3, avg_pool, max_pool, last_state, x_feats), 1) # h_lstm_atten, h_gru_atten

                conc = self.relu(self.linear(conc))

                conc = self.bn(conc)

                conc = self.dropout(conc)



                out = self.out(conc)



                return out

        # TRAIN-----#

        gpu = 0

        batch_size = 512

        start_epoch = 0

        np.save('train_sequences.npy',train_sequences)

        np.save('training_labels.npy',training_labels)

        np.save('train_features.npy',train_features)

        np.save('test_sequences.npy',test_sequences)

        np.save('test_features.npy',test_features)

        print('PREPARE FOR PYTORCH')

        x_train = torch.tensor(train_sequences, dtype=torch.long).cuda()

        y_train = torch.tensor(training_labels[:, np.newaxis], dtype=torch.float32).cuda()

        x_train_feats = torch.tensor(train_features, dtype=torch.float32).cuda()

        print('FINISH TRAIN PREMINILARY FOR PYTORCH')

        x_train_data = torch.utils.data.TensorDataset(x_train, x_train_feats, y_train)

        train = torch.utils.data.DataLoader(x_train_data, batch_size=batch_size, shuffle=True)

        del x_train, x_train_feats, x_train_data, y_train

        gc.collect()

        print('FINISH TRAIN FOR PYTORCH')

        x_test = torch.tensor(test_sequences, dtype=torch.long).cuda()

        x_test_feats = torch.tensor(test_features, dtype=torch.float32).cuda()

        x_test_data = torch.utils.data.TensorDataset(x_test, x_test_feats)

        test = torch.utils.data.DataLoader(x_test_data, batch_size=batch_size, shuffle=False)



        del train_sequences, train_features, test_sequences, test_features

        gc.collect()



        del x_test, x_test_feats, x_test_data

        gc.collect()



        # fasttext-RNN2

        embed_google_index = load_embedding('google-news')

        embedding_matrix = build_embedding_matrix(embed_google_index, one=False)



        rnn_kwargs = {

            'max_features': num_words,

            'embedding_matrix': embedding_matrix,

            'hidden_size': 90,

            'linear1_in': 1082,

            'linear1_out': 70,

            'linear2_out':16,

            'lockout': 0.1,

            'dropout': 0.1,

        }

        del embedding_matrix

        gc.collect()

        epochs = 4

        model = ModelRNN_fc2(**rnn_kwargs)

        del rnn_kwargs

        gc.collect()

        model.cuda()

        criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")

        lr=0.003

        for epoch in range(start_epoch, epochs):

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            train_model(train, model, criterion, optimizer, epoch, gpu=gpu)

            lr = lr*0.8



        preds = np.zeros((len(test_idx)))

        preds = test_model(test, model, gpu=gpu)

        preds = preds.ravel()

        if split>0:

            best_score, _ = f1_smart(testing_labels, preds)

            print ('RNN2-googlenews :', best_score)   

        np.save('preds_googlenews.npy', preds)



        del model, criterion, optimizer, preds

        gc.collect()

        

        # paragram-google-RNN2

        embedding_matrix = build_concatenate_embedding_matrix('paragram',embed_google_index, two=False)

        del embed_google_index

        gc.collect()

        rnn_kwargs = {

            'max_features': num_words,

            'embedding_matrix': embedding_matrix,

            'hidden_size': hidden_size,

            'linear1_in': 543,

        }



        del embedding_matrix

        gc.collect()

        start_epoch = 0

        model = CapsNet(**rnn_kwargs)

        del rnn_kwargs

        gc.collect()

        model.cuda()

        criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")

        epochs = 3

        lr=0.003

        for epoch in range(start_epoch, epochs):

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            train_model(train, model, criterion, optimizer, epoch, gpu=gpu)

            lr = lr*0.8



        preds = np.zeros((len(test_idx)))

        preds = test_model(test, model, gpu=gpu)

        preds = preds.ravel()

        if split>0:

            best_score, _ = f1_smart(testing_labels, preds)

            print ('RNN2-paragram_google :', best_score)   

        np.save('preds_paragram_google.npy', preds)

        

        del train, test

        gc.collect()

        

        train_sequences = np.load('train_sequences.npy')

        #training_labels = np.load('training_labels.npy')

        train_features = np.load('train_features.npy')

        test_sequences = np.load('test_sequences.npy')

        test_features = np.load('test_features.npy')

        oof_pred_LR = np.load('oof_pred_LR.npy')



        training_labels_boost = training_labels - oof_pred_LR

        del oof_pred_LR

        gc.collect()

        print('PREPARE FOR PYTORCH')

        x_train = torch.tensor(train_sequences, dtype=torch.long).cuda()

        y_train = torch.tensor(training_labels_boost[:, np.newaxis], dtype=torch.float32).cuda()

        x_train_feats = torch.tensor(train_features, dtype=torch.float32).cuda()

        print('FINISH TRAIN PREMINILARY FOR PYTORCH')

        x_train_data = torch.utils.data.TensorDataset(x_train, x_train_feats, y_train)

        train = torch.utils.data.DataLoader(x_train_data, batch_size=batch_size, shuffle=True)

        del x_train, x_train_feats, x_train_data, y_train

        gc.collect()

        print('FINISH TRAIN FOR PYTORCH')

        x_test = torch.tensor(test_sequences, dtype=torch.long).cuda()

        x_test_feats = torch.tensor(test_features, dtype=torch.float32).cuda()

        x_test_data = torch.utils.data.TensorDataset(x_test, x_test_feats)

        test = torch.utils.data.DataLoader(x_test_data, batch_size=batch_size, shuffle=False)



        del train_sequences, train_features, test_sequences, test_features, training_labels_boost

        gc.collect()

        

        # glove-RNN2

        embed_glove_index = load_embedding('glove')

        #model = Word2Vec(all_text, size=100, window=5, min_count=10, workers=1)

        #del all_text

        #gc.collect()

        embedding_matrix = build_embedding_matrix(embed_glove_index, one=False)

        del model

        gc.collect()

        rnn_kwargs = {

            'max_features': num_words,

            'embedding_matrix': embedding_matrix,

            'hidden_size': 90,

            'linear1_in': 1082,

            'linear1_out': 70,

            'lockout': 0.1,

            'dropout': 0.1,

        }

        epochs = 3

        model = ModelRNN(**rnn_kwargs)

        del embedding_matrix, rnn_kwargs

        gc.collect()

        model.cuda()

        criterion = torch.nn.MSELoss()

        lr=0.003

        for epoch in range(start_epoch, epochs):

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            train_model_boost(train, model, criterion, optimizer, epoch, gpu=gpu)

            lr = lr*0.8



        preds = np.zeros((len(test_idx)))

        preds = test_model_boost(test, model, gpu=gpu)

        preds = preds.ravel()

        test_pred_LR = np.load('test_pred_LR.npy')

        preds = preds + test_pred_LR

        del test_pred_LR

        gc.collect()

        if split>0:

            best_score, _ = f1_smart(testing_labels, preds)

            print ('RNN2-glove :', best_score)   

        np.save('preds_glove.npy', preds)



        del model, criterion, optimizer, preds, train

        gc.collect()

        

        

        train_sequences = np.load('train_sequences.npy')

        #training_labels = np.load('training_labels.npy')

        train_features = np.load('train_features.npy')



        print('PREPARE FOR PYTORCH')

        x_train = torch.tensor(train_sequences, dtype=torch.long).cuda()

        y_train = torch.tensor(training_labels[:, np.newaxis], dtype=torch.float32).cuda()

        x_train_feats = torch.tensor(train_features, dtype=torch.float32).cuda()

        print('FINISH TRAIN PREMINILARY FOR PYTORCH')

        x_train_data = torch.utils.data.TensorDataset(x_train, x_train_feats, y_train)

        train = torch.utils.data.DataLoader(x_train_data, batch_size=batch_size, shuffle=True)

        del x_train, x_train_feats, x_train_data, y_train

        gc.collect()



        del train_sequences, train_features

        gc.collect()

        # glove+fasttext-RNN

        embedding_matrix = build_concatenate_embedding_matrix(embed_glove_index, 'wiki-news', one=False)

        del embed_glove_index

        gc.collect()

        

        

        rnn_kwargs = {

            'max_features': num_words,

            'embedding_matrix': embedding_matrix,

            'hidden_size': hidden_size,

            'linear1_in': 543,

        }

        epochs = 3

        model = CapsNet(**rnn_kwargs)

        del embedding_matrix, rnn_kwargs

        gc.collect()

        model.cuda()

        criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")

        lr=0.003

        for epoch in range(start_epoch, epochs):

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            train_model(train, model, criterion, optimizer, epoch, gpu=gpu)

            lr=lr*0.8



        preds = np.zeros((len(test_idx)))

        preds = test_model(test, model, gpu=gpu)

        preds = preds.ravel()

        if split>0:

            best_score, _ = f1_smart(testing_labels, preds)

            print ('RNN-glove+fasttext :', best_score)   

        np.save('preds_glove_fasttext.npy', preds)



        del model, criterion, optimizer, preds

        gc.collect()



        x_train = {'sparse_data_one': scipy.sparse.load_npz('train_char_features.npz'),

                'sparse_data_two': scipy.sparse.load_npz('train_name_bi.npz'),

                'sparse_data_three': scipy.sparse.load_npz('train_word_features.npz'),

                'feats': np.load('train_sparse_feats.npy'),

            }



        def sparseNN():   

            sparse_data_one = Input( shape=[x_train["sparse_data_one"].shape[1]], 

                dtype = 'float32',   sparse = True, name='sparse_data_one') 

            sparse_data_two = Input( shape=[x_train["sparse_data_two"].shape[1]], 

                dtype = 'float32',   sparse = True, name='sparse_data_two') 

            sparse_data_three = Input( shape=[x_train["sparse_data_three"].shape[1]], 

                dtype = 'float32',   sparse = True, name='sparse_data_three') 



            feats = Input(shape=[6], name="feats")



            x_one = Dense(200 , kernel_initializer=he_uniform(seed=0) )(sparse_data_one)    

            x_one = PReLU()(x_one)



            x_two = Dense(200 , kernel_initializer=he_uniform(seed=0) )(sparse_data_two)    

            x_two = PReLU()(x_two)



            x_three = Dense(200 , kernel_initializer=he_uniform(seed=0) )(sparse_data_three)    

            x_three = PReLU()(x_three)



            x = concatenate( [x_one, x_two, x_three, feats] )

            x = Dense(200 , kernel_initializer=he_uniform(seed=0) )(x)

            x = PReLU()(x)

            x = Dense(100 , kernel_initializer=he_uniform(seed=0) )(x)

            x = PReLU()(x)

            x= Dense(1, activation='sigmoid')(x)

            model = Model([sparse_data_one, sparse_data_two, sparse_data_three, feats], x)

            optimizer = Adam(.0005)

            model.compile(loss="binary_crossentropy", optimizer=optimizer)

            return model

        BATCH_SIZE = 1024

        epochs = 1

        sparse_nn = sparseNN()

        sparse_nn.fit(  x_train, training_labels, batch_size=BATCH_SIZE, epochs=1, verbose=1 )

        del x_train, training_labels

        gc.collect()

        x_test = {'sparse_data_one': scipy.sparse.load_npz('test_char_features.npz'),

                'sparse_data_two': scipy.sparse.load_npz('test_name_bi.npz'),

                'sparse_data_three': scipy.sparse.load_npz('test_word_features.npz'),

                'feats': np.load('test_sparse_feats.npy'),

            }

        preds = sparse_nn.predict(x_test)[:,0]

        np.save('preds_sparsenn.npy', preds)

    

import scipy

def do_sparses(dfs):

    def is_chinese(x):

        if re.search(u'[\u4e00-\u9fff]', x):

            return 1

        else: 

            return 0

    def get_sentiment(df):

        sid = SIA()

        df['nltk'] = df['question_text'].apply(lambda x: sid.polarity_scores(x))

        df['neg'] = df['nltk'].apply(lambda x: x['neg'])

        df['neu'] = df['nltk'].apply(lambda x: x['neu'])

        df['pos'] = df['nltk'].apply(lambda x: x['pos'])

        df['compound'] = df['nltk'].apply(lambda x: x['compound'])     

        df['chinese']= df['question_text'].apply(lambda x: is_chinese(x))

    def create_count_features(df_data):

        def lg(text):

            text = [x for x in text.split() if x!='']

            return len(text)

        df_data['nb_words_description'] = df_data['question_text'].apply(lg).astype(np.uint16)

        return df_data

    train = pd.read_csv(dfs[0])

    test = pd.read_csv(dfs[1])

    train_target = train['target'].values

    np.save('train_target.npy',train_target)

    print('AHA IM HERE fitting new tfidf !')

    TOKENIZER = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

    def tokenize(s):

        return TOKENIZER.sub(r' \1 ', s).split()

    def lemmatize_sentence(sentence):

        tokens=nltk.word_tokenize(sentence)

        lemmatizer = WordNetLemmatizer()

        lem = [lemmatizer.lemmatize(t) for t in tokens]

        return " ".join(lem)

    train['question_text_lemma'] = train['question_text'].apply(lambda x: lemmatize_sentence(x))

    tfidf_vectorizer = TfidfVectorizer(

        ngram_range=(1,3),

        tokenizer=tokenize,

        min_df=3,

        max_df=0.9,

        strip_accents='unicode',

        use_idf=True,

        smooth_idf=True,

        sublinear_tf=True,

        max_features = 100000

    ).fit(train['question_text_lemma'])

    print('FINISH fitting new tfidf !')

    X_tfidf = tfidf_vectorizer.transform(train['question_text_lemma'])

    scipy.sparse.save_npz('X_tfidf.npz', X_tfidf)

    del X_tfidf, train['question_text_lemma']

    gc.collect()

    print('FINISH transforming new tfidf in train!')

    test['question_text_lemma'] = test['question_text'].apply(lambda x: lemmatize_sentence(x))

    X_tfidf_test = tfidf_vectorizer.transform(test['question_text_lemma'])

    scipy.sparse.save_npz('X_tfidf_test.npz', X_tfidf_test)

    del X_tfidf_test, tfidf_vectorizer, test['question_text_lemma']

    gc.collect()

    

    

    char_vectorizer = TfidfVectorizer(

        sublinear_tf=True,

        strip_accents='unicode',

        analyzer='char',

        stop_words='english',

        ngram_range=(2, 4),

        max_features=30000)

    char_vectorizer.fit(train['question_text'])

    print('Char TFIDF 1/3')

    train_char_features = char_vectorizer.transform(train['question_text'])

    scipy.sparse.save_npz('train_char_features.npz', train_char_features)

    del train_char_features

    gc.collect()

    print('Char TFIDF 2/3')

    test_char_features = char_vectorizer.transform(test['question_text'])

    scipy.sparse.save_npz('test_char_features.npz', test_char_features)

    del char_vectorizer, test_char_features

    gc.collect()

    print('Char TFIDF 3/3')



    word_vectorizer = TfidfVectorizer(

        sublinear_tf=True,

        strip_accents='unicode',

        analyzer='word',

        token_pattern=r'\w{1,}',

        ngram_range=(1, 2),

        max_features=30000)

    word_vectorizer.fit(train['question_text'])

    print('Word TFIDF 1/3')

    train_word_features = word_vectorizer.transform(train['question_text'])

    scipy.sparse.save_npz('train_word_features.npz', train_word_features)

    del train_word_features

    gc.collect()

    print('Word TFIDF 2/3')

    test_word_features = word_vectorizer.transform(test['question_text'])

    scipy.sparse.save_npz('test_word_features.npz', test_word_features)

    del word_vectorizer, test_word_features

    gc.collect()

    print('Word TFIDF 3/3')   

    

    get_sentiment(train)

    train = create_count_features(train)

    np.save('train_sparse_feats.npy',train[['chinese','neg','neu','pos','compound','nb_words_description']].values)

    train.drop(['chinese','neg','neu','pos','compound','nb_words_description'],axis=1, inplace=True)

    

    get_sentiment(test)

    test = create_count_features(test)

    np.save('test_sparse_feats.npy',test[['chinese','neg','neu','pos','compound','nb_words_description']].values)

    test.drop(['chinese','neg','neu','pos','compound','nb_words_description'],axis=1, inplace=True)

    

    print('START LOADING TFIDF')

    train_target = np.load('train_target.npy')

    n_folds = 5

    test_pred_LR = 0

    oof_pred_LR = np.zeros([train_target.shape[0],])

    X_tfidf = scipy.sparse.load_npz('X_tfidf.npz')

    skf = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X_tfidf, train_target))

    del train_target,  X_tfidf

    gc.collect()

    for i, (train_index, val_index) in tqdm(enumerate(skf)):

        X_tfidf = scipy.sparse.load_npz('X_tfidf.npz')

        x_train, x_val = X_tfidf[list(train_index)], X_tfidf[list(val_index)]

        del X_tfidf

        gc.collect()

        train_target = np.load('train_target.npy')

        y_train, y_val = train_target[train_index], train_target[val_index]

        del train_target

        gc.collect()

        #classifier = LogisticRegression(C=5, solver='sag')

        classifier = Ridge()

        classifier.fit(x_train, y_train)

        del x_train, y_train

        gc.collect()

        val_preds = classifier.predict(x_val)

        X_tfidf_test = scipy.sparse.load_npz('X_tfidf_test.npz')

        preds = classifier.predict(X_tfidf_test)

        del X_tfidf_test,x_val

        gc.collect()

        test_pred_LR += 0.2*preds

        oof_pred_LR[val_index] = val_preds

        print(f1_score(y_val, val_preds > 0.26))

        del classifier, preds, val_preds

        gc.collect()

    np.save('oof_pred_LR.npy',oof_pred_LR)

    del oof_pred_LR

    gc.collect()

    np.save('test_pred_LR.npy',test_pred_LR)

    del test_pred_LR

    gc.collect()

    

    print('END Parallel')

    

    wordnet_lemmatizer = WordNetLemmatizer()

    def word_count(text, dc):

        text = set( text.split(' ') ) 

        for w in text:

            dc[w]+=1

    def remove_low_freq(text, dc):

        return ' '.join( [w for w in text.split() if w in dc] )

    stop_words = set(stopwords.words('english'))

    def create_bigrams(text):

        #try:

        text = np.unique( [ wordnet_lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words ] )

        lst_bi = []

        for combo in combinations(text, 2):

            cb1=combo[0]+combo[1]

            cb2=combo[1]+combo[0]

            in_dict=False

            if cb1 in word_count_dict_one:

                new_word = cb1

                in_dict=True

            if cb2 in word_count_dict_one:

                new_word = cb2

                in_dict=True

            if not in_dict:

                new_word = combo[0]+'___'+combo[1]

            if len(cb1)>=0:

                lst_bi.append(new_word)

        return ' '.join( lst_bi )

        #except:

        #    return ' '

    def create_bigrams_df(df):

        return df.apply( create_bigrams )

    cores = 2

    max_text_length=60###################

    min_df_one=5

    min_df_bi=5

    def parallelize_dataframe(df, func):

        df_split = np.array_split(df, cores)

        pool = Pool(cores)

        df = pd.concat(pool.map(func, df_split))

        pool.close()

        pool.join()

        return df



    word_count_dict_one = defaultdict(np.uint32)



    train['question_text'].apply(lambda x : word_count(x, word_count_dict_one) )

    rare_words = [key for key in word_count_dict_one if  word_count_dict_one[key]<min_df_one ]

    for key in rare_words :

        word_count_dict_one.pop(key, None)



    train['question_text']      = train['question_text'].apply( lambda x : remove_low_freq(x, word_count_dict_one) )

    word_count_dict_one=dict(word_count_dict_one)





    start_time = time.time()

    word_count_dict_bi=defaultdict(np.uint32)

    def word_count_bi(text):

        text =  text.split(' ') 

        for w in text:

            word_count_dict_bi[w]+=1



    train['question_textbi'] = train['question_text'].apply( lambda x : ' '.join( x.split()[5:] ))

    test['question_textbi'] = test['question_text'].apply( lambda x : ' '.join( x.split()[5:] ))



    train['name_bi']  = create_bigrams_df(train['question_textbi'])

    test['name_bi']  = create_bigrams_df(test['question_textbi'])

    train.drop('question_textbi',axis=1, inplace=True)

    test.drop('question_textbi',axis=1, inplace=True)

    gc.collect()

    train['name_bi'].apply(word_count_bi )

    rare_words = [key for key in word_count_dict_bi if  word_count_dict_bi[key]<min_df_bi ]

    for key in rare_words :

        word_count_dict_bi.pop(key, None)

    train['name_bi']      = train['name_bi'].apply( lambda x : remove_low_freq(x, word_count_dict_bi) )

    test['name_bi']      = test['name_bi'].apply( lambda x : remove_low_freq(x, word_count_dict_bi) )



    print('[{}] Finished CREATING BIGRAMS...'.format(time.time() - start_time))



    #####################################



    start_time = time.time()

    word_count_dict_bi = dict(word_count_dict_bi)

    vocabulary_one = word_count_dict_one.copy()

    vocabulary_bi = word_count_dict_bi.copy()

    for dc in [vocabulary_one,  vocabulary_bi]:

        cpt=0

        for key in dc:

            dc[key]=cpt

            cpt+=1

    print('[{}] Finished CREATING VOCABULARY ...'.format(time.time() - start_time))



    def tokenize(text):

        return [w for w in text.split()]

    start_time = time.time()

    vect_item_one            = CountVectorizer(vocabulary= vocabulary_one,   dtype=np.uint8, 

                                               tokenizer=tokenize, binary=True ) 

    train_item_one  = vect_item_one.fit_transform( train['question_text']  )

    scipy.sparse.save_npz('train_item_one.npz', train_item_one)

    

    

    

    del train_item_one, vocabulary_one

    gc.collect()

    test_item_one = vect_item_one.transform(     test['question_text']  )

    scipy.sparse.save_npz('test_item_one.npz', test_item_one)

    del test_item_one

    gc.collect()

    print('[{}] Finished Vectorizing Onegram Item Description'.format(time.time() - start_time))

    start_time = time.time()

    vect_name_bi           = CountVectorizer(vocabulary= vocabulary_bi,   dtype=np.uint8, 

                                         tokenizer=tokenize, binary=True ) 

    train_name_bi  = vect_name_bi.fit_transform( train['name_bi']  )

    scipy.sparse.save_npz('train_name_bi.npz', train_name_bi)

    del train['name_bi'], train_name_bi, vocabulary_bi

    gc.collect()

    print('[{}] Finished Vectorizing BiGram Name'.format(time.time() - start_time))

    

    test_name_bi = vect_name_bi.transform(     test['name_bi']  )

    scipy.sparse.save_npz('test_name_bi.npz', test_name_bi)

    del test_name_bi, word_count_dict_one, rare_words

    #time.sleep(10*60)

    



#     X_tfidf = scipy.sparse.load_npz('X_tfidf.npz')

#     #classifier = LogisticRegression(C=5, solver='sag')

#     print('LOADED TFIDF')

#     classifier = Ridge()

#     classifier.fit(X_tfidf, train_target)

#     del X_tfidf

#     gc.collect()

#     print('fitted tfidf')

#     X_tfidf_test = scipy.sparse.load_npz('X_tfidf_test.npz')

#     test_pred_LR = classifier.predict(X_tfidf_test)

#     del X_tfidf_test

#     gc.collect()

#     print('Predicted tfidf')

#     np.save(test_pred_LR, 'test_pred_LR.npy')

#     del test_pred_LR

#     gc.collect()

    

    print('END Parallel')



    



from multiprocessing import Pool

df_two = [train, test, True]

test_idx = list(test.qid.values)

del train, test

gc.collect()





pool = Pool(1)

pool.apply_async(do_sparses, [[f'{FILE_DIR}/train.csv', f'{FILE_DIR}/test.csv']])

do_rnns(df_two)

!ls
models_predictions = defaultdict(list)

models_predictions['RNN2-googlenews'] += list(np.load('preds_googlenews.npy'))  

models_predictions['RNN2-paragram_google'] += list(np.load('preds_paragram_google.npy')) 

models_predictions['RNN2-glove'] += list(np.load('preds_glove.npy'))  

models_predictions['RNN2-glove_fasttext'] += list(np.load('preds_glove_fasttext.npy'))

models_predictions['RNN2-sparsenn'] += list(np.load('preds_sparsenn.npy'))
submission_preds_df = pd.DataFrame(models_predictions)

submission_preds_df.corr()

mysubmission=pd.DataFrame()

mysubmission['qid'] = test_idx

preds = submission_preds_df.mean(axis=1)

#preds = submission_preds_df['RNN2-googlenews']*0.1 + submission_preds_df['RNN2-paragram_google']*0.2 + submission_preds_df['RNN2-glove']*0.35 + \

#        submission_preds_df['RNN2-glove_fasttext']*0.2 + submission_preds_df['RNN2-sparsenn']*0.15

mysubmission['prediction'] = preds > 0.44190952125936747 #0.7028983812108168 CV

mysubmission.to_csv('submission.csv', index=False)

print (mysubmission.shape)
mysubmission.head()