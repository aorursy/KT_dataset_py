# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

tqdm.pandas()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import string

import re    #for regex

import nltk

from nltk.corpus import stopwords

import spacy

from nltk import pos_tag

from nltk.stem.wordnet import WordNetLemmatizer 

from nltk.tokenize import word_tokenize

# Tweet tokenizer does not split at apostophes which is what we want

from nltk.tokenize import TweetTokenizer   

import time

import re



## Load data

train = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')

test = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')

print("Train shape : ",train.shape)

print("Test shape : ",test.shape)
## Building vocubulary from our Quest Data

def build_vocab(sentences, verbose =  True):

    """

    :param sentences: list of list of words

    :return: dictionary of words and their count

    """

    vocab = {}

    for sentence in tqdm(sentences, disable = (not verbose)):

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab



##Apply the vocab function to get the words and the corresponding counts

sentences = train["question_body"].progress_apply(lambda x: x.split()).values

vocab = build_vocab(sentences)

print({k: vocab[k] for k in list(vocab)[:20]})
%%time 

from gensim.models import KeyedVectors



news_path = '/kaggle/input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=False)
%%time

import operator 

## This is a common function to check coverage between our quest data and the word embedding

def check_coverage(vocab,embeddings_index):

    a = {}

    oov = {}

    k = 0

    i = 0

    for word in tqdm(vocab):

        try:

            a[word] = embeddings_index[word]

            k += vocab[word]

        except:



            oov[word] = vocab[word]

            i += vocab[word]

            pass



    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))

    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]



    return sorted_x
%%time

oov = check_coverage(vocab,embeddings_index)
## List 10 out of vocabulary word

oov[:10]
def decontract(text):

    text = re.sub(r"(W|w)on(\'|\’)t ", "will not ", text)

    text = re.sub(r"(C|c)an(\'|\’)t ", "can not ", text)

    text = re.sub(r"(Y|y)(\'|\’)all ", "you all ", text)

    text = re.sub(r"(Y|y)a(\'|\’)ll ", "you all ", text)

    text = re.sub(r"(I|i)(\'|\’)m ", "i am ", text)

    text = re.sub(r"(A|a)isn(\'|\’)t ", "is not ", text)

    text = re.sub(r"n(\'|\’)t ", " not ", text)

    text = re.sub(r"(\'|\’)re ", " are ", text)

    text = re.sub(r"(\'|\’)d ", " would ", text)

    text = re.sub(r"(\'|\’)ll ", " will ", text)

    text = re.sub(r"(\'|\’)t ", " not ", text)

    text = re.sub(r"(\'|\’)ve ", " have ", text)

    return text

train["question_body"] = train["question_body"].progress_apply(lambda x: decontract(x))

sentences = train["question_body"].apply(lambda x: x.split())

vocab = build_vocab(sentences)

oov = check_coverage(vocab,embeddings_index)
def clean_apostrophes(x):

    apostrophes = ["’", "‘", "´", "`"]

    for s in apostrophes:

        x = re.sub(s, "'", x)

    return x



train["question_body"] = train["question_body"].progress_apply(lambda x: clean_apostrophes(x))

sentences = train["question_body"].apply(lambda x: x.split())

vocab = build_vocab(sentences)

oov = check_coverage(vocab,embeddings_index)
# clean weird / special characters



letter_mapping = {'\u200b':' ', 'ũ': "u", 'ẽ': 'e', 'é': "e", 'á': "a", 'ķ': 'k', 

                  'ï': 'i', 'Ź': 'Z', 'Ż': 'Z', 'Š': 'S', 'Π': ' pi ', 'Ö': 'O', 

                  'É': 'E', 'Ñ': 'N', 'Ž': 'Z', 'ệ': 'e', '²': '2', 'Å': 'A', 'Ā': 'A',

                  'ế': 'e', 'ễ': 'e', 'ộ': 'o', '⧼': '<', '⧽': '>', 'Ü': 'U', 'Δ': 'delta',

                  'ợ': 'o', 'İ': 'I', 'Я': 'R', 'О': 'O', 'Č': 'C', 'П': 'pi', 'В': 'B', 'Φ': 

                  'phi', 'ỵ': 'y', 'օ': 'o', 'Ľ': 'L', 'ả': 'a', 'Γ': 'theta', 'Ó': 'O', 'Í': 'I',

                  'ấ': 'a', 'ụ': 'u', 'Ō': 'O', 'Ο': 'O', 'Σ': 'sigma', 'Â': 'A', 'Ã': 'A', 'ᗯ': 'w', 

                  'ᕼ': "h", "ᗩ": "a", "ᖇ": "r", "ᗯ": "w", "O": "o", "ᗰ": "m", "ᑎ": "n", "ᐯ": "v", "н": 

                  "h", "м": "m", "o": "o", "т": "t", "в": "b", "υ": "u",  "ι": "i","н": "h", "č": "c", "š":

                  "s", "ḥ": "h", "ā": "a", "ī": "i", "à": "a", "ý": "y", "ò": "o", "è": "e", "ù": "u", "â": 

                  "a", "ğ": "g", "ó": "o", "ê": "e", "ạ": "a", "ü": "u", "ä": "a", "í": "i", "ō": "o", "ñ": "n",

                  "ç": "c", "ã": "a", "ć": "c", "ô": "o", "с": "c", "ě": "e", "æ": "ae", "î": "i", "ő": "o", "å": 

                  "a", "Ä": "A","&gt":" greater than","&lt" :"lesser than", "(not" : "not" , "});":"",">" :"greater","<":"lesser" ,"$":"dollar","\\\\":" ","\\": " "} 



def clean_special_chars(text):

    new_text = ''

    for i in range(len(text)):

        if i in letter_mapping:

            c = letter_mapping[i]

        else:

            c = text[i]

        new_text += c

    return new_text



train["question_body"] = train["question_body"].progress_apply(lambda x: clean_special_chars(x))

sentences = train["question_body"].apply(lambda x: x.split())

vocab = build_vocab(sentences)

oov = check_coverage(vocab,embeddings_index)

# remove useless punctuations



useless_punct = ['च', '不', 'ঢ়', '平', 'ᠠ', '錯', '判', '∙',

                 '言', 'ς', 'ل', '្', 'ジ', 'あ', '得', '水', 'ь', '◦', '创', 

                 '康', '華', 'ḵ', '☺', '支', '就', '„', '」', '어', '谈', '陈', '团', '腻', '权', 

                 '年', '业', 'マ', 'य', 'ا', '売', '甲', '拼', '˂', 'ὤ', '贯', '亚', 'ि', '放', 'ʻ', 'ទ', 'ʖ', 

                 '點', '્', '発', '青', '能', '木', 'д', '微', '藤', '̃', '僕', '妒', '͜', 'ន', 'ध', '이', '希', '特',

                 'ड', '¢', '滢', 'ส', '나', '女', 'క', '没', '什', 'з', '天', '南', 'ʿ', 'ค', 'も', '凰', '步', '籍', '西',

                 'ำ', '−', 'л', 'ڤ', 'ៃ', '號', 'ص', 'स', '®', 'ʋ', '批', 'រ', '치', '谢', '生', '道', '═', '下', '俄', 'ɖ',

                 '觀', 'வ', '—', 'ی', '您', '♥', '一', 'や', '⊆', 'ʌ', '語', 'ี', '兴', '惶', '瀛', '狐', '⁴', 'प', '臣', 'ద',

                 '―', 'ì', 'ऌ', 'ీ', '自', '信', '健', '受', 'ɨ', '시', 'י', 'ছ', '嬛', '湾', '吃', 'ち', 'ड़', '反', '红', '有',

                 '配', 'ে', 'ឯ', '宮', 'つ', 'μ', '記', '口', '℅ι', 'ो', '狸', '奇', 'о', 'ट', '聖', '蘭', '読', 'ū', '標', '要', 

                 'ត', '识', 'で', '汤', 'ま', 'ʀ', '局', 'リ', '्', 'ไ', '呢', '工', 'ल', '沒', 'τ', 'ិ', 'ö', 'せ', '你', 'ん', 'ュ', 

                 '枚', '部', '大', '罗', 'হ', 'て', '表', '报', '攻', 'ĺ', 'ฉ', '∩', '宝', '对', '字', '文', '这', '∑', '髪', 'り', '่', '능',

                 '罢', '내', '阻', '为', '菲', 'ي', 'न', 'ί', 'ɦ', '開', '†', '茹', '做', '東', 'ত', 'に', 'ت', '晓', '키', '悲', 'સ', 

                 '好', '›', '上', '存', '없', '하', '知', 'ធ', '斯', ' ', '授', 'ł', '傳', '兰', '封', 'ோ', 'و', 'х', 'だ', '人', '太', 

                 '品', '毒', 'ᡳ', '血', '席', '剔', 'п', '蛋', '王', '那', '梦', 'ី', '彩', '甄', 'и', '柏', 'ਨ', '和', '坊', '⌚', '广', 

                 '依', '∫', 'į', '故', 'ś', 'ऊ', '几', '日', 'ک', '音', '×', '”', '▾', 'ʊ', 'ज', 'ด', 'ठ', 'उ', 'る', '清', 'ग', 'ط',

                 'δ', 'ʏ', '官', '∛', '়', '้', '男', '骂', '复', '∂', 'ー', '过', 'য', '以', '短', '翻', 'র', '教', '儀', 'ɛ', '‹', 'へ', 

                 '¾', '合', '学', 'ٌ', '학', '挑', 'ष', '比', '体', 'م', 'س', 'អ', 'ת', '訓', '∀', '迎', 'វ', 'ɔ', '٨', '▒', '化', 'చ', '‛', 

                 'প', 'º', 'น', '업', '说', 'ご', '¸', '₹', '儿', '︠', '게', '骨', 'ท', 'ऋ', 'ホ', '茶', '는', 'જ', 'ุ', '羡', '節', 'ਮ', 

                 'উ', '番', 'ড়', '讲', 'ㅜ', '등', '伟', 'จ', '我', 'ล', 'す', 'い', 'ញ', '看', 'ċ', '∧', 'भ', 'ઘ', 'ั', 'ម', '街', 'ય', 

                 '还', '鰹', 'ខ', 'ు', '訊', 'म', 'ю', '復', '杨', 'ق', 'त', '金', '味', 'ব', '风', '意', '몇', '佬', '爾', '精', '¶', 

                 'ం', '乱', 'χ', '교', 'ה', '始', 'ᠰ', '了', '个', '克', '্', 'ห', '已', 'ʃ', 'わ', '新', '译', '︡', '本', 'ง', 'б', 'け', 

                 'ి', '明', '¯', '過', 'ك', 'ῥ', 'ف', 'ß', '서', '进', 'ដ', '样', '乐', '寧', '€', 'ณ', 'ル', '乡', '子', 'ﬁ', 'ج', '慕',

                 '–', 'ᡵ', 'Ø', '͡', '제', 'Ω', 'ប', '絕', '눈', 'फ', 'ম', 'గ', '他', 'α', 'ξ', '§', 'ஜ', '黎', 'ね', '복', 'π', 'ú', '鸡',

                 '话', '会', 'ক', '八', '之', '북', 'ن', '¦', '가', 'ו', '恋', '地', 'ῆ', '許', '产', 'ॡ', 'ش', '़', '野', 'ή', 'ɒ', '啧',

                 'យ', '᠌', 'ᠨ', 'ب', '皎', '老', '公', '☆', 'व', 'ি', 'ល', 'ر', 'គ', '행', 'ង', 'ο', '让', 'ំ', 'λ', 'خ', 'ἰ', '家',

                 'ট', 'ब', '理', '是', 'め', 'र', '√', '기', 'ν', '玉', '한', '入', 'ד', '别', 'د', 'ะ', '电', 'ા', '♫', 'ع', 'ં', '堵',

                 '嫉', '伊', 'う', '千', '관', '篇', 'क', '非', '荣', '粵', '瑜', '英', '를', '美', '条', '`', '宋', '←', '수', '後', '•',

                 '³', 'ी', '고', '肉', '℃', 'し', '漢', '싱', 'ϵ', '送', 'ه', '落', 'న', 'ក', 'க', 'ℇ', 'た', 'ះ', '中', '射', '♪', '符',

                 'ឃ', '谷', '分', '酱', 'び', 'থ', 'ة', 'г', 'σ', 'と', '楚', '胡', '饭', 'み', '禮', '主', '直', '÷', '夢', 'ɾ', 'চ', '⃗',

                 '統', '高', '顺', '据', 'ら', '頭', 'よ', '最', 'ా', 'ੁ', '亲', 'ស', '花', '≡', '眼', '病', '…', 'の', '發', 'ா', '汝',

                 '★', '氏', 'ร', '景', 'ᡠ', '读', '件', '仲', 'শ', 'お', 'っ', 'پ', 'ᡤ', 'ч', '♭', '悠', 'ं', '六', '也', 'ռ', 'য়', '恐', 

                 'ह', '可', '啊', '莫', '书', '总', 'ষ', 'ք', '̂', '간', 'な', '此', '愛', 'ర', 'ใ', '陳', 'Ἀ', 'ण', '望', 'द', '请', '油',

                 '露', '니', 'ş', '宗', 'ʍ', '鳳', 'अ', '邋', '的', 'ព', '火', 'ा', 'ก', '約', 'ட', '章', '長', '商', '台', '勢', 'さ',

                 '국', 'Î', '簡', 'ई', '∈', 'ṭ', '經', '族', 'ु', '孫', '身', '坑', 'স', '么', 'ε', '失', '殺', 'ž', 'ર', 'が', '手',

                 'ា', '心', 'ਾ', '로', '朝', '们', '黒', '欢', '早', '️', 'া', 'आ', 'ɸ', '常', '快', '民', 'ﷺ', 'ូ', '遢', 'η', '国', 

                 '无', '江', 'ॠ', '「', 'ন', '™', 'ើ', 'ζ', '紫', 'ె', 'я', '“', '♨', '國', 'े', 'อ', '∞', 

                  '\n', "{\n', '}\n", "=&gt;", '}\n\n', '-&gt;', '\n\ni', '&lt;','/&gt;\n','{\n\n','\\','|','&','\\n\\n',"\\appendix"]

useless_punct.remove(' ')



def remove_useless_punct(text):

    return re.sub(f'{"|".join(useless_punct)}', '', text)

train["question_body"] = train["question_body"].progress_apply(lambda x: remove_useless_punct(x))

sentences = train["question_body"].apply(lambda x: x.split())

vocab = build_vocab(sentences)

oov = check_coverage(vocab,embeddings_index)
## ReLoad data

train = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')

test = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')

print("Train shape : ",train.shape)

print("Test shape : ",test.shape)
def clean_text(x):



    x = str(x)

    for punct in "/-'":

        x = x.replace(punct, ' ')

    for punct in '&':

        x = x.replace(punct, f' {punct} ')

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        x = x.replace(punct, '')

    return x



train["question_body"] = train["question_body"].progress_apply(lambda x: clean_text(x))

sentences = train["question_body"].apply(lambda x: x.split())

vocab = build_vocab(sentences)



oov = check_coverage(vocab,embeddings_index)
oov[:500]
## Checking how the numbers are present in the crawl embedding .

'1234567'in embeddings_index
'14528' in embeddings_index
import re



def clean_numbers(x):



    x = re.sub('[0-9]{5,}', '12345', x)

    x = re.sub('[0-9]{4}', '1234', x)

    x = re.sub('[0-9]{3}', '123', x)

    x = re.sub('[0-9]{2}', '12', x)

    return x



train["question_body"] = train["question_body"].progress_apply(lambda x: clean_numbers(x))

sentences = train["question_body"].apply(lambda x: x.split())

vocab = build_vocab(sentences)

oov = check_coverage(vocab,embeddings_index)
oov[:20]
def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re





mispell_dict = {'colour':'color',

                'centre':'center',

                'didnt':'did not',

                'doesnt':'does not',

                'isnt':'is not',

                'shouldnt':'should not',

                'favourite':'favorite',

                'travelling':'traveling',

                'counselling':'counseling',

                'theatre':'theater',

                'cancelled':'canceled',

                'labour':'labor',

                'organisation':'organization',

                'wwii':'world war 2',

                'citicise':'criticize',

                'instagram': 'social medium',

                'whatsapp': 'social medium',

                'snapchat': 'social medium'



                }

mispellings, mispellings_re = _get_mispell(mispell_dict)



def replace_typical_misspell(text):

    def replace(match):

        return mispellings[match.group(0)]



    return mispellings_re.sub(replace, text)



train["question_body"] = train["question_body"].progress_apply(lambda x: replace_typical_misspell(x))

sentences = train["question_body"].progress_apply(lambda x: x.split())

to_remove = ['a','to','of','and']

sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]

vocab = build_vocab(sentences)

oov = check_coverage(vocab,embeddings_index)
import pickle
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')





def load_embeddings(path):

    with open(path,'rb') as f:

        emb_arr = pickle.load(f)

    return emb_arr



def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    unknown_words = []

    

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            unknown_words.append(word)

    return embedding_matrix, unknown_words
## Reload the data 

train = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')

test = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')

print("Train shape : ",train.shape)

print("Test shape : ",test.shape)
GLOVE_EMBEDDING_PATH = '/kaggle/input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl' 
import operator 



def check_coverage(vocab,embeddings_index):

    a = {}

    oov = {}

    k = 0

    i = 0

    for word in tqdm(vocab):

        try:

            a[word] = embeddings_index[word]

            k += vocab[word]

        except:



            oov[word] = vocab[word]

            i += vocab[word]

            pass



    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))

    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]



    return sorted_x



def build_vocab(sentences, verbose =  True):

    """

    :param sentences: list of list of words

    :return: dictionary of words and their count

    """

    vocab = {}

    for sentence in tqdm(sentences, disable = (not verbose)):

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab
tic = time.time()

glove_embeddings = load_embeddings(GLOVE_EMBEDDING_PATH)

print(f'loaded {len(glove_embeddings)} word vectors in {time.time()-tic}s')
vocab = build_vocab(list(train['question_body'].apply(lambda x:x.split())))

oov = check_coverage(vocab,glove_embeddings)

oov[:10]
def decontract(text):

    text = re.sub(r"(W|w)on(\'|\’)t ", "will not ", text)

    text = re.sub(r"(C|c)an(\'|\’)t ", "can not ", text)

    text = re.sub(r"(Y|y)(\'|\’)all ", "you all ", text)

    text = re.sub(r"(Y|y)a(\'|\’)ll ", "you all ", text)

    text = re.sub(r"(I|i)(\'|\’)m ", "i am ", text)

    text = re.sub(r"(A|a)isn(\'|\’)t ", "is not ", text)

    text = re.sub(r"n(\'|\’)t ", " not ", text)

    text = re.sub(r"(\'|\’)re ", " are ", text)

    text = re.sub(r"(\'|\’)d ", " would ", text)

    text = re.sub(r"(\'|\’)ll ", " will ", text)

    text = re.sub(r"(\'|\’)t ", " not ", text)

    text = re.sub(r"(\'|\’)ve ", " have ", text)

    return text



def clean_text(x):



    x = str(x)

    for punct in "/-'":

        x = x.replace(punct, ' ')

    for punct in '&':

        x = x.replace(punct, f' {punct} ')

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        x = x.replace(punct, '')

    return x



def clean_numbers(x):



    x = re.sub('[0-9]{5,}', '12345', x)

    x = re.sub('[0-9]{4}', '1234', x)

    x = re.sub('[0-9]{3}', '123', x)

    x = re.sub('[0-9]{2}', '12', x)

    return x



def preprocess(x):

    x= decontract(x)

    x=clean_text(x)

    x=clean_number(x)

    return x
train["question_body"] = train["question_body"].progress_apply(lambda x: decontract(x))

sentences = train["question_body"].apply(lambda x: x.split())

vocab = build_vocab(sentences)

oov = check_coverage(vocab,glove_embeddings)
train["question_body"] = train["question_body"].progress_apply(lambda x: clean_text(x))

sentences = train["question_body"].apply(lambda x: x.split())

vocab = build_vocab(sentences)

oov = check_coverage(vocab,glove_embeddings)
oov[:500]
'1234567' in glove_embeddings
train["question_body"] = train["question_body"].progress_apply(lambda x: clean_numbers(x))

sentences = train["question_body"].apply(lambda x: x.split())

vocab = build_vocab(sentences)

oov = check_coverage(vocab,glove_embeddings)
oov[:20]