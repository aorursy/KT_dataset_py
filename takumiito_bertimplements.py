import sys

# copy our file into the working directory

sys.path.insert(0, "../input/pytorch-pretrained-BERT/pytorch-pretrained-BERT/pytorch-pretrained-bert/")
import torch

from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch

from pytorch_pretrained_bert import BertTokenizer, BertModel
batch_size = 256



n_seeds = 1

n_splits = 10

n_epochs = 15

EMBED_SIZE = 300

SUBGROUP_NEGATIVE_WEIGHT_COEF = 1

BACKGROUND_POSITIVE_WEIGHT_COEF = 0



ENSEMBLE_START_EPOCH = 3



MAX_LEN = 220



EMB_DROPOUT = 0.3

MIDDLE_DROPOUT = 0.3



BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'

# 分散表現がuncasedだった場合は、単語を全て小文字で扱う

BERT_DO_LOWER = 'uncased' in BERT_MODEL_PATH



WORK_DIR = "../working/"



batch_size = 32



OUT_DROPOUT = 0.3



BERT_HIDDEN_SIZE = 768
from contextlib import contextmanager

import os



# 標準出力をnullに変換

@contextmanager

def suppress_stdout():

    # nullの書き込み

    with open(os.devnull, "w") as devnull:

        # 標準出力をnullに変換

        old_stdout = sys.stdout

        sys.stdout = devnull

        try:  

            yield

        finally:

            sys.stdout = old_stdout
import pandas as pd

raw_train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

#test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
import gc

AUX_COLUMNS = ['target', 'severe_toxicity','obscene','identity_attack','insult','threat']

train_data = raw_train[:100000]

train_text = train_data['comment_text']

y_target = (train_data['target'] >= 0.5).astype('float32')

#print(y_target)

y_aux_target = (train_data[AUX_COLUMNS] >= 0.5).astype('float32')

#x_test = test['comment_text'][:300]

#print(y_aux_target)

#print(len(y_target.values))

#print(y_aux_target.shape)

#y_target = torch.tensor(y_target.values.reshape((-1, 1)), dtype=torch.float32).cuda()

#y_aux_target = torch.tensor(y_aux_target.values, dtype=torch.float32).cuda()

y_label = pd.concat([y_target, y_aux_target], axis=1)

#print(y_label)

#print(train_text)
del raw_train, train_data, y_target, y_aux_target

gc.collect()
import re

import emoji

import unicodedata



# 特定文字の変換器を作成

CUSTOM_TABLE = str.maketrans(

    {

        "\xad": None,

        "\x7f": None,

        "\x10": None,

        "\x9d": None,

        "\xa0": None,

        "\ufeff": None,

        "\u200b": None,

        "\u200e": None,

        "\u202a": None,

        "\u202c": None,

        "\uf0d8": None,

        "\u2061": None,

        "‘": "'",

        "’": "'",

        "`": "'",

        "“": '"',

        "”": '"',

        "«": '"',

        "»": '"',

        "ɢ": "G",

        "ɪ": "I",

        "ɴ": "N",

        "ʀ": "R",

        "ʏ": "Y",

        "ʙ": "B",

        "ʜ": "H",

        "ʟ": "L",

        "ғ": "F",

        "ᴀ": "A",

        "ᴄ": "C",

        "ᴅ": "D",

        "ᴇ": "E",

        "ᴊ": "J",

        "ᴋ": "K",

        "ᴍ": "M",

        "Μ": "M",

        "ᴏ": "O",

        "ᴘ": "P",

        "ᴛ": "T",

        "ᴜ": "U",

        "ᴡ": "W",

        "ᴠ": "V",

        "ĸ": "K",

        "в": "B",

        "м": "M",

        "н": "H",

        "т": "T",

        "ѕ": "S",

        "—": "-",

        "–": "-",

    }

)



# 下品な単語の規制後と規制前の単語の辞書を作成

WORDS_REPLACER = [

    ("sh*t", "shit"),

    ("s**t", "shit"),

    ("f*ck", "fuck"),

    ("fu*k", "fuck"),

    ("f**k", "fuck"),

    ("f*****g", "fucking"),

    ("f***ing", "fucking"),

    ("f**king", "fucking"),

    ("p*ssy", "pussy"),

    ("p***y", "pussy"),

    ("pu**y", "pussy"),

    ("p*ss", "piss"),

    ("b*tch", "bitch"),

    ("bit*h", "bitch"),

    ("h*ll", "hell"),

    ("h**l", "hell"),

    ("cr*p", "crap"),

    ("d*mn", "damn"),

    ("stu*pid", "stupid"),

    ("st*pid", "stupid"),

    ("n*gger", "nigger"),

    ("n***ga", "nigger"),

    ("f*ggot", "faggot"),

    ("scr*w", "screw"),

    ("pr*ck", "prick"),

    ("g*d", "god"),

    ("s*x", "sex"),

    ("a*s", "ass"),

    ("a**hole", "asshole"),

    ("a***ole", "asshole"),

    ("a**", "ass"),

]



# 下品な単語の規制部分の特殊文字を無効化し、大文字小文字を区別しない判別器を作成

REGEX_REPLACER = [

    (re.compile(pat.replace("*", "\*"), flags=re.IGNORECASE), repl)

    for pat, repl in WORDS_REPLACER

]



# 空白の判別器を作成

RE_SPACE = re.compile(r"\s")

RE_MULTI_SPACE = re.compile(r"\s+")



# Unicodeから異体字セレクタ（直前の文字の種類を変えるコード）をkeyとする辞書を作成

NMS_TABLE = dict.fromkeys(

    i for i in range(sys.maxunicode + 1) if unicodedata.category(chr(i)) == "Mn"

)



# 特定のUnicode（学習で使えない文字？）を別の文字で置換する辞書作成

HEBREW_TABLE = {i: "א" for i in range(0x0590, 0x05FF)}

ARABIC_TABLE = {i: "ا" for i in range(0x0600, 0x06FF)}

CHINESE_TABLE = {i: "是" for i in range(0x4E00, 0x9FFF)}

KANJI_TABLE = {i: "ッ" for i in range(0x2E80, 0x2FD5)}

HIRAGANA_TABLE = {i: "ッ" for i in range(0x3041, 0x3096)}

KATAKANA_TABLE = {i: "ッ" for i in range(0x30A0, 0x30FF)}



TABLE = dict()

TABLE.update(CUSTOM_TABLE)

TABLE.update(NMS_TABLE)

# Non-english languages

TABLE.update(CHINESE_TABLE)

TABLE.update(HEBREW_TABLE)

TABLE.update(ARABIC_TABLE)

TABLE.update(HIRAGANA_TABLE)

TABLE.update(KATAKANA_TABLE)

TABLE.update(KANJI_TABLE)



# 絵文字の判別器を作成

EMOJI_REGEXP = emoji.get_emoji_regexp()



# 絵文字のエイリアスを文章化

UNICODE_EMOJI_MY = {

    k: f" EMJ {v.strip(':').replace('_', ' ')} "

    for k, v in emoji.UNICODE_EMOJI_ALIAS.items()

}



# 文章内の絵文字をエイリアスの文章に変換

def my_demojize(string: str) -> str:

    # sub関数でmatchした単語を、それを含む絵文字のエイリアスの文章に置換

    def replace(match):

        return UNICODE_EMOJI_MY.get(match.group(0), match.group(0))



    # テストデータの文章中の絵文字を、絵文字のエイリアスの文章に置換し、そこから異体字セレクタを削除

    return re.sub("\ufe0f", "", EMOJI_REGEXP.sub(replace, string))



# 文章中の特定の特殊単語を解析用に変換

def normalize(text: str) -> str:

    #text_len = len(text)

    #print(text)

    # 文章内の絵文字をエイリアスの文章に変換

    text = my_demojize(text)



    # 空白を全てスペースに変換

    text = RE_SPACE.sub(" ", text)

    # 文字表現（Unicode）の規格を統一化

    text = unicodedata.normalize("NFKD", text)

    # 文章中のTABLE_keyをvalueに変換

    text = text.translate(TABLE)

    # 連続した空白を一つのスペースに変換し、文章の両端の空白を削除

    text = RE_MULTI_SPACE.sub(" ", text).strip()



    # 下品な単語の規制された部分を修復

    for pattern, repl in REGEX_REPLACER:

        text = pattern.sub(repl, text)

        

    #if text_len != len(text):

        #print(text + "\n")

    

    return text



# 文章の特殊単語を一般単語の変換

train_text = train_text.apply(lambda x: normalize(x))
del UNICODE_EMOJI_MY, EMOJI_REGEXP, TABLE, HEBREW_TABLE, ARABIC_TABLE, CHINESE_TABLE, KANJI_TABLE, HIRAGANA_TABLE, KATAKANA_TABLE, NMS_TABLE, RE_MULTI_SPACE, RE_SPACE, REGEX_REPLACER, WORDS_REPLACER, CUSTOM_TABLE

gc.collect()
#if "EMJ" in crawl_emb_dict:

#    print("EMJ")

#if "א" in crawl_emb_dict:

#    print("HEB")

#if "ا" in crawl_emb_dict:

#    print("ARA")

#if "是" in crawl_emb_dict:

#    print("CHI")

#if "ッ" in crawl_emb_dict:

#    print("JAP")
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', '\n', '\r', "'", "'", 'θ', '÷', 'α', 'β', '∅', 'π', '₹', '´']



def clean_text(x: str) -> str:

    for punct in puncts:

        if punct in x:

            # 特定文字の両側に空白付ける

            x = x.replace(punct, ' {} '.format(punct))

    return x



import re

def clean_numbers(x):

    return re.sub('\d+', ' ', x)



# 句読点の両側にスペースを付与

train_text = train_text.apply(lambda x: clean_text(x))



# 文章から数字削除

train_text = train_text.apply(lambda x: clean_numbers(x))



# Bertの語彙を読み込み

#with open(BERT_MODEL_PATH + 'vocab.txt', 'r') as f:

#    raw_dict = f.readlines()



# データ内の改行を削除と重複行の削除

#crawl_emb_dict = set([t.replace('\n', '') for t in raw_dict])

#print(len(crawl_emb_dict))

    

import joblib

# ファイルデータ（単語の分散表現の辞書）を並列処理形式で保存

with open('../input/reducing-oov-of-crawl300d2m-no-appos-result/jigsaw-crawl-300d-2M.joblib', 'rb') as f:

    crawl_emb_dict = joblib.load(f)

crawl_emb_dict = set(crawl_emb_dict.keys())

#print(crawl_emb_dict)

#for k in list(crawl_emb_dict)[:10]:

#    print({k:crawl_emb_dict[k]})



# グーグルが禁止している単語（禁句）集を取得

with open('../input/googleprofanitywords/google-profanity-words/google-profanity-words/profanity.js', 'r') as f:

    p_words = f.readlines()

    

set_puncts = set(puncts)

#print(set_puncts)



# データ内の改行を削除と重複行の削除

p_word_set = set([t.replace('\n', '') for t in p_words])

#print(p_word_set)
del puncts, p_words#, raw_dict

gc.collect()
import operator

from typing import Dict, List

from tqdm import tqdm_notebook as tqdm



def check_coverage(texts, embeddings_word: Dict) -> List[str]:

    known_words = []

    unknown_words = []

    for text in tqdm(texts):

        text = text.split()

        for word in text:

            if word in embeddings_word:

                known_words.append(word)

                #print(word)

            else:

                unknown_words.append(word)



    print('Found embeddings for {:.2%} of vocab'.format(float(len(set(known_words))) / (float(len(set(known_words))) + float(len(set(unknown_words))))))

    print('Found embeddings for {:.2%} of all text'.format(float(len(known_words)) / (float(len(known_words)) + float(len(unknown_words)))))



    return set(unknown_words)



# ３通りのStemmer（接尾辞除去=語幹検出アルゴリズム）用の関数を呼び出し

from nltk.stem import PorterStemmer

p_stemmer = PorterStemmer()

from nltk.stem.lancaster import LancasterStemmer

l_stemmer = LancasterStemmer()

from nltk.stem import SnowballStemmer

s_stemmer = SnowballStemmer("english")



import copy

def edits1(word):

    """

    wordの編集距離1の単語のリストを返す

    """

    letters    = 'abcdefghijklmnopqrstuvwxyz'

    # 単語を左右２つに分割したパターンを作成

    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]

    # 単語から一文字を消したパターンを作成

    deletes    = [L + R[1:]               for L, R in splits if R]

    # 単語から隣同士の文字を一組入れ替えたパターンを作成

    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]

    # 単語内の一文字を別の文字（アルファベット中の全ての文字）にそれぞれ変換したパターンを作成

    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]

    # 単語のどこかに別の文字（アルファベット中の全ての文字）を加えたパターンを作成

    inserts    = [L + c + R               for L, R in splits for c in letters]

    return set(deletes + transposes + replaces + inserts)



def known(words, embed): 

    "The subset of `words` that appear in the dictionary of WORDS."

    # 単語が辞書内に入っていたら返す

    return set(w for w in words if w in embed)



# 単語の誤字脱字を復元

def spellcheck(word, word_rank_dict):

    # 単語に一文字入れたり、隣同士の文字を入れ替えたりした物の内、単語辞書にある物の中から、一番文字数が少ない文字列を返す

    return min(known(edits1(word), word_rank_dict), key=lambda word_rank_dict:word_rank_dict)





import unicodedata

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "",

                 "€": "e", "™": "tm", "√": " sqrt ", "×": "x",

                 "²": "2", "—": "-", "–": "-", "’": "'",

                 "_": "-", "`": "'", '“': '"', '”': '"',

                 '“': '"', "£": "e", '∞': 'infinity',

                 'θ': 'theta', '÷': '/', 'α': 'alpha',

                 '•': '.', 'à': 'a', '−': '-', 'β': 'beta',

                 '∅': '', '³': '3', 'π': 'pi', 'SB91':'senate bill','tRump':'trump','utmterm':'utm term','FakeNews':'fake news','Gʀᴇat':'great','ʙᴏᴛtoᴍ':'bottom','washingtontimes':'washington times','garycrum':'gary crum','htmlutmterm':'html utm term','RangerMC':'car','TFWs':'tuition fee waiver','SJWs':'social justice warrior','Koncerned':'concerned','Vinis':'vinys','Yᴏᴜ':'you','Trumpsters':'trump','Trumpian':'trump','bigly':'big league','Trumpism':'trump','Yoyou':'you','Auwe':'wonder','Drumpf':'trump','utmterm':'utm term','Brexit':'british exit','utilitas':'utilities','ᴀ':'a', '😉':'wink','😂':'joy','😀':'stuck out tongue', 'theguardian':'the guardian','deplorables':'deplorable', 'theglobeandmail':'the globe and mail', 'justiciaries': 'justiciary','creditdation': 'Accreditation','doctrne':'doctrine','fentayal': 'fentanyl','designation-': 'designation','CONartist' : 'con-artist','Mutilitated' : 'Mutilated','Obumblers': 'bumblers','negotiatiations': 'negotiations','dood-': 'dood','irakis' : 'iraki','cooerate': 'cooperate','COx':'cox','racistcomments':'racist comments','envirnmetalists': 'environmentalists',}

def process_stemmer(texts, embed):

    oov_word_set = set()

    new_texts = []

    for text in tqdm(texts):

        text = text.split()

        new_text = ""

        for word in text:

            #print(word)

            # 分散表現辞書から色々な表現で変換した文中の単語の分散表現を検索

            if word in embed:

                new_text += word + " "

                #print("embed:",word)

                continue



            # 単語を全て小文字化した物で検索

            if word.lower() in embed:

                new_text += word.lower() + " "

                #print("lower:",word)

                continue



            # 単語を全て大文字化した物で検索

            if word.upper() in embed:

                new_text += word.upper() + " "

                #print("upper:",word)

                continue



            # 単語の頭文字だけを大文字化した物で検索

            if word.capitalize() in embed:

                new_text += word.capitalize() + " "

                #print("cap:",word)

                continue



            # 特殊文字の分散表現を検索

            corr_word = punct_mapping.get(word, None)

            if corr_word is not None:

                new_text += corr_word + " "

                #print("punct:",word)

                continue



            try:

                # PorterStemmerアルゴリズムで抽出した単語の語幹で検索

                vector = p_stemmer.stem(word)

            except:

                # 失敗したら、文字コードを変更して再実行

                vector = p_stemmer.stem(word.decode('utf-8'))

            if vector in embed:

                new_text += vector + " "

                #print("p_st:",vector)

                continue

                

            try:

                # LancasterStemmerアルゴリズムで抽出した単語の語幹で検索

                vector = l_stemmer.stem(word)

            except:

                vector = l_stemmer.stem(word.decode('utf-8'))

            if vector in embed:

                new_text += vector + " "

                #print("l_st:",vector)

                continue



            try:

                # SnowballStemmerアルゴリズムで抽出した単語の語幹で検索

                vector = s_stemmer.stem(word)

            except:

                vector = s_stemmer.stem(word.decode('utf-8'))

            if vector in embed:

                new_text += vector + " "

                #print("s_st:",vector)

                continue



            # 単語の分散表現が検索できなかった単語を記録

            oov_word_set.add(word)

            new_text += word + " "

            #print("oov:",word)

        if len(new_text.strip()) == 0:

            print(text)

            print("0:None!")

            new_text += "0" + " "

            

        new_texts.append(new_text.strip())

            

        #print(new_texts)

    return new_texts, oov_word_set



def process_small_capital(texts, embed, oov_set):

    oov_word_set = set()

    new_texts = []

    for text in tqdm(texts):

        text = text.split()

        new_text = ""

        for word in text:

            #print(word)

            if word not in oov_set:

                new_text += word + " "

                continue



            char_list = []

            any_small_capitial = False

            # 単語を一文字ずつ取得

            for char in word:

                try:

                    # 文字に割り振られている名前（"a"="LATIN SMALL LETTER A"）を取得

                    uni_name = unicodedata.name(char)

                except ValueError:

                    continue



                # 文字がラテン文字だった場合

                if 'LATIN SMALL LETTER' or 'LATIN CAPITAL LETTER' in uni_name:

                    # 文字に割り振られた名前の最後（文字が大文字化した物）を取得

                    char = uni_name[-1]

                    any_small_capitial = True

                # キリル文字の"ғ"は、"F"に変換

                if 'CYRILLIC SMALL LETTER GHE WITH STROKE' in uni_name:

                    char = 'F'

                    any_small_capitial = True



                char_list.append(char)



            # 単語内の全ての文字が、名前が割り当てられていない、ラテン文字ではなく、キリル文字の'ғ'でもない場合

            if not any_small_capitial:

                oov_word_set.add(word)

                new_text += word + " "

                #print("oov_small_cap:",word)

                continue



            # 変換した文字を一つの単語に戻す

            legit_word = ''.join(char_list)



            if legit_word in embed:

                new_text += legit_word + " "

                #print("embed:",legit_word)

                continue



            if legit_word.lower() in embed:

                new_text += legit_word.lower() + " "

                #print("lower:",legit_word)

                continue



            if legit_word.upper() in embed:

                new_text += legit_word.upper() + " "

                #print("upper:",legit_word)

                continue



            if legit_word.capitalize() in embed:

                new_text += legit_word.capitalize() + " "

                #print("cap:",legit_word)

                continue



            corr_word = punct_mapping.get(legit_word, None)

            if corr_word is not None:

                new_text += corr_word + " "

                #print("punct:",legit_word)

                continue



            try:

                vector = p_stemmer.stem(legit_word)

            except:

                vector = p_stemmer.stem(legit_word.decode('utf-8'))

            if vector in embed:

                new_text += vector + " "

                #print("p_st:",vector)

                continue



            try:

                vector = l_stemmer.stem(legit_word)

            except:

                vector = l_stemmer.stem(legit_word.decode('utf-8'))

            if vector in embed:

                new_text += vector + " "

                #print("l_st:",vector)

                continue



            try:

                vector = s_stemmer.stem(legit_word)

            except:

                vector = s_stemmer.stem(legit_word.decode('utf-8'))

            if vector in embed:

                new_text += vector + " "

                #print("s_st:",vector)

                continue



            oov_word_set.add(word)

            new_text += word + " "

            #print("oov:",word)

            

        if len(new_text.strip()) == 0:

            print(text)

            print("0:None!")

            new_text += "0" + " "



        new_texts.append(new_text.strip())

        

        #print(new_texts)

    return new_texts, oov_word_set



def process_spellcheck(texts, embed, oov_set):

    oov_word_set = set()

    new_texts = []

    for text in tqdm(texts):

        text = text.split()

        new_text = ""

        for word in text:

            if word not in oov_set:

                new_text += word + " "

                continue



            try:

                vector = spellcheck(word, embed)

            except:

                oov_word_set.add(word)

                new_text += word + " "

                #print("oov:",word)

                continue

            if vector is not None:

                new_text += vector + " "

                #print("original:",word)

                #print("miss_sp:",vector)

                continue



            oov_word_set.add(word)

            new_text += word + " "

            #print("oov:",word)

            

        if len(new_text.strip()) == 0:

            print(text)

            print("0:None!")

            new_text += "0" + " "

            

        new_texts.append(new_text.strip())

            

        #print(new_texts)

    return new_texts, oov_word_set



from nltk import TweetTokenizer

# Tweet専用の解析ツールを作成

# reduce_len：単語の長さの標準化（短縮化）をするかどうかを設定

tknzr = TweetTokenizer(reduce_len=True)

def twitter_stemmer(texts, embed, oov_set):

    oov_word_set = set()

    new_texts = []

    for text in tqdm(texts):

        text = text.split()

        new_text = ""

        for word in text:

            if word not in oov_set:

                new_text += word + " "

                continue

            

            tokens = tknzr.tokenize(word)

            if (tokens[0] == "'" or '"') and len(tokens) > 1:

                word = tokens[1]

                #print("tokens_1:")

            else:

                word = tokens[0]

                #print("tokens_0:")

            #print(word)

            # 分散表現辞書から色々な表現で変換した文中の単語の分散表現を検索

            if word in embed:

                new_text += word + " "

                #print("embed:",word)

                continue



            # 単語を全て小文字化した物で検索

            if word.lower() in embed:

                new_text += word.lower() + " "

                #print("lower:",word)

                continue



            # 単語を全て大文字化した物で検索

            if word.upper() in embed:

                new_text += word.upper() + " "

                #print("upper:",word)

                continue



            # 単語の頭文字だけを大文字化した物で検索

            if word.capitalize() in embed:

                new_text += word.capitalize() + " "

                #print("cap:",word)

                continue



            # 特殊文字の分散表現を検索

            corr_word = punct_mapping.get(word, None)

            if corr_word is not None:

                new_text += corr_word + " "

                #print("punct:",word)

                continue



            try:

                # PorterStemmerアルゴリズムで抽出した単語の語幹で検索

                vector = p_stemmer.stem(word)

            except:

                # 失敗したら、文字コードを変更して再実行

                vector = p_stemmer.stem(word.decode('utf-8'))

            if vector in embed:

                new_text += vector + " "

                #print("p_st:",vector)

                continue

                

            try:

                # LancasterStemmerアルゴリズムで抽出した単語の語幹で検索

                vector = l_stemmer.stem(word)

            except:

                vector = l_stemmer.stem(word.decode('utf-8'))

            if vector in embed:

                new_text += vector + " "

                #print("l_st:",vector)

                continue



            try:

                # SnowballStemmerアルゴリズムで抽出した単語の語幹で検索

                vector = s_stemmer.stem(word)

            except:

                vector = s_stemmer.stem(word.decode('utf-8'))

            if vector in embed:

                new_text += vector + " "

                #print("s_st:",vector)

                continue



            # 単語の分散表現が検索できなかった単語を記録

            oov_word_set.add(word)

            new_text += word + " "

            #print("oov:",word)

        if len(new_text.strip()) == 0:

            print("0:None!")

            print(text)

            new_text += "0" + " "

            

        new_texts.append(new_text.strip())

            

        #print(new_texts)

    return new_texts, oov_word_set



def bytes_to_unicode():

    # ラテン文字を表すUnicodeを取得

    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))

    cs = bs[:]

    n = 0

    # 0~255のUnicodeを取得

    for b in range(2**8):

        if b not in bs:

            # 文字以外のUnicode（PCの命令コード）も取得

            bs.append(b)

            # ラテン文字拡張A（PCの命令コード+256に位置する文字）のUnicodeを取得

            cs.append(2**8+n)

            n += 1

    # 配列内のUnicodeを単語に変換

    cs = [chr(n) for n in cs]

    return dict(zip(bs, cs))



# 単語の先頭文字から二文字ずつのタプルを取得

def get_pairs(word):

    pairs = set()

    prev_char = word[0]

    for char in word[1:]:

        pairs.add((prev_char, char))

        prev_char = char

    return pairs



# ファイルからmerge文字取得

MERGES_PATH = "../input/transformer-tokenizers/gpt2/merges.txt"

bpe_data = open(MERGES_PATH, encoding='utf-8').read().split('\n')[1:-1]

# merge文字のfirstとsecondをタプル化

bpe_merges = [tuple(merge.split()) for merge in bpe_data]

# keyがmerge文字のタブル、valueが0からのidの辞書作成

bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

def bpe(token):

    word = tuple(token)

    # 単語の先頭文字から二文字ずつのタプルを取得

    pairs = get_pairs(word)



    if not pairs:

        return token



    while True:

        # 単語中でマッチしたmerge文字のタプルから、idが一番小さいものを取得

        bigram = min(pairs, key = lambda pair: bpe_ranks.get(pair, float('inf')))

        if bigram not in bpe_ranks:

            break

        first, second = bigram

        new_word = []

        i = 0

        while i < len(word):

            try:

                # i番目の文字から後ろのmerge文字のfirstのidを取得

                j = word.index(first, i)

                # i番目の文字からmerge文字のfirstの直前までの文字列を取得

                new_word.extend(word[i:j])

                i = j

            except:

                # merge文字のfirstが単語に含まれてない場合は、そのまま取得

                new_word.extend(word[i:])

                break



            # merge文字のfirstとsecondの二文字が連続で続いた時も取得

            if word[i] == first and i < len(word)-1 and word[i+1] == second:

                new_word.append(first+second)

                i += 2

            else:

                new_word.append(word[i])

                i += 1

        new_word = tuple(new_word)

        word = new_word

        #print("new_word:",new_word)

        if len(word) == 1:

            break

        else:

            pairs = get_pairs(word)

    word = ' '.join(word)

    return word



def merge_spellcheck(texts, embed, oov_set):

    # key（ラテン文字とPC命令コード）、value（ラテン文字とラテン文字拡張）である辞書を取得

    byte_encoder = bytes_to_unicode()

    # 文章を単語（解析するパーツ）に分ける文字の判別器

    #pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d|[^('s)('t)('re)('ve)('m)('ll)('d)]+""")

    oov_word_set = set()

    new_texts = []

    for text in tqdm(texts):

        text = text.split()

        new_text = ""

        for pre_word in text:

            if pre_word not in oov_set:

                new_text += pre_word + " "

                continue

            #print("original:",pre_word)

            bpe_tokens = []

            token_list = re.split("('s)|('t)|('re)|('ve)|('m)|('ll)|('d)", pre_word)

            token_list = [x for x in token_list if (x is not None) and (x != "")]

            #print("token_list:",token_list)

            # 単語中の単語をmerge文字で細分化

            for token in token_list:

                #print("token:",token)

                # 単語中のラテン文字とPC命令コードをラテン文字拡張Aに変換した文字のみの単語を生成

                token = ''.join(byte_encoder[b] for b in token.encode('utf-8'))

                # merge文字によって分かち書きした単語を保存

                bpe_tokens.extend(bpe_token for bpe_token in bpe(token).split(' '))

                if (len(bpe_tokens) == 1) and (bpe_tokens[0] == token) and (len(token_list) == 0):

                    oov_word_set.add(word)

                    new_text += word + " "

                    #print("no_tokens:")

                    continue

                if len(bpe_tokens) == 1:

                    bpe_tokens.append(" ")



                for word in bpe_tokens:

                    #print("word:",word)

                    # 分散表現辞書から色々な表現で変換した文中の単語の分散表現を検索

                    if word in embed:

                        new_text += word + " "

                        #print("embed:",word)

                        continue



                    # 単語を全て小文字化した物で検索

                    if word.lower() in embed:

                        new_text += word.lower() + " "

                        #print("lower:",word)

                        continue



                    # 単語を全て大文字化した物で検索

                    if word.upper() in embed:

                        new_text += word.upper() + " "

                        #print("upper:",word)

                        continue



                    # 単語の頭文字だけを大文字化した物で検索

                    if word.capitalize() in embed:

                        new_text += word.capitalize() + " "

                        #print("cap:",word)

                        continue



                    # 特殊文字の分散表現を検索

                    corr_word = punct_mapping.get(word, None)

                    if corr_word is not None:

                        new_text += corr_word + " "

                        #print("punct:",word)

                        continue



                    try:

                        # PorterStemmerアルゴリズムで抽出した単語の語幹で検索

                        vector = p_stemmer.stem(word)

                    except:

                        # 失敗したら、文字コードを変更して再実行

                        vector = p_stemmer.stem(word.decode('utf-8'))

                    if vector in embed:

                        new_text += vector + " "

                        #print("p_st:",vector)

                        continue



                    try:

                        # LancasterStemmerアルゴリズムで抽出した単語の語幹で検索

                        vector = l_stemmer.stem(word)

                    except:

                        vector = l_stemmer.stem(word.decode('utf-8'))

                    if vector in embed:

                        new_text += vector + " "

                        #print("l_st:",vector)

                        continue



                    try:

                        # SnowballStemmerアルゴリズムで抽出した単語の語幹で検索

                        vector = s_stemmer.stem(word)

                    except:

                        vector = s_stemmer.stem(word.decode('utf-8'))

                    if vector in embed:

                        new_text += vector + " "

                        #print("s_st:",vector)

                        continue



                    # 単語の分散表現が検索できなかった単語を記録

                    oov_word_set.add(word)

                    new_text += word + " "

                    #print("oov:",word)

        if len(new_text.strip()) == 0:

            print(text)

            print("0:None!")

            new_text += "0" + " "

            

        new_texts.append(new_text.strip())

            

        #print(new_texts)

    return new_texts, oov_word_set



def head(enumerable, n=10):

    #print(enumerable)

    for i, item in enumerate(enumerable):

        print(str(i) + '\n',item)

        if i > n:

            return



print("only_data:")

# 分散表現の辞書に登録されていない単語を取得

oov = check_coverage(train_text, crawl_emb_dict)

head(oov)



# テストデータの単語に接尾辞除去を用いて、分散表現の辞書作成

train_text, oov_stemer = process_stemmer(train_text, crawl_emb_dict)

#print(train_text)

print("stemmer_data:")

oov = check_coverage(train_text, crawl_emb_dict)

head(oov)

#print(oov_stemer)



# 単語中の文字を解析して、分散表現を検索

train_text, oov_small_capital = process_small_capital(train_text, crawl_emb_dict, oov_stemer)

print("small_capital_data:")

oov = check_coverage(train_text, crawl_emb_dict)

head(oov)

#print(oov_small_capital)



# 単語の誤字脱字を解析し、分散表現を検索

#train_text_test = train_text

train_text, oov_spell = process_spellcheck(train_text, crawl_emb_dict, oov_small_capital)

print("spellcheck_data:")

oov = check_coverage(train_text, crawl_emb_dict)

head(oov)

#print(oov_spell)



# twitterの分かち書きを使って、単語を変換し、分散表現を検索

#train_text_test = train_text

train_text, oov_twitter = twitter_stemmer(train_text, crawl_emb_dict, oov_spell)

print("twitter_stemmer_data:")

oov = check_coverage(train_text, crawl_emb_dict)

head(oov)

#print(oov_twitter)



# merge文字を使って、単語を変換し、分散表現を検索

#train_text_test = train_text

train_text, oov_mearge = merge_spellcheck(train_text, crawl_emb_dict, oov_twitter)

print("merge_data:")

oov = check_coverage(train_text, crawl_emb_dict)

head(oov)
del oov, oov_stemer, oov_small_capital, oov_spell, oov_mearge, oov_twitter, bpe_data, bpe_merges, bpe_ranks

gc.collect()
# 文章の特徴データ生成

def sentence_fetures(text):

    word_list = text.split()

    #print(word_list)

    # 単語数

    word_count = len(word_list)

    # 大文字を含む単語の数

    n_upper = len([word for word in word_list if any([c.isupper() for c in word])])

    # 含まれる単語の種類

    n_unique = len(set(word_list))

    # ビックリマークの数

    n_ex = word_list.count('!')

    #print(n_ex)

    # クエスチョンマークの数

    n_que = word_list.count('?')

    # 特殊文字（句読点）の数

    n_puncts = len([word for word in word_list if word in set_puncts])

    # 禁句の数

    n_prof = len([word for word in word_list if word in p_word_set])

    # unknown単語の数

    n_oov = len([word for word in word_list if word not in crawl_emb_dict])

    

    return word_count, n_upper, n_unique, n_ex, n_que, n_puncts, n_prof, n_oov



from collections import defaultdict

sentence_feature_cols = ['word_count', 'n_upper', 'n_unique', 'n_ex', 'n_que', 'n_puncts', 'n_prof', 'n_oov']

# key不要のvalueがlist型の辞書を作成

feature_dict = defaultdict(list)

#print(raw_train)

for text in train_text:

    #print(text)

    # 文章の特徴データを取得

    feature_list = sentence_fetures(text)

    for i_feature, feature_name in enumerate(sentence_feature_cols):

        # keyを特徴の名前、valueを文章の特徴値とした辞書を作成

        feature_dict[sentence_feature_cols[i_feature]].append(feature_list[i_feature])

        

sentence_df = pd.DataFrame.from_dict(feature_dict)

#print(sentence_df['word_count'])

# 各特徴データを単語の数で平均化

for col in ['n_upper', 'n_unique', 'n_ex', 'n_que', 'n_puncts', 'n_prof', 'n_oov']:

    sentence_df[col + '_ratio'] = sentence_df[col] / sentence_df['word_count']

#print(sentence_df)

    

# 特徴データからunknownワード関連のデータを削除

sentence_feature_mat = sentence_df.drop(columns=['n_oov', 'n_oov_ratio']).values

#print(sentence_feature_mat)



# 特徴データを標準化（平均0、分散1の集合化）

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(sentence_feature_mat)

sentence_feature_mat = scaler.transform(sentence_feature_mat)

#print(sentence_feature_mat)

#print(scaler.var_)
del sentence_feature_cols, feature_dict, sentence_df, scaler

gc.collect()
#sentence_feature_mat = torch.tensor(sentence_feature_mat, dtype=torch.float32).cuda()

sentence_feature_size = sentence_feature_mat.shape[-1]

y_label_size = y_label.shape[-1]

#print(y_label_size)

#print(sentence_feature_mat.shape)

#print(sentence_feature_size)

import numpy as np

label_data = torch.tensor(np.concatenate([y_label, sentence_feature_mat], axis=1), dtype=torch.float32).cuda()

#print(y_target)

#print(y_target.shape)

n_samples = len(train_text)

train_size = int(n_samples * 0.4)

valid_size = int(n_samples * 0.3)

#train_index = list(range(0, train_size))

#valid_index = list(range(train_size, train_size + valid_size))

#test_index = list(range(train_size + valid_size, n_samples))

#print(n_samples)

train = pd.Series(list(train_text[0:train_size]))

train_label = label_data[0:train_size]

valid = pd.Series(list(train_text[train_size:train_size + valid_size]))

valid_label = label_data[train_size:train_size + valid_size]

test = pd.Series(list(train_text[train_size + valid_size:n_samples]))

test_label = label_data[train_size + valid_size:n_samples]

#print(len(train))

#print(len(valid))

#print(len(test))

#print(valid_label)

#print(label_data.shape)
del y_label, n_samples, train_size, valid_size, sentence_feature_mat#, train_index, valid_index, test_index

gc.collect()
tokenizer = BertTokenizer.from_pretrained(

    BERT_MODEL_PATH, cache_dir=None, do_lower_case=BERT_DO_LOWER)
# 文章データを整地

def tokenize(text, max_len, tokenizer):

    # vocab.txtに乗っている文字だけを文章内に取得

    tokenized_text = tokenizer.tokenize(text)[:max_len-2]

    return ["[CLS]"]+tokenized_text+["[SEP]"]



# 以前のpandasファイルの進捗バーを初期化

#import pandas as pd

#from tqdm import tqdm

#tqdm.pandas()

#from tqdm import tqdm_notebook as tqdm

#from tqdm._tqdm_notebook import tqdm_notebook

#tqdm_notebook.pandas()

train = train.apply(lambda x: tokenize(x, MAX_LEN, tokenizer))

valid = valid.apply(lambda x: tokenize(x, MAX_LEN, tokenizer))

test = test.apply(lambda x: tokenize(x, MAX_LEN, tokenizer))
# 文章データをvocab.txt内の分散表現に変換

train = train.apply(lambda x: tokenizer.convert_tokens_to_ids(x))

valid = valid.apply(lambda x: tokenizer.convert_tokens_to_ids(x))

test = test.apply(lambda x: tokenizer.convert_tokens_to_ids(x))
del tokenizer

gc.collect()



torch.cuda.empty_cache()

torch.cuda.memory_allocated()
import shutil

# printなどの標準出力メッセージを非表示

with suppress_stdout():

    # 事前学習済みのBertモデル（TendorflowModel）をPytorch上に作成

    convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(

        # Bert（TendorflowModel）の事前学習時（checkpoint）の重みをモデルに設定

        # ファイルの名前で出力は既に決定されている？

        # ファイル内の変数名に'squad'がある時、事前学習の出力はClassification（最初の時系列データの出力）になる？

        BERT_MODEL_PATH + 'bert_model.ckpt',

        BERT_MODEL_PATH + 'bert_config.json',

        # 重みを設定したBERTモデルの設定ファイルをWORK_DIR上に保存

        WORK_DIR + 'pytorch_model.bin')



# 使用したBertの設定ファイル（PytorchModel）を別のフォルダに保存

shutil.copyfile(BERT_MODEL_PATH + 'bert_config.json', WORK_DIR + 'bert_config.json')
gc.collect()

torch.cuda.empty_cache()

torch.cuda.memory_allocated()
from torch import nn

from torch.nn import functional as F

class NeuralNet(nn.Module):

    def __init__(self, num_aux_targets, sentence_feature_size):

        super(NeuralNet, self).__init__()

        # WORK_DIR上の設定ファイルで事前学習済みのBERTを作成

        #self.bert = BertModel.from_pretrained(WORK_DIR)

        # bertモデルは学習しない

        #for param in self.bert.parameters():

            #param.requires_grad=False

            #print(f'bert-{param.requires_grad}')

        self.dropout = nn.Dropout(OUT_DROPOUT)

        

        # Bertの特徴を学習

        self.before_linear = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)

        self.before_linear2 = nn.Linear(BERT_HIDDEN_SIZE, 50)

        

        # sentence_featureの特徴を学習

        self.sentence_feature_linear = nn.Linear(sentence_feature_size, sentence_feature_size)

        

        # Bertとsentence_featureの抱き合わせを学習

        n_hidden = 50 + sentence_feature_size

        self.mix_linear = nn.Linear(n_hidden, n_hidden)

        

        # 出力

        self.linear_out = nn.Linear(n_hidden, 1)

        #nn.init.xavier_uniform_(self.linear_out.weight)

        self.linear_aux_out = nn.Linear(n_hidden, num_aux_targets)

        #nn.init.xavier_uniform_(self.linear_aux_out.weight)

        

    def forward(self, bert_output, sentence_feature):

        # encodeセルの最初の時系列（Classification）以外は出力しないBertモデル

        # 変数前の*は、変数をアンパックする（引数を増やしてる？）

        #_, bert_output = self.bert(*x_features, output_all_encoded_layers=False)

        bert_output = self.dropout(bert_output)

        

        bert_relu  = F.relu(bert_output)

        

        before_nn = self.before_linear(bert_relu)

        before_relu = F.relu(before_nn)

        before_nn2 = self.before_linear2(before_relu)

        before_relu2 = F.relu(before_nn2)

        

        sentence_feature_nn = self.sentence_feature_linear(sentence_feature)

        sentence_feature_relu = F.relu(sentence_feature_nn)

        

        h_cat = torch.cat((before_relu2, sentence_feature_relu), 1)

        mix_nn = self.mix_linear(h_cat)

        mix_relu = F.relu(mix_nn)

        

        result = self.linear_out(mix_relu)

        aux_result = self.linear_aux_out(mix_relu)

        out = torch.cat([result, aux_result], 1)

        

        return out
import math

# テストデータをバッチ毎に分けるジェネレータ関数

class DynamicBucketIterator(object):

    # テストデータをバッチ毎のデータセットに分割

    def __init__(self, data, label, capacity, pad_token, shuffle, length_quantile, max_batch_size, for_bert):

        self.data = data

        self.label = label

        self.pad_token = pad_token

        self.capacity = capacity

        self.shuffle = shuffle

        self.length_quantile = length_quantile

        self.for_bert = for_bert

        

        # 文章が短い順に文章データのindexをソート

        self.index_sorted = sorted(range(len(self.data)), key=lambda i: len(self.data[i]))

        

        old_separator_index = 0

        self.separator_index_list = [0]

        for i_sample in range(len(self.data)):

            # 文章が短い順に文章データ取得

            sample_index = self.index_sorted[i_sample]

            sample = self.data[sample_index]

            current_batch_size = i_sample - old_separator_index + 1

            if min(len(sample), MAX_LEN) * current_batch_size <= self.capacity and current_batch_size <= max_batch_size:

                pass

            else:

                # バッチの最後の文章データのindexを記録

                old_separator_index = i_sample

                self.separator_index_list.append(i_sample)

                

        # 文章データの最後のindexを記録

        self.separator_index_list.append(len(self.data)) # [0, ..., start_separator_index, end_separator_index, ..., len(data)]

        

        if not self.shuffle:

            # バッチ数取得

            self.bucket_index = range(self.__len__())

        

        self.reset_index()



    def reset_index(self):

        self.i_batch = 0

        

        if self.shuffle:

            self.index_sorted = sorted(np.random.permutation(len(self.data)), key=lambda i: len(self.data[i]))

            self.bucket_index = np.random.permutation(self.__len__())

    

    def __len__(self):

        return len(self.separator_index_list) - 1

    

    # イテレータ関数により呼び出される関数

    def __iter__(self):

        return self

    

    # イテレータにバッチ毎のデータを返す

    def __next__(self):

        # バッチ数が全て呼び出されていたら、初期化して、イテレーション終了

        try:

            i_bucket = self.bucket_index[self.i_batch]

        except IndexError as e:

            self.reset_index()

            raise StopIteration

            

        start_index, end_index = self.separator_index_list[i_bucket : i_bucket + 2]

        

        # データのindexを使用順に保存

        index_batch = self.index_sorted[start_index : end_index]



        raw_batch_data = [self.data[i] for i in index_batch]

        

        batch_label = self.label[index_batch]

        # ???

        math.ceil(1)

        

        # バッチ中で一番長い文章の単語数を取得

        max_len = int(math.ceil(np.quantile([len(x) for x in raw_batch_data], self.length_quantile)))

        max_len = min([max_len, MAX_LEN])

        if max_len == 0:

            max_len = 1

        

        # BERT用にデータを整形して、返す

        if self.for_bert:

            # バッチの文章データの空配列

            segment_id_batch = np.zeros((len(raw_batch_data), max_len))

            padded_batch = []

            input_mask_batch = []

            for sample in raw_batch_data:

                # バッチ内で最長のデータに合わせて、単語が入ってる所に1、入ってない所に0

                input_mask = [1] * len(sample) + [0] * (max_len - len(sample))

                input_mask_batch.append(input_mask[:max_len])



                # データ内をバッチ内で最長のデータに合わせて、0でパディング

                sample = sample + [self.pad_token for _ in range(max_len - len(sample))]

                padded_batch.append(sample[:max_len])



            self.i_batch += 1



            # バッチ毎に、パディングされた文章データ、文章データの空配列、

            # 文章データのmask、文章毎の学習の正解データ、文章データのindexを返す

            return padded_batch, segment_id_batch, input_mask_batch, batch_label, index_batch

        

        else:

            padded_batch = []

            for sample in raw_batch_data:

                sample = sample + [self.pad_token for _ in range(max_len - len(sample))]

                padded_batch.append(sample[:max_len])



            self.i_batch += 1



            return padded_batch, batch_label, index_batch

        

def sigmoid(x):

    return np.where(x<-709.0, 0.0, 1 / (1 + np.exp(-x)))
import time

import torch.optim as optim



test_dict = {}

fold_list = [0]

epochs = 7



start_time = time.time()



# encodeセルの最初の時系列（Classification）以外は出力しないBertモデル

# WORK_DIR上の設定ファイルで事前学習済みのBERTを作成

bert = BertModel.from_pretrained(WORK_DIR).cuda()

#for param in bert.parameters():

    #param.requires_grad=False

    #print(f'bert-{param.requires_grad}')





model = NeuralNet(y_label_size-1, sentence_feature_size)

# 最適化手法のパラメータ設定

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

model.cuda()



# クラス分類が1のデータのロス関数（BCEWithLogitsLoss）への影響度

#pos_weight = (len(y_targetlabel) - y_target.sum(0)) / y_target.sum(0)

#pos_weight[pos_weight == float("Inf")] = 1

#print(pos_weight)

loss_fn=nn.BCEWithLogitsLoss(reduction='mean')#, pos_weight=pos_weight)



#for param in model.parameters():

    #print(param.requires_grad)



highest_accuracy = 0

lowest_loss = len(valid) * 100

#print(lowest_loss)

for i_fold in fold_list:

    fold_start_time = time.time()



    train_loader = DynamicBucketIterator(train, 

                                        train_label,

                                        capacity=MAX_LEN*batch_size, pad_token=0, shuffle=False, length_quantile=1, max_batch_size=2048, for_bert=True)

    

    #print(valid_label)

    valid_loader = DynamicBucketIterator(valid, 

                                         valid_label,

                                         capacity=MAX_LEN*batch_size,

                                         pad_token=0,

                                         shuffle=False,

                                         length_quantile=1,

                                         max_batch_size=2048,

                                         for_bert=True)



    print(torch.cuda.memory_allocated())



    eval_start_time = time.time()

    for epoch in range(epochs):

        batch_i = 0

        train_loss_validation = 0

        model.train()

        for batch in train_loader:

            x_batch = batch[0]

            segment_id_batch = batch[1]

            input_mask_batch = batch[2]

            y_batch = batch[3]

            #print(y_batch)

            y_targets_batch = y_batch[:, :y_label_size]

            sentence_feature = y_batch[:, -sentence_feature_size:]

            #print(y_targets_batch)

            #print(sentence_feature)

            index_batch = batch[4]

            #sample_weight_batch = y_batch[:, len(y_batch[1])-1]

            x_features = [torch.tensor(feature, dtype=torch.long).cuda() for feature in [x_batch, segment_id_batch, input_mask_batch]]



            with torch.no_grad():

                # 変数前の*は、変数をアンパックする（引数を増やしてる？）

                _, bert_output = bert(*x_features, output_all_encoded_layers=False)

            #print(bert_output.grad_fn)

            # 勾配の初期化

            optimizer.zero_grad()

            out = model(bert_output, sentence_feature)

            #print(model.linear_out.weight.grad)

            #print(y_targets_batch)

            test_dict[f'{epoch}-{batch_i}'] = sigmoid(out.detach().cpu().numpy())

            batch_i += 1

            

            # クラス分類が1のデータのロス関数（BCEWithLogitsLoss）への影響度

            #pos_weight = (len(y_targets_batch) - y_targets_batch.sum(0)) / y_targets_batch.sum(0)

            #pos_weight[pos_weight == float("Inf")] = 1

            #print(pos_weight)

            #loss_fn=nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

            

            loss = loss_fn(out, y_targets_batch)

            #print(f'{epoch}-{batch_i}:{loss.item() / len(y_batch)}')

            

            train_loss_validation += loss.item()

            # 勾配の計算

            loss.backward()

            #print(model.linear_out.weight.grad)

            #print(bert_output.grad_fn.grad)

            

            # パラメータの更新

            optimizer.step()

            del x_batch, segment_id_batch, input_mask_batch, y_batch, y_targets_batch, sentence_feature, index_batch, x_features, out, loss, bert_output

            torch.cuda.empty_cache()

        print("train_loss_validation:", train_loss_validation)

        

        valid_pred = np.zeros(len(valid))

        batch_i = 0

        loss_validation = 0

        model.eval()

        for batch in valid_loader:

            x_batch = batch[0]

            segment_id_batch = batch[1]

            input_mask_batch = batch[2]

            y_batch = batch[3]

            #print(y_batch)

            y_targets_batch = y_batch[:, :y_label_size]

            sentence_feature = y_batch[:, -sentence_feature_size:]

            #print(y_targets_batch)

            #print(sentence_feature)

            index_batch = batch[4]

            x_features = [torch.tensor(feature, dtype=torch.long).cuda() for feature in [x_batch, segment_id_batch, input_mask_batch]]



            with torch.no_grad():

                # 変数前の*は、変数をアンパックする（引数を増やしてる？）

                _, bert_output = bert(*x_features, output_all_encoded_layers=False)

            #print(bert_output.grad_fn)

            

            y_pred = model(bert_output, sentence_feature)

            

            #print("y_pred:", y_pred[:, 0])

            #print("y_batch:", y_batch[:, 0])

            # クラス分類が1のデータのロス関数（BCEWithLogitsLoss）への影響度

            #pos_weight = (len(y_targets_batch) - y_targets_batch.sum(0)) / y_targets_batch.sum(0)

            #pos_weight[pos_weight == float("Inf")] = 1

            #print(pos_weight)

            #loss_fn=nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

            

            loss = loss_fn(y_pred, y_targets_batch)

            

            loss_validation += loss.item()

            #print(y_batch)

            #test_dict[f'{epoch}-{batch_i}'] = sigmoid(out.detach().cpu().numpy())

            batch_i += 1

            

            valid_pred[index_batch] = sigmoid(y_pred[:, 0].detach().cpu().numpy())

            

            del x_batch, segment_id_batch, input_mask_batch, y_batch, y_targets_batch, sentence_feature, index_batch, x_features#, loss_fn

            torch.cuda.empty_cache()

        

        valid_pred = (torch.from_numpy(valid_pred) >= 0.5).to(torch.float32).cuda()

        correct = (valid_pred == valid_label[:, 0]).to(torch.float32).sum(0)

        positive_correct = ((valid_pred == valid_label[:, 0]) & (valid_label[:, 0] == 1.0)).to(torch.float32).sum(0)

        negative_correct = correct - positive_correct

        accuracy = correct / len(valid_label)

        positive_sum = valid_label[:, 0].sum(0)

        positive_accuracy = positive_correct / positive_sum

        negative_accuracy = negative_correct / (len(valid_label) - positive_sum)

        if (accuracy >= highest_accuracy) | (lowest_loss >= loss_validation):

            print(epoch)

            print(accuracy)

            print(positive_accuracy)

            print(negative_accuracy)

            highest_accuracy = accuracy

            lowest_loss = loss_validation

            print("lowest_loss:", lowest_loss)

            # 重みやパラメータの値のみ保存（保存されたメモリ番地などは保存しない）

            torch.save(model.state_dict(), '../fine-tuning')

            

        print("loss_validation:", loss_validation)

        print(f'uncased {i_fold} finishd in {time.time() - fold_start_time}', file=open('time.txt', 'a'))



    del train_loader, valid_loader, correct, accuracy, batch_i, positive_correct, negative_correct, positive_sum, positive_accuracy, negative_accuracy

    gc.collect()

    torch.cuda.empty_cache()
del train, valid, valid_label, train_label, optimizer, loss_fn

gc.collect()

torch.cuda.empty_cache()



print(f'uncased finishd in {time.time() - start_time}', file=open('time.txt', 'a'))
#test_dict
test_dict = {}

fold_list = [0]

#min_out = torch.tensor(-709, dtype=torch.float32).cuda()



i_epoch = 2

start_time = time.time()



model.load_state_dict(torch.load('../fine-tuning'))

model.eval()

#for param in model.parameters():

    #print(param.requires_grad)



for i_fold in fold_list:

    fold_start_time = time.time()



    test_loader = DynamicBucketIterator(test, 

                                        test_label,

                                        capacity=MAX_LEN*batch_size, pad_token=0, shuffle=False, length_quantile=1, max_batch_size=2048, for_bert=True)



    print(torch.cuda.memory_allocated())



    test_pred = np.zeros(len(test))

    eval_start_time = time.time()

    for batch in test_loader:

        x_batch = batch[0]

        segment_id_batch = batch[1]

        input_mask_batch = batch[2]

        y_batch = batch[3]

        sentence_feature = y_batch[:, -sentence_feature_size:]

        index_batch = batch[4]

        x_features = [torch.tensor(feature, dtype=torch.long).cuda() for feature in [x_batch, segment_id_batch, input_mask_batch]]

        #                 print('x_features', torch.cuda.memory_allocated())

        

        with torch.no_grad():

            _, bert_output = bert(*x_features, output_all_encoded_layers=False)

        #print(bert_output.grad_fn)

        

        y_pred = model(bert_output, sentence_feature)

        #                 print('after_prediction', torch.cuda.memory_allocated())

        #y_pred = torch.where(y_pred < min_out, min_out, y_pred)

        #print(y_pred)

        test_pred[index_batch] = sigmoid(y_pred[:, 0].detach().cpu().numpy())

        del x_batch, segment_id_batch, input_mask_batch, index_batch, y_batch, sentence_feature, x_features, y_pred, bert_output

        torch.cuda.empty_cache()

    print(f'uncased {i_fold} finishd in {time.time() - fold_start_time}', file=open('time.txt', 'a'))



    test_dict[i_fold] = test_pred

    #print(epoch_test_pred)

    del model, test_loader

    gc.collect()

    torch.cuda.empty_cache()
test_pred = (torch.from_numpy(test_pred) >= 0.5).to(torch.float32).cuda()

correct = (test_pred == test_label[:, 0]).to(torch.float32).sum(0)

accuracy = correct / len(test_label)

print(accuracy)



del accuracy, correct, test_pred

gc.collect()

torch.cuda.empty_cache()

print(f'uncased finishd in {time.time() - start_time}', file=open('time.txt', 'a'))
#min_out = torch.tensor(-700, dtype=torch.float32)

#y_pred = torch.tensor([-709, -710], dtype=torch.float32)

#pred = torch.where(y_pred < min_out, min_out, y_pred)

#sigmoid(pred.detach().cpu().numpy())
#test_dict
#df_submit = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

#df_submit.prediction = test_dict[0]

#df_submit.to_csv('submission.csv', index=False)