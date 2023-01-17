import os

import re

import pandas as pd

import numpy as np

from nltk.corpus import stopwords

import string

from textblob import TextBlob
def load_data(datadir):

    train_df = pd.read_csv(os.path.join(datadir, 'train.csv'))   

    test_df = pd.read_csv(os.path.join(datadir, 'test.csv'))   

    print("Train shape : ", train_df.shape)

    print("Test shape : ", test_df.shape)

    return train_df, test_df
def build_vocab(texts):

    sentences = texts.apply(lambda x: x.split()).values

    vocab = {}

    for sentence in sentences:

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab
def add_features(df):

    print('Adding some features to data')

    

    ## Transform it to string

    df["question_text"] = df["question_text"].apply(lambda x: str(x))

    print('To Sting')

    ## Number of words in the text ##

    df["num_words"] = df["question_text"].apply(lambda x: len(x.split()))

    print('Number of words')

    ## Number of unique words in the text ##

    df["num_unique_words"] = df["question_text"].apply(lambda x: len(set(x.split())))

    print('Number of unique words')

    ## Number of characters in the text ##

    df["num_chars"] = df["question_text"].apply(lambda x: len(x))

    print('Number of chars')

    ## Number of stopwords in the text ##

    df["num_stopwords"] = df["question_text"].apply(lambda x: len([w for w in x.lower().split() if w in set(stopwords.words('english'))]))

    print('Number of Stopwords')

    ## Number of punctuations in the text ##

    df["num_punctuations"] =df['question_text'].apply(lambda x: len([c for c in x if c in string.punctuation]) )

    print('Number of punctuations')

    ## Number of upper case words in the text ##

    df["num_words_upper"] = df["question_text"].apply(lambda x: len([w for w in x.split() if w.isupper()]))

    print('Number of upper words')

    ## Number of title case words in the text ##

    df["num_words_title"] = df["question_text"].apply(lambda x: len([w for w in x.split() if w.istitle()]))

    print('Number of title')

    ## Number of numbers in the text ##

    df["num_num"] = df["question_text"].apply(lambda x: len([n for n in x.split() if n.isnumeric()]))

    print('Number of numbers')

    ## Average length of the words in the text ##

    df["mean_word_len"] = df["question_text"].apply(lambda x: np.mean([len(w) for w in x.split()]))

    print('mean len word')

    return df
def load_embed(file):

    def get_coefs(word,*arr): 

        return word, np.asarray(arr, dtype='float32')



    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))    

    return embeddings_index
def build_vocab(texts):

    sentences = texts.apply(lambda x: x.split()).values

    vocab = set()

    for sentence in sentences:

        for word in sentence:

            vocab.add(word)

    return vocab
def add_lower(embedding, df_column):

    vocab = build_vocab(df_column)

    count = 0

    for word in vocab:

        if word in embedding and word.lower() not in embedding:  

            embedding[word.lower()] = embedding[word]

            count += 1

    print(f"Added {count} words to embedding")
def clean_lower(df):

    df["question_text"] = df["question_text"].apply(lambda x: x.lower())

    return df
abbreviations = {

    "ain't": "is not",

    "aren't": "are not",

    "can't": "cannot",

    "'cause": "because",

    "could've": "could have",

    "couldn't": "could not",

    "didn't": "did not",

    "doesn't": "does not",

    "don't": "do not",

    "hadn't": "had not",

    "hasn't": "has not",

    "haven't": "have not",

    "he'd": "he would",

    "he'll": "he will",

    "he's": "he is",

    "how'd": "how did",

    "how'd'y": "how do you",

    "how'll": "how will",

    "how's": "how is",

    "I'd": "I would",

    "I'd've": "I would have",

    "I'll": "I will",

    "I'll've": "I will have",

    "I'm": "I am",

    "I've": "I have",

    "i'd": "i would",

    "i'd've": "i would have",

    "i'll": "i will",

    "i'll've": "i will have",

    "i'm": "i am",

    "i've": "i have",

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

    "so's": "so as",

    "this's": "this is",

    "that'd": "that would",

    "that'd've": "that would have",

    "that's": "that is",

    "there'd": "there would",

    "there'd've": "there would have",

    "there's": "there is",

    "here's": "here is",

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

    "you'll": "you will",

    "you'll've": "you will have",

    "you're": "you are",

    "you've": "you have",

    "who'd": "who would",

    "who're": "who are",

    "'re": " are",

    "tryin'": "trying",

    "doesn'": "does not",

    'howdo': 'how do',

    'whatare': 'what are',

    'howcan': 'how can',

    'howmuch': 'how much',

    'howmany': 'how many',

    'whydo': 'why do',

    'doI': 'do I',

    'theBest': 'the best',

    'howdoes': 'how does',

}
def clean_abbreviation(df, abbreviations):

    compiled_abbreviation = re.compile('(%s)' % '|'.join(abbreviations.keys()))

    def replace(match):

        return abbreviations[match.group(0)]

    df['question_text'] = df["question_text"].apply(

        lambda x: _clean_abreviation(x, compiled_abbreviation, replace)

    )

    return df

    

def _clean_abreviation(x, compiled_re, replace):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        x = x.replace(s, "'")

    return compiled_re.sub(replace, x)
## I added social media to the list

spells = {

    'colour': 'color',

    'centre': 'center',

    'favourite': 'favorite',

    'travelling': 'traveling',

    'counselling': 'counseling',

    'theatre': 'theater',

    'cancelled': 'canceled',

    'labour': 'labor',

    'organisation': 'organization',

    'wwii': 'world war 2',

    'citicise': 'criticize',

    'youtu.be': 'youtube',

    'youtu ': 'youtube ',

    'qoura': 'quora',

    'sallary': 'salary',

    'Whta': 'what',

    'whta': 'what',

    'narcisist': 'narcissist',

    'mastrubation': 'masturbation',

    'mastrubate': 'masturbate',

    "mastrubating": 'masturbating',

    'pennis': 'penis',

    'Etherium': 'ethereum',

    'etherium': 'ethereum',

    'narcissit': 'narcissist',

    'bigdata': 'big data',

    '2k': '2000',

    '2k10': '2010',

    '2k11': '2011',

    '2k12': '2012',

    '2k13': '2013',

    '2k14': '2014',

    '2k15': '2015',

    '2k16': '2016',

    '2k17': '2017',

    '2k18': '2018',

    'qouta': 'quota',

    'exboyfriend': 'ex boyfriend',

    'exgirlfriend': 'ex girlfriend',

    'airhostess': 'air hostess',

    "whst": 'what',

    'watsapp': 'whatsapp',

    'demonitisation': 'demonetization',

    'demonitization': 'demonetization',

    'demonetisation': 'demonetization',

    'quorans': 'quora user',

    'quoran': 'quora user',

    'pokémon': 'pokemon',

    'instagram': 'social medium',

    'whatsapp': 'social medium',

    'snapchat': 'social medium',

    'pubg': 'video game',

    'dota': 'video game',

    'dota2': 'video game',

    'fortnite': 'video game',

    'league of legends': 'video game'

}
def clean_spells(df, spells):

    compiled_spells = re.compile('(%s)' % '|'.join(spells.keys()))

    def replace(match):

        return spells[match.group(0)]

    df['question_text'] = df["question_text"].apply(

        lambda x: _clean_spells(x, compiled_spells, replace)

    )

    return df



def _clean_spells(x, compiled_re, replace):

    return compiled_re.sub(replace, x)
def clean_numbers(df):  

    df['question_text'] = df['question_text'].apply(lambda x: _clean_numbers(x))

    return df





def _clean_numbers(x):

    if bool(re.search(r'\d', x)):

        x = re.sub('[0-9]{5,}', '#####', x)

        x = re.sub('[0-9]{4}', '####', x)

        x = re.sub('[0-9]{3}', '###', x)

        x = re.sub('[0-9]{2}', '##', x)

    return x
## I removed '#' sign from the list

all_puncts={',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '*', '+', '\\', 

        '•', '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 

        '█', '…', '“', '★', '”', '–', '●', '►', '−', '¢', '¬', '░', '¡', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', 

        '—', '‹', '─', '▒', '：', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', '¯', '♦', '¤', '▲', '¸', '⋅', '‘', '∞', 

        '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '・', '╦', '╣', '╔', '╗', '▬', '❤', '≤', '‡', '√', '◄', '━', 

        '⇒', '▶', '≥', '╝', '♡', '◊', '。', '✈', '≡', '☺', '✔', '↵', '≈', '✓', '♣', '☎', '℃', '◦', '└', '‟', '～', '！', '○', 

        '◆', '№', '♠', '▌', '✿', '▸', '⁄', '□', '❖', '✦', '．', '÷', '｜', '┃', '／', '￥', '╠', '↩', '✭', '▐', '☼', '☻', '┐', 

        '├', '«', '∼', '┌', '℉', '☮', '฿', '≦', '♬', '✧', '〉', '－', '⌂', '✖', '･', '◕', '※', '‖', '◀', '‰', '\x97', '↺', 

        '∆', '┘', '┬', '╬', '،', '⌘', '⊂', '＞', '〈', '⎙', '？', '☠', '⇐', '▫', '∗', '∈', '≠', '♀', '♔', '˚', '℗', '┗', '＊', 

        '┼', '❀', '＆', '∩', '♂', '‿', '∑', '‣', '➜', '┛', '⇓', '☯', '⊖', '☀', '┳', '；', '∇', '⇑', '✰', '◇', '♯', '☞', '´', 

        '↔', '┏', '｡', '◘', '∂', '✌', '♭', '┣', '┴', '┓', '✨', '\xa0', '˜', '❥', '┫', '℠', '✒', '［', '∫', '\x93', '≧', '］', 

        '\x94', '∀', '♛', '\x96', '∨', '◎', '↻', '⇩', '＜', '≫', '✩', '✪', '♕', '؟', '₤', '☛', '╮', '␊', '＋', '┈', '％', 

        '╋', '▽', '⇨', '┻', '⊗', '￡', '।', '▂', '✯', '▇', '＿', '➤', '✞', '＝', '▷', '△', '◙', '▅', '✝', '∧', '␉', '☭', 

        '┊', '╯', '☾', '➔', '∴', '\x92', '▃', '↳', '＾', '׳', '➢', '╭', '➡', '＠', '⊙', '☢', '˝', '∏', '„', '∥', '❝', '☐', 

        '▆', '╱', '⋙', '๏', '☁', '⇔', '▔', '\x91', '➚', '◡', '╰', '\x85', '♢', '˙', '۞', '✘', '✮', '☑', '⋆', 'ⓘ', '❒', 

        '☣', '✉', '⌊', '➠', '∣', '❑', '◢', 'ⓒ', '\x80', '〒', '∕', '▮', '⦿', '✫', '✚', '⋯', '♩', '☂', '❞', '‗', '܂', '☜', 

        '‾', '✜', '╲', '∘', '⟩', '＼', '⟨', '·', '✗', '♚', '∅', 'ⓔ', '◣', '͡', '‛', '❦', '◠', '✄', '❄', '∃', '␣', '≪', '｢', 

        '≅', '◯', '☽', '∎', '｣', '❧', '̅', 'ⓐ', '↘', '⚓', '▣', '˘', '∪', '⇢', '✍', '⊥', '⎯', '↠', '۩', '☰', '◥', 

        '⊆', '✽', '⚡', '↪', '❁', '☹', '◼', '☃', '◤', '❏', 'ⓢ', '⊱', '➝', '̣', '✡', '∠', '｀', '▴', '┤', '∝', '♏', 'ⓐ', 

        '✎', ';', '␤', '＇', '❣', '✂', '✤', 'ⓞ', '☪', '✴', '⌒', '˛', '♒', '＄', '✶', '▻', 'ⓔ', '◌', '◈', '❚', '❂', '￦', 

        '◉', '╜', '̃', '✱', '╖', '❉', 'ⓡ', '↗', 'ⓣ', '♻', '➽', '׀', '✲', '✬', '☉', '▉', '≒', '☥', '⌐', '♨', '✕', 'ⓝ', 

        '⊰', '❘', '＂', '⇧', '̵', '➪', '▁', '▏', '⊃', 'ⓛ', '‚', '♰', '́', '✏', '⏑', '̶', 'ⓢ', '⩾', '￠', '❍', '≃', '⋰', '♋', 

        '､', '̂', '❋', '✳', 'ⓤ', '╤', '▕', '⌣', '✸', '℮', '⁺', '▨', '╨', 'ⓥ', '♈', '❃', '☝', '✻', '⊇', '≻', '♘', '♞', 

        '◂', '✟', '⌠', '✠', '☚', '✥', '❊', 'ⓒ', '⌈', '❅', 'ⓡ', '♧', 'ⓞ', '▭', '❱', 'ⓣ', '∟', '☕', '♺', '∵', '⍝', 'ⓑ', 

        '✵', '✣', '٭', '♆', 'ⓘ', '∶', '⚜', '◞', '்', '✹', '➥', '↕', '̳', '∷', '✋', '➧', '∋', '̿', 'ͧ', '┅', '⥤', '⬆', '⋱', 

        '☄', '↖', '⋮', '۔', '♌', 'ⓛ', '╕', '♓', '❯', '♍', '▋', '✺', '⭐', '✾', '♊', 

        '➣', '▿', 'ⓑ', '♉', '⏠', '◾', '▹', 

        '⩽', '↦', '╥', '⍵', '⌋', '։', '➨', '∮', '⇥', 'ⓗ', 'ⓓ', '⁻', '⎝', '⌥', '⌉', '◔', '◑', '✼', '♎', '♐', '╪', '⊚', 

        '☒', '⇤', 'ⓜ', '⎠', '◐', '⚠', '╞', '◗', '⎕', 'ⓨ', '☟', 'ⓟ', '♟', '❈', '↬', 'ⓓ', '◻', '♮', '❙', '♤', '∉', '؛', 

        '⁂', 'ⓝ', '־', '♑', '╫', '╓', '╳', '⬅', '☔', '☸', '┄', '╧', '׃', '⎢', '❆', '⋄', '⚫', '̏', '☏', '➞', '͂', '␙', 

        'ⓤ', '◟', '̊', '⚐', '✙', '↙', '̾', '℘', '✷', '⍺', '❌', '⊢', '▵', '✅', 'ⓖ', '☨', '▰', '╡', 'ⓜ', '☤', '∽', '╘', 

        '˹', '↨', '♙', '⬇', '♱', '⌡', '⠀', '╛', '❕', '┉', 'ⓟ', '̀', '♖', 'ⓚ', '┆', '⎜', '◜', '⚾', '⤴', '✇', '╟', '⎛', 

        '☩', '➲', '➟', 'ⓥ', 'ⓗ', '⏝', '◃', '╢', '↯', '✆', '˃', '⍴', '❇', '⚽', '╒', '̸', '♜', '☓', '➳', '⇄', '☬', '⚑', 

        '✐', '⌃', '◅', '▢', '❐', '∊', '☈', '॥', '⎮', '▩', 'ு', '⊹', '‵', '␔', '☊', '➸', '̌', '☿', '⇉', '⊳', '╙', 'ⓦ', 

        '⇣', '｛', '̄', '↝', '⎟', '▍', '❗', '״', '΄', '▞', '◁', '⛄', '⇝', '⎪', '♁', '⇠', '☇', '✊', 'ி', '｝', '⭕', '➘', 

        '⁀', '☙', '❛', '❓', '⟲', '⇀', '≲', 'ⓕ', '⎥', '\u06dd', 'ͤ', '₋', '̱', '̎', '♝', '≳', '▙', '➭', '܀', 'ⓖ', '⇛', '▊', 

        '⇗', '̷', '⇱', '℅', 'ⓧ', '⚛', '̐', '̕', '⇌', '␀', '≌', 'ⓦ', '⊤', '̓', '☦', 'ⓕ', '▜', '➙', 'ⓨ', '⌨', '◮', '☷', 

        '◍', 'ⓚ', '≔', '⏩', '⍳', '℞', '┋', '˻', '▚', '≺', 'ْ', '▟', '➻', '̪', '⏪', '̉', '⎞', '┇', '⍟', '⇪', '▎', '⇦', '␝', 

        '⤷', '≖', '⟶', '♗', '̴', '♄', 'ͨ', '̈', '❜', '̡', '▛', '✁', '➩', 'ா', '˂', '↥', '⏎', '⎷', '̲', '➖', '↲', '⩵', '̗', '❢', 

        '≎', '⚔', '⇇', '̑', '⊿', '̖', '☍', '➹', '⥊', '⁁', '✢'}





## checked for GLOVE embedding

glove_puncs = set("/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&')



## I'll fix these puncs using a dict

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi'}

        

## Remove other puncs

remove_punc = all_puncts - glove_puncs



## Remove useless variables

del all_puncts



def clean_and_map_special_chars(df, punct, mapping, removes):

    df["question_text"] = df["question_text"].apply(lambda x: _clean_and_map_special_chars(x, punct, mapping, removes))

    return df



def _clean_and_map_special_chars(text, punct, mapping, removes):

    for r in removes:

        if r in text:

            text = text.replace(r, '')

        

    for p in punct:

        if p in text:

            text = text.replace(p, f' {p} ')

        

    for m in mapping:

        if m in text:

            text = text.replace(m, mapping[m])

     

    specials = {'\u200b': ' ', '…': '', '\ufeff': '', 'करना': '', 'है': '',

               '\x7f':'', '\xa0':'', '\ufeff':'', '\u200e':'', '\u202a':'',

                '\u202c':'', '\u2060':'', '\uf0d8':'', '\ue019':'', '\uf02d':'',

                '\u200f':'', '\u2061':'', '\ue01b':'', '\n':' ', '\t':' ' }  # Other special characters that I have to deal with in last

    

    for s in specials:

        text = text.replace(s, specials[s])

    

    return text
def clean_space(df):

    compiled_re = re.compile(r"\s+")

    df['question_text'] = df["question_text"].apply(lambda x: _clean_space(x, compiled_re))

    return df



def _clean_space(x, compiled_re):

    return compiled_re.sub(" ", x)
def clean(df):

    df = clean_lower(df)

    print('Lower is done')

    df = clean_abbreviation(df, abbreviations)

    print('Abbreviation is done')

    df = clean_spells(df, spells)

    print('Spells is done')

    df = clean_and_map_special_chars(df, glove_puncs, punct_mapping, remove_punc)

    print('Special Chars is done')

    df = clean_numbers(df)

    print('Remove numbers is done')

    df = clean_space(df)

    print('Spaces is done')

    return df
datadir = '../input/quora-insincere-questions-classification'

train_df, test_df = load_data(datadir)
glovedir = os.path.join('../input/quora-insincere-questions-classification', 'embeddings', 'glove.840B.300d', 'glove.840B.300d.txt')

embed_glove = load_embed(glovedir)
add_lower(embed_glove,

          pd.concat((train_df['question_text'], test_df['question_text']),axis=0, ignore_index=True))
# train_df = add_features(train_df)

# test_df = add_features(test_df)
train_df = clean(train_df)

test_df = clean(test_df)
## split to train and val

from sklearn.model_selection import train_test_split



train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2019)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



## Some parameters

embed_size = 300 # how big is each word vector

max_features = 85000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 64 # max number of words in a question



## Fill up the missing values

train_X = train_df["question_text"].fillna("_na_").values

val_X = val_df["question_text"].fillna("_na_").values

test_X = test_df["question_text"].fillna("_na_").values



## Tokenize the sentences

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)

val_X = tokenizer.texts_to_sequences(val_X)

test_X = tokenizer.texts_to_sequences(test_X)



## Pad the sentences 

train_X = pad_sequences(train_X, maxlen=maxlen)

val_X = pad_sequences(val_X, maxlen=maxlen)

test_X = pad_sequences(test_X, maxlen=maxlen)



## Word index

word_index = tokenizer.word_index



## Get the target values

train_y = train_df['target'].values

val_y = val_df['target'].values



def make_embed_matrix(embeddings_index, word_index, max_features):

    

    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size+2))

    

    for word, i in word_index.items():

        if i < max_features:

            embedding_vector = embeddings_index.get(word)

            word_sent = TextBlob(word).sentiment

            ## Extra information we are passing to our embeddings

            extra_embed = [word_sent.polarity,word_sent.subjectivity]

            if embedding_vector is not None:

                embedding_matrix[i] = np.append(embedding_vector,extra_embed)

                

    return embedding_matrix
embedding_matrix = make_embed_matrix(embed_glove, word_index, max_features)

del word_index

del embed_glove

del train_df

del val_df

del test_df

del tokenizer
from keras import backend as K



def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
import tensorflow as tf





inp = tf.keras.layers.Input(shape=(maxlen,))

x = tf.keras.layers.Embedding(max_features, embed_size+2, weights=[embedding_matrix], trainable=False)(inp)

x = tf.keras.layers.SpatialDropout1D(0.3)(x)

x1 = tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(128, return_sequences=True))(x)

x2 = tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(64, return_sequences=True))(x1)

max_pl = tf.keras.layers.GlobalMaxPooling1D()(x1)

avg_pl = tf.keras.layers.GlobalMaxPooling1D()(x2)

x = tf.compat.v1.keras.layers.concatenate([max_pl, avg_pl])

# x = tf.keras.layers.Dense(32, activation="relu")(x)

# x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Dense(1, activation="sigmoid")(x)



model = tf.keras.models.Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer=tf.compat.v1.keras.optimizers.Adam(), metrics=['accuracy', f1])

print(model.summary())



epochs = 7

batch_size = 512



# from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

checkpoints_treated = tf.keras.callbacks.ModelCheckpoint('weights.hdf5', monitor="val_loss", mode="min", verbose=True, save_best_only=True)
history = model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, 

                            validation_data=[val_X, val_y], callbacks=[checkpoints_treated,])
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))

plt.plot(history.history['loss'], label='Train Accuracy')

plt.plot(history.history['val_loss'], label='Test Accuracy')

plt.show()
model.load_weights('weights.hdf5')
pred_val = model.predict([val_X], batch_size=512, verbose=1)
from sklearn.metrics import f1_score

best_score = 0

best_thresh = 0

for thresh in np.arange(0.1, 0.501, 0.0001):

    thresh = np.round(thresh, 4)

    temp_score = f1_score(val_y, (pred_val>thresh).astype(int))

    if temp_score >= best_score:

        best_score = temp_score

        best_thresh = thresh
print( best_score, best_thresh)
pred_test_y = model.predict([test_X], batch_size=512, verbose=0)

pred_test_y = (pred_test_y>best_thresh).astype(int)

out_df = pd.DataFrame({"qid":test_df["qid"].values})

out_df['prediction'] = pred_test_y

out_df.to_csv("submission.csv", index=False)