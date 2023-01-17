import os
import json
import string
import numpy as np
import pandas as pd
import plotly.offline as py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go

from pandas.io.json import json_normalize
from plotly import tools
py.init_notebook_mode(connected=True)
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
color = sns.color_palette()
np.random.seed(13)
%matplotlib inline
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ", train_df.shape)
print("Test shape : ", test_df.shape)
train_df.head(8)
sincere_questions = train_df[train_df['target'] == 0]
insincere_questions = train_df[train_df['target'] == 1]
insincere_questions.tail(5)
sincere_questions['length'] = sincere_questions['question_text'].apply(lambda x : len(x))
insincere_questions['length'] = insincere_questions['question_text'].apply(lambda x : len(x))
train_df['length'] = train_df['question_text'].apply(lambda x : len(x))
sincere = go.Box(y=sincere_questions['length'].values, name = 'Sincere Questions', boxmean=True)
insincere = go.Box(y=insincere_questions['length'].values, name = 'Insincere Questions', boxmean=True)
data = [sincere, insincere]
layout = go.Layout(title = "Average Length of Sincere vs. Insincere")
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)
print(train_df.iloc[443216]['question_text'])
train_df[train_df['length'] == train_df['length'].max()]
from tqdm import tqdm # I love this handy tool! 
print(">> Generating Count Based And Demographical Features")
for df in ([train_df]):
    df['length'] = df['question_text'].apply(lambda x : len(x))
    df['capitals'] = df['question_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['length']),axis=1)
    df['num_exclamation_marks'] = df['question_text'].apply(lambda comment: comment.count('!'))
    df['num_question_marks'] = df['question_text'].apply(lambda comment: comment.count('?'))
    df['num_punctuation'] = df['question_text'].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))
    df['num_symbols'] = df['question_text'].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))
    df['num_words'] = df['question_text'].apply(lambda comment: len(comment.split()))
    df['num_unique_words'] = df['question_text'].apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
    df['num_smilies'] = df['question_text'].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
    df['num_sad'] = df['question_text'].apply(lambda comment: sum(comment.count(w) for w in (':-<', ':()', ';-()', ';(')))
train_df[train_df.columns[2:]].head(8)
# List Of Bad Words by Google-Profanity Words 
bad_words = ['cockknocker', 'n1gger', 'ing', 'fukker', 'nympho', 'fcuking', 'gook', 'freex', 'arschloch', 'fistfucked', 'chinc', 'raunch', 'fellatio', 'splooge', 'nutsack', 'lmfao', 'wigger', 'bastard', 'asses', 'fistfuckings', 'blue', 'waffle', 'beeyotch', 'pissin', 'dominatrix', 'fisting', 'vullva', 'paki', 'cyberfucker', 'chuj', 'penuus', 'masturbate', 'b00b*', 'fuks', 'sucked', 'fuckingshitmotherfucker', 'feces', 'panty', 'coital', 'wh00r.', 'whore', 'condom', 'hells', 'foreskin', 'wanker', 'hoer', 'sh1tz', 'shittings', 'wtf', 'recktum', 'dick*', 'pr0n', 'pasty', 'spik', 'phukked', 'assfuck', 'xxx', 'nigger*', 'ugly', 's_h_i_t', 'mamhoon', 'pornos', 'masterbates', 'mothafucks', 'Mother', 'Fukkah', 'chink', 'pussy', 'palace', 'azazel', 'fistfucking', 'ass-fucker', 'shag', 'chincs', 'duche', 'orgies', 'vag1na', 'molest', 'bollock', 'a-hole', 'seduce', 'Cock*', 'dog-fucker', 'shitz', 'Mother', 'Fucker', 'penial', 'biatch', 'junky', 'orifice', '5hit', 'kunilingus', 'cuntbag', 'hump', 'butt', 'fuck', 'titwank', 'schaffer', 'cracker', 'f.u.c.k', 'breasts', 'd1ld0', 'polac', 'boobs', 'ritard', 'fuckup', 'rape', 'hard', 'on', 'skanks', 'coksucka', 'cl1t', 'herpy', 's.o.b.', 'Motha', 'Fucker', 'penus', 'Fukker', 'p.u.s.s.y.', 'faggitt', 'b!tch', 'doosh', 'titty', 'pr1k', 'r-tard', 'gigolo', 'perse', 'lezzies', 'bollock*', 'pedophiliac', 'Ass', 'Monkey', 'mothafucker', 'amcik', 'b*tch', 'beaner', 'masterbat*', 'fucka', 'phuk', 'menses', 'pedophile', 'climax', 'cocksucking', 'fingerfucked', 'asswhole', 'basterdz', 'cahone', 'ahole', 'dickflipper', 'diligaf', 'Lesbian', 'sperm', 'pisser', 'dykes', 'Skanky', 'puuker', 'gtfo', 'orgasim', 'd0ng', 'testicle*', 'pen1s', 'piss-off', '@$$', 'fuck', 'trophy', 'arse*', 'fag', 'organ', 'potty', 'queerz', 'fannybandit', 'muthafuckaz', 'booger', 'pussypounder', 'titt', 'fuckoff', 'bootee', 'schlong', 'spunk', 'rumprammer', 'weed', 'bi7ch', 'pusse', 'blow', 'job', 'kusi*', 'assbanged', 'dumbass', 'kunts', 'chraa', 'cock', 'sucker', 'l3i+ch', 'cabron', 'arrse', 'cnut', 'how', 'to', 'murdep', 'fcuk', 'phuked', 'gang-bang', 'kuksuger', 'mothafuckers', 'ghey', 'clit', 'licker', 'feg', 'ma5terbate', 'd0uche', 'pcp', 'ejaculate', 'nigur', 'clits', 'd0uch3', 'b00bs', 'fucked', 'assbang', 'mutha', 'goddamned', 'cazzo', 'lmao', 'godamn', 'kill', 'coon', 'penis-breath', 'kyke', 'heshe', 'homo', 'tawdry', 'pissing', 'cumshot', 'motherfucker', 'menstruation', 'n1gr', 'rectus', 'oral', 'twats', 'scrot', 'God', 'damn', 'jerk', 'nigga', 'motherfuckin', 'kawk', 'homey', 'hooters', 'rump', 'dickheads', 'scrud', 'fist', 'fuck', 'carpet', 'muncher', 'cipa', 'cocaine', 'fanyy', 'frigga', 'massa', '5h1t', 'brassiere', 'inbred', 'spooge', 'shitface', 'tush', 'Fuken', 'boiolas', 'fuckass', 'wop*', 'cuntlick', 'fucker', 'bodily', 'bullshits', 'hom0', 'sumofabiatch', 'jackass', 'dilld0', 'puuke', 'cums', 'pakie', 'cock-sucker', 'pubic', 'pron', 'puta', 'penas', 'weiner', 'vaj1na', 'mthrfucker', 'souse', 'loin', 'clitoris', 'f.ck', 'dickface', 'rectal', 'whored', 'bookie', 'chota', 'bags', 'sh!t', 'pornography', 'spick', 'seamen', 'Phukker', 'beef', 'curtain', 'eat', 'hair', 'pie', 'mother', 'fucker', 'faigt', 'yeasty', 'Clit', 'kraut', 'CockSucker', 'Ekrem*', 'screwing', 'scrote', 'fubar', 'knob', 'end', 'sleazy', 'dickwhipper', 'ass', 'fuck', 'fellate', 'lesbos', 'nobjokey', 'dogging', 'fuck', 'hole', 'hymen', 'damn', 'dego', 'sphencter', 'queef*', 'gaylord', 'va1jina', 'a55', 'fuck', 'douchebag', 'blowjob', 'mibun', 'fucking', 'dago', 'heroin', 'tw4t', 'raper', 'muff', 'fitt*', 'wetback*', 'mo-fo', 'fuk*', 'klootzak', 'sux', 'damnit', 'pimmel', 'assh0lez', 'cntz', 'fux', 'gonads', 'bullshit', 'nigg3r', 'fack', 'weewee', 'shi+', 'shithead', 'pecker', 'Shytty', 'wh0re', 'a2m', 'kkk', 'penetration', 'kike', 'naked', 'kooch', 'ejaculation', 'bang', 'hoare', 'jap', 'foad', 'queef', 'buttwipe', 'Shity', 'dildo', 'dickripper', 'crackwhore', 'beaver', 'kum', 'sh!+', 'qweers', 'cocksuka', 'sexy', 'masterbating', 'peeenus', 'gays', 'cocksucks', 'b17ch', 'nad', 'j3rk0ff', 'fannyflaps', 'God-damned', 'masterbate', 'erotic', 'sadism', 'turd', 'flipping', 'the', 'bird', 'schizo', 'whiz', 'fagg1t', 'cop', 'some', 'wood', 'banger', 'Shyty', 'f', 'you', 'scag', 'soused', 'scank', 'clitorus', 'kumming', 'quim', 'penis', 'bestial', 'bimbo', 'gfy', 'spiks', 'shitings', 'phuking', 'paddy', 'mulkku', 'anal', 'leakage', 'bestiality', 'smegma', 'bull', 'shit', 'pillu*', 'schmuck', 'cuntsicle', 'fistfucker', 'shitdick', 'dirsa', 'm0f0']
print(">> Words in bad_word list:", len(bad_words))
print(">> Generating Features on Bad Words")
for df in ([train_df]):
    df["badwordcount"] = df['question_text'].apply(lambda comment: sum(comment.count(w) for w in bad_words))
    df['num_chars'] =    df['question_text'].apply(len)
    df["normchar_badwords"] = df["badwordcount"]/df['num_chars']
    df["normword_badwords"] = df["badwordcount"]/df['num_words']
train_df[['badwordcount','num_chars','normchar_badwords','normword_badwords']].head(8)
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

def tag_part_of_speech(text):
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    pos_list = pos_tag(text_splited)
    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])
    adjective_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])
    verb_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])
    return[noun_count, adjective_count, verb_count]
print(">> Generating POS Features")
for df in ([train_df]):
    df['nouns'], df['adjectives'], df['verbs'] = zip(*df['question_text'].apply(
        lambda comment: tag_part_of_speech(comment)))
    df['nouns_vs_length'] = df['nouns'] / df['length']
    df['adjectives_vs_length'] = df['adjectives'] / df['length']
    df['verbs_vs_length'] = df['verbs'] /df['length']
    df['nouns_vs_words'] = df['nouns'] / df['num_words']
    df['adjectives_vs_words'] = df['adjectives'] / df['num_words']
    df['verbs_vs_words'] = df['verbs'] / df['num_words']
    # More Handy Features
    df["count_words_title"] = df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    df["mean_word_len"] = df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    df['punct_percent']= df['num_punctuation']*100/df['num_words']
train_df[['nouns','nouns_vs_length','adjectives_vs_length','verbs_vs_length','nouns_vs_words','adjectives_vs_words','verbs_vs_words']].head(8)
f, ax = plt.subplots(figsize= [20,15])
sns.heatmap(train_df.drop(['qid','question_text'], axis=1).corr(), annot=True, fmt=".2f", ax=ax, 
            cbar_kws={'label': 'Correlation Coefficient'}, cmap='viridis')
ax.set_title("Correlation Matrix for Insincerity and New Features", fontsize=18)
plt.show()