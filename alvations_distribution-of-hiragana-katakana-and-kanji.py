hiragana = {'え', 'ぁ', 'で', 'め', 'を', 'こ', 'い', '゙', '゜', 'あ', 'ゎ', 'ぺ', 'げ', 'る', 'ず', 'ち', 'ぐ', 'ゖ', 'ゆ', 'ゞ', 'ご', 'ね', 'の', 'ゑ', 'し', 'っ', 'び', 'ぶ', 'へ', 'わ', 'ほ', 'ゅ', 'さ', 'べ', 'ぜ', 'も', 'だ', 'ぱ', 'ぇ', 'ろ', 'ゕ', 'く', 'ま', 'ば', 'ざ', 'は', 'お', 'ぼ', 'ゔ', 'よ', 'ぞ', 'つ', 'が', '\u3097', 'り', 'そ', 'ん', 'ら', 'ゝ', 'れ', 'ぃ', 'ぅ', 'む', 'ぴ', '゛', 'す', 'ぽ', 'み', 'か', 'や', 'ゐ', 'ひ', 'ぷ', 'け', 'せ', '\u3098', 'て', 'ゃ', 'づ', 'ふ', 'ょ', 'じ', '゚', '\u3040', 'ゟ', 'ぉ', 'ぢ', 'な', 'に', 'た', 'ぎ', 'ぬ', 'ど', 'と', 'き', 'う'}

katakana = {'ヅ', 'プ', 'ヂ', 'ァ', 'ヵ', 'ヿ', 'ブ', 'ョ', 'ム', 'ポ', 'エ', 'ノ', 'カ', 'シ', 'ュ', 'モ', 'ナ', 'ト', 'ェ', 'ロ', 'ハ', 'オ', 'ヒ', 'ホ', 'ィ', 'ペ', 'コ', '・', '゠', 'ヶ', 'ク', 'メ', 'ギ', 'ゼ', 'ユ', 'パ', 'ビ', 'ソ', 'ピ', 'ヲ', 'ス', 'ゾ', 'ン', 'ヤ', 'リ', 'ォ', 'ッ', 'ウ', 'ツ', 'ザ', 'グ', 'ベ', 'フ', 'ヘ', 'ニ', 'ジ', 'ゥ', 'キ', 'セ', 'ヱ', 'ャ', 'ヽ', 'ケ', 'ゴ', 'ヾ', 'ラ', 'ヌ', 'タ', 'ヺ', 'サ', 'ヴ', 'ダ', 'レ', 'ヮ', 'テ', 'ヰ', 'ガ', 'ヹ', 'ヷ', 'マ', 'ミ', 'ー', 'ア', 'デ', 'ボ', 'ド', 'チ', 'ゲ', 'イ', 'バ', 'ル', 'ネ', 'ズ', 'ヨ', 'ヸ', 'ワ'}
# E.g. we want to check whether ある is hiragana.

s = 'ある'

all(True for ch in s if ch in hiragana)
s = 'ある'

s[0] in hiragana
len(set(s).intersection(hiragana)) == len(s)
from collections import defaultdict



import pandas as pd



df = pd.read_csv('../input/japanese-lemma-frequency/japanese_lemmas.csv')

df.head()
def is_charset(s, charset):

    return len(set(s).intersection(charset)) == len(s)



# We'll store the charset as the keys

# and list of row index as the values.

charset2idx = defaultdict(list)



for idx, row in df.iterrows():

    lemma = row['lemma']

    if is_charset(lemma, hiragana):

        k = 'hiragana'

    elif is_charset(lemma, katakana):

        k = 'katakana'

    else: # i.e. Kanji.

        k = 'kanji'

    charset2idx[k].append(idx)
num_lemmas = len(df)

print(len(charset2idx['hiragana']), 'out of', num_lemmas, 'are hirgana.')

print(len(charset2idx['katakana']), 'out of', num_lemmas, 'are katakana.')

print(len(charset2idx['kanji']), 'out of', num_lemmas, 'are kanji.')
import matplotlib.pyplot as plt

from matplotlib import rc

 

# Data to plot

labels = 'Hiragana', 'Katakana', 'Kanji', 

sizes = [len(charset2idx['hiragana']),

         len(charset2idx['katakana']),

         len(charset2idx['kanji']), 

        ]

colors = ['lightcoral', 'yellowgreen', 'lightskyblue']

explode = (0.2, 0.1, 0.1)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.rcParams["figure.figsize"] = [16,9]



font = {'size': 22}



rc('font', **font)

plt.show()
from collections import Counter



idx2charset = defaultdict(Counter)



for idx, row in df.iterrows():

    lemma = row['lemma']

    # Iterate through each character in lemma.

    for ch in lemma:

        if ch in hiragana:

            idx2charset[idx]['Hiragana'] += 1

        elif ch in katakana:

            idx2charset[idx]['Katakana'] += 1

        else:

            idx2charset[idx]['Kanji'] += 1
next((idx, idx2charset[idx]) for idx in idx2charset if len(idx2charset[idx]) > 1)
df.iloc[43]
Counter(' + '.join(sorted(charset_in_lemma.keys())) 

        for idx, charset_in_lemma in idx2charset.items())
charset_counter = Counter(' + '.join(sorted(charset_in_lemma.keys())) 

                          for idx, charset_in_lemma in idx2charset.items())



num_lemmas = len(df)

for cs, count in charset_counter.most_common():

    print(count, 'out of', num_lemmas, 'are', cs)
import matplotlib.pyplot as plt

from matplotlib import rc

 

# Close the previous plot

plt.close()



# Data to plot

labels, sizes = zip(*charset_counter.most_common())



# Red = Hiragana

# Green = Katakana

# Blue = Kanji

# Purple = Kanji + Hiragana

# Yellow = Hiragana + Katakana

# Cyan = Kanji + Katakana



colors = ['lightskyblue', 'orchid', 'yellowgreen', 

          'lightcoral',  'cyan', 'yellow']

explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.0 )  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=False, startangle=120)

plt.axis('equal')

plt.rcParams["figure.figsize"] = [14,7]



font = {'size': 12}



rc('font', **font)

plt.show()
charset2idx = defaultdict(list)



for idx, charset_in_lemma in idx2charset.items():

    _charset = ' + '.join(sorted(charset_in_lemma.keys())) 

    charset2idx[_charset].append(idx)
for idx in charset2idx['Kanji + Katakana']:

    print(df.iloc[idx]['lemma'])
for idx in charset2idx['Hiragana + Katakana']:

    print(df.iloc[idx]['lemma'])
for idx in charset2idx['Kanji'][:20]:

    print(df.iloc[idx]['lemma'])
import pickle

with open('../input/charguana/japanese.pkl', 'rb') as fin:

    japanese = pickle.load(fin)



katakana = set(japanese['katakana'])

hiragana = set(japanese['hiragana'])

kanji = set(japanese['kanji'])

romanji = set(japanese['romanji'])
from collections import Counter



idx2charset = defaultdict(Counter)

otherchars = set()

for idx, row in df.iterrows():

    lemma = row['lemma']

    # Iterate through each character in lemma.

    for ch in lemma:

        if ch in hiragana:

            idx2charset[idx]['Hiragana'] += 1

        elif ch in katakana:

            idx2charset[idx]['Katakana'] += 1

        elif ch in romanji:

            idx2charset[idx]['Romanji'] += 1

        elif ch in kanji:

            idx2charset[idx]['Kanji'] += 1

        else:

            otherchars.add(ch)
''.join(sorted(otherchars))
romanji = set(japanese['romanji']).union(otherchars)
idx2charset = defaultdict(Counter)

otherchars = set()

for idx, row in df.iterrows():

    lemma = row['lemma']

    # Iterate through each character in lemma.

    for ch in lemma:

        if ch in hiragana:

            idx2charset[idx]['Hiragana'] += 1

        elif ch in katakana:

            idx2charset[idx]['Katakana'] += 1

        elif ch in romanji:

            idx2charset[idx]['Romanji'] += 1

        elif ch in kanji:

            idx2charset[idx]['Kanji'] += 1

        else: 

            # Now, we should have caught everything so 

            # there shouldn't be anything falling in these gaps. 

            print(ch)
charset_counter = Counter(' + '.join(sorted(charset_in_lemma.keys())) 

                          for idx, charset_in_lemma in idx2charset.items())



num_lemmas = len(df)

for cs, count in charset_counter.most_common():

    print(count, 'out of', num_lemmas, 'are', cs)
import matplotlib.pyplot as plt

from matplotlib import rc

 

# Close the previous plot

plt.close()



# Data to plot

labels, sizes = zip(*charset_counter.most_common()[:5])

_, size_of_others = zip(*charset_counter.most_common()[5:])



# Added the label and counts of 'Others'

labels = ['Others'] + list(labels) 

sizes = [sum(size_of_others)] + list(sizes)



# Blue = Kanji

# Purple = Kanji + Hiragana

# Green = Katakana

# Red = Hiragana

# Grey = Romanji

# Light Grey = Others



colors = ['gainsboro', 'lightskyblue', 'orchid', 'yellowgreen', 

          'lightcoral',  'silver']

explode = (0.0, 0.1, 0.1, 0.1, 0.1, 0.1 )  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=False, startangle=120)

plt.axis('equal')

plt.rcParams["figure.figsize"] = [14,7]



font = {'size': 12}



rc('font', **font)

plt.show()
df.head()
# Initialize the new columns with zeros

df['#Katakana'] = 0

df['#Hiragana'] = 0

df['#Kanji'] = 0

df['#Romanji'] = 0
df.head()
for idx, row in df.iterrows():

    lemma = row['lemma']

    for ch in lemma:

        if ch in hiragana:

            df.iloc[idx, df.columns.get_loc('#Hiragana')] += 1

        elif ch in katakana:

            df.iloc[idx, df.columns.get_loc('#Katakana')] += 1

        elif ch in romanji:

            df.iloc[idx, df.columns.get_loc('#Romanji')] += 1

        elif ch in kanji:

            df.iloc[idx, df.columns.get_loc('#Kanji')] += 1
df.head()