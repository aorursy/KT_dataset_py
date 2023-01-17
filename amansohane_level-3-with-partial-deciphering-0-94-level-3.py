# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

import collections, itertools
import sklearn.feature_extraction.text as sktext
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
train_data.head()
test_data = pd.read_csv("../input/test.csv")
test_data.head()
from sklearn.datasets import fetch_20newsgroups
plaintext_data = fetch_20newsgroups(subset='all', download_if_missing=True)
category_names = plaintext_data.target_names
plain_counts = pd.Series(collections.Counter(itertools.chain.from_iterable(plaintext_data.data))) \
    .rename("counts").to_frame() \
    .sort_values("counts", ascending = False)
plain_counts = 1000000 * plain_counts / plain_counts.sum()
plain_counts = plain_counts.reset_index().rename(columns = {"index":"plain_char"})

diff_counts = []
for i in range(1,5):
    counts = pd.Series(
        collections.Counter(itertools.chain.from_iterable(train_data.query("difficulty == @i")["ciphertext"].values)) + \
        collections.Counter(itertools.chain.from_iterable(test_data.query("difficulty == @i")["ciphertext"].values))
        ).rename("counts").to_frame() \
        .sort_values("counts", ascending = False)
    counts = 1000000 * counts / counts.sum()
    counts = counts.reset_index().rename(columns = {"index":"diff_{}".format(i)})
    diff_counts.append(counts)

pd.concat([plain_counts] + diff_counts, axis = 1).head(20)
##diff2 -> plain, subs
key = [['8', ' '],['$', 'e'],['{', 't'],['V', 'a'],['e', 'o'],['h', 'i'],['\x10', 'n'],['}', 's'],['*', 'r'],['7', 'h'],['?', 'l'],['z', '\n'],['H', 'd'],['f', 'c'],['j', 'u'],['4', 'm'],['\x1a', '.'],['x', '-'],['.', 'p'],['k', 'g'],['v', 'y'],['d', 'f'],['^', 'w'],['\x18', 'b'],['b', '>'],['[', ','],['\x03', 'v'],['6', 'A'],['A', 'I'],['m', 'k'],['B', "'"],['N', ':'],['E', 'S'],['S', '1'],['\x06', 'T'],['(', 'X'],['c', '0'],['l', 'M'],['9', 'C'],['%', '*'],['\x02', ')'],['\x08', '('],['O', '2'],['a', '='],['\x0c', 'N'],['&', 'R'],['@', 'P'],['|', '3'],['\x7f', 'D'],[',', 'O'],['i', '@'],['L', 'E'],['M', 'L'],['=', '"'],['C', '9'],['X', '\t'],['\x1b', '5'],['1', 'F'],['n', 'H'],['Q', 'B'],['3', '4'],['0', '_'],['2', 'x'],['s', 'W'],['<', '6'],[')', 'G'],['_', 'j'],['G', 'U'],['u', '8'],['\x19', '?'],['-', '?'],['o', 'z'],['F', '/'],[';', '|'],['\t', 'J'],['~', 'K'],['W', '!'],['!', 'V'],["'", '<'],[' ', 'Y'],['\n', '+'],['#', 'q'],['I', '$'],[':', '#'],[']', 'Q'],['/', '^'],['g', '#'],['\x1e', '%'],['p', ']'],['5', ']'],['\\', '['],['`', 'Z'],['t', '&'],['y', '&'],['R', 'Z'],['P', '}'],['r', '{'],['"', '\r'],['T', 'u'],['Z', '\x02']]
decrypt_map_2 = {i:j for i,j in key}
diff_3_data = train_data.query("difficulty == 3").copy()
diff_3_data["trans"] = diff_3_data["ciphertext"].apply(
    lambda x:''.join([decrypt_map_2.get(k,'?') for k in x])
)

##top starting letters 
diff_3_data.trans.str[:5].value_counts().head()
collections.Counter([i[:5] for i in plaintext_data.data]).most_common(5)
diff_3_data.loc[diff_3_data["trans"].apply(lambda x:x[:5] == 'FrMmZ')].head(3)
diff_3_data.query("Id == 'ID_fb163c212'").trans.iloc[0]
target_11_data = [i[:300] for i,j in zip(plaintext_data.data, plaintext_data.target) if j == 11] 
[i for i in target_11_data if i.find('Russell') > 0 and i.find("whatever") > 0]
pd.options.display.max_columns = 300

pd.options.display.max_rows = 300

plain_1, cipher_1 = [
    '''From: trussell@cwis.unomaha.edu (Tim Russell)\nSubject: Re: Once tapped, your code is no good any more.\nOrganization: University of Nebraska at Omaha\nDistribution: na\nLines: 18\n\ngeoff@ficus.cs.ucla.edu (Geoffrey Kuenning) writes:\n\n>It always amazes me how quick people are to blame whatever\n>administr''',
    '''FrMmZ tr8ssellkWw#s.znWm}h\t.e2H (T#m R]ssell)\n L?bjeut/ Re? Onue tUppeE, @fzr &&?e zs n@ ugo} $n\n m?re.# OrzUnM>?tkWnd $nzversMt8 gf Nebr?s\n\x02 :t Om9h\x02g DMstr8b@tkWnd n\t# B#nesu 10k f 0eMizk?#g-s.\ns.Mkl\x02.eU] (GeoM?re\n K>ennznz) wrWtesZk & #2t }lw\t0s \x02m?8es me how q{8uo peMple \x02re t] bl\x02me wh$tever& #'''
    
]

plain_2, cipher_2 = [
    '''From: hollasch@kpc.com (Steve Hollasch)\nSubject: Re: Raytracing Colours?\nSummary: Illumination Equations\nOrganization: Kubota Pacific Computer, Inc.\nLines: 44\n\nasecchia@cs.uct.ac.za (Adrian Secchia) writes:\n| When an incident ray (I) strikes an object at point P ...  The reflected\n| ray (R) and the ''',
    '''FrMmZ h]ll\x02sWhi{p&.\n#m (Eteve {fll'sch)i ZHbje-t? ReZ RU\ntr$okn& C&l#?rs?f ?]mmarfI Sll>mzn\x02tW@n ?q@Ity&ns> Or{\x02n#8at]{nZ K>bMt\x02 PUu80k& C&mpMter, dnu.& 1inesa 44k f :sekuhW\x02yos.z&t.}i.y\x02 (AUrHan LeWuh\n9) wrMtes?f | chen ?n kn&>?ent r9\n (\x02) strHkes ?n {bjeut :t pMMnt P ...  The rezleute?& | rIu (R) '''
]

plain_3, cipher_3 = [
    '''From: snichols@adobe.com (Sherri Nichols)\nSubject: Re: Braves Pitching UpdateDIR\nOrganization: Adobe Systems Incorporated\nLines: 13\n\nIn article <1993Apr15.010745.1@acad.drake.edu> sbp002@acad.drake.edu writes:\n>or second starter.  It seems to me that when quality pitchers take the\n>mound, the other ''',
    '''FrMmZ snWuh{lsiIa&be.iMm ($herr# NHchkls)k $>bjektZ Re? ?r$ves P#tghyno $p2\x02teD\x02Rc Org?n8W:tz@n? AE{be ?=stems 2n\n#rpMr\x02teUc L]nesI !3{ # Sn UrtMWle <ZdB3AprL5.o\x02#?45.ZzaH$E.\x02r9?e.e1Hz sbp>i90\tk\x02U.:ra8e.e?H wr\ntes?f k@r seWfnL st?rter.  ?t seems t# me th9t when qH?lHty p]tWhers t\x02oe the# cm]Hn?, the'''
]

plain_4, cipher_4 = [
    '''From: art@cs.UAlberta.CA (Art Mulder)\nSubject: comp.windows.x: Getting more performance out of X.  FAQ\nSummary: This posting contains a list of suggestions about what you can do to get the best performance out of X on your workstation -- without buying more hardware.\nKeywords: FAQ speed X\nNntp-Posti''',
    '''FrMmZ UrtkWs./AlbertI.CA (Art \nfl?er)# $@bjeMt: H{mp.wMn\x02Mws.xZ GettWnz m{re per0urm?n\ne #?t @0 X.  FAQc S{mm$r\n/ Thzs p@stWnz WfntIyns } lust Mi s@?gest]{ns \x02b??t wh\x02t #@? oIn a& t# {et the best perifrm'nce k?t @y X Mn \n]Hr w{rMstIty&n \no wMthf]t buf8nz m?re h9rEwUre.f Ke@wfrLs' FAQ spee' X> NntpoP'''   
]

plain_5, cipher_5 = [
    '''From: joslin@pogo.isp.pitt.edu (David Joslin)\nSubject: Apology to Jim Meritt (Was: Silence is concurance)\nDistribution: usa\nOrganization: Intelligent Systems Program\nLines: 39\n\nm23364@mwunix.mitre.org (James Meritt) writes:\n>}So stop dodging the question.  What is hypocritical about my\n>}criticizing''',
    '''FrMmZ j]slMnypf]W.>sp.putt.e2H (D?vHd Jksl8n)f !?bjeut? Ap@l{u= tW J>m Herztt (??su S]lenWe Ms -Mnu8r\x02nWe)& DkstrybftuMnZ @s'\n OrW$nM#9tM]nZ :ntell#]ent 1]stems Pr#{r\x02m& 1inesa 3}f { ma3364kmw8nMx.m8tre.fr] (J?mes cerutt) wrztesZ& i}L{ st@p \x02MEyMn- the q@estkWn.  kh\tt zs h\npfMrit]W\x02l :bMHt m#f z2gry'''
]
print(pd.DataFrame({'plain':list(plain_1)[:len(cipher_1)], 'cipher':list(cipher_1)}).T)
plain, cipher = [], []

pairs = [(plain_1, cipher_1), (plain_2, cipher_2), (plain_3, cipher_3), (plain_4, cipher_4), (plain_5, cipher_5)]
for p_temp, c_temp in pairs:
    i1,i2 = 0,0
    while 1:
        p,c = p_temp[i1], c_temp[i2]
        plain.append(p)
        cipher.append(c)
        if p == '\n':
            i2+=1
        i1 += 1 
        i2 += 1
        if i2 == 300:
            break

pd.DataFrame({'plain':list(plain), 'cipher':list(cipher)}).T

possible_maps = collections.defaultdict(list)
for i,j in zip(plain, cipher):
    possible_maps[i].append(j)

sure_map = {}
unsure_map = {}
for i,j in possible_maps.items():
    if (len(set(j)) == 1) and len(j) !=1:
        sure_map[i] = j[0]
    else:
        unsure_map[i] = set(j)
print(len(sure_map), len(unsure_map))
d = train_data.query("difficulty == 3").copy()
d["trans"] = d["ciphertext"].apply(
    lambda x:''.join([decrypt_map_2.get(k,'?') for k in x])
)
d["trans"] = d["trans"].apply(
    lambda x:''.join([k if k in sure_map else '?' for k in x])
)
X_train = [''.join([k if k in sure_map else '?' for k in i]) for i in plaintext_data.data]
y_train = plaintext_data.target

X_test = d["trans"].values
y_test = d["target"].values

clf = Pipeline([
    ('vectorizer', sktext.CountVectorizer(lowercase=True, ngram_range = (1,2))),
    ('tfidf', sktext.TfidfTransformer()),
    ('clf', KNeighborsClassifier(n_neighbors = 1))
])

clf.fit(X_train, y_train)
print(classification_report(y_train, clf.predict(X_train)))
print(classification_report(y_test, clf.predict(X_test)))