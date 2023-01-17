# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def read_input():

    return pd.read_csv('/kaggle/input/22000-scotch-whisky-reviews/scotch_review.csv', index_col=0) # Head column is the index columns.

original = read_input()

original.info()
# Why the price column's dtype is not number ?

try:

    original.price.astype(float)

except Exception as e:

    print(e)
# Is there any illregular price?

original.loc[~original.price.str.match('^[\d]+[\.]*[\d]*$')]
# Name column has many informations and some duplicates.

original.name.value_counts()[:20]
# Are all descriptions unique?

original.description.value_counts()[:5]
# What is the review point distribution like?

original['review.point'].hist(bins=25)
# Is there any duplicate?

original[original.duplicated(keep=False)]
# What values can currency take ?

original.currency.value_counts()
def cleanse(df):

    def cleanse_str(sr):

        result = sr.str.normalize('NFKC')

        result = result.str.replace("’", "'")

        return result.str.lower()

    result = df.drop(columns='currency') # We don't need it.

    result = result.drop_duplicates() # We don't need duplicated rows.

    result.name = cleanse_str(result.name)

    result.name = result.name.str.replace('^\s+', '') # Remove the first sapce 

    result.name = result.name.str.replace('^the ', '') # Remove the first 'The' 

    result.price = result.price.str.replace('\$15,000 or \$|/[a-z]*|\,', '').astype(float).astype(np.uint16) # price as int

    result.rename(columns={'review.point':'review_point'}, inplace=True) # Dot separator is hard to use.

    result.review_point = result.review_point.astype(np.uint8) # Make it smaller.

    result.description = cleanse_str(result.description)

    result.description = result.description.str.replace('\r\n', '') # Remove new line chars.

    result['name_pruned'] = df.name # Copy for prune

    return result

# cleansed = cleanse(original)

# cleansed.info()
def extract_alcohol(df):

    # Be careful, 'Kilchoman 100% Islay 3rd Edition, 50%' has two percentages and the first one is not telling about alcohol.

    result = df.name.str.extract('((?<!\d)\d{2}|(?<!\d)\d{2}\.[\d]+)%')[0].astype(float)

    df['name_pruned'] = df.name.str.replace('((?<!\d)\d{2}|(?<!\d)\d{2}\.[\d]+)%', '')

    result.loc[1648] = 57.1 # This row's name contains a word 'ABV' between numeric and '%' which I don't make sence. So I'll do this by hard cording.

    return result

# cleansed.loc[extract_alcohol(cleansed).isnull(), 'name'].values # Debug code.

# extract_alcohol(cleansed).describe()
def extract_age(df):

    result = df.name.str.extract('(\d+) [yY]ear')[0].astype(float)

    df.name_pruned = df.name_pruned.str.replace('(\d+) [yY]ear( [oO]ld)?', '')

    return result

# cleansed.loc[extract_age(cleansed).isnull(), 'name'].values[800:900] # Debug code.

# extract_age(cleansed).describe()
def extract_birth(df):

    result = df.name.str.extract('((?<!#)20[01]\d|(?<!#)19[\d]{2})')

    df.name_pruned = df.name_pruned.str.replace('((?<!#)20[01]\d|(?<!#)19[\d]{2})', '')

    return result

# extract_birth(cleansed)[0].astype(float).describe()
def extract_distilled_at(df):

    result = df.name.str.extract('[Dd]istilled at (.+?)[,\);]')

    df.name_pruned = df.name_pruned.str.replace('\(?[Dd]istilled at (.+?)[,\);]', '')

    return result

# extract_distilled_at(cleansed)[0].value_counts().index.values
# I extracted brands by a huristic way with a text editor, because I couldn't find any algorithm to do that. 

brands = ["a.d.rattray","aberfeldy","aberlour","abhainn dearg","adelphi","ailsa bay","alexander murray & co.", "anchor bay", "ancnoc",

          "annasach","antiquary","ardbeg",

          "ardmore","arran","auchentoshan","auchroisk","auld reekie","aultmore","balblair","ballantine's","balvenie","ben nevis","benriach",

          "benrinnes", "benromach","berry brothers & rudd","big peat","black bottle","black bull","black grouse","blackadder","bladnoch",

          "blair athol","blue hanger","borders","bowmore","brora","bruichladdich","buchanan's","bunnahabhain","cadenhead's","caledonian",

          "cambus","caol ila","cardhu","carlyle","carn mor","cask & thistle","chapter 7","chieftain's","clan denny","clansman",

          "classic cask","clynelish","collectivum xxviii","compass box","connoisseurs choice","convalmore","cooper's choice","copper dog",

          "coronation","cragganmore","craigellachie","creative whisky co.","cuatro series","cutty sark","cù bòcan","d&m",

          "dailuaine","dalmore","dalwhinnie","darkness!","deanston","deerstalker","deveron","dewar's",

          "distillery select","double barrel","douglas","dun bheagan","duncan taylor","duncansby head","eades","edradour","epicurean", "exclusive",

          "famous grouse","famous jubilee","fat trout","feathery","fettercairn","five distinguished and rare","girvan","glen deveron",

          "glen elgin","glen garioch","glen grant","glen moray","glen ord","glen scotia","glen spey","glen turner",

          "glenburgie","glencadam","glendronach","glenfarclas","glenfiddich","glengarioch","glenglassaugh","glengoyne","glenkeir treasures",

          "glenkinchie","glenlivet","glenmorangie","glenrothes","glenturret","glenugie","glenury royal","golden age","gordon & macphail",

          "grand macnish","grangestone","grant's","haig club","half century blend","hankey bannister heritage blend","hart brothers",

          "hazelburn","hepburn's choice","high commissioner","highland journey","highland park", "highland queen","house of hazelwood","hunter laing","inchgower",

          "inchmoan","inchmurrin","islay mist","isle of jura","isle of skye","j&b","j.mossman","james brookes","jamie stewart","jura","lombard","jewels of scotland",

          "john barr","john mcdougall's","john walker","johnnie walker","kilchoman","kilkerran","king's crest","kininvie","kirkland signature","knockando",

          "label 5","lady of the glen","ladyburn","lagavulin","langside distillers","laphroaig","last drop","ledaig","linkwood","littlemill",

          "loch lomond","lonach","longmorn","longrow","lord elcho","lost distiller","macallan","macduff","mackillop's choice","mackinlay's",

          "macnamara","macphail's collection","macqueen's","maltman","mannochmore","master of malt","mcdougall's selection","mcgibbons provenance","miltonduff",

          "monarch of the glen","montgomerie's","moon harbour pier","mortlach","murray mcdavid","naked grouse","noss head","oban","octomore islay barley",

          "old malt cask","old masters freemason whisky","old particular","old pulteney","pearls of scotland","peerless","pentland skerries",

          "pittyvaich","poit dhubh blended malt","port askaig","port charlotte","port dundas","port ellen","provenance","pure scot",

          "raasay while we wait","ragnvald","rare cask reserves blended reserve","robert burns","rock oyster","ron burgundy","rosebank",

          "royal brackla","royal lochnagar","royal mile whiskies","royal salute","scallywag","scapa","scotch malt whisky society","scott's selection",

          "shackleton","sheep dip","shieldaig","sia","signatory","sigurd","single cask nation","single malts of scotland","singleton","sir edward's",

          "smokehead","smoking ember","sovereign","spey royal choice","speyburn","speyside","springbank","storm","strathclyde","strathisla",

          "strathmill","stronachie","syndicate","talisker","tamdhu","te bheag blended whisky","teaninich","that boutique-y","thorfinn",

          "timorous beastie","tobermory","tomatin","tomintoul","tormore","trader joe's","treacle chest","tullibardine","tweeddale","usquaebach",

          "wemyss","whisky exchange","whisky galore","wild scotsman","william grant's","wolfburn"

         ]
def extract_brand(df):

    df.name_pruned = df.name_pruned.str.replace(',', ' ')

    df.name_pruned = df.name_pruned.str.replace('\s{2,}', ' ')

    df.name_pruned = df.name_pruned.str.replace('\. ', '.')

    df.name_pruned = df.name_pruned.str.replace('berry bros\.', 'berry brothers ')

    df.name_pruned = df.name_pruned.str.replace("berry's|berrys'", 'berry brothers & rudd')

    df.name_pruned = df.name_pruned.str.replace("black bowmore", 'bowmore black')

    df.name_pruned = df.name_pruned.str.replace("cadenhead", "cadenhead's")

    df.name_pruned = df.name_pruned.str.replace("gold bowmore", 'bowmore gold')

    df.name_pruned = df.name_pruned.str.replace("macdougall's", "mcdougall's")

    df.name_pruned = df.name_pruned.str.replace("scott|scott selection", "scott's selection")

    df.name_pruned = df.name_pruned.str.replace("traditional ben nevis", 'ben nevis traditional')

    df.name_pruned = df.name_pruned.str.replace("white bowmore", 'bowmore white') 

    df.name_pruned = df.name_pruned.str.replace("^“double malt”", 'eades “double malt”')

    df.name_pruned = df.name_pruned.str.replace("william grant", "william grant's")



    brand_sr = pd.Series(np.repeat('', len(df)), index=df.index)

    for brand in brands:

        mask = df.name_pruned.str.startswith(brand)

        brand_sr[mask] = brand

        df.loc[mask, 'name_pruned'] = df.loc[mask, 'name_pruned'].str.replace(brand, '')

    for brand in ("elements of islay", "chivas", "john walker & sons"):

        mask = df.name_pruned.str.contains(brand)

        brand_sr[mask] = brand

        df.loc[mask, 'name_pruned'] = df.loc[mask, 'name_pruned'].str.replace(brand, '')

    return brand_sr

# extract_brand(cleansed)
def split_name_df():

    df = read_input()

    cleansed = cleanse(df)

    cleansed['alc'] = extract_alcohol(cleansed)

    cleansed['age'] = extract_age(cleansed)

    cleansed['birth'] = extract_birth(cleansed)

    cleansed['distilled_at'] = extract_distilled_at(cleansed)

    cleansed.sort_values('name_pruned', inplace=True)

    cleansed['brand'] = extract_brand(cleansed)

    cleansed.name_pruned = cleansed.name_pruned.str.replace('\s{2,}', ' ')

    cleansed.name_pruned = cleansed.name_pruned.str.replace('^\s+|$\s+', '')

    cleansed.to_csv('/kaggle/working/work.csv', encoding='utf-8 sig')

    return cleansed

# cleansed = split_name_df()
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet



tag_dict = {"J": wordnet.ADJ,

            "N": wordnet.NOUN,

            "V": wordnet.VERB,

            "R": wordnet.ADV}



def get_wordnet_pos(word):

    """Map POS tag to first character lemmatize() accepts"""

    tag = nltk.pos_tag([word])[0][1][0].upper()

    return tag_dict.get(tag, wordnet.NOUN)



def bow(description):

    stop_words = set(stopwords.words('english'))

    wordnet_lemmatizer = WordNetLemmatizer()

    description = description.str.replace("[\,\.!\(\):“”;?%#$€£0-9]", '')

    description = description.str.replace("[\/—\-'‘]| & ", ' ')

    def preproc(sentences):

        result = word_tokenize(sentences)

        result = filter(lambda word: word not in stop_words and len(word) > 2, result)

        result = ' '.join((wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in result))

        return result

    

    count_vectorizer = CountVectorizer()

    count_matrix = count_vectorizer.fit_transform([preproc(a_desc) for a_desc in description.values])

    return pd.DataFrame(data=count_matrix.toarray(), columns=count_vectorizer.get_feature_names(), index=description.index)



def sqeeze_bow_matrix(bow_matrix):

    freq = bow_matrix.sum(axis=0)

    lower_freq_bnd = np.quantile(freq, .75)

    result = bow_matrix.loc[:, (freq > lower_freq_bnd)]

    words = result.columns

    drop_words = ['year', 'old', 'bottle', 'whisky', 'ago', 'along', 'alongside', 'already', 'also', 'although', 'among', 'bros', 'brother', 

                  'chivas', 'could', 'date', 'day', 'do', 'dose', 'get', 'give', 'go', 'however', 'john', 'johnnie', 'make', 'may', 'maybe',

                  'might', 'must', 'na', 'onto', 'otherwise', 'please', 'take', 'though', 'whether', 'whose', 'within', 'would', 'name', 

                  'category', 'price', 'age', 'brand'] + brands

    result.drop(columns = words[words.isin(drop_words)], inplace=True)

    # lemmatize again

    result['allow'] += result['allows']

    result['become'] += result['becomes']

    result['blend'] += result['blending']

    result['bring'] += result['brings'] + result['brought']

    result['build'] += result['building']

    result['burn'] += result['burning'] + result['burnt']

    result['contain'] += result['contains']

    result['creamy'] += result['creamier']

    result['dark'] += result['darker']

    result['deep'] += result['deeper']

    result['deliver'] += result['delivers']

    result['develop'] += result['develops']

    result['dominate'] += result['dominates']

    result['drink'] += result['drinking']

    result['drive'] += result['driven']

    result['emerge'] += result['emerges']

    result['evolve'] += result['evolves']

    result['fade'] += result['fading']

    result['feel'] += result['felt']

    result['find'] += result['found']

    result['floral'] += result['florals']

    result['fresh'] += result['fresher']

    result['fruity'] += result['fruitier']

    result['full'] += result['fuller']

    result['grow'] += result['grown']

    result['hold'] += result['held']

    result['keep'] += result['kept']

    result['linger'] += result['lingers']

    result['long'] += result['longer']

    result['marry'] += result['married']

    result['mix'] += result['mixed']

    result['north'] += result['northern']

    result['oak'] += result['oaked']

    result['offer'] += result['offering']

    result['open'] += result['opening']

    result['peat'] += result['peated']

    result['replace'] += result['replaces']

    result['reveal'] += result['reveals']

    result['rich'] += result['richer']

    result['sherry'] += result['sherried']

    result['suggest'] += result['suggests']

    result['surprise'] += result['surprising']

    result['sweet'] += result['sweeter']

    result['texture'] += result['textured']

    result['thick'] += result['thicker']

    result['toward'] += result['towards']

    result['wax'] += result['waxed']

    result.drop(columns = ['allows', 'becomes', 'blending', 'brings', 'brought', 'building','burning', 'burnt', 

                           'contains', 'darker', 'deeper', 'delivers', 'develops', 'dominates', 'drinking', 'driven', 'emerges',

                           'evolves', 'fading', 'felt', 'found', 'florals', 'fresher', 'fruitier', 'fuller', 'grown', 'held', 

                           'kept', 'lingers', 'longer', 'married', 'mixed', 'northern', 'oaked', 'offering', 'opening', 'peated', 

                           'replaces', 'reveals', 'richer', 'suggests', 'surprising', 'sweeter', 'textured', 'thicker', 'towards', 

                           'waxed', 'sherried', 

                          ], inplace=True)

    return result



def description_bow_matrix():

    df = read_input()

    cleansed = cleanse(df)

    desc_bow_mat = bow(cleansed.description)

    sqeezed = sqeeze_bow_matrix(desc_bow_mat)

    return sqeezed



# desc_bow_mat = description_bow_matrix()

# desc_bow_mat.columns.values[:1000]
df = split_name_df().merge(description_bow_matrix(), how='inner', left_index=True, right_index=True)

df.info()

df.to_csv('all_features.csv', index=False)
base_df = df[df.columns[:11]]

desc_df = df[df.columns[11:]]
import matplotlib.pyplot as plt

import seaborn as sns
print(base_df['price'].describe())

sns.distplot(base_df['price'])
base_df['log_price'] = np.log(base_df['price'])

print(base_df['log_price'].describe())

sns.distplot(base_df['log_price'])
print(base_df['age'].describe())

sns.distplot(base_df['age'].dropna())
base_df['log_age'] = np.log(base_df['age'])

print(base_df['log_age'].describe())

sns.distplot(base_df['log_age'].dropna())
print(base_df['alc'].describe())

sns.distplot(base_df['alc'].dropna())
print(base_df[['review_point', 'log_price', 'alc', 'log_age']].corr())

sns.pairplot(base_df[['review_point', 'log_price', 'alc', 'log_age']])
plt.figure(figsize=(12, 6))

print(base_df.category.value_counts())

sns.countplot(data=base_df, x='category')
plt.figure(figsize=(12, 8))

sns.boxplot(data=base_df, y='review_point', x='category')
plt.figure(figsize=(12, 8))

sns.boxplot(data=base_df, y='log_price', x='category')
freq = desc_df.sum(axis=0)

freq.nlargest(100).index.values
plt.figure(figsize=(20, 20))

sns.heatmap(df[['review_point'] + freq.nlargest(30).index.tolist()].corr(), annot=True)
bottler_brands = base_df.loc[base_df.distilled_at.notnull(), 'brand'].unique()

top20_cnt = base_df.brand.value_counts()[:20]

print(top20_cnt)

print('{} out of 20 are bottlers'.format(top20_cnt.index.isin(bottler_brands).sum()))
top20_review_point = base_df.groupby('brand')['review_point'].mean().nlargest(20)

print(top20_review_point)

print('{} out of 20 are bottlers'.format(top20_review_point.index.isin(bottler_brands).sum()))