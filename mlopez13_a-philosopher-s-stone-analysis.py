import numpy as np

import pandas as pd



# for the plots

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

%matplotlib inline

import seaborn as sns



import string

import pickle



# for linear regressions

from scipy import stats



# for linear regressions with train and test data

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score
filename = '/kaggle/input/harry-potter-and-the-philosophers-stone-script/hp_script.csv'

script = pd.read_csv(filename, encoding='cp1252')

script.head()
script.drop(columns='ID_number', inplace=True)

script.head()
script.loc[0, 'dialogue']
script.isna().sum()
lines = script['character_name'].value_counts()

lines['Harry Potter']
character = pd.DataFrame(script['character_name'].unique(), columns=['name'])

character['lines'] = character['name'].apply(lambda name: lines[name])

character.sort_values(by='lines', ascending=False, inplace=True)

character.head()
character['color'] = 'grey'



character.set_index('name', inplace=True)



character.loc['Harry Potter', 'color'] = 'green'

character.loc['Ron Weasley', 'color'] = 'red'

character.loc['Hermione Granger', 'color'] = 'brown'

character.head()
top10 = character.head(10)



plt.title('Top 10 number of lines of characters.')

plt.barh(top10.index, top10['lines'], color=top10['color'])

plt.gca().invert_yaxis()

plt.xlabel('Number of lines')

plt.ylabel('Character')

plt.grid()

plt.show()
script['words'] = script['dialogue'].apply(lambda x: len(x.split()))

script.head()
words = script[['character_name', 'words']].groupby('character_name').sum()

words.loc['Harry Potter', 'words']
character.reset_index(inplace=True)

character['words'] = character['name'].apply(lambda name: words.loc[name, 'words'])

character.set_index('name', inplace=True)

character.head()
top10lines = character.sort_values(by='lines', ascending=False).head(10)

top10words = character.sort_values(by='words', ascending=False).head(10)



fig, ax = plt.subplots(1, 2, figsize=(12, 5))



ax[0].set_title('Top 10 number of lines of characters.')

ax[0].barh(top10lines.index, top10lines['lines'], color=top10lines['color'])

ax[0].invert_yaxis()

ax[0].set_xlabel('Number of lines')

ax[0].set_ylabel('Character')

ax[0].grid()



ax[1].set_title('Top 10 number of words of characters.')

ax[1].barh(top10words.index, top10words['words'], color=top10words['color'])

ax[1].invert_yaxis()

ax[1].set_xlabel('Number of words')

ax[1].set_ylabel('Character')

ax[1].grid()



fig.tight_layout()



fig.show()
character.describe()
def line(slope, intercept, x):

    return slope * x + intercept
def words_vs_lines(df, scale = 'lin'):

    

    copy = df.copy()

    

    if scale == 'log':

        copy['lines'] = np.log(df['lines'])

        copy['words'] = np.log(df['words'])

    

    x = copy['lines']

    y = copy['words']

    c = copy['color']

    

    # subset for text labels

    mask = df['words'] > 400

    t = copy[mask]



    # LINEAR REGRESSION with H, R, H

    s1, i1, r1, p1, e1 = stats.linregress(x, y)

    if scale == 'log':

        label1=f'HRH: log(words) = {s1.round(1)} * log(lines) + {i1.round(1)}, r2 = {r1.round(2)}'

    else:

        label1=f'HRH: words = {s1.round(1)} * lines + {i1.round(1)}, r2 = {r1.round(2)}'

    

    # LINEAR REGRESSION without H, R, H

    main_characters = ['Harry Potter', 'Ron Weasley', 'Hermione Granger']

    xx = x[~x.index.isin(main_characters)]

    yy = y[~y.index.isin(main_characters)]

    s2, i2, r2, p2, e2 = stats.linregress(xx, yy)

    if scale == 'log':

        label2=f'noHRH: log(words) = {s2.round(1)} * log(lines) + {i2.round(1)}, r2 = {r2.round(2)}'

    else:

        label2=f'noHRH: words = {s2.round(1)} * lines + {i2.round(1)}, r2 = {r2.round(2)}'

    

    # FIGURE

    plt.title('Words versus lines.')

    if scale == 'log':

        plt.xlabel('Log of number of lines')

        plt.ylabel('Log of number of words')

    else:

        plt.xlabel('Number of lines')

        plt.ylabel('Number of words')



    # scatter

    plt.scatter(x, y, c='none', edgecolor=c)

    t.apply(lambda row: plt.text(row['lines'], row['words'], row.name, c=row['color']), axis=1)



    # lines

    x_array = np.array([min(x), max(x)])

    plt.plot(x_array, line(s1, i1, x_array), c='g', ls='--', lw=1, label=label1)

    plt.plot(x_array, line(s2, i2, x_array), c='k', lw=1, label=label2)



    # limits

    margin = (max(y) - min(y)) / 10

    plt.ylim([min(y) - margin, max(y) + margin])

        

    # legend

    plt.legend()

    

    plt.show()
words_vs_lines(character)

words_vs_lines(character, 'log')
def linreg(X_train, X_test, y_train, y_test):



    # 1. Linear Regression.

    reg = LinearRegression()

    reg.fit(X_train, y_train)



    pred = reg.predict(X_test)



    print(f'y = {reg.intercept_.round(2)}', end=' ')

    i = 0

    for c in reg.coef_.round(2):

        i += 1

        print(f'+ {c} * x{i}', end=' ')

    

    r2_test = r2_score(y_test, pred)

    r2_train = r2_score(y_train, reg.predict(X_train))



    up = max(max(pred), max(y_test))

    down = min(min(pred), min(y_test))

    margin = (up - down) / 20

    

    # 2. Plot.

    plt.title('Linear regression.')

    plt.scatter(pred, y_test, color='black', label=f'test, r2 = {r2_test.round(2)}')

    plt.scatter(y_train, reg.predict(X_train), color='none', edgecolor='black', label=f'train, r2 = {r2_train.round(2)}')

    plt.plot([down - margin, up + margin], [down - margin, up + margin], linewidth=1, color='blue')

    plt.xlabel('Prediction')

    plt.ylabel('Truth')

    plt.xlim([down - margin, up + margin])

    plt.ylim([down - margin, up + margin])

    plt.legend()

    plt.grid()



    plt.show()
X = character[['lines']]

y = character['words']



X_train, X_test, y_train, y_test = train_test_split(X, y)



linreg(X_train, X_test, y_train, y_test)
X_train = np.log(X_train)

X_test = np.log(X_test)

y_train = np.log(y_train)

y_test = np.log(y_test)



linreg(X_train, X_test, y_train, y_test)
character['wpl'] = character.apply(lambda row: row.words / row.lines, axis=1)

character.head()
top30wpl = character.sort_values(by='wpl', ascending=False).head(30)



plt.figure(figsize=(10,10))



plt.title('Top 30 number of words per line of characters.')

plt.barh(top30wpl.index, top30wpl['wpl'], color=top30wpl['color'])

plt.gca().invert_yaxis()

plt.xlabel('Words per line')

plt.ylabel('Character')

plt.grid()

plt.show()
words = script[['character_name', 'dialogue']].copy()

words.columns = ['character', 'word']

words.head()
words['word'] = words['word'].str.replace('[^\w\s]', '')

words['word'] = words['word'].str.lower()

words.head()
words['word'] = words['word'].str.split()

words.head()
words = words.explode('word').reset_index(drop=True)

words.head()
def say_my_name(name):

    df = words[words['word'] == name]

    df = df.groupby('character').count()

    df = df.sort_values(by='word', ascending=False)

    

    top10 = df.copy().head(10)

    

    top10['color'] = 'grey'

    top10.loc['Harry Potter', 'color'] = 'green'

    top10.loc['Ron Weasley', 'color'] = 'red'

    top10.loc['Hermione Granger', 'color'] = 'brown'

    

    plt.title(f'Who said {name}?')

    plt.barh(top10.index, top10['word'], color=top10['color'])

    plt.gca().invert_yaxis()

    plt.xlabel(f'Number of times a character says {name}')

    plt.ylabel('Character')

    plt.grid()

    plt.show()
say_my_name('voldemort')
from nltk import word_tokenize

from nltk.tag import pos_tag

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords
contractions_dict = {"ain't": 'am not', "aren't": 'are not', "can't": 'cannot', "can't've": 'cannot have', "'cause": 'because', "could've": 'could have', "couldn't": 'could not', "couldn't've": 'could not have', "didn't": 'did not', "doesn't": 'does not', "don't": 'do not', "hadn't": 'had not', "hadn't've": 'had not have', "hasn't": 'has not', "haven't": 'have not', "he'd": 'he would', "he'd've": 'he would have', "he'll": 'he will', "he'll've": 'he will have', "he's": 'he is', "how'd": 'how did', "how'd'y": 'how do you', "how'll": 'how will', "how's": 'how is', "i'd": 'i would', "i'd've": 'i would have', "i'll": 'i will', "i'll've": 'i will have', "i'm": 'i am', "i've": 'i have', "isn't": 'is not', "it'd": 'it had', "it'd've": 'it would have', "it'll": 'it will', "it'll've": 'it will have', "it's": 'it is', "let's": 'let us', "ma'am": 'madam', "mayn't": 'may not', "might've": 'might have', "mightn't": 'might not', "mightn't've": 'might not have', "must've": 'must have', "mustn't": 'must not', "mustn't've": 'must not have', "needn't": 'need not', "needn't've": 'need not have', "o'clock": 'of the clock', "oughtn't": 'ought not', "oughtn't've": 'ought not have', "shan't": 'shall not', "sha'n't": 'shall not', "shan't've": 'shall not have', "she'd": 'she would', "she'd've": 'she would have', "she'll": 'she will', "she'll've": 'she will have', "she's": 'she is', "should've": 'should have', "shouldn't": 'should not', "shouldn't've": 'should not have', "so've": 'so have', "so's": 'so is', "that'd": 'that would', "that'd've": 'that would have', "that's": 'that is', "there'd": 'there had', "there'd've": 'there would have', "there's": 'there is', "they'd": 'they would', "they'd've": 'they would have', "they'll": 'they will', "they'll've": 'they will have', "they're": 'they are', "they've": 'they have', "to've": 'to have', "wasn't": 'was not', "we'd": 'we had', "we'd've": 'we would have', "we'll": 'we will', "we'll've": 'we will have', "we're": 'we are', "we've": 'we have', "weren't": 'were not', "what'll": 'what will', "what'll've": 'what will have', "what're": 'what are', "what's": 'what is', "what've": 'what have', "when's": 'when is', "when've": 'when have', "where'd": 'where did', "where's": 'where is', "where've": 'where have', "who'd": 'who would', "who'll": 'who will', "who'll've": 'who will have', "who's": 'who is', "who've": 'who have', "why's": 'why is', "why've": 'why have', "will've": 'will have', "won't": 'will not', "won't've": 'will not have', "would've": 'would have', "wouldn't": 'would not', "wouldn't've": 'would not have', "y'all": 'you all', "y'alls": 'you alls', "y'all'd": 'you all would', "y'all'd've": 'you all would have', "y'all're": 'you all are', "y'all've": 'you all have', "you'd": 'you had', "you'd've": 'you would have', "you'll": 'you you will', "you'll've": 'you you will have', "you're": 'you are', "you've": 'you have'}
script['tokens'] = script['dialogue'].str.lower()

script.head()
def dict_replace(sentence):

    for key in contractions_dict:

        sentence = sentence.replace(key, contractions_dict[key])

    return sentence
script['tokens'] = script['tokens'].apply(dict_replace)

script.head()
script['tokens'] = script['tokens'].apply(word_tokenize)

script.head()
stop_words = stopwords.words('english')



def clean_tokens(tokens_list):

    cleaned_tokens_list = []

    

    # Identify Part Of Speech (POS)

    for token, tag in pos_tag(tokens_list):

        if tag == 'NN' or tag == 'NNS':

            # Noun (non proper)

            pos = 'n'

        elif tag.startswith('VB'):

            # Verb

            pos = 'v'

        elif tag.startswith('JJ'):

            # Adjective

            pos = 'a'

        else:

            continue

        

        # Lemmatize (for instance, cats -> cat, bringing -> bring, great -> good)

        lemmatizer = WordNetLemmatizer()

        token = lemmatizer.lemmatize(token, pos)

        

        # Filter out punctuation marks and stop_words

        if token not in string.punctuation and token.lower() not in stop_words:

            cleaned_tokens_list.append(token.lower())

        

    return cleaned_tokens_list
script['clean_tokens'] = script['tokens'].apply(clean_tokens)

script.head()
filename = '/kaggle/input/twitter-sentiment-analysis/twitter_model.sav'

model = pickle.load(open(filename, 'rb'))
dist = model.prob_classify({'kill': True})

dist.prob('pos')
script['dict'] = script.apply(lambda row: dict([token, True] for token in row['clean_tokens']), axis=1)

script.head()
script['sentiment'] = script.apply(lambda row: model.prob_classify(row['dict']).prob('pos'), axis=1)

script.head()
words = ['good', 'evening', 'professor', 'dumbledore', 'rumour', 'true', 'hagrid', 'bring']



print('Probability that a tweet with some word is positive:\n')



for word in words:

    dist = model.prob_classify({word: True})

    x = dist.prob('pos') * 100

    category = 'POS' if x > 55 else ('NEG' if x < 45 else 'neutral')

    print(f'p({word}) = {round(x, 2)} % ({category})')
mask = script['character_name'] == 'Draco Malfoy'

columns = ['dialogue', 'clean_tokens', 'sentiment']

script.loc[mask, columns].head(10)
df = script.copy()

df['sentiment'] = (2 * df['sentiment'] - 1).round(5)

df['sentiment'] = df.apply(lambda row: row['sentiment'] if len(row['dict']) > 0 else 0, axis=1)

df['sentiment_cat'] = df['sentiment'].apply(lambda x: 'POS' if x > 0.05 else ('NEG' if x < -0.05 else 'neutral'))

df.drop(columns=['tokens', 'dict'], inplace=True)

df.head()
np.mean(df['sentiment'])
plt.title('Sentiment distribution.')

plt.hist(df['sentiment'], bins=11, color='b', edgecolor='k')

plt.xlabel('Sentiment')

plt.ylabel('Counts')

plt.grid()

plt.show()
total = df[['character_name', 'sentiment']].groupby('character_name').sum()

total.loc['Harry Potter', 'sentiment']
character.reset_index(inplace=True)

character['sentiment'] = character.apply(lambda row: total.loc[row['name'], 'sentiment'] / lines[row['name']], axis=1)

character['sentiment_cat'] = character['sentiment'].apply(lambda x: 'POS' if x > 0.05 else ('NEG' if x < -0.05 else 'neutral'))

character.set_index('name', inplace=True)

character.head()
character.describe()
mask = character['words'] > 200

columns = ['sentiment', 'sentiment_cat']

character.loc[mask, columns].sort_values(by='sentiment', ascending=False)
mask = script['character_name'] == 'Petunia Dursley'

columns = ['dialogue', 'clean_tokens', 'sentiment', 'sentiment_cat']

df.loc[mask, columns].head(10)
mask = script['character_name'] == 'Rubeus Hagrid'

columns = ['dialogue', 'clean_tokens', 'sentiment', 'sentiment_cat']

df.loc[mask, columns].head(10)
tokens = ['wizard', 'harry']



print('Probability that a tweet with some word is positive:\n')



for token in tokens:

    dist = model.prob_classify({token: True})

    x = dist.prob('pos') * 100

    category = 'POS' if x > 55 else ('NEG' if x < 45 else 'neutral')

    print(f'p({token}) = {round(x, 2)} % ({category})')