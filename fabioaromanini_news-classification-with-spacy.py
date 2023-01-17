import pandas as pd

news_data = pd.read_json('../input/News_Category_Dataset_v2.json', lines=True)

news_data.head()
news_data.info()
news_example = news_data.loc[0]

print(f"""

Headline

{news_example['headline']}



Short Description

{news_example['short_description']}



Category

{news_example['category']}



Link

{news_example['link']}

""")
print(f"""

Category count: {len(news_data['category'].value_counts())}

{news_data['category'].value_counts()}

""")

%matplotlib inline

import seaborn as sns



sns.set(rc={'figure.figsize':(15,9)})
values = news_data['category'].value_counts()[

    ['POLITICS', 'ENTERTAINMENT', 'TRAVEL', 'STYLE & BEAUTY', 'QUEER VOICES', 'DIVORCE','SCIENCE', 'MONEY', 'EDUCATION']

]



sns.barplot(values.index, values)
print(f"""

Authors count: {len(news_data['authors'].value_counts())}

{news_data['authors'].value_counts()}

""")

values = news_data['authors'].value_counts()[['', 'Ron Dicker', 'Reuters, Reuters', 'Curtis M. Wong', 'Julie R. Thomson', 'Hugh Bronstein, Reuters']]

sns.barplot(values.index, values)
news_data['category'] = news_data['category'].apply(lambda row: 'WORLDPOST' if row == 'THE WORLDPOST' else row)

len(news_data['category'].unique())
df = pd.DataFrame()

category_sample_size = 200



for category in news_data['category'].unique():

    df = df.append(

        news_data[news_data['category'] == category][:category_sample_size].drop(['authors', 'date',  'link'], axis=1),

        ignore_index=True

    )



df.info()
df['++category++'] = df['category']

df['++headline++'] = df['headline']

df['++short_description++'] = df['short_description']



df = df.drop(['category', 'headline', 'short_description'], axis=1)

df.info()
import spacy

nlp = spacy.load('en_core_web_sm')
from collections import Counter



def valid(token):

    return not (token.is_punct or token.is_stop)



def proccess(token):

    if token.is_digit: return '++digit++' 

    else: return token.lemma_.lower()



def get_words(text):

    doc = nlp(text)

    tokens = [proccess(token) for token in doc if valid(token)]

    word_freq = Counter(tokens)

    return word_freq.most_common(5)



def create_bag(news):

    return get_words(f"""{news['++headline++']} {news['++short_description++']}""")



df['++wc++'] = df.apply(create_bag, axis=1)



for i in range(5):

    news_instance = df.loc[i] 

    print(f"""{news_instance['++headline++']} {news_instance['++short_description++']}""")

    print(news_instance['++wc++'])

all_tokens = set()



def add_to_set(tokens):

    [all_tokens.add(token[0]) for token in tokens]



df['++wc++'].apply(add_to_set)



for token in all_tokens:

    df[token] = [0] * len(df['++headline++'])



print(len(all_tokens))

df.info()
def one_hot(news_instance):

    for token in news_instance['++wc++']:

        news_instance[token[0]] = token[1]

    

    return news_instance



df = df.apply(one_hot, axis=1)



for i in range(5):

    news_instance = df.loc[i]

    for col in news_instance.index:

        if news_instance[col] != 0:

            print(f"""{col}: {news_instance[col]}""")
from sklearn.model_selection import train_test_split



X = df.drop(['++category++', '++headline++', '++short_description++', '++wc++'], axis=1)

y = pd.get_dummies(df['++category++'])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.tree import DecisionTreeClassifier



clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score



accuracy_score(y_test, y_pred)