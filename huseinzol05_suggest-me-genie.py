import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import seaborn as sns
import spacy
import re
sns.set()
py.init_notebook_mode(connected = True)
nlp = spacy.load('en')
def clearstring(string):
    string = re.sub('[^a-z ]+', '', string)
    string = string.split(' ')
    string = filter(None, string)
    string = ' '.join([y.strip() for y in string])
    tags = ','.join([str(i) for i in nlp(string) if i.pos_ in ['NOUN']])
    return tags
# df = pd.read_csv('../input/ks-projects-201801.csv',encoding = "ISO-8859-1", keep_default_na=False)
# df['year'] = pd.DatetimeIndex(df['launched']).year
# tags = []
# for i in range(df.shape[0]):
#     try:
#         tags.append(clearstring(df.iloc[i,1].lower()))
#     except:
#         print(df.iloc[i,1])
# df['tags'] = tags
# df.head()

# Skip this process to reduce time completion. I already uploaded processed CSV in this kernel
df = pd.read_csv('../input/dear-genie-kickstarter/dear_jin.csv',encoding = "ISO-8859-1", keep_default_na=False)
df.head()
year_unique, year_count = np.unique(df['year'], return_counts = True)
data = [go.Bar(
            x=year_unique[1:],
            y=year_count[1:],
    text=year_count[1:],
    textposition = 'auto',
            marker=dict(
                color='rgb(179, 224, 255)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'Year count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
state = df.state.unique().tolist()
del state[state.index('live')]
data_bar = []
for i in state:
    year_unique, year_count = np.unique(df[df.state==i]['year'], return_counts = True)
    data_bar.append(go.Bar(x=year_unique[1:],y=year_count[1:],name=i))
layout = go.Layout(
    title = 'State per Year count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data_bar, layout = layout)
py.iplot(fig)
main_category_unique, main_category_count = np.unique(df.main_category,return_counts = True)
data = [go.Bar(
            x=main_category_unique,
            y=main_category_count,
    text=main_category_count,
    textposition = 'auto',
            marker=dict(
                color='rgb(179, 204, 255)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'Category count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
data_bar = []
for i in state:
    main_category_unique, main_category_count = np.unique(df[df.state==i]['main_category'], return_counts = True)
    data_bar.append(go.Bar(x=main_category_unique,y=main_category_count,name=i))
layout = go.Layout(
    title = 'State per Main Category count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data_bar, layout = layout)
py.iplot(fig)
film_category_unique, film_category_count = np.unique(df[df.main_category == 'Film & Video'].category, return_counts=True)
data = [go.Bar(
            x=film_category_unique,
            y=film_category_count,
    text=film_category_count,
    textposition = 'auto',
            marker=dict(
                color='rgb(255, 224, 179)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'Category inside Film & Video',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
data_bar = []
for i in state:
    main_category_unique, main_category_count = np.unique(df[(df.state==i) & (df.main_category == 'Film & Video')].category, return_counts = True)
    data_bar.append(go.Bar(x=main_category_unique,y=main_category_count,name=i))
layout = go.Layout(
    title = 'State per Category inside Film & Video count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data_bar, layout = layout)
py.iplot(fig)
film_category_unique, film_category_count = np.unique(df[df.main_category == 'Music'].category, return_counts=True)
data = [go.Bar(
            x=film_category_unique,
            y=film_category_count,
    text=film_category_count,
    textposition = 'auto',
            marker=dict(
                color='rgb(255, 224, 179)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'Category inside Music',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
data_bar = []
for i in state:
    main_category_unique, main_category_count = np.unique(df[(df.state==i) & (df.main_category == 'Music')].category, return_counts = True)
    data_bar.append(go.Bar(x=main_category_unique,y=main_category_count,name=i))
layout = go.Layout(
    title = 'State per Category inside Music count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data_bar, layout = layout)
py.iplot(fig)
tags_succesful=(','.join(df[df.state=='successful'].tags.values.tolist())).split(',')
tags_succesful_unique, tags_succesful_count = np.unique(tags_succesful,return_counts = True)
ids=(-tags_succesful_count).argsort()[:20]
data = [go.Bar(
            x=tags_succesful_unique[ids],
            y=tags_succesful_count[ids],
    text=tags_succesful_count[ids],
    textposition = 'auto',
            marker=dict(
                color='rgb(217, 217, 217)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'Top 20 Unigram keywords for successful projects',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
trigram = []
for i in range(len(tags_succesful)-3):
    trigram.append(', '.join(tags_succesful[i:i+3]))
trigram_succesful_unique, trigram_succesful_count = np.unique(trigram,return_counts = True)
ids=(-trigram_succesful_count).argsort()[:20]
data = [go.Bar(
            x=trigram_succesful_unique[ids],
            y=trigram_succesful_count[ids],
    text=trigram_succesful_count[ids],
    textposition = 'auto',
            marker=dict(
                color='rgb(217, 217, 217)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'Top 20 Trigram keywords for successful projects',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
gram_5 = []
for i in range(len(tags_succesful)-5):
    gram_5.append(', '.join(tags_succesful[i:i+5]))
gram_5_succesful_unique, gram_5_succesful_count = np.unique(gram_5,return_counts = True)
ids=(-gram_5_succesful_count).argsort()[:20]
data = [go.Bar(
            x=gram_5_succesful_unique[ids],
            y=gram_5_succesful_count[ids],
    text=gram_5_succesful_count[ids],
    textposition = 'auto',
            marker=dict(
                color='rgb(217, 217, 217)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'Top 20 5-Gram keywords for successful projects',
     margin = dict(
        t = 50,
         b= 200
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
decision = df.state.values
data_array = df.pledged.values
data_bar = []
for no, k in enumerate(state):
    weights = np.ones_like(data_array[decision == k])/float(len(data_array[decision == k]))
    n, bins, _ = plt.hist(data_array[decision == k], 10,weights=weights)
    loc = np.where(n >= 0.5)[0]
    plt.clf()
    data_bar.append(go.Bar(x=bins[loc],y=n[loc],name=k))
layout = go.Layout(
    title = 'Probability how much pledged to be for states',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data_bar, layout = layout)
py.iplot(fig)
decision = df.state.values
data_array = df.backers.values
data_bar = []
for no, k in enumerate(state):
    weights = np.ones_like(data_array[decision == k])/float(len(data_array[decision == k]))
    n, bins, _ = plt.hist(data_array[decision == k], 10,weights=weights)
    loc = np.where(n >= 0.5)[0]
    plt.clf()
    data_bar.append(go.Bar(x=bins[loc],y=n[loc],name=k))
layout = go.Layout(
    title = 'Probability how much backers to be for states',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data_bar, layout = layout)
py.iplot(fig)
sns.pairplot(df[['goal','pledged','state','backers']], hue="state",size=5)
plt.cla()
plt.show()
tags_to_train = df.tags.iloc[np.where((df.tags != '') & (df.state != 'live'))[0]].tolist()
label_to_train = df.state.iloc[np.where((df.tags != '') & (df.state != 'live'))[0]]
# change to binary classification
label_to_train[label_to_train == 'canceled'] = 'failed'
label_to_train[label_to_train == 'undefined'] = 'failed'
label_to_train[label_to_train == 'suspended'] = 'failed'
label_to_train = label_to_train.tolist()
for i in range(len(tags_to_train)):
    tags_to_train[i] = tags_to_train[i].replace(',', ' ')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
Y = LabelEncoder().fit_transform(label_to_train)
bow = CountVectorizer().fit(tags_to_train)
X = bow.transform(tags_to_train)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
clf_huber = SGDClassifier(loss = 'modified_huber', 
                                  penalty = 'l2', alpha = 1e-3, 
                                  n_iter = 50).fit(X, Y)
clf_bayes = MultinomialNB().fit(X, Y)
stacked=np.hstack([clf_bayes.predict_proba(X), clf_huber.predict_proba(X)])
import xgboost as xgb
params_xgd = {
    'max_depth': 7,
    'objective': 'binary:logistic',
    'learning_rate': 0.05,
    'nthread': -1,
    'silent': False,
    'n_estimators': 100
    }
clf = xgb.XGBClassifier(**params_xgd)
clf.fit(stacked,Y)
fig, ax = plt.subplots(figsize=(12,5))
xgb.plot_importance(clf, ax=ax)
plt.show()
print(np.mean(clf_bayes.predict(X) == Y))
print(np.mean(clf_huber.predict(X) == Y))
print(np.mean(clf.predict(stacked) == Y))
from fuzzywuzzy import fuzz
import random
guess = ['bicyle', 'shit']
result_guess = []
tags = df.tags.values.tolist()
for i in guess:
    picked=np.argmax([fuzz.ratio(guess[1], n) for n in tags_succesful_unique])
    results=np.where(np.array([n.find(tags_succesful_unique[picked]) for n in tags]) >=0)[0][:10]
    for k in results:
        count = 0
        while True and count < 10:
            selected = random.choice(tags[k].split(','))
            if selected not in result_guess:
                result_guess.append(selected)
                break
            count+=1
result_guess
from sklearn.neighbors import NearestNeighbors
from random import shuffle
def help_me_genie(wish, suggest_count):
    if wish.find('oh genie, suggest me') < 0:
        return "you need to call me by 'oh genie, suggest me'"
    guess = [i.strip() for i in wish[len('oh genie, suggest me '):].split(',')]
    print('your wish is:',guess)
    result_guess = []
    for i in guess:
        picked=np.argmax([fuzz.ratio(guess[1], n) for n in tags_succesful_unique])
        results=np.where(np.array([n.find(tags_succesful_unique[picked]) for n in tags]) >=0)[0][:10]
        for k in results:
            for n in range(20):
                selected = random.choice(tags[k].split(','))
                if selected not in result_guess:
                    result_guess.append(selected)
                    break
                
    print('your result guess:', result_guess)
    jin_guess=np.zeros((df.shape[0], len(result_guess)))
    for i in range(df.shape[0]):
        for k in range(len(result_guess)):
            if tags[i].find(result_guess[k]) >= 0:
                jin_guess[i, k] += 1
    nbrs = NearestNeighbors(n_neighbors=suggest_count, algorithm='auto', metric='sqeuclidean').fit(jin_guess)
    id_entry = np.argmax(np.sum(jin_guess,axis=1)) 
    xtest = jin_guess[id_entry, :].reshape(1, -1)
    distances, indices = nbrs.kneighbors(xtest)
    results = []
    for i in indices[0][:]:
        items = tags[i].split(',') + [random.choice(result_guess)]
        shuffle(items)
        results.append(' '.join(items))
    prob=clf.predict_proba(np.hstack([clf_huber.predict_proba(bow.transform(results)), clf_bayes.predict_proba(bow.transform(results))]))
    for i in range(len(results)):
        print(results[i], ', successful rate:', prob[i,1]*100, '%')
help_me_genie('oh genie, suggest me bicycle, invest', 10)
help_me_genie('oh genie, suggest me taylor swift, broom', 10)
help_me_genie('oh genie, suggest me guitar, shit, trousers', 30)
help_me_genie('guitar, shit, trousers', 20)
help_me_genie('oh genie, suggest me guitar, shit, trousers, smart phone, aloe vera', 20)
