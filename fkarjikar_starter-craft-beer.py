%autosave 0
%matplotlib inline
import string
from  collections import Counter
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import psycopg2
import seaborn as sbn
from altair import Chart, X, Y, Color, Scale
import altair as alt
import requests
import nltk
from nltk.corpus import stopwords 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from graphviz import Digraph

matplotlib.style.use('ggplot')


bf = pd.read_csv('Data/beer_train.csv')
bf.head()
bf.isna().any(axis=0)
print("total len = ", len(bf))
print("missing: ", len(bf[bf['user/gender'].isna() == True]))
alt.Chart(bf[['review/overall']].sample(5000)).mark_bar().encode(alt.X('review/overall', bin=True), alt.Y('count()'))


dot = Digraph(comment='The Round Table')

dot.node('A', 'Load Data')
dot.node('B', 'Group Data')
dot.node('L', 'Aggregate')
dot.node('S', 'Sort on rating from high to low')
dot.node('J', 'Join back to original')
dot.node('F', 'filter out extra columns')
dot.edges(['AB', 'BL', 'LS', 'SJ', 'JF'])

dot
byid = bf.groupby('beer/beerId')[['review/overall']].mean().sort_values(['review/overall'], ascending=False)
byid.merge(bf[['beer/beerId', 'beer/name', 'beer/style']], left_index=True, right_on='beer/beerId').head(10)

bf['beer/style'].unique()
bf.groupby('beer/style')['review/overall'].agg(['mean','count']).sort_values('mean', ascending=False).head(10)

alt.Chart(bf.sample(5000)).mark_point().encode(y='review/overall',x='review/taste')
alt.Chart(bf[bf['user/gender'].isna() == False].sample(5000)).mark_point().encode(y='review/overall',x='review/taste', size='count()', color='user/gender:N')

cm = bf[['review/appearance','review/overall','review/palate', 'review/taste', 'beer/ABV']].corr()
cm
cm.reset_index(inplace=True)
cm = cm.melt(id_vars='index', value_vars=['review/appearance','review/overall','review/palate', 'review/taste', 'beer/ABV'],  value_name='corr', var_name='attribute')

cm.head()
alt.Chart(cm,height=300, width=300).mark_rect().encode(x='index',y='attribute',color='corr',tooltip='corr')
# The default size makes the graph smaller than the legend and it lookks rather odd. specifying height and width improves the aesthetics
alt.Chart(bf.sample(5000)).mark_point().encode(y='review/overall',x='review/aroma', size='count()')
beer_X = bf[['review/taste']]
beer_y = bf[['review/overall']]
train_X, test_X, train_y, test_y = train_test_split(beer_X, beer_y, random_state=42)

rmodel = LinearRegression()
rmodel.fit(train_X.values, train_y.values)
preds = rmodel.predict(test_X)
preds[:20]
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("MAE = ", mean_absolute_error(test_y, preds))
print("MSE = ", mean_squared_error(test_y, preds))

alt.Chart(pd.DataFrame({'actual': test_y['review/overall'].values[:5000], 'predicted':preds.ravel()[:5000]})).mark_point().encode(x='actual',y='predicted',size='count()')
alt.Chart(pd.DataFrame({'rating': np.concatenate((preds.ravel()[:2500], test_y['review/overall'].values[:2500])), 
                        'label':['pred']*2500 + ['actual']*2500 })).mark_bar().encode(
    x=alt.X('rating', bin=True), y='count()', color='label')
train_y['review/overall'].mean()
len(preds)
avg_preds = np.full(9373, train_y['review/overall'].mean())
abs(avg_preds-test_y.values).mean()
beer_X = bf[['review/taste', 'review/palate', 'review/appearance']]
beer_y = bf[['review/overall']]
train_X, test_X, train_y, test_y = train_test_split(beer_X, beer_y, random_state=42)
rmodel = LinearRegression()
rmodel.fit(train_X.values, train_y.values)
preds = rmodel.predict(test_X)
abs(preds-test_y.values).mean()
beer_X = bf[['review/taste', 'review/palate', 'review/appearance', 'beer/ABV']]
beer_y = bf[['review/overall']]
train_X, test_X, train_y, test_y = train_test_split(beer_X, beer_y, random_state=42)
rmodel = LinearRegression()
rmodel.fit(train_X.values, train_y.values)
preds = rmodel.predict(test_X)
abs(preds-test_y.values).mean()
scaler = StandardScaler()
scaled_X = scaler.fit_transform(beer_X)
train_X, test_X, train_y, test_y = train_test_split(scaled_X, beer_y, random_state=42)
rmodel = LinearRegression()
rmodel.fit(train_X, train_y.values)
preds = rmodel.predict(test_X)
abs(preds-test_y.values).mean()
np.sqrt(((preds-test_y) ** 2.0).mean())
alt.Chart(bf.sample(5000)).mark_bar().encode(alt.X('beer/ABV', bin=True), y='count()')
bf['beer/ABV'].describe()

# start by converting everything to a thumbs up / thumbs down rating.
bf['thumbs'] = bf['review/overall'].map(lambda x : 1 if x >= 3.5 else 0)
beer_X = bf[['review/taste', 'review/palate', 'review/appearance', 'beer/ABV']]
beer_y = bf[['thumbs']]
train_X, test_X, train_y, test_y = train_test_split(beer_X, beer_y, random_state=42)
rmodel = LogisticRegression(solver='lbfgs')
rmodel.fit(train_X.values, train_y.values.ravel())
preds = rmodel.predict(test_X)
mean_absolute_error(test_y, preds)
alt.Chart(pd.DataFrame({'preds':preds.ravel()[:5000]})).mark_bar().encode(alt.X('preds',bin=True), y='count()') + \
alt.Chart(bf.sample(5000)).mark_bar(color='red',opacity=0.5).encode(alt.X('thumbs',bin=True),y='count()')
accuracy_score(test_y, preds)


bf['review/text'] = bf['review/text'].str.lower()
bf.loc[:5,'review/text']
import string
bf['review/text'] = bf['review/text'].str.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))

bf['text_list'] = bf['review/text'].str.split()
bf.dropna(subset=['review/text'], inplace=True)
c = Counter()

tl = 0
for row in bf.text_list:
    c.update(row)
    tl += len(row)

print('total number of words ', tl)
print('unique words = ', len(c.keys()))
sw = set(stopwords.words('english'))
for w in sw:
    del c[w]
ll = 0
for w in c:
    if c[w] < 50:
        ll += 1

ll
len(c) - ll

remove_list = []
for w in c:
    if c[w] < 50:
        remove_list.append(w)

for w in remove_list:
    del c[w]

len(c)

%%time

for w in c.keys():
    bf[w] = bf.text_list.map(lambda x : 1 if w in x else 0)
    
    
worddf = bf[list(c.keys())]
ratings = bf['review/overall']
train_X, test_X, train_y, test_y = train_test_split(worddf, ratings, random_state=43)
%%time

model = LinearRegression(n_jobs=-1)
model.fit(train_X, train_y)
preds = model.predict(test_X)
mean_absolute_error(test_y, preds)
alt.Chart(pd.DataFrame({'preds':preds[:5000]})).mark_bar().encode(alt.X('preds:Q', bin=True),y='count()')
worddf.describe()
num_ones = worddf.sum().sum()
total_cells = len(worddf.columns) * len(worddf)

print(num_ones, total_cells, num_ones/total_cells)
word_freqs = sorted(c.values(), reverse=True)
plt_df = pd.DataFrame({'x':range(len(word_freqs)), 'freq':word_freqs})
plt_df.head()
alt.Chart(plt_df).mark_line().encode(x='x',y='freq',tooltip='freq').interactive()
model.coef_[:20].tolist()
weight_words = zip(train_X.columns.tolist(), model.coef_.tolist())
#list(weight_words)[:10]
from operator import itemgetter
get_firsti = itemgetter(1)
sorted_weights = sorted(list(weight_words), key=lambda x: x[1], reverse=True)

sorted_weights[:10]
sorted_weights[-10:]
weight_word_dict = dict(sorted_weights)
for word in ['awesome', 'tasty', 'awful', 'delicious']:
    print(word, weight_word_dict[word])
# create our sets
top_weights = set([x[0] for x in sorted_weights[:10]])
bottom_weights = set([x[0] for x in sorted_weights[-10:]])
bf[bf.text_list.map(lambda x:  len(set(x).intersection(top_weights)) > 0)]['review/overall'].mean()
bf[bf.text_list.map(lambda x: not set(x).isdisjoint(top_weights))]['review/text'].head()
bf[bf.text_list.map(lambda x:  len(set(x).intersection(bottom_weights)) > 0)]['review/overall'].mean()
tot = sum(c.values(), 0.0)
for w in c:
    c[w] /= tot


c.most_common(10)
c.most_common()[-10:]


%%time

for w in c.keys():
    bf[w] = bf.text_list.map(lambda x : c[w] if w in x else 0)

bf.head()
worddf = bf[list(c.keys())]
ratings = bf['review/overall']
train_X, test_X, train_y, test_y = train_test_split(worddf, ratings, random_state=43)
%%time

model = LinearRegression(n_jobs=-1)
model.fit(train_X, train_y)
preds = model.predict(test_X)
mean_absolute_error(test_y, preds)
comp_df = pd.DataFrame({'predict':preds.tolist(), 'actual':test_y.values.tolist()})
comp_df.head()
comp_df['predict'] = comp_df.predict.map(lambda x : round(x,2))
comp_df.head()
alt.Chart(comp_df.sample(5000)).mark_point().encode(x='actual:Q',y='predict:Q',size='count()')
sbn.violinplot(data=comp_df,x='actual', y='predict')
comp_df.corr()
comp_dfs = comp_df.sample(5000)
lower_box = 'q1(predict):Q'  # returns the upper boundary of the lower quartile of values.
lower_whisker = 'min(predict):Q'
upper_box = 'q3(predict):Q' # returns the lower boundary of the upper quartile of values.
upper_whisker = 'max(predict):Q'

# Compose each layer individually
lower_plot = alt.Chart(comp_dfs, width=300).mark_rule().encode(
    y=alt.Y(lower_whisker, axis=alt.Axis(title="prediction")),
    y2=lower_box,
    x='actual:O'
)

middle_plot = alt.Chart(comp_dfs,width=300).mark_bar(size=5.0).encode(
    y=lower_box,
    y2=upper_box,
    x='actual:O'
)

upper_plot = alt.Chart(comp_dfs, width=300).mark_rule().encode(
    y=upper_whisker,
    y2=upper_box,
    x='actual:O'
)

middle_tick = alt.Chart(comp_dfs, width=300).mark_tick(
    color='white',
    size=5.0
).encode(
    y='median(predict):Q',
    x='actual:O',
)

# the plus operator overlays the individual plots and the result is displayed all together.
lower_plot + upper_plot + middle_plot + middle_tick











