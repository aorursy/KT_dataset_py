import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from collections import Counter
import matplotlib.pyplot as plt
#import seaborn as sns
%matplotlib inline
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import FeatureUnion

train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
sub = pd.read_csv('../input/sample_submission.csv')
print('size of train data',train.shape)
print('size of test data',test.shape)
train.head()
test.head()
sub.head()
# checking missing data
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total missing', 'Percent missing'])
missing_train_data.head(20)
# checking missing data
total = test.isnull().sum().sort_values(ascending = False)
percent = (test.isnull().sum()/test.isnull().count()*100).sort_values(ascending = False)
missing_test_data  = pd.concat([total, percent], axis=1, keys=['Total missing', 'Percent missing'])
missing_test_data.head(20)
temp = train['cuisine'].value_counts()
trace = go.Bar(
    y=temp.index[::-1],
    x=(temp / temp.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color='blue',
    ),
)

layout = go.Layout(
    title = "Top cuisine with recipe count (%)",
    xaxis=dict(
        title='Recipe count',
        tickfont=dict(size=14,)),
    yaxis=dict(
        title='Cuisine',
        titlefont=dict(size=16),
        tickfont=dict(
            size=14)),
    margin=dict(
    l=200,
),
    
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
n=6714 # total ingredients in train data
top = Counter([item for sublist in train.ingredients for item in sublist]).most_common(n)
temp= pd.DataFrame(top)
temp.columns = ['ingredient','total_count']
temp = temp.head(20)
trace = go.Bar(
    y=temp.ingredient[::-1],
    x=temp.total_count[::-1],
    orientation = 'h',
    marker=dict(
        color='green',
    ),
)

layout = go.Layout(
    title = "Top most used ingredients",
    xaxis=dict(
        title='ingredient count',
        tickfont=dict(size=14,)),
    yaxis=dict(
        title='ingredient',
        titlefont=dict(size=16),
        tickfont=dict(
            size=14)),
    margin=dict(
    l=200,
),
    
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp1 = train[train['cuisine'] == 'italian']
n=6714 # total ingredients in train data
top = Counter([item for sublist in temp1.ingredients for item in sublist]).most_common(n)
temp= pd.DataFrame(top)
temp.columns = ['ingredient','total_count']
temp = temp.head(20)
trace = go.Bar(
    y=temp.ingredient[::-1],
    x=temp.total_count[::-1],
    orientation = 'h',
    marker=dict(
        color='#57B8FF',
    ),
)

layout = go.Layout(
    title = "Top most used ingredients in 'italian' cuisine",
    xaxis=dict(
        title='ingredient count',
        tickfont=dict(size=14,)),
    yaxis=dict(
        title='ingredient',
        titlefont=dict(size=16),
        tickfont=dict(
            size=14)),
    margin=dict(
    l=200,
),
    
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp1 = train[train['cuisine'] == 'mexican']
n=6714 # total ingredients in train data
top = Counter([item for sublist in temp1.ingredients for item in sublist]).most_common(n)
temp= pd.DataFrame(top)
temp.columns = ['ingredient','total_count']
temp = temp.head(20)
trace = go.Bar(
    y=temp.ingredient[::-1],
    x=temp.total_count[::-1],
    orientation = 'h',
    marker=dict(
        color='#B66D0D',
    ),
)

layout = go.Layout(
    title = "Top most used ingredients in 'mexican' cuisine",
    xaxis=dict(
        title='ingredient count',
        tickfont=dict(size=14,)),
    yaxis=dict(
        title='ingredient',
        titlefont=dict(size=16),
        tickfont=dict(
            size=14)),
    margin=dict(
    l=200,
),
    
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp1 = train[train['cuisine'] == 'southern_us']
n=6714 # total ingredients in train data
top = Counter([item for sublist in temp1.ingredients for item in sublist]).most_common(n)
temp= pd.DataFrame(top)
temp.columns = ['ingredient','total_count']
temp = temp.head(20)
trace = go.Bar(
    y=temp.ingredient[::-1],
    x=temp.total_count[::-1],
    orientation = 'h',
    marker=dict(
        color='#009FB7',
    ),
)

layout = go.Layout(
    title = "Top most used ingredients in 'southern_us' cuisine",
    xaxis=dict(
        title='ingredient count',
        tickfont=dict(size=14,)),
    yaxis=dict(
        title='ingredient',
        titlefont=dict(size=16),
        tickfont=dict(
            size=14)),
    margin=dict(
    l=200,
),
    
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp1 = train[train['cuisine'] == 'indian']
n=6714 # total ingredients in train data
top = Counter([item for sublist in temp1.ingredients for item in sublist]).most_common(n)
temp= pd.DataFrame(top)
temp.columns = ['ingredient','total_count']
temp = temp.head(20)
trace = go.Bar(
    y=temp.ingredient[::-1],
    x=temp.total_count[::-1],
    orientation = 'h',
    marker=dict(
        color='#FBB13C',
    ),
)

layout = go.Layout(
    title = "Top most used ingredients in 'indian' cuisine",
    xaxis=dict(
        title='ingredient count',
        tickfont=dict(size=14,)),
    yaxis=dict(
        title='ingredient',
        titlefont=dict(size=16),
        tickfont=dict(
            size=14)),
    margin=dict(
    l=200,
),
    
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp1 = train[train['cuisine'] == 'chinese']
n=6714 # total ingredients in train data
top = Counter([item for sublist in temp1.ingredients for item in sublist]).most_common(n)
temp= pd.DataFrame(top)
temp.columns = ['ingredient','total_count']
temp = temp.head(20)
trace = go.Bar(
    y=temp.ingredient[::-1],
    x=temp.total_count[::-1],
    orientation = 'h',
    marker=dict(
        color='red',
    ),
)

layout = go.Layout(
    title = "Top most used ingredients in 'chinese' cuisine",
    xaxis=dict(
        title='ingredient count',
        tickfont=dict(size=14,)),
    yaxis=dict(
        title='ingredient',
        titlefont=dict(size=16),
        tickfont=dict(
            size=14)),
    margin=dict(
    l=200,
),
    
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp1 = train[train['cuisine'] == 'thai']
n=6714 # total ingredients in train data
top = Counter([item for sublist in temp1.ingredients for item in sublist]).most_common(n)
temp= pd.DataFrame(top)
temp.columns = ['ingredient','total_count']
temp = temp.head(20)
trace = go.Bar(
    y=temp.ingredient[::-1],
    x=temp.total_count[::-1],
    orientation = 'h',
    marker=dict(
        color='#4FB5A5',
    ),
)

layout = go.Layout(
    title = "Top most used ingredients in 'thai' cuisine",
    xaxis=dict(
        title='ingredient count',
        tickfont=dict(size=14,)),
    yaxis=dict(
        title='ingredient',
        titlefont=dict(size=16),
        tickfont=dict(
            size=14)),
    margin=dict(
    l=200,
),
    
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
train['number_of_ingredients']= [len(x) for x in train['ingredients']]
test['number_of_ingredients']= [len(x) for x in test['ingredients']]
plt.figure(figsize=(12,5))
plt.title("Distribution of ingredients")
ax = sns.distplot(train["number_of_ingredients"])
tfhash = [("tfidf", TfidfVectorizer(stop_words='english',max_df=.95)),
        ("hashing", HashingVectorizer (stop_words='english',ngram_range=(1,2)))]
X_train = FeatureUnion(tfhash).fit_transform(train.ingredients.str.join(' '))
X_test = FeatureUnion(tfhash).transform(test.ingredients.str.join(' '))
y = train.cuisine
sub['id'] = test.sort_values(by='id' , ascending=True)
sub['cuisine'] = LinearSVC(C = 0.499, dual=False).fit(X_train,y).predict(X_test) 
sub[['id' , 'cuisine' ]].to_csv("svc.csv", index=False)