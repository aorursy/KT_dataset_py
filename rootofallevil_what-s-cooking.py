## Essential libraries for data exploration

import numpy as np
import pandas as pd

from itertools import chain

import warnings
warnings.filterwarnings("ignore")
## Comment out these two lines if you are running this notebook on Kaggle

# train_data = pd.read_json('./data/train.json')
# test_data = pd.read_json('./data/test.json')
## Uncomment these two lines if you are running this notebook on Kaggle

train_data = pd.read_json('../input/train.json') 
test_data = pd.read_json('../input/test.json')
train_data.info()
train_data.head()
test_data.info()
test_data.head()
## Take a look at our target variable

print("Number of cuisine classes: {}".format(len(train_data.cuisine.unique())))
train_data.cuisine.unique()
## Import plotpy library for data visualization

import plotly
import plotly.offline as offline
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
## Helper function for generating a list of n random colors for visualization purposes.
import random

def random_colors(n):
    colors = []
    for i in range(n):
        colors.append("#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
    return colors
trace = go.Table(
    header=dict(
        values=['Cuisine', 'Number of Recipes'],
        fill={'color': '#ddeaff'},
        align = ['left'] * 5
    ),
    cells = dict(
        values=[train_data.cuisine.value_counts().index, train_data.cuisine.value_counts()],
        align=['left'] * 5
    ))

layout = go.Layout(
    title='Number of Recipes in Each Cuisine Class',
    titlefont={'size': 20},
    width=500, height=550,
    paper_bgcolor='#ffffff',
    plot_bgcolor='#ffffff',
    autosize = False,
    margin=dict(l=30,r=30,b=50,t=50,pad=1)
)

iplot({'data': [trace], 'layout': layout})
percentages = []
for i in train_data.cuisine.value_counts():
    percent = (i/sum(train_data.cuisine.value_counts())) * 100
    percent = "%.2f" % percent 
    percent = str(percent) + '%'
    percentages.append(percent)
    
trace = go.Bar(
    x=train_data.cuisine.value_counts().values[::-1],             # x-axis: value_counts, sorted in descending
    y=[i for i in train_data.cuisine.value_counts().index][::-1], # y-axis: cuisine label
    text = percentages[::-1],                                     # bar-plot associated with bars
    textposition = "outside",                                     # place text outside of bars
    orientation = 'h',                                            # horizontal
    marker = {'color': random_colors(20)}                         # colors for bars
)

layout = go.Layout(
    title="Cuisine Class Distribution",                           # title of diagram
    titlefont={'size': 25},                                       # size of title
    width=1000, height=450,                                       # diagram size
    plot_bgcolor='rgba(0,0,0,0)',                                 # color
    paper_bgcolor='rgba(0,0,0,0)',                                # ... and color
    margin=dict(l=75,r=75,b=50,t=50,pad=1)                        # margin
)

iplot({'data': [trace], 'layout': layout})
trace = go.Histogram(
    x=train_data['ingredients'].str.len(),
    xbins={'start': 0, 'end': 90, 'size': 1},
    marker={'color': '#7a5634'},
    opacity=0.75)

layout = go.Layout(
    title="Distribution of Recipe Length",
    titlefont={'size': 25},
    xaxis={'title': 'Number of Ingredients'},
    yaxis={'title': 'Count of Recipes'},
    bargap=0.1, bargroupgap=0.2
)

iplot({'data': [trace], 'layout': layout})
long_recipes = train_data[train_data['ingredients'].str.len() > 30]
print("There are {} recipes consist of more than 30 ingredients.".format(len(long_recipes)))

short_recipes = train_data[train_data['ingredients'].str.len() < 2]
print("There are {} recipes consist of less than 2 ingredients.".format(len(short_recipes)))
colors = random_colors(21)
cuisines = [i for i in train_data.cuisine.value_counts().index][::-1]
data = []

for i in range(20):
    trace = go.Box(
        y=train_data[train_data['cuisine'] == cuisines[i]]['ingredients'].str.len(),
        name=cuisines[i], 
        marker={'color': colors[i]}
    )
    data.append(trace)
    
layout = go.Layout(
    title = "Recipe Length Distribution by Cuisine"
)

iplot({'data': data, 'layout': layout})
from collections import Counter

all_ingredients = []
for item in train_data['ingredients']:
    for ingr in item:
        all_ingredients.append(ingr)
        
counter = Counter()
for ingredient in all_ingredients:
    counter[ingredient] += 1
    
print("Among {} unique ingredients in our training sample," 
      "the most commonly used 20 are: ".format(len(counter)))
counter.most_common(20)
most_common = counter.most_common(20)
most_common_ingredients = [i[0] for i in most_common]
most_common_ingredients_count = [i[1] for i in most_common]

trace = go.Bar(
    x=most_common_ingredients_count[::-1],
    y=most_common_ingredients[::-1],
    orientation='h',
    marker={'color': random_colors(20)}
)

layout = go.Layout(
    xaxis={'title': "Number of Occurrences in All Reciples (Training Sample)"},
    yaxis={'title': "Ingredient"},
    title="The 20 Most Common Ingredients",
    titlefont={'size': 20},
    width=800, height=400,
    margin=dict(l=150,r=10,b=80,t=50,pad=5),
)

iplot({'data': [trace], 'layout': layout})
cuisine_all_ingredients = []
for cuisine in cuisines:
    ingredients = []
    for item in train_data[train_data['cuisine'] == cuisine]['ingredients']:
        for ingr in item:
            ingredients.append(ingr)
    result = (cuisine, len(list(set(ingredients))))
    cuisine_all_ingredients.append(result)
    
trace = go.Bar(
    y=[i[0] for i in cuisine_all_ingredients],
    x=[i[1] for i in cuisine_all_ingredients],
    orientation='h',
    marker={'color': random_colors(20)}
)

layout = go.Layout(
    xaxis={'title': 'Count of different ingredients'},
    yaxis={'title': "Cuisine"},
    title="Number of Unique Ingredientse Used In a Given Cuisine",
    titlefont={'size': 20},
    margin=dict(l=100,r=10,b=60,t=60),
    width=800, height=500
)

iplot({'data': [trace], 'layout': layout})
all_ingredients = list(set(all_ingredients)) # now unique

def top_cuisine_specific_ingredient(cuisine, top_num):
    ingredients_used_by_other_cuisines = []
    for item in train_data[train_data.cuisine != cuisine]['ingredients']:
        for ingr in item:
            ingredients_used_by_other_cuisines.append(ingr)
    ingredients_used_by_other_cuisines = list(set(ingredients_used_by_other_cuisines))
    ingredients_used_only_by_this_cuisine = [x for x in all_ingredients if x not in ingredients_used_by_other_cuisines]
    
    myCounter = Counter()
    for item in train_data[train_data.cuisine == cuisine]['ingredients']:
        for ingr in item:
            myCounter[ingr] += 1
    
    for cuisine in list(myCounter):
        if cuisine not in ingredients_used_only_by_this_cuisine:
            del myCounter[cuisine]
            
    cuisine_specific = pd.DataFrame(myCounter.most_common(top_num), columns=['ingredient', 'count'])
    return cuisine_specific
labels = [i for i in train_data.cuisine.value_counts().index]
numPlots = 20
y = [[i]*2 for i in range(1, 20)]
y = list(chain.from_iterable(y))
z = [1, 2]*int(numPlots/2)

fig = tools.make_subplots(
    rows=10, cols=2, subplot_titles=labels,
    specs=[[{}, {}],[{}, {}],[{}, {}],[{}, {}],[{}, {}],
            [{}, {}],[{}, {}],[{}, {}],[{}, {}],[{}, {}]],  
    horizontal_spacing=0.20,
    print_grid=False)

traces = []
for i, e in enumerate(labels):
    cuisine_specific = top_cuisine_specific_ingredient(e, 5)
    trace = go.Bar(
        x = cuisine_specific['count'].values[::-1],
        y = cuisine_specific['ingredient'].values[::-1],
        orientation = 'h', 
        marker = {'color': random_colors(5)}
    )
    traces.append(trace)
    
for trace, y, z in zip(traces, y, z):
    fig.append_trace(trace, y, z)
    fig['layout'].update(
        height = 1600, width = 840, showlegend = False,
        margin = dict(l=170,r=5,b=40,t=90,pad=5),
        title = "Ingredients Used Only in One Cuisine"
    );
    
iplot(fig)
features =  []# list of list of ingredients
all_ingredients = [] # all ingredients (with duplicate)
for ingredient_list in train_data['ingredients']: # item here is a list of ingredients
    features.append(ingredient_list)
    all_ingredients += ingredient_list
    
test_features = []
for ingredient_list in test_data['ingredients']: # item here is a list of ingredients
    test_features.append(ingredient_list)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    vocabulary = list(set([str(i).lower() for i in all_ingredients])), 
    max_df=0.99, norm='l2', ngram_range=(1, 4)
).fit([str(i) for i in features])

X_tr = tfidf.transform([str(i) for i in features]) # X_tr - matrix of tf-idf scores
to_predict = tfidf.transform([str(i) for i in test_features])
feature_names = tfidf.get_feature_names()
# Define a function for finding the most important features in a given cuisine according to Tf-Idf measure 
def top_feats_by_class(min_tfidf=0.1, top_n=10):
    ''' 
     Input:
         trainsample - the tf-idf transformed training sample;
         target - the target variable;
         featurenames - array mapping from feature integer indices (position in the dataset) to feature name (ingredient in our case) in the Tf-Idf transformed dataset; 
         min_tfidf - features having tf-idf value below the min_tfidf will be excluded ;
         top_n - how many important features to show.
     Output:
          Returns a list of dataframe objects, where each dataframe holds top_n features and their mean tfidf value
         calculated across documents (recipes) with the same class label (cuisine). 
     '''
    dfs = []
    labels = np.unique(target)
    
    for label in labels:
        
        ids = np.where(target==label)
        D = X_tr[ids].toarray()
        D[D < min_tfidf] = 0
        tfidf_means = np.nanmean(D, axis=0)
        
        topn_ids = np.argsort(tfidf_means)[::-1][:top_n] #  Get top n tfidf values
        top_feats = [(feature_names[i], tfidf_means[i]) for i in topn_ids] # find their corresponding feature names
        df = pd.DataFrame(top_feats)
        df.columns = ['feature', 'tfidf']
        
        df['cuisine'] = label
        dfs.append(df)
        
    return dfs
target = train_data['cuisine']
result_tfidf = top_feats_by_class(min_tfidf=0.1, top_n=5)
cuisines = []
for i, e in enumerate(result_tfidf):
    cuisines.append(result_tfidf[i].cuisine[0])

totalPlot = 20
y = [[i] * 2 for i in range(1, 20)]
y = list(chain.from_iterable(y))
z = [1,2] * int((totalPlot/2))

fig = tools.make_subplots(
    rows=10, cols=2, subplot_titles=cuisines, 
    specs=[[{}, {}],[{}, {}],[{}, {}],[{}, {}],[{}, {}],
           [{}, {}],[{}, {}],[{}, {}],[{}, {}],[{}, {}]],  
    horizontal_spacing=0.20, 
    print_grid=False)

traces = []
for index,element in enumerate(result_tfidf): 
    trace = go.Bar(
        x=result_tfidf[index].tfidf[::-1],
        y=result_tfidf[index].feature[::-1],
        orientation='h',
        marker={'color': random_colors(5)}
    )
    traces.append(trace)

for trace, y, z in zip(traces, y, z):
    fig.append_trace(trace, y, z)
    fig['layout'].update(
        height=1600, width=840, showlegend=False,
        margin=dict(l=110,r=5,b=60,t=90,pad=5), 
        title='Feature Importance based on Tf-Idf measure'
    )

iplot(fig)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

import warnings
warnings.filterwarnings("ignore")

encoder = LabelEncoder()
y_transformed = encoder.fit_transform(train_data.cuisine)

X_train, X_test, y_train, y_test = train_test_split(X_tr, y_transformed, random_state=42)
from sklearn.linear_model import LogisticRegression

clf1_cv = LogisticRegression(C=10, verbose=True)
clf1_cv.fit(X_train, y_train)

y_pred = encoder.inverse_transform(clf1_cv.predict(X_train))
y_true = encoder.inverse_transform(y_train)

print("Accuracy score on train data: {}".format(accuracy_score(y_true, y_pred)))
print("Accuracy score on test data: {}".format(clf1_cv.score(X_test, y_test)))
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

_SVC = SVC(C=50, kernel='rbf', gamma=1.4, coef0=1, cache_size=3000, probability=True, verbose=True)
OvRSVC = OneVsRestClassifier(_SVC, n_jobs=-1)
OvRSVC.fit(X_train, y_train)

y_pred = encoder.inverse_transform(OvRSVC.predict(X_train))
y_true = encoder.inverse_transform(y_train)

print("Accuracy score on train data: {}".format(accuracy_score(y_true, y_pred)))
print("Accuracy score on test data: {}".format(OvRSVC.score(X_test, y_test)))
from sklearn.ensemble import VotingClassifier
vclf=VotingClassifier(estimators=[('clf1',clf1_cv),('clf2',OvRSVC)],voting='soft',weights=[1,2])
vclf.fit(X_train , y_train)
vclf.score(X_test, y_test)
predicted_result = vclf.predict(to_predict)
predicted_result_encoded = encoder.inverse_transform(predicted_result)
result_to_submit = pd.DataFrame({'cuisine' : predicted_result_encoded , 'id' : test_data.id })
result_to_submit = result_to_submit[[ 'id' , 'cuisine']]
result_to_submit.to_csv('submit.csv', index = False)