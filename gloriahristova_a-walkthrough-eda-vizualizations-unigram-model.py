# Data processing 
import pandas as pd
import json
from collections import Counter
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

# Data vizualizations
import random
import plotly
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
init_notebook_mode(connected=True)
import plotly.offline as offline
import plotly.graph_objs as go

# Data Modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import model_selection 
import warnings
warnings.filterwarnings('ignore')
train_data = pd.read_json('../input/train.json') # store as dataframe objects
test_data = pd.read_json('../input/test.json')
train_data.info()
train_data.shape # 39774 observations, 3 columns
print("The training data consists of {} recipes".format(len(train_data)))
print("First five elements in our training sample:")
train_data.head()
test_data.info()
test_data.shape # 9944 observations, 2 columns
print("The test data consists of {} recipes".format(len(test_data)))
print("First five elements in our test sample:")
test_data.head()
print("Number of cuisine categories: {}".format(len(train_data.cuisine.unique())))
train_data.cuisine.unique()
def random_colours(number_of_colors):
    '''
    Simple function for random colours generation.
    Input:
        number_of_colors - integer value indicating the number of colours which are going to be generated.
    Output:
        Color in the following format: ['#E86DA4'] .
    '''
    colors = []
    for i in range(number_of_colors):
        colors.append("#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
    return colors
trace = go.Table(
                header=dict(values=['Cuisine','Number of recipes'],
                fill = dict(color=['#EABEB0']), 
                align = ['left'] * 5),
                cells=dict(values=[train_data.cuisine.value_counts().index,train_data.cuisine.value_counts()],
               align = ['left'] * 5))

layout = go.Layout(title='Number of recipes in each cuisine category',
                   titlefont = dict(size = 20),
                   width=500, height=650, 
                   paper_bgcolor =  'rgba(0,0,0,0)',
                   plot_bgcolor = 'rgba(0,0,0,0)',
                   autosize = False,
                   margin=dict(l=30,r=30,b=1,t=50,pad=1),
                   )
data = [trace]
fig = dict(data=data, layout=layout)
iplot(fig)
#  Label distribution in percents
labelpercents = []
for i in train_data.cuisine.value_counts():
    percent = (i/sum(train_data.cuisine.value_counts()))*100
    percent = "%.2f" % percent
    percent = str(percent + '%')
    labelpercents.append(percent)
trace = go.Bar(
            x=train_data.cuisine.value_counts().values[::-1],
            y= [i for i in train_data.cuisine.value_counts().index][::-1],
            text =labelpercents[::-1],  textposition = 'outside', 
            orientation = 'h',marker = dict(color = random_colours(20)))
layout = go.Layout(title='Number of recipes in each cuisine category',
                   titlefont = dict(size = 25),
                   width=1000, height=450, 
                   plot_bgcolor = 'rgba(0,0,0,0)',
                   paper_bgcolor = 'rgba(255, 219, 227, 0.88)',
                   margin=dict(l=75,r=110,b=50,t=60),
                   )
data = [trace]
fig = dict(data=data, layout=layout)
iplot(fig, filename='horizontal-bar')
print('Maximum Number of Ingredients in a Dish: ',train_data['ingredients'].str.len().max())
print('Minimum Number of Ingredients in a Dish: ',train_data['ingredients'].str.len().min())
trace = go.Histogram(
    x= train_data['ingredients'].str.len(),
    xbins=dict(start=0,end=90,size=1),
   marker=dict(color='#7CFDF0'),
    opacity=0.75)
data = [trace]
layout = go.Layout(
    title='Distribution of Recipe Length',
    xaxis=dict(title='Number of ingredients'),
    yaxis=dict(title='Count of recipes'),
    bargap=0.1,
    bargroupgap=0.2)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
longrecipes = train_data[train_data['ingredients'].str.len() > 30]
print("It seems that {} recipes consist of more than 30 ingredients!".format(len(longrecipes)))
print("Explore the ingredients in the longest recipe in our training set:" + "\n")
print(str(list(longrecipes[longrecipes['ingredients'].str.len() == 65].ingredients.values)) + "\n")
print("Cuisine: " + str(list(longrecipes[longrecipes['ingredients'].str.len() == 65].cuisine)))
shortrecipes = train_data[train_data['ingredients'].str.len() <= 2]
print("It seems that {} recipes consist of less than or equal to 2 ingredients!".format(len(shortrecipes)))
print("Explore the ingredients in the shortest recipes in our training set:" + "\n")
print(list(train_data[train_data['ingredients'].str.len() == 1].ingredients.values))
print("And there corresponding labels" + "\n")
print(list(train_data[train_data['ingredients'].str.len() == 1].cuisine.values))
boxplotcolors = random_colours(21)
labels = [i for i in train_data.cuisine.value_counts().index][::-1]
data = []
for i in range(20):
    trace = go.Box(
    y=train_data[train_data['cuisine'] == labels[i]]['ingredients'].str.len(), name = labels[i],
    marker = dict(color = boxplotcolors[i]))
    data.append(trace)
layout = go.Layout(
    title = "Recipe Length Distribution by cuisine"
)

fig = go.Figure(data=data,layout=layout)
iplot(fig, filename = "Box Plot Styling Outliers")
allingredients = [] # this list stores all the ingredients in all recipes (with duplicates)
for item in train_data['ingredients']:
    for ingr in item:
        allingredients.append(ingr) 
# Count how many times each ingredient occurs
countingr = Counter()
for ingr in allingredients:
     countingr[ingr] += 1
print("The most commonly used ingredients (with counts) are:")
print("\n")
print(countingr.most_common(20))
print("\n")
print("The number of unique ingredients in our training sample is {}.".format(len(countingr)))
# Extract the first 20 most common ingredients in order to vizualize them for better understanding
mostcommon = countingr.most_common(20)
mostcommoningr = [i[0] for i in mostcommon]
mostcommoningr_count = [i[1] for i in mostcommon]
trace = go.Bar(
            x=mostcommoningr_count[::-1],
            y= mostcommoningr[::-1],
            orientation = 'h',marker = dict(color = random_colours(20),
))
layout = go.Layout(
    xaxis = dict(title= 'Number of occurences in all recipes (training sample)', ),
    yaxis = dict(title='Ingredient',),
    title= '20 Most Common Ingredients', titlefont = dict(size = 20),
    margin=dict(l=150,r=10,b=60,t=60,pad=5),
    width=800, height=500, 
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='horizontal-bar')
# Define a function that returns how many different ingredients can be found in all recipes part of a given cuisine
def findnumingr(cuisine):
    '''
    Input:
        cuisine - cuisine category (ex. greek,souther_us etc.)
    Output:
        The number of unique ingredients used in all recipes part of the given cuisine. 
    '''
    listofinrg = []
    for item in train_data[train_data['cuisine'] == cuisine]['ingredients']:
        for ingr in item:
            listofinrg.append(ingr)
    result = (cuisine,len(list(set(listofinrg))))         
    return result 
cuisineallingr = []
for i in labels:
    cuisineallingr.append(findnumingr(i))
# Vizualize the results
trace = go.Bar(
            x=[i[1] for i in cuisineallingr],
            y= [i[0] for i in cuisineallingr],
            orientation = 'h',marker = dict(color = random_colours(20),
))
layout = go.Layout(
    xaxis = dict(title= 'Count of different ingredients', ),
    yaxis = dict(title='Cuisine',),
    title= 'Number of all the different ingredients used in a given cuisine', titlefont = dict(size = 20),
    margin=dict(l=100,r=10,b=60,t=60),
    width=800, height=500, 
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='horizontal-bar')
allingredients = list(set(allingredients)) # list containing all unique ingredients
# Define a function that returns a dataframe with top unique ingredients in a given cuisine 
def cuisine_unique(cuisine,numingr, allingredients):
    '''
    Input:
        cuisine - cuisine category (ex. 'brazilian');
        numingr - how many specific ingredients do you want to see in the final result; 
        allingredients - list containing all unique ingredients in the whole sample.
    
    Output: 
        dataframe giving information about the name of the specific ingredient and how many times it occurs in the chosen cuisine (in descending order based on their counts)..

    '''
    allother = []
    for item in train_data[train_data.cuisine != cuisine]['ingredients']:
        for ingr in item:
            allother .append(ingr)
    allother  = list(set(allother ))
    
    specificnonly = [x for x in allingredients if x not in allother]
    
    mycounter = Counter()
    
    for item in train_data[train_data.cuisine == cuisine]['ingredients']:
        for ingr in item:
            mycounter[ingr] += 1
    keep = list(specificnonly)
    
    for word in list(mycounter):
        if word not in keep:
            del mycounter[word]
    
    cuisinespec = pd.DataFrame(mycounter.most_common(numingr), columns = ['ingredient','count'])
    
    return cuisinespec
cuisinespec= cuisine_unique('mexican', 10, allingredients)
print("The top 10 unique ingredients in Mexican cuisine are:")
cuisinespec 
# Vizualization of specific ingredients in the first 10 cuisines
labels = [i for i in train_data.cuisine.value_counts().index][0:10]
totalPlot = 10
y = [[item]*2 for item in range(1,10)]
y = list(chain.from_iterable(y))
z = [1,2]*int((totalPlot/2))

fig = tools.make_subplots(rows= 5, cols=2, subplot_titles= labels, specs = [[{}, {}],[{}, {}],[{}, {}],[{}, {}],[{}, {}]],  horizontal_spacing = 0.20)
traces = []
for i,e in enumerate(labels): 
    cuisinespec= cuisine_unique(e, 5, allingredients)
    trace = go.Bar(
            x= cuisinespec['count'].values[::-1],
            y=  cuisinespec['ingredient'].values[::-1],
            orientation = 'h',marker = dict(color = random_colours(5),))
    traces.append(trace)

for t,y,z in zip(traces,y,z):
    fig.append_trace(t, y,z)

    fig['layout'].update(height=800, width=840,
    margin=dict(l=265,r=5,b=40,t=90,pad=5), showlegend=False, title='Ingredients used only in one cuisine')

iplot(fig, filename='horizontal-bar')
# Vizualization of specific ingredients in the second 10 cuisines
labels = [i for i in train_data.cuisine.value_counts().index][10:20]
totalPlot = 10
y = [[item]*2 for item in range(1,10)]
y = list(chain.from_iterable(y))
z = [1,2]*int((totalPlot/2))

fig = tools.make_subplots(rows= 5, cols=2, subplot_titles= labels, specs = [[{}, {}],[{}, {}],[{}, {}],[{}, {}],[{}, {}]],  horizontal_spacing = 0.20)
traces = []
for i,e in enumerate(labels): 
    cuisinespec= cuisine_unique(e, 5, allingredients)
    trace = go.Bar(
            x= cuisinespec['count'].values[::-1],
            y=  cuisinespec['ingredient'].values[::-1],
            orientation = 'h',marker = dict(color = random_colours(5),))
    traces.append(trace)

for t,y,z in zip(traces,y,z):
    fig.append_trace(t, y,z)

    fig['layout'].update(height=800, width=840,
    margin=dict(l=170,r=5,b=40,t=90,pad=5), showlegend=False, title='Ingredient used only in one cuisine')

iplot(fig, filename='horizontal-bar')
# Prepare the data 
features = [] # list of list containg the recipes
for item in train_data['ingredients']:
    features.append(item)
ingredients = [] # this list stores all the ingredients in all recipes (with duplicates)
for item in train_data['ingredients']:
    for ingr in item:
        ingredients.append(ingr) 
len(features) # 39774 recipes
# Fit the TfidfVectorizer to data
tfidf = TfidfVectorizer(vocabulary= list(set([str(i).lower() for i in ingredients])), max_df=0.99, norm='l2', ngram_range=(1, 4))
X_tr = tfidf.fit_transform([str(i) for i in features]) # X_tr - matrix of tf-idf scores
feature_names = tfidf.get_feature_names()
# Define a function for finding the most important features in a given cuisine according to Tf-Idf measure 
def top_feats_by_class(trainsample,target,featurenames, min_tfidf=0.1, top_n=10):
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
        D = trainsample[ids].toarray()
        D[D < min_tfidf] = 0
        tfidf_means = np.nanmean(D, axis=0)
        
        topn_ids = np.argsort(tfidf_means)[::-1][:top_n] #  Get top n tfidf values
        top_feats = [(featurenames[i], tfidf_means[i]) for i in topn_ids] # find their corresponding feature names
        df = pd.DataFrame(top_feats)
        df.columns = ['feature', 'tfidf']
        
        df['cuisine'] = label
        dfs.append(df)
        
    return dfs
# Extract the target variable
target = train_data['cuisine']
result_tfidf = top_feats_by_class(X_tr, target, feature_names, min_tfidf=0.1, top_n=5)
# Exctract labels from the resulting dataframe
labels = []
for i, e in enumerate(result_tfidf):
    labels.append(result_tfidf[i].cuisine[0])

# Set the plot
totalPlot = 10
y = [[item]*2 for item in range(1,10)]
y = list(chain.from_iterable(y))
z = [1,2]*int((totalPlot/2))

fig = tools.make_subplots(rows= 5, cols=2, subplot_titles= labels[0:10], specs = [[{}, {}],[{}, {}],[{}, {}],[{}, {}],[{}, {}]],  horizontal_spacing = 0.20)
traces = []
for index,element in enumerate(result_tfidf[0:10]): 
    trace = go.Bar(
            x= result_tfidf[index].tfidf[::-1],
            y= result_tfidf[index].feature[::-1],
            orientation = 'h',marker = dict(color = random_colours(5),))
    traces.append(trace)

for t,y,z in zip(traces,y,z):
    fig.append_trace(t, y,z)

    fig['layout'].update(height=800, width=840,
    margin=dict(l=110,r=5,b=40,t=90,pad=5), showlegend=False, title='Feature Importance based on Tf-Idf measure')

iplot(fig, filename='horizontal-bar')
# Set the plot
totalPlot = 10
y = [[item]*2 for item in range(1,10)]
y = list(chain.from_iterable(y))
z = [1,2]*int((totalPlot/2))

fig = tools.make_subplots(rows= 5, cols=2, subplot_titles= labels[10:20], specs = [[{}, {}],[{}, {}],[{}, {}],[{}, {}],[{}, {}]],  horizontal_spacing = 0.20)
traces = []
for index,element in enumerate(result_tfidf[10:20]): 
    trace = go.Bar(
            x= result_tfidf[10:20][index].tfidf[::-1],
            y= result_tfidf[10:20][index].feature[::-1],
            orientation = 'h',marker = dict(color = random_colours(5),))
    traces.append(trace)

for t,y,z in zip(traces,y,z):
    fig.append_trace(t, y,z)

    fig['layout'].update(height=800, width=840,
    margin=dict(l=100,r=5,b=40,t=90,pad=5), showlegend=False, title='Feature Importance based on Tf-Idf measure')

iplot(fig, filename='horizontal-bar')
# Train sample 
print("How training data looks like at this stage (example of one recipe):")
print(str(features[0]) + '\n' )
print("Number of instances: "+ str(len(features)) + '\n')
print("And the target variable:")
print(target[0])
# Test Sample - only features - the target variable is not provided.
features_test = [] # list of lists containg the recipes
for item in test_data['ingredients']:
    features_test.append(item)
print("How test data looks like at this stage (example of one recipe):")
print(str(features_test[0]) + '\n')
print("Number of instances: "+ str(len(features_test)))
# Both train and test samples are processed in the exact same way
# Train
features_processed= [] # here we will store the preprocessed training features
for item in features:
    newitem = []
    for ingr in item:
        ingr.lower() # Case Normalization - convert all to lower case 
        ingr = re.sub("[^a-zA-Z]"," ",ingr) # Remove punctuation, digits or special characters 
        ingr = re.sub((r'\b(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\b'), ' ', ingr) # Remove different units  
        newitem.append(ingr)
    features_processed.append(newitem)

# Test 
features_test_processed= [] 
for item in features_test:
    newitem = []
    for ingr in item:
        ingr.lower() 
        ingr = re.sub("[^a-zA-Z]"," ",ingr)
        ingr = re.sub((r'\b(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\b'), ' ', ingr) 
        newitem.append(ingr)
    features_test_processed.append(newitem) 
# Check for empty instances in train and test samples after processing before proceeding to next stage of the analysis    
count_m = []    
for recipe in features_processed:
    if not recipe:
        count_m.append([recipe])
    else: pass
print("Empty instances in the preprocessed training sample: " + str(len(count_m)))  
count_m = []    
for recipe in features_test_processed:
    if not recipe:
        count_m.append([recipe])
    else: pass
print("Empty instances in the preprocessed test sample: " + str(len(count_m)))    
# Binary representation of the training set will be employed
vectorizer = CountVectorizer(analyzer = "word",
                             ngram_range = (1,1), # unigrams
                             binary = True, #  (the default is counts)
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,  
                             max_df = 0.99) # any word appearing in more than 99% of the sample will be discarded
# Fit the vectorizer on the training data and transform the test sample
train_X = vectorizer.fit_transform([str(i) for i in features_processed])
test_X =  vectorizer.transform([str(i) for i in features_test_processed])
# Apply label encoding on the target variable (before model development)
lb = LabelEncoder()
train_Y = lb.fit_transform(target)
# Ensemble Unigram model (baseline model) - parameters are not tuned at this stage
vclf=VotingClassifier(estimators=[('clf1',LogisticRegression(random_state = 42)),
                                  ('clf2',SVC(kernel='linear',random_state = 42,probability=True)),
                                  ('clf3',RandomForestClassifier(n_estimators = 600,random_state = 42))], 
                                    voting='soft', weights = [1,1,1]) 
vclf.fit(train_X, train_Y)
# 10-fold Cross validation of  the results
kfold = model_selection.KFold(n_splits=10, random_state=42)
valscores = model_selection.cross_val_score(vclf, train_X, train_Y, cv=kfold)
print('Mean accuracy on 10-fold cross validation: ' + str(np.mean(valscores))) #  0.8005731359034913
# Generate predictions on test sample
predictions = vclf.predict(test_X) 
predictions = lb.inverse_transform(predictions)
predictions_final = pd.DataFrame({'cuisine' : predictions , 'id' : test_data.id }, columns=['id', 'cuisine'])
predictions_final.to_csv('Final_submission.csv', index = False)