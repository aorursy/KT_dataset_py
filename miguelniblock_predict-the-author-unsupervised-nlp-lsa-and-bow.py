# General-purpose Libraries

import numpy as np

import pandas as pd

import scipy

import sklearn

import spacy

import matplotlib.pyplot as plt

import seaborn as sns

import re

from collections import Counter

import spacy

from time import time

%matplotlib inline



# Tools for processing data

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Normalizer

from sklearn.decomposition import TruncatedSVD, PCA

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix, make_scorer, adjusted_rand_score, silhouette_score, homogeneity_score, normalized_mutual_info_score

# Classifiers, supervised and unsupervised

from sklearn import ensemble

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

from sklearn.cluster import MeanShift, estimate_bandwidth

from sklearn.cluster import SpectralClustering

from sklearn.cluster import AffinityPropagation



import warnings

warnings.filterwarnings("ignore")
# Read data into a DataFrame

data = pd.read_csv('../input/articles1.csv')
# Preview the data

data.head(3)
data.info()
lengths = pd.Series([len(x) for x in data.content])

print('Statistical Summary of Article Lengths')

print(lengths.describe())



sns.distplot(lengths,kde=False)

plt.title('Distribution of Article Lengths (All)')

plt.show()

sns.distplot(lengths[lengths<10000],kde=False)

plt.title('Distribution of Articles Lengths < 10,000 Characters')

plt.show()
# First ten authors with more than X articles

print(data.author.value_counts()[data.author.value_counts()>100][-10:])
# Make a DataFrame with articles by our chosen authors

# Include author names and article titles.



# Make a list of the 10 chosen author names

names = data.author.value_counts()[data.author.value_counts()>100][-10:].index.tolist()



# DataFrame for articles of all chosen authors

authors_data = pd.DataFrame()

for name in names:

    # Select each author's data

    articles = data[data.author==name][:100][['title','content','author']]

    # Append it to the DataFrame

    authors_data = authors_data.append(articles)



authors_data = authors_data.reset_index().drop('index',1)

    

authors_data.head()
# Look for duplicates

print('Number of articles:',authors_data.shape[0])

print('Unique articles:',len(np.unique(authors_data.index)))



# Number of authors

print('Unique authors:',len(np.unique(authors_data.author)))

print('')

print('Articles by author:\n')



# Articles counts by author

print(authors_data.author.value_counts())
lengths = pd.Series([len(x) for x in authors_data.content])

print('Statistical Summary of Article Lengths')

print(lengths.describe())



sns.distplot(lengths,kde=False)

plt.title('Distribution of Article Lengths (All)')

plt.show()

sns.distplot(lengths[lengths<10000],kde=False)

plt.title('Distribution of Articles Lengths < 10,000 Characters')

plt.show()
t0 = time()



# Load spacy NLP object

nlp = spacy.load('en')



# A list to store common words by all authors

common_words = []



# A dictionary to store the spacy_doc object of each author

authors_docs = {}



for name in names:

    # Corpus is all the text written by that author

    corpus = ""

    # Grab all rows of current author, along the 'content' column

    author_content = authors_data.loc[authors_data.author==name,'content']

    

    # Merge all articles in to the author's corpus

    for article in author_content:

        corpus = corpus + article

    # Let Spacy parse the author's body of text

    doc = nlp(corpus)

    

    # Store the doc in the dictionary

    authors_docs[name] = doc

        

    # Filter out punctuation and stop words.

    lemmas = [token.lemma_ for token in doc

                if not token.is_punct and not token.is_stop]

        

    # Return the most common words of that author's corpus.

    bow = [item[0] for item in Counter(lemmas).most_common(1000)]

    

    # Add them to the list of words by all authors.

    for word in bow:

        common_words.append(word)



# Eliminate duplicates

common_words = set(common_words)

    

print('Total number of common words:',len(common_words))

print("done in %0.3fs" % (time() - t0))
# Let's see our 10 authors in the dictionary

lengths = []

for k,v in authors_docs.items():

    print(k,'corpus contains',len(v),' words.')

    lengths.append(len(v))
sns.barplot(x=lengths,y=names,orient='h')

plt.title('Word Count per Author in Chosen Data')

plt.show()
# check for lower case words

common_words = pd.Series(pd.DataFrame(columns=common_words).columns)

print('Count of all common_words:',len(common_words))

print('Count of lowercase common_words:',np.sum([word.islower() for word in common_words]))



# Turn all common_words into lower case

common_words = [word.lower() for word in common_words]

print('Count of lowercase common_words (After Conversion):',np.sum([word.islower() for word in common_words]))
# We must remove these in to avoid conflicts with existing features.

if 'author' in common_words:

    common_words.remove('author')

if 'title' in common_words:

    common_words.remove('title')

if 'content' in common_words:

    common_words.remove('content')
# Count the number of times a common_word appears in each article

# (about 3Hrs processing)



bow_counts = pd.DataFrame()

for name in names:

    # Select X articles of that author

    articles = authors_data.loc[authors_data.author==name,:][:50]

    bow_counts = bow_counts.append(articles)

bow_counts = bow_counts.reset_index().drop('index',1)



# Use common_words as the columns of a temporary DataFrame

df = pd.DataFrame(columns=common_words)



# Join BOW features with the author's content

bow_counts = bow_counts.join(df)



# Initialize rows with zeroes

bow_counts.loc[:,common_words] = 0



# Fill the DataFrame with counts of each feature in each article

t0 = time()

for i, article in enumerate(bow_counts.content):

    doc = nlp(article)

    for token in doc:

        if token.lemma_.lower() in common_words:

            bow_counts.loc[i,token.lemma_.lower()] += 1

    # Print a message every X articles

    if i % 50 == 0:

        if time()-t0 < 3600: # if less than an hour in seconds

            print("Article ",i," done after ",(time()-t0)/60,' minutes.')

        else:

            print("Article ",i," done after ",(time()-t0)/60/60,' hours.')
bow_counts.head(3)
# This saves the long-awaited data into a pickle file for easy recovery

#bow_counts.to_pickle('bow_counts')



# Read it back in with the following

#bow_counts = pd.read_pickle('bow_counts')
# Make sure we have 50 articles per author

bow_counts.author.value_counts()
# Establish outcome and predictors

y = bow_counts['author']

X = bow_counts.drop(['content','author','title'], 1)



X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                    test_size=0.24,

                                                    random_state=0,

                                                    stratify=y)
# Make sure classes are balanced after train-test-split

y_test.value_counts()
# Store our results in a DataFrame

metrics = ['Algorithm','n_train','Features','ARI','Homogeneity',

           'Silhouette','Mutual_Info','Cross_Val','Train_Accuracy',

           'Test_Accuracy']

performance = pd.DataFrame(columns=metrics)
# Function to quickly evaluate clustering solutions

def evaluate_clust(clust,params,features,i):

    t0 = time()

    print('\n','-'*40,'\n',clust.__class__.__name__,'\n','-'*40)

    

    # Find best parameters based on scoring of choice

    score = make_scorer(normalized_mutual_info_score)

    search = GridSearchCV(clust,params,scoring=score,cv=3).fit(X,y)

    print("Best parameters:",search.best_params_)

    y_pred = search.best_estimator_.fit_predict(X)



    ari = adjusted_rand_score(y, y_pred)

    performance.loc[i,'ARI'] = ari 

    print("Adjusted Rand-Index: %.3f" % ari)

    

    hom = homogeneity_score(y,y_pred)

    performance.loc[i,'Homogeneity'] = hom

    print("Homogeneity Score: %.3f" % hom)

    

    sil = silhouette_score(X,y_pred)

    performance.loc[i,'Silhouette'] = sil

    print("Silhouette Score: %.3f" % sil)

    

    nmi = normalized_mutual_info_score(y,y_pred)

    performance.loc[i,'Mutual_Info'] = nmi

    print("Normed Mutual-Info Score: %.3f" % nmi)

    

    performance.loc[i,'n_train'] = len(X)

    performance.loc[i,'Features'] = features

    performance.loc[i,'Algorithm'] = clust.__class__.__name__

    

    # Print contingency matrix

    crosstab = pd.crosstab(y, y_pred)

    plt.figure(figsize=(8,5))

    sns.heatmap(crosstab, annot=True,fmt='d', cmap=plt.cm.copper)

    plt.show()

    print(time()-t0,"seconds.")
clust=KMeans()

params={

    'n_clusters': np.arange(10,30,5),

    'init': ['k-means++','random'],

    'n_init':[10,20],

    'precompute_distances':[True,False]

}

evaluate_clust(clust,params,features='BOW',i=0)
#Declare and fit the model

clust = MeanShift()



params={}

evaluate_clust(clust,params,features='BOW',i=1)
#Declare and fit the model.

clust = AffinityPropagation()



params = {

    'damping':[.5,.7,.9],

    'max_iter':[200,500]

}

evaluate_clust(clust,params,features='BOW',i=2)
clust= SpectralClustering()



params = {

    'n_clusters':np.arange(10,26,5),

    #'eigen_solver':['arpack','lobpcg',None],

    'n_init':[15,25],

    'assign_labels':['kmeans','discretize']

}



features='BOW'



i=3



t0=time()



y_pred = clust.fit_predict(X)



ari = adjusted_rand_score(y, y_pred)

performance.loc[i,'ARI'] = ari 

print("Adjusted Rand-Index: %.3f" % ari)



hom = homogeneity_score(y,y_pred)

performance.loc[i,'Homogeneity'] = hom

print("Homogeneity Score: %.3f" % hom)



sil = silhouette_score(X,y_pred)

performance.loc[i,'Silhouette'] = sil

print("Silhouette Score: %.3f" % sil)



nmi = normalized_mutual_info_score(y,y_pred)

performance.loc[i,'Mutual_Info'] = nmi

print("Normed Mutual-Info Score: %.3f" % nmi)



performance.loc[i,'n_train'] = len(X)

performance.loc[i,'Features'] = features

performance.loc[i,'Algorithm'] = clust.__class__.__name__



# Print contingency matrix

crosstab = pd.crosstab(y, y_pred)

plt.figure(figsize=(8,5))

sns.heatmap(crosstab, annot=True,fmt='d', cmap=plt.cm.copper)

plt.show()

print(time()-t0,"seconds.")
performance.iloc[:,:7]
def score_optimization(clf,params,features,i):

    t0 = time()

    # Heading

    print('\n','-'*40,'\n',clf.__class__.__name__,'\n','-'*40)

    

    # Find best parameters based on scoring of choice

    score = make_scorer(normalized_mutual_info_score)

    search = GridSearchCV(clf,params,

                          scoring=score,cv=3).fit(X,y)

    # Extract best estimator

    best = search.best_estimator_

    print("Best parameters:",search.best_params_)



    # Cross-validate on all the data

    cv = cross_val_score(X=X,y=y,estimator=best,cv=5)

    print("\nCross-val scores(All Data):",cv)

    print("Mean cv score:",cv.mean())

    performance.loc[i,'Cross_Val'] = cv.mean() 

    

    # Get train accuracy

    best = best.fit(X_train,y_train)

    train = best.score(X=X_train,y=y_train)

    performance.loc[i,'Train_Accuracy'] = train 

    print("\nTrain Accuracy Score:",train)



    # Get test accuracy

    test = best.score(X=X_test,y=y_test)

    performance.loc[i,'Test_Accuracy'] = test 

    print("\nTest Accuracy Score:",test)

    

    y_pred = best.predict(X_test)

    

    ari = adjusted_rand_score(y_test, y_pred)

    performance.loc[i,'ARI'] = ari 

    print("\nAdjusted Rand-Index: %.3f" % ari)

    

    hom = homogeneity_score(y_test,y_pred)

    performance.loc[i,'Homogeneity'] = hom

    print("Homogeneity Score: %.3f" % hom)

    

    sil = silhouette_score(X_test,y_pred)

    performance.loc[i,'Silhouette'] = sil

    print("Silhouette Score: %.3f" % sil)

    

    nmi = normalized_mutual_info_score(y_test,y_pred)

    performance.loc[i,'Mutual_Info'] = nmi

    print("Normed Mutual-Info Score: %.3f" % nmi)



    #print(classification_report(y_test, y_pred))



    conf_matrix = pd.crosstab(y_test,y_pred)

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.copper)

    plt.show()

    

    performance.loc[i,'n_train'] = len(X_train)

    performance.loc[i,'Features'] = features

    performance.loc[i,'Algorithm'] = clf.__class__.__name__

    print(time()-t0,'seconds.')
# Parameters to optimize

params = [{

    'solver': ['newton-cg', 'lbfgs', 'sag'],

    'C': [0.3, 0.5, 0.7, 1],

    'penalty': ['l2']

    },{

    'solver': ['liblinear','saga'],

    'C': [0.3, 0.5, 0.7, 1],

    'penalty': ['l1','l2']

}]



clf = LogisticRegression(

    n_jobs=-1 # Use all CPU

)



score_optimization(clf=clf,params=params,features='BOW',i=4)
# Parameters to compare

params = {

    'criterion':['entropy','gini'],

}



# Implement the classifier

clf = ensemble.RandomForestClassifier(

    n_estimators=100,

    max_features=None,

    n_jobs=-1,

)



score_optimization(clf=clf,params=params,features='BOW',i=5)
# Parameters to compare

params = {

    'learning_rate':[0.3,0.5,0.7,1]

}



# Implement the classifier

clf = ensemble.GradientBoostingClassifier(

    max_features=None

)



score_optimization(clf=clf,params=params,features='BOW',i=6)
performance.iloc[:7].sort_values('Mutual_Info',ascending=False)[['Algorithm','n_train','Features','Mutual_Info','Test_Accuracy']]
vectorizer = TfidfVectorizer(max_df=0.3, # drop words that occur in more than X percent of documents

                             min_df=8, # only use words that appear at least X times

                             stop_words='english', 

                             lowercase=True, #convert everything to lower case 

                             use_idf=True,#we definitely want to use inverse document frequencies in our weighting

                             norm=u'l2', #Applies a correction factor so that longer paragraphs and shorter paragraphs get treated equally

                             smooth_idf=True #Adds 1 to all document frequencies, as if an extra document existed that used every word once.  Prevents divide-by-zero errors

                            )



#Pass pandas series to our vectorizer model

counts_tfidf = vectorizer.fit_transform(bow_counts.content)



counts_tfidf
svd = TruncatedSVD(460)

svd.fit(counts_tfidf)

svd.explained_variance_ratio_.sum()
lsa = make_pipeline(svd, Normalizer(copy=False))

lsa_data = lsa.fit_transform(counts_tfidf)

lsa_data.shape
lsa_data = pd.DataFrame(lsa_data)

lsa_data.head()
#First, establish X and Y

y = bow_counts['author']

X = lsa_data



X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    y,

                                                    test_size=0.24,

                                                    random_state=0,

                                                   stratify=y)
y_test.value_counts()
clust=KMeans()

params={

    'n_clusters': np.arange(10,30,5),

    'init': ['k-means++','random'],

    'n_init':[10,20],

    'precompute_distances':[True,False]

}

evaluate_clust(clust,params,features='LSA',i=7)
#Declare and fit the model

clust = MeanShift()



params={

    'bandwidth':[0.5,0.7,0.9]

}

evaluate_clust(clust,params,features='LSA',i=8)
#Declare and fit the model.

clust = AffinityPropagation()



params = {

    'damping':[.5,.7,.9],

    'max_iter':[200,500]

}

evaluate_clust(clust,params,features='LSA',i=9)
clust= SpectralClustering()



params = {

    'n_clusters':np.arange(10,26,5),

    #'eigen_solver':['arpack','lobpcg',None],

    'n_init':[15,25],

    'assign_labels':['kmeans','discretize']

}



features='LSA'



i=10



t0=time()



y_pred = clust.fit_predict(X)



ari = adjusted_rand_score(y, y_pred)

performance.loc[i,'ARI'] = ari 

print("Adjusted Rand-Index: %.3f" % ari)



hom = homogeneity_score(y,y_pred)

performance.loc[i,'Homogeneity'] = hom

print("Homogeneity Score: %.3f" % hom)



sil = silhouette_score(X,y_pred)

performance.loc[i,'Silhouette'] = sil

print("Silhouette Score: %.3f" % sil)



nmi = normalized_mutual_info_score(y,y_pred)

performance.loc[i,'Mutual_Info'] = nmi

print("Normed Mutual-Info Score: %.3f" % nmi)



performance.loc[i,'n_train'] = len(X)

performance.loc[i,'Features'] = features

performance.loc[i,'Algorithm'] = clust.__class__.__name__



# Print contingency matrix

crosstab = pd.crosstab(y, y_pred)

plt.figure(figsize=(8,5))

sns.heatmap(crosstab, annot=True,fmt='d', cmap=plt.cm.copper)

plt.show()

print(time()-t0,"seconds.")
performance.iloc[:11].sort_values('Mutual_Info',ascending=False)[['Algorithm','n_train','Features','Mutual_Info','Test_Accuracy']]
# Parameters to optimize

params = [{

    'solver': ['newton-cg', 'lbfgs', 'sag'],

    'C': [0.3, 0.5, 0.7, 1],

    'penalty': ['l2']

    },{

    'solver': ['liblinear','saga'],

    'C': [0.3, 0.5, 0.7, 1],

    'penalty': ['l1','l2']

}]



clf = LogisticRegression(

    n_jobs=-1 # Use all CPU

)



score_optimization(clf=clf,params=params,features='LSA',i=11)
# Parameters to compare

params = {

    'criterion':['entropy','gini'],

}



# Implement the classifier

clf = ensemble.RandomForestClassifier(

    n_estimators=100,

    max_features=None,

    n_jobs=-1,

)



score_optimization(clf=clf,params=params,features='LSA',i=12)
# Parameters to compare

params = {

    'learning_rate':[0.3,0.5,0.7,1]

}



# Implement the classifier

clf = ensemble.GradientBoostingClassifier(

    max_features=None

)



score_optimization(clf=clf,params=params,features='LSA',i=13)
performance.iloc[:14].sort_values('Mutual_Info',ascending=False)[['Algorithm','n_train','Features','Mutual_Info','Test_Accuracy']].iloc[:9]
vectorizer = TfidfVectorizer(max_df=0.3, # drop words that occur in more than X percent of documents

                             min_df=8, # only use words that appear at least X times

                             stop_words='english', 

                             lowercase=True, #convert everything to lower case 

                             use_idf=True,#we definitely want to use inverse document frequencies in our weighting

                             norm=u'l2', #Applies a correction factor so that longer paragraphs and shorter paragraphs get treated equally

                             smooth_idf=True #Adds 1 to all document frequencies, as if an extra document existed that used every word once.  Prevents divide-by-zero errors

                            )



#Pass pandas series to our vectorizer model

counts_tfidf = vectorizer.fit_transform(authors_data.content)



counts_tfidf
svd = TruncatedSVD(900)

svd.fit(counts_tfidf)

svd.explained_variance_ratio_.sum()
lsa = make_pipeline(svd, Normalizer(copy=False))

lsa_data = lsa.fit_transform(counts_tfidf)

lsa_data.shape
lsa_data = pd.DataFrame(lsa_data)

lsa_data.head()
#First, establish X and Y

y = authors_data['author']

X = lsa_data



X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    y,

                                                    test_size=0.24,

                                                    random_state=0,

                                                   stratify=y)
y_test.value_counts()
clust=KMeans()

params={

    'n_clusters': np.arange(10,30,5),

    'init': ['k-means++','random'],

    'n_init':[10,20],

    'precompute_distances':[True,False]

}

evaluate_clust(clust,params,features='LSA',i=14)
#Declare and fit the model

clust = MeanShift()



params={

    'bandwidth':[0.5,0.7,0.9]

}

evaluate_clust(clust,params,features='LSA',i=15)
#Declare and fit the model.

clust = AffinityPropagation()



params = {

    'damping':[.5,.7,.9],

    'max_iter':[200,500]

}

evaluate_clust(clust,params,features='LSA',i=16)
clust= SpectralClustering()



params = {

    'n_clusters':np.arange(10,26,5),

    #'eigen_solver':['arpack','lobpcg',None],

    'n_init':[15,25],

    'assign_labels':['kmeans','discretize']

}



features='LSA'



i=17



t0=time()



y_pred = clust.fit_predict(X)



ari = adjusted_rand_score(y, y_pred)

performance.loc[i,'ARI'] = ari 

print("Adjusted Rand-Index: %.3f" % ari)



hom = homogeneity_score(y,y_pred)

performance.loc[i,'Homogeneity'] = hom

print("Homogeneity Score: %.3f" % hom)



sil = silhouette_score(X,y_pred)

performance.loc[i,'Silhouette'] = sil

print("Silhouette Score: %.3f" % sil)



nmi = normalized_mutual_info_score(y,y_pred)

performance.loc[i,'Mutual_Info'] = nmi

print("Normed Mutual-Info Score: %.3f" % nmi)



performance.loc[i,'n_train'] = len(X)

performance.loc[i,'Features'] = features

performance.loc[i,'Algorithm'] = clust.__class__.__name__



# Print contingency matrix

crosstab = pd.crosstab(y, y_pred)

plt.figure(figsize=(8,5))

sns.heatmap(crosstab, annot=True,fmt='d', cmap=plt.cm.copper)

plt.show()

print(time()-t0,"seconds.")
# Parameters to optimize

params = [{

    'solver': ['newton-cg', 'lbfgs', 'sag'],

    'C': [0.3, 0.5, 0.7, 1],

    'penalty': ['l2']

    },{

    'solver': ['liblinear','saga'],

    'C': [0.3, 0.5, 0.7, 1],

    'penalty': ['l1','l2']

}]



clf = LogisticRegression(

    n_jobs=-1 # Use all CPU

)



score_optimization(clf=clf,params=params,features='LSA',i=18)
# Parameters to compare

params = {

    'criterion':['entropy','gini'],

}



# Implement the classifier

clf = ensemble.RandomForestClassifier(

    n_estimators=100,

    max_features=None,

    n_jobs=-1,

)



score_optimization(clf=clf,params=params,features='LSA',i=19)
# Parameters to compare

params = {

    'learning_rate':[0.3,0.5,0.7,1]

}



# Implement the classifier

clf = ensemble.GradientBoostingClassifier(

    max_features=None

)



score_optimization(clf=clf,params=params,features='LSA',i=20)
performance.sort_values('Mutual_Info',ascending=False)[['Algorithm','n_train','Features','Mutual_Info','Test_Accuracy']].iloc[:10]
performance.sort_values('Mutual_Info',ascending=False)[['Mutual_Info','ARI','Homogeneity','Cross_Val','Train_Accuracy','Test_Accuracy']].iloc[:10]
performance.sort_values('Test_Accuracy',ascending=False)[['Algorithm','n_train','Features','Mutual_Info','Test_Accuracy']].iloc[:10]
plot_data = performance.sort_values('Test_Accuracy',ascending=False)[['Algorithm','Features','n_train','Test_Accuracy']].iloc[:9]



plot_data.n_train = plot_data.n_train.apply(lambda x: str(x))

plot_data['method'] = plot_data.Features+'_'+plot_data.n_train

%matplotlib inline



sns.catplot(col='Algorithm',x='method',y='Test_Accuracy',

            data=plot_data,kind='bar')

plt.show()