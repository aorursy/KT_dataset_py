import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

TrumpTweets = pd.read_csv("../input/trumptweets/TrumpTweets.csv",encoding ='latin1',nrows = 20000)

TrumpTweets.head()
TrumpTweets['created_at']= pd.to_datetime(TrumpTweets['created_at']) 

TrumpTweets['created_at'] = TrumpTweets['created_at'] - + pd.Timedelta(hours=1)

TrumpTweets['created_at'] = TrumpTweets['created_at'].dt.date
# TrumpTweets['favorite_line'] = TrumpTweets['created_at']-TrumpTweets['created_at'][19999]

# TrumpTweets['favorite_line']=TrumpTweets['favorite_line'].dt.days

# from sklearn.linear_model import LinearRegression

# reg = LinearRegression().fit(TrumpTweets['favorite_line'].values.reshape(-1, 1),TrumpTweets['favorite_count'] )

# print(reg.score(TrumpTweets['favorite_line'].values.reshape(-1, 1),TrumpTweets['favorite_count']))

# TrumpTweets['expected_favorites'] = reg.predict(TrumpTweets['favorite_line'].values.reshape(-1, 1))

# TrumpTweets['above_average'] = TrumpTweets['favorite_count']>=TrumpTweets['expected_favorites']

TrumpTweets['Average_past_20_tweets'] = np.maximum(TrumpTweets['favorite_count'].iloc[::-1].shift().rolling(min_periods=1, window=21).mean().iloc[::-1],1)

TrumpTweets['Difference_over_average'] = (TrumpTweets['favorite_count']-TrumpTweets['Average_past_20_tweets'])/TrumpTweets['Average_past_20_tweets']

TrumpTweets['above_average'] = TrumpTweets['favorite_count']>=TrumpTweets['Average_past_20_tweets']
TrumpTweets['Difference_over_average']=pd.cut(TrumpTweets['Difference_over_average'], bins=[-float('inf'), -0.5, 0.5, float('inf')], labels=['low', 'mid', 'high'])
wordDict = {}

for i in TrumpTweets['text']:

    if i is not None:

        for word in i.split():

            if word.lower() in wordDict:

                wordDict[word.lower()] = wordDict[word.lower()]+1

            else:

                wordDict[word.lower()]=1



import operator, collections

wordDictCounts = sorted(wordDict.items(), key=operator.itemgetter(1),reverse=True)

wordDict = sorted_dict = collections.OrderedDict(wordDictCounts)

wordDict = {k:v for k,v in wordDict.items() if not v == 1}

    

words = list(wordDict.keys())

import numpy as np

AllWords = np.zeros((len(TrumpTweets['text']),len(wordDict)), dtype=int)



   

tweet_index=0    

for tweet in TrumpTweets['text']:

    for word in tweet.split():

        if word.lower() in words:

            AllWords[tweet_index][words.index(word.lower())]=AllWords[tweet_index][words.index(word.lower())]+1

    tweet_index+=1
import gc

WordCountsOfTweets = pd.DataFrame(AllWords, columns = words) 

del AllWords

gc.collect()#Collect garbage to allocate memory
print(WordCountsOfTweets.columns[63])
WordCountsOfTweets=WordCountsOfTweets.drop(columns=WordCountsOfTweets.columns[0:63])
WordCountsOfTweets
from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# X, X_test, y, y_test = train_test_split(WordCountsOfTweets[WordCountsOfTweets.columns[0:20000]], TrumpTweets.above_average, test_size=0.3, random_state=1)

X=WordCountsOfTweets[1000:19999][WordCountsOfTweets.columns]

y=TrumpTweets.Difference_over_average[1000:19999]

X_test = WordCountsOfTweets[:1000][WordCountsOfTweets.columns]

y_test = TrumpTweets.Difference_over_average[:1000]
print(X.shape)

print(y.shape)

print(X_test.shape)

print(y_test.shape)

y_test.value_counts().plot(kind='bar')

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Accuracy by always guessing most occured category:",len(y_test[y_test=='low'])/len(y_test))
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn import tree

import graphviz



# Create Decision Tree classifer object

clf = DecisionTreeClassifier()



# Train Decision Tree Classifer

clf = clf.fit(X,y)



tree.plot_tree(clf, max_depth = 2)

dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True, special_characters=True, max_depth = 2, feature_names=WordCountsOfTweets.columns, class_names=["low", "mid", "high"])  

graph = graphviz.Source(dot_data)  



#Predict the response for test dataset

y_pred = clf.predict(X_test)

graph

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=5, min_samples_split=10,

            min_weight_fraction_leaf=0.0, n_estimators=91, n_jobs=1,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X,y)



y_pred = clf.predict(X_test)



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# print("Accuracy by always guessing above:",len(y_test[y_test])/len(y_test))

#print("Accuracy by always guessing below:",1-len(y_test[y_test])/len(y_test))
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=100, random_state=0)

clf.fit(X,y)



y_pred = clf.predict(X_test)



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='adam', warm_start = True,alpha=1e-5,hidden_layer_sizes=(144,12,3), random_state=1)

clf.fit(X, y)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.naive_bayes import GaussianNB



#Create a Gaussian Classifier

gnb = GaussianNB()



#Train the model using the training sets

gnb.fit(X, y)



#Predict the response for test dataset

y_pred = gnb.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# from sklearn import svm

# clf = svm.SVC()

# clf.fit(X, y)

# y_pred = clf.predict(X_test)

# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# print(len(y_test[y_test])/len(y_test))
from sklearn.decomposition import PCA

pca = PCA(n_components=400)

principalComponents = pca.fit_transform(WordCountsOfTweets[WordCountsOfTweets.columns[0:30000]])

principalDf = pd.DataFrame(data = principalComponents)
from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

X, X_test, y, y_test = train_test_split(principalDf, TrumpTweets.above_average, test_size=0.3, random_state=1)

from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X,y)



y_pred = clf.predict(X_test)



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
import pandas as pd

SP500 = pd.read_csv("../input/sp500/SP500.csv")

SP500['Buy'] = SP500['Open']<SP500['Close']

SP500 = SP500.reindex(index=SP500.index[::-1])

SP500['Date'] = pd.to_datetime(SP500['Date']) 

SP500['Date'] = SP500['Date'].dt.date

SP500
import numpy as np

AllWords = np.zeros((len(TrumpTweets.created_at.unique()),len(wordDict)), dtype=int)



   

tweet_index=0    

for date in TrumpTweets.created_at.unique():

    for tweet in TrumpTweets.text[TrumpTweets['created_at']==date]:

        for word in tweet.split():

            if word in words:

                AllWords[tweet_index][words.index(word)]=AllWords[tweet_index][words.index(word)]+1

    tweet_index+=1
import gc

WordCountsOfTweets = pd.DataFrame(AllWords, columns = words) 

del AllWords

gc.collect()
validDates = TrumpTweets.created_at.unique()[np.isin(TrumpTweets.created_at.unique(),SP500['Date'].values)]

WordCountsOfTweets = WordCountsOfTweets[np.isin(TrumpTweets.created_at.unique(),SP500['Date'].values)]

SP500 = SP500[np.isin(SP500['Date'].values,validDates)]
from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# X, X_test, y, y_test = train_test_split(WordCountsOfTweets[WordCountsOfTweets.columns[0:20000]], TrumpTweets.above_average, test_size=0.3, random_state=1)

X=WordCountsOfTweets[600:1200]

y=SP500.Buy[600:1200]

X_test = WordCountsOfTweets[1200:]

y_test = SP500.Buy[1200:]
from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf= RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=5, min_samples_split=10,

            min_weight_fraction_leaf=0.0, n_estimators=201, n_jobs=1,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)

clf.fit(X,y)



y_pred = clf.predict(X_test)



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(len(y_test[y_test])/len(y_test))
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=100, random_state=0)

clf.fit(X,y)



y_pred = clf.predict(X_test)



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(len(y_test[y_test])/len(y_test))
print("Accuracy:",metrics.accuracy_score(y_test, ~y_pred))

print(len(y_test[y_test])/len(y_test))