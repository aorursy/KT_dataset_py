# ALL imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
# Create dataframes train and test
train = pd.read_csv('../input/drugsComTrain_raw.csv')
test = pd.read_csv('../input/drugsComTest_raw.csv')
train.head()
test.head()
list(train) == list(test)
list(train)
train.values.shape[0], test.values.shape[0], train.values.shape[0] / test.values.shape[0]
train.condition.unique().size, test.condition.unique().size
train.drugName.unique().size, test.drugName.unique().size
# I previously did this by creating and sorting a dictionary -- here's an easier way with pandas! (Inspiration from Sayan Goswami)
conditions = train.condition.value_counts().sort_values(ascending=False)
conditions[:10]
plt.rcParams['figure.figsize'] = [12, 8]
conditions[:10].plot(kind='bar')
plt.title('Top 10 Most Common Conditions')
plt.xlabel('Condition')
plt.ylabel('Count');
# Look at bias in review (also shown on 'Data' page in competition: distribution of ratings)
train.rating.hist(color='skyblue')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.xticks([i for i in range(1, 11)]);
rating_avgs = (train['rating'].groupby(train['drugName']).mean())
rating_avgs.hist(color='skyblue')
plt.title('Distribution of average drug ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
rating_avgs = (train['rating'].groupby(train['condition']).mean())
rating_avgs.hist(color='skyblue')
plt.title('Averages of medication reviews for each disease')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
# Is rating correlated with usefulness of the review?
plt.scatter(train.rating, train.usefulCount, c=train.rating.values, cmap='tab10')
plt.title('Useful Count vs Rating')
plt.xlabel('Rating')
plt.ylabel('Useful Count')
plt.xticks([i for i in range(1, 11)]);
# Create a list (cast into an array) containing the average usefulness for given ratings
use_ls = []

for i in range(1, 11):
    use_ls.append([i, np.sum(train[train.rating == i].usefulCount) / np.sum([train.rating == i])])
    
use_arr = np.asarray(use_ls)
plt.scatter(use_arr[:, 0], use_arr[:, 1], c=use_arr[:, 0], cmap='tab10', s=200)
plt.title('Average Useful Count vs Rating')
plt.xlabel('Rating')
plt.ylabel('Average Useful Count')
plt.xticks([i for i in range(1, 11)]);
# Sort train dataframe from most to least useful
useful_train = train.sort_values(by='usefulCount', ascending=False)
useful_train.iloc[:10]
# Print top 10 most useful reviews
for i in useful_train.review.iloc[:3]:
    print(i, '\n')
# Print 10 of the least useful reviews
for i in useful_train.review.iloc[-3:]:
    print(i, '\n')
sid = SentimentIntensityAnalyzer()
# Create list (cast to array) of compound polarity sentiment scores for reviews
sentiments = []

for i in train.review:
    sentiments.append(sid.polarity_scores(i).get('compound'))
    
sentiments = np.asarray(sentiments)
sentiments
useful_train['sentiment'] = pd.Series(data=sentiments)
useful_train = useful_train.reset_index(drop=True)
useful_train.head()
useful_train.sentiment.hist(color='skyblue', bins=30)
plt.title('Compound Sentiment Score Distribution')
plt.xlabel('Scores')
plt.ylabel('Count');
useful_train.plot(x='sentiment', y='usefulCount', kind='scatter', alpha=0.01)
plt.title('Usefulness vs Sentiment')
plt.ylim(0, 200);
temp_ls = []

for i in range(1, 11):
    temp_ls.append(np.sum(useful_train[useful_train.rating == i].sentiment) / np.sum(useful_train.rating == i))
plt.scatter(x=range(1, 11), y=temp_ls, c=range(1, 11), cmap='tab10', s=200)
plt.title('Average Sentiment vs Rating')
plt.xlabel('Rating')
plt.ylabel('Average Sentiment')
plt.xticks([i for i in range(1, 11)]);
# Create a list of all drugs and their average ratings, cast to dataframe
rate_ls = []

for i in train.drugName.unique():
    
    # Only consider drugs that have at least 10 ratings
    if np.sum(train.drugName == i) >= 10:
        rate_ls.append((i, np.sum(train[train.drugName == i].rating) / np.sum(train.drugName == i)))
    
avg_rate = pd.DataFrame(rate_ls)
# Sort drugs by their ratings, look at top 10 best and worst rated drugs
avg_rate = avg_rate.sort_values(by=[1], ascending=False).reset_index(drop=True)
avg_rate[:10]
avg_rate[-10:]
# Make dictionary of conditions, each value will be a dataframe of all of the drugs used to treat the given condition
help_dict = {}

# Iterate over conditions
for i in train.condition.unique():
    
    temp_ls = []
    
    # Iterate over drugs within a given condition
    for j in train[train.condition == i].drugName.unique():
        
        # If there are at least 10 reviews for a drug, save its name and average rating in temporary list
        if np.sum(train.drugName == j) >= 10:
            temp_ls.append((j, np.sum(train[train.drugName == j].rating) / np.sum(train.drugName == j)))
        
    # Save temporary list as a dataframe as a value in help dictionary, sorted best to worst drugs
    help_dict[i] = pd.DataFrame(data=temp_ls, columns=['drug', 'average_rating']).sort_values(by='average_rating', ascending=False).reset_index(drop=True)
help_dict['Birth Control'].iloc[:10]
help_dict['Depression'].iloc[:10]
help_dict['Acne'].iloc[:10]
help_dict['Acne'].iloc[-10:]
# Creates TF-IDF vectorizer and transforms the corpus
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train.review)

# transforms test reviews to above vectorized format
X_test = vectorizer.transform(test.review)
# Create a column with binary rating indicating the polarity of a review
train['binary_rating'] = train['rating'] > 5

y_train_rating = train.binary_rating
clf = MultinomialNB().fit(X_train, y_train_rating)

# Evaluates model on test set
test['binary_rating'] = test.rating > 5
y_test_rating = test.binary_rating
pred = clf.predict(X_test)

print("Accuracy: %s" % str(clf.score(X_test, y_test_rating)))
print("Confusion Matrix")
print(confusion_matrix(pred, y_test_rating))
# Trains random forest classifier
start = time.time()
rfc_rating = RandomForestClassifier(n_estimators=100, random_state=42, max_depth = 10000, min_samples_split = 0.001)
rfc_rating.fit(X_train, y_train_rating)
end = time.time()
print("Training time: %s" % str(end-start))

# Evaluates model on test set
pred = rfc_rating.predict(X_test)

print("Accuracy: %s" % str(rfc_rating.score(X_test, y_test_rating)))
print("Confusion Matrix")
print(confusion_matrix(pred, y_test_rating))
b = "'@#$%^()&*;!.-"
X_train = np.array(train['review'])
X_test = np.array(test['review'])

def clean(X):
    for index, review in enumerate(X):
        for char in b:
            X[index] = X[index].replace(char, "")
    return(X)

X_train = clean(X_train)
X_test = clean(X_test)
print(X_train[:2])
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from keras.utils import to_categorical
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
import nltk

vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'),lowercase=True, max_features=5000)
#vectorizer = TfidfVectorizer(binary=True, stop_words=stopwords.words('english'), lowercase=True, max_features=5000)
test_train = np.concatenate([X_train, X_test])
print(test_train.shape)
X_onehot = vectorizer.fit_transform(test_train)
stop_words = vectorizer.get_stop_words()
print(type(X_onehot))
print(X_onehot.shape)
print(X_onehot.toarray())
names_list = vectorizer.get_feature_names()
names = [[i] for i in names_list]
names = Word2Vec(names, min_count=1)
print(len(list(names.wv.vocab)))
print(list(names.wv.vocab)[:5])

def score_transform(X):
    y_reshaped = np.reshape(X['rating'].values, (-1, 1))
    for index, val in enumerate(y_reshaped):
        if val >= 8:
            y_reshaped[index] = 1
        elif val >= 5:
            y_reshaped[index] = 2
        else:
            y_reshaped[index] = 0
    y_result = to_categorical(y_reshaped)
    return y_result
    
    print(X_onehot)
y_train_test = pd.concat([train, test], ignore_index=True)
y_train = score_transform(y_train_test)
print(y_train)
print(y_train.shape)
from numpy.random import seed

np.random.seed(1)
model = Sequential()
model.add(Dense(units=256, activation='relu', input_dim=len(vectorizer.get_feature_names())))
model.add(Dense(units=3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary
history = model.fit(X_onehot[:-53866], y_train[:-53866], epochs=6, batch_size=128, verbose=1, validation_data=(X_onehot[157382:157482], y_train[157382:157482]))
scores = model.evaluate(X_onehot[157482:], y_train[157482:], verbose=1)
scores[1]
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
all_names = [i.split() for i in X_train]
np.random.seed(1)
all_names_rand = [all_names[np.random.randint(low=1, high=150000)] for i in range(5000)]
print(len(all_names_rand))
all_names_list = Word2Vec(all_names_rand, min_count=1)
all_names_vec = all_names_list[all_names_list.wv.vocab]
print(all_names[0])
kclusterer_all = KMeansClusterer(5, distance=nltk.cluster.util.cosine_distance, repeats=10)
assigned_clusters_all = kclusterer_all.cluster(all_names_vec, assign_clusters=True)
print(len(assigned_clusters_all))
def generate_df(feature_names): 
    
    #  creates a zipped dictionary with every word from all of the reviews as a key and its assigned cluster as its value
    all_words_dict = dict(zip(all_names_list.wv.vocab, assigned_clusters_all))
    
    #iterates through and deletes any word that isn't a feature.
    for key in list(all_words_dict.keys()):
        if key in list(feature_names):
            pass
        else:
            del all_words_dict[key]
            
    #dictionary is then converted into nested list, with each inner list corresponding to one cluster.
    sorted_names = []
    for cluster in range(5):
        cluster_list = []
        for key, value in all_words_dict.items():
            if value == cluster:
                cluster_list.append(key)
        sorted_names.append(cluster_list)
        
    #inner list word features are sorted alphabetically then converted into a pandas DataFrame.
    for index, entry in enumerate(sorted_names):
        entry.sort()
    
    df_all = pd.DataFrame(sorted_names).T
    print(df_all[:50])
    
    #returns pandas dataframe with each cluster as a column and a list of lists where each list is all of the words assigned to that cluster.
    return df_all, sorted_names
    

df,sorted_names_all = generate_df(names.wv.vocab)
def test_clusters(cluster_list):
    #function iterates through a list of lists. these lists contain names of the feature words we want to test our model with.
    
    #score_list for accuracy of each cluster on model.
    score_list = []
    lens = []
    
    #number of reviews tested on.
    reviewnum = 15000
    
    #random sampling of reviews.
    np.random.seed(3)
    indicies = [np.random.randint(low=1, high=150000) for i in range(reviewnum)]
    X_sample =  test_train[indicies]
    y_sample = y_train[indicies]
    
    #beginning iteration through words per list.
    for cluster in cluster_list:
        
        #appending length for print statement at the end.
        lens.append(len(cluster))

        X_onehot = vectorizer.fit_transform(X_sample)
        
        X_onehot = X_onehot.toarray()

        cluster_indexes = []

        #if the feature name in the feature vocab is found in the cluster, the index is appended to the cluster index list.
        for index, feature_name in enumerate(list(names.wv.vocab)):
            if feature_name in cluster:
                cluster_indexes.append(index)

        #amount of features for the input layer dimension of the neural net
        features = len(cluster_indexes)
        
        #creating specific X_onehot matrix with only columns of corresponding vector columns
        X_onehot = X_onehot[:, cluster_indexes]

        model_cluster = Sequential()
        model_cluster.add(Dense(units=256, activation='relu', input_dim=features))
        model_cluster.add(Dense(units=3, activation='softmax'))

        model_cluster.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_cluster.summary
        
        #indexing so 2000 reviews are saved for testing
        history = model_cluster.fit(X_onehot[:reviewnum-2000], y_sample[:reviewnum-2000], epochs=10, batch_size=128, verbose=2, validation_data=(X_onehot[reviewnum-2000:reviewnum-1900], y_sample[reviewnum-2000:reviewnum-1900]))
        
        y_test = score_transform(test)
        scores = model_cluster.evaluate(X_onehot[reviewnum-2000:], y_sample[reviewnum-2000:], verbose=1)
        
        #score of model is appended to list returned at the end
        score_list.append(scores[1])
    for index, entry in enumerate(score_list):
        print("cluster", index + 1, "accuracy: ", str(entry) + ". number of words for cluster: ", lens[index])
test_clusters(sorted_names_all)
test_clusters([list(names.wv.vocab)[:1000], list(names.wv.vocab)[2000:3000], list(names.wv.vocab)[3000:4000], list(names.wv.vocab)[4000:5000]])
vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'),
                             lowercase=True, min_df=3, max_df=0.9, max_features=1000)
X_onehot = vectorizer.fit_transform(test_train)
names_list = vectorizer.get_feature_names()
names = [[i] for i in names_list]
names = Word2Vec(names, min_count=1)

df, sorted_names_all = generate_df(names.wv.vocab)

test_clusters(sorted_names_all)
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#Reimporting data due to all of the weird transformations that have been applied to our original X_train as well as different variable names

train = pd.read_csv('../input/drugsComTrain_raw.csv')
test = pd.read_csv('../input/drugsComTest_raw.csv')

# Get review text
reviews = np.vstack((train.review.values.reshape(-1, 1), 
                     test.review.values.reshape(-1, 1)))

# Set up function to re-vectorize reviews. This time binary is set to false, we only have 500 max features and min and max_df arguments have been set.
vectorizer = CountVectorizer(binary=False, stop_words=stopwords.words('english'),
                             lowercase=True, min_df=3, max_df=0.9, max_features=500)

# Vectorize reviews
X = vectorizer.fit_transform(reviews.ravel()).toarray()

# Get ratings
ratings = np.concatenate((train.rating.values, test.rating.values)).reshape(-1, 1)

y = ratings

X_train, X_test = X[:train.values.shape[0], :], X[train.values.shape[0]:, :] 
y_train, y_test = y[:train.values.shape[0]], y[train.values.shape[0]:]
X.shape, y.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train[:5000], y_train[:5000])
pred = lin_reg.predict(X_train[5000:])
np.sum(np.abs(y_train[5000:] - pred[:])) / (161297 - 5000)