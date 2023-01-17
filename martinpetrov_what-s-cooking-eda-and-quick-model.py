# Import needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
pd.options.display.max_colwidth = 300
from wordcloud import WordCloud, STOPWORDS
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
# Import datasets
df_train = pd.read_json("../input/train.json")
df_test = pd.read_json("../input/test.json")
# Check the head of the training dataset
df_train.head()
# Check what are the cuisines occuring the most in the dataset. 
# It seems italian, mexican and southern US are the winners
sns.countplot(y=df_train['cuisine'],order=df_train['cuisine'].value_counts().index,orient='')
# Make sure we display the ingredients without commas
df_train['ingredients']=[" ".join(x) for x in df_train['ingredients'].values]
df_test['ingredients']=[" ".join(x) for x in df_test['ingredients'].values]
# Plot a word cloud of the ingredients to see what are the most common ones

ingredients = ' '
stopwords = set(STOPWORDS)
 
# iterate through the csv file
for val in df_train['ingredients']:
    # typecaste each val to string
    val = str(val)
    # split the value
    tokens = val.split()
         
    for words in tokens:
        ingredients = ingredients + words + ' '

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = STOPWORDS,
                min_font_size = 10).generate(ingredients)

plt.figure(figsize = (10, 10), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
# Create a series to store the labels: y
y = df_train.cuisine

# Create training and test sets and set random state
X_train, X_test, y_train, y_test = train_test_split(df_train['ingredients'],y,test_size=0.33,random_state=53)

#Initialize and fit a count vectorizer
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)
# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(count_train,y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test,pred)
print(score)
# Now try the same with a TfidfVectorizer

# Initialize a TfidfVectorizer object: tfidf_vectorizer and remove the terms appearing in more than 70% of the recipes
tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)

# Transform the training data: tfidf_train 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data: tfidf_test 
tfidf_test = tfidf_vectorizer.transform(X_test)

# Fit the classifier to the training data
nb_classifier.fit(tfidf_train,y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(tfidf_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test,pred)
print(score)
# Create the list of alphas: alphas
alphas = np.arange(0.1,1,0.1)

# Define train_and_predict()
def train_and_predict(alpha):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=alpha)
    # Fit to the training data
    nb_classifier.fit(tfidf_train,y_train)
    # Predict the labels: pred
    pred = nb_classifier.predict(tfidf_test)
    # Compute accuracy: score
    score = metrics.accuracy_score(y_test,pred)
    return score

# Iterate over the alphas and print the corresponding score
for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()
# Fit the classifier, predict on the test set and submit the results
nb_classifier = MultinomialNB(alpha=0.1)
# Fit to the training data
nb_classifier.fit(tfidf_train,y_train)

# Predict on the test data
tfids_score = tfidf_vectorizer.transform(df_test['ingredients'])
predictions=nb_classifier.predict(tfids_score)

df_test['cuisine']=predictions
submission=df_test[['id','cuisine']]
submission.to_csv('submission1.csv',index=False)
