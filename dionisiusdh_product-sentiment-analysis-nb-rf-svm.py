import numpy as np
import pandas as pd 

# Visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# sklearn for feature extraction & modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Iteratively read files
import glob
import os

# For displaying images in ipython
import seaborn as sns
sns.set(color_codes = True)
%matplotlib inline
train = pd.read_csv('../input/student-shopee-code-league-sentiment-analysis/train.csv')
train.head()
test = pd.read_csv('../input/student-shopee-code-league-sentiment-analysis/test.csv')
test.head()
#Library for building wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
heading_1 = train[train["rating"]==1]["review"] # Extract only Summary of reviews
collapsed_heading_1 = heading_1.str.cat(sep=' ')

heading_2 = train[train["rating"]==2]["review"] # Extract only Summary of reviews
collapsed_heading_2 = heading_2.str.cat(sep=' ')

heading_3 = train[train["rating"]==3]["review"] # Extract only Summary of reviews
collapsed_heading_3 = heading_3.str.cat(sep=' ')

heading_4 = train[train["rating"]==4]["review"] # Extract only Summary of reviews
collapsed_heading_4 = heading_4.str.cat(sep=' ')

heading_5 = train[train["rating"]==5]["review"] # Extract only Summary of reviews
collapsed_heading_5 = heading_5.str.cat(sep=' ')
# Create stopword list:
stopwords = set(STOPWORDS)
#stopwords.update(["Subject","re","fw","fwd"])

print("Word Cloud for Rating 1")

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white",max_words=50).generate(collapsed_heading_1)

# Display the generated image:
# the matplotlib way:1
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

print("\nWord Cloud for Rating 2")

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white",max_words=50).generate(collapsed_heading_2)

# Display the generated image:
# the matplotlib way:1
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

print("\nWord Cloud for Rating 3")
# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white",max_words=50).generate(collapsed_heading_3)

# Display the generated image:
# the matplotlib way:1
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

print("\nWord Cloud for Rating 4")

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white",max_words=50).generate(collapsed_heading_4)

# Display the generated image:
# the matplotlib way:1
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
     
print("\nWord Cloud for Rating 5")
# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white",max_words=50).generate(collapsed_heading_5)

# Display the generated image:
# the matplotlib way:1
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# Building Pipeline for raw text transformation
clf = Pipeline([
    ('vect', CountVectorizer(stop_words= "english")),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
    ])
# Remove records with blank values
train_1 = train.dropna()
train_1.shape , train.shape
X_train, X_test, y_train, y_test = train_test_split(train["review"], train["rating"],random_state = 42,
                                                   test_size = 0.10)
X_train.shape,X_test.shape,y_train.shape
model = clf.fit(X_train,y_train)
print("Accuracy of Naive Bayes Classifier is {}".format(model.score(X_test,y_test)))
y_predicted = model.predict(X_test)
y_predicted[0:10]
test = pd.read_csv('../input/student-shopee-code-league-sentiment-analysis/test.csv')
test.head()
preds = model.predict(test['review'])
preds[0:10]
subs = pd.DataFrame(test['review_id'])
subs['rating'] = preds
subs
subs.to_csv('subs.csv', index=False)