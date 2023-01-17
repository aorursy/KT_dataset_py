# import required libraries
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from textblob.classifiers import NaiveBayesClassifier
from sklearn.metrics import confusion_matrix
import re
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# load the dataset and take a look at the first few records
df = pd.read_csv('../input/London_hotel_reviews.csv', encoding = "ISO-8859-1")
print(df.shape)
df.head()
print(df.isna().sum(), end = '\n\n')
df[df.isnull().any(axis=1)].head()
df[df["Date Of Review"].isnull()]
print(len(df[df['Review Title'].str.contains("<U")]), 'reviews that are probably gibberish.')
df[df['Review Title'].str.contains("<U")].head()
df = df[df['Review Title'].str.contains("<U") == False]
sns.set(rc={'figure.figsize':(15,10)}) # this will set the size of all the following graphs (default is too small)
grid = sb.countplot(x = 'Property Name', data = df, order = df['Property Name'].value_counts().index)
grid.set_title('Number Of Reviews Per Hotel')
grid.set_xticklabels(grid.get_xticklabels(), rotation=90)
grid = sb.countplot(x = 'Review Rating', data = df, order = df['Review Rating'].value_counts().index)
grid.set_title('Number Of Ratings')
# get average review rating for each hotel
average_rating_df = pd.DataFrame(columns = ['Property Name', 'Average Rating'])
count = 0
for i in df['Property Name'].unique():
    average_rating = sum(df['Review Rating'][df['Property Name'] == i]) / sum(df['Property Name'] == i)
    average_rating_df.loc[count] = [i, average_rating]
    count += 1 
average_rating_df = average_rating_df.sort_values('Average Rating', ascending = False)
average_rating_df.plot(kind = 'bar', x = 'Property Name')
df = df.fillna('NA') # 3,953 NA's in the "Location Of The Reviewer" column, so replace these with the string 'NA' so the following loop doesn't run into an error
specific_locations = []
for i in df['Location Of The Reviewer']:
    if ", " not in i:
        specific_locations.append(i)
    else:
        specific_locations.append(i.rsplit(", ", 1)[1])
df['Specific Location'] = specific_locations

grid = sb.countplot(x = 'Specific Location', data = df, order = df['Specific Location'].value_counts().iloc[:20].index)
grid.set_title('Locations Of Reviewers')
grid.set_xticklabels(grid.get_xticklabels(), rotation=90)
years = []
for i in df['Date Of Review']:
    years.append(i[-4:])
df['Year'] = years
grid = sb.countplot(x = 'Year', data = df)
grid.set_title('Review Count Per Year')
grid.set_xticklabels(grid.get_xticklabels(), rotation=90)
df['Complete Review'] = df['Review Title'] + ' ' + df['Review Text']
df.loc[df['Review Rating'] > 4, 'Good Review'] = 1
df.loc[df['Review Rating'] <= 4, 'Good Review'] = 0
print(sum(df['Good Review'] == 0) / len(df['Good Review']) * 100, 'percent of reviews are bad (less than 5 star).')
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews

reviews_train_clean = preprocess_reviews(train['Complete Review'])
reviews_test_clean = preprocess_reviews(test['Complete Review'])
cv = CountVectorizer(binary=True)
cv.fit(reviews_train_clean)
X_train = cv.transform(reviews_train_clean)
X_test = cv.transform(reviews_test_clean)
target_train = train['Good Review']
target_test = test['Good Review']

model = LogisticRegression()
model.fit(X_train, target_train)
print ("Accuracy: %s" % accuracy_score(target_test, model.predict(X_test)))
feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), model.coef_[0]
    )
}
print('Good words:', end = "\n\n")
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:5]:
    print (best_positive)
print('')
print('Bad words:', end = "\n\n")
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:5]:
    print (best_negative)