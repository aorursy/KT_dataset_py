import time

start = time.time()



# Import modules

import pandas as pd

from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

import re

import pickle



end=time.time()

print(end-start)
start = time.time()



# Read in the data

df = pd.read_csv('../input/7282_1.csv')

df2= pd.read_csv('../input/Datafiniti_Hotel_Reviews.csv')



# Important columns. City, Country, and Province columns are excluded because they are not reliable

imp_col_list = ['address', 'name', 'reviews.date', 'reviews.text','reviews.title']

df = df.loc[:,imp_col_list]

df2 = df2.loc[:,imp_col_list]

df= df.append(df2)



end=time.time()

print(end-start)
start = time.time()



# Prep text, add some columns and fillna, and rename columns

df['reviews.text'] = df['reviews.text'].str.lower()

df['reviews.text'] = df['reviews.text'].replace(to_replace='[^A-Za-z0-9]+', regex=True, value=' ')

df['reviews.text'] = df['reviews.text'].fillna('')

df['review_date'] = pd.to_datetime(df['reviews.date']).dt.date

df['review_month'] = pd.to_datetime(df['reviews.date']).dt.month

df['words_in_review'] = [len(i.split()) for i in df['reviews.text']]

season_dict = {1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Summer',

               7:'Summer',8:'Summer',9:'Fall',10:'Fall',11:'Fall',12:'Winter'}

df['review_season'] = df['review_month'].map(season_dict).fillna('Summer')

df.rename(columns={'address': 'hotel_address', 'city': 'hotel_city','country':'hotel_country', 

                   'name':'hotel_name'},inplace=True)



end=time.time()

print(end-start)
df.head()
df.describe()
# Size of dataframe

df.memory_usage().sum()/1024/1024
df.shape, df.isnull().sum()
# Import and initiate a vectorizer.





# The max features is how many words for which I want to create columns. Note I also use the 'stop_words' 

# parameter that removes English language stop words. There are issues with this method, see documentation.

# I'll leave ngram_range at (1, 1) for now but may increase this in the final model.

vectorizer = CountVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 1))
# Vectorize the reviews to transform sentences into columns.

X = vectorizer.fit_transform(df['reviews.text'])



# And then put all of that in a new dataframe.

bag_of_words = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
# Join our bag of words back to our initial hotel data.

full_df = df.join(bag_of_words)

full_df.head()
full_df.memory_usage().sum()/1024/1024
# X is the list of features. In this case, it's the bag of words. 

X = bag_of_words



# y the target

y = df['hotel_name']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)
# Import the random forest model classifier.

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(min_samples_leaf=10, random_state=8675309)
import time

start = time.time()

# Fit the model to the data.

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)



print("Accuracy score: ",round((accuracy_score(y_test, y_pred)*100),2), "%")



end=time.time()

print(end-start)
test_review = 'I loved the beach, the nearby bars, the live music, and the walkable neighborhood. The weather was great and it was sunny.'
test_review = test_review.lower()

test_review = re.sub('[^A-Za-z0-9]+', ' ', test_review)

test_review = [test_review]



# Convert your test review into a vector.

X_test = vectorizer.transform(test_review).toarray()



# Make a prediction of which hotel your review would be a review:

prediction = clf.predict(X_test)[0]



# Return the essential information about your match. Note the head() option. You can set it to 1 to only

# get back the hotel information, but this also includes the review date information. This is because

# it tells you what time of year this place is most often visited. If you search this result, you'll see

# "Inn & Suites 2540 S Mccall Rd" is a hotel in Florida. It's most often visited in Winter, which makes 

# sense because people are likely to visit Florida when the weather is bad elsewhere.

df[df['hotel_name'] == prediction][['hotel_name', 'hotel_address', 'review_date','review_month','review_season']].head(15)
start = time.time()



# Initialize a vectorizer

vectorizer = TfidfVectorizer(max_features=None, stop_words='english', ngram_range=(1, 3))



# Vectorize the reviews to transform sentences into columns

X = vectorizer.fit_transform(df['reviews.text'])



end=time.time()

print(end-start, X.shape)
# A memory error happens when I try to expand the sparse matrix into a dense array.

bag_of_words = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
start = time.time()



# Create a vocab and bag of words with the most popular words

keep_cols = X.mean(axis=0)*100

keep_cols = pd.DataFrame(keep_cols, columns=vectorizer.get_feature_names())



end=time.time()

print(end-start)
start = time.time()



keep_cols = keep_cols.transpose().reset_index().rename(columns={'index':'feature',0:'freq'})

keep_cols = keep_cols.reset_index().sort_values('freq')



end=time.time()

print(end-start)
# Note I'm increasing the threshold because I added the 10,000 updated reviews

threshold = 0.00450472037

keep_cols = keep_cols.loc[keep_cols['freq'] > threshold]

X = sparse.csc_matrix(X)

keep_list = keep_cols['index'].tolist()

vocab = keep_cols['feature'].tolist()

X = X[:,keep_list]

bag_of_words = pd.DataFrame(X.toarray(), columns=(vocab))

bag_of_words.shape
start = time.time()

df_s = df.reset_index(drop=True)

df_s = df_s.reset_index(drop=False)

df_s['review_season'].value_counts()
df_s1 = df_s.loc[df_s['review_season'] == 'Spring']

df_s2 = df_s.loc[df_s['review_season'] == 'Summer']

df_s3 = df_s.loc[df_s['review_season'] == 'Fall']

df_s4 = df_s.loc[df_s['review_season'] == 'Winter']

df_s1.shape, df_s2.shape, df_s3.shape, df_s4.shape
l_s1 = df_s1['index'].tolist()

l_s2 = df_s2['index'].tolist()

l_s3 = df_s3['index'].tolist()

l_s4 = df_s4['index'].tolist()

len(l_s1),len(l_s2),len(l_s3),len(l_s4), len(l_s1)+len(l_s2)+len(l_s3)+len(l_s4)
start = time.time()



# Assign X and y

X_s1 = bag_of_words[bag_of_words.index.isin(l_s1)]

y_s1 = df_s1['hotel_name']

X_s2 = bag_of_words[bag_of_words.index.isin(l_s2)]

y_s2 = df_s2['hotel_name']

X_s3 = bag_of_words[bag_of_words.index.isin(l_s3)]

y_s3 = df_s3['hotel_name']

X_s4 = bag_of_words[bag_of_words.index.isin(l_s4)]

y_s4 = df_s4['hotel_name']



# Train test split X and y

X_s1_train, X_s1_test, y_s1_train, y_s1_test = train_test_split(X_s1, y_s1, test_size=0.20, random_state=30)

X_s2_train, X_s2_test, y_s2_train, y_s2_test = train_test_split(X_s2, y_s2, test_size=0.20, random_state=30)

X_s3_train, X_s3_test, y_s3_train, y_s3_test = train_test_split(X_s3, y_s3, test_size=0.20, random_state=30)

X_s4_train, X_s4_test, y_s4_train, y_s4_test = train_test_split(X_s4, y_s4, test_size=0.20, random_state=30)



# Declare the classifiers

clf_s1 = RandomForestClassifier(min_samples_leaf=3, random_state=8675309)

clf_s2 = RandomForestClassifier(min_samples_leaf=3, random_state=8675309)

clf_s3 = RandomForestClassifier(min_samples_leaf=3, random_state=8675309)

clf_s4 = RandomForestClassifier(min_samples_leaf=3, random_state=8675309)



end=time.time()

# print(end-start)

X_s1.shape, X_s2.shape, X_s3.shape, X_s4.shape, y_s1.shape, y_s2.shape, y_s3.shape, y_s4.shape
X_s1_train.shape, X_s2_train.shape, X_s3_train.shape, X_s4_train.shape, X_s1_test.shape, X_s2_test.shape, X_s3_test.shape, X_s4_test.shape
y_s1_train.shape, y_s2_train.shape, y_s3_train.shape, y_s4_train.shape, y_s1_test.shape, y_s2_test.shape, y_s3_test.shape, y_s4_test.shape
start = time.time()



# Fit the model to the data

clf_s1.fit(X_s1_train,y_s1_train)

y_s1_pred = clf_s1.predict(X_s1_test)



print(accuracy_score(y_s1_test, y_s1_pred))



end=time.time()

print(end-start)
start = time.time()



clf_s2.fit(X_s2_train,y_s2_train)

y_s2_pred = clf_s2.predict(X_s2_test)



print(accuracy_score(y_s2_test, y_s2_pred))



end=time.time()

print(end-start)
start = time.time()



clf_s3.fit(X_s3_train,y_s3_train)

y_s3_pred = clf_s3.predict(X_s3_test)



print(accuracy_score(y_s3_test, y_s3_pred))



end=time.time()

print(end-start)
start = time.time()



clf_s4.fit(X_s4_train,y_s4_train)

y_s4_pred = clf_s4.predict(X_s4_test)



print(accuracy_score(y_s4_test, y_s4_pred))



end=time.time()

print(end-start)
start = time.time()



# Reinitialize and refit the vectorizer with the vocabulary

vectorizer = TfidfVectorizer(max_features=None, vocabulary=vocab, stop_words='english', ngram_range=(1, 3))

X = vectorizer.fit_transform(df['reviews.text'])



end=time.time()

print(end-start)
start = time.time()



# Create a review to feed the model

test_review = 'I loved the beach, the nearby bars, the live music, and the walkable neighborhood#@!$?@#!. The weather was great and it was sunny.'



# Test season has to match case perfectly - use dropdown from website

test_season = 'Fall'



# Clean the text and convert your test review into a vector.

test_review = test_review.lower()

test_review = re.sub('[^A-Za-z0-9]+', ' ', test_review)

test_review = [test_review]

X_test = vectorizer.transform(test_review).toarray()



end=time.time()

print(end-start)
# Note how we've cleaned the test_review to be easier for the machine to read:

print(test_review)
# Define a prediction function (note that the X_test global thing may cause an error if 

# # I did it wrong, remember this during troubleshooting)

def make_prediction(season):

    global X_test

    global prediction

    if test_season == 'Spring':

        prediction = clf_s1.predict(X_test)[0]

    elif test_season == 'Summer':

        prediction = clf_s2.predict(X_test)[0]

    elif test_season == 'Fall':

        prediction = clf_s3.predict(X_test)[0]

    else:

        prediction = clf_s4.predict(X_test)[0]

    return df[df['hotel_name'] == prediction][['hotel_name', 'hotel_address']].head(1)
start = time.time()



print(make_prediction(test_season))



end=time.time()

print(end-start)
test_review = 'This was an amazing spot to go hiking. The crowd was young and the food was delicious.'

test_season = 'Fall'



# Clean the text and convert your test review into a vector.

test_review = test_review.lower()

test_review = re.sub('[^A-Za-z0-9]+', ' ', test_review)

test_review = [test_review]



X_test = vectorizer.transform(test_review).toarray()

print(make_prediction(test_season))
test_review = 'I loved the fishing. It was a relaxing vacation and this hotel really lived up to its reputation.'

test_season = 'Summer'



# Clean the text and convert your test review into a vector.

test_review = test_review.lower()

test_review = re.sub('[^A-Za-z0-9]+', ' ', test_review)

test_review = [test_review]



X_test = vectorizer.transform(test_review).toarray()

print(make_prediction(test_season))
test_review = 'Fun for the whole family. The area had a lot of activities for children which adults could enjoy too.'

test_season = 'Fall'



# Clean the text and convert your test review into a vector.

test_review = test_review.lower()

test_review = re.sub('[^A-Za-z0-9]+', ' ', test_review)

test_review = [test_review]



X_test = vectorizer.transform(test_review).toarray()

print(make_prediction(test_season))
test_review = 'The snow was incredible. Fresh powder, skiing, snowboarding, jacuzzis at night. This hotel was right by the ski lift which made for quick access to the mountain.'

test_season = 'Winter'



# Clean the text and convert your test review into a vector.

test_review = test_review.lower()

test_review = re.sub('[^A-Za-z0-9]+', ' ', test_review)

test_review = [test_review]



X_test = vectorizer.transform(test_review).toarray()

print(make_prediction(test_season))
test_review = 'I\'m a big art fan. The number of museums, operas, and collections nearby made this visit a once-in-a-lifetime experience!'

test_season = 'Spring'



# Clean the text and convert your test review into a vector.

test_review = test_review.lower()

test_review = re.sub('[^A-Za-z0-9]+', ' ', test_review)

test_review = [test_review]



X_test = vectorizer.transform(test_review).toarray()

print(make_prediction(test_season))
test_review = 'The snow was incredible. Fresh powder, skiing, snowboarding, jacuzzis at night. This hotel was right by the ski lift which made for quick access to the mountain.'

test_season = 'Winter'
def suggest_destination(review, season):

    review = review.lower()

    review = re.sub('[^A-Za-z0-9]+', ' ', review)

    review = [review]

    X_test = vectorizer.transform(review).toarray()

    if season == 'Spring':

        prediction = clf_s1.predict(X_test)[0]

    elif season == 'Summer':

        prediction = clf_s2.predict(X_test)[0]

    elif season == 'Fall':

        prediction = clf_s3.predict(X_test)[0]

    else:

        prediction = clf_s4.predict(X_test)[0]

    df_answer = df[df['hotel_name'] == prediction][['hotel_name', 'hotel_address']].head(1)

    df_answer = df_answer.reset_index(drop=True)

    answer = df_answer['hotel_name'][0], df_answer['hotel_address'][0]

    url_str = str(answer[0]).replace(" ", "%20")+"_"+str(answer[1]).replace(" ", "%20")

    url = "https://www.google.com/search?q={}".format(url_str)

    return answer, url
answer, url = suggest_destination(test_review, test_season)

print(answer, test_season)
print(url)
start = time.time()



# Pickle out the trained models

pickle_out_s1 = open("clf_s1.pickle","wb")

pickle.dump(clf_s1, pickle_out_s1, protocol=0)

pickle_out_s1.close()



end=time.time()

print(end-start)
start = time.time()



# Pickle out the trained models

pickle_out_s2 = open("clf_s2.pickle","wb")

pickle.dump(clf_s2, pickle_out_s2, protocol=0)

pickle_out_s2.close()



end=time.time()

print(end-start)
start = time.time()



# Pickle out the trained models

pickle_out_s3 = open("clf_s3.pickle","wb")

pickle.dump(clf_s3, pickle_out_s3, protocol=0)

pickle_out_s3.close()



end=time.time()

print(end-start)
start = time.time()



# Pickle out the trained models

pickle_out_s4 = open("clf_s4.pickle","wb")

pickle.dump(clf_s4, pickle_out_s4, protocol=0)

pickle_out_s4.close()



end=time.time()

print(end-start)
start = time.time()



# Pickle out the fitted vectorizer

pickle_vec_out = open("vectorizer.pickle","wb")

pickle.dump(vectorizer, pickle_vec_out, protocol=0)

pickle_vec_out.close()



end=time.time()

print(end-start)