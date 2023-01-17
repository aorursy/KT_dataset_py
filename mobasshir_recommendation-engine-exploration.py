print('>> Installing Libraries')



!pip3 install pandas matplotlib numpy scikit-surprise



print('>> Libraries Installed')
print('>> Importing Libraries')



import pandas as pd



from surprise import Reader, Dataset, SVD



from surprise.accuracy import rmse, mae

from surprise.model_selection import cross_validate



print('>> Libraries imported.')
df = pd.read_csv('/kaggle/input/movielensratings/ratings.csv')

df.head()
df.drop('timestamp', axis=1, inplace=True)

df.head()
df.isna().sum()
n_movies = df['movieId'].nunique()

n_users = df['userId'].nunique()

print(f'Number of unique movies: {n_movies}')

print(f'Number of unique users: {n_users}')
available_ratings = df['rating'].count()

total_ratings = n_movies*n_users

missing_ratings = total_ratings - available_ratings

sparsity = (missing_ratings/total_ratings) * 100

print(f'Sparsity: {sparsity}')
df['rating'].value_counts().plot(kind='bar')
filter_movies = df['movieId'].value_counts() > 3

filter_movies = filter_movies[filter_movies].index.tolist()
filter_users = df['userId'].value_counts() > 3

filter_users = filter_users[filter_users].index.tolist()
print(f'Original shape: {df.shape}')

df = df[(df['movieId'].isin(filter_movies)) & (df['userId'].isin(filter_users))]

print(f'New shape: {df.shape}')
cols = ['userId', 'movieId', 'rating']
reader = Reader(rating_scale = (0.5, 5))

data = Dataset.load_from_df(df[cols], reader)
trainset = data.build_full_trainset()

antiset = trainset.build_anti_testset()
algo = SVD(n_epochs =25, verbose = True)
cross_validate(algo, data, measures = ['RMSE', 'MAE'], cv=5, verbose= True)

print('>> Training Done')
predictions = algo.test(antiset)
predictions[0]
from collections import defaultdict

def get_top_n(predictions, n=3):

    top_n = defaultdict(list)

    for uid, iid, _, est, _ in predictions:

        top_n[uid].append((iid, est))

        

    for uid, user_ratings in top_n.items():

        user_ratings.sort(key = lambda x: x[1], reverse = True)

        top_n[uid] = user_ratings[:n]

        

    return top_n



top_n = get_top_n(predictions, n=3)



for uid, user_ratings in top_n.items():

    print(uid, [iid for (iid, rating) in user_ratings])
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords 

from nltk.tokenize import WordPunctTokenizer
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/yelpdata/yelp_review_arizona.csv')

df_business = pd.read_csv('/kaggle/input/yelpdata/yelp_business.csv')
df.head()
df_business.head()
#Select only stars and text

yelp_data = df[['business_id', 'user_id', 'stars', 'text']]
import string

from nltk.corpus import stopwords

stop = []

for word in stopwords.words('english'):

    s = [char for char in word if char not in string.punctuation]

    stop.append(''.join(s))
def text_process(mess):

    """

    Takes in a string of text, then performs the following:

    1. Remove all punctuation

    2. Remove all stopwords

    3. Returns a list of the cleaned text

    """

    # Check characters to see if they are in punctuation

    nopunc = [char for char in mess if char not in string.punctuation]



    # Join the characters again to form the string.

    nopunc = ''.join(nopunc)

    

    # Now just remove any stopwords

    return " ".join([word for word in nopunc.split() if word.lower() not in stop])
yelp_data['text'] = yelp_data['text'].apply(text_process)
#Split train test for testing the model later

vld_size=0.15

X_train, X_valid, y_train, y_valid = train_test_split(yelp_data['text'], df['business_id'], test_size = vld_size)
userid_df = yelp_data[['user_id','text']]

business_df = yelp_data[['business_id', 'text']]
userid_df[userid_df['user_id']=='ZwVz20be-hOZnyAbevyMyQ']['text']
userid_df = userid_df.groupby('user_id').agg({'text': ' '.join})

business_df = business_df.groupby('business_id').agg({'text': ' '.join})
userid_df.head()
userid_df.loc['ZwVz20be-hOZnyAbevyMyQ']['text']


from sklearn.feature_extraction.text import TfidfVectorizer
#userid vectorizer

userid_vectorizer = TfidfVectorizer(tokenizer = WordPunctTokenizer().tokenize, max_features=5000)

userid_vectors = userid_vectorizer.fit_transform(userid_df['text'])



#Business id vectorizer

businessid_vectorizer = TfidfVectorizer(tokenizer = WordPunctTokenizer().tokenize, max_features=5000)

businessid_vectors = businessid_vectorizer.fit_transform(business_df['text'])
userid_rating_matrix = pd.pivot_table(yelp_data, values='stars', index=['user_id'], columns=['business_id'])

userid_rating_matrix.head()
P = pd.DataFrame(userid_vectors.toarray(), index=userid_df.index, columns=userid_vectorizer.get_feature_names())

Q = pd.DataFrame(businessid_vectors.toarray(), index=business_df.index, columns=businessid_vectorizer.get_feature_names())
P.head()
Q.head()
def matrix_factorization(R, P, Q, steps=25, gamma=0.001,lamda=0.02):

    for step in range(steps):

        for i in R.index:

            for j in R.columns:

                if R.loc[i,j]>0:

                    eij=R.loc[i,j]-np.dot(P.loc[i],Q.loc[j])

                    P.loc[i]=P.loc[i]+gamma*(eij*Q.loc[j]-lamda*P.loc[i])

                    Q.loc[j]=Q.loc[j]+gamma*(eij*P.loc[i]-lamda*Q.loc[j])

        e=0

        for i in R.index:

            for j in R.columns:

                if R.loc[i,j]>0:

                    e= e + pow(R.loc[i,j]-np.dot(P.loc[i],Q.loc[j]),2)+lamda*(pow(np.linalg.norm(P.loc[i]),2)+pow(np.linalg.norm(Q.loc[j]),2))

        if e<0.001:

            break

        

    return P,Q
# %%time

# P, Q = matrix_factorization(userid_rating_matrix, P, Q, steps=25, gamma=0.001,lamda=0.02) ## Only when you train on new examples either load from pickle
# Store P, Q and vectorizer in pickle file

# import pickle

# output = open('yelp_recommendation_model_8.pkl', 'wb')

# pickle.dump(P,output)

# pickle.dump(Q,output)

# pickle.dump(userid_vectorizer,output)

# output.close()



# Use when only you train on new data, then export
import pickle

input = open('../input/yelpdata/yelp_recommendation_model_5.pkl','rb')

P = pickle.load(input)

Q = pickle.load(input)

userid_vectorizer = pickle.load(input)

input.close()
P.head()
Q.head()
Q.iloc[0].sort_values(ascending=False).head(10)
words = "i want to have dinner with beautiful views"

test_df= pd.DataFrame([words], columns=['text'])

test_df['text'] = test_df['text'].apply(text_process)

test_vectors = userid_vectorizer.transform(test_df['text'])

test_v_df = pd.DataFrame(test_vectors.toarray(), index=test_df.index, columns=userid_vectorizer.get_feature_names())



predictItemRating=pd.DataFrame(np.dot(test_v_df.loc[0],Q.T),index=Q.index,columns=['Rating'])

topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:7]



for i in topRecommendations.index:

    print(df_business[df_business['business_id']==i]['name'].iloc[0])

    print(df_business[df_business['business_id']==i]['categories'].iloc[0])

    print(str(df_business[df_business['business_id']==i]['stars'].iloc[0])+ ' '+str(df_business[df_business['business_id']==i]['review_count'].iloc[0]))

    print('')