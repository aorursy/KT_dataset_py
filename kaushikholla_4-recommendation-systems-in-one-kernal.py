# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing libraries



import pandas as pd

import numpy as np

import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import RegexpTokenizer

from nltk.tokenize import word_tokenize

from nltk.stem.wordnet import WordNetLemmatizer

#To track function execution

from tqdm import tqdm

from bs4 import BeautifulSoup



#Libraries for Sentimental analysis

from nltk.sentiment.vader import SentimentIntensityAnalyzer



#Libraries for visualization

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.sparse import csr_matrix

import warnings; warnings.simplefilter('ignore')

%matplotlib inline

from scipy.sparse.linalg import svds



#Libraries for ML

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import train_test_split

from sklearn.neighbors import NearestNeighbors

from sklearn import neighbors

from scipy.spatial.distance import cosine

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc



from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score
# Reading dataframe

df = pd.read_csv("../input/amazon-fine-food-reviews/Reviews.csv")
#Looking at top 5 rows

df.head(5)
# Checking for null values in the dataframe.

df.isnull().sum()
# Inspecting entries with Null value in profileName column 

df[df["ProfileName"].isnull()]
# Dropping Null values

df.dropna(inplace=True)
# Checking if null value exist again

df.isnull().sum()
# Checking the columns of the reviews.

df.columns

#Checking the shape of the dataframe.

df.shape
# Checking for the info of the dataframe.

df.info()
# Statistical analysis of the dataframe.

df.describe()
# Checking number of reviews for each score.

df["Score"].value_counts()
total = df["Score"].count()

print(total)
percent_plot = pd.DataFrame({"Total":df["Score"].value_counts()})

percent_plot.reset_index(inplace = True)

percent_plot.rename(columns={"index":"Rating"},inplace=True)
percent_plot
sns.barplot(x="Rating",y="Total", data=percent_plot)
percent_plot["Percent"] = percent_plot["Total"].apply(lambda x: (x/total)*100)
#percent_plot.drop(['percent'],axis=1, inplace = True)
percent_plot
sns.barplot(x="Rating", y="Percent", data = percent_plot)
df.columns
df["word_count"] = df["Text"].apply(lambda x: len(str(x).split(" ")))

df[["Text","word_count"]].head()
# Checking the statistics of word count to check for range and average number of the words in each article.

df["word_count"].describe()
#Checking for top 20 most repeated words - Gives insights on data specific stop words.



common_words = pd.Series(' '.join(df["Text"]).split()).value_counts()

common_words[:20]
# Checking 20 most uncommon words

common_words[-20:]
# Removing Stopwords

stop_words = set(stopwords.words("english"))



# Adding common words from our document to stop_words



add_words = ["the","I","and","a","to","of","is","it","for","in","this","that","my","with",     

"have",     

"but",      

"are",      

"was",      

"not",      

"you"]



stop_words = stop_words.union(add_words)
#Below Function is to clean the text and prepare it for the next phase.



from tqdm import tqdm

corpus = []



def clean_content(df):

    cleaned_content = []

    

    for sent in tqdm(df["Text"]):

        

        #Removing HTML comtent

        review_content = BeautifulSoup(sent).get_text()

        

        #Removing non-alphabetic charecters

        review_content = re.sub("[^a-zA-Z]"," ", review_content)

        

        #Tokenize the sentences

        words = word_tokenize(review_content.lower())

        

        #Removing the stop words

        sto_words_removed = [word for word in words if not word in stop_words]

        sto_words_removed = " ".join(sto_words_removed)

        corpus.append(sto_words_removed)

        cleaned_content.append(sto_words_removed)

        

    return (cleaned_content)
df["cleaned_text"] = clean_content(df)
df.head()
wordcloud = WordCloud(

                    background_color = "white",

                    stopwords = stop_words,

                    max_words = 100,

                    max_font_size = 50).generate(str(corpus))
# Displaying the word cloud

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

#fig.savefig("word1.png", dpi=900)
rec_df = df[['ProductId', 'UserId','Score']]
rec_df.columns
# Finding the unique userId and ProductID

print("Number of unique users = ", df['UserId'].nunique())

print("Number of unique products = ", df['ProductId'].nunique())
# Score description

rec_df[['Score']].describe().transpose()
counts = rec_df['UserId'].value_counts()

#df_top50 = rec_df[rec_df['UserID']]
df_top50 = rec_df[rec_df['UserId'].isin(counts[counts >= 50].index)]
df_top50.head()
print("Number of users who have rated 50 or more items = ", len(df_top50))

print("Number of unique users who have rated more than 50 items = ", df_top50['UserId'].nunique())

print("Number of unique items in the list = ", df_top50['ProductId'].nunique())
#Calculating the density for this matrix

final_rating_matrix = pd.pivot_table(df_top50, index='UserId', columns = 'ProductId', values = 'Score')
final_rating_matrix.fillna(0, inplace = True)
final_rating_matrix.shape
# Matrix for Item based Recommender system where product on the column and User on the row.

final_rating_matrix_item = final_rating_matrix.transpose()

final_rating_matrix_item.head()
# Creating train test split of the data into 70:30

train_data, test_data = train_test_split(df_top50, test_size = 0.3, random_state = 0)

train_data.head()
train_data_group = train_data.groupby('ProductId').agg
train_data_grouped = train_data.groupby('ProductId').agg({'UserId':'count'}).reset_index()
train_data_grouped.head()
train_data_grouped.rename(columns= {'UserId':'Score'},inplace=True)

train_data_grouped.head()
train_data_sort = train_data_grouped.sort_values(['Score','ProductId'],ascending = False)
train_data_sort.head()
# Generating Recommendation Rank

train_data_sort['Rank'] = train_data_sort['Score'].rank(ascending=0, method='first')

train_data_sort.head()
# Top 10 popularity based recommendations are:

pop_rec = train_data_sort.head(10)

pop_rec
# Using Popularity Based Recommendation System to make Prediction.

def make_recommendation(user_id):

    user_recommendation = pop_rec

    # Adding user Id to popularity based recommender system

    user_recommendation['user_id'] = user_id

    

    cols = user_recommendation.columns.tolist()

    cols = cols[-1:] + cols[:-1]

    user_recommendation = user_recommendation[cols]

    

    return user_recommendation
user_list = [15, 121, 200]



for i in user_list:

    print("Recommendation for userId: {}".format(i))

    print(make_recommendation(i))

    print()
count = df.groupby("ProductId", as_index = False).count()

mean = df.groupby('ProductId', as_index = False).mean()
df1 = pd.merge(df,count, how='right', on=["ProductId"])
df1.head()
df1["count"] = df1["UserId_y"]

df1["Score"] = df1["Score_x"]

df1["Summary"] = df1["Summary_x"]
df1.head()
df1 = df1[['ProductId','Summary','Score','count']]
# Choosing those products having over 100 reviews

df1 = df1.sort_values(["count"], ascending = False)

df_100 = df1[df1['count'] >= 100]
df_100.head()
combined_summary = df_100.groupby("ProductId")["Summary"].apply(list)
df3 = df_100.groupby("ProductId", as_index = False ).mean()
# Adding combined summary to the dataframe

df4 = pd.merge(df3, combined_summary, on = "ProductId", how = "inner")
df4.head()
df4 = df4[['ProductId','Score','Summary']]
df4.head()
# Tokenizing

cleanu = re.compile('[^a-z]+')



def cleanup(sent):

    sent = [i.lower() for i in sent]

    sent = ",".join(sent)

    sent = sent.lower()

    sent = cleanu.sub(' ',sent).strip()

    sent = " ".join(nltk.word_tokenize(sent))

    return sent
df4["summary_cleaned"] = df4["Summary"].apply(cleanup)
df4.head()
df4 = df4.drop_duplicates(['Score'], keep='last')
df4 = df4.reset_index()
df4.head()
document = df4["summary_cleaned"]

vect = CountVectorizer(max_features = 100, stop_words = 'english')

X = vect.fit_transform(document)
df5 = pd.DataFrame(X.A, columns = vect.get_feature_names())
df5 = df5.astype(int)
df5.head()
X = df5.to_numpy()
type(X)
df5_train, df5_test = train_test_split(X, test_size = 0.1, random_state = 42)
print(df5_train.size)

print(df5_test.size)
knn = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')
knn.fit(df5_train)
distances, indices = knn.kneighbors(df5_train)
len_train = len(df5_train)

len_test = len(df5_test)
for i in range(len_test):

    a = knn.kneighbors([df5_test[i]])

    related_product_list = a[1]

    

    first_related_product = [item[0] for item in related_product_list]

    first_related_product = str(first_related_product).strip('[]')

    first_related_product = int(first_related_product)

    second_related_product = [item[1] for item in related_product_list]

    second_related_product = str(second_related_product).strip('[]')

    second_related_product = int(second_related_product)

    

    print ("Based on product reviews, for ", df4["ProductId"][len_train + i] ," and this average Score is ",df4["Score"][len_train + i])

    print ("The first similar product is ", df4["ProductId"][first_related_product] ," and this average Score is ",df4["Score"][first_related_product])

    print ("The second similar product is ", df4["ProductId"][second_related_product] ," and this average Score is ",df4["Score"][second_related_product])

    print ("-----------------------------------------------------------")
count = df.groupby("UserId", as_index = False).count()

mean = df.groupby("UserId", as_index = False).mean()



df1 = pd.merge(df,count, how = "right", on=["UserId"])
df1["Count"] = df1["ProductId_y"]

df1["Score"] = df1["Score_x"]

df1["Summary"] = df1["Summary_x"]



df1 = df1[["UserId","Summary","Score","Count"]]
df1.head()
df1 = df1.sort_values(['Count'], ascending = False)

df2 = df1[df1.Count >= 100]
df2.head()
df3 = df.groupby("UserId", as_index = False).mean()

df3.head()
total_summary = df2.groupby("UserId")["Summary"].apply(list)
df4 = pd.merge(df3, total_summary, on="UserId", how = "inner")

df4 = df4[["UserId","Summary","Score"]]
df4.head()
df4["CleanedSummary"] = df4["Summary"].apply(cleanup)

df4.drop_duplicates(["Score"], keep = "last")

df4.reset_index()

df4.head()
docs = df4["CleanedSummary"]

tfidf = TfidfVectorizer(stop_words = "english")

X = tfidf.fit_transform(docs)
tfidf.get_feature_names()
df5 = pd.DataFrame(X.A, columns = tfidf.get_feature_names())

df5.head()
X = np.array(df5)



df5_train, df5_test = train_test_split(X, test_size = 0.05, random_state = 42)
df6 = df.drop_duplicates(['Summary'], keep='last')

df6 = df6.reset_index()
len_train = len(df5_train)

len_test = len(df5_test)



knn_u = NearestNeighbors(n_neighbors = 3, algorithm = 'ball_tree')

knn_u.fit(df5_train)



distance, indices = knn_u.kneighbors(df5_train)
# Finding the similar users and their products



for i in range(len_test):

    a = knn_u.kneighbors([df5_test[i]])

    related_product_list = a[1]

    

    first_related_product = [item[0] for item in related_product_list]

    first_related_product = str(first_related_product).strip('[]')

    first_related_product = int(first_related_product)

    second_related_product = [item[1] for item in related_product_list]

    second_related_product = str(second_related_product).strip('[]')

    second_related_product = int(second_related_product)

    

    

    print ("Based on  reviews, for the user", df3["UserId"][len_train + i])

    print ("The first similar user is ", df3["UserId"][first_related_product]) 

    print ("He/She likes following products")

    count = 0

    for i in range(295736):

        if (df6["UserId"][i] == df4["UserId"][first_related_product]) & (df6["Score"][i] == 5):

            aaa= df6["ProductId"][i]

            count += 1

            print (aaa)

        

        if count == 10:

            break

    print ("--------------------------------------------------------------------")
df["ProductId"][df["ProductId"][df["UserId"] == "#oc-R1CSQFEG6ZI93U"]]
df["UserId"][df["UserId"] == "#oc-R1CSQFEG6ZI93U"]
counts = df['UserId'].value_counts()

df_svd = df[df['UserId'].isin(counts[counts >= 50].index)]
df_svd.head()
df_svd = df_svd.drop(['word_count','cleaned_text','Id','ProfileName','Time','HelpfulnessNumerator','HelpfulnessDenominator','Text','Summary'], axis = 1)
df_svd.head()
train_data, test_data = train_test_split(df_svd, test_size = 0.3, random_state = 0)
df_svd = df_svd.reset_index()
df_svd.head()
# Creating Sparse matrix with users on row and items on column.

pivot_df = pd.pivot_table(df_svd, index = ["UserId"], columns = "ProductId", values = "Score")

pivot_df.fillna(0, inplace = True)

pivot_df.head()
pivot_df.shape
# Adding index and removing UserId as Index

pivot_df['user_index'] = np.arange(0, pivot_df.shape[0],1)
pivot_df.set_index(['user_index'], inplace = True)



pivot_df.head()
u, sigma, vt = svds(pivot_df, k = 50)
sigma = np.diag(sigma)
sigma
predicted_ratings = np.dot(np.dot(u,sigma),vt)



# Predicted Ratings

pred_df = pd.DataFrame(predicted_ratings, columns = pivot_df.columns)

pred_df.head()
# Recommending the items to the given user based on the predicted rating 



def svd_recommender(userId, pivot_df, pred_df, num_recommendations):

    

    user_idx = userId - 1

    # Getting the items that user has already rated

    sorted_user_ratings = pivot_df.iloc[120].sort_values(ascending = False)

    # Getting the prediction value for the item

    sorted_user_predictions = pred_df.iloc[120].sort_values(ascending = False)

    

    # Creating a temp data frame for the user ratings and predictions

    temp = pd.concat([sorted_user_ratings, sorted_user_predictions], axis = 1)

    temp.index.name = "Recommended Items"

    temp.columns = ["User_ratings", "user_predictions"]

    # Selecting the items that the user hasnt rated i.e for which score is 0

    temp = temp.loc[temp.User_ratings == 0]

    temp = temp.sort_values("user_predictions", ascending = False)

    

    print("\n Below are the recommended items for user(user_id = {}:\n".format(userId))

    print(temp['user_predictions'].head(num_recommendations))
userId = 121

num_recommendations = 5

svd_recommender(userId, pivot_df, pred_df, num_recommendations)
pred_df.head()
#pred_df.iloc[120]
#type(pred_df)
#pivot_df.iloc[120][pivot_df.iloc[120] >= 1]