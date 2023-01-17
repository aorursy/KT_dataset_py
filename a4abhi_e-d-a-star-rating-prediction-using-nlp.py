import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS 
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
print(train.shape)

print(test.shape)
print(train.isnull().sum())

print(test.isnull().sum())
dup = train.duplicated().sum()

dup1 = test.duplicated().sum()

print("Number of duplicate values in train data : " + str(dup))

print("Number of duplicate values in test data : " + str(dup))
rev = train['Review Title'].count()

rev1 = test['Review Title'].count()

print(" Total no. of people who gave Review title in train data are " + str(rev))

print(" Total no. of people who gave Review title in test data are " + str(rev1))
avg = train['Star Rating'].mean()

Avg = round(avg,1)

print("Average rating given by users is " + str(Avg))
oldest = train['App Version Name'].min()

latest = train['App Version Name'].max()



print("Oldest App Version is " + str(oldest))

print("Latest App version is " + str(latest))



train.loc[(train['App Version Name']<2),'App Version'] = 1      #Version1

train.loc[(train['App Version Name']>=2) ,'App Version'] = 2    #Version2
train.head()
sns.countplot("App Version", data = train)

plt.ylabel('Number of Users')

plt.show()
sns.countplot(x="Star Rating" ,data=train)

plt.show()
sns.countplot(x="Star Rating", hue = 'App Version', data=train)

plt.show()
df = train[['Review Title','Star Rating']]  #Creating a new dataframe with Review title and rating variable

df1 = df.dropna()                           #Removing null values

sns.countplot('Star Rating', data = df1)

plt.show()
train["Review_Length"]= train["Review Text"].str.len()     #Calculating and storing review's length

train["Title_Length"] = train["Review Title"].str.len()    #Calculating and storing title's review
sns.distplot(train['Review_Length'].dropna())

plt.show()
sns.distplot(train['Title_Length'].dropna())

plt.show()
plt.scatter(train['Review_Length'], train['Star Rating'])

plt.title('Review_Length vs Star Rating')

plt.xlabel('Review Length')

plt.ylabel('Star Rating')

plt.show()

print("Review Length to Star Rating Correlation:",train['Star Rating'].corr(train['Review_Length']))
plt.scatter(train['Title_Length'], train['Star Rating'])

plt.title('Title_Length vs Star Rating')

plt.xlabel('Title Length')

plt.ylabel('Star Rating')

plt.show()

print("Title Length to Star Rating Correlation:",train['Star Rating'].corr(train['Title_Length']))
comment_words = ' '

stopwords = set(STOPWORDS) 

  

# iterate through the csv file 

for val in train["Review Text"]: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

          

    for words in tokens: 

        comment_words = comment_words + words + ' '

  

  

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (9, 9), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 

comment_words = ' '

stopwords = set(STOPWORDS) 

  

# iterate through the csv file 

for val in test["Review Text"]: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

          

    for words in tokens: 

        comment_words = comment_words + words + ' '

  

  

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (9, 9), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 

f = train['Review Title'].dropna() #Extracting coloumn and removing null values from train data.

g = test['Review Title'].dropna()  #Extracting coloumn and removing null values from test data.
comment_words = ' '

stopwords = set(STOPWORDS) 

  

# iterate through the csv file 

for val in f: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

          

    for words in tokens: 

        comment_words = comment_words + words + ' '

  

  

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (9, 9), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 

comment_words = ' '

stopwords = set(STOPWORDS) 

  

# iterate through the csv file 

for val in g: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

          

    for words in tokens: 

        comment_words = comment_words + words + ' '

  

  

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (9, 9), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 

import numpy as np

import pandas as pd

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

stop = stopwords.words('english')

from sklearn.metrics import f1_score, accuracy_score

from sklearn import model_selection, naive_bayes

from sklearn.feature_extraction.text import TfidfVectorizer

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
df = train[['Review Text', 'Star Rating']]

df.head()
print(df.isnull().sum())

print('---------')

df1 = df.dropna()   # Creating new dataframe without null values

print(df1.isnull().sum())
df1['Cleaned'] = df1['Review Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

df1.head()
df1['Cleaned'] = df1['Cleaned'].str.replace('[^\w\s]','')

df1.head()
df1['Cleaned'] = df1['Cleaned'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

df1.head()
df1['Cleaned'] = df1['Cleaned'].str.replace('\d+', '')

df1.head()
lemmatizer = WordNetLemmatizer()

df1['Cleaned'] = [lemmatizer.lemmatize(row) for row in df1['Cleaned']]

df1.head()
x = df1['Cleaned']       # Independent Variable

y = df1['Star Rating']   # Dependent Variable
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(x,y, random_state = 2)
# ngram level tf-idf 

tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), max_features=5000)

tfidf_vect_ngram.fit(x)

xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)

xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
nb = naive_bayes.MultinomialNB(alpha = 0.6)



model = nb.fit(xtrain_tfidf_ngram, train_y)



pred = model.predict(xvalid_tfidf_ngram)



acc = accuracy_score(valid_y,pred)



print('Accuracy of validation set is :', acc)
score = f1_score(valid_y, pred, average='weighted')

print("Weighted F score is ",score)
df2 = test[['id','Review Text']]

df3 = df2.dropna()              #Removing null values

df3.head()
#To lower case

df3['Cleaned'] = df3['Review Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))



#Removing punctuation

df3['Cleaned'] = df3['Cleaned'].str.replace('[^\w\s]','')



#Removing Stopwords

df3['Cleaned'] = df3['Cleaned'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))



#Removing digits

df3['Cleaned'] = df3['Cleaned'].str.replace('\d+', '')



#Lemmatizing

lemmatizer = WordNetLemmatizer()

df3['Cleaned'] = [lemmatizer.lemmatize(row) for row in df3['Cleaned']]

df3.head()
x1 = df3['Cleaned']
# ngram level tf-idf 

tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), max_features=5000)

tfidf_vect_ngram.fit(x1)

xtest_tfidf_ngram =  tfidf_vect_ngram.transform(x1)
test_pred = model.predict(xtest_tfidf_ngram)

df3['Star Rating'] = test_pred

df4 = df3[['id','Star Rating']]

df4.to_csv("predictions.csv")