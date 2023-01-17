#loading libraries
import pandas as pd
import string
import seaborn as sns
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from sklearn.ensemble import forest 
from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import cross_val_score
import statistics as stats
from sklearn.feature_extraction.text import TfidfVectorizer
from skopt import BayesSearchCV



nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


# !pip install scikit-optimize
#reading the data            
df = pd.read_csv("../input/traincsv/train.csv")


#dummy variables
id = df["id"]
location = df["location"]
keyword = df["keyword"]
df = df.rename(columns={"text":"tweet"})
df["original tweet"] = df["tweet"]
df.head()
#removing irrelevant features
df = df.drop(columns=["id","location","keyword"],axis=1)
df.head()
#Dropping variables that has more than 60 percent missing values

#Checking the percentage missing values by columns
missing_column = (df.isna().sum()/len(df))*100
print(missing_column)


duplicate = df.duplicated().sum()
print(duplicate)#we can see that there are duplicates values
 
df = df.drop_duplicates()
sns.countplot(df["target"])
#We can see the target column is balanced.
df["tweet"] = df['tweet'].str.replace('http\S+|www.\S+', '', case=False)

df.head()
#Removing punctuation
df["tweet"] = df['tweet'].str.replace('[{}]'.format(string.punctuation), '')


#Changing the special characters to the usual alphabet letters
df['tweet'] = df["tweet"].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
df.head()


df['tweet'] = df['tweet'].str.replace('\d+', '')
df["tweet"] = [word_tokenize(word) for word in df["tweet"]]
df.head()

df["tweet"] = [[word.lower() for word in words ] for words in df["tweet"]]
df.head()
stop_words = stopwords.words('english')

df["tweet"] = [[words for words in word if words not in stop_words] for word in df["tweet"]]
df.head()
lem = nltk.WordNetLemmatizer()
df["tweet"] = [[lem.lemmatize(lema,"v") for lema in i]for i in df["tweet"]]

x = df["tweet"]
y = df["target"]
x_train,x_test,y_train,y_test = train_test_split = train_test_split(x,y, test_size=0.2, random_state=0)
print(x_train.shape)
print(x_test.shape)
# # Creating a dummy fuction so it can be passed to the  (tokenizer and preprocessor) parameter
# def dummy(doc):
#     return doc

# cv = CountVectorizer(
#         tokenizer=dummy,
#         preprocessor=dummy,
#         min_df = 0.000167
        
#     )  

# x_train = cv.fit_transform(x_train)

# x_test =  cv.transform(x_test)


def dummy(doc):
    return doc

tfidf = TfidfVectorizer(
         tokenizer=dummy,
         preprocessor=dummy,
         min_df = 0.000167

)

x_train = tfidf.fit_transform(x_train)

x_test =  tfidf.transform(x_test)




xg = xgb.XGBClassifier()
fo =  forest.RandomForestClassifier()
tr = tree.DecisionTreeClassifier()
lo = linear_model.LogisticRegression()
sv = svm.SVC()

xgb_score = cross_val_score(xg,x_train,y_train,cv=5)
ran_score = cross_val_score(fo,x_train,y_train,cv=5)
dtree_score = cross_val_score(tr,x_train,y_train,cv=5)
log_score = cross_val_score(lo,x_train,y_train,cv=5)
svm_score = cross_val_score(sv,x_train,y_train,cv=5)

# This Dataframe outputs the average score for each algorithms
df_score = pd.DataFrame({"model":["xgboost","RandomForestClassifier","DecisionTreeClassifier","LogisticRegression","Support vector machine"],"score":[stats.mean(xgb_score),stats.mean(ran_score),stats.mean(dtree_score),stats.mean(log_score),stats.mean(svm_score)]})
df_score
# We can see that Support vector machine classifier gave the best score

#Checking initial model score
initial_model = svm.SVC()
initial_model = initial_model.fit(x_train,y_train)
original_score = initial_model.score(x_test,y_test)
print(f'Original Score = {original_score}')

# Count vectorization score = Score = 0.7906976744186046
# Tfidf score = Score = 0.7933554817275748

#We see Tfidf gives the best score..so tfidf will be used for vectorization

# Finding the best parameter
# optimize_model  = svm.SVC()
# param = {'C': [0.1,1, 10, 52,100], 'gamma': ('auto','scale'),'kernel': ['linear','rbf', 'poly', 'sigmoid']}
# search = BayesSearchCV(optimize_model,param,scoring="accuracy")
# search = search.fit(x_train,y_train)
# print(search.best_params_)

#best_param = C=1.0,gamma='scale',kernel='rbf'

model = svm.SVC(C=1.0,gamma=1.0,kernel='rbf')
model = model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print(f'model Score = {score}')
test = pd.read_csv("../input/clean-test/clean_test.csv")
#dummy variables
tweet_id =  test["id"]
tweets = test["tweet"]
print(tweets.shape)
tweets = tfidf.transform(tweets)
pred = model.predict(tweets)
new_df = {"id":tweet_id,"target":pred}
new_df = pd.DataFrame(new_df)
new_df.head()
disaster_pred = new_df.to_csv("disaster_pred.csv",index = False)
print(disaster_pred)


 
