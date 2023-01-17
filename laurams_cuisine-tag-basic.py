# import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Import the required libraries 
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd
import json

import seaborn as sns
import matplotlib.pyplot as plt 

import os
print(os.listdir("../input")) 
df_train =pd.read_json('../input/train.json')
df_test =pd.read_json('../input/test.json')
df_train.head()
df_train['seperated_ingredients'] = df_train['ingredients'].apply(','.join)
df_test['seperated_ingredients'] = df_test['ingredients'].apply(','.join)
df_train.shape
counter_cui = Counter(df_train.cuisine.values.tolist()).most_common()

df_tmp = pd.DataFrame(counter_cui, columns=['cuisine','count']) 
sns.barplot(y='cuisine', x="count", data=df_tmp);
# liste ingredient concatenate
liste_ingredient = [item for sublist in df_train.ingredients.tolist() for item in sublist]
counter_ing = Counter(liste_ingredient).most_common(10)

# top ingredient
df_tmp = pd.DataFrame(counter_ing, columns=['ing','count']) 
sns.barplot(y='ing', x="count", data=df_tmp );
print("nb mean ing in recipe :", int(df_train.ingredients.str.len().mean() ))
print("nb min ing in recipe :", df_train.ingredients.str.len().min())
print("nb max ing in recipe :", df_train.ingredients.str.len().max())
df_train[df_train.ingredients.str.len()==1]
sns.countplot(df_train.ingredients.str.len() );
plt.gcf().set_size_inches(16,8)
plt.title('Number of ingredients distribution')
new_list=[]
for i in df_train.ingredients.tolist(): 
    new_list.append(','.join(i)) 
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(tokenizer=lambda x: [i.strip() for i in x.split(',')])
corpus = new_list 
X_train_vectorized = vectorizer.fit_transform(corpus)   

corpus_test = [','.join(i) for i in df_test.ingredients.tolist()]
X_test_vectorized = vectorizer.transform(corpus_test)

count=dict(zip(vectorizer.get_feature_names(), X_train_vectorized.sum(axis=0).tolist()[0]))
count=pd.DataFrame(list(count.items()),columns=['Ingredient','Count'])
count.sort_values('Count',ascending=False).head(10)
from sklearn.preprocessing import LabelEncoder 
encoder = LabelEncoder()
y_transformed = encoder.fit_transform(df_train.cuisine)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_vectorized, y_transformed , random_state = 42)

from sklearn.linear_model import LogisticRegression

clf1 = LogisticRegression(C=10,dual=False)
clf1.fit(X_train , y_train)
clf1.score(X_test, y_test)
y_test_count = clf1.predict(X_test_vectorized)

y_predicted_final = encoder.inverse_transform(y_test_count)

predictions = pd.DataFrame({'cuisine' : y_predicted_final , 'id' : df_test.id })
predictions = predictions[[ 'id' , 'cuisine']]
predictions.to_csv('submit.csv', index = False)


new_list=[]
for i in df_train.ingredients.tolist(): 
    new_list.append(','.join(i)) 
    
bigram_vectorizer = CountVectorizer(tokenizer=lambda x: [i.strip() for i in x.split(',')],ngram_range=(2, 2),  token_pattern=r'\b\w+\b', min_df=1)

corpus = new_list 
X_train_vectorized = bigram_vectorizer.fit_transform(corpus)     

count=dict(zip(bigram_vectorizer.get_feature_names(), X_train_vectorized.sum(axis=0).tolist()[0]))
count=pd.DataFrame(list(count.items()),columns=['Ingredient','Count'])
def check_important_ing_by_cuisine_bigram(cuisine_name,bigram_vectorizer):
    liste_index=list(np.where(df_train.cuisine==cuisine_name)[0])
    # create dataframe average tf idf weight + feature name
    df_weight = pd.DataFrame(X_train_vectorized[liste_index ,:].mean(axis=0)).T 
    df_name =pd.DataFrame(bigram_vectorizer.get_feature_names())
    # concat weight + name
    df_mean =pd.concat([df_weight, df_name], axis=1  )
    df_mean.columns=['weight','name']
    # sort output
    return df_mean.sort_values('weight', ascending=False).reset_index(drop=True)

check_important_ing_by_cuisine_bigram('chinese',bigram_vectorizer).iloc[:10]
count.sort_values('Count',ascending=False).head(10)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_vectorized, y_transformed , random_state = 42)

from sklearn.linear_model import LogisticRegression

clf1 = LogisticRegression(C=10,dual=False)
clf1.fit(X_train , y_train)
clf1.score(X_test, y_test)
# credits https://buhrmann.github.io/tfidf-analysis.html
# https://www.kaggle.com/edchen/tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(tokenizer=lambda x: [i.strip() for i in x.split(',')]).fit(df_train['seperated_ingredients'].values)
X_train_vectorized = vect.transform(df_train['seperated_ingredients'].values)
X_train_vectorized = X_train_vectorized.astype('float')
Result_transformed = vect.transform(df_test['seperated_ingredients'].values)
Result_transformed = Result_transformed.astype('float')
def check_important_ing_by_cuisine(cuisine_name):
    liste_index=list(np.where(df_train.cuisine==cuisine_name)[0])
    # create dataframe average tf idf weight + feature name
    df_weight = pd.DataFrame(X_train_vectorized[liste_index ,:].mean(axis=0)).T 
    df_name =pd.DataFrame(vect.get_feature_names())
    # concat weight + name
    df_mean =pd.concat([df_weight, df_name], axis=1  )
    df_mean.columns=['weight','name']
    # sort output
    return df_mean.sort_values('weight', ascending=False).reset_index(drop=True)

def plot_top_tfidf(cuisine_name, top_n=25):
    df_tmp = check_important_ing_by_cuisine(cuisine_name).loc[:top_n] 
    fig = plt.figure(figsize=(8, 6), facecolor="w")
    sns.barplot(y='name', x="weight", data=df_tmp,  color='#3F5D7D');
    plt.xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
    plt.title("label = " + str(cuisine_name), fontsize=16)
plot_top_tfidf('french', top_n=15)
plot_top_tfidf('british', top_n=15) 
plot_top_tfidf('chinese', top_n=15) 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_vectorized, y_transformed , random_state = 42)

from sklearn.linear_model import LogisticRegression

clf1 = LogisticRegression(C=10,dual=False)
clf1.fit(X_train , y_train)
clf1.score(X_test, y_test)
def create_vocab(df_train):
    liste_all_ing =[]
    for i,j in df_train.ingredients.iteritems():
        liste_all_ing+=j 
    liste_unique_train_lower = [x.lower() for x in liste_all_ing]
    return list(set(liste_unique_train_lower))

liste_unique_train = create_vocab(df_train)
liste_unique_test = create_vocab(df_test)
liste_all = list(set(liste_unique_train+liste_unique_test))

print("num ingredients diff :", len(liste_unique_train)) 
print("num ingredients diff :", len(liste_all))