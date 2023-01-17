import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv("../input/ireland-historical-news/irishtimes-date-text.csv")
data.shape
data.isna().sum()
data.duplicated().sum()
data.head()
data.drop_duplicates(inplace=True) 
data.shape
year = [] 

month = [] 

day = [] 



dates = data.publish_date.values



for date in dates:

    str_date = list(str(date))

    year.append(int("".join(str_date[0:4]))) 

    month.append(int("".join(str_date[4:6])))

    day.append(int("".join(str_date[6:8])))
data['year'] = year

data['month'] = month

data['day'] = day



data.drop(['publish_date'] , axis=1,inplace=True) 
data.head()
print('Unique Headlines Categories: {}'.format(len(data.headline_category.unique())))
set([category for category in data.headline_category if "." not in category] ) 
data.headline_category = data.headline_category.apply(lambda x: x.split(".")[0]) 
plt.figure(figsize=(10,5))

ax = sns.countplot(data.headline_category) 
from nltk.corpus import stopwords 

from nltk.tokenize import WordPunctTokenizer

from string import punctuation

from nltk.stem import WordNetLemmatizer

import regex



wordnet_lemmatizer = WordNetLemmatizer()



stop = stopwords.words('english')



for punct in punctuation:

    stop.append(punct)



def filter_text(text, stop_words):

    word_tokens = WordPunctTokenizer().tokenize(text.lower())

    filtered_text = [regex.sub(u'\p{^Latin}', u'', w) for w in word_tokens if w.isalpha()]

    filtered_text = [wordnet_lemmatizer.lemmatize(w, pos="v") for w in filtered_text if not w in stop_words] 

    return " ".join(filtered_text)
data["filtered_text"] = data.headline_text.apply(lambda x : filter_text(x, stop)) 
data.head()
plt.figure(figsize=(10,5))

ax = sns.lineplot(x=data.year.value_counts().index.values,y=data.year.value_counts().values)

ax = plt.title('Number of Published News by Year')
plt.figure(figsize=(10,5))

ax = sns.lineplot(x=data.month.value_counts().index.values,y=data.month.value_counts().values)

ax = plt.title('Number of Published News by Month')
plt.figure(figsize=(10,5))

ax = sns.lineplot(x=data.day.value_counts().index.values,y=data.day.value_counts().values)

ax = plt.title('Number of Published News by Day')
from wordcloud import WordCloud



def make_wordcloud(words,title):

    cloud = WordCloud(width=1920, height=1080,max_font_size=200, max_words=300, background_color="white").generate(words)

    plt.figure(figsize=(20,20))

    plt.imshow(cloud, interpolation="gaussian")

    plt.axis("off") 

    plt.title(title, fontsize=60)

    plt.show()
all_text = " ".join(data[data.headline_category == "news"].filtered_text) 

make_wordcloud(all_text, "News") 
all_text = " ".join(data[data.headline_category == "culture"].filtered_text) 

make_wordcloud(all_text, "Culture")
all_text = " ".join(data[data.headline_category == "opinion"].filtered_text) 

make_wordcloud(all_text, "Opinion")
all_text = " ".join(data[data.headline_category == "business"].filtered_text) 

make_wordcloud(all_text, "Business")
all_text = " ".join(data[data.headline_category == "sport"].filtered_text) 

make_wordcloud(all_text, "Sport")
all_text = " ".join(data[data.headline_category == "lifestyle"].filtered_text) 

make_wordcloud(all_text, "Lifestyle")
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf = TfidfVectorizer(lowercase=False)

ml_data = tfidf.fit_transform(data['filtered_text'])
ml_data.shape
data['classification'] = data['headline_category'].replace(['news','culture','opinion','business','sport','lifestyle'],[0,1,2,3,4,5])
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(ml_data,data['classification'], stratify=data['classification'], test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix



model = LogisticRegression(solver='lbfgs',multi_class='auto', max_iter=1000)

model.fit(x_train,y_train)
predicted = model.predict(x_test)

print("Test score: {:.2f}".format(accuracy_score(y_test,predicted)))

print("Cohen Kappa score: {:.2f}".format(cohen_kappa_score(y_test,predicted)))

plt.figure(figsize=(15,10))

ax = sns.heatmap(confusion_matrix(y_test,predicted),annot=True)

ax = ax.set(xlabel='Predicted',ylabel='True',title='Confusion Matrix',

            xticklabels=(['news','culture','opinion','business','sport','lifestyle']),

            yticklabels=(['news','culture','opinion','business','sport','lifestyle']))
def get_most_important_words(model,index,category):

    base = {'news':0,'culture':1,'opinion':2,'business':3,'sport':4,'lifestyle':5}

    t=pd.DataFrame(model.coef_[base[category]].T, index=tfidf.get_feature_names()) 

    return pd.concat([t.nlargest(5,0),t.nsmallest(5,0)])
index = tfidf.get_feature_names()
get_most_important_words(model,index,'news')
get_most_important_words(model,index,'culture')
get_most_important_words(model,index,'opinion')
get_most_important_words(model,index,'business')
get_most_important_words(model,index,'sport')
get_most_important_words(model,index,'lifestyle')