import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import itertools
import warnings
import string
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

warnings.filterwarnings('ignore')
review = pd.read_csv("../input/scotch_review.csv")
review.shape
review.head()
review.rename(columns={'Unnamed: 0':'S No'},inplace=True)
review.head()

review.info()
review['category'].unique()
plt.figure(figsize=(8,8))
sns.distplot(review['review.point'],color='red')
plt.title('Distribution of review points over 5 categories')
plt.xlabel('Points')
plt.ylabel('Freq')
plt.figure(figsize=(8,8))
p=sns.boxplot(review['category'],review['review.point'],palette=sns.color_palette(palette='Set3'))
p.set_xticklabels(p.get_xticklabels(),rotation=90)
plt.title("Boxplot of Review Vs Category",size=16)
plt.xlabel('Category',size=10)
plt.ylabel('Review Score',size=10)
review_length = pd.DataFrame(review.groupby('category')['description'].count().sort_values(ascending=False))
review_length.reset_index(inplace = True)
review_length.head()
plt.figure(figsize=(10,6))
sns.barplot(review_length.category, review_length.description,palette=sns.color_palette(palette='cool'))
plt.xticks(rotation='vertical')
plt.ylabel('Number of Words', fontsize=12)
plt.xlabel('Category Name', fontsize=12)
plt.title('Total Review Count by Category')
plt.show()
##Code inspiration - https://www.kaggle.com/sudalairajkumar/simple-feature-engg-notebook-spooky-author

grouped_df = review.groupby('category')
for name, group in grouped_df:
    print("Author name : ", name)
    cnt = 0
    for ind, row in group.iterrows():
        print(row["description"])
        print("\n")
        cnt += 1
        if cnt == 2:
            break
    
    print("\n\n")
### Create a list for each category

SMS=review[review.category=="Single Grain Whisky"]['description'].values
BSW=review[review.category=='Blended Scotch Whisky']['description'].values
BMSW=review[review.category=='Blended Malt Scotch Whisky']['description'].values
SGW=review[review.category=='Single Grain Whisky']['description'].values
GSW=review[review.category=='Grain Scotch Whisky']['description'].values
#Creating a function for worcloud 
#Code inspiration:https://www.kaggle.com/duttadebadri/analysing-the-olympics-for-last-120-yrs/notebook & Nick Brooks from comments ..

from wordcloud import WordCloud,STOPWORDS
stopwords=set(STOPWORDS)
def show_wordcloud(data,title=None):
    wc=WordCloud(background_color="black", max_words=10000,stopwords=STOPWORDS, max_font_size= 40)
    wc.generate(" ".join(data))
    fig=fig = plt.figure(figsize=[8,5], dpi=80)
    plt.axis('off')
    if title:
        fig.suptitle(title,fontsize=16)
        fig.subplots_adjust(top=1)
        plt.imshow(wc.recolor( colormap= 'Pastel2' , random_state=17), alpha=1,interpolation='bilinear')
        plt.show()
        

        

show_wordcloud(SMS,title="Wordcloud for Single Grain Whisky")
show_wordcloud(BSW,title="Wordcloud for Blended Scotch Whisky")
show_wordcloud(BMSW,title="Wordcloud for Blended Malt Scotch Whisky")
show_wordcloud(SGW,title="Wordcloud for Single Grain Whisky")
show_wordcloud(GSW,title="Wordcloud for Grain Scotch Whisky")
### Creating the features 

review['count_word']=review["description"].apply(lambda x: len(str(x).split()))
review['count_stopwords']=review['description'].apply(lambda x:len([w for w in str(x).lower().split() if w in stopwords]))
review['count_punct']=review['description'].apply(lambda x:len([p for p in str(x) if p in string.punctuation]))
plt.figure(figsize=(15,12))

plt.subplot(221)
g = sns.boxplot(x=review['category'],y=review['count_word'],palette=sns.color_palette(palette="Set1"))
g.set_title("Distribution of words in each sentences by category", fontsize=15)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.subplot(222)
g1 = sns.boxplot(x=review['category'],y=review['count_stopwords'],palette=sns.color_palette(palette="dark"))
g1.set_title("Distribution of stopwords in each sentences by category", fontsize=15)
g1.set_xticklabels(g1.get_xticklabels(),rotation=90)
g1.set_xlabel("")
g1.set_ylabel("Count", fontsize=12)

plt.subplots_adjust(wspace = 1, hspace = 0.6,top = 0.9)
review.head()
### Code inspiration - 1.https://www.kaggle.com/juanumusic/to-predict-or-not-to-predict-python-tutorial - Very neat and exploratory kernel.Do check out.
### 2.https://stackoverflow.com/questions/47557417/understanding-text-feature-extraction-tfidfvectorizer-in-python-scikit-learn


### Creating count vectoriser:
vect=CountVectorizer(ngram_range=(1,1),analyzer='word',stop_words=stopwords,token_pattern=r'\w{1,}')
review_vect = vect.fit_transform(review['description'])
review_vect.get_shape()
tf_idf=TfidfVectorizer(ngram_range=(1,1),stop_words=stopwords,analyzer='word',token_pattern=r'\w{1,}')
review_tfidf=tf_idf.fit_transform(review['description'])
review_tfidf.get_shape()
### Split into train and test data:
## Count Vectorizer model:
x_train_vec,x_test_vec,y_train_vec,y_test_vec=train_test_split(review_vect,review['category'],train_size=0.8,random_state=100)
### Applying the Multinomial Logistic Regression :

logit=LogisticRegression(class_weight='balanced',multi_class='multinomial',solver='lbfgs')
logit.fit(x_train_vec,y_train_vec)
logit.get_params()

### Test over the data:
predictions=logit.predict(x_test_vec)

##Checking the accuracy:
print("Accuracy Score with count Vectorizer: {:0.3f}".format(accuracy_score(predictions,y_test_vec)))

conf_matrix_vec=confusion_matrix(y_test_vec,predictions)

### Confusion Matrix: 

### Code Source - http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if 'd' else '0'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
np.set_printoptions(precision=2)
plt.figure(figsize=(8,8))
plot_confusion_matrix(conf_matrix_vec, classes=['Blended Scotch Whisky', 'Single Malt Scotch','Blended Malt Scotch Whisky', 'Grain Scotch Whisky','Single Grain Whisky'],
                      title='Confusion matrix for Count Vectorizer Model')
plt.show()
## We implement similar approach with TFIDF.
# Split into train and test:

x_train_tf,x_test_tf,y_train_tf,y_test_tf=train_test_split(review_tfidf,review['category'],train_size=0.8,random_state=100)

### Applying the Multinomial Logistic Regression :

logit=LogisticRegression(class_weight='balanced',multi_class='multinomial',solver='lbfgs')
logit.fit(x_train_tf,y_train_tf)
#logit.get_params()

### Test over the data:
predictions=logit.predict(x_test_tf)

##Checking the accuracy:
print("Accuracy Score with count Vectorizer: {:0.3f}".format(accuracy_score(predictions,y_test_tf)))

conf_matrix_vec=confusion_matrix(y_test_tf,predictions)


np.set_printoptions(precision=2)
plt.figure(figsize=(8,8))
plot_confusion_matrix(conf_matrix_vec, classes=['Blended Scotch Whisky', 'Single Malt Scotch','Blended Malt Scotch Whisky', 'Grain Scotch Whisky','Single Grain Whisky'],
                      title='Confusion matrix for TFIDF Model')
plt.show()