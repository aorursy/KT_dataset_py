# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


df = pd.read_csv("/kaggle/input/amazon-fine-food-reviews/Reviews.csv")
df.head()
df.keys()
df.shape
df.info()
df.describe()
df.Score.value_counts()
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.countplot(x='Score', data=df, palette='RdBu')
plt.xlabel('Score (Rating)')
plt.show()
    
#copying the original dataframe to 'temp_df'.
temp_df = df[['UserId','HelpfulnessNumerator','HelpfulnessDenominator', 'Summary', 'Text','Score']].copy()

#Adding new features to dataframe.
temp_df["Sentiment"] = temp_df["Score"].apply(lambda score: "positive" if score > 3 else \
                                              ("negative" if score < 3 else "not defined"))
temp_df["Usefulness"] = (temp_df["HelpfulnessNumerator"]/temp_df["HelpfulnessDenominator"]).apply\
(lambda n: ">75%" if n > 0.75 else ("<25%" if n < 0.25 else ("25-75%" if n >= 0.25 and\
                                                                        n <= 0.75 else "useless")))

temp_df.loc[temp_df.HelpfulnessDenominator == 0, 'Usefulness'] = ["useless"]
# Removing all rows where 'Score' is equal to 3
#temp_df = temp_df[temp_df.Score != 3]
#Lets now observe the shape of our new dataframe.
temp_df.shape
temp_df.describe()
temp_df.info()
#Lets view the dataframe when Score=5
temp_df[temp_df.Score == 5].head(10)
sns.countplot(x='Sentiment', order=["positive", "negative"], data=temp_df, palette='RdBu')
plt.xlabel('Sentiment')
plt.show()
temp_df.Sentiment.value_counts()
pos = temp_df.loc[temp_df['Sentiment'] == 'positive']
pos = pos[0:25000]

neg = temp_df.loc[temp_df['Sentiment'] == 'negative']
neg = neg[0:25000]
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import string
import matplotlib.pyplot as plt


def create_Word_Corpus(temp):
    words_corpus = ''
    for val in temp["Summary"]:
        text = str(val).lower()
        tokens = []
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        for words in tokens:
            words_corpus = words_corpus + ' ' + words
    return words_corpus
        
# Generate a word cloud image
pos_wordcloud = WordCloud(width=900, height=500).generate(create_Word_Corpus(pos))
neg_wordcloud = WordCloud(width=900, height=500).generate(create_Word_Corpus(neg))
# Plot cloud
def plot_Cloud(wordCloud):
    plt.figure( figsize=(20,10), facecolor='w')
    plt.imshow(wordCloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    plt.savefig('wordclouds.png', facecolor='w', bbox_inches='tight')
#Visuallizing popular positive words
plot_Cloud(pos_wordcloud)
#Visuallizing popular negative words
plot_Cloud(neg_wordcloud)
#Checking the value count for 'Usefulness'
temp_df.Usefulness.value_counts()
sns.countplot(x='Usefulness', order=['useless', '>75%', '25-75%', '<25%'], data=temp_df, palette='RdBu')
plt.xlabel('Usefulness')
plt.show()
temp_df[temp_df.Score==5].Usefulness.value_counts()
temp_df[temp_df.Score==2].Usefulness.value_counts()
sns.countplot(x='Sentiment', hue='Usefulness', order=["positive", "negative"], \
              hue_order=['>75%', '25-75%', '<25%'], data=temp_df, palette='RdBu')
plt.xlabel('Sentiment')
plt.show()
temp_df["text_word_count"] = temp_df["Text"].apply(lambda text: len(text.split()))
temp_df.head()
temp_df[temp_df.Score==5].text_word_count.median()
temp_df[temp_df.Score==4].text_word_count.median()
temp_df[temp_df.Score==3].text_word_count.median()
temp_df[temp_df.Score==2].text_word_count.median()
temp_df[temp_df.Score==1].text_word_count.median()
sns.boxplot(x='Score',y='text_word_count', data=temp_df, palette='RdBu', showfliers=False)
plt.show()
sns.violinplot(x='Usefulness', y='text_word_count', order=[">75%", "<25%"], \
               data=temp_df, palette='RdBu')
plt.ylim(-50, 400)
plt.show()
x = temp_df.UserId.value_counts()
x.to_dict()
print("converted Series to dictionary")
temp_df["reviewer_freq"] = temp_df["UserId"].apply(lambda counts: "Frequent (>50 reviews)" \
                                                                 if x[counts]>50 else "Not Frequent (1-50)")
temp_df.head()
ax = sns.countplot(x='Score', hue='reviewer_freq', data=temp_df, palette='RdBu')
ax.set_xlabel('Score (Rating)')
plt.show()
y = temp_df[temp_df.reviewer_freq=="Frequent (>50 reviews)"].Score.value_counts()
z = temp_df[temp_df.reviewer_freq=="Not Frequent (1-50)"].Score.value_counts()

tot_y = y.sum()

y = (y/tot_y)*100

tot_z = z.sum()

z = (z/tot_z)*100

ax1 = plt.subplot(121)
y.plot(kind="bar",ax=ax1)
plt.xlabel("Score")
plt.ylabel("Percentage")
plt.title("Frequent (>50 reviews) Distribution")

ax2 = plt.subplot(122)
z.plot(kind="bar",ax=ax2)
plt.xlabel("Score")
plt.ylabel("Percentage")
plt.title("Not Frequent (1-50) Distribution")
plt.show()
sns.countplot(x='Usefulness', order=['useless', '>75%', '25-75%', '<25%'], \
              hue='reviewer_freq', data=temp_df, palette='RdBu')
plt.xlabel('Helpfulness')
plt.show()
sns.violinplot(x='reviewer_freq', y='text_word_count',  \
               data=temp_df, palette='RdBu')
plt.xlabel('Frequency of Reviewer')
plt.ylim(-50, 400)
plt.show()
#Let's import pandas to read the csv file.
import pandas as pd
dataset = pd.read_csv("/kaggle/input/amazon-fine-food-reviews/Reviews.csv")
dataset.head()
#Here we will sort the dataframe according to 'Time' feature
dataset.sort_values(['Time'], ascending=True, inplace=True)
dataset.head()
#Dropping the unwanted columns from our data frame.
dataset.drop("Id", inplace=True, axis=1)
dataset.drop("ProductId", inplace=True, axis=1)
dataset.drop("ProfileName", inplace=True, axis=1)
dataset.drop("HelpfulnessNumerator", inplace=True, axis=1)
dataset.drop("HelpfulnessDenominator", inplace=True, axis=1)
dataset.drop("Time", inplace=True, axis=1)
dataset.head()
#Make all 'Score' less than 3 equal to -ve class and 
# 'Score' greater than 3 equal to +ve class.
dataset.loc[dataset['Score']<3, 'Score'] = [0]
dataset.loc[dataset['Score']>3, 'Score'] = [1]
dataset.head()
#import numpy as np
#from sklearn.model_selection import train_test_split
#train, test = train_test_split(dataset, test_size = 0.3)

total_size=len(dataset)

train_size=int(0.70*total_size)

#training dataset
train=dataset.head(train_size)
#test dataset
test=dataset.tail(total_size - train_size)
train.Score.value_counts()
test.Score.value_counts()
# Removing all rows where 'Score' is equal to 3
train = train[train.Score != 3]
test = test[test.Score != 3]
print(train.shape)
print(test.shape)
train['Score'].value_counts()
test.Score.value_counts()
#Taking the 'Text' & 'Summary' column in seperate list for further 
#text preprocessing.
lst_text = train['Text'].tolist()
lst_summary = train['Summary'].tolist()
test_text = test['Text'].tolist()
#Converting the whole list to lower-case.
lst_text = [str(item).lower() for item in lst_text]
lst_summary = [str(item).lower() for item in lst_summary]
test_text = [str(item).lower() for item in test_text]
#Lets now remove all HTML tags from the list of strings.
import re
def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

for i in range(len(lst_text)):
    lst_text[i] = striphtml(lst_text[i])
    lst_summary[i] = striphtml(lst_summary[i])
for i in range(len(test_text)):
    test_text[i] = striphtml(test_text[i])
lst_text[0:5]
#Now we will remove all special characters from the strings.
for i in range(len(lst_text)):
    lst_text[i] = re.sub(r'[^A-Za-z]+', ' ', lst_text[i])
    lst_summary[i] = re.sub(r'[^A-Za-z]+', ' ', lst_summary[i])
for i in range(len(test_text)):
    test_text[i] = re.sub(r'[^A-Za-z]+', ' ', test_text[i])
lst_text[0:5]
#Removing Stop Words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#word_tokenize accepts a string as an input, not a file.
stop_words = set(stopwords.words('english'))
for i in range(len(lst_text)):
    text_filtered = []
    summary_filtered = []
    text_word_tokens = []
    summary_word_tokens = []
    text_word_tokens = lst_text[i].split()
    summary_word_tokens = lst_summary[i].split()
    for r in text_word_tokens:
        if not r in stop_words:
            text_filtered.append(r)
    lst_text[i] = ' '.join(text_filtered)
    for r in summary_word_tokens:
        if not r in stop_words:
            summary_filtered.append(r)
    lst_summary[i] = ' '.join(summary_filtered)
for i in range(len(test_text)):
    text_filtered = []
    text_word_tokens = []
    text_word_tokens = test_text[i].split()
    for r in text_word_tokens:
        if not r in stop_words:
            text_filtered.append(r)
    test_text[i] = ' '.join(text_filtered)
#Lets now stem each word.
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
for i in range(len(lst_text)):
    text_filtered = []
    summary_filtered = []
    text_word_tokens = []
    summary_word_tokens = []
    text_word_tokens = lst_text[i].split()
    summary_word_tokens = lst_summary[i].split()
    for r in text_word_tokens:
        text_filtered.append(str(stemmer.stem(r)))
    lst_text[i] = ' '.join(text_filtered)
    for r in summary_word_tokens:
        summary_filtered.append(str(stemmer.stem(r)))
    lst_summary[i] = ' '.join(summary_filtered)
for i in range(len(test_text)):
    text_filtered = []
    text_word_tokens = []
    text_word_tokens = test_text[i].split()
    for r in text_word_tokens:
        if not r in stop_words:
            text_filtered.append(str(stemmer.stem(r)))
    test_text[i] = ' '.join(text_filtered)
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vect = CountVectorizer()
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
X_train_dtm = vect.fit_transform(lst_text)
# Numpy arrays are easy to work with, so convert the result to an 
# array
#train_data_features = train_data_features.toarray()
# examine the document-term matrix
X_train_dtm
# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(test_text)
X_test_dtm
# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
# train the model using X_train_dtm (timing it with an IPython 
#"magic command")
%time nb.fit(X_train_dtm, train.Score)
# make class predictions for X_test_dtm
y_pred_class_nb = nb.predict(X_test_dtm)
# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(test.Score, y_pred_class_nb)
# print the confusion matrix
con_metrics_nb = metrics.confusion_matrix(test.Score, y_pred_class_nb)
con_metrics_nb
#ploting heatmap for confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(con_metrics_nb, annot=True, fmt='d')
plt.title("Confusion Matrix: Naive Bayes")
plt.show()
#Checking Precision, Recall and F1 Score
print(metrics.classification_report(test.Score, y_pred_class_nb))
# calculate AUC - Method 1
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(test.Score, y_pred_class_nb)
print(metrics.auc(false_positive_rate, true_positive_rate))
# calculate AUC _ Method - 2
auc_nb = metrics.roc_auc_score(test.Score, y_pred_class_nb)
print(auc_nb)
#Plotting Area Under the Curve
plt.plot(false_positive_rate,true_positive_rate,label="Naive Bayes, AUC="+str(auc_nb))
plt.plot([0,1],[0,1],'r--')
plt.title('ROC curve: Naive Bayes')
plt.legend(loc='lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# import and instantiate a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg_l2 = LogisticRegression()
# train the model using X_train_dtm
%time logreg_l2.fit(X_train_dtm, train.Score)
# make class predictions for X_test_dtm
y_pred_class_l2 = logreg_l2.predict(X_test_dtm)
# calculate accuracy
metrics.accuracy_score(test.Score, y_pred_class_l2)
# print the confusion matrix
con_metrics_l2 = metrics.confusion_matrix(test.Score, y_pred_class_l2)
con_metrics_l2
#ploting heatmap for confusion matrix
sns.heatmap(con_metrics_l2, annot=True, fmt='d')
plt.title("Confusion Matrix: Logistic Regression with L2 Regularizor")
plt.show()
