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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import nltk 
import re
from textblob import TextBlob
import seaborn as sns
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') 
from matplotlib import style;
style.use('ggplot')

df1 = pd.read_csv("/kaggle/input/medicine-side-effects-analysis/train.csv")
df1.shape
dfTest = pd.read_csv("/kaggle/input/medicine-side-effects-analysis/test.csv")
dfTest.shape
df1.columns
str(df1)
df1.info()
df1.describe()
df1.head(20)
df1.isnull().sum()
drugs = pd.value_counts(df1.drugName)
drugs.head(20)
drugs[drugs == drugs.min()].head(20)
fig = plt.figure(figsize = (80,25))
sns.countplot(drugs)
df2 = df1[['Id','condition','review','rating']].copy()
df2Test = dfTest[['Id','condition','review']].copy()
df2.head(5)
df2Test.head(5)
dfUniqueCond = pd.DataFrame(df2.condition.unique(),columns=['condition'])
dfUniqueCond = dfUniqueCond.dropna(subset=['condition'])
dfUniqueCond['cleanCondition'] = dfUniqueCond['condition'].apply(lambda x: " ".join(x.lower() for x in x.split()))
dfUniqueCond['cleanCondition'] = dfUniqueCond['cleanCondition'].str.replace('\d+', '')
dfCondList = []
for index, row in dfUniqueCond.iterrows():
#     print(row['cleanCondition']) 
    text = row["cleanCondition"].split()
    for i in range(len(text)): 
        dfCondList.append(text[i])
dfCondList.extend(['side','effect','year','medication','medicine'])
dfUniqueCond = pd.DataFrame(df2Test.condition.unique(),columns=['condition'])
dfUniqueCond = dfUniqueCond.dropna(subset=['condition'])
dfUniqueCond['cleanCondition'] = dfUniqueCond['condition'].apply(lambda x: " ".join(x.lower() for x in x.split()))
dfUniqueCond['cleanCondition'] = dfUniqueCond['cleanCondition'].str.replace('\d+', '')
for index, row in dfUniqueCond.iterrows():
#     print(row['cleanCondition']) 
    text = row["cleanCondition"].split()
    for i in range(len(text)): 
        dfCondList.append(text[i])
dfCondList
stopwords = stopwords.words('english')
df2['cleanReview'] = df2['review'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))  
df2['cleanReview'] = df2['review'].apply(lambda x: ' '.join([item for item in x.split() if item not in dfCondList]))  
   
df2['cleanReview'] = df2['cleanReview'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df2['cleanReview'] = df2['cleanReview'].apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))

df2['cleanReview'] = df2['cleanReview'].str.replace('[^\w\s]', '')
df2['cleanReview'] = df2['cleanReview'].str.replace('\d+', '')
df2['cleanReview'] = df2['cleanReview'].str.replace(r'\b\w{1,3}\b', '')
                     
df2['cleanReview'] = df2['cleanReview'].apply(lambda x: " ".join(x.strip() for x in x.split()))
#df2['cleanReview'] = df2['cleanReview'].apply(lemmatize_text)
df_all = df2
#Test data
# remove stopwords from review
df2Test['cleanReview'] = df2Test['review'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))  
df2Test['cleanReview'] = df2Test['review'].apply(lambda x: ' '.join([item for item in x.split() if item not in dfCondList]))  
   
df2Test['cleanReview'] = df2Test['cleanReview'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df2Test['cleanReview'] = df2Test['cleanReview'].apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))

df2Test['cleanReview'] = df2Test['cleanReview'].str.replace('[^\w\s]', '')
df2Test['cleanReview'] = df2Test['cleanReview'].str.replace('\d+', '')
df2Test['cleanReview'] = df2Test['cleanReview'].str.replace(r'\b\w{1,3}\b', '')
                     
df2Test['cleanReview'] = df2Test['cleanReview'].apply(lambda x: " ".join(x.strip() for x in x.split()))
#df2Test['cleanReview'] = df2['cleanReview'].apply(lemmatize_text)
############### Word Cloud                 ##########################
 
df_positive = df2[df2['rating'].isin([8,9,10])]
df_neutral  = df2[df2['rating'].isin([5,6,7])]
df_negative = df2[df2['rating'].isin([1,2,3,4])]

pos_string = []
neg_string = []
neutr_string = []
for t in df_positive.cleanReview:
    pos_string.append(t)
pos_string = pd.Series(pos_string).str.cat(sep=' ')
for t in df_negative.cleanReview:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')
for t in df_neutral.cleanReview:
    neutr_string.append(t)
neutr_string = pd.Series(neutr_string).str.cat(sep=' ')
######### Positive Word Cloud  ##################
wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(pos_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show() 
######### Negative Word Cloud  ##################
wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show() 
######### Neutral  Word Cloud  ##################
wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neutr_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
#df_all.columns 
y = df_all['rating']  
smote=SMOTE()
df_all_dup = df_all.head(6000)
cv=CountVectorizer()
word_count_vector=cv.fit_transform(df_all_dup['cleanReview'])
feature_names=cv.get_feature_names()
############## Tfidf Only   ###################################

df_all.shape
df_all_dup = df_all
df_all_dup.shape
#df_all= df_all_dup.head(30000)
cv=CountVectorizer()
word_count_vector=cv.fit_transform(df_all['cleanReview'])
tf=TfidfVectorizer(min_df=4,max_features=13000)
#tf=TfidfVectorizer()
text_tf= tf.fit_transform(df_all['cleanReview'])
feature_names=cv.get_feature_names()
text_tf.shape
X = text_tf.toarray()
X = pd.DataFrame(X)
X.shape
df2TestCopy = df2Test
df2TestCopy.shape
#df2Test = df2Test.head(10000)
tf=TfidfVectorizer(min_df=2,max_features=13000)
#tf=TfidfVectorizer()
X_TestData= tf.fit_transform(df2Test['cleanReview'])
X_TestData = X_TestData.toarray()
X_TestData = pd.DataFrame(X_TestData)
X_TestData.shape
one_hot_encoded_train = pd.get_dummies(X)
one_hot_encoded_test = pd.get_dummies(X_TestData)
del X
del X_TestData 
one_hot_encoded_train.shape
one_hot_encoded_test.shape
final_train, final_test = one_hot_encoded_train.align(one_hot_encoded_test,join='left',axis=1)
y_sm = df_all_dup['rating']
X_sm =  final_train
X_Test = final_test
X_Test.apply(lambda x: x.count(), axis=1)
X_Test.isnull().count()
X_Test.head(5)
X_Test = X_Test.fillna(0)
X_Test.replace(np.nan,0)
X_train, X_test, y_train, y_test = train_test_split(text_tf, y, test_size=0.2, random_state=1)
X_sm, y_sm = smote.fit_sample(X_train,y_train)
clf = LinearSVC().fit(X_sm, y_sm)
#clf = LinearSVC().fit(X_train, y_train)  
pickle.dump(clf,open('model.pkl','wb'))
predicted= clf.predict(X_test)
print("LinearSVC Test Accuracy:",metrics.accuracy_score(y_test, predicted))
predicted= clf.predict(X_sm)
print("LinearSVC Test Data Accuracy:",metrics.accuracy_score(y_sm, predicted))
cm = confusion_matrix(predicted, y_sm)
print("Train Confusion Matrix:  \n", cm)
print("                    Train Classification Report \n",classification_report(predicted, y_sm))
output = pd.DataFrame(df2Test['Id'])
predicted = pd.DataFrame(predicted,columns = ['Predicted'])
output = output.join(predicted)
output.to_csv('C:\\Users\\Medha\\Desktop\\Medha\\ExcelR\\Live Projects\\Medicine Side Effects\\DataSet\\Ouput.csv', index = False)
#del output










