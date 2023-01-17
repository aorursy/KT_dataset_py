import numpy as np 
import pandas as pd 
import nltk
import string as s
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS

from sklearn.feature_extraction.text  import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics  import f1_score,accuracy_score
from sklearn.metrics import  confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from lightgbm import LGBMClassifier
train_data=pd.read_csv("/kaggle/input/ag-news-classification-dataset/train.csv",header=0,names=['classid','title','desc'])
test_data=pd.read_csv("/kaggle/input/ag-news-classification-dataset/test.csv",header=0,names=['classid','title','desc'])
train_data.head()
test_data.head()
train_x=train_data.desc[:70000]
test_x=test_data.desc
train_y=train_data.classid[:70000] 
test_y=test_data.classid
df=train_data[:70000]
sns.countplot(df.classid);
from PIL import Image
c_mask = np.array(Image.open("/kaggle/input/masks/comment.png"))
u_mask = np.array(Image.open("/kaggle/input/masks/upvote.png"))
stopwordset= set(STOPWORDS)
morestop={'lt','gt','href','HREF','quot','aspx'}
stopwordset= stopwordset.union(morestop)
world = df.desc[df.classid[df.classid==1].index]
plt.figure(figsize = (15,20)) ;
wordcloud = WordCloud(min_font_size = 3,  max_words = 500 ,background_color="white",contour_width=3, contour_color='firebrick',mask=c_mask, stopwords=stopwordset,colormap='Dark2').generate(" ".join(world))
plt.imshow(wordcloud,interpolation = 'bilinear');
plt.axis("off");
sports = df.desc[df.classid[df.classid==2].index]
plt.figure(figsize = (15,20)) ;
wordcloud = WordCloud(min_font_size = 3,  max_words = 500 ,background_color="white",contour_width=3, contour_color='skyblue',mask=u_mask, stopwords=stopwordset,colormap='Dark2').generate(" ".join(sports))
plt.imshow(wordcloud,interpolation = 'bilinear');
plt.axis("off");
biz = df.desc[df.classid[df.classid==3].index]
plt.figure(figsize = (15,20)) ;
wordcloud = WordCloud(min_font_size = 3,  max_words = 500 ,background_color="white",contour_width=3, contour_color='firebrick',mask=c_mask, stopwords=stopwordset,colormap='Dark2').generate(" ".join(biz))
plt.imshow(wordcloud,interpolation = 'bilinear');
plt.axis("off");
sci = df.desc[df.classid[df.classid==4].index]
plt.figure(figsize = (15,20)) ;
wordcloud = WordCloud(min_font_size = 3,  max_words = 500 ,background_color="white",contour_width=3, contour_color='skyblue',mask=u_mask, stopwords=stopwordset,colormap='Dark2').generate(" ".join(sci))
plt.imshow(wordcloud,interpolation = 'bilinear');
plt.axis("off");
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

train_x=train_x.apply(remove_urls)
test_x=test_x.apply(remove_urls)
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

train_x=train_x.apply(remove_html)
test_x=test_x.apply(remove_html)
def word_tokenize(txt):
    tokens = re.findall("[\w']+", txt)
    return tokens
train_x=train_x.apply(word_tokenize)
test_x=test_x.apply(word_tokenize)
def lowercasing(lst):
    new_lst=[]
    for  i in  lst:
        i=i.lower()
        new_lst.append(i) 
    return new_lst
train_x=train_x.apply(lowercasing)
test_x=test_x.apply(lowercasing)
def remove_stopwords(lst):
    stop=stopwords.words('english')
    new_lst=[]
    for i in lst:
        if i not in stop:
            new_lst.append(i)
    return new_lst

train_x=train_x.apply(remove_stopwords)
test_x=test_x.apply(remove_stopwords)  
def remove_punctuations(lst):
    new_lst=[]
    for i in lst:
        for  j in  s.punctuation:
            i=i.replace(j,'')
        new_lst.append(i)
    return new_lst
train_x=train_x.apply(remove_punctuations) 
test_x=test_x.apply(remove_punctuations)
def remove_numbers(lst):
    nodig_lst=[]
    new_lst=[]

    for i in  lst:
        for j in  s.digits:
            i=i.replace(j,'')
        nodig_lst.append(i)
    for i in  nodig_lst:
        if  i!='':
            new_lst.append(i)
    return new_lst
train_x=train_x.apply(remove_numbers)
test_x=test_x.apply(remove_numbers)
import nltk

def stemming(text):
    porter_stemmer = nltk.PorterStemmer()
    roots = [porter_stemmer.stem(each) for each in text]
    return (roots)

train_x=train_x.apply(stemming)
test_x=test_x.apply(stemming)
lemmatizer=nltk.stem.WordNetLemmatizer()
def lemmatzation(lst):
    new_lst=[]
    for i in lst:
        i=lemmatizer.lemmatize(i)
        new_lst.append(i)
    return new_lst
train_x=train_x.apply(lemmatzation)
test_x=test_x.apply(lemmatzation)
def remove_extrawords(lst):
    stop=['href','lt','gt','ii','iii','ie','quot','com']
    new_lst=[]
    for i in lst:
        if i not in stop:
            new_lst.append(i)
    return new_lst

train_x=train_x.apply(remove_extrawords)
test_x=test_x.apply(remove_extrawords)
train_x=train_x.apply(lambda x: ''.join(i+' ' for i in x))
test_x=test_x.apply(lambda x: ''.join(i+' '  for i in x))
tfidf=TfidfVectorizer(min_df=3)
train_1=tfidf.fit_transform(train_x)
test_1=tfidf.transform(test_x)
print("No. of features extracted")
print(len(tfidf.get_feature_names()))


train_arr=train_1.toarray()
test_arr=test_1.toarray()
NB_MN=MultinomialNB(alpha=0.16)
NB_MN.fit(train_arr,train_y)
pred=NB_MN.predict(test_arr)

print("first 20 actual labels")
print(test_y.tolist()[:20])
print("first 20 predicted labels")
print(pred.tolist()[:20])
def eval_model(y,y_pred):
    print("F1 score of the model")
    print(f1_score(y,y_pred,average='micro'))
    print("Accuracy of the model")
    print(accuracy_score(y,y_pred))
    print("Accuracy of the model in percentage")
    print(round(accuracy_score(y,y_pred)*100,3),"%")
def confusion_mat(color):
    cof=confusion_matrix(test_y, pred)
    cof=pd.DataFrame(cof, index=[i for i in range(1,5)], columns=[i for i in range(1,5)])
    sns.set(font_scale=1.5)
    plt.figure(figsize=(8,8));

    sns.heatmap(cof, cmap=color,linewidths=1, annot=True,square=True, fmt='d', cbar=False,xticklabels=['World','Sports','Business','Science'],yticklabels=['World','Sports','Business','Science']);
    plt.xlabel("Predicted Classes");
    plt.ylabel("Actual Classes");
    
eval_model(test_y,pred)
    
a=round(accuracy_score(test_y,pred)*100,3)
confusion_mat('YlGnBu')
DT=DecisionTreeClassifier()
DT.fit(train_arr,train_y)
pred=DT.predict(test_arr)

print("first 20 actual labels")
print(test_y.tolist()[:20])
print("first 20 predicted labels")
print(pred.tolist()[:20])
eval_model(test_y,pred)
    
b=round(accuracy_score(test_y,pred)*100,3)

confusion_mat('Blues')
NB=GaussianNB(var_smoothing=0.1)
NB.fit(train_arr,train_y)
pred=NB.predict(test_arr)
eval_model(test_y,pred)
    
c=round(accuracy_score(test_y,pred)*100,3)
confusion_mat('Greens')
SGD=SGDClassifier(early_stopping=True,penalty='l2',alpha=0.00001)
SGD.fit(train_arr,train_y)
pred=SGD.predict(test_arr)
eval_model(test_y,pred)
    
d=round(accuracy_score(test_y,pred)*100,3)
confusion_mat('Reds')
lgbm=LGBMClassifier(learning_rate=0.35)
lgbm.fit(train_arr,train_y)
pred=lgbm.predict(test_arr)
eval_model(test_y,pred)

e=round(accuracy_score(test_y,pred)*100,3)
confusion_mat('YlOrBr')
sns.set()
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
Models = ['MultinomialNB', 'DecisionTree', 'GaussianNB', 'SGD','LGBM']
Accuracy=[a,b,c,d,e]
ax.bar(Models,Accuracy,color=['#702963','#8a2be2','#9966cc','#df73ff','#702763']);
for i in ax.patches:
    ax.text(i.get_x()+.1, i.get_height()-5.5, str(round(i.get_height(),2))+'%', fontsize=15, color='white')
plt.title('Comparison of Different Classification Models');
plt.ylabel('Accuracy');
plt.xlabel('Classification Models');

plt.show();