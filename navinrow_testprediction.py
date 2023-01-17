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
df = pd.read_csv("/kaggle/input/amazon-fine-food-reviews/Reviews.csv")
df.dropna()

df['HelpfulnessDenominator']=df['HelpfulnessDenominator'].astype(float)

df['HelpfulnessNumerator']=df['HelpfulnessNumerator'].astype(float)
df['Score']=df['Score'].astype(float)
# yummydummy = []
# for yum in range(len(df['Score'])):
#     if(df['Score'][yum]>=4):
#         yummydummy.append("Positive")
#     elif(df['Score'][yum]>=3 and df['Score'][yum]<4):
#         yummydummy.append("Neutral")
#     else:
#         yummydummy.append("Negative")
# df['senti'] = yummydummy
        

df['senti'] = df['Score']!=3.0
df['senti'] = df['Score']>=4.0
df['senti'] =df["senti"].replace([True , False] , ["pos" , "neg"])
import matplotlib as mpl
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)


mpl.rcParams['font.size']=12                #10 
mpl.rcParams['savefig.dpi']=100             #72 
mpl.rcParams['figure.subplot.bottom']=.1 


def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=300,
        max_font_size=40, 
        scale=3,
        random_state=1 
        
    ).generate(str(data))
    
    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
show_wordcloud(df['Summary'][df.senti=="neg"],title="Negative Words")
df.head()
df['Text'][3]
real = []
negu = []
for j in range(len(df['senti'])):
    if(df['senti'][j]=="pos"):
        real.append(df['Summary'][j])
        
    else:
        negu.append(df['Summary'][j])
        
        
print("TOP 200 positive sentences summary!!!!!!!!!!!!:")
print("-------------------------------")
for i in range(200):
    
    print(real[i])
    
print("top 200 negative sentences summary!!!!!!!!!!!!!!!!!!!!!!!:")
print("-------------------------------")
for ku in range(200):
    print(negu[ku])
    



# print("top 20 positive Sentences in short:",real)
# print("top 20 negatives Sentences in short:",negu)
! pip install nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# from nltk.tokenize import word_tokenize
import nltk

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
print(stopwords.words("english"))
# r =[]
# for k in range(len(df['Score'])):
#     text = df['Text'][k]
#     text_tokens = word_tokenize(text)

#     tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
#     r.append(tokens_without_sw)
    



# df['no_stop_words'] =r 

features= ['HelpfulnessNumerator','HelpfulnessDenominator']
X = df[features]
y = df.Score
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=1,max_leaf_nodes=3000)
model.fit(X,y)
prediction = model.predict(X)
prediction
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

# print(mean_absolute_error(y,prediction))
print(model.score(X,y))
# model accuracy very hard to predict score from helpful numerator and denominator
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.model_selection import cross_val_predict
y_p = cross_val_predict(model, X, y, cv=3)

# print(confusion_matrix(y, y_p.argmax(axis=0)))
# print(classification_report(y, y_p))
# print("Accuray score:",accuracy_score(y,y_p))

from sklearn.naive_bayes import GaussianNB
model1 =GaussianNB() 
model1.fit(X,y)
# second model gaussian
prediction1 = model1.predict(X)
prediction1
print("In sample prediction:",model1.predict(X.head(50)))
print("Actual values:",y.head(50).tolist())

print(model1.score(X,y))

#  model gaussian nb have better accuracy 
from sklearn.linear_model import LinearRegression
model2 = LinearRegression()
model2.fit(X,y)
prediction2 = model2.predict(X)
prediction2


print("in sample prediction:",model2.predict(X.head(50)))
print("actual Score:",y.head(50).tolist())
model2.score(X,y)
# using linear regression even worse accuracy the best till now is naive bayes for predicting score based on numerator and denominator
from sklearn.naive_bayes import MultinomialNB
model3 = MultinomialNB(alpha=40.0,fit_prior=True)
model3.fit(X,y)
prediction3 = model3.predict(X.head(50))
print("Prediction with multinomial:",prediction3)
print("Actual values:",y.head(50).tolist())
print(model3.score(X,y))
from sklearn.tree import DecisionTreeRegressor
model4 = DecisionTreeRegressor()
model4.fit(X,y)
prediction4 = model4.predict(X.head(50))
print("Top few predictions:",prediction4)
print("Real Values",y.head(50).tolist())
print(model4.score(X,y))
###predictiing Positive negative from scoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee in training set
yolo = df[['Score','senti','Text']]
yolo.head()
from nltk.corpus import stopwords
stop = stopwords.words('english')

yolo['Text'] = yolo['Text'].str.lower().str.split()
yolo['Text']=yolo['Text'].apply(lambda x: [item for item in x if item not in stop])
yolo.head()
yolo['Text'][0]
X1 = yolo[['Score']]
y1 =yolo.senti
from sklearn.naive_bayes import GaussianNB
modela = GaussianNB()
modela.fit(X1,y1)
pred1 = modela.predict(X1.head(50))
print("Top 50 Values of pos and neg based on score",pred1)
print("Top 50 Actual pos and neg values",y1.head(50).tolist())
print(modela.score(X1,y1)*100)
type(pred1)
ohlo = y1.head(20).tolist()
from sklearn.naive_bayes import MultinomialNB
modela1 = MultinomialNB()
modela1.fit(X1,y1)

pred2 = modela1.predict(X1.head(20))
print("Top 20 positive by multinomial nb:",pred2 )
print("Top 20 actual senti",y1.head(20).tolist())
print(modela1.score(X1,y1))
from sklearn.linear_model import LinearRegression
modela2 = LinearRegression()

temporary = yolo
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
temporary['senti'] = labelencoder.fit_transform(temporary['senti'])
temporary.head()
y2 = temporary.senti
X2 = temporary[['Score']]
modela2.fit(X2,y2)
print("top 50 by linear regression prediction:",modela2.predict(X2.head(50)))
print("top 50 actual:",y2.head(50).tolist())
print(modela2.score(X2,y2)*100)


print(modela2.score(X2,y2))
modela4 = GaussianNB()
modela4.fit(X2,y2)
print("top 20 by gaussian label encoder:",modela4.predict(X2.head(20)))
print("top 20 actual:",y2.head(20).tolist())

print("Score of gaussian with label encoder:",modela4.score(X2,y2)*100)
from matplotlib import pyplot as plt 
poyi = []
noyp = []
yoppii = []
for uipp in range(200):
    yoppii.append(uipp)
    poyi.append(X1['Score'][uipp])
    
    
noyp = y1.head(200).tolist()
    
df
yoppii

plt.xlabel("senti")
plt.ylabel("total")
plt.bar(noyp,yoppii)
# plt.show()
# here is the top 200 number of positive and negative where number of positive sentences in tokens are abit more than number of negative sentences meaning more of the overall socre would be 4.0 or 5.0 than 2.0 or 1.0
