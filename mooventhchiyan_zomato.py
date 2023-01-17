import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.datasets import load_files

from nltk.corpus import stopwords

from nltk.sentiment.vader import SentimentIntensityAnalyzer
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
zomato = pd.read_csv("/kaggle/input/zomato-bangalore-restaurants/zomato.csv")

zomato.head()
zomato.info()
# Percentage of Nul Values

pd.DataFrame(round(zomato.isnull().sum()/zomato.shape[0] * 100,3), columns = ["Nan"])  
zomato=zomato.drop(['url','dish_liked','phone'],axis=1)
zomato.columns
zomato= zomato.rename(columns={"approx_cost(for two people)" : "Cost_of_two",

                                "reviews_list": "reviews",

                                 "menu_item"  : "menu",

                                 "listed_in(type)": "type",

                                 "listed_in(city)": "city"})
zomato.head()
zomato['rate']=zomato['rate'].str.split('/').str[0]
sd= zomato.copy(deep=True)

sd= zomato[zomato['rate']=='NEW']

fd= zomato[zomato['rate'].isna()]

sfd= pd.concat([sd,fd])

sfd= sfd[sfd['reviews']!=" "] ## Making sure that the review is not empty
import nltk

nltk.download("stopwords")



stopwords= nltk.corpus.stopwords.words('english')

stopwords.remove('not')

stemmer= nltk.stem.PorterStemmer()

import re



def clean_data(doc):

    doc= doc.lower()

    doc= re.sub(r"\W"," ",doc)

    doc= re.sub(r"\d"," ",doc)

    doc= re.sub(r"\s+"," ",doc)

    doc=re.sub("[^a-z\s]","",doc)

    words= doc.split(" ")

    doc= re.sub(r" [@#$%\&\*\(\)\<\>\?\'\":;\]\[-] ", " ",doc)

    word_imp= [ stemmer.stem(word) for word in words if word not in stopwords]

    doc_cleaned=" ".join(word_imp)

    return doc_cleaned

sid= SentimentIntensityAnalyzer()

sfd["reviews"]=sfd["reviews"].apply(clean_data)
sfd['scores'] = sfd['reviews'].apply(lambda reviews: sid.polarity_scores(reviews))



sfd['compound']  = sfd['scores'].apply(lambda score_dict: score_dict['compound'])



sfd['neu']  = sfd['scores'].apply(lambda score_dict: score_dict['neu'])



sfd['comp_score'] = sfd['compound'].apply(lambda c: 'pos' if c >=0.2 else 'neg')



sfd.head()
sfd["comp_score"].value_counts()
def comp(comp_score):

    if comp_score =='pos':

        return 4

    else:

        return 3

sfd['new_rate']=sfd['comp_score'].apply(comp)
nsd= sfd.drop(['scores','compound','neu','rate','comp_score'],1)

nsd=nsd.rename(columns={"new_rate":"rate"})

nzomato= zomato.loc[(zomato['rate']!='NEW')]

nzomato =nzomato.loc[zomato['rate'].notna()]

frame=[nzomato,nsd]

final=pd.concat(frame)
final.head()
final= final[final['rate']!='-']

final['Cost_of_two']=final['Cost_of_two'].replace(",","",regex=True).astype(float)
final['Cost_of_two'].plot(kind='hist',figsize=(8,8))
### Adjustments on the data types

final['rate']=final['rate'].astype("float")

# round(final['Cost_of_two'].mean()) ---- > 586

final['Cost_of_two']=final['Cost_of_two'].fillna('586')

final['rest_type']= final['rest_type'].fillna('Quick Bites')  # Filled with mode(maximum occurance)
menu= final[final['cuisines'].isna()]

# There are around 10 restrants without the cuisines type we can fill after analysing the menu enterily
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
WC= menu[menu['menu'].map(lambda d: len(d)) > 5]

# Start with one menu:

# Create and generate a word cloud image:

for i in range(0,4):

    text= WC.menu.iloc[i]

    wordcloud = WordCloud().generate(text)



# Display the generated image:

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.title(WC['name'].iloc[i])

    plt.show()
final.loc[(pd.isnull(final.cuisines)), 'cuisines'] = "Briyani"

final.loc[(final.name=="Lassi Spot"),"cuisines"]= "Desserts, Ice Cream "
final.isnull().sum()
# Top 20 shops in the zomato list

final['name'].value_counts()[:20].plot(kind='bar')
# Bangalore a IT hub everone needs a quick bites (reallly sad)

final['rest_type'].value_counts()[:20].plot(kind='bar')
plt.figure(figsize=(14,4))

plt.grid()

sns.distplot(final['Cost_of_two'])

#final['Cost_of_two'].value_counts().sort_values(ascending=False).plot(kind='bar')

plt.show()
print(final[final['name'].str.lower().str.contains("briyani")].shape)

Briyani= final[final['name'].str.lower().str.contains("briyani")]

Briyani.shape
Briyani.groupby('name')['location'].count().plot(figsize=(8,8),kind='pie')
Briyani.groupby('location')['location'].count().plot(figsize=(8,8),kind='pie')
Briyani.groupby('location')['Cost_of_two'].mean().plot(figsize=(12,8),kind='line')

# Briyani is costly in IndiraNagar and lesser in JP Nagar,Peenya
Briyani.groupby('online_order')['location'].count()
# Getting dummies is not going to work here as we have columns with address,name,menu

# Using factorize with reference to one of the kaggle kernels



def Encode(zomato):

    for column in zomato.columns[~zomato.columns.isin(['rate', 'Cost_of_two', 'votes'])]:

        zomato[column] = zomato[column].factorize()[0]

    return zomato



zomato_en = Encode(final.copy())
zomato_en
zomato_en['Cost_of_two']=zomato_en['Cost_of_two'].astype(int)
X= zomato_en.drop('Cost_of_two',axis=1)

y= zomato_en['Cost_of_two']
import scipy.stats as st

import statsmodels.api         as     sm
#Adding constant column of ones, mandatory for sm.OLS model

X_1 = sm.add_constant(X)

#Fitting sm.OLS model

model = sm.OLS(y,X_1).fit()

model.pvalues

#model.summary()
#Backward Elimination

cols = list(X.columns)

pmax = 1

while (len(cols)>0):

    p= []

    X_1 = X[cols]

    X_1 = sm.add_constant(X_1)

    model = sm.OLS(y,X_1).fit()

    p = pd.Series(model.pvalues.values[1:],index = cols)      

    pmax = max(p)

    feature_with_p_max = p.idxmax()

    if(pmax>0.05):

        cols.remove(feature_with_p_max)

    else:

        break

selected_features_BE = cols

print(selected_features_BE)
X= X[selected_features_BE]
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score,accuracy_score

from sklearn.model_selection import train_test_split

import xgboost
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
RFR=RandomForestRegressor()

RFR.fit(X_train,y_train)



GB= GradientBoostingRegressor()

GB.fit(X_train,y_train)



Ada= AdaBoostRegressor()

Ada.fit(X_train,y_train)



XG= xgboost.XGBRegressor()

XG.fit(X_train,y_train)
RFR_train_pred=RFR.predict(X_train)

GB_train_predict=GB.predict(X_train)

Ada_train_predict=Ada.predict(X_train)

XG_train_predict=XG.predict(X_train)
RFR_test_pred=RFR.predict(X_test)

GB_test_predict=GB.predict(X_test)

Ada_test_predict=Ada.predict(X_test)

XG_test_predict=XG.predict(X_test)
## r2_score



r2_score_train=[r2_score(y_train,RFR_train_pred),r2_score(y_train,GB_train_predict),r2_score(y_train,Ada_train_predict),r2_score(y_train,XG_train_predict)]

r2_score_test =[r2_score(y_test,RFR_test_pred),r2_score(y_test,GB_test_predict),r2_score(y_test,Ada_test_predict),r2_score(y_test,XG_test_predict)]

a=pd.DataFrame({'r2_score_train':r2_score_train,'r2_score_test':r2_score_test})

plt.figure(figsize=[10,5]) 

a.plot(kind='bar')

plt.xticks(np.arange(4),['RFR','GB','ada','XG'])

plt.show()