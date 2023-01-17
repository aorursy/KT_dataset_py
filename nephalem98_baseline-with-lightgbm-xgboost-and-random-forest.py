import pandas as pd #Data Analysis

import numpy as np #Linear Algebra

import seaborn as sns #Data Visualization

import matplotlib.pyplot as plt #Data Visualization\

%matplotlib inline
import json

import string

from pandas.io.json import json_normalize

color = sns.color_palette()

from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999
import os

print(os.listdir("../input"))
#This is the Product_sales_train_and_test dataset but without the "[]" in the Customer Basket.

df=pd.read_csv("../input/removed//data.csv")
train=pd.read_csv("../input/discount-prediction/Train.csv")
test=pd.read_csv("../input/discount-prediction/test.csv")
product=pd.read_csv("../input/discount-prediction/product_details.csv",encoding="mac-roman")
#Removing the front and trailing spaces

df['Customer_Basket']=df['Customer_Basket'].str.lstrip()

df['Customer_Basket']=df['Customer_Basket'].str.rstrip()
#The count of the number of Product Id's in the Customer Basket

df['Length']=df['Customer_Basket'].str.split(' ').apply(len)
df.head()
train.head()
#We can see a lot of null values in the train dataset

train.info()
#Let us see number of null values there are

train.isnull().sum()
test.isnull().sum()
train[train['BillNo'].isna()].head(10)
train.dropna(axis=0,how='all',inplace=True)
train.isnull().sum()
train.fillna(float(0.0),inplace=True)
train.isnull().sum()
train['Customer'].value_counts().head()
len(set(test['Customer']).difference(set(train['Customer'])))
train['Discount 5%'].value_counts()
sns.set_style("whitegrid")

sns.countplot(x='Discount 5%',data=train)
sns.set_style("whitegrid")

sns.countplot(x='Discount 12%',data=train)
sns.set_style("whitegrid")

sns.countplot(x='Discount 18%',data=train)
sns.set_style("whitegrid")

sns.countplot(x='Discount 28%',data=train)
#lol=df2["Customer"].str.split(", ",n=1,expand=True)

#df2['CustomerName']=lol[0]

#df2["Location"]=lol[1]
#lol1=df3["Customer"].str.split(", ",n=1,expand=True)

#df3['CustomerName']=lol1[0]

#df3["Location"]=lol1[1]
#set(df3['Location']).difference(set(df2["Location"]))
#df3[df3["Location"]=='T.M.M. HOSPITAL, THIRUVALLA.']
#sns.countplot(x="discount",hue="Location",data=df3)
#df2["discount"].value_counts()
#len(set(trailtest['Customer']).difference(set(trailtrain['Customer'])))
discount=[]

for i, row in train.iterrows():

    if row["Discount 5%"]==1.0:

        discount.append(1)

    elif row["Discount 12%"]==1.0:

        discount.append(2)

    elif row["Discount 18%"]==1.0:

        discount.append(3)

    elif row["Discount 28%"]==1.0:

        discount.append(4)

    else:

        discount.append(5)        
train["discount"]=discount
from wordcloud import WordCloud, STOPWORDS

from collections import defaultdict

train1_df = train[train["discount"]==1]

train2_df = train[train["discount"]==2]

train3_df = train[train["discount"]==3]

train4_df = train[train["discount"]==4]

train5_df = train[train["discount"]==5]



## custom function for ngram generation ##

def generate_ngrams(text, n_gram=1):

    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



## custom function for horizontal bar chart ##

def horizontal_bar_chart(df, color):

    trace = go.Bar(

        y=df["word"].values[::-1],

        x=df["wordcount"].values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace



## Get the bar chart from sincere questions ##

freq_dict = defaultdict(int)

for sent in train1_df["Customer"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



## Get the bar chart from insincere questions ##

freq_dict = defaultdict(int)

for sent in train2_df["Customer"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



freq_dict = defaultdict(int)

for sent in train3_df["Customer"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



freq_dict = defaultdict(int)

for sent in train4_df["Customer"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace3 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



freq_dict = defaultdict(int)

for sent in train5_df["Customer"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace4 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



# Creating two subplots

fig = tools.make_subplots(rows=3, cols=2, vertical_spacing=0.04,

                          subplot_titles=["Frequent words of Discount 5%", 

                                          "Frequent words of Discount 12%",

                                         "Frequent words of Discount 18%",

                                         "Frequent words of Discount 28%","Frequent words of No Discount"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace3, 2, 2)

fig.append_trace(trace4, 3, 1)



fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

py.iplot(fig, filename='word-plots')



#plt.figure(figsize=(10,16))

#sns.barplot(x="ngram_count", y="ngram", data=fd_sorted.loc[:50,:], color="b")

#plt.title("Frequent words for Insincere Questions", fontsize=16)

#plt.show()
train.drop(['Discount 5%','Discount 12%','Discount 18%','Discount 28%'],axis=1,inplace=True)
from sklearn.feature_extraction.text import CountVectorizer

cv1 = CountVectorizer(max_features=500)

y = cv1.fit_transform(df["Customer_Basket"]).toarray()
len(cv1.vocabulary_)
thirty= list(y)

thirty1=pd.DataFrame(thirty)
final=pd.concat([df,thirty1],axis=1)
finaltrain=pd.merge(train,final,on="BillNo",how="inner")

finaltest=pd.merge(test,final,on="BillNo",how="inner")
finaltrain.head()
finaltest.head()
#df2=df2[df2["BillNo"]!=float(0.0)]
finaltrain.drop(["BillNo","Customer_Basket","Customer","Date"],axis=1,inplace=True)

finaltest.drop(["BillNo","Customer_Basket","Customer","Date"],axis=1,inplace=True)
X=finaltrain.drop("discount",axis=1)

y=finaltrain["discount"]
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X, y.ravel())
X_train=pd.DataFrame(X_train_res)
y_train=pd.DataFrame(y_train_res)
X_train["smote"]=y_train_res
X1=X_train.drop(["smote"],axis=1)

y1=X_train["smote"]
import lightgbm as lgb
model = lgb.LGBMClassifier( class_weight = 'balanced',

                               objective = 'multiclass', n_jobs = -1, n_estimators = 400)
model.fit(X1,y1)
pred_lg=model.predict(finaltest)
pred_lg
import xgboost as xgb

from xgboost.sklearn import XGBClassifier
xgb = XGBClassifier(max_depth=5, learning_rate=0.2, n_estimators=300,

                    objective='multi:softprob', subsample=0.6, colsample_bytree=0.6, seed=0, silent=0)                  
#xgb.fit(X1, y1)
#pred_xg=xgb.predict(finaltest)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=500)
rfc.fit(X1,y1)
rfcpredict=rfc.predict(finaltest)
rfcpredict
#from sklearn.model_selection import StratifiedKFold

#kfold = 5

#skf = StratifiedKFold(n_splits=5)
#folds = 3

#param_comb = 5



#skf = KFold(n_splits=folds, shuffle = True, random_state = 1001)



#random_search = RandomizedSearchCV(rfc, param_distributions=params, n_iter=param_comb, scoring=rmsle, n_jobs=4, cv=skf.split(X,y), verbose=3, random_state=1001 )
#random_search.fit(X, y)
#print('\n All results:')

#print(random_search.cv_results_)

#print('\n Best estimator:')

#print(random_search.best_estimator_)

#print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))

#print(random_search.best_score_ * 2 - 1)

#print('\n Best hyperparameters:')

#print(random_search.best_params_)