# !pip install newspaper3k
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error as RMSE
from sklearn.decomposition import PCA
import seaborn as sns
from newspaper import Article
import requests
from bs4 import BeautifulSoup
data = pd.read_csv("../input/uci-online-news-popularity-data-set/OnlineNewsPopularity.csv")
def cleancols(cols):
    x = [y.lower().strip() for y in cols]
    return x
data.columns = cleancols(data.columns)
y_labels = data['shares'] #the feature to be predicted
df = data.drop(columns = ['url','timedelta','shares'],axis = 1) #url and timedelta are of no use to us
url = "https://www.bbc.com/news/world"
page = requests.get(url)
soup = BeautifulSoup(page.content,'html.parser')
imgnews = soup.findAll('a',attrs = {'class':'gs-c-promo-heading gs-o-faux-block-link__overlay-link gel-pica-bold nw-o-link-split__anchor'})
newsarts = []
for arts in imgnews:
    newsarts.append("https://www.bbc.com"+arts['href'])
pca = PCA(n_components = 12)
df_scaled = pd.DataFrame(preprocessing.scale(df),columns = df.columns)
df_red = pca.fit_transform(df_scaled) #applying PCA on the standardized data
explainedfeats = pd.DataFrame(pca.components_,index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7','PC-8','PC-9','PC-10','PC-11','PC-12'],columns = df.columns)
xgb = XGBRegressor(max_depth = 10,random_state = 42)
xgb.fit(df,y_labels)
fig, ax = plt.subplots(1,1,figsize=(10,10))
impplot = plot_importance(xgb,ax = ax)
plt.show()
impfeats = [impplot.get_yticklabels()[::-1][i].get_text() for i in range(0,20)]
print(impfeats)
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
stopwords=set(stopwords.words('english'))
from textblob import TextBlob #for subjectivity and polarity purpose
def tokenizetext(text):
    return word_tokenize(text)
def words(text):
    l = [word for word in word_tokenize(text) if word.isalpha()]
    return l
def unique_words(text):
    return list(set(words(text)))
def rate_uni_words(text):
    uni_words = len(unique_words(text))/len(words(text))
    return uni_words
def avglengthtoken(text):
    w = words(text)
    sum = 0
    for item in w:
        sum+=len(item)
    avglen = sum/len(w)
    return avglen
def n_non_stop_unique_tokens(text):
    uw = unique_words(text)
    n_uw = [item for item in uw if item not in stopwords]
    w = words(text)
    n_w = [item for item in w if item not in stopwords]
    rate_nsut = len(n_uw)/len(n_w)
    return rate_nsut
def numlinks(article):
    return len(BeautifulSoup(sampletext.html).findAll('link'))
def get_subjectivity(a_text):
    return a_text.sentiment.subjectivity
def get_polarity(a_text):
    return a_text.sentiment.polarity
def word_polarity(words):
    pos_words = []
    ppos_words = [] # polarity of pos words
    neg_words = []
    pneg_words = [] # polarity of negative words
    neu_words = []
    pneu_words = [] # polarity of neutral words
    for w in words:
        an_word = TextBlob(w)
        val = an_word.sentiment.polarity
        if val > 0:
            pos_words.append(w)
            ppos_words.append(val)
        if val < 0:
            neg_words.append(w)
            pneg_words.append(val)
        if val == 0 :
            neu_words.append(w)
            pneu_words.append(val)
    return pos_words,ppos_words,neg_words,pneg_words,neu_words,pneu_words
def avg_pol_pw(text):    
    totalwords = words(text)
    res = word_polarity(totalwords)
    return np.sum(res[1])/len(res[0])
def avg_pol_nw(text):    
    totalwords = words(text)
    res = word_polarity(totalwords)
    return np.sum(res[3])/len(res[2])
finrows = []
for article in newsarts[0:25]:
    sampletext = Article(article, language = 'en')
    sampletext.download()
    sampletext.parse()
    sampletext.nlp() 
    
    row = {}
    row['n_tokens_title'] = len(words(sampletext.title))
    row['n_tokens_content'] = len(words(sampletext.text))
    row['n_unique_tokens'] = len(unique_words(sampletext.text))
    row['average_token_length'] = avglengthtoken(sampletext.text)
    row['n_non_stop_unique_tokens'] = n_non_stop_unique_tokens(sampletext.text)
    row['num_hrefs'] = numlinks(sampletext)
    
    analysed_text = TextBlob(sampletext.text)
    row['global_subjectivity'] = get_subjectivity(analysed_text)
    row['avg_positive_polarity'] = avg_pol_pw(sampletext.text)
    row['global_sentiment_polarity'] = get_polarity(analysed_text)
    finrows.append(row)
#converting the list to a dataframe
masterdf = pd.DataFrame(finrows, columns = ['n_tokens_title','n_tokens_content','n_unique_tokens','average_token_length','n_non_stop_unique_tokens','num_hrefs','global_subjectivity',
                                   'avg_positive_polarity','global_sentiment_polarity'])
df_reduced = df[masterdf.columns]
xtrain, xtest, ytrain, ytest = train_test_split(df_reduced, y_labels, test_size = 0.2, shuffle = True, random_state = 42)
xgb2 = XGBRegressor(random_state = 42)
paramsxgb = {'max_depth':[5,20,50,100]}
gsc = GridSearchCV(estimator = xgb2,param_grid = paramsxgb, cv = 3, scoring = 'neg_root_mean_squared_error')
gscres = gsc.fit(xtrain,ytrain)
gscres.best_params_
xgb2.max_depth = gscres.best_params_['max_depth']
boost = ['gbtree','gblinear']
rmsescores = {}
for b in boost:
    xgb2.booster = b
    xgb2.fit(xtrain,ytrain)
    predicted = xgb2.predict(xtest)
    rmsescores['xgb-'+b] = RMSE(ytest,predicted,squared = False)
    print("RMSE error with {} booster is {} :".format(b,RMSE(ytest,predicted,squared = False)))
xgb2.booster = 'gblinear'
rf = RandomForestRegressor(random_state = 42)
paramsrf = {'max_depth':[5,20,50,100]}
gsc = GridSearchCV(estimator = rf,param_grid = paramsrf, cv = 3, scoring = 'neg_root_mean_squared_error')
gscres = gsc.fit(xtrain,ytrain)
gscres.best_params_
rf.max_depth = gscres.best_params_['max_depth']
rf.fit(xtrain,ytrain)
predictedrf = rf.predict(xtest)
rmsescores['rf'] = RMSE(ytest,predictedrf,squared = False)
print("RMSE error with Random Forest Regressor is {} :".format(RMSE(ytest,predicted,squared = False)))
lr = RidgeCV(alphas = [0.001,0.1,1,5,10,100],scoring = 'neg_root_mean_squared_error', cv = None, store_cv_values = True)
lr.fit(xtrain,ytrain)
predictedlr = lr.predict(xtest)
rmsescores['ridgecv'] = RMSE(ytest,predictedlr,squared = False)
print("RMSE error with Linear Regression via RidgeCV is {} :".format(RMSE(ytest,predictedlr,squared = False)))
cb = CatBoostRegressor(verbose = 0,random_state = 42,eval_metric = 'RMSE')
paramscb = {'iterations':[1,10,50,100],'learning_rate':[0.03,0.1,0.5,1],'depth':[3,5,8,10]}
gsc = GridSearchCV(estimator = cb,param_grid = paramscb, cv = 3, scoring = 'neg_root_mean_squared_error')
gscres = gsc.fit(xtrain,ytrain)
gscres.best_params_
cb.iterations = gscres.best_params_['iterations']
cb.learning_rate = gscres.best_params_['learning_rate']
cb.depth = gscres.best_params_['depth']
cb.fit(xtrain,ytrain)
predictedcb = cb.predict(xtest)
rmsescores['catboost'] = RMSE(ytest,predictedcb,squared = False)
print("RMSE error with CatBoost is {} :".format(RMSE(ytest,predictedcb,squared = False)))
ax = sns.barplot(x = list(rmsescores.keys()),y = list(rmsescores.values()))
ax.set_ylim(10750,12000)
ax.set_xlabel('Regressors')
ax.set_ylabel('RMSE Scores')
plt.show()
predlr = lr.predict(masterdf)
lrdata = {'Links':list(newsarts[:25]),'Predicted Virality':list(predlr)}
pd.DataFrame(lrdata).reindex(np.arange(0,25,1))
predcb = cb.predict(masterdf)
cbdata = {'Links':list(newsarts[0:25]),'Predicted Virality':list(predcb)}
pd.DataFrame(cbdata).reindex(np.arange(0,25,1))
predrf = rf.predict(masterdf)
rfdata = {'Links':list(newsarts[0:25]),'Predicted Virality':list(predrf)}
pd.DataFrame(rfdata).reindex(np.arange(0,25,1))
predxgb = xgb2.predict(masterdf)
xgbdata = {'Links':list(newsarts[0:25]),'Predicted Virality':list(predxgb)}
pd.DataFrame(xgbdata).reindex(np.arange(0,25,1))