#Loading in relevant packages

import numpy as np

import pandas as pd

import os

pd.set_option('precision', 3)#Setting the third-decimal point

from tqdm import tqdm_notebook



#Visualisation packages

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

import plotly.offline as py

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)

%matplotlib inline

warnings.filterwarnings('ignore')



#NLP packages

import nltk

from nltk import word_tokenize,pos_tag

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords,wordnet

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 



#Model packages

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score,f1_score

from sklearn.model_selection import KFold

from sklearn.base import clone

from sklearn.decomposition import PCA
# Load in the train and test datasets

tr = pd.read_csv('../input/review_train.csv')

te = pd.read_csv('../input/review_test.csv')



# Store our Review ID for the final results

ID = te['REVIEW_ID']



train= tr.copy()

test= te.copy()



train.head()
train.info()
test.info()
#There are no null values in the test file, it is reasonable to drop the observations with null values in the training set 

#cause the percentage is relatively low as the whole set has 14700 entries

train=train.dropna()

train.info()
#The histogram regarding rating and label

trace1 = go.Histogram(

    x=train[train['LABEL']==0]['RATING'],

    opacity=0.75,

    name='Normal',

    xbins=dict(

        size=0.5,

    ),

)

trace2 = go.Histogram(

    x=train[train['LABEL']==1]['RATING'],

    opacity=0.75,

    name='Fake',

    xbins=dict(

        size=0.5,

    ),

)



data = [trace1, trace2]

layout = go.Layout(title='Rating Histogram',

                   width=700,

                   barmode='overlay')

fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
#The histogram regarding Verified-Purchase and label

trace1 = go.Histogram(

    x=train[train['LABEL']==0]['VERIFIED_PURCHASE'],

    opacity=0.75,

    name='Normal',

    xbins=dict(

        size=0.5

    ),

)

trace2 = go.Histogram(

    x=train[train['LABEL']==1]['VERIFIED_PURCHASE'],

    opacity=0.75,

    name='Fake',

    xbins=dict(

        size=0.5

    ),

)



data = [trace1, trace2]

layout = go.Layout(title='Verified-Purchase Histograms',

                   width=500,

                   barmode='overlay')

fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
#The histogram regarding Product category and label

trace1 = go.Histogram(

    x=train[train['LABEL']==0]['PRODUCT_CATEGORY'],

    opacity=0.75,

    name='Normal',

    xbins=dict(

        size=0.5

    ),

)

trace2 = go.Histogram(

    x=train[train['LABEL']==1]['PRODUCT_CATEGORY'],

    opacity=0.75,

    name='Fake',

    xbins=dict(

        size=0.5

    ),

)



data = [trace1, trace2]

layout = go.Layout(title='Product category Histograms',

                   width=950,

                   barmode='overlay')

fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
#Drop useless and non-valuable columns 

train = train.drop(columns = ['REVIEW_ID','PRODUCT_ID','PRODUCT_TITLE'])

test  = test.drop(columns = ['REVIEW_ID','PRODUCT_ID','PRODUCT_TITLE'])
#Copy training set and test set for further processes

X_train= train.copy()

X_test= test.copy()





#Replace Y,N by 1,0 in 'VERIFIED_PURCHASE'

X_train['VERIFIED_PURCHASE'] = X_train['VERIFIED_PURCHASE'].apply(lambda x: 1 if x == 'Y' else 0)

X_test['VERIFIED_PURCHASE'] = X_test['VERIFIED_PURCHASE'].apply(lambda x: 1 if x == 'Y' else 0)

#Apply one-hot labeling to 'PRODUCT_CATEGORY'

s = pd.get_dummies(X_train['PRODUCT_CATEGORY'])

X_train = pd.concat([X_train,s],axis = 1)

X_train = X_train.drop(columns='PRODUCT_CATEGORY')

s = pd.get_dummies(X_test['PRODUCT_CATEGORY'])

X_test = pd.concat([X_test,s],axis = 1)

X_test = X_test.drop(columns='PRODUCT_CATEGORY')

#Derive the length of 'REVIEW_TITLE'

X_train['TITLE_LENGTH'] = X_train['REVIEW_TITLE'].apply(lambda x:  len(x.split()))

X_test['TITLE_LENGTH'] = X_test['REVIEW_TITLE'].apply(lambda x:  len(x.split()))

#Derive the length of 'REVIEW_TEXT'

X_train['TEXT_LENGTH'] = X_train['REVIEW_TEXT'].apply(lambda x:  len(x.split()))

X_test['TEXT_LENGTH'] = X_test['REVIEW_TEXT'].apply(lambda x:  len(x.split()))



#Concate the training set and test set

alldata = pd.concat([X_train.drop(columns='LABEL'),X_test])

print(alldata.shape)

alldata.head()
#The histogram regarding Title length

trace1 = go.Histogram(

    x=X_train[X_train['LABEL']==0]['TITLE_LENGTH'],

    opacity=0.75,

    name='Normal',

)

trace2 = go.Histogram(

    x=X_train[X_train['LABEL']==1]['TITLE_LENGTH'],

    opacity=0.75,

    name='Fake',

)



data = [trace1, trace2]

layout = go.Layout(title='Title length Histogram',

                   width=700,

                   barmode='overlay')

fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
#The histogram regarding Text length

trace1 = go.Histogram(

    x=X_train[X_train['LABEL']==0]['TEXT_LENGTH'],

    opacity=0.75,

    name='Normal',

        xbins=dict(

        start=0,

        end=600,

        size=10,

    ),

)

trace2 = go.Histogram(

    x=X_train[X_train['LABEL']==1]['TEXT_LENGTH'],

    opacity=0.75,

    name='Fake',

        xbins=dict(

        start=0,

        end=600,

        size=10,

    ),

)



data = [trace1, trace2]

layout = go.Layout(title='Text length Histogram',

                   width=700,

                   barmode='overlay')

fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
#Defining a function using NLTK to clean the text

def clean_text(sentence):

    #Tokenize the sentences and add tags

    token_word = word_tokenize(sentence)  

    token_words = pos_tag(token_word)

    

    #Normalization

    words_lematizer = []

    wordnet_lematizer = WordNetLemmatizer()

    for word, tag in token_words:

        if tag.startswith('N'):

            word_lematizer =  wordnet_lematizer.lemmatize(word, wordnet.NOUN)

        elif tag.startswith('V'): 

            word_lematizer =  wordnet_lematizer.lemmatize(word, wordnet.VERB)

        elif tag.startswith('J'): 

            word_lematizer =  wordnet_lematizer.lemmatize(word, wordnet.ADJ)

        elif tag.startswith('R'): 

            word_lematizer =  wordnet_lematizer.lemmatize(word, wordnet.ADV)

        else: 

            word_lematizer =  wordnet_lematizer.lemmatize(word)

        words_lematizer.append(word_lematizer)

    

    #Decapitalise the words

    words_list = [x.lower() for x in words_lematizer ]

    #Clean characters and abberviations

    abbreviations = ["'s","'ve","'d","'t","'ll","'n","'m","n't"]

    words_list = [word for word in words_list if word not in abbreviations]

    characters = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','-','...','^','{','}']

    words_list = [word for word in words_list if word not in characters]

    #Clean Stop words    

    cleaned_words = [word for word in words_list if word not in stopwords.words('english')]

    

    words = ' '.join(cleaned_words)



    

    return words
#Due to the amount of samples, text_title would be the possbile feature for analysis

for i in tqdm_notebook(range(0,train.shape[0])):

    train.loc[i,'REVIEW_TITLE'] = clean_text(train.iloc[i]['REVIEW_TITLE'])

for i in tqdm_notebook(range(0,test.shape[0])):

    test.loc[i,'REVIEW_TITLE'] = clean_text(test.iloc[i]['REVIEW_TITLE'])

    

train=train.dropna()

test=test.dropna()
#Calculate the TF-IDF of review_title

words = pd.concat([train['REVIEW_TITLE'],test['REVIEW_TITLE']],axis = 0)

vectorizer = CountVectorizer()

transformer = TfidfTransformer()

#Count the frequency

count_vect = vectorizer.fit_transform(words)

#Derive the TF-IDF

tfidf = transformer.fit_transform(count_vect)  

word_names = vectorizer.get_feature_names()

weight = tfidf.toarray() 

word_vectors = pd.DataFrame(weight,columns=word_names)

#Columns that have no meanings or inappropriate

drop_columns=['00', '001', '007', '009', '016', '02', '03', '04', '043801', '06', '0p', '10', '100', '1000th', '100ft', '1055cm',

              '107', '1080', '1080p', '109069mptt', '10pc', '10pcs', '11', '1160us', '1186', '11ac', '12', '120hz', '12mic', 

              '12oz', '12v', '12w', '13', '1332', '14', '140', '143', '1430', '1478', '14mm', '15', '150', '152', '155', 

              '15a', '15in', '15mths', '15th', '16', '16gb', '16mm', '17', '1700', '172', '1750mah', '18', '180', '1800', '180s',

              '186t', '18u', '19', '1995', '1996', '1do', '1er', '1pc', '1st', '1x', '20', '200', '2000', '2002', '2003', '2005',

              '2006', '2007', '20070910', '2008', '200a', '200m', '200mm', '2010', '2011', '2012', '2013', '2014', '2015', '20pcs', 

              '20x30x3', '21', '210', '2155mx', '21932', '22', '2200mah', '22240', '2300mah', '23g', '24', '25', '250', '251', 

              '2557lmt', '25cttw', '26', '2605dn', '264', '27742', '28', '29', '2ds', '2g', '2k11', '2k13', '2l', '2nd', '2wd', 

              '2x', '2yo', '30', '300', '30c', '3131kit', '313us', '32', '3212', '32a320', '32oz', '33bv', '34', '34090', '3490', 

              '35', '350', '3500stb', '357', '35pc', '36', '360', '37', '375', '37v', '38', '380', '39', '3d', '3ds', '3g', '3m', 

              '3rd', '3weeks', '3yo', '40', '4000', '403', '40v', '42', '420', '42lf5800', '44', '45', '45mn', '45v', '46', '48', 

              '48b', '4g', '4gb', '4k', '4s', '4th', '4x', '50', '500', '5050', '50th', '5200', '525', '53', '55343', '55mm', 

              '571', '58mm', '5d', '5five', '5i', '5mm', '5s', '5x', '60', '600', '6093', '60d', '60hz', '60w', '610', '6201', '65', 

              '65012', '66', '6610', '69', '6g', '6mos', '6s', '6th', '6v', '70', '7113', '717', '720p', '74', '75', '7512', '78', '7a',

              '7d', '7e', '80', '800', '802', '80525', '8061', '81', '810', '84', '8gb', '8mm', '90', '910xt', '911', '911150', '92', 

              '9227c', '94', '9400', '95', '950', '9525', '97', '970', '98', '99', '9m', '9th', '9v', '9w', '9yo', 'a1', 'a2', 'aa', 

              'aaa', 'aarrrrgghh', 'ab', 'ítem', 'עלית', '很好like', '非常好']



word_vectors = word_vectors.drop(columns = drop_columns)
#Applying PCA for dimension reduction

pca = PCA(n_components=1000,

         svd_solver = 'randomized',

         random_state = 1234,)

pca.fit(word_vectors)

word_vectors_pca = pca.transform(word_vectors)

print("original shape:   ", word_vectors.shape)

print("transformed shape:", word_vectors_pca.shape)
alldata = alldata.drop(columns=['REVIEW_TITLE','REVIEW_TEXT'])

Xtr_number=train.shape[0]



X=np.concatenate((alldata[:Xtr_number].values, word_vectors_pca[:Xtr_number]),axis=1)

Y= X_train['LABEL'].ravel()

Xt=np.concatenate((alldata[Xtr_number:].values,word_vectors_pca[Xtr_number:]),axis=1)

print(X.shape)

print(Y.shape)

print(Xt.shape)
#Random_state and number of folds

SEED = 1234 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction
#The function to calculate the oof predictions        

def get_oof(clf, x_train, y_train, x_test, n_folds):

    oof_train = np.zeros(x_train.shape[0])

    oof_test = np.zeros(x_test.shape[0])

    oof_test_kf = np.zeros((n_folds, x_test.shape[0]))

    kf = KFold(n_splits= n_folds, shuffle=True, random_state=SEED)

    for i, (train_index, fold_index) in enumerate(kf.split(x_train,y_train)):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_fo = x_train[fold_index]



        clf.fit(x_tr, y_tr)

        

        oof_train[fold_index] = clf.predict(x_fo)

        oof_test_kf[i] = clf.predict(x_test)



    oof_test[:] = oof_test_kf.mean(axis=0)

    

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
#Logistic regression

lr_params = {

    'max_iter':100,

    'random_state' : SEED,

    'n_jobs': -1, 

}



lr = LogisticRegression(**lr_params)

lr.fit(X,Y)

lr_train=lr.predict(X)

lr_test=lr.predict(Xt)

print('LR Accuracy:',lr.score(X,Y))

coef=lr.coef_[0][0:34]



lr_oof_train, lr_oof_test = get_oof(lr, X, Y, Xt, n_folds=NFOLDS)

print('LR CV Accuracy:',accuracy_score(Y, lr_oof_train))

print('F1 Score:' ,f1_score(Y, lr_oof_train))


feature_coef = pd.DataFrame({'feature': alldata.columns, 'score': coef},columns=['feature','score'])

feature_coef = feature_coef.sort_values(by='score',ascending=True)

data = [go.Bar(

            x=feature_coef['score'],

            y=feature_coef['feature'],

            orientation = 'h'

)]

layout = go.Layout(height=900, width=800,title='Logistic regression original feature coefficient')

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
#SVC

svc_params = {

    'max_iter': 250,

    'n_jobs': -1, 

    'random_state' : SEED,

    'warm_start': True,

}



svc = SGDClassifier(**svc_params)

svc.fit(X,Y)

svc_train=svc.predict(X)

svc_test=svc.predict(Xt)

print('SVC Accuracy:',svc.score(X,Y))



svc_oof_train, svc_oof_test = get_oof(svc, X, Y, Xt, n_folds=NFOLDS)

print('SVC CV Accuracy:',accuracy_score(Y, svc_oof_train))
# Random Forest 

rf_params = {

    'n_jobs': -1,

    'n_estimators': 20,

    'max_features' : 0.3,

    'max_depth' : 10,

    'verbose': 0,

    'random_state':SEED,

    'oob_score' : True

}



rf = RandomForestClassifier(**rf_params)

rf.fit(X,Y)

rf_train=rf.predict(X)

rf_test=rf.predict(Xt)

print('RF Accuracy:',rf.score(X,Y))



rf_oof_train, rf_oof_test= get_oof(rf, X, Y, Xt, n_folds=NFOLDS) 

print('RF CV Accuracy:',accuracy_score(Y, rf_oof_train))
# Gradient Boosting 

gb_params = {

    'n_estimators': 20,

    'max_features' : 0.3,

    'max_depth': 10,

    'verbose': 0,

    'random_state':SEED

}



gb = GradientBoostingClassifier(**gb_params)

gb.fit(X,Y)

gb_train=gb.predict(X)

gb_test=gb.predict(Xt)

print('GB Accuracy:',gb.score(X,Y))



gb_oof_train, gb_oof_test= get_oof(gb, X, Y, Xt, n_folds=NFOLDS) 

print('GB CV Accuracy:',accuracy_score(Y, gb_oof_train))
#The class for stacking models to build a new model containing 2 tiers of models

class StackedModel():

    def __init__(self, base_models, meta_model, n_folds):

        self.base_models = base_models

        self.meta_model = clone(meta_model)

        self.n_folds = n_folds

   

    # Fit the data on clones of the tier 1 models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=SEED)

        

        # Train cloned tier 1 models then create out-of-fold predictions

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kf.split(X, y):

                instance = clone(model)

                instance.fit(X[train_index], y[train_index])

                self.base_models_[i].append(instance)

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # Now train the cloned  tier 2 model using the out-of-fold predictions as new feature

        self.meta_model.fit(out_of_fold_predictions, y)

        return 

   

    #Do the predictions of all base models on the test data and use the averaged predictions as meta-features 

    #for the final prediction which is done through the tier 2 model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1) 

            for base_models in self.base_models_ ])

        return self.meta_model.predict(meta_features)

    

    def coef_(self):

        return self.meta_model.coef_

    

    def intercept_(self):

        return self.meta_model.intercept_
lr_params = {

    'max_iter':100,

    'random_state' : SEED,

    'n_jobs': -1, 

}



tier2 = LogisticRegression(**lr_params)

stacked_model = StackedModel(base_models =(rf, gb, svc, lr), 

                            meta_model = tier2, n_folds = NFOLDS) 

stacked_model.fit(X, Y)

stacked_train = stacked_model.predict(X)

stacked_test = stacked_model.predict(Xt)

print('Stacked Accuracy:',accuracy_score(Y, stacked_train))

print('F1 Score:' ,f1_score(Y, stacked_train))
#Showing the coefficient of the stacked model

print('Stacked model intercept:',stacked_model.intercept_()[0])

model_importance = pd.DataFrame({'Model': ['RF', 'GB', 'SVC', 'LR'], 'Score': stacked_model.coef_()[0]},columns=['Model','Score'])

model_importance = model_importance.sort_values(by='Score',ascending=True)

data = [go.Bar(

            x=model_importance['Score'],

            y=model_importance['Model'],

            orientation = 'h'

)]

layout = go.Layout(height=400, width=600,title='Stacked model coefficient')

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
submission = pd.DataFrame(columns=['REVIEW_ID','LABEL'])

submission['REVIEW_ID']=ID

submission['LABEL']=stacked_test.astype(int)

submission=submission.set_index('REVIEW_ID')

submission.to_csv('Result.csv',index=True)

print(submission.head())