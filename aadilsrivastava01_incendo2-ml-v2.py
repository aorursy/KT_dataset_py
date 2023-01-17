import numpy as np

import pandas as pd 

import string
train = pd.read_csv('../input/train_dataset.csv')

test = pd.read_csv('../input/test_dataset.csv')
train.head()
train.shape
train.dtypes
train.isna().sum()
test.isna().sum()
import matplotlib.pyplot as plt

%matplotlib inline
train['Essayset'].value_counts(normalize=True).plot.bar()
train['max_score'].value_counts().plot.bar()
from collections import defaultdict



def count_value(df):

        dic = defaultdict(int)

        dic1 = defaultdict(int)

        df = df.dropna()

        for val in df['Essayset']:

            if val in [1.0,2.0,5.0,6.0]:

                dic[val]+=1

            else:

                dic1[val]+=1

        return dic,dic1
X = train.copy()

X_test = test.copy()
count_value(X)
X.loc[(X['Essayset'].isna() == True) & (X['max_score'] == 3),['Essayset']] = 6.0

X.loc[(X['Essayset'].isna() == True) & (X['max_score'] == 2),['Essayset']] = 8.0
import seaborn as sns
plt.figure(1,figsize=(16,16))





plt.subplot(321)

sns.distplot(X.loc[X['score_3'].isna()!=True,['score_3']])

plt.subplot(322)

sns.boxplot(y=X.loc[X['score_3'].isna()!=True,['score_3']])



plt.subplot(323)

sns.distplot(X.loc[X['score_4'].isna()!=True,['score_4']])

plt.subplot(324)

sns.boxplot(y=X.loc[X['score_4'].isna()!=True,['score_4']])



plt.subplot(325)

sns.distplot(X.loc[X['score_5'].isna()!=True,['score_5']])

plt.subplot(326)

sns.boxplot(y=X.loc[X['score_5'].isna()!=True,['score_5']])



plt.show()
mean_3_3 = np.mean(train.loc[train['max_score']==3,'score_3'])

mean_4_3 = np.mean(train.loc[train['max_score']==3,'score_4'])

mean_5_3 = np.mean(train.loc[train['max_score']==3,'score_5'])



mean_3_2 = np.mean(train.loc[train['max_score']==2,'score_3'])

mean_4_2 = np.mean(train.loc[train['max_score']==2,'score_4'])

mean_5_2 = np.mean(train.loc[train['max_score']==2,'score_5'])
X.loc[(X['score_3'].isna()==True) & (X['max_score']==3),'score_3'] = mean_3_3

X.loc[(X['score_4'].isna()==True) & (X['max_score']==3),'score_4'] = mean_4_3

X.loc[(X['score_5'].isna()==True) & (X['max_score']==3),'score_5'] = mean_5_3



X.loc[(X['score_3'].isna()==True) & (X['max_score']==2),'score_3'] = mean_3_2

X.loc[(X['score_4'].isna()==True) & (X['max_score']==2),'score_4'] = mean_4_2

X.loc[(X['score_5'].isna()==True) & (X['max_score']==2),'score_5'] = mean_5_2
X.isna().sum()
X['score'] = X.loc[:,['score_1','score_2','score_3','score_4','score_5']].mean(axis=1)

X = X.drop(labels = ['score_1','score_2','score_3','score_4','score_5'],axis =1)
X['score'] = X['score'].round().astype('category')

X.head()
X['score'].value_counts(normalize=True).plot.bar()
from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected = True)

import plotly.graph_objs as go
count = X['score'].value_counts()

labels = count.index

value = np.array((count/count.sum())*100)



plot = go.Pie(labels=labels,values = value)

layout = go.Layout(title='Target Value Distribution')

fig = go.Figure(data=[plot],layout=layout)

py.iplot(fig,filename='Target Distribution')
from wordcloud import STOPWORDS,WordCloud



def wcloud(text,title=None,figure_size=(24.0,16.0)):

    stopwords = set(STOPWORDS)

    stopwords = stopwords.union({'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'})

    

    wordcloud = WordCloud(stopwords=stopwords,random_state = 42,width=800, 

                    height=400,).generate(str(text))

    

    plt.figure(figsize=figure_size)

    plt.title(title,fontdict={'size': 40,})

    plt.imshow(wordcloud)
wcloud(X[X['score']==3]['EssayText'],'3 Marks: Essay_Text_Cloud')
wcloud(X[X['score']==2]['EssayText'],'2 Marks: Essay_Text_Cloud')
wcloud(X[X['score']==1]['EssayText'],'1 Mark: Essay_Text_Cloud')
wcloud(X[X['score']==0]['EssayText'],'0 Mark: Essay_Text_Cloud')
X['words'] = X['EssayText'].apply(lambda x: len(x.split()))

X_test['words'] = X_test['EssayText'].apply(lambda x: len(x.split()))



X['unique'] = X['EssayText'].apply(lambda x: len(set(x.split())))

X_test['unique'] = X_test['EssayText'].apply(lambda x: len(set(x.split())))



X['char'] = X['EssayText'].apply(lambda x: len(x))

X_test['char'] = X_test['EssayText'].apply(lambda x: len(x))



X['stop'] = X['EssayText'].apply(lambda x: len([word for word in str(x).lower().split() if word in set(STOPWORDS)]))

X_test['stop'] = X_test['EssayText'].apply(lambda x: len([word for word in str(x).lower().split() if word in set(STOPWORDS)]))



X['punct'] = X['EssayText'].apply(lambda x: len([punct for punct in str(x) if punct in string.punctuation]))

X_test['punct'] = X_test['EssayText'].apply(lambda x: len([punct for punct in str(x) if punct in string.punctuation]))



X['upper'] = X['EssayText'].apply(lambda x: len([word for word in x.split() if word.isupper()]))

X_test['upper'] = X_test['EssayText'].apply(lambda x: len([word for word in x.split() if word.isupper()]))



X['title'] = X['EssayText'].apply(lambda x: len([word for word in x.split() if word.istitle()]))

X_test['title'] = X_test['EssayText'].apply(lambda x: len([word for word in x.split() if word.istitle()]))



X['avg_word'] = X['EssayText'].apply(lambda x: (np.sum([len(word) for word in x.split()]))/len(x.split()))

X_test['avg_word'] = X_test['EssayText'].apply(lambda x: (np.sum([len(word) for word in x.split()]))/len(x.split()))
X.head()
X.iloc[15010].EssayText
# Truncate some extreme values for better visuals ##

X['words'].loc[X['words']>100] = 100 #truncation for better visuals

X['punct'].loc[X['punct']>10] = 10 #truncation for better visuals

X['char'].loc[X['char']>450] = 450 #truncation for better visuals



f, axes = plt.subplots(3, 1, figsize=(10,20))

sns.boxplot(x='score', y='words', data=X, ax=axes[0])

axes[0].set_xlabel('Score', fontsize=12)

axes[0].set_title("Number of words in each class", fontsize=15)



sns.boxplot(x='score', y='punct', data=X, ax=axes[1])

axes[1].set_xlabel('Score', fontsize=12)

axes[1].set_title("Number of characters in each class", fontsize=15)



sns.boxplot(x='score', y='char', data=X, ax=axes[2])

axes[2].set_xlabel('Score', fontsize=12)

axes[2].set_title("Number of punctuations in each class", fontsize=15)

plt.show()
# X['words'].loc[X['words']>100] = 100 #truncation for better visuals

# X['punct'].loc[X['punct']>10] = 10 #truncation for better visuals

# X['char'].loc[X['char']>450] = 450 #truncation for better visuals



f, axes = plt.subplots(5, 1, figsize=(10,30))

sns.boxplot(x='score', y='unique', data=X, ax=axes[0])

axes[0].set_xlabel('Score', fontsize=12)

axes[0].set_title("Number of unique words in each class", fontsize=15)



sns.boxplot(x='score', y='stop', data=X, ax=axes[1])

axes[1].set_xlabel('Score', fontsize=12)

axes[1].set_title("Number of stop words in each class", fontsize=15)



sns.boxplot(x='score', y='upper', data=X, ax=axes[2])

axes[2].set_xlabel('Score', fontsize=12)

axes[2].set_title("Number of Upper Case in each class", fontsize=15)





sns.boxplot(x='score', y='title', data=X, ax=axes[3])

axes[3].set_xlabel('Score', fontsize=12)

axes[3].set_title("Number of Title Case in each class", fontsize=15)



sns.boxplot(x='score', y='avg_word', data=X, ax=axes[4])

axes[4].set_xlabel('Score', fontsize=12)

axes[4].set_title("Number of average in each class", fontsize=15)



plt.show()
X.clarity.value_counts()
X['clarity'].loc[(X['clarity'].isna()==True) & (X['score'] == 0) & (X['max_score']==3.0)] = 'worst'

X['clarity'].loc[(X['clarity'].isna()==True) & (X['score'] == 1) & (X['max_score']==3.0)] = 'average'

X['clarity'].loc[(X['clarity'].isna()==True) & (X['score'] == 2) & (X['max_score']==3.0)] = 'above_average'

X['clarity'].loc[(X['clarity'].isna()==True) & (X['score'] == 3) & (X['max_score']==3.0)] = 'excellent'



X['clarity'].loc[(X['clarity'].isna()==True) & (X['score'] == 0) & (X['max_score']==2.0)] = 'worst'

X['clarity'].loc[(X['clarity'].isna()==True) & (X['score'] == 1) & (X['max_score']==2.0)] = 'average'

X['clarity'].loc[(X['clarity'].isna()==True) & (X['score'] == 2) & (X['max_score']==2.0)] = 'excellent'



X['coherent'].loc[(X['coherent'].isna()==True) & (X['score'] == 0) & (X['max_score']==3.0)] = 'worst'

X['coherent'].loc[(X['coherent'].isna()==True) & (X['score'] == 1) & (X['max_score']==3.0)] = 'average'

X['coherent'].loc[(X['coherent'].isna()==True) & (X['score'] == 2) & (X['max_score']==3.0)] = 'above_average'

X['coherent'].loc[(X['coherent'].isna()==True) & (X['score'] == 3) & (X['max_score']==3.0)] = 'excellent'



X['coherent'].loc[(X['coherent'].isna()==True) & (X['score'] == 0) & (X['max_score']==2.0)] = 'worst'

X['coherent'].loc[(X['coherent'].isna()==True) & (X['score'] == 1) & (X['max_score']==2.0)] = 'average'

X['coherent'].loc[(X['coherent'].isna()==True) & (X['score'] == 2) & (X['max_score']==2.0)] = 'excellent'
X.isna().sum()
from sklearn.preprocessing import LabelEncoder
le_clarity = LabelEncoder()

le_coherent = LabelEncoder()



X['clarity'] = le_clarity.fit_transform(X['clarity'])

X_test['clarity'] = le_clarity.transform(X_test['clarity'])



X['coherent'] = le_coherent.fit_transform(X['coherent'])

X_test['coherent'] = le_coherent.transform(X_test['coherent'])
X = X.drop(labels = ['ID','min_score','max_score','EssayText','avg_word'],axis=1)

X_test = X_test.drop(labels = ['ID','min_score','max_score','EssayText','avg_word'],axis=1)
X.head()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score,classification_report, log_loss,f1_score

from sklearn.svm import LinearSVC
kf = KFold(n_splits=5,shuffle=True,random_state=42)

eng_features = ['Essayset','clarity','coherent','words','unique','char','stop','punct','upper','title']

cv_scores = []

pred_val = np.zeros([X.shape[0]])

for train_index, val_index in kf.split(X):

    X_train, X_val = X.loc[train_index][eng_features].values,X.loc[val_index][eng_features].values

    y_train, y_val = X.loc[train_index]['score'].values,X.loc[val_index]['score'].values

    classifier = LogisticRegression(penalty= 'l1',class_weight='balanced', C = 1.0,

                                    multi_class = 'auto',solver='liblinear',random_state=42,max_iter=200)

    classifier.fit(X_train,y_train)

    pred_prob = classifier.predict_proba(X_val)

    pred = classifier.predict(X_val)

    pred_val[val_index] = pred

    print(accuracy_score(y_val,pred))
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier
kf = KFold(n_splits=5,shuffle=True,random_state=42)

eng_features = ['Essayset','clarity','coherent','words','unique','char','stop','punct','upper','title']

cv_scores = []

pred_val = np.zeros([X.shape[0]])

for train_index, val_index in kf.split(X):

    X_train, X_val = X.loc[train_index][eng_features].values,X.loc[val_index][eng_features].values

    y_train, y_val = X.loc[train_index]['score'].values,X.loc[val_index]['score'].values

    

    rfc = RandomForestClassifier(n_estimators = 70,random_state=42,n_jobs=-1,criterion='entropy',

                                min_samples_leaf=20,min_samples_split=10)

    rfc.fit(X_train,y_train)

    

    pred_prob = rfc.predict_proba(X_val)

    pred = rfc.predict(X_val)

    pred_val[val_index] = pred

    print(accuracy_score(y_val,pred))
kf = KFold(n_splits=5,shuffle=True,random_state=42)

eng_features = ['Essayset','clarity','coherent','words','unique','char','stop','punct','upper','title']

cv_scores = []

pred_val = np.zeros([X.shape[0]])

for train_index, val_index in kf.split(X):

    X_train, X_val = X.loc[train_index][eng_features].values,X.loc[val_index][eng_features].values

    y_train, y_val = X.loc[train_index]['score'].values,X.loc[val_index]['score'].values

    

    abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=42,

                                                                   criterion='entropy',

                                                                   min_samples_leaf=20,

                                                                   min_samples_split=10))

    abc.fit(X_train,y_train)

    

    pred_prob = abc.predict_proba(X_val)

    pred = abc.predict(X_val)

    pred_val[val_index] = pred

    print(accuracy_score(y_val,pred))
kf = KFold(n_splits=5,shuffle=True,random_state=42)

eng_features = ['Essayset','clarity','coherent','words','unique','char','stop','punct','upper','title']

cv_scores = []

pred_val = np.zeros([X.shape[0]])

for train_index, val_index in kf.split(X):

    X_train, X_val = X.loc[train_index][eng_features].values,X.loc[val_index][eng_features].values

    y_train, y_val = X.loc[train_index]['score'].values,X.loc[val_index]['score'].values

    

    etc = ExtraTreesClassifier(n_estimators = 30,random_state=42,n_jobs=-1,criterion='entropy',

                               min_samples_leaf=10,min_samples_split=10)

    etc.fit(X_train,y_train)

    

    pred_prob = etc.predict_proba(X_val)

    pred = etc.predict(X_val)

    pred_val[val_index] = pred

    print(accuracy_score(y_val,pred))
kf = KFold(n_splits=5,shuffle=True,random_state=42)

eng_features = ['Essayset','clarity','coherent','words','unique','char','stop','punct','upper','title']

cv_scores = []

pred_val = np.zeros([X.shape[0]])

for train_index, val_index in kf.split(X):

    X_train, X_val = X.loc[train_index][eng_features].values,X.loc[val_index][eng_features].values

    y_train, y_val = X.loc[train_index]['score'].values,X.loc[val_index]['score'].values

    

    knb = KNeighborsClassifier(n_neighbors=15, weights='distance',n_jobs=-1)

    knb.fit(X_train,y_train)

    

    pred_prob = knb.predict_proba(X_val)

    pred = knb.predict(X_val)

    pred_val[val_index] = pred

    print(accuracy_score(y_val,pred))
kf = KFold(n_splits=5,shuffle=True,random_state=42)

eng_features = ['Essayset','clarity','coherent','words','unique','char','stop','punct','upper','title']

cv_scores = []

pred_val = np.zeros([X.shape[0]])

for train_index, val_index in kf.split(X):

    X_train, X_val = X.loc[train_index][eng_features].values,X.loc[val_index][eng_features].values

    y_train, y_val = X.loc[train_index]['score'].values,X.loc[val_index]['score'].values

    

    mnb = MultinomialNB()

    mnb.fit(X_train,y_train)

    

    pred_prob = mnb.predict_proba(X_val)

    pred = mnb.predict(X_val)

    pred_val[val_index] = pred

    print(accuracy_score(y_val,pred))
pred_sub = rfc.predict(X_test)

X_test['essay_score'] = pred_sub

X_test.head()

sub = test.copy()

sub['essay_score'] = pred_sub

sub = sub.drop(labels=['min_score','max_score','clarity','coherent','EssayText'],axis=1)

sub.columns = ['id','essay_set', 'essay_score']

sub.head()
sub.to_csv(path_or_buf = 'submission.csv',index=False)