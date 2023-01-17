import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import eli5
from IPython.display import Image
import os

!ls ../input/

mpl.style.use('ggplot')
mpl.rcParams['patch.force_edgecolor']=True
df = pd.read_csv('../input/detect-sarcasm-in-comments/Train.csv').reset_index(drop=True)
df.head()
df.isnull().sum()
df.columns
df['date'] = pd.to_datetime(df['date'],yearfirst=True)
df['year'] = df['date'].apply(lambda d: d.year)

comments_by_year = df.groupby('year')['label'].agg([np.sum,np.mean])
comments_by_year
plt.figure(figsize=(8,6))
comments_by_year['mean'].plot(kind='line')
plt.ylabel('Mean Sarcasm')
plt.title('Rate of Sarcasm on Reddit')
print('Minimum and Maximum Scores')
df['score'].min(), df['score'].max()
# Mean and STD of Score Rating
mean = df['score'].mean()
std = df['score'].std()
print('Mean Score and Standard Deviation')
mean, std
# Distribution of Scores for Sarcastic and Non-Sarcastic Comments
plt.figure(figsize=(8,6))
df[(df['score'].abs()<(10-((df['score'].abs()-mean)/std))) & (df['label']==1)]['score'].hist(alpha=0.5,label='Sarcastic')
df[(df['score'].abs()<(10-((df['score'].abs()-mean)/std))) & (df['label']==0)]['score'].hist(alpha=0.5,label='Not Sarcastic')
plt.yscale('linear')
plt.ylabel('Frequency')
plt.xlabel('Score')
plt.legend()
plt.title('Scores for Sarcastic vs. None-Sarcastic Comments')
# Distribution of LogBase10 Scores for Sarcastic and Non-Sarcastic Comments 
plt.figure(figsize=(8,6))
df[(df['score'].abs()<(10-((df['score'].abs()-mean)/std))) & (df['label']==1)]['score'].hist(alpha=0.5,label='Sarcastic')
df[(df['score'].abs()<(10-((df['score'].abs()-mean)/std))) & (df['label']==0)]['score'].hist(alpha=0.5,label='Not Sarcastic')
plt.yscale('log')
plt.ylabel('Log Base10 Frequency')
plt.xlabel('Score')
plt.legend()
plt.title('LogBase10-Scores for Sarcastic vs. None-Sarcastic Comments')
# Natural Log Length of Comments for Sarcastic and Non-Sarcastic Comments
plt.figure(figsize=(8,6))
df['log_comment'] = df['comment'].apply(lambda text: np.log1p(len(text)))
df[df['label']==1]['log_comment'].hist(alpha=0.5,label='Sarcastic')
df[df['label']==0]['log_comment'].hist(alpha=0.5,label='Sarcastic')
plt.legend()
plt.title('Natural Log Length of Comments')
# Sarcastic Comments by Reddit Users
df.groupby('user')['label'].agg([np.sum,np.mean,np.size]).sort_values(by='sum',ascending=False).head(5)
df.columns
X = df['comment']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=28)
model = Pipeline([('tfidf',TfidfVectorizer(min_df=2)),
                    ('logit',LogisticRegression(solver='lbfgs', max_iter=3000))])
parameters = {'tfidf__ngram_range':[(1,1),(1,2)],'tfidf__use_idf':(True,False)}
grid = GridSearchCV(estimator=model,param_grid=parameters,verbose=2,n_jobs=-1, cv=3, refit=True)
%%time
grid.fit(X_train,y_train)
grid.best_params_
%%time
chosen_model = Pipeline([('tfidf',TfidfVectorizer(min_df=2, ngram_range=(1,2),use_idf=True)),
                    ('logit',LogisticRegression(solver='lbfgs', max_iter=3000))])
chosen_model.fit(X_train,y_train)
predictions = chosen_model.predict(X_test)
print('Accuracy Score: {:.2%}'.format(accuracy_score(y_test,predictions)),'\n')
# Plot Confusion Matrix

cm = pd.DataFrame(confusion_matrix(y_test,predictions), index=['NOT SARCASTIC','SARCASTIC'],columns=['NOT SARCASTIC','SARCASTIC'])

fig = plt.figure(figsize=(8,6))
ax = sns.heatmap(cm,annot=True,cbar=False, cmap='Blues',linewidths=0.5,fmt='.0f')
ax.set_title('SARCASM DETECTION CONFUSION MATRIX',fontsize=16,y=1.25)
ax.set_ylabel('ACTUAL',fontsize=14)
ax.set_xlabel('PREDICTED',fontsize=14)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.tick_params(labelsize=12)