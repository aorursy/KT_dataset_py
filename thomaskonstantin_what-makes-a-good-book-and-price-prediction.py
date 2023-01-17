# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

sns.set_style('darkgrid')
b_data = pd.read_csv('/kaggle/input/top-270-rated-computer-science-programing-books/prog_book.csv')

b_data.head(3)
def average_word_length(sir):

    splited = sir.split(' ')

    aux = 0

    for word in splited:

        aux += len(word)

    aux/=len(splited) 

    return aux



def number_of_words(sir):

    splited = sir.split(' ')

    return len(splited)

b_data['Title_Average_Word_Length'] = b_data.Book_title.apply(average_word_length)

b_data['Title_Number_Of_Words'] = b_data.Book_title.apply(number_of_words)

b_data['Description_Average_Word_Length'] = b_data.Description.apply(average_word_length)

b_data['Description_Number_Of_Words'] = b_data.Description.apply(number_of_words)

lencoder = LabelEncoder()

lencoder.fit(b_data.Type)

b_data.Type = lencoder.transform(b_data.Type)
b_data.Reviews = b_data.Reviews.apply(lambda x : int(x.replace(',','')))
b_data.head(3)

plt.figure(figsize=(20,11))

ax = sns.distplot(b_data.Rating,label="Ratings",color='green')

ax.set_xlabel("Rating",fontsize=20)

ax.set_ylabel("Density",fontsize=20)

textstr = '\n'.join(

        (r'$\mu=%.2f$' % (b_data.Rating.mean(),), r'$\mathrm{median}=%.2f$' % (b_data.Rating.median(),),

         r'$\sigma=%.2f$' % (b_data.Rating.std(),)))

props = dict(boxstyle='round', facecolor='green', alpha=0.5)

ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,

            verticalalignment='top', bbox=props)



ax.set_title('Distribution Of Rating Scores Across Our Dataset',fontsize=21)

plt.show()
plt.figure(figsize=(20,11))

ax = sns.distplot(b_data.Reviews,label="Reviews",color='teal')

ax.set_xlabel("Reviews",fontsize=20)

ax.set_ylabel("Density",fontsize=20)

textstr = '\n'.join(

        (r'$\mu=%.2f$' % (b_data.Reviews.mean(),), r'$\mathrm{median}=%.2f$' % (b_data.Reviews.median(),),

         r'$\sigma=%.2f$' % (b_data.Reviews.std(),)))

props = dict(boxstyle='round', facecolor='teal', alpha=0.5)

ax.text(0.85, 0.85, textstr, transform=ax.transAxes, fontsize=14,

            verticalalignment='center', bbox=props)



ax.set_title('Distribution Of Reviews Across Our Dataset',fontsize=21)

plt.show()
b_data.Reviews = b_data.Reviews.replace(0,1)

b_data.Reviews = np.log(b_data.Reviews)

plt.figure(figsize=(20,11))

ax = sns.distplot(b_data.Reviews,label="Reviews",color='teal')

ax.set_xlabel("Reviews",fontsize=20)

ax.set_ylabel("Density",fontsize=20)

textstr = '\n'.join(

        (r'$\mu=%.2f$' % (b_data.Reviews.mean(),), r'$\mathrm{median}=%.2f$' % (b_data.Reviews.median(),),

         r'$\sigma=%.2f$' % (b_data.Reviews.std(),)))

props = dict(boxstyle='round', facecolor='teal', alpha=0.5)

ax.text(0.85, 0.85, textstr, transform=ax.transAxes, fontsize=14,

            verticalalignment='center', bbox=props)



ax.set_title('Distribution Of Reviews Across Our Dataset After Log Transformation',fontsize=21)

plt.show()
plt.figure(figsize=(20,11))

ax = sns.distplot(b_data.Number_Of_Pages,label="Number_Of_Pages",color='red')

ax.set_xlabel("Number Of Pages",fontsize=20)

ax.set_ylabel("Density",fontsize=20)

textstr = '\n'.join(

        (r'$\mu=%.2f$' % (b_data.Number_Of_Pages.mean(),), r'$\mathrm{median}=%.2f$' % (b_data.Number_Of_Pages.median(),),

         r'$\sigma=%.2f$' % (b_data.Number_Of_Pages.std(),)))

props = dict(boxstyle='round', facecolor='red', alpha=0.5)

ax.text(0.85, 0.85, textstr, transform=ax.transAxes, fontsize=14,

            verticalalignment='center', bbox=props)



ax.set_title('Distribution Of Book Page Number Counts',fontsize=21)

plt.show()
plt.figure(figsize=(20,11))

ax = sns.countplot(b_data.Type)

ax.set_xticklabels(lencoder.inverse_transform([0,1,2,3,4,5]))

ax.set_xlabel("Book Type",fontsize=20)

ax.set_ylabel("Count",fontsize=20)

ax.set_title('Distibution Of Different Book Types In Our Data',fontsize=22)
plt.figure(figsize=(20,11))

ax = sns.distplot(b_data.Price,label="Price")

ax.set_xlabel("Price",fontsize=20)

ax.set_ylabel("Density",fontsize=20)

textstr = '\n'.join(

        (r'$\mu=%.2f$' % (b_data.Price.mean(),), r'$\mathrm{median}=%.2f$' % (b_data.Price.median(),),

         r'$\sigma=%.2f$' % (b_data.Price.std(),)))

props = dict(boxstyle='round', facecolor='red', alpha=0.5)

ax.text(0.85, 0.85, textstr, transform=ax.transAxes, fontsize=14,

            verticalalignment='center', bbox=props)



ax.set_title('Distribution Of Book Prices',fontsize=21)

plt.show()
from wordcloud import WordCloud,STOPWORDS

import re



stopwords = list(STOPWORDS)



title_w = ''



for word in b_data.Book_title:

    word = word.lower()

    splited = re.findall(r'\b[A-Za-z]+\b',word)

    splited = [w for w in splited if w not in stopwords]

    title_w += ' '.join(splited)+ ' '





wordcloud = WordCloud(width = 1100, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 8).generate(title_w) 

  

# plot the WordCloud image                        

plt.figure(figsize = (18, 11), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
desc_w = ''



for word in b_data.Description:

    word = word.lower()

    splited = re.findall(r'\b[A-Za-z]+\b',word)

    splited = [w for w in splited if w not in stopwords]

    desc_w += ' '.join(splited)+ ' '





wordcloud = WordCloud(width = 1100, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 8).generate(desc_w) 

  

# plot the WordCloud image                        

plt.figure(figsize = (18, 11), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
plt.figure(figsize=(20,11))

ax = sns.jointplot(x=b_data.Price,y=b_data.Rating,height=10,kind='kde',cmap='mako')

#ax.set_xlabel("Price",fontsize=20)

#ax.set_ylabel("Density",fontsize=20)

#ax.set_title('Distribution Of Book Prices',fontsize=21)

plt.show()
correlations = b_data.corr('pearson')

plt.figure(figsize=(20,11))

ax = sns.clustermap(correlations,annot=True,cmap='coolwarm',figsize=(20,11))
b_data = b_data[b_data['Number_Of_Pages']<1500]

b_data = b_data[b_data['Price']<150]
from sklearn.metrics import mean_squared_error,r2_score

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor
features = ['Number_Of_Pages','Type','Title_Average_Word_Length']

train_x,test_x,train_y,test_y = train_test_split(b_data[features],b_data.Price)



LR_pipe = Pipeline(steps=[('model',LinearRegression())])



LR_scores = np.sqrt(-1*cross_val_score(LR_pipe,b_data[features],b_data.Price,cv=6,scoring='neg_mean_squared_error'))



plt.figure(figsize=(20,11))

ax = sns.pointplot(x=np.arange(0,6),y=LR_scores)

ax.set_title('Cross Validation RMSE for LinearRegression',fontsize=20)

ax.set_ylabel('RMSE',fontsize=18)

ax.set_xlabel('Fold Number',fontsize=18)
print("LinearRegression Average Cross Validation Score:",LR_scores.mean())
RF_pipe = Pipeline(steps=[('model',RandomForestRegressor(n_estimators=50,max_leaf_nodes=15,random_state=42))])



RF_scores = np.sqrt(-1*cross_val_score(RF_pipe,b_data[features],b_data.Price,cv=6,scoring='neg_mean_squared_error'))



plt.figure(figsize=(20,11))

ax = sns.pointplot(x=np.arange(0,6),y=RF_scores,color='teal')

ax.set_title('Cross Validation RMSE for RandomForest',fontsize=20)

ax.set_ylabel('RMSE',fontsize=18)

ax.set_xlabel('Fold Number',fontsize=18)
print("RandomForest Average Cross Validation Score:",RF_scores.mean())
plt.figure(figsize=(20,11))

ax = sns.residplot(x=b_data.Number_Of_Pages,y=b_data.Price)

ax.set_title('Absolute residuals vs Price',fontsize=19,fontweight='bold')
plt.figure(figsize=(20,11))

ax = sns.scatterplot(x=b_data.Number_Of_Pages,y=b_data.Price,hue=b_data.Type,palette='coolwarm')
from sklearn.preprocessing import PolynomialFeatures,StandardScaler

pf = PolynomialFeatures(degree = 2)

PR_pipe = Pipeline(steps = [('scale',StandardScaler()),('pf',pf), ('model',LinearRegression()) ])







PR_scores = np.sqrt(-1*cross_val_score(PR_pipe,b_data[features],b_data.Price,cv=6,scoring='neg_mean_squared_error'))





plt.figure(figsize=(20,11))

ax = sns.pointplot(x=np.arange(0,6),y=PR_scores)

ax.set_title('Cross Validation RMSE for Polynomial Regression',fontsize=20)

ax.set_ylabel('RMSE',fontsize=18)

ax.set_xlabel('Fold Number',fontsize=18)
print("Polynomial Regression Average Cross Validation Score:",PR_scores.mean())
KNR_pipe = Pipeline(steps=[('model',KNeighborsRegressor(n_neighbors=25))])



KNR_scores = np.sqrt(-1*cross_val_score(KNR_pipe,b_data[features],b_data.Price,cv=6,scoring='neg_mean_squared_error'))



plt.figure(figsize=(20,11))

ax = sns.pointplot(x=np.arange(0,6),y=KNR_scores,color='teal')

ax.set_title('Cross Validation RMSE for KNN',fontsize=20)

ax.set_ylabel('RMSE',fontsize=18)

ax.set_xlabel('Fold Number',fontsize=18)
print("KNN Average Cross Validation Score:",KNR_scores.mean())
b_data = b_data.sample(frac=1)

LR_pipe.fit(b_data[features],b_data.Price)

PR_pipe.fit(b_data[features],b_data.Price)

KNR_pipe.fit(b_data[features],b_data.Price)

RF_pipe.fit(b_data[features],b_data.Price)
LR_Predict = LR_pipe.predict(b_data[features])



plt.figure(figsize=(20,11))

ax= sns.lineplot(x=np.arange(0,b_data.shape[0]),y=b_data.Price,label='Actual',color='green')

ax= sns.lineplot(x=np.arange(0,b_data.shape[0]),y=LR_Predict,label='LinearReg Prediciton',color='red')

ens_Predict =LR_pipe.predict(b_data[features])*0.1+PR_pipe.predict(b_data[features])*0.3 + RF_pipe.predict(b_data[features])*0.6



plt.figure(figsize=(20,11))

ax= sns.lineplot(x=np.arange(0,b_data.shape[0]),y=b_data.Price,label='Actual',color='green')

ax= sns.lineplot(x=np.arange(0,b_data.shape[0]),y=ens_Predict,label='Stacked Prediciton',color='red')

print('Stacked Model RMSE: ',np.sqrt(mean_squared_error(ens_Predict,b_data.Price)))