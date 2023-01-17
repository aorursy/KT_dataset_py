# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
users=pd.read_csv('/kaggle/input/bookcrossing-dataset/Book reviews/BX-Users.csv',sep=';',encoding='latin')
users.head()
i_cols = ['isbn', 'book_title' ,'book_author','year_of_publication', 'publisher', 'img_s', 'img_m', 'img_l']
books = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Books.csv', sep=';', names=i_cols, encoding='latin-1',low_memory=False)
books.head()
ratings=pd.read_csv('/kaggle/input/bookcrossing-dataset/Book reviews/BX-Book-Ratings.csv',sep=';',encoding='latin')
ratings=ratings.rename(columns={'ISBN':'isbn'})
ratings.head()
user=pd.merge(users,ratings,on='User-ID')
dataset=pd.merge(user,books,on='isbn')
dataset.head()
location=dataset.Location.str.split(', ',n=2,expand=True)
location.columns=['City','State','Country']
dataset['City']=location['City']
dataset['State']=location['State']
dataset['Country']=location['Country']
dataset.head()
data=dataset.drop(columns=['Age','Location','img_s','img_m','img_l'],axis=1)
data.head()
data['Reviews']=np.where(data['Book-Rating']>4,1,0)
data.head()
data['Book_title_City_Reviewed']=data[['book_title','City','State']].apply(lambda x:','.join(x),axis=1)
data.head()
new_data=data[data['Country']!='n/a']
new_data.head()
sns.pairplot(data)
plt.show()
print('Average Of Reviews with not good Remarks', len(data.loc[data['Reviews']==0])/len(data.loc[data['Reviews']]))
print('Average Of Reviews with  good Remarks', len(data.loc[data['Reviews']==1])/len(data.loc[data['Reviews']]))     
good_remarks=[len(x) for x in data.loc[data['Reviews']==1,'book_title']]              
notgood_remarks=[len(x) for x in data.loc[data['Reviews']==0,'book_title']] 
print('Average length of Book Title for good remarks',np.mean(good_remarks))
print('Average length of Book Title for not good remarks',np.mean(notgood_remarks))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
X=new_data['Book_title_City_Reviewed']
y=new_data['Reviews']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
vect=CountVectorizer(min_df=5,ngram_range=(1,2)).fit(X_train)
X_train_trans=vect.transform(X_train)
lr=MultinomialNB(alpha=0.1).fit(X_train_trans,y_train)
predict=lr.predict(vect.transform(X_test))
names=np.array(vect.get_feature_names())
coeff=lr.coef_[0].argsort()
print('roc_auc_score {:.2%}'.format(metrics.roc_auc_score(y_test,predict)))

largest=names[coeff[:-11:-1]]
smallest=names[coeff[:10]]
not_good_remarks=pd.DataFrame(coeff[:10],index=smallest,columns=['Count'])
good_remarks=pd.DataFrame(coeff[:-11:-1],index=largest,columns=['Count'])

not_good_remarks.head()
print('Average Frequency titles' ,np.mean(not_good_remarks['Count']))
good_remarks.head()
print('Average Frequency of titles',np.mean(good_remarks['Count']))
plt.plot(not_good_remarks['Count'])
plt.xticks(rotation=45)
plt.xlabel('Titles')
plt.ylabel('Frequency')
plt.title('NotGood_Remarks Title with its Frequency')
plt.show()
plt.plot(good_remarks['Count'])
plt.xticks(rotation=45)
plt.xlabel('Titles')
plt.ylabel('Frequency')
plt.title('Good_Remarks Title with its Frequency')
plt.show()
