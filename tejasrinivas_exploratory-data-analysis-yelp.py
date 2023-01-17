# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings("ignore")


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
yelp_business = pd.read_csv('../input/yelp_business.csv')
yelp_business.head()
yelp_business.isnull().sum()
plt.figure(figsize=(12,10))
f = sns.heatmap(yelp_business.isnull(),yticklabels=False, cbar=False, cmap = 'viridis')
plt.figure(figsize=(6,6))
sns.countplot(x='is_open',data=yelp_business);
#let's look at the number of unique values in the stars variable.

yelp_business['stars'].nunique()
sns.countplot(x='stars',data=yelp_business);
sns.distplot(yelp_business['review_count'].apply(np.log1p));
yelp_business_attributes = pd.read_csv('../input/yelp_business_attributes.csv')

by_state = yelp_business.groupby('state')
import squarify    # pip install squarify (algorithm for treemap)
plt.figure(figsize=(12,12))

a = by_state['business_id'].count()

a.sort_values(ascending=False,inplace=True)

squarify.plot(sizes= a[0:15].values, label= a[0:15].index, alpha=0.9)

plt.axis('off')
plt.tight_layout()


business_cats=';'.join(yelp_business['categories'])
cats=pd.DataFrame(business_cats.split(';'),columns=['category'])
cats_ser = cats.category.value_counts()


cats_df = pd.DataFrame(cats_ser)
cats_df.reset_index(inplace=True)

plt.figure(figsize=(12,10))
f = sns.barplot( y= 'index',x = 'category' , data = cats_df.iloc[0:20])
f.set_ylabel('Category')
f.set_xlabel('Number of businesses');
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

plt.figure(figsize=(12,10))

wordcloud = WordCloud(background_color='white',
                          width=1200,
                      stopwords = STOPWORDS,
                          height=1000
                         ).generate(str(yelp_business['name']))


plt.imshow(wordcloud)
plt.axis('off');
from sklearn.cross_validation import train_test_split
yelp_business.head()
X = pd.get_dummies(yelp_business['city'])
yelp_business = pd.concat([yelp_business,X], axis=1)
del X;
drop_cols = ['business_id',
 'name',
 'neighborhood',
 'address',
 'city',
 'state',
 'postal_code',
  'is_open',           
 'categories']
cols = [ i for i in yelp_business.columns if i not in drop_cols]
cols1 = ['latitude',
 'longitude',
 'stars',
 'review_count']

X = yelp_business[cols1]
y = yelp_business['is_open']
from imblearn.over_sampling import SMOTE
train_X, test_X, train_y, test_y = train_test_split(X,y,test_size = 0.3, random_state = 42)
train_X.fillna(0,inplace=True)
test_X.fillna(0,inplace=True)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(train_X, train_y)
X_res = pd.DataFrame(X_res)
y_res = pd.DataFrame(y_res)
test_X = pd.DataFrame(test_X)
test_y = pd.DataFrame(test_y)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
L = [0.0001,0.001,0.01,0.1,1,10]
accuracy = []
for i in L:
    LR = LogisticRegression(C=i)
    LR.fit(X_res,y_res)
    pred_y = LR.predict(test_X)
    
    accuracy.append(accuracy_score(test_y,pred_y))
accuracy
y_res[0].value_counts()
LR = LogisticRegression(C=0.001)
LR.fit(X_res,y_res)
pred_y = LR.predict(test_X)
confusion_matrix(test_y,pred_y)
from sklearn.metrics import accuracy_score

accuracy_score(test_y,pred_y)
review = pd.read_csv('../input/yelp_review.csv')
checkin = pd.read_csv('../input/yelp_checkin.csv')
review.head()
review_busines = review.groupby(by=['review_id'])
review_businesid = pd.DataFrame()
review_businesid['25-percentile'] = np.percentile(review_busines['stars'],25)
review_businesid['50-percentile'] = np.percentile(review_busines['stars'],50)
review_businesid['75-percentile'] = np.percentile(review_busines['stars'],75)
review_businesid['Mean'] = review_busines['stars'].mean()



checkin.head()







yelp_business_att = pd.read_csv('../input/yelp_business_attributes.csv')
plt.figure(figsize=(12,10))
f = sns.heatmap(yelp_business_att.isnull(),cbar=False,yticklabels=False,cmap='viridis')
sns.factorplot(yelp_business_attributes['AcceptsInsurance'])
yelp_business_hours = pd.read_csv('../input/yelp_business_hours.csv')
yelp_business_hours.head()
yelp_tip = pd.read_csv('../input/yelp_tip.csv')
yelp_tip.head()
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

plt.figure(figsize=(12,10))

wordcloud = WordCloud(background_color='white',
                          width=1200,
                      stopwords = STOPWORDS,
                          height=1000
                         ).generate(str(yelp_tip['text']))


plt.imshow(wordcloud)
plt.axis('off')
plt.show()

