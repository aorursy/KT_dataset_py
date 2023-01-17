import pandas as pd
import numpy as np

df=pd.read_csv('../input/Amazon_Unlocked_Mobile.csv')

df=df.sample(frac=0.1,random_state=10)
df.head()
df.shape
#Droping missing values
df.dropna(inplace=True)


#Removing any neutral rating =3
df=df[df['Rating']!=3]

#encode 4 an 5 as 1
#1 and 2 as 0
df['Positively Rated']=np.where(df['Rating']>3,1,0)
df.head()
df['Positively Rated'].mean()
from sklearn.model_selection import train_test_split

#spliting data into training and test 
X_train,X_test,y_train,y_test=train_test_split(df['Reviews'],df['Positively Rated'],random_state=0)
X_train.iloc[0]
X_train.shape
from sklearn.feature_extraction.text import CountVectorizer

vect=CountVectorizer().fit(X_train)
vect.get_feature_names()[::2000]
len(vect.get_feature_names())
X_train_vectorized = vect.transform(X_train)

X_train_vectorized
from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(X_train_vectorized,y_train)
from sklearn.metrics import roc_auc_score

predictions=model.predict(vect.transform(X_test))
print('AUC:',roc_auc_score(y_test,predictions))
# get the feature names as numpy array
feature_names = np.array(vect.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
from sklearn.feature_extraction.text import TfidfVectorizer

# Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 5
vect = TfidfVectorizer(min_df=5).fit(X_train)
len(vect.get_feature_names())
X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))
#features with smallest and largest tfidf
feature_names = np.array(vect.get_feature_names())

sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()

print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))
sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
# These reviews are treated the same by our current model
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))
# Fit the CountVectorizer to the training data specifiying a minimum 
# document frequency of 5 and extracting 1-grams and 2-grams
vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

len(vect.get_feature_names())
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))
feature_names = np.array(vect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

