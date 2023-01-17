import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
amazonmusic = pd.read_csv('/kaggle/input/amazon-music-reviews/Musical_instruments_reviews.csv')
amazonmusic.head()
amazonmusic.drop(['reviewerID', 'asin', 'reviewerName', 'helpful', 'unixReviewTime', 'reviewTime'],axis=1,inplace=True)
amazonmusic.isna().mean()
amazonmusic.fillna(amazonmusic.mean())
amazonmusic.isna().mean()
def rating(x):
    if x == 5:
        return('Excellent')
    elif x==4:
        return('Excellent')
    elif x == 3:
        return('ok')
    elif x==2:
        return('not good')
    elif x==1:
        return('not good')
amazonmusic['overall'] = amazonmusic['overall'].apply(lambda x: rating(x))
amazonmusic['overall'].value_counts()
amazonmusic = amazonmusic.astype(str)
import seaborn as sns
sns.countplot(x='overall',data=amazonmusic,orient='v')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(amazonmusic['summary'].astype(str))
amazonmusic['summary'] = le.transform(amazonmusic['summary'].astype(str))
le.fit(amazonmusic['reviewText'].astype(str))
amazonmusic['reviewText'] = le.transform(amazonmusic['reviewText'].astype(str))
from sklearn.model_selection import train_test_split
y = amazonmusic['overall']
X = amazonmusic.drop(['overall'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=700)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
dc = DecisionTreeClassifier()
amazonmusic.info()
dc.fit(X_train,y_train)
predictions = dc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
ab = AdaBoostClassifier()
ab.fit(X_train,y_train)
pre = ab.predict(X_test)
print(classification_report(y_test,pre))
