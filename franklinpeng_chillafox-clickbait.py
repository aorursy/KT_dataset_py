# data analysis, splitting and wrangling

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



# visualization

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv("../input/clickbait-thumbnail-detection/train.csv") 

test = pd.read_csv("../input/clickbait-thumbnail-detection/test_2.csv")
# Calculate ratio of the number of Upper letter's in title divide by the total amount of word
df=train

df['Uppercase'] = df['title'].str.findall(r'[A-Z]').str.len()

df['Lowercase'] = df['title'].str.findall(r'[a-z]').str.len()

df['totalwords'] = df['title'].str.split().str.len()

df['upper/total ratio'] = df['Uppercase']/df['totalwords']
# calculate the Total words in side description.
df['Uppercase_Des'] = df['description'].str.findall(r'[A-Z]').str.len()

df['puncuation_Des'] = df['description'].str.count(r'[^\w\s]')

df['dash_count_Des']= df['description'].str.count(r'/')

df['Other_puncuation_Des']= df['puncuation_Des']- df['dash_count_Des']

df['totalwords_Des'] = df['description'].str.split().str.len()

df['Uppercaseratio_Des']=df['Uppercase_Des']/df['totalwords_Des']
# calculate like count Ratio.
df['dislikeCount']=df['dislikeCount'].replace(0,1)

df['like_ratio'] = round(df['likeCount']/df['dislikeCount'],2)
df['1'] = df['user_comment_1'].str.split().str.len()

df['2'] = df['user_comment_2'].str.split().str.len()

df['3'] = df['user_comment_3'].str.split().str.len()

df['4'] = df['user_comment_4'].str.split().str.len()

df['5'] = df['user_comment_5'].str.split().str.len()

df['6'] = df['user_comment_6'].str.split().str.len()

df['7'] = df['user_comment_7'].str.split().str.len()

df['8'] = df['user_comment_8'].str.split().str.len()

df['9'] = df['user_comment_9'].str.split().str.len()

df['10'] = df['user_comment_10'].str.split().str.len()

df['Comment_Word_Count'] = df['1'] + df['2'] + df['3'] + df['4'] + df['5'] + df['6'] + df['7'] + df['8'] + df['9'] + df['10']
df['1'] = df['user_comment_1'].str.count(r'[^\w\s]')

df['2'] = df['user_comment_2'].str.count(r'[^\w\s]')

df['3'] = df['user_comment_3'].str.count(r'[^\w\s]')

df['4'] = df['user_comment_4'].str.count(r'[^\w\s]')

df['5'] = df['user_comment_5'].str.count(r'[^\w\s]')

df['6'] = df['user_comment_6'].str.count(r'[^\w\s]')

df['7'] = df['user_comment_7'].str.count(r'[^\w\s]')

df['8'] = df['user_comment_8'].str.count(r'[^\w\s]')

df['9'] = df['user_comment_9'].str.count(r'[^\w\s]')

df['10'] = df['user_comment_10'].str.count(r'[^\w\s]')

df['Comment_Puncuation_Count'] = df['1'] + df['2'] + df['3'] + df['4'] + df['5'] + df['6'] + df['7'] + df['8'] + df['9'] + df['10']
df['1_'] = df['user_comment_1'].str.findall(r'[A-Z]').str.len()

df['2_'] = df['user_comment_2'].str.findall(r'[A-Z]').str.len()

df['3_'] = df['user_comment_3'].str.findall(r'[A-Z]').str.len()

df['4_'] = df['user_comment_4'].str.findall(r'[A-Z]').str.len()

df['5_'] = df['user_comment_5'].str.findall(r'[A-Z]').str.len()

df['6_'] = df['user_comment_6'].str.findall(r'[A-Z]').str.len()

df['7_'] = df['user_comment_7'].str.findall(r'[A-Z]').str.len()

df['8_'] = df['user_comment_8'].str.findall(r'[A-Z]').str.len()

df['9_'] = df['user_comment_9'].str.findall(r'[A-Z]').str.len()

df['10_'] = df['user_comment_10'].str.findall(r'[A-Z]').str.len()

df['Comment_UppercaseCount'] = df['1_'] + df['2_'] + df['3_'] + df['4_'] + df['5_'] + df['6_'] + df['7_'] + df['8_'] + df['9_'] + df['10_']
import time

from datetime import date

df['start_date']=df.timestamp.apply(lambda x: x[:10])

df['start_date']=pd.to_datetime(df['start_date'])

df['start_date']

today = date.today()

df['today']=pd.to_datetime(today)
df['video_publish_time']=df['today']-df['start_date']

df['video_publish_time']=df['video_publish_time'].dt.days.astype('int16')
df['viewCount/Date']=df['viewCount']/df['video_publish_time']

df['likecount/Date']=df['likeCount']/df['video_publish_time']

df['dislikecount/Date']=df['dislikeCount']/df['video_publish_time']
test['Uppercase'] = test['title'].str.findall(r'[A-Z]').str.len()

test['Lowercase'] = test['title'].str.findall(r'[a-z]').str.len()

test['totalwords'] = test['title'].str.split().str.len()

test['upper/total ratio'] = test['Uppercase']/test['totalwords']
test['Uppercase_Des'] = test['description'].str.findall(r'[A-Z]').str.len()

test['puncuation_Des'] = test['description'].str.count(r'[^\w\s]')

test['dash_count_Des']= test['description'].str.count('/')

test['Other_puncuation_Des']= test['puncuation_Des']- test['dash_count_Des']

test['totalwords_Des'] = test['description'].str.split().str.len()

test['Uppercaseratio_Des']=test['Uppercase_Des']/test['totalwords_Des']
test['1'] = test['user_comment_1'].str.split().str.len()

test['2'] = test['user_comment_2'].str.split().str.len()

test['3'] = test['user_comment_3'].str.split().str.len()

test['4'] = test['user_comment_4'].str.split().str.len()

test['5'] = test['user_comment_5'].str.split().str.len()

test['6'] = test['user_comment_6'].str.split().str.len()

test['7'] = test['user_comment_7'].str.split().str.len()

test['8'] = test['user_comment_8'].str.split().str.len()

test['9'] = test['user_comment_9'].str.split().str.len()

test['10'] = test['user_comment_10'].str.split().str.len()

test['Comment_Word_Count'] = test['1'] + test['2'] + test['3'] + test['4'] + test['5'] + test['6'] + test['7'] + test['8'] + test['9'] + test['10']
test['1'] = test['user_comment_1'].str.count(r'[^\w\s]')

test['2'] = test['user_comment_2'].str.count(r'[^\w\s]')

test['3'] = test['user_comment_3'].str.count(r'[^\w\s]')

test['4'] = test['user_comment_4'].str.count(r'[^\w\s]')

test['5'] = test['user_comment_5'].str.count(r'[^\w\s]')

test['6'] = test['user_comment_6'].str.count(r'[^\w\s]')

test['7'] = test['user_comment_7'].str.count(r'[^\w\s]')

test['8'] = test['user_comment_8'].str.count(r'[^\w\s]')

test['9'] = test['user_comment_9'].str.count(r'[^\w\s]')

test['10'] = test['user_comment_10'].str.count(r'[^\w\s]')

test['Comment_Puncuation_Count'] = test['1'] + test['2'] + test['3'] + test['4'] + test['5'] + test['6'] + test['7'] + test['8'] + test['9'] + test['10']
test['1_'] = test['user_comment_1'].str.findall(r'[A-Z]').str.len()

test['2_'] = test['user_comment_2'].str.findall(r'[A-Z]').str.len()

test['3_'] = test['user_comment_3'].str.findall(r'[A-Z]').str.len()

test['4_'] = test['user_comment_4'].str.findall(r'[A-Z]').str.len()

test['5_'] = test['user_comment_5'].str.findall(r'[A-Z]').str.len()

test['6_'] = test['user_comment_6'].str.findall(r'[A-Z]').str.len()

test['7_'] = test['user_comment_7'].str.findall(r'[A-Z]').str.len()

test['8_'] = test['user_comment_8'].str.findall(r'[A-Z]').str.len()

test['9_'] = test['user_comment_9'].str.findall(r'[A-Z]').str.len()

test['10_'] = test['user_comment_10'].str.findall(r'[A-Z]').str.len()

test['Comment_UppercaseCount'] = test['1_'] + test['2_'] + test['3_'] + test['4_'] + test['5_'] + test['6_'] + test['7_'] + test['8_'] + test['9_'] + test['10_']
test['start_date']=test.timestamp.apply(lambda x: x[:10])

test['start_date']=pd.to_datetime(test['start_date'])

test['today']=pd.to_datetime(today)
test['video_publish_time']=test['today']-test['start_date']

test['video_publish_time']=test['video_publish_time'].dt.days.astype('int16')
test['viewCount/Date']=test['viewCount']/test['video_publish_time']

test['likecount/Date']=test['likeCount']/test['video_publish_time']

test['dislikecount/Date']=test['dislikeCount']/test['video_publish_time']
train_data = df[["class", "viewCount", "commentCount","likeCount","dislikeCount","viewCount/Date","totalwords","upper/total ratio","totalwords_Des","Other_puncuation_Des","Uppercaseratio_Des",'Comment_Puncuation_Count',"Comment_UppercaseCount",'Comment_Word_Count']]

test_data = test[["ID", "viewCount", "commentCount","likeCount","dislikeCount","viewCount/Date","totalwords","upper/total ratio","totalwords_Des","Other_puncuation_Des","Uppercaseratio_Des",'Comment_Puncuation_Count',"Comment_UppercaseCount",'Comment_Word_Count']]
y = train_data["class"]

X = train_data.drop("class", axis=1)
X.head()
# split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2606)

print ("train_set_x shape: " + str(X_train.shape))

print ("train_set_y shape: " + str(y_train.shape))

print ("test_set_x shape: " + str(X_test.shape))

print ("test_set_y shape: " + str(y_test.shape))
X_train = X_train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

X_test = X_test.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# machine learning

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

ada_clf= AdaBoostClassifier(DecisionTreeClassifier(max_depth=15),n_estimators=260,algorithm="SAMME.R",learning_rate=0.72)

ada_clf.fit(X_train,y_train)

y_pred_4 = ada_clf.predict(X_test)

f1_score(y_pred_4,y_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_pred_4,y_test)
fin_test = test_data.drop("ID", axis=1)
X_X = train_data.drop("class", axis=1)

X_X = X.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

fin_test = fin_test.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
ada_fin_clf= AdaBoostClassifier(DecisionTreeClassifier(max_depth=15),n_estimators=260,algorithm="SAMME.R",learning_rate=0.72)

ada_fin_clf.fit(X_X,y)
Y_pred = ada_fin_clf.predict(fin_test)
test_data["class"] = Y_pred

test_data["class"] = test_data["class"].map(lambda x: "True" if x==1 else "False")

result = test_data[["ID","class"]]

result.to_csv("submission.csv", index=False)