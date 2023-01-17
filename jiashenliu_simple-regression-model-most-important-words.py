import pandas as pd

df = pd.read_csv('../input/Hotel_Reviews.csv')

df.head()
print(df.shape)
df['all_review'] = df.apply(lambda x:x['Positive_Review']+' '+x['Negative_Review'],axis=1)
from sklearn.model_selection import train_test_split

train,test1 = train_test_split(df,test_size=0.8,random_state=42)

test1,test2 = train_test_split(test1,test_size=0.67,random_state=42)

test2,test3 = train_test_split(test2,test_size=0.5,random_state=42)

print(train.shape);print(test1.shape);print(test2.shape);print(test3.shape)
from sklearn.feature_extraction.text import TfidfVectorizer

t = TfidfVectorizer(max_features=10000)

train_feats = t.fit_transform(train['all_review'])

test_feats1 = t.transform(test1['all_review'])

test_feats2 = t.transform(test2['all_review'])

test_feats3 = t.transform(test3['all_review'])
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error
gbdt = GradientBoostingRegressor(max_depth=5,learning_rate=0.1,n_estimators=150) # Large iteration, fewer estimators

gbdt.fit(train_feats,train['Reviewer_Score'])
pred_inbag = gbdt.predict(train_feats)

pred_test1 = gbdt.predict(test_feats1)

pred_test2 = gbdt.predict(test_feats2)

pred_test3 = gbdt.predict(test_feats3)
MAEs = pd.DataFrame({'data':['in_bag','out_bag1','out_bag2','out_bag3'],'MAE':[mean_absolute_error(train['Reviewer_Score'],pred_inbag),mean_absolute_error(test1['Reviewer_Score'],pred_test1),mean_absolute_error(test2['Reviewer_Score'],pred_test2),mean_absolute_error(test3['Reviewer_Score'],pred_test3)]})
MAEs
from ggplot import *

p = ggplot(MAEs,aes(x='data',weight='MAE')) + geom_bar()+theme_bw()+ggtitle('Mean Absolute Error of GBDT models')

print(p)
RMSEs = pd.DataFrame({'data':['in_bag','out_bag1','out_bag2','out_bag3'],'RMSE':[mean_squared_error(train['Reviewer_Score'],pred_inbag)**0.5,mean_squared_error(test1['Reviewer_Score'],pred_test1)**0.5,mean_squared_error(test2['Reviewer_Score'],pred_test2)**0.5,mean_squared_error(test3['Reviewer_Score'],pred_test3)**0.5]})
RMSEs
p = ggplot(RMSEs,aes(x='data',weight='RMSE')) + geom_bar()+theme_bw()+ggtitle('Rooted Mean Squared Error of GBDT models')

print(p)
words = t.get_feature_names()

importance = gbdt.feature_importances_

impordf = pd.DataFrame({'Word' : words,

'Importance' : importance})

impordf = impordf.sort_values(['Importance', 'Word'], ascending=[0, 1])

## Check the top 30 most important words

impordf.head(30)
impordf.to_csv('Most_important_words.csv',index=False)