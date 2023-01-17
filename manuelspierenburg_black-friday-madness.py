import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib as mlp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
print(os.listdir("../input"))
df = pd.read_csv("../input/BlackFriday.csv")
df.info()
df.isna().sum()
# remove categories with NaNs
df = df.loc[:, ~df.isna().any()]
# Top 5 users by count
user_counts = df.groupby('User_ID').size().reset_index(name='counts') 
user_counts.sort_values('counts', ascending=False)[0:5]
# Top 5 users by purchase
user_purchase = df[['User_ID','Purchase']].groupby('User_ID').agg('sum')
user_purchase.sort_values('Purchase', ascending=False)[0:5]
# Top 5 products by count
product_counts = df.groupby('Product_ID').size().reset_index(name='counts') 
product_counts.sort_values('counts', ascending=False)[0:5]
# Top 5 cities by purchase
city_purchase = df[['City_Category','Purchase']].groupby('City_Category').agg('sum')
city_purchase.sort_values('Purchase', ascending=False)[0:5]
# show purchase distribution
# we can clearly see the long tail only few customers purchase a lot
user_purchase.hist(bins=100) 
# compare number of female to male customers
df[['User_ID', 'Gender']].drop_duplicates()\
                         .groupby('Gender')['Gender']\
                         .agg('count').plot.pie()
# count customers per age group
df[['User_ID', 'Age']].drop_duplicates()\
                      .groupby('Age')['Age']\
                      .agg('count').plot.bar()
# compare married customers to not married customers
df[['User_ID', 'Marital_Status']].drop_duplicates()\
                                 .groupby('Marital_Status')['Marital_Status']\
                                 .agg('count').plot.pie()
# prepare matrices for sklearn learners
X = np.zeros((len(user_counts), len(product_counts)))
y = np.zeros(len(user_counts))

# map user_ids, product_ids and gender to 0,1,2,...
user_map =  {v: k for k,v in enumerate(np.unique(df.User_ID))} 
product_map =  {v: k for k,v in enumerate(np.unique(df.Product_ID))} 
gender_map = {v: k for k,v in enumerate(np.unique(df.Gender))} 

# for each user create a vector with ones for product he bought
print("Total transactions: {}".format(len(df)))
for i in range(len(df)):
    if i % 100000 == 0:
        print("{} processed".format(i))
    X[user_map[df.iloc[0]['User_ID']]][product_map[df.iloc[0]['Product_ID']]] = 1
    y[user_map[df.iloc[0]['User_ID']]] = gender_map[df.iloc[0]['Gender']]
# calculate sparsity
total = X.size
zeros = total - np.count_nonzero(X)
sparsity = zeros / total
print("Total: {}".format(total))
print("Zeros: {}".format(zeros))
print("Sparsity: {}".format(sparsity))
# even though data is sparse random forest may be a good classifier for this problem
scores = cross_val_score(RandomForestClassifier(n_estimators=10), X, y, cv=3)
print("Accuracy: {}".format(np.mean(scores)))
# 100% accuracy ????? Wooooooooooooooooooooooot
# Maybe I made a mistake but seems women buy very different articles compared to man
# lets investigate the random forest features
forest = RandomForestClassifier(n_estimators=10)
forest.fit(X, y)
importances = forest.feature_importances_
idx = np.argsort(importances)[::-1]
sorted_importances = importances[idx]
product_ids = np.array(list(product_map.keys()))[idx]

print("Best 5 features indices: {}".format(idx[0:5]))
print("Best 5 features importances: {}".format(sorted_importances[0:5]))
print("Best 5 features (product_ids): {}".format(product_ids[0:5]))
# mmh only one feature has a score bigger than 0.
# let's see the male vs female ratio on that product
df[df.Product_ID == 'P00069042'].groupby('Gender')['Gender'].agg('count').plot.pie()
# that's weird, I would expect a product which was only bought by man or women. 
# let's see if we can find such products ourself
product_ratios = pd.DataFrame(columns=['w_ratio', 'm_ratio'], index=list(product_map.keys()))
print('Number of products: {}'.format(len(product_map)))
for pid, i in product_map.items():
    tmp = df[df.Product_ID == pid].groupby('Gender')['Gender'].agg('count')
    total = len(df[df.Product_ID == pid])
    w_ratio = tmp.get('F', 0) / total
    m_ratio = tmp.get('M', 0) / total
    product_ratios.loc[pid]= [w_ratio, m_ratio]
    if i % 1000 == 0:
        print("{} processed".format(i))
# products only bought by women
w_products = product_ratios[product_ratios.w_ratio == 1].index
print("Number of products bought only by women: {}".format(len(w_products)))
print("Examples: {}".format(w_products[:5]))
# products only bought by men
m_products = product_ratios[product_ratios.w_ratio == 0].index
print("Number of products bought only by men: {}".format(len(m_products)))
print("Examples: {}".format(m_products[:5]))
# There are quite a lot of products which are only bought by women or men which makes it easy to build 
# a classifier and explains the good result
# input this time is just one value per user = total amount purchased
# create output vector = age group per user
y = np.zeros(len(user_purchase))

# map age groups to 0,1,2,...
age_map = {v: k for k,v in enumerate(np.unique(df.Age))} 

for i in range(len(user_purchase)):
    user_id = user_purchase.index[i]
    age = df[df['User_ID'] == user_id]['Age'].iloc[0]
    y[i] = age_map[age]
# let's try random forest again
scores = cross_val_score(RandomForestClassifier(n_estimators=10), user_purchase, y, cv=3)
print("Random Forest Accuracy: {}".format(np.mean(scores)))
# does not seem to work that well for age group
# Lets try SVC and LinearDiscriminantAnalysis 
scores = cross_val_score(SVC(gamma='auto'), user_purchase, y,  cv=3)
print("SVC Accuracy: {}".format(np.mean(scores)))
scores = cross_val_score(LinearDiscriminantAnalysis(), user_purchase, y, cv=3)
print("LDA Accuracy: {}".format(np.mean(scores)))
# we can use same X as for gender classifier
# prepare y 
y = np.zeros(len(user_counts))
for uid, i in user_map.items():
    y[user_map[uid]] = df[df.User_ID == uid].iloc[0].Marital_Status
# predict maritial status 
scores = cross_val_score(RandomForestClassifier(n_estimators=10), X, y, cv=3)
print("Accuracy: {}".format(np.mean(scores)))
# also for maritial status the bought products seem not to work very well