# Import modules

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn import metrics 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
#Import the Dataset

BabyProduct = pd.read_csv("../input/amazon-baby-product/amazon_baby.csv")

BabyProduct.head()
# Understand dataset

print(BabyProduct.shape)

print(BabyProduct.head(5))

print()

print(BabyProduct['rating'].value_counts())

BabyProduct['rating'].value_counts().plot(kind='bar', title='count of each rating')
#checking null in reviews

print('Total NaN in review =' , BabyProduct['review'].isnull().sum())

BabyProduct.loc[BabyProduct['review'].isnull(),'review']
#Replacing NaN values with spaces

BabyProduct['review'].fillna(value='',inplace=True)

print(BabyProduct.loc[38,:])
# PreProcessing

# Remove 3 ratings

BabyProduct = BabyProduct.loc[BabyProduct['rating'] != 3,:]

print(BabyProduct.shape)

print(BabyProduct['rating'].value_counts())

# Forming sentiments: 4 and 5 rating +ve sentiment, 1 and 2 form -ve sentiment

BabyProduct['sentiment'] = (BabyProduct['rating'] >= 4).astype(int)

print(BabyProduct.head(5))
# Step 1 : Load Data

X = BabyProduct['review']

y = BabyProduct['sentiment']

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=5)

print(X.size, Xtrain.size , Xtest.size , ytrain.size, ytest.size)
# CountVectorizer : Converts a collection of text documents to a matrix of token counts

vect1 = CountVectorizer()

# Fit : Learn a vocabulary dictionary of all tokens in the raw documents, takes care of stop word removal  & lowercasing

vect1.fit(Xtrain)

# Transform : Transform documents to document-term matrix

Xtrain_dtm = vect1.transform(Xtrain)

print(Xtrain_dtm.shape)
# Step 2: Build Model

log_reg = LogisticRegression()
# Step 3: Learn

log_reg.fit(Xtrain_dtm,ytrain);
# Step 4: Predict

Xtest_dtm = vect1.transform(Xtest)

ypredict = log_reg.predict(Xtest_dtm)
# Step 5: Evaluate

print('accuracy: ' ,metrics.accuracy_score(ytest,ypredict))

print('confusion matrix')

print(metrics.confusion_matrix(ytest,ypredict))

fpr,tpr,threshold = metrics.roc_curve(ytest,ypredict)

print('area under curve: ', metrics.roc_auc_score(ytest,ypredict))
plt.plot(fpr,tpr)

plt.title('ROC curve')

plt.xlabel('false postive rate')

plt.ylabel('true positive rate')

plt.grid(True)
#Top 5 products

BabyProduct_name5 =BabyProduct['name'].value_counts().head()

BabyProduct_name5.plot(kind='bar');
# Probablistic estimates on top product

Giraffe_Teether =BabyProduct.loc[BabyProduct['name']=='Vulli Sophie the Giraffe Teether',:]

print(Giraffe_Teether['rating'].value_counts())

Xgt = Giraffe_Teether['review']

ygt = Giraffe_Teether['rating']

Xgt_dtm = vect1.transform(Xgt)

print(Xgt_dtm.shape)
ygt_prob = log_reg.predict_proba(Xgt_dtm)

ygt_pred = log_reg.predict(Xgt_dtm)

print(ygt_prob.shape)

print(ygt_pred.shape)

print(ygt_prob[:5])

print(ygt_pred[:5])
Giraffe_Teether_indices = Giraffe_Teether.index.values

ygt_prob_df = pd.DataFrame(ygt_prob,index=Giraffe_Teether_indices,columns=['pred_prob0','pred_prob1'])

print(ygt_prob_df.head(5))
dict_maxprob = {}

for ind, row in ygt_prob_df.iterrows():

    dict_maxprob[ind] = max(row['pred_prob0'],row['pred_prob1'])



se_maxprob = pd.Series(dict_maxprob)
Giraffe_Teether['pred_prob'] = se_maxprob

print(Giraffe_Teether.head(5))
#sorting sentiments based on their high probability

Giraffe_Teether.sort_values('pred_prob',ascending=False, inplace=True)

print(Giraffe_Teether.head(5))
gt_pos_sorted = Giraffe_Teether.loc[Giraffe_Teether['sentiment']==1,['review']]

print(gt_pos_sorted.head(5))
gt_neg_sorted = Giraffe_Teether.loc[Giraffe_Teether['sentiment']==0,['review']]

print(gt_neg_sorted.head(5))
list_pos_sorted = []

for i in gt_pos_sorted.head()['review']:

    list_pos_sorted.append(i)



print(list_pos_sorted[0])
list_neg_sorted = []

for i in gt_neg_sorted.head()['review']:

    list_neg_sorted.append(i)



print(list_neg_sorted[1])
Xvoc = BabyProduct['review']

yvoc = BabyProduct['sentiment']
vect2 = CountVectorizer(vocabulary=['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate'])

vect2.fit(Xvoc)

vect2.get_feature_names()

Xvoc_dtm = vect2.transform(Xvoc)
# Converting sparse matrix to DataFrame

Xvoc_dtm_pd = pd.DataFrame(Xvoc_dtm.toarray(),columns=vect2.get_feature_names(),index=Xvoc.index)

print(Xvoc_dtm_pd.shape)

print(Xvoc_dtm_pd.head(5))
print(Xvoc_dtm_pd['awesome'].sum())

print(Xvoc_dtm_pd['awesome'].value_counts())
featurecount_dict = {}

for i in vect2.get_feature_names():

    featurecount_dict[i] = Xvoc_dtm_pd[i].sum()

    

print(featurecount_dict)
Xvoc = BabyProduct['review']

yvoc = BabyProduct['sentiment']



Xvoctrain, Xvoctest, yvoctrain, yvoctest = train_test_split(Xvoc,yvoc,random_state=5)

print(Xvoc.shape, Xvoctrain.shape , Xvoctest.shape , yvoctrain.shape, yvoctest.shape)
vect2.fit(Xvoctrain)

Xvoctrain_dtm = vect2.transform(Xvoctrain)
log_reg2 = LogisticRegression()

log_reg2.fit(Xvoctrain_dtm,yvoctrain);
print(vect2.get_feature_names())

# we can see +ve weights for positive words and -ve weights for negative words

print(log_reg2.coef_)

print(log_reg2.intercept_)

Xvoctest_dtm = vect2.transform(Xvoctest)

yvocpredict = log_reg2.predict(Xvoctest_dtm)
print('accuracy: ' , metrics.accuracy_score(yvoctest,yvocpredict))

print('confusion matrix')

print(metrics.confusion_matrix(yvoctest,yvocpredict))

fpr,tpr,threshold = metrics.roc_curve(yvoctest,yvocpredict)

print('roc_auc: ' , metrics.roc_auc_score(yvoctest,yvocpredict))
plt.plot(fpr,tpr)

plt.title('ROC curve')

plt.xlabel('false postive rate')

plt.ylabel('true positive rate')

plt.grid(True)
