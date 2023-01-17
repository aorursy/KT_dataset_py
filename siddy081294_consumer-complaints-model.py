# Mid-Term Project - PGP Program IIT Guwahati
#Importing all the required files
import pandas as pd

import numpy as np

from scipy.stats import randint

import seaborn as sns

import matplotlib.pyplot as plt

import nltk

import re

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB

from sklearn.linear_model import LogisticRegression,LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,classification_report

from sklearn.preprocessing import LabelEncoder,StandardScaler
#Reading Input Files
consumer_complaints_train = pd.read_csv('../input/consumer-complaints/Edureka_Consumer_Complaints_train.csv')

consumer_complaints_train.head(3)
consumer_complaints_train.shape
# Data Pre-processing and Exploratory Data Analysis 
fig = plt.figure(figsize=(20,20))

consumer_complaints_train.groupby(['State'])['Complaint ID'].count().sort_values().plot.barh(

    color='red', title='NUMBER OF COMPLAINTS REGISTERED IN EACH STATE\n')

plt.xlabel('Statewise Count', fontsize=15)

#Here California has most number of the complaints being registered
consumer_complaints_train[consumer_complaints_train['State'] == 'CA']['Product'].value_counts()
#Taking top 5 product categories complained for california state

fig = plt.figure(figsize=(6,6))

consumer_complaints_train[consumer_complaints_train['State'] == 'CA']['Product'].value_counts().head().plot.pie(shadow=True)
#looking at the complaints raised according to each category

fig = plt.figure(figsize=(6,6))

consumer_complaints_train.groupby(['Product'])['Complaint ID'].count().sort_values().plot.bar()
fig = plt.figure(figsize=(6,6))

consumer_complaints_train[consumer_complaints_train['Product']=='Mortgage']['Issue'].value_counts().plot.barh(color='green')
fig = plt.figure(figsize=(10,8))

consumer_complaints_train[consumer_complaints_train['Consumer disputed?']=='Yes']['Product'].value_counts().head().plot.pie(shadow=True,explode=[0.2,0,0,0,0])
fig = plt.figure(figsize=(6,6))

consumer_complaints_train[consumer_complaints_train['Consumer disputed?']=='Yes']['State'].value_counts().head().plot.bar(color='maroon', width=0.4)
fig = plt.figure(figsize=(6,6))

consumer_complaints_train[consumer_complaints_train['Timely response?']=='Yes']['Consumer disputed?'].value_counts().plot.bar(width=0.1)

plt.xlabel('Consumer Disputed')

plt.ylabel('Count of the consumers')

plt.title('THE DISTRIBUTION OF CONSUMERS WHO DISPUTED BESIDES TIMELY RESPONSE', fontsize=12)
fig = plt.figure(figsize=(6,6))

consumer_complaints_train[(consumer_complaints_train['Consumer disputed?']=='Yes') & (consumer_complaints_train['Timely response?']=='Yes')]['Company'].value_counts().head(10).plot.bar(color='red')
consumer_complaints_train['Submitted via'].value_counts()
fig = plt.figure(figsize=(10,8))

consumer_complaints_train[(consumer_complaints_train['Consumer disputed?'] == 'Yes')&(consumer_complaints_train['Timely response?'] == 'Yes')]['Submitted via'].value_counts().plot.pie(explode=[0.2,0,0,0,0,0],shadow=True)
consumer_complaints_train[consumer_complaints_train['Company'] == 'Bank of America']['Company response to consumer'].value_counts().plot.barh()
temp_df = consumer_complaints_train[consumer_complaints_train['Timely response?'] == 'Yes']

temp_df.groupby(['Company'])['Timely response?'].count().sort_values(ascending=False).head(15).plot.bar(color='green')

plt.title('companies distribution of their timely response', fontsize=13)
consumer_complaints_train.groupby(['Company'])['Complaint ID'].count().sort_values(ascending=False).head(15).plot.barh(width=0.75, color='brown')
consumer_complaints_train['Date received'] = pd.to_datetime(consumer_complaints_train['Date received'])

consumer_complaints_train['year_received'] = consumer_complaints_train['Date received'].dt.year

consumer_complaints_train['month_received'] = consumer_complaints_train['Date received'].dt.month

consumer_complaints_train.head()
jan = consumer_complaints_train[(consumer_complaints_train['month_received'] == 1) & (consumer_complaints_train['Consumer disputed?'] == 'Yes')]['Consumer disputed?'].value_counts()

feb = consumer_complaints_train[(consumer_complaints_train['month_received'] == 2) & (consumer_complaints_train['Consumer disputed?'] == 'Yes')]['Consumer disputed?'].value_counts()

mar = consumer_complaints_train[(consumer_complaints_train['month_received'] == 3) & (consumer_complaints_train['Consumer disputed?'] == 'Yes')]['Consumer disputed?'].value_counts()

apr = consumer_complaints_train[(consumer_complaints_train['month_received'] == 4) & (consumer_complaints_train['Consumer disputed?'] == 'Yes')]['Consumer disputed?'].value_counts()

may = consumer_complaints_train[(consumer_complaints_train['month_received'] == 5) & (consumer_complaints_train['Consumer disputed?'] == 'Yes')]['Consumer disputed?'].value_counts()

jun = consumer_complaints_train[(consumer_complaints_train['month_received'] == 6) & (consumer_complaints_train['Consumer disputed?'] == 'Yes')]['Consumer disputed?'].value_counts()

jul = consumer_complaints_train[(consumer_complaints_train['month_received'] == 7) & (consumer_complaints_train['Consumer disputed?'] == 'Yes')]['Consumer disputed?'].value_counts()

aug = consumer_complaints_train[(consumer_complaints_train['month_received'] == 8) & (consumer_complaints_train['Consumer disputed?'] == 'Yes')]['Consumer disputed?'].value_counts()

sep = consumer_complaints_train[(consumer_complaints_train['month_received'] == 9) & (consumer_complaints_train['Consumer disputed?'] == 'Yes')]['Consumer disputed?'].value_counts()

oct = consumer_complaints_train[(consumer_complaints_train['month_received'] == 10) & (consumer_complaints_train['Consumer disputed?'] == 'Yes')]['Consumer disputed?'].value_counts()

nov = consumer_complaints_train[(consumer_complaints_train['month_received'] == 11) & (consumer_complaints_train['Consumer disputed?'] == 'Yes')]['Consumer disputed?'].value_counts()

dec = consumer_complaints_train[(consumer_complaints_train['month_received'] == 12) & (consumer_complaints_train['Consumer disputed?'] == 'Yes')]['Consumer disputed?'].value_counts()
fig = plt.figure(figsize=(8,6))

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',

         'October', 'November', 'December']



dispute = [jan,feb,mar,apr,may,jun,jul,aug,sep,oct,nov,dec]



sns.barplot(x=dispute, y=months)
len(consumer_complaints_train[(consumer_complaints_train['Timely response?'] == 'Yes') & (consumer_complaints_train['Consumer disputed?'] == 'Yes')])/len(consumer_complaints_train[consumer_complaints_train['Consumer disputed?'] == 'Yes'])
len(consumer_complaints_train[(consumer_complaints_train['Timely response?'] == 'No') & (consumer_complaints_train['Consumer disputed?'] == 'Yes')])/len(consumer_complaints_train[consumer_complaints_train['Consumer disputed?'] == 'Yes'])
#Create a new dataset for the Text Base Modelling

nlp_dataset = consumer_complaints_train[['Product','Consumer complaint narrative']]
nlp_dataset.isnull().sum()
nlp_dataset = nlp_dataset.dropna()
nlp_dataset.shape
nlp_dataset['Product'].unique()
nlp_dataset_sampled = nlp_dataset.sample(10000, random_state=1)
X = nlp_dataset_sampled['Consumer complaint narrative']

y = nlp_dataset_sampled['Product']
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1,2), stop_words='english')

features = tfidf.fit_transform(nlp_dataset_sampled['Consumer complaint narrative'])

labels = LabelEncoder().fit_transform(nlp_dataset_sampled['Product'])
models = [

    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),

    LinearSVC(),

    MultinomialNB(),

]



#6-Fold Cross-validation

CV = 6

cv_df = pd.DataFrame(index=range(CV * len(models)))



entries = []

for model in models:

    model_name = model.__class__.__name__

    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)

    for fold_idx, accuracy in enumerate(accuracies):

        entries.append((model_name, fold_idx, accuracy))

    

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
mean_accuracy = cv_df.groupby('model_name').accuracy.mean()

std_accuracy = cv_df.groupby('model_name').accuracy.std()



acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 

          ignore_index=True)

acc.columns = ['Mean Accuracy', 'Standard deviation']

acc
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25,random_state = 0)
tfidf =TfidfVectorizer(sublinear_tf=True, min_df=4, ngram_range=(1,2), stop_words='english')

fit_v = tfidf.fit(X_train)

features = fit_v.transform(X_train)

model = LinearSVC()

model.fit(features,y_train)

y_pred = model.predict(fit_v.transform(X_test))

round(accuracy_score(y_test,y_pred),4)

print(classification_report(y_test, y_pred, target_names= nlp_dataset_sampled['Product'].unique()))
consumer_complaints_test = pd.read_csv('../input/test-data/Edureka_Consumer_Complaints_test.csv')
consumer_complaints_test.head()
nlp_dataset_test = consumer_complaints_test[['Product','Consumer complaint narrative']]
nlp_dataset_test = nlp_dataset_test.dropna()

nlp_dataset_test.shape
y_pred_test = model.predict(fit_v.transform(nlp_dataset_test['Consumer complaint narrative']))
y_pred_test.shape
round(accuracy_score(nlp_dataset_test['Product'],y_pred_test),4)
pd.DataFrame(y_pred_test).to_csv('predictions_nlp.csv', index=False)
train_df = pd.read_csv('../input/consumer-complaints/Edureka_Consumer_Complaints_train.csv')
train_df.head()
train_df.shape
train_df_1 = train_df
train_df_1.drop(['Date received','Date sent to company','Sub-product', 'Sub-issue','Consumer complaint narrative',

                'ZIP code','Complaint ID', 'Tags'], axis=1, inplace=True)
train_df_1.isnull().sum()
train_df_1['Company public response'].value_counts()
train_df_1['Company public response'].fillna('Company chooses not to provide a public response', inplace=True)
train_df_1['Consumer consent provided?'].value_counts()
train_df_1['Consumer consent provided?'].fillna('Other', inplace=True)
train_df_1.isnull().sum()
train_df_1.dropna(inplace=True)
train_df_1.isnull().sum()
train_df_1.shape
train_df_1['Consumer disputed?'].value_counts()
train_df_sampled = train_df_1.sample(10000, random_state=1)
from sklearn.metrics import roc_auc_score,roc_curve
converted_df = train_df_sampled.apply(LabelEncoder().fit_transform)
converted_df.head()
X = converted_df.drop(['Consumer disputed?'], axis=1)

y = converted_df['Consumer disputed?']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=0)
lr = LogisticRegression(random_state=1, solver='liblinear', class_weight='balanced') #being imbalanced we use class_weight='balanced'

lr.fit(X_train,y_train)

y_pred_lr = lr.predict(X_test)

accuracy_score(y_test,y_pred_lr)
import seaborn as sns

print(accuracy_score(y_test,y_pred_lr))

print('********************************************')

print('Confusion matrix')

lr_cfm=confusion_matrix(y_test, y_pred_lr)





lbl1=["Predicted 1", "Predicted 2"]

lbl2=["Actual 1", "Actual 2"]



sns.heatmap(lr_cfm, annot=True, cmap="rainbow", fmt="d", xticklabels=lbl1, yticklabels=lbl2)

plt.show()



print('**********************************************')

print(classification_report(y_test,y_pred_lr))
rf = RandomForestClassifier(class_weight='balanced')

rf.fit(X_train,y_train)

y_pred_rf = rf.predict(X_test)

accuracy_score(y_test,y_pred_rf)
print(accuracy_score(y_test,y_pred_rf))

print('********************************************')

print('Confusion matrix')

lr_cfm=confusion_matrix(y_test, y_pred_rf)





lbl1=["Predicted 1", "Predicted 2"]

lbl2=["Actual 1", "Actual 2"]



sns.heatmap(lr_cfm, annot=True, cmap="rainbow", fmt="d", xticklabels=lbl1, yticklabels=lbl2)

plt.show()



print('**********************************************')

print(classification_report(y_test,y_pred_rf))
# Actual Values of y_test

print (y_test.value_counts())

print ("Null Accuracy:",y_test.value_counts().head(1) / len(y_test))
from sklearn.svm import SVC
svc = SVC(class_weight='balanced')

svc.fit(X_train,y_train)

y_pred_svc = svc.predict(X_test)

accuracy_score(y_test,y_pred_svc)
print(accuracy_score(y_test,y_pred_svc))

print('********************************************')

print('Confusion matrix')

lr_cfm=confusion_matrix(y_test, y_pred_svc)





lbl1=["Predicted 1", "Predicted 2"]

lbl2=["Actual 1", "Actual 2"]



sns.heatmap(lr_cfm, annot=True, cmap="rainbow", fmt="d", xticklabels=lbl1, yticklabels=lbl2)

plt.show()



print('**********************************************')

print(classification_report(y_test,y_pred_svc))
from xgboost.sklearn import XGBClassifier
xgb = XGBClassifier(num_class=2,objective='multi:softprob',eval_metric="mlogloss", seed=42)
xgb.fit(X_train,y_train)
y_pred_xgb = xgb.predict(X_test)

accuracy_score(y_test,y_pred_xgb)
print(accuracy_score(y_test,y_pred_xgb))

print('********************************************')

print('Confusion matrix')

lr_cfm=confusion_matrix(y_test, y_pred_xgb)





lbl1=["Predicted 1", "Predicted 2"]

lbl2=["Actual 1", "Actual 2"]



sns.heatmap(lr_cfm, annot=True, cmap="rainbow", fmt="d", xticklabels=lbl1, yticklabels=lbl2)

plt.show()



print('**********************************************')

print(classification_report(y_test,y_pred_xgb))
test_df = pd.read_csv('../input/test-data/Edureka_Consumer_Complaints_test.csv')
test_df.head()
test_df_1 = test_df
test_df_1.drop(['Date received','Date sent to company','Sub-product', 'Sub-issue','Consumer complaint narrative',

                'ZIP code','Complaint ID', 'Tags'], axis=1, inplace=True)
test_df_1.isnull().sum()
test_df_1['Company public response'].value_counts()
test_df_1['Company public response'].fillna('Company chooses not to provide a public response', inplace=True)
test_df_1.isnull().sum()
test_df_1['Consumer consent provided?'].value_counts()
test_df_1['Consumer consent provided?'].fillna('Other', inplace=True)
test_df_1.isnull().sum()
test_df_1.dropna(inplace=True)
test_df_1.shape
converted_test = test_df_1.apply(LabelEncoder().fit_transform)
converted_test.head()
predict_test = xgb.predict(converted_test)
predict_test.shape
pd.DataFrame(predict_test, columns=['Consumer Disputed']).to_csv('Cust_Dispute_predictions.csv', index=False)