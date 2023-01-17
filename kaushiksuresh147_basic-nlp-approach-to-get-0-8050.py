



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O

import missingno as mno

from sklearn.feature_extraction.text import TfidfVectorizer # For extracting the features from the tweet text

from sklearn.linear_model import LogisticRegression #To build a logistic model

from sklearn.metrics import accuracy_score  #To obtain the evaluation metrics



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
import seaborn as sns

occurences=train['target'].value_counts().reset_index().rename(columns={'index':'Class','target':'Number of Occurences'})

sns.barplot(x=occurences['Class'],y=occurences['Number of Occurences'])

occurences['Percentage(%)']=(occurences['Number of Occurences']/occurences['Number of Occurences'].sum())*100

occurences.set_index('Class')
traindata = list(np.array(train.iloc[:,3])) #Extracting the text feature alone from the train data

testdata = list(np.array(test.iloc[:,3]))#Extracting the text feature alone from the test data

y = np.array(train.iloc[:,4]).astype(int)#Extracting the target varaible from the train data



X_all = traindata + testdata #combining both the test and train data

lentrain = len(traindata)
# Implementing TFIDF to extract the features from the text

tfidf = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  

        analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)



print("Implementing TFIDF to both the test and train data")

tfidf.fit(X_all)

print("Transforming the data")

X_all = tfidf.transform(X_all)
X = X_all[:lentrain] # Seperating the train data from the entire data

X_test = X_all[lentrain:] # Seperating the test data from the entire data



log = LogisticRegression(penalty='l2',dual=False, tol=0.0001, 

                             C=1, fit_intercept=True, intercept_scaling=1.0, 

                             class_weight=None, random_state=None) #initialising the logistic regression function with the respective parameters



print("Training on the train data")

log.fit(X,y)



#Evaluating with the train data's target variable to obatin the training accuracy!

y_pred_X=log.predict(X)

print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))



predictions = log.predict(X_test) #Prediciting the target for the test data



predictions
test
test_ids=test['id']

submission = pd.DataFrame(predictions,index=test_ids,columns=['target'])

submission.to_csv('submission_nlprnot.csv')

print("submission file created..")