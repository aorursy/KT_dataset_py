# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#reading the training file

training_text_data=pd.read_csv("/kaggle/input/msk-redefining-cancer-treatment/training_text",sep='\|\|', header=None,skiprows=1,names=["ID","Text"])


training_text_data.head()
#reading the training_variants file

training_variants_data=pd.read_csv("/kaggle/input/msk-redefining-cancer-treatment/training_variants")
training_variants_data.head()
#merging training text data and training variant data

total_data=pd.merge(left=training_text_data,

    right=training_variants_data,

    how='inner',

    on="ID",)
total_data.head(16)
#checking for null values on our data

total_data.isnull().sum()
#removing the null data

total_data = total_data[~total_data['Text'].isnull()]
total_data.shape
total_data.isnull().sum()
total_data.info()



y=total_data.Class

y
X=total_data.Text
print(X.shape)



X.head()
# splitting into test and train

from sklearn.model_selection  import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# vectorizing the sentences; removing stop words

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(stop_words='english')
#converting the xtrain in to bag of words

vect.fit(X_train)

vect.fit(X_test)

# X_train_dtm = vect.transform(X_train)
#printing the number of words in our text column

vect.vocabulary_
# transforming the train and test datasets

X_train_transformed = vect.transform(X_train)

X_test_transformed =vect.transform(X_test)
# note that the type is transformed matrix

print(type(X_train_transformed))

print(X_train_transformed)
# training the NB model and making predictions

from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()

from sklearn.metrics import classification_report





# fit

mnb.fit(X_train_transformed,y_train)



# predict class

y_pred_class = mnb.predict(X_test_transformed)



# predict probabilities

y_pred_proba =mnb.predict_proba(X_test_transformed)





# printing the overall accuracy

from sklearn import metrics

metrics.accuracy_score(y_test, y_pred_class)



print(classification_report(y_test, y_pred_class))
from sklearn.model_selection import GridSearchCV



from sklearn.tree import DecisionTreeClassifier



from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

from sklearn.metrics import classification_report



SVC=LinearSVC()
SVM_model= SVC.fit(X_train_transformed, y_train)

y_pred = SVC.predict(X_test_transformed)

print(classification_report(y_test, y_pred))
import statsmodels.api as sm
y_train_numeric=y_train.astype(int)

X_train_transformed_numeric=X_train_transformed.astype(int)
y_train.dtype
X_train_transformed.dtype
#tunig naive bayes algorithm

# GridSearchCV to find optimal max_depth

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'alpha': range(1, 3)}

# instantiate the model

mnb = MultinomialNB()





# fit tree on training data

naiv = GridSearchCV(mnb, parameters, 

                    cv=n_folds, 

                   scoring="neg_log_loss")

naiv.fit(X_train_transformed, y_train)
# scores of GridSearch CV

scores = naiv.cv_results_

pd.DataFrame(scores).head()

test_text_data=pd.read_csv("/kaggle/input/msk-redefining-cancer-treatment/stage2_test_text.csv",sep='\|\|', header=None,skiprows=1,names=["ID","Text"])

test_variant_data=pd.read_csv("/kaggle/input/msk-redefining-cancer-treatment/stage2_test_variants.csv")
test_text_data.isnull().sum()
test_variant_data.isnull().sum()
test_total_data=pd.merge(left=test_text_data,

    right=test_variant_data,

    how='inner',

    on="ID",)

test_total_data.head()

len(test_total_data)
X_test_given_transformed=vect.transform(test_total_data['Text'])

mnb = MultinomialNB(alpha=1)

mnb.fit(X_train_transformed, y_train)

y_pred_test =mnb.predict_proba(X_test_given_transformed)

print(y_pred_test)
y_pred_test=pd.DataFrame(y_pred_test)  

y_pred_test.head(10)



y_pred_test.rename(columns={0:'class1',1:'class2',2:'class3',3:'class4',4:'class5',5:'class6',6:'class7',7:'class8',8:'class9'}, 

                 inplace=True)
y_pred_test.head()
test_total_data.head()
y_pred_test.head()

Submission_File=pd.concat([test_total_data,y_pred_test],axis=1)

Submission_File.head()
Submission_File=Submission_File.drop(columns=["Text","Gene","Variation"])

Submission_File.head()
Submission_File["ID"]=Submission_File["ID"].astype(int)
Submission_File.tail()

#Submission_File = Submission_File[:-1]
Submission_File.to_csv('Submission_File',sep=',',header=True,index=None)

Submission_File.to_csv(r'Submission_File.csv',index=False)
from IPython.display import FileLink

FileLink(r'Submission_File.csv')