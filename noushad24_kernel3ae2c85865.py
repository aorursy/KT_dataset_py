# IMPORTING THE LIBRARIES

import pandas as pd

review=pd.read_csv('/kaggle/input/amazon-reviews/Amazon_Reviews.csv')

 #label column change the values 0 and 1

review['Label']=review['Label'].map({'__label__2 ':1,'__label__1 ':0})

review

#Spliting the Data

from sklearn.model_selection import train_test_split

y=review['Label']

review.drop(columns='Label',axis=1,inplace=True)

x_train,x_test,y_train,y_test=train_test_split(review,y,random_state=42,test_size=0.2)

x_train

# preprocessing for tokenize,stopwords removal,stemming/lemmatization

from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer,PorterStemmer



tokenizer=RegexpTokenizer(r'\w+')

lemmatizer=WordNetLemmatizer()

stemmer=PorterStemmer()



def preprocessing(review):

    

    final_tokens=' '

    tokens=tokenizer.tokenize(review)

    pure_tokens=[token.lower() for token in tokens if token.lower() not in stopwords.words('english')]

    lemmas_tokens=[lemmatizer.lemmatize(pure_tokens) for pure_tokens in pure_tokens]

    

    final_tokens=final_tokens.join(lemmas_tokens)

    

    return final_tokens



x_train['cleaned_text']=x_train['Review'].apply(preprocessing)

x_train['cleaned_text']



# data preprocessing

x_test['cleaned_text']=x_test['Review'].apply(preprocessing)

x_test['cleaned_text']
#to find the tfidf vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer=TfidfVectorizer(stop_words='english',use_idf=True)



x_train_TfIdf=vectorizer.fit_transform(x_train['cleaned_text'])

x_test_TfIdf=vectorizer.transform(x_test['cleaned_text'])



vectorizer.get_feature_names
#build a navie bayes classifier model

from sklearn.naive_bayes import MultinomialNB,GaussianNB

from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,f1_score,accuracy_score,precision_score

import matplotlib.pyplot as plt



clf=MultinomialNB().fit(x_train_TfIdf.toarray(),y_train)



y_pred_navie=clf.predict(x_test_TfIdf.toarray())



y_pred_navie



# confusion_matrix

confusion_matrix(y_test,y_pred_navie)

#f1_score

NB=f1_score(y_test,y_pred_navie)

NB
#accuracy score

nb=accuracy_score(y_test,y_pred_navie)

nb
# plotting AUC-ROC CURVE

y_proba_pred=clf.predict_proba(x_test_TfIdf.toarray())[::,1]



fpr,tpr,thershold=roc_curve(y_test,y_proba_pred)#to draw a roc curve



plt.plot(fpr,tpr)



auc=roc_auc_score(y_test,y_proba_pred)#calculate auc score





plt.plot(fpr,tpr,label="auc="+str(auc))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('AUC-ROC Curve')

plt.legend(loc=8)

plt.show()

#Build a Logistic Regression

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train_TfIdf.toarray(),y_train)

y_pred=lr.predict(x_test_TfIdf.toarray())

y_pred
# confusion_matrix

confusion_matrix(y_test,y_pred)
#f1_score

LR=f1_score(y_test,y_pred)

LR
# accuracy_score

lr=accuracy_score(y_test,y_pred)

lr
#roc_auc_score

roc_auc_score(y_test,y_pred)
#roc curve

fpr,tpr,thershold=roc_curve(y_test,y_pred)



plt.plot(fpr,tpr)



auc=roc_auc_score(y_test,y_pred)#calculate auc score





plt.plot(fpr,tpr,label="auc="+str(auc))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('AUC-ROC Curve')

plt.legend(loc=8)
#Build a DecisionTree

from sklearn.tree import DecisionTreeClassifier

decision=DecisionTreeClassifier()

decision.fit(x_train_TfIdf,y_train)

y_pred_dec=decision.predict(x_test_TfIdf)

y_pred_dec
#Hyperparameter Tunning for Gridsearchcv

from sklearn.model_selection import GridSearchCV

dict_a={'max_depth':[5,6,7,9],

            'min_samples_leaf':[4,5,6,7],

            'min_samples_split':[2,3,4],

            'max_leaf_nodes':[3,6,7,8]}



gd=GridSearchCV(estimator=decision,param_grid=dict_a,cv=5)

gd.fit(x_train_TfIdf,y_train)

gd.best_estimator_

gd.best_score_
#Hyperparameter Tunning for Randzedomisearchcv

from sklearn.model_selection import RandomizedSearchCV

dict_a={'max_depth':[5,6,7,9],

            'min_samples_leaf':[4,5,6,7],

            'min_samples_split':[2,3,4],

            'max_leaf_nodes':[3,6,7,8]}



rd=RandomizedSearchCV(estimator=decision,param_distributions=dict_a,n_iter=90)

rd.fit(x_train_TfIdf,y_train)

rd.best_score_
#confusion_matrix

confusion_matrix(y_test,y_pred_dec)
#accuracy_score

dt=accuracy_score(y_test,y_pred_dec)

dt
#f1_score

DT=f1_score(y_test,y_pred_dec)

DT
#to draw a roc curve

y_proba_pred=decision.predict_proba(x_test_TfIdf.toarray())[::,1]



fpr,tpr,thershold=roc_curve(y_test,y_proba_pred)



plt.plot(fpr,tpr)



auc=roc_auc_score(y_test,y_proba_pred)#calculate auc score





plt.plot(fpr,tpr,label="auc="+str(auc))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('AUC-ROC Curve')

plt.legend(loc=8)

plt.show()
#  RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

random=RandomForestClassifier(oob_score=True)

random.fit(x_train_TfIdf,y_train)

y_pred_random=random.predict(x_test_TfIdf)

y_pred_random
from sklearn.model_selection import GridSearchCV

dict_a={'max_depth':[4,5,6,7],

        'min_samples_split':[5,9,8,7],

        'n_estimators':[10],

       'min_samples_leaf':[8,6,3,2]}



gd=GridSearchCV(estimator=random,param_grid=dict_a,cv=5)

gd.fit(x_train_TfIdf,y_train)

gd.best_score_
from sklearn.model_selection import RandomizedSearchCV

dict_a={'max_depth':[4,5,6,7],

        'min_samples_split':[5,9,8,7],

        'n_estimators':[10],

       'min_samples_leaf':[8,6,3,2]}



rd=RandomizedSearchCV(estimator=random,param_distributions=dict_a,n_iter=90)

rd.fit(x_train_TfIdf,y_train)

rd.best_score_
confusion_matrix(y_test,y_pred_random)
rf=accuracy_score(y_test,y_pred_random)

rf
RF=f1_score(y_test,y_pred_random)

RF
roc_auc_score(y_test,y_pred_random)


y_proba_pred=random.predict_proba(x_test_TfIdf.toarray())[::,1]



fpr,tpr,thershold=roc_curve(y_test,y_proba_pred)#to draw a roc curve



plt.plot(fpr,tpr)



auc=roc_auc_score(y_test,y_proba_pred)#calculate auc score





plt.plot(fpr,tpr,label="auc="+str(auc))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('AUC-ROC Curve')

plt.legend(loc=8)

plt.show()
from sklearn.ensemble import GradientBoostingClassifier

gbc=GradientBoostingClassifier()

gbc.fit(x_train_TfIdf,y_train)

y_pred_gbc=gbc.predict(x_test_TfIdf)

y_pred_gbc
from sklearn.model_selection import GridSearchCV

dict_a={'max_depth':[5,6],

            'min_samples_leaf':[4,5],

            'min_samples_split':[2,3],

            'max_leaf_nodes':[3,6],

             'n_estimators':[5,10,20],

              'learning_rate':[0.1,1.0,0.1]}



gd=GridSearchCV(estimator=gbc,param_grid=dict_a,cv=5)

gd.fit(x_train_TfIdf,y_train)

gd.best_score_
from sklearn.model_selection import RandomizedSearchCV

dict_a={'max_depth':[5,6,4],

            'min_samples_leaf':[4,5,2],

            'min_samples_split':[2,3,9],

            'max_leaf_nodes':[3,6],

             'n_estimators':[50,100,200],

              'learning_rate':[0.1,1.0,0.1]}



rd=RandomizedSearchCV(param_distributions=dict_a,estimator=gbc,n_iter=100)

rd.fit(x_train_TfIdf,y_train)

rd.best_score_
confusion_matrix(y_test,y_pred_gbc)
gb=accuracy_score(y_test,y_pred_gbc)

gb
GB=f1_score(y_test,y_pred_gbc)

GB
#visualizing the accuracy score for models

import seaborn as sns



check=["Naive Bayes","Logistic","DecisionTree","RandomForest","Gradientboosting"]

overall_accuracy=[nb,lr,dt,rf,gb]



sns.barplot(x=check,y=overall_accuracy)

plt.xlabel('supervisied Learning Models')

plt.ylabel('Accuracy')

plt.title('Supervisied Learning vs Accuracy')

plt.show()



check=["Naive Bayes","Logistic","DecisionTree",  "RandomForest",   "Gradientboosting"]

overall_f1_score=[NB,LR,DT,RF,GB]

sns.barplot(x=check,y=overall_f1_score)

plt.xlabel('supervisied Learning Models')

plt.ylabel('f1_score')

plt.title('Supervisied Learning vs Accuracy')

plt.show()
