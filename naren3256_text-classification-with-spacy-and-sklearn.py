# import the libraries

import numpy as np 

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

import spacy

from spacy import displacy
# Dataset

df = pd.read_csv('../input/reviews/Restaurant_Reviews.tsv',sep = '\t')
df.head()
df.shape
# check for the missing values

df.isna().sum()

review_ = []



for i in range(0,11):

    text = df["Review"][i]

    review_.append(text)

print(review_)

# check the word dependancy using spacy.displacy()

for data in review_:

    nlp = spacy.load('en_core_web_sm')

    data = nlp(data)

    displacy.render(data,style = 'dep', options = {'font':'Areal','distance':100

                                              ,'color': 'green','bg':'white','compact' : True,}, jupyter =True)

    
# remove the empty string from the review column.

empty_loc  = []

for i, Rv,lk in df.itertuples():

    if type(Rv) == str:

        if Rv.isspace() == True:

            empty_loc.append(i)

print(empty_loc)
# check the number of positive and negative reviews

df["Liked"].value_counts()

x = df["Review"]

y = df["Liked"]



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4)   # 40% of the data is reserved for testing

print(y_test.value_counts())

print(y_train.value_counts())
#Linear Classifier

Classifier_svc = Pipeline([('tfIdf',TfidfVectorizer()),('cl',LinearSVC()),])

Classifier_svc.fit(x_train,y_train)

pred = Classifier_svc.predict(x_test)
# model evaluation

cm = confusion_matrix(y_test,pred)

print(cm)

print('\n')



print("Accuracy : ", accuracy_score(y_test,pred))

print('\n')

print(classification_report(y_test,pred))



sns.heatmap(cm,annot =True)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1)  # 90% of the data used for training

print(y_train.value_counts())

Classifier_svc.fit(x_train,y_train)

pred = Classifier_svc.predict(x_test)
# model evaluation -->  Linearsvc with trainset_percentage == 100

print(y_test.value_counts())



print('\n')

print("confusion matrix : ")

cm = confusion_matrix(y_test,pred)

print(cm)

print('\n')



print("Accuracy : ", accuracy_score(y_test,pred))

print('\n')

print(classification_report(y_test,pred))



sns.heatmap(cm,annot =True)
print(Classifier_svc.predict(["I was ate just on a whim because the parking lot was full. I had the Irish Skillet and it was Delicious. Not bad prices either between my friend and I we only paid just over 20 dollars. Service here is great even on a full day."]))

print(Classifier_svc.predict(["Stopped here for breakfast because this has been a good restaurant for meals at any time of day for many years now. You can just count on a decent meal when you stop here. I like the breakfast skillets."]))
print(Classifier_svc.predict(["We can't get a decent hamburger, they over cook them & they don't know the difference between cornbread or cake. They only have 1 soup worth eating & the waitresses on Saturdays are terrible, they are rude & don't listen while your trying to place your order"]))

print(Classifier_svc.predict(["Stopped in there one evening while traveling through Monticello. Ordered the fish and chips that the menu AND the waitress said was Cod fillets. What was brought to us was overcooked mincemeat fish sticks that had been cooked for a while and just heated in the microwave. Will NOT stop there again."]))