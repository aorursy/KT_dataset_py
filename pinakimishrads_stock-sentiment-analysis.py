import pandas as pd
df = pd.read_csv('../input/stock-sentiment-analysis/Stock_Dataa.csv', encoding = "ISO-8859-1")
df.head()
train = df[df['Date']< '20150101']
test = df[df['Date'] > '20141231']
#removing punchuations

data = train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ", regex= True, inplace = True)
data.head()
#renaming the column names by number 1-25 for easy access
list1 = [i for i in range(25)]
new_Index = [str(i) for i in list1]
data.columns = new_Index
data.head()
#converting to lower case 
for index in new_Index:
    data[index] = data[index].str.lower()
data.head(1)    
#combine all 25 headlines to one paragraph
headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))
headlines[0]
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
## implement BAG OF WORDS
countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(headlines)
# implement RandomForest Classifier
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])
## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)
## Import library to check accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)
# USING RANDOM FOREST CLASSIFIER WITH TF-IDF VECTORIZER
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
## IMPLEMENTING TF-IDF VECTORIZER
tfidf= TfidfVectorizer(ngram_range=(2,2))
traindataset= tfidf.fit_transform(headlines)
# Implement RandomForestClassifier on traindataset
random_classifier= RandomForestClassifier(n_estimators=200,criterion='entropy')
random_classifier.fit(traindataset,train['Label'])
# WE WILL BE PERFORMING SAME STEPS FOR TEST DATA ALSO.

test_transform=[]
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset= tfidf.transform(test_transform)
predictions= random_classifier.predict(test_dataset)
# ACCURACY AFTER USING TF-IDF VECTORIZER
matrix= confusion_matrix(test["Label"],predictions)
print(matrix)
score= accuracy_score(test["Label"],predictions)
print(score)
report= classification_report(test['Label'],predictions)
print(report)
from sklearn.naive_bayes import MultinomialNB
naive= MultinomialNB()
# WE WILL FIRST USE BAG OF WORDS MODEL FOR CONVERTING TEXT INTO VECTORS
## IMPLEMENTING BAG OF WORDS MODEL
countvector= CountVectorizer(ngram_range=(2,2))
traindataset= countvector.fit_transform(headlines)
# FITTING TRAIN DATA INTO  NAIVE BAYES CLASSIFIER 
naive.fit(traindataset,train['Label'])
# WE WILL BE PERFORMING SAME STEPS FOR TEST DATA ALSO.

test_transform=[]
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset= countvector.transform(test_transform)
predictions= naive.predict(test_dataset)
matrix= confusion_matrix(test["Label"],predictions)
print(matrix)
score= accuracy_score(test["Label"],predictions)
print(score)
report= classification_report(test['Label'],predictions)
print(report)
# NOW WE WILL USE TF-IDF VECTORIZER WITH NAIVE BAYES CLASSIFIER
traindataset= tfidf.fit_transform(headlines)
naive.fit(traindataset,train['Label'])
# WE WILL BE PERFORMING SAME STEPS FOR TEST DATA ALSO.

test_transform=[]
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset= countvector.transform(test_transform)
predictions= naive.predict(test_dataset)
# ACCURACY AFTER USING TF-IDF VECTORIZER IN NAIVE BAYES CLASSIFIER
matrix= confusion_matrix(test["Label"],predictions)
print(matrix)
score= accuracy_score(test["Label"],predictions)
print(score)
report= classification_report(test['Label'],predictions)
print(report)
