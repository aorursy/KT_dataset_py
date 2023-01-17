import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
data = pd.read_csv("../input/train.tsv",delimiter='\t')
data.shape
adjectives = []
for i in range(0,data.shape[0]):
    pos_tagged = pos_tag(word_tokenize(data.iloc[i,2]))
    string = ""
    for j in range(0,len(pos_tagged)):
        if pos_tagged[j][1] in ("JJ","JJR", "JJS", "RB", "RBR", "RBS"):
            string = string + " " + pos_tagged[j][0]
    adjectives.append(string)
data['Adjective Review'] = adjectives
data.head()
data = data.drop("PhraseId",axis=1)
data = data.drop("SentenceId",axis=1)
data = data.drop("Phrase",axis=1)
print(data.head())
predictors = data['Adjective Review']
predictors.shape
response = data["Sentiment"]
response.shape
tv = TfidfVectorizer(min_df=0.0, max_df=1.0, ngram_range=(1,2),sublinear_tf=True,max_features=1000)
tv_features = tv.fit_transform(predictors)
print(tv_features.shape)
train_predictors, test_predictors, train_response, test_response = train_test_split(tv_features, response, random_state = 0)
print(train_predictors.shape)
print(test_predictors.shape)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_features = 10).fit(train_predictors, train_response)
predicted_test_response = model.predict(test_predictors)
accuracy_score(test_response, predicted_test_response)
test = pd.read_csv("../input/test.tsv",delimiter='\t')
test.shape
test_adjectives = []
for i in range(0,test.shape[0]):
    pos_tagged = pos_tag(word_tokenize(test.iloc[i,2]))
    string = ""
    for j in range(0,len(pos_tagged)):
        if pos_tagged[j][1] in ("JJ","JJR", "JJS", "RB", "RBR", "RBS"):
            string = string + " " + pos_tagged[j][0]
    test_adjectives.append(string)
test['Adjective Review'] = test_adjectives
test = test.drop("SentenceId",axis=1)
test = test.drop("Phrase",axis=1)
print(test.head())
test_predictors = test['Adjective Review']
test_predictors.shape
test_tv = TfidfVectorizer(min_df=0.0, max_df=1.0, ngram_range=(1,2),sublinear_tf=True,max_features=1000)
test_tv_features = tv.fit_transform(test_predictors)
test_response = model.predict(test_tv_features)
len(test_response)
test['Sentiment'] = test_response
test.head()
test.shape
test = test.drop("Adjective Review",axis=1)
test.to_csv("Submission.csv", sep=',',index=False)
