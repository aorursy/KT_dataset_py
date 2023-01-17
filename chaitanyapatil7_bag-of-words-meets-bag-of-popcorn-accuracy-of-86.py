import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
tfidf = TfidfTransformer()
cv = CountVectorizer()
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
pipe = Pipeline([('bow',CountVectorizer()),
                 ('tfidf',TfidfTransformer()),
                 ('model', MultinomialNB())])
df = pd.read_csv('../input/labeledTrainData.tsv',sep = '\t')
X = df['review']
y= df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
pipe.fit(X_train,y_train)
predictions = pipe.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,predictions))


test_data = pd.read_csv('../input/testData.tsv',sep = '\t')
test_data.shape
Xt  = test_data['review']
Xt.shape
predictions_new = pipe.predict(Xt)
predictions_new.shape
type(predictions_new)
test_data_id = test_data['id']
test_data_id.shape
type(test_data_id)
Label=[]
for num in predictions_new:
    Label.append(num)
type(Label)
len(Label)
sentiment=pd.DataFrame({'sentiment':Label})
sentiment.head()
idx=pd.DataFrame({'id':test_data_id})

idx.head()
OUTPUT_RESULT="submission_pipeline.csv"
submission=pd.concat([idx,sentiment],axis=1)
submission.to_csv(OUTPUT_RESULT,index=False)
