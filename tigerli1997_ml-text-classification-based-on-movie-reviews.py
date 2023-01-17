import pandas as pd # Pandas has been imported before and erase this line if you start from Data Preparation
from sklearn.model_selection import train_test_split

file = "../input/labelled_full_dataset.csv" # erase this line if you start from Data Preparation
df = pd.read_csv(file) # erase this line if you start from Data Preparation

X, y = df['review'], df['label']

# Splitting in Train and Test Sets, We want to use 30% of the data as test data
X_train, X_test, y_train, y_test = train_test_split(\
X, y, test_size=0.3, stratify=y, random_state=100)

print('data counts in each set:',\
      len(X_train),len(X_test),len(y_train),len(y_test))
print('='*50)
print('ratio of positive to negative reviews in train and test sets:',sum(y_train)/len(y_train),sum(y_test)/len(y_test))
#import necessary modules
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report

# define a model evaluation function for repeated use
def model_score(model):
    model.fit(X_train, y_train) # fit model
    y_pred = model.predict(X_test) # make predictions
    print("Accuracy : {:.4f}".format(accuracy_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred))
clf1 = Pipeline([('vec', CountVectorizer()), ('nb', MultinomialNB())]) 
model_score(clf1)
clf2 = Pipeline([('vec', TfidfVectorizer()), ('nb', MultinomialNB())])
model_score(clf2)
clf3 = Pipeline([('vec', CountVectorizer()), ('LR', LogisticRegression())])
model_score(clf3)
clf4 = Pipeline([('vec', TfidfVectorizer()), ('LR', LogisticRegression())])
model_score(clf4)
# it doesn't matter if you see 'future warning'
clf5 = Pipeline([('vec', CountVectorizer(ngram_range=(1,2))), ('nb', MultinomialNB())])
model_score(clf5)
clf6 = Pipeline([('vec', TfidfVectorizer(ngram_range=(1,2))), ('nb', MultinomialNB())])
model_score(clf6)
clf7 = Pipeline([('vec', CountVectorizer(ngram_range=(1,2))), ('LR', LogisticRegression())])
model_score(clf7)
clf8 = Pipeline([('vec', TfidfVectorizer(ngram_range=(1,2))), ('LR', LogisticRegression())])
model_score(clf8)
# try to type some movie reviews here
content = '''
This movie is not your ordinary Hollywood flick. It has a great and deep message. 
This movie has a foundation and just kept on being built on from their and that foundation is hope.
''' 
# An example if user review from movie: The Shawshank Redemption (rated in 10/10)

# you could try different models here: clf 12345678
prediction = clf7.predict([content])
# be patient to check the prediction output and set conditions
print('result of prediction: ',prediction[0])

if prediction[0] == 1:
    print ("This is a positive review!")
if prediction[0] == 0:
    print ('This is a negative review!')