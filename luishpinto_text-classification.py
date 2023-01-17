import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline
flist = dict()

flist['yelp'] = '../input/textclassificationdb/yelp_labelled.txt'
flist['imdb'] = '../input/textclassificationdb/imdb_labelled.txt'
flist['amazon'] = '../input/textclassificationdb/amazon_cells_labelled.txt'
dflist = []

for source,path in flist.items():
    df = pd.read_csv(path,names = ['sentence','label'],sep = '\t')
    df['source'] = source
    dflist.append(df)

df = pd.concat(dflist)

print(df.iloc[0])
sentences = ['John likes ice cream','John hates chocolate.']

vectorizer = CountVectorizer(min_df = 0,lowercase = False)
vectorizer.fit(sentences)

print(vectorizer.vocabulary_)
print(vectorizer.transform(sentences).toarray())
classifiers = [LogisticRegression(),
               DecisionTreeClassifier(),
               SGDClassifier()]

for clf in classifiers:

    print('\n\nClassifier: {}\n'.format(clf))

    for i in df['source'].unique():
        dfsource = df[df['source'] == i]
        S = dfsource['sentence'].values
        y = dfsource['label'].values

        Strain,Stest,ytrain,ytest = train_test_split(S,y,test_size = 0.25,random_state = 1000)

        vectorizer = CountVectorizer()
        vectorizer.fit(Strain)

        Xtrain = vectorizer.transform(Strain)
        Xtest = vectorizer.transform(Stest)

        pipe = Pipeline(steps = [('classifier',clf)])
        pipe.fit(Xtrain,ytrain)
        score = pipe.score(Xtest,ytest)

        print('Accuracy for {} dataset: {:.3f}'.format(i,score))
mood = ['Bad mood','Good mood']

## enter the sentence
sentence = "The bluetooth doesn't work fine"

print(mood[pipe.predict(vectorizer.transform([sentence]))[0]])