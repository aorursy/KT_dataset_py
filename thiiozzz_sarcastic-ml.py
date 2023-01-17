import pandas as pd





raw_df = pd.read_json('../input/Sarcasm_Headlines_Dataset.json', lines=True)

raw_df.head(2)
from nltk.corpus import stopwords

import string

import re



cleaned_df = raw_df



cleaned_df.pop('article_link')

cleaned_df.dropna()



stop = stopwords.words('english') + list(string.punctuation)

cleaned_df['headline'] = cleaned_df['headline'].apply(lambda s: ' '.join([re.sub(r'\W+', '', word.lower()) for word in s.split(' ') if word not in stop]))



cleaned_df.head(2)
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import train_test_split

import sklearn.metrics



train, test = train_test_split(cleaned_df, test_size=0.2)



reg_text_clf = Pipeline([

    ('vect', CountVectorizer()),

    ('tfidf', TfidfTransformer()),

    ('reg', SGDClassifier())

])



reg_text_clf.fit(train['headline'], train['is_sarcastic'])
reg_predicted = reg_text_clf.predict(test['headline'])

sklearn.metrics.f1_score(test['is_sarcastic'], reg_predicted, average='micro')
sentence_to_predict = ['youre tall', 'youre tall as a giant dwarf', 'youre very nice thank you so much !','youre very nice little pumpkin !']

text_to_predict = reg_text_clf.predict(sentence_to_predict)



for i in range(0, len(text_to_predict)):

    print("%s -> %s" % (sentence_to_predict[i], ('it looks fair', 'sounds like a troll !') [text_to_predict[i]]))
