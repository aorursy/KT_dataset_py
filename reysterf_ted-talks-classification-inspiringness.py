import pandas as pd
transcript_df = pd.read_csv('../input/transcripts.csv')
details_df = pd.read_csv('../input/ted_main.csv')
df = pd.merge(details_df, transcript_df, how='inner', on=['url'])
df = df.drop(['main_speaker', 'title', 'views', 'description','comments', 
              'duration', 'event', 'film_date', 'languages', 'num_speaker', 
              'published_date', 'related_talks', 'speaker_occupation', 'tags', 'url'], axis=1)
df
import ast
insp_ratios = []
for i in range(0,df.shape[0]):
    test = ast.literal_eval(df.iloc[i]['ratings'])

    rating_count = 0
    inspiring_count = 0
    for rating in test:
        rating_count += rating['count']
        if(rating['name'] == 'Inspiring'):
            inspiring_count = rating['count']
    insp_ratios.append(inspiring_count/rating_count)
x = 0
for ratio in insp_ratios:
    if ratio >= 0.20:
        x += 1
print(x/len(insp_ratios))
for i in range(0,df.shape[0]):
    if insp_ratios[i] > 0.2:
        df.at[i, 'ratings'] = 1
    else:
        df.at[i, 'ratings'] = 0
import numpy as np

y = np.asarray(df['ratings'], dtype=np.int64)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df = 0.05) 

vect = vectorizer.fit(df['transcript'])
transcript_dtm = vectorizer.transform(df['transcript'])
x = pd.DataFrame(transcript_dtm.toarray(), columns=vect.get_feature_names())
x = x.rename(columns = {'fit': 'fit_feature'})
x.shape
from sklearn.model_selection import train_test_split

#70% training and 30% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
x_train
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100),max_iter=15000,learning_rate_init=0.01, solver='sgd')
mlp.fit(x_train,y_train)
predictions = mlp.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix

print (confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
