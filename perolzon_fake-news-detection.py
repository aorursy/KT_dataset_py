import pandas as pd

fake_data=pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")

true_data=pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")



#Set labels for the datasets

fake_data["target"]=0

true_data["target"]=1



#Concat and reindex the data

all_data=pd.concat([fake_data,true_data],ignore_index=True)



# View the dataset



all_data
import nltk

from nltk.tokenize import word_tokenize

from spacy.lang.en.stop_words import STOP_WORDS



def preprocess(data):

    txt = data.text.str.lower().str.cat(sep=' ')

    words = nltk.tokenize.word_tokenize(txt)

    words=[word.lower() for word in words if word.isalpha()]

    

    alsowordstoremove=["t","s","don","wouldn","won","couldn"]

    words = [w for w in words if not w in STOP_WORDS] 

    words = [w for w in words if not w in alsowordstoremove]

    

    tags=nltk.pos_tag(words)

    res_chunk = nltk.ne_chunk(tags)

    return words





filtered_words=preprocess(true_data)

word_dist = nltk.FreqDist(filtered_words)

top_N=30



df = pd.DataFrame(word_dist.most_common(top_N),

                        columns=['Word true data', 'Frequency true data'])



filtered_fake_words=preprocess(fake_data)

word_dist = nltk.FreqDist(filtered_fake_words)

df2=pd.DataFrame(word_dist.most_common(top_N),

                      columns=['Word fake data','Frequency fake data'])



result=pd.merge(df,df2,left_index=True, right_index=True)

print('All frequencies, not including STOPWORDS:')

print('=' * 60)

print(result)
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score



#Using linear SVC to speed up classification

pipe = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', LinearSVC())])



x_train,x_test,y_train,y_test = train_test_split(all_data['text'], all_data.target, test_size=0.2,

                                                 random_state=20

                                                )



model = pipe.fit(x_train,y_train)



prediction= model.predict(x_test)

score=accuracy_score(y_test,prediction)

print(round(score*100,3))
test=pd.Series(["Reuters is a fake news outlet that destroys democratic values."])

prediction = model.predict(test)

prediction[0]

# 0 is fake news, while 1 is a true news.