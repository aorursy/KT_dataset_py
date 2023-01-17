import pandas as pd

import numpy as np

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import  WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')
data.shape
data.head()
data.drop(data.iloc[:, 2:], inplace = True, axis = 1) 
data.head()
def textProcessing(df):

    tokens=[]

    for i in range(len(df)):

        tokens.append(word_tokenize(df['v2'].iloc[i].lower()))

        

    eng_stopwords=stopwords.words('english')

    eng_stopwords.extend([',','.','!','@','#','?','-'])

    

    main_words=[]    

    for i in range(len(tokens)):

        words=[]

        for token in tokens[i]:

            if token not in eng_stopwords:

                words.append(token)

        main_words.append(words)

        

    wnet=WordNetLemmatizer()

    

    for i in range(len(main_words)):

        for j in range(len(main_words[i])):

            main_words[i][j]=wnet.lemmatize(main_words[i][j],pos='v')

            

    for i in range(len(main_words)):

        main_words[i]=" ".join(main_words[i])

        

    

    return main_words
wordlist=textProcessing(data)
data["v1"]=data["v1"].map({'spam':1,'ham':0})
tfidf=TfidfVectorizer()
vector = tfidf.fit_transform(wordlist).toarray()
print(vector.shape)
x_train,x_test,y_train,y_test = train_test_split(vector,data['v1'],test_size=0.33,random_state=32)
print([np.shape(x_train), np.shape(x_test)])
nb=MultinomialNB()
nb.fit(x_train,y_train)
y_pred = nb.predict(x_test)
accuracy_score(y_test,y_pred)
conf_matrix=confusion_matrix(y_test,y_pred)

pd.DataFrame(data = conf_matrix, columns = ['Predicted 0', 'Predicted 1'],

            index = ['Actual 0', 'Actual 1'])