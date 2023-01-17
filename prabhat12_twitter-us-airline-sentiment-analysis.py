# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

data=pd.read_csv("../input/Tweets.csv")

data.head()
data["airline_sentiment"]=data.airline_sentiment.map({"neutral":"0","positive":"1","negative":"-1"})

data.head()
X=data.iloc[:,10]

print(X.size)

Y=data.iloc[:,1]

sentiment=data['airline_sentiment'].value_counts()



label=['Negative','Neutral','Positive']

index = np.arange(len(label))



plt.bar(index,sentiment)

plt.xticks(index,label,rotation=45)

plt.ylabel('Sentimen Count')

plt.xlabel('Sentiment')

plt.title('Sentiment')
def plot_for_each_airline(airline_name):

        airline_data=data[data['airline']==airline_name]

        sentiment=airline_data['airline_sentiment'].value_counts()

        label=['Negative','Neutral','Positive']

        index = np.arange(len(label))

        

        plt.bar(index,sentiment)

        plt.xticks(index,label,rotation=45)

        plt.ylabel('Sentimen Count')

        plt.xlabel('Sentiment')

        plt.title(airline_name)
plot_for_each_airline("Virgin America")
plot_for_each_airline("American")
plot_for_each_airline("United")

plot_for_each_airline("Southwest")

plot_for_each_airline("Delta")
plot_for_each_airline("US Airways")
from sklearn.feature_extraction.text import CountVectorizer



vectorizer=CountVectorizer()

bag_of_words=vectorizer.fit_transform(X)

vc=vectorizer.get_feature_names

# uncomment to see bag of words , its lengthy so not showing

#print(bag_of_words.toarray())

#print( vectorizer.vocabulary_)
X_train,X_test,Y_train,Y_test=train_test_split(bag_of_words,Y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier()

model.fit(X_train.toarray(),Y_train)



predict=model.predict(X_test.toarray())

print(model.score(X_test.toarray(),Y_test))

print(model.score(bag_of_words,Y))

def user_defined_sentences(statement):

    st=[s]

    vect=vectorizer.transform(st)

    print(vect.toarray())

    predict=model.predict(vect.toarray())

    print(predict)

    polarity=model.predict_proba(vect.toarray())

    print(polarity)



    # uncomment to see linear graph

    # plt.plot(index,polarity.reshape(-1,1))

    # plt.show()

    

    # plotting horizontal bar graph for better demonstration 

    plt.bar(index,polarity[0])

    plt.ylabel('Probability',fontsize=10)

    plt.xlabel('Prediction',fontsize=10)

    plt.xticks(index, label, fontsize=10, rotation=40)

    plt.title('Sentiment of Sentence')

    plt.show()

s="Not happy with the flight, too boring and late"

user_defined_sentences(s)
