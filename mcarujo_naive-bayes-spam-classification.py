!pip install nb_black -q
%load_ext nb_black
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv("../input/spam-ham-emails/emails.csv")

data.head()
data.info()
from wordcloud import WordCloud

import matplotlib.pyplot as plt





def wordcloud(df, column_name, spam, title):

    text = df.query(f"spam == '{spam}'")

    # Juntando todos os textos na mesma string

    all_words = " ".join([text for text in text[column_name]])

    # Gerando a nuvem de palavras

    wordcloud = WordCloud(

        width=800, height=500, max_font_size=110, collocations=False

    ).generate(all_words)

    # Plotando nuvem de palavras

    plt.figure(figsize=(24, 12))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.title(title)

    plt.show()





wordcloud(data, "text", "1", "Word cloud for SPAM")
wordcloud(data, "text", "0", "Word cloud for HAM")
import plotly.graph_objects as go



counts = data.spam.value_counts()





def counts_percent(cv):

    aux = []

    percent = cv / sum(cv)

    for i in range(0, len(percent)):

        aux.append(f"{cv[i]} ({round(percent[i],2)*100} %)")

    return aux





fig = go.Figure(

    data=[

        go.Bar(

            x=counts.index,

            y=counts.values,

            text=counts_percent(counts.values),

            textposition="auto",

        )

    ]

)



fig.update_layout(

    title="Counts classes: Spam as 1 and Ham as 0",

    yaxis=dict(title="Count value", titlefont_size=16, tickfont_size=14,),

    xaxis=dict(title="Classes", titlefont_size=16, tickfont_size=14,),

)

fig.show()
from sklearn.feature_extraction.text import CountVectorizer



vectorize = CountVectorizer()



X = vectorize.fit_transform(data.text)

Y = data.spam.values

print("How many features (bag of words): ", len(vectorize.get_feature_names()))
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



X_train, X_test, y_train, y_test = train_test_split(

    X, Y, test_size=0.33, random_state=42

)



model = MultinomialNB()



model.fit(X_train, y_train)





# Let's see the results

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions) * 100

print("The accuracy was %.2f%%" % accuracy)
from sklearn.metrics import confusion_matrix



m_c = confusion_matrix(y_test, predictions)

plt.figure(figsize=(5, 4))

sns.heatmap(m_c, annot=True, cmap="Reds", fmt="d").set(xlabel="Predict", ylabel="Real")