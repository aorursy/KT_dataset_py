import pandas as pd

import numpy as np



#Gráficos

import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from PIL import Image



#Machine Learning

from sklearn.feature_extraction.text import CountVectorizer



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import MultinomialNB
data = pd.read_json('../input/amazon-music-reviews/Musical_Instruments_5.json', lines=True)

data
data.info()
data.describe()
pd.isnull(data).sum()
data['Review']=data['reviewText']+''+data['summary']

data
data['Month'],data['Day'],data['Year']=data['reviewTime'].str.split().str

data
data=data.drop(['reviewerID','reviewerName','helpful','reviewText','summary','reviewTime'],axis=1)

data
condicion=[(data['overall']==1), (data['overall']==2), (data['overall']==3),

          (data['overall']==4), (data['overall']==5)]

resultados=[0,0,0,1,1]

data['R_Review']=np.select(condicion,resultados,default=np.nan)
data
data.groupby('R_Review').size()
fig= px.pie(data,values='overall',

            names='overall',

            title='Calificación del instrumento',

           color_discrete_sequence=px.colors.sequential.Rainbow_r)



fig.show()
rating=data.groupby('overall').size()



fig=px.bar(rating, text=rating, color=rating,

          color_continuous_scale=px.colors.diverging.Portland)



fig.update_layout(title='Calificación del instrumento',

                 xaxis_title='Calificaciones',

                 yaxis_title='Reseñas',

                 legend_title='Calificaciones')



fig.show()
ventas_de_productos_por_asin=data.groupby('asin').size()

Productos_mas_vendidos=ventas_de_productos_por_asin[ventas_de_productos_por_asin>20]



fig=px.bar(Productos_mas_vendidos,

           text=Productos_mas_vendidos,

           color=Productos_mas_vendidos,

           color_continuous_scale=px.colors.diverging.Portland)



fig.update_layout(title='Instrumentos más vendidos',

                 xaxis_title='asin de los productos',

                 yaxis_title='Cantidad')



fig.show()
Productos_menos_vendidos=ventas_de_productos_por_asin[ventas_de_productos_por_asin<11]



fig=px.bar(Productos_menos_vendidos,

           text=Productos_menos_vendidos,

           color=Productos_menos_vendidos,

           color_continuous_scale=px.colors.diverging.Portland)



fig.update_layout(title='Instrumentos menos vendidos',

                 xaxis_title='asin de los productos',

                 yaxis_title='Cantidad')



fig.show()
fig= px.pie(data,values='Year',

            names='Year',

            title='Distribución de Instrumentos según el año',

           color_discrete_sequence=px.colors.sequential.Rainbow_r)



fig.show()
Sentiment=data.groupby('R_Review').size()



fig= px.pie(values=Sentiment,

            names=Sentiment,

            title='Distribución de sentimientos',

            color_discrete_sequence=px.colors.sequential.GnBu_r)



fig.show()
cv=CountVectorizer()

x_train_cv=cv.fit_transform(data['Review'])

cv.vocabulary_
x_train_cv.shape
tfidf_vectorizer=TfidfVectorizer()

x_train_tfidf=tfidf_vectorizer.fit_transform(data['Review'])

tfidf_vectorizer.vocabulary_
sum_words=x_train_cv.sum(axis=0)



words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]

words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])
frequency.head()
fig=px.bar(frequency[:41],x='word',y='freq',color='freq',

           color_continuous_scale=px.colors.diverging.Portland)



fig.update_layout(title='Las 40 palabras más usadas',

                 xaxis_title='Palabras',

                 yaxis_title='Cantidad')



fig.show()
fig=px.bar(frequency[-41:],x='word',y='freq',color='freq',

           color_continuous_scale=px.colors.diverging.Portland)



fig.update_layout(title='Las 40 palabras menos usadas',

                 xaxis_title='Palabras',

                 yaxis_title='Cantidad')



fig.show()
wordcloud = WordCloud(background_color = 'white',

                      max_words=200, 

                      width = 1000, 

                      height = 1000).generate_from_frequencies(dict(words_freq))



plt.figure(figsize=(10, 10))

plt.axis('off')

plt.imshow(wordcloud)

plt.title("Palabras más comunes", fontsize = 20)

plt.show()
amazon_mask=np.array(Image.open("../input/images/amazon_logo.jpg"))



wc = WordCloud(background_color="black", max_words=2000, mask=amazon_mask).generate_from_frequencies(dict(words_freq))



plt.figure(figsize=(15,10))

plt.imshow(wc)

plt.axis("off")

plt.figure()
wc.to_file("amazon.png")
data=data.drop(['asin','overall','unixReviewTime','Month','Day','Year'],axis=1)

data.head()
x=data['Review']

y=data['R_Review']
x_cv=cv.fit_transform(x)
X_train, X_test, y_train ,y_test = train_test_split(x_cv, y, test_size = 0.2 , random_state = 42)
AD=DecisionTreeClassifier()

AD.fit(X_train,y_train)

y_pred=AD.predict(X_test)

print('Presición del algoritmo Árbol de Decisiones es: {}'.format(AD.score(X_train,y_train)*100))

print('Presición del algoritmo Árbol de Decisiones es: {}'.format(accuracy_score(y_test,y_pred)*100))
mnb_model=MultinomialNB()

mnb_model.fit(X_train, y_train)

y_pred=mnb_model.predict(X_test)

print('Presición del algoritmo Árbol de Decisiones es: {}'.format(mnb_model.score(X_train,y_train)*100))

print('Presición del algoritmo Árbol de Decisiones es: {}'.format(accuracy_score(y_test,y_pred)*100))
x_tfidf=tfidf_vectorizer.fit_transform(x)
X_train, X_test, y_train ,y_test = train_test_split(x_tfidf, y, test_size = 0.2 , random_state = 42)
AD.fit(X_train,y_train)

y_pred=AD.predict(X_test)

print('Presición del algoritmo Árbol de Decisiones es: {}'.format(AD.score(X_train,y_train)*100))

print('Presición del algoritmo Árbol de Decisiones es: {}'.format(accuracy_score(y_test,y_pred)*100))
mnb_model.fit(X_train, y_train)

y_pred=mnb_model.predict(X_test)

print('Presición del algoritmo Árbol de Decisiones es: {}'.format(mnb_model.score(X_train,y_train)*100))

print('Presición del algoritmo Árbol de Decisiones es: {}'.format(accuracy_score(y_test,y_pred)*100))