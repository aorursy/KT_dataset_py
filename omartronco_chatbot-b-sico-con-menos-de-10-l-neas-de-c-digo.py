# Primero hay que importar pandas

import pandas as pd
# Se importa una base de datos si se cuenta con ella

# data = pd.read_csv('train_data.csv')



# Si no se cuenta con ella, se simula tener una

data = pd.DataFrame([['el seguro cubre accidentes','Sí cubre accidentes.'],

                    ['el seguro cubre robo','Sí cubre robo total. El de autopartes, no.'],

                    ['el seguro cubre choques','Sí cubre accidentes.'],

                    ['cubre si le cae un arbol','Sí cubre esas locuras.'],

                    ['cubre inundación?','Sí cubre inundación.']])

data.columns = ['question','answer']



# NOTA: Esta base contiene una mala redacción y faltas de ortografía en las preguntas.

# Esto es intencional, ya que queremos replicar el lenguaje natural que usan los usuarios.

# Sin embargo debemos cuidar que las respuestas estén bien redactadas, pues es como responderá el bot.



data
from sklearn.feature_extraction.text import TfidfVectorizer # Para convertir el texto a vectores

from scipy import spatial # Para medir la distancia entre vectores
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data.question) 

# La línea de código anterior convierte la columna 'question' en vectores,

# es decir, cada pregunta la convierte en una serie de números.
pd.DataFrame(vectorizer.get_feature_names())
pd.DataFrame(X.toarray()[0])
question = ['este seguro loco me cubre los todos los choques que tenga con mi coche?']

question
vectorized_question = vectorizer.transform(question).toarray()

pd.DataFrame(vectorized_question)
pd.DataFrame(vectorizer.get_feature_names())
tree = spatial.KDTree(X.toarray())

pd.DataFrame(tree.query(vectorized_question))
data.iloc[tree.query(vectorized_question)[1][0]].question
data.iloc[tree.query(vectorized_question)[1][0]].answer
#import pandas as pd

#from sklearn.feature_extraction.text import TfidfVectorizer

#from scipy import spatial



#data = pd.read_csv('train_data.csv')

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data.question)

tree = spatial.KDTree(X.toarray())



def answer(question):

    vectorized_question = vectorizer.transform(question).toarray()

    tree.query(vectorized_question)

    print('La pregunta fue: ',question)

    print('El bot entendió: ',data.iloc[tree.query(vectorized_question)[1][0]].question)

    print('El bot responde: ',data.iloc[tree.query(vectorized_question)[1][0]].answer)
# Se introduce una oración y encuentra la definición más cercana

sentence = ['este seguro loco me cubre los todos los choques que tenga con mi coche?']

answer(sentence)