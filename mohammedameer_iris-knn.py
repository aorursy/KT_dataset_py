from sklearn import datasets
iris = datasets.load_iris()
iris
iris.data
# Sepal Length
iris.data[:,0]
#sepal width
iris.data[:,1]
#petal length
iris.data[:,2]
#petal width
iris.data[:,3]

import pandas as pd
df = pd.DataFrame({'Sepal Length':iris.data[:,0],
                   'Sepal Width':iris.data[:,1],
                   'Petal Length':iris.data[:,2],
                   'Petal Width':iris.data[:,3]
                  })
df
df['Species'] = iris.target
df
iris.target_names
x = df.iloc[:,0:4].values
y = df.iloc[:,4].values
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x,y)
model.predict([[5.9,3.0,5.1,1.8]])
iris.target_names
%%writefile app.py
# It saves the file in .py format 
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

def user_input():
  sepal_length = st.sidebar.slider('Sepal Length',4.3,7.9,5.4)   # min,max,initial
  sepal_width = st.sidebar.slider('Sepal Width',2.0,4.4,3.4)
  petal_length = st.sidebar.slider('Petal Length',1.0,6.9,1.4)
  petal_width = st.sidebar.slider('Petal Width',0.1,2.5,0.2)
  data = {'sepal_length':sepal_length,
          'sepal_width' : sepal_width,
          'petal_length': petal_length,
          'petal_width': petal_width}
  features = pd.DataFrame(data,index = [0])
  return features

st.write("# Simple **Iris Flower** Prediction App")
st.write("# Iris Dataset")
st.write(" Number of Classes : 3")
st.write(" Classifier : KNN")
st.write("Accuracy = 0.95")
st.sidebar.header('User Input Parameters')
st.subheader('User Input Parameters')

df = user_input()
st.write(df)

iris = datasets.load_iris()
x = iris.data
y = iris.target

# Apply KNN algorithm

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x,y)
prediction = model.predict(df)

st.subheader('Class Labels and their corresponding index number')
st.write(iris.target_names)


st.subheader('Prediction')
st.write(iris.target_names[prediction])

!pip install pyngrok
!pip install streamlit

from pyngrok import ngrok

public_url = ngrok.connect(port='8501')
print(public_url)

!streamlit run app.py


