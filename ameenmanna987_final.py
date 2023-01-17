import pandas as pd
df= pd.read_csv('../input/r-left/r_left.csv')
df.info()
df
x =df.iloc[:,13:14].values # X is our INPUT  #X1 = r_left.iloc[:,[15]]  # AMBIENT TEMP COLUMN
y =df.iloc[:,4].values  # y is our OUTPUT #AC POWER COLUMN

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test =tts(x,y,test_size =0.3,random_state=0) #xtrain,xtest,yrain,ytest
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x_train,y_train)
y_pred =lin_reg.predict(x_test)
y_pred    # prediction value
y_test
#1
import pickle
pickle.dump(lin_reg,open('model.pkl','wb'))  #write bit for wb

 #2
!pip install streamlit
!pip install pyngrok
# %%writefile amn.py
# import streamlit as st
# import pickle 
# st.title('MY STREAMLIT APP')
# st.subheader('WRITE ANY DESCRIPTION')
# st.write('Write predicting what vs what')
# xyz = pickle.load(open('model.pkl','rb')) #read bit -rd
# at = st.number_input('Enter AMBIENT TEMP')
# xyz.predict([[at]])
# if st.button('Predict'):
#     st.title(xyz.predict([[at]]))
%%writefile amn.py
import streamlit as st
import pickle 
import matplotlib.pyplot as plt
st.title('MY STREAMLIT APP')
st.subheader('WRITE ANY DESCRIPTION')
st.write('Write predicting what vs what')
xyz = pickle.load(open('model.pkl','rb')) #read bit -rd
at = st.number_input('Enter AMBIENT TEMP')
plt.scatter(x_test,y_test,color ='gray')
plt.scatter(x_test,y_pred,color='red')
xyz.predict([[at]])
if st.button('Predict'):
    st.title(xyz.predict([[at]]))
from pyngrok import ngrok
url = ngrok.connect(port = '8501')
url


!streamlit run amn.py