import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
dataset=pd.read_csv('../input/train.csv')

dataset.head()
dataset.info()
dataset.describe()
X=dataset.drop('price_range',axis=1)

y=dataset['price_range']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train,y_train)
score = knn.score(X_test,y_test)

print("Test score: {0:.2f} %".format(100 * score))
import pickle



Pkl_Filename = "Pickle_KNN_Model.pkl"  
# Save the Modle to file in the current working directory



with open(Pkl_Filename, 'wb') as file:  

    pickle.dump(knn, file)
# Load the Model back from file

with open(Pkl_Filename, 'rb') as file:  

    Pickled_KNN_Model = pickle.load(file)



Pickled_KNN_Model

# Use the Reloaded Model to 

# Calculate the accuracy score and predict target values



# Calculate the Score 

score = Pickled_KNN_Model.score(X_test,y_test)  

# Print the Score

print("Test score: {0:.2f} %".format(100 * score)) 
from sklearn.metrics import classification_report,confusion_matrix
pred = Pickled_KNN_Model.predict(X_test)
print(classification_report(y_test,pred))
matrix=confusion_matrix(y_test,pred)

print(matrix)
#DATA INPUT

id=1

battery_power=2000

blue=1

clock_speed=1.8

dual_sim=1

fc=14

four_g=1

int_memory=5

m_dep=0.1

mobile_wt=193

n_cores=8

pc=16

px_height=1720

px_width=720

ram=4000

sc_h=12

sc_w=7

talk_time=2

three_g=1

touch_screen=1

wifi=1



data_test = pd.DataFrame({'id':[id],'battery_power':[battery_power],'blue':[blue],'clock_speed':[clock_speed],'dual_sim':[dual_sim],'fc':[fc],'four_g':[four_g],'int_memory':[int_memory],'m_dep':[m_dep],'mobile_wt':[mobile_wt],'n_cores':[n_cores],'pc':[pc],'px_height':[px_height],'px_width':[px_width],'ram':[ram],'sc_h':[sc_h],'sc_w':[sc_w],'talk_time':[talk_time],'three_g':[three_g],'touch_screen':[touch_screen],'wifi':[wifi]})

data_test=data_test.drop('id',axis=1)

data_test.head()
predicted_price=Pickled_KNN_Model.predict(data_test)
print("PREDIKSI HARGA = ", predicted_price[0])
data_test['price_range']=predicted_price

data_test