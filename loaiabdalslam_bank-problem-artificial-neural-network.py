import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
from keras.models import Sequential
from keras.layers import Dense

# Read Data From and Reprocessing it
dataset = pd.read_csv('../input/Churn_Modelling.csv')
x=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values
dataset.head()
# Label Encoder Categorical Data [1:2]
Label_x1=LabelEncoder()
Label_x2=LabelEncoder()

x[:,1]=Label_x1.fit_transform(x[:,1])
x[:,2]=Label_x2.fit_transform(x[:,2])
#Let's Switch This Label to Coulmn
OneHotEncoder=OneHotEncoder(categorical_features=[1])
x=OneHotEncoder.fit_transform(x).toarray()

#removing Dummy Variable
x=x[:,1:]
#Splitting Data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#StandardScaler Variable
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
#Let's Create our CNN
model = Sequential()
# Adding the first hidden layer
model.add(Dense(input_dim=11,output_dim=6,init='uniform',activation='relu'))
# Adding the second hidden layer
model.add(Dense(output_dim=6,init='uniform',activation='relu'))
# Adding the output hidden layer
model.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

# Compiling The Model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, batch_size = 10, nb_epoch = 10)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.50)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test,y_pred)*100,'%')
