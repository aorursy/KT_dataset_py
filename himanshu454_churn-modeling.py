import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , LabelEncoder


import keras
from keras.models import Sequential
from keras.layers import Dense , Dropout
data = pd.read_csv("../input/annchurn-modelling/Churn_Modelling.csv")
data.head(10)
data.drop(["RowNumber" , "CustomerId" , "Surname"] , axis=1 , inplace = True)
data.head(5)
data.info()
data.describe()
l1 = LabelEncoder()
data["Geography"] = l1.fit_transform(data["Geography"])
l2 = LabelEncoder()
data["Gender"] = l2.fit_transform(data["Gender"])
data.head(5)
sns.heatmap(data.isnull() , yticklabels=False)
sns.countplot(data["Exited"])
sns.boxplot(data["CreditScore"])
sns.boxplot(data["Tenure"])
sns.boxplot(data["NumOfProducts"])
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
train_x , test_x , train_y , test_y = train_test_split(x,y,test_size = 0.2 , random_state = 43)
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.fit_transform(test_x)
train_x.shape
model = Sequential()
model.add(Dense(units = train_x.shape[0] , activation = 'relu' , input_dim = 10))
model.add(Dropout(0.5))
model.add(Dense(512 , activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(128 ,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1 ,activation = 'sigmoid'))
model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(train_x , train_y , batch_size = 128 , epochs = 100)
plt.plot(history.history["loss"])
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.plot(history.history["accuracy"])
plt.xlabel("Epcohs")
plt.ylabel("Accuracy")