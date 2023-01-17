import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt

#load the data 
data = '../input/pima-indians-diabetes-database/diabetes.csv'
df = pd.read_csv(data)
df.describe()
df.isnull().sum()
#separate the data into x and y
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
#split the data train/test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10,random_state=0)
x_train.shape


my_model = KNeighborsClassifier(n_neighbors=27)
my_model.fit(x_train,y_train)
y_pred = my_model.predict(x_test)

accuracy = round(accuracy_score(y_test,y_pred),2)*100
print("accuracy of our model is {}%",format(accuracy))

cm = confusion_matrix(y_test,y_pred)
print(cm)


#plotting the error rate versus k value
error =[]
for a in range(1,40):
    model = KNeighborsClassifier(n_neighbors=a)
    model.fit(x_train,y_train)
    y_model = model.predict(x_test)
    error.append(np.mean(y_model!=y_test))
    
    
plt.plot(range(1,40),error,'o',color='red',linewidth=0)
plt.xlabel('K value')
plt.ylabel('error rate')
