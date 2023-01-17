import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("../input/diabetes1/diabetes.csv")
data.head()

data.info()
non_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for coloumn in non_zero:
    data[coloumn] = data[coloumn].replace(0,np.NaN)
    mean = int(data[coloumn].mean(skipna = True))
    data[coloumn] = data[coloumn].replace(np.NaN,mean)
    print(data[coloumn])
sns.countplot(x="Outcome",data = data)
data["Age"].plot.hist()
data["BMI"].plot.hist()
sns.heatmap(data.isnull(),yticklabels=False, cbar=False)
X =data.iloc[:,0:8]
y =data.iloc[:,8]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()  # creating an instance of logistice regression
logmodel.fit(X_train,y_train)
pred = logmodel.predict(X_test)
pred
from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,pred)
accuracy_score(y_test,pred)
df=pd.DataFrame({'Actual':y_test, 'Predicted':pred})
df
import seaborn as sns
plt.figure(figsize=(5, 7))


ax = sns.distplot(y, hist=False, color="r", label="Actual Value")
sns.distplot(pred, hist=False, color="b", label="Fitted Values" , ax=ax)


plt.title('Actual vs Fitted Values for outCome')


plt.show()
plt.close()
