
import pandas as pd
iris_data = pd.read_csv("../input/iris/Iris.csv")
print("First 11 rows of dataset:")
print(iris_data.head(11))
print(iris_data.isnull())
print(iris_data.fillna("Nan"))
# Q.2 Given the score of CSK, KKR, DC, and MI such that no two teams has the same score, chalk out an appropriate graph for the best display of the scores. Also, highlight the team having the highest score in the graph. 
import matplotlib.pyplot as plt
teams=['CSK','KKR','DC','MI']
scores=[65,100,82,45]
cols=['y','r','g','b']
plt.pie(scores,labels=teams,colors=cols,startangle=90,shadow=True,explode=(0,0.5,0,0),autopct='%1.1f%%')
plt.title("KKR is the winner as highlighted")
plt.show()
import numpy as np
a = np.array([0, 10, 20, 40, 60])
print("Array1: ",a)
b = np.array([10, 30, 40])
print("Array2: ",b)
print("Common values between two arrays:")
print(np.intersect1d(a, b))

for i, val in enumerate(a):
    if val in b:
        a = np.delete(a, np.where(a == val)[0][0])
for i, val in enumerate(b):
    if val in a:
        a = np.delete(a, np.where(a == val)[0][0])
print("Array after deletion of common element : ")
print(a)
print(b)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('../input/iris/Iris.csv', error_bad_lines=False)
df = df.drop(['Id'], axis=1)
df['Species'] = pd.factorize(df["Species"])[0] 
Target = 'Species'
Features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

model = LogisticRegression(solver='lbfgs', multi_class='auto')
Features = ['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

x, y = train_test_split(df, 
                        test_size = 0.2, 
                        train_size = 0.8, 
                        random_state= 3)

x1 = x[Features]
x2 = x[Target]
y1 = y[Features]
y2 = y[Target]

nb_model = GaussianNB() 
nb_model.fit(X=x1, y=x2)
result= nb_model.predict(y[Features]) 

f1_sc = f1_score(y2, result, average='micro')
confusion_m = confusion_matrix(y2, result)

print("F1 Score    : ", f1_sc)
print("Confusion Matrix: ")
print(confusion_m)