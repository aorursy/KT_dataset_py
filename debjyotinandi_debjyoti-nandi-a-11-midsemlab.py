import pandas as pd
import numpy as np
data = pd.read_csv("../input/ipl/deliveries.csv")
print("The first 11 rows of the data set is the following ")
print(data.iloc[0:11])
print(" The row with a missing value : ")
print(data[data['player_dismissed'].isnull()])
print(" Replacing all the null with nan")
data[data.isnull().any(axis=1)].replace(0,np.nan)
from matplotlib import pyplot as plt

team=['CSK','KKR','DC','MI']
Scores=[110,140,102,95]

plt.bar(team, Scores, width = 0.8, color=['orange','red','orange','orange'])

plt.xlabel('Team') 
plt.ylabel('Points')
plt.title('Team Points Chart')
plt.show()
import numpy as np
a = np.array([1,2,3,4,5,6,7,8])
print("Array 1: ",a)
b = np.array([1,16,4,38,7,9])
print("Array 2: ",b)
print("Common values between two arrays:")
print(np.intersect1d(a, b))

for i, val in enumerate(a):
    if val in b:
        a = np.delete(a, np.where(a == val)[0][0])
for i, val in enumerate(b):
    if val in a:
        a = np.delete(a, np.where(a == val)[0][0])
print("Arrays after deletion of common elements : ")
print("First array")
print(a)
print("Second array")
print(b)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

train = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")


X = train.drop("species",axis=1)
y = train["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

print("F1 Score(macro):",f1_score(y_test, predictions, average='macro'))
print("F1 Score(micro):",f1_score(y_test, predictions, average='micro'))
print("F1 Score(weighted):",f1_score(y_test, predictions, average='weighted')) 
print("\nConfusion Matrix(below):\n")
confusion_matrix(y_test, predictions)