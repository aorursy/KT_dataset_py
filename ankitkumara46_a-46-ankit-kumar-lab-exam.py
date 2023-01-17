# Q2.
import pandas as pd
df = pd.read_csv("../input/titanic/train_and_test2.csv")
df.head()

df.dropna(axis=1, how='all')
print(df.head())
print(df.shape)

df[:50].mean()

df[df['Sex']==1].mean()

df['Fare'].max()
# Q1.
import numpy as np
a=np.array([9,5,4,3,2,6])
b=np.array([5,8,6,9,2,1])
print("CHECK IF B HAS SAME VIEWS TO MEMORY IN A")
print(b.base is a)
print("CHECK IF A HAS SAME VIEWS TO MEMORY IN B")
print(a.base is b)
div_by_3=a%3==0
div1_by_3=b%3==0
print("Divisible By 3")
print(a[div_by_3])
print(b[div1_by_3])
b[::-1].sort()
print("SECOND ARRAY SORTED")
print(b)
print("SUM OF ELEMENTS OF FIRST ARRAY")
print(np.sum(a))
# Q3.
import matplotlib.pyplot as plt
import numpy as np
x=["English","Math","Science","History","Geography"]
y=[86,83,86,90,88]
plt.bar(x,y)
plt.bar(x[1],y[1],color='r')
# Q4.
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pandas as pd
train = pd.read_csv("../input/iris/Iris.csv")


X = train.drop("Species",axis=1)
y = train["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

print("F1 Score:",f1_score(y_test, predictions,average='weighted'))
 
print("\nConfusion Matrix(below):\n")
confusion_matrix(y_test, predictions)

