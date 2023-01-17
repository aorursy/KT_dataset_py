import pandas as pd

import numpy as np



cricket_data = pd.read_csv("../input/custom1/customcricket_dataset.csv")

df = pd.DataFrame(cricket_data)

print(cricket_data) #this to check which column has null/missing values



print('\n\n')

print(df.head(3))



print('\n\n')

null = df[df['score'].isnull()] #score with NULL 

print(null)

print('\n\n')

from matplotlib import pyplot as plt



team=['CSK','KKR','DC','MI']

points=[4,2,3,1]



plt.bar(team, points, width = 0.8, color=['red','green','green','green'])



plt.xlabel('Team') 

plt.ylabel('Points')

plt.title('Team Points Chart')

plt.show()
import numpy as np

import pandas as pd



a = np.array([1,2,3,4,5,6])



b = np.array([2,4,6])



c = np.intersect1d(a,b) #printing the common items



print(c)

print("\n")

for i in b:

    for j in a:

        if i == j:

            a = a[a!=j] #removing the common items from the array "a"

print(a)

print("\n")

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score



train = pd.read_csv("../input/titanic/train_data.csv")





X = train.drop("Survived",axis=1)

y = train["Survived"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)



logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)



predictions = logmodel.predict(X_test)



print("F1 Score:",f1_score(y_test, predictions))

 

print("\nConfusion Matrix(below):\n")

confusion_matrix(y_test, predictions)
