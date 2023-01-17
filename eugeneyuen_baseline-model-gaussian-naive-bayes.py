# Import the dependencies

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/ufcdata/data.csv")
#Prepare the training set and selecting features

#Dropped na values and columns that is not significant

df.dropna(inplace = True)

df.drop(df[df['Winner'] == 'Draw' ].index, inplace = True)

x_clean = df.drop(['Winner','R_fighter','B_fighter','Referee','date','location','R_Stance', 'B_Stance','weight_class'], axis=1)



x = x_clean # x = feature values





# Target value

y = df.loc[:, df.columns == 'Winner']
x.head()
y.head()
#Split the data into 80% training and 20% testing

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#Train the model

model = GaussianNB()

model.fit(x_train, y_train) #Training the model
#Test the model

predictions = model.predict(x_test)



#Check precision, recall, f1-score

print(classification_report(y_test, predictions) )
