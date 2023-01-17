import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import csv





from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



bill_contents = pd.read_csv('../input/discount-prediction/Product_sales_train_and_test.csv')

train = pd.read_csv('../input/discount-prediction/Train.csv')

test = pd.read_csv('../input/discount-prediction/test.csv')



final_train = pd.read_csv('../input/preprocessed/training_set.csv')

final_train.info()



# Code for pre-processing is present in another file

# After Pre-processing of data



X = pd.read_csv('../input/preprocessed/training_input.csv')

y = pd.read_csv('../input/preprocessed/training_output.csv')



# Data Visualization

categories = list(y.columns.values)

sns.set(font_scale = 1)

plt.figure(figsize=(5,2))

ax= sns.barplot(categories, y.iloc[:,0:].sum().values)





# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)





# Test data

X_testt = pd.read_csv('../input/preprocessed/test_input.csv')



# Fitting K-NN to the Training set

classifier = KNeighborsClassifier(n_neighbors = 47, metric = 'minkowski', p = 2, n_jobs = -1)

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)



print("Accuracy :",accuracy_score(y_test,y_pred))



# Fitting classifier to test data

y_prediction = classifier.predict(X_testt)





# Submission file

submission = pd.DataFrame(y_prediction)

submission.columns = ['Discount 5%', 'Discount 12%', 'Discount 18%', 'Discount 28%']





col=[]

with open('../input/preprocessed/test_input.csv', mode='rt') as file:

    data=csv.reader(file)

    for row in data:

        col.append(row[0])

        

del col[0]

submission.insert(0, 'BillNo', col)



# Results are mapped to a csv file

submission.to_csv("final_predictions.csv", index=False)
