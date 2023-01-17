# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, roc_curve, confusion_matrix, accuracy_score

from sklearn.neighbors import KNeighborsClassifier

plt.style.use('seaborn')
# read in csv file of diabetes

df = pd.read_csv('/kaggle/input/diabetes.csv')

df.head()
# use info() method to get an idea of the data-types, column names, and null value counts

df.info()
df.describe().T
df.shape
# create a copy of data frame 

df_new = df.copy(deep=True)
# calculate mean of each column/feature that has null values

gluc_avg = df_new['Glucose'].mean()

blood_avg = df_new['BloodPressure'].mean()

skin_avg = df_new['SkinThickness'].mean()

insu_avg = df_new['Insulin'].mean()

bmi_avg = df_new['BMI'].mean()
# replace null values with mean

df_new['Glucose'].replace(0, gluc_avg, inplace = True)

df_new['BloodPressure'].replace(0, blood_avg, inplace = True)

df_new['SkinThickness'].replace(0, skin_avg, inplace = True)

df_new['Insulin'].replace(0, insu_avg, inplace = True)

df_new['BMI'].replace(0, bmi_avg, inplace = True)
df_new.describe().T
# pd.to_csv(df_new)
# view distribution of variable

df_new.hist(figsize=(15,15))
# identify 'outcome' column and count values for each outcome

outcome_counts = df['Outcome'].value_counts().sort_values()

print(outcome_counts)

outcome_counts.plot(kind='bar')

plt.title('Outcome')

plt.ylabel('Total')
sns.pairplot(df_new, hue='Outcome')
# identify correlation between dependent and independent variables

df_new.corr()['Outcome'].sort_values()
# independent/feature variables

X = df_new[['Pregnancies', 'Glucose', 

            'BloodPressure', 'SkinThickness', 

            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 

            'Age']]

# dependent/target variablea

Y = df_new['Outcome']



# create scale object

scale = StandardScaler()

# scale x_data ie features

X = scale.fit_transform(X)
# train test and split data

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)



print(x_train.shape, y_train.shape)

print(x_test.shape, y_train.shape)
ks = 30

train_score = []

test_score = []



for i in range(1,ks):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train, y_train)

    yhat = knn.predict(x_test)

    

    train_score.append(knn.score(x_train, y_train))

    test_score.append(knn.score(x_test, y_test))     # alternatively use metrics.accuracy_score(y_test, yhat)
# convert to numpy array for access of best k value

test_score = np.array(test_score)

train_score = np.array(train_score)
# best k value

best_k = test_score.argmax()

print('Best value for k: ', best_k)

high_score = test_score.max()  # alternatively use test_score[9]

print('Accuracy score of: ', high_score)
# visualise training and testing data

sns.lineplot(range(ks-1), test_score, color='g', label='Testing Score')

sns.lineplot(range(ks-1), train_score, color='b', label='Training Score')

plt.title('Best Values for K')

plt.xlabel('K Values')

plt.ylabel('Accuracy Scores')

plt.show()
k = 23



# build model

KNN = KNeighborsClassifier(n_neighbors=k)

KNN.fit(x_train, y_train)



# prediction

yhat = KNN.predict(x_test)

final_score = KNN.score(x_test, y_test)

print('Score: ', final_score)

# Confusion Matrix

matrix = confusion_matrix(y_test, yhat)



sns.heatmap(pd.DataFrame(matrix), annot=True,  cmap="YlGnBu" , fmt='.2g')

plt.title('Confusion matrix')

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
# Classification report

class_report = classification_report(y_test,yhat)

print(class_report)
# another way to see results from the confusion matrix

print('TP - True Negative: ',matrix[0,0])

print('FP - False Positive: ', matrix[0,1])

print('FN - False Negative: ', matrix[1,0])

print('TP - True Positive: ', matrix[1,1])

print('Accuracy Rate: ', np.divide(np.sum([matrix[0,0],matrix[1,1]]),np.sum(matrix)))

print('Misclassification Rate: ', np.divide(np.sum([matrix[0,1],matrix[1,0]]),np.sum(matrix)))