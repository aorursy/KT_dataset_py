import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

import sklearn.metrics as metrics





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
data.head()
# getting to know our column names in one place

data.columns
# setting our feature and target from the data

# we'll do away with the target and one other feature 

features = data.drop(['SkinThickness','Outcome'], axis=1)

target = data.Outcome



# for simplicity

X = features

y = target
# splitting our data into train and validation set

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=1)
# making our prediction with LogisticRegressor



# instantiate the model

model = LogisticRegression()



# fitting i.e training the model in a subtle manner

model.fit(X_train, y_train)
# making prediction so soon :)

prediction = model.predict(X_valid)
CF_matrix = metrics.confusion_matrix(y_valid, prediction)

print('diagonal values for Actdual prediction made \n None diagonal for inacurrate ones' )

CF_matrix
clases = [0, 1]

fig, ax = plt.subplots()

tick_marks = np.arange(len(clases))

plt.xticks(tick_marks, clases)

plt.yticks(tick_marks, clases)



# creating a heatmap

# convert matrix into dataframe

CFmatrix = pd.DataFrame(CF_matrix)

# **************** #####



sns.heatmap(CFmatrix, annot=True,cmap='YlGnBu', fmt='g' )



plt.tight_layout()

plt.title('Cconfusion Matrix', y=1.1)

plt.ylabel('Actual Label')

plt.xlabel('Predicted label')
print('Accuracy:', metrics.accuracy_score(y_valid, prediction))

print('Precision:', metrics.precision_score(y_valid, prediction))

print('Recall:', metrics.recall_score(y_valid, prediction))

print('That was a good Accuracy')
prediction_proba = model.predict_proba(X_valid)[::, 1]

fpr, tpr, _ = metrics.roc_curve(y_valid, prediction_proba)

auc = metrics.roc_auc_score(y_valid, prediction_proba)

plt.plot(fpr, tpr, label='data 1, auc='+str(auc))

plt.legend(loc=9)

plt.show()
# AUC score for the case is 0.82.

# AUC score 1 represents perfect classifier,

# and 0.5 represents a worthless classifier.
output = pd.DataFrame({'row_id':pd.DataFrame(prediction).index, 'Diabetes_or_Not':prediction})

output.to_csv('submission.csv', index=False)