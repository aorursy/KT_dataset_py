#Loading Libraries

import pandas as pd

import numpy as np



#ML Algorithm Libraries

from sklearn.tree import DecisionTreeClassifier  #classifier algorithm

from sklearn.model_selection import train_test_split #dataset splitting function

from sklearn import metrics
#Loading Dataset

data = pd.read_csv('../input/cancer-data-set/CancerData.csv',header='infer')
data.shape
#Checking for null/missing values

data.isna().sum()
data.head()
data.columns
#Dropping the id column

data = data.drop(columns='id',axis=1)
data.head()
#Randomly selecting 10 records for testing 

test_df = data.sample(n=10)
features = ['radius_mean', 'texture_mean', 'perimeter_mean',

       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst']

target = ['diagnosis']



X = data[features]

y = data[target]
#Splitting the dataset (train & test)

size = 0.1  #10% of the dataset will used of testing the model & metrics calculation

state = 1



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=size, random_state=state)
dtc = DecisionTreeClassifier(max_depth=2)
dtc.fit(X_train,y_train)
#Getting the Accuracy Score of the Trained Model

print('Accuracy of the model: ','{:.2%}'.format(dtc.score(X_train,y_train)) )
#Prediction on the test data

y_pred = dtc.predict(X_test)
print("Accuracy Score: ",'{:.2%}'.format(metrics.accuracy_score(y_test, y_pred)))
# Confusion Matrix Heatmap

from matplotlib import pyplot as plt

import seaborn as sns



class_names=[0,1] 

confMatrix = metrics.confusion_matrix(y_test,y_pred)

fig, ax = plt.subplots(figsize=(8,8))

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(confMatrix), annot=True, cmap="Blues" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
test_df
# Making a prediction

testdata_pred = dtc.predict(test_df.iloc[:,1:31])

# Merging the prediction with the test-data

test_df['Predicted_Diagnosis'] = testdata_pred
test_df.head(10)
#Calculating the accuracy

print("Accuracy Score: ",'{:.2%}'.format(metrics.accuracy_score(test_df['diagnosis'], testdata_pred)))