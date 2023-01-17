import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv("../input/titanic/train.csv")

test_df = pd.read_csv("../input/titanic/test.csv")
train_df.head()
print("Train Data shape :: ",train_df.shape)

print("Test Data shape :: ",test_df.shape)
train_df.describe()
train_df.info()
test_df.info()
train_df.isna().sum()
test_df.isna().sum()
train_set = train_df.drop(columns=["Survived"])

test_set = test_df.copy(deep = True)

frame = [train_set,test_set]

final_set = pd.concat(frame,axis=0)
plt.figure(figsize = (30,10))

sns.heatmap(final_set.isnull(),yticklabels = False, cbar = True)
def fill_nan_categorical(categorical_features,data):

    for i in categorical_features:

        data[i] = data[i].fillna(data[i].mode()[0])

    

    return data
categorical_features = final_set.select_dtypes(exclude=["number","bool_"]).columns.tolist()

final_set = fill_nan_categorical(categorical_features,final_set)
plt.figure(figsize = (30,10))

sns.heatmap(final_set.isnull(),yticklabels = False, cbar = True)
def fill_nan_continuous(continuous_features,data):

    for i in categorical_features:

        data[i] =  data[i].fillna(data[i].mean()[0])

    

    return data
continuous_features = final_set.select_dtypes(include=["number","object_"]).columns.tolist()

final_set = fill_nan_categorical(continuous_features,final_set)
plt.figure(figsize = (30,10))

ax = plt.axes()

sns.heatmap(final_set.isnull(),yticklabels = False, cbar = True)

ax.set_title('Data Post all Nan values Removal',fontsize =20,color = "red");
def one_hot_encoder(final_set):

    df  = final_set.copy(deep= True)

    dummies = pd.get_dummies(df,prefix="column_",drop_first=True)

    return dummies
final_set = final_set.drop(columns = ["Name"])

ohe_set = one_hot_encoder(final_set)
train_data = pd.DataFrame(ohe_set[:891])

test_data = pd.DataFrame(ohe_set[891:])

X = train_data

y = train_df["Survived"]
train_data.tail()
test_data.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1, random_state = 42)
import xgboost



xgb_classifier = xgboost.XGBRFClassifier(n_estimators=30,

                                        learning_rate=1,

                                        reg_lambda=0.15,

                                        gamma=0.06,

                                        max_depth=20)

xgb_classifier.fit(X_train, y_train)

y_test_pred = xgb_classifier.predict(X_test).tolist()

accuracy_score(y_test_pred,y_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



# actual values

actual = y_test

# predicted values

predicted = y_test_pred



# confusion matrix

matrix = confusion_matrix(actual,predicted, labels=[1,0])

print('Confusion matrix : \n',matrix)



# outcome values order in sklearn

tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)

print('Outcome values : \n', tp, fn, fp, tn)



# classification report for precision, recall f1-score and accuracy

matrix = classification_report(actual,predicted,labels=[1,0])

print('Classification report : \n',matrix)
y_pred = xgb_classifier.predict(test_data).tolist()
predictions = pd.DataFrame(y_pred)

datasets = pd.concat([test_set["PassengerId"],predictions],axis=1)

datasets.columns = ["PassengerId","Survived"]

datasets.to_csv("gender_submission.csv",index=False)