import pandas as pd
import numpy as np
import tensorflow as tf
# Import data
df = pd.read_csv("train.csv")
df.head()
# No name, passenger ID or ticket column
df1 = df.drop(["PassengerId","Name","Ticket"],axis=1)
df1.head()
df1.isnull().any()
## I decided to remove cabin as there are too many different styles of cabin
df1 = df1.drop("Cabin",axis=1)
df1.head()
df1["Age"].fillna(df1["Age"].mean(),inplace=True)
df1.isnull().any()
df1["Embarked"].unique()
import random
embarked_cabin_list = ["S","C","Q"]
df1["Embarked"].fillna(random.choice(embarked_cabin_list),inplace=True)
df1.isnull().any()
df1.columns
# Creating continous feature columns
SibSp = tf.feature_column.numeric_column("SibSp")
Parch = tf.feature_column.numeric_column("Parch")
Fare = tf.feature_column.numeric_column("Fare")
Age = tf.feature_column.numeric_column("Age")
Pclass = tf.feature_column.numeric_column("Pclass")
Sex = tf.feature_column.categorical_column_with_hash_bucket("Sex", hash_bucket_size=1000)
Embarked = tf.feature_column.categorical_column_with_hash_bucket("Embarked", hash_bucket_size=1000)
feat_cols = [SibSp,Parch,Fare,Age,Pclass,Sex,Embarked]
print(feat_cols)
x_data = df1.drop("Survived",axis=1)
x_data.head()
# Create labels
labels = df1["Survived"]
print("Length of labels is {0} and length of variables is {1}".format(len(labels),len(x_data)))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data,labels,test_size=0.33, random_state=101)
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)
model.train(input_fn=input_func,steps=1000)
eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)
results = model.evaluate(eval_input_func)
results
pred_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)
predictions = model.predict(pred_input_func)
# Predictions is a generator! 
predictions = model.predict(pred_input_func)
list_pred = list(predictions)
len(list_pred)
final_preds = []
for pred in list_pred:
    final_preds.append(pred['class_ids'][0])
len(final_preds)
from sklearn.metrics import classification_report
print(classification_report(y_test,final_preds))
real_df = pd.read_csv("test.csv")
real_df.head()
predict_df = real_df.drop(["PassengerId","Name","Ticket","Cabin"],axis=1)
predict_df.head()
predict_df["Age"].fillna(predict_df["Age"].mean(),inplace=True)
predict_embarked_cabin_list = ["S","C","Q"]
predict_df["Embarked"].fillna(random.choice(embarked_cabin_list),inplace=True)
predict_df.isnull().any()
predict_df["Fare"].fillna(predict_df["Fare"].mean(),inplace=True)
predict_df.isnull().any()
real_pred_input_func = tf.estimator.inputs.pandas_input_fn(
      x=predict_df,
      batch_size=10,
      num_epochs=1,
      shuffle=False)
real_predictions = model.predict(real_pred_input_func)
real_predictions
real_list_pred = list(real_predictions)
len(real_list_pred)
real_preds = []
for pred in real_list_pred:
    real_preds.append(pred['class_ids'][0])
#convert list of predictions into series
series_predictions = pd.Series(real_preds)
submission_data["Survived"] = series_predictions.values
submission_df = pd.DataFrame({'PassengerId':real_df["PassengerId"],"Survived":series_predictions})
submission_df.to_csv("kaggle_submission.csv")