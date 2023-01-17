import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('fivethirtyeight')

import itertools
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
diabetes = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
cols_to_norm = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction']
columns=diabetes.columns[:8]

plt.subplots(figsize=(18,15))

length=len(columns)

for i,j in itertools.zip_longest(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    diabetes[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x:(x - x.min())/(x.max()-x.min()))
diabetes
diabetes.columns
diabetes.isnull().any()
sns.countplot(x='Outcome',data=diabetes)

plt.show()
Pregnan=tf.feature_column.numeric_column('Pregnancies')

Gluco=tf.feature_column.numeric_column('Glucose')

BloodPre=tf.feature_column.numeric_column('BloodPressure')

SkinThi=tf.feature_column.numeric_column('SkinThickness')

Insu=tf.feature_column.numeric_column('Insulin')

BMI=tf.feature_column.numeric_column('BMI')

DiabPedi=tf.feature_column.numeric_column('DiabetesPedigreeFunction')

age=tf.feature_column.numeric_column('Age')
# another way for categorical data

#assigned_group= tf.feature_column.categorical_column_with_hash_bucket('group', hash_bucket_size=10)
diabetes['Age'].hist(bins=20)
age_bucket = tf.feature_column.bucketized_column(age,boundaries=[20,30,40,50,60,70,80])
feat_cols= [Pregnan,Gluco,BloodPre,SkinThi,Insu,BMI,DiabPedi,age_bucket]
age_bucket
labels = diabetes.pop('Outcome')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes,labels,test_size=.3,random_state=101)
X_train.head()
y_train.head()
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y= y_train, batch_size=10,num_epochs=1000, shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2) 
modelDNN =tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols)
model.train(input_fn= input_func,steps=1000)
modelDNN.train(input_fn= input_func,steps=1000)
eval_input_func=tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=10,num_epochs=1,shuffle=False)
results=model.evaluate(eval_input_func)
resultsDNN =modelDNN.evaluate(eval_input_func)
# Accuracy is much better by other models but i just wanted to learn tf.estimator

results

resultsDNN
### Now we can put data for predictions 
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=10,num_epochs=1,shuffle=False)
predictions = model.predict(pred_input_func)
predictions

list(predictions)