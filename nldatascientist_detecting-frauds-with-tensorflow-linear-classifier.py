# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import seaborn as sns

import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/creditcard.csv")

df.describe()
fraud_indices = df[df.Class == 1].index

number_records_fraud = len(fraud_indices)



normal_indices = df[df.Class == 0].index

number_records_normal = len(normal_indices)



print("Normal transactions: ", number_records_normal)

print("Fraud transactions: ", number_records_fraud)
df.isnull().sum().plot(kind='bar')
y = df.Class

x = df.drop(['Class','Time'],axis=1)

x_scaled = (x - x.min()) / (x.max()-x.min()) 



# scaling is necessary to have the same range on the y-axis



chtdata = pd.concat([y,x_scaled.iloc[:,0:15]],axis=1)

chtdata = pd.melt(chtdata,id_vars="Class",var_name="Features",value_name='Scaled value')

plt.figure(figsize=(20,10))

sns.violinplot(x="Features", y="Scaled value", hue="Class", data=chtdata, split=True, inner="quart")

plt.xticks(rotation=90)
chtdata = pd.concat([y,x_scaled.iloc[:,15:]],axis=1)

chtdata = pd.melt(chtdata,id_vars="Class",var_name="Features",value_name='Scaled value')

plt.figure(figsize=(20,10))

sns.violinplot(x="Features", y="Scaled value", hue="Class", data=chtdata, split=True, inner="quart")

plt.xticks(rotation=90)
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn import metrics
nV01 = tf.feature_column.numeric_column('V1')

nV02 = tf.feature_column.numeric_column('V2')

nV03 = tf.feature_column.numeric_column('V3')

nV04 = tf.feature_column.numeric_column('V4')

nV05 = tf.feature_column.numeric_column('V5')

nV06 = tf.feature_column.numeric_column('V6')

nV07 = tf.feature_column.numeric_column('V7')

nV08 = tf.feature_column.numeric_column('V8')

nV09 = tf.feature_column.numeric_column('V9')

nV10 = tf.feature_column.numeric_column('V10')

nV11 = tf.feature_column.numeric_column('V11')

nV12 = tf.feature_column.numeric_column('V12')

nV13 = tf.feature_column.numeric_column('V13')

nV14 = tf.feature_column.numeric_column('V14')

nV15 = tf.feature_column.numeric_column('V15')

nV16 = tf.feature_column.numeric_column('V16')

nV17 = tf.feature_column.numeric_column('V17')

nV18 = tf.feature_column.numeric_column('V18')

nV19 = tf.feature_column.numeric_column('V19')

nV20 = tf.feature_column.numeric_column('V20')

nV21 = tf.feature_column.numeric_column('V21')

nV22 = tf.feature_column.numeric_column('V22')

nV23 = tf.feature_column.numeric_column('V23')

nV24 = tf.feature_column.numeric_column('V24')

nV25 = tf.feature_column.numeric_column('V25')

nV26 = tf.feature_column.numeric_column('V26')

nV27 = tf.feature_column.numeric_column('V27')

nV28 = tf.feature_column.numeric_column('V28')

nV30 = tf.feature_column.numeric_column('Amount')



features = [nV01 , nV02 , nV03 , nV04 , nV05 , nV06 , nV07 , nV08 , nV09 , nV10 , 

             nV11 , nV12 ,

            # nV13 , 

            nV14 ,

            # nV15 , 

            nV16 , nV17 , nV18 , nV19 , 

            # nV20 , 

             nV21 , nV22 , nV23 , 

#             nV24 , nV25 , nV26 , 

             nV27 , nV28 , nV30]
trainsize = 0.7



X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, train_size=trainsize, random_state=101)
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=100, num_epochs=1000,shuffle=True)

 

model = tf.estimator.LinearClassifier(feature_columns=features,n_classes=2)

 

model.train(input_fn=input_func,steps=1000)

 

#resultaten trainingset

results=model.evaluate(tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10, num_epochs=1, shuffle=False))
print(results)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=10, num_epochs=1, shuffle=False)

results=model.evaluate(eval_input_func)

print(results)
pred_input_func= tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)

predictions = model.predict(pred_input_func)



y_pred= [d['logits'] for d in predictions]



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(10,10))

plt.title('ROC - Tensorflow')

plt.plot(fpr, tpr, 'b',label='Area under curve = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()