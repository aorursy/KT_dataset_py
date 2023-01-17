# Library

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, gc, warnings

warnings.filterwarnings("ignore")



# Plot

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# TensorFlow

import tensorflow as tf

tf.random.set_seed(42)



# Sklearn

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve



# Display

from IPython.display import clear_output
# Load Data

url = '../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv'

df = pd.read_csv(url, header='infer')

print("Total Records: ", df.shape[0])
# Check for empty / missing values

print("Is Dataset Empty?: ", df.empty)
num_cols = ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium', 'time']

cat_cols = ['anaemia', 'diabetes', 'high_blood_pressure','sex', 'smoking', 'DEATH_EVENT']



#Stat Summary

df[num_cols].describe().transpose()
#Garbage Collection

gc.collect()
# Map

cat_map = {"anaemia":  {0:"No", 1:"Yes"},

           "diabetes": {0:"No", 1:"Yes"},

           "high_blood_pressure": {0:"No", 1:"Yes"},

           "sex": {0:"Male", 1:"Female"},

           "smoking": {0:"No", 1:"Yes"},

           "DEATH_EVENT": {0:"No", 1:"Yes"}} 

          



# Creating a seperate dataframe

df_1 = df.replace(cat_map)



# Creating Age Group 

bins = [18, 30, 40, 50, 60, 70, 100]

labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']

df_1['age_grp'] = pd.cut(df_1.age, bins, labels = labels,include_lowest = True)



df_1.head()
plt.figure(figsize=(10,8))

df.age.hist(bins=20, histtype='bar',color='wheat')

plt.title("Age Distribution", fontsize=20)

plt.show()
plt.figure(figsize=(10,8))

flatui = ["#e74c3c", "#34495e"]

sns.countplot(x="age_grp", hue="sex", data=df_1, saturation=0.25, dodge=True, palette=sns.color_palette(flatui))

plt.title("Gender Distribution per Age Group", fontsize=20)

plt.show()
# Correlation Heatmap



plt.figure(figsize=(10,10))

plt.title ("Correlation Heatmap", fontsize=20)

corr = df_1.corr()



ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
plt.figure(figsize=(15,10))

sns.stripplot(x="age_grp", y="serum_creatinine", hue="sex",

              data=df_1, dodge=True, zorder=1, palette=sns.cubehelix_palette(2), jitter=True)

plt.title("Age vs Serum Creatinine", fontsize=20)
sns.catplot(x="ejection_fraction", y="serum_sodium", hue="sex", data=df_1, 

            height=8, kind="boxen", aspect=2.5, palette = sns.cubehelix_palette(8, start=.5, rot=-.75))

plt.title("Ejection Fraction vs Serum Sodium", fontsize=25)
# Garbage Collection

gc.collect()
''' Feature Engineering & Data Split '''



target = ['DEATH_EVENT']

features = df.columns[:-1]



X = df[features]

y = df[target]



#Training = 90% & Validation = 10%

test_size = 0.1

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True) 



print("Training Data Records: ", X_train.shape[0])

print("Validation Data Records: ", X_val.shape[0])



#Reset Index

X_val.reset_index(drop=True, inplace=True)

y_val.reset_index(drop=True, inplace=True)
'''TensorFlow Feature Columns Creation '''



cat_cols = ['anaemia','diabetes','high_blood_pressure','sex','smoking']   # Categorical Data Columns

float_cols = ['age','platelets','serum_creatinine']  # Num Float Data Columns

int_cols = ['creatinine_phosphokinase','ejection_fraction','serum_sodium', 'time']  # Num Int Data Columns





# One Hot Encoding Custom Function

def one_hot_encode(feature, vocab):

    return tf.feature_column.indicator_column(

        tf.feature_column.categorical_column_with_vocabulary_list(feature, vocab))





features_cols = []



# Categorical Features

for feature in cat_cols:

    vocabulary = X_train[feature].unique()

    features_cols.append(one_hot_encode(feature,vocabulary))



# Numerical Float Features    

for feature in float_cols:

    features_cols.append(tf.feature_column.numeric_column(feature, dtype=tf.float32))



# Numerical Int Features    

for feature in int_cols:

    features_cols.append(tf.feature_column.numeric_column(feature, dtype=tf.int32))

''' TensorFlow Input Function Creation  '''



num_examples = len(y_train)



def make_input_fn(X, y, n_epochs=None, shuffle=True):

       

    def input_fn():

        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))



        if shuffle:

            dataset = dataset.shuffle(num_examples)

            

        # For training, cycle thru dataset as many times as need (n_epochs=None).

        dataset = dataset.repeat(n_epochs)

        

        # In memory training doesn't use batching.

        dataset = dataset.batch(num_examples)

        

        return dataset

    

    return input_fn



# Training and evaluation input functions.

train_input_fn = make_input_fn(X_train, y_train)

val_input_fn = make_input_fn(X_val, y_val, shuffle=False, n_epochs=1)
''' Linear Classifier: Train & Evaluate Model'''

linear_clf = tf.estimator.LinearClassifier(features_cols)



# Train

linear_clf.train(train_input_fn, max_steps=100)



# Evaluation

result = linear_clf.evaluate(val_input_fn)

clear_output()

res = [(result[k]) for k in ['accuracy'] if k in result]

for acc in res:

    print("Linear Classifier Benchmark Accuracy: ",'{:.1%}'.format(acc))
#Garbage Collection

gc.collect()
'''Using entire dataset per layer'''

batches = 1



bt_clf = tf.estimator.BoostedTreesClassifier(features_cols, n_batches_per_layer=batches)



# Train

bt_clf.train(train_input_fn, max_steps=100)



# Evaluation

result = bt_clf.evaluate(val_input_fn)

clear_output()

res = [(result[k]) for k in ['accuracy'] if k in result]

for acc in res:

    print("Boosted Trees Classifier Accuracy: ",'{:.1%}'.format(acc))
# Predicted Probabilities

pred_dicts = list(bt_clf.predict(val_input_fn))

probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])



#ROC

fpr, tpr, _ = roc_curve(y_val, probs)





# Plot

fig=plt.figure(figsize=(20,15))



ax1=fig.add_subplot(221)

sns.distplot(probs, bins=15, ax=ax1,color='maroon')

ax1.set_title('Predicted Probabilities',size=15)



ax2=fig.add_subplot(222)

sns.lineplot(fpr, tpr, ax=ax2,color='darkolivegreen')

ax2.set_title('ROC Curve',size=15)

ax2.set_xlabel('false positive rate')

ax2.set_ylabel('true positive rate')
#Garbage Collection

gc.collect()
# Class Prediction

cls_id = pd.DataFrame([pred['class_ids'][0] for pred in pred_dicts])



# Validation dataframe

X_Val_df = X_val.copy()



# Adding Actual Death Event Column

X_Val_df['ACTUAL_DEATH_EVENT'] = y_val



# Adding Predicted Death Event Column

X_Val_df['PRED_DEATH_EVENT'] = cls_id



X_Val_df.head(10)