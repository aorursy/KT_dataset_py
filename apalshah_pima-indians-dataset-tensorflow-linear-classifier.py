import pandas as pd

diabetes = pd.read_csv('../input/diabetes.csv')
diabetes.head()
import matplotlib.pyplot as plt
%matplotlib inline
diabetes['Age'].hist(bins=20)
cols_to_norm = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction']
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x : (x - x.min()) / (x.max() - x.min()))
diabetes.head()
import tensorflow as tf
pregnancies          = tf.feature_column.numeric_column('Pregnancies')
glucose              = tf.feature_column.numeric_column('Glucose')
blood_pressure       = tf.feature_column.numeric_column('BloodPressure')
skin_thickness       = tf.feature_column.numeric_column('SkinThickness')
insulin              = tf.feature_column.numeric_column('Insulin')
bmi                  = tf.feature_column.numeric_column('BMI')
diabetes_pedigree_fn = tf.feature_column.numeric_column('DiabetesPedigreeFunction')
age                  = tf.feature_column.numeric_column('Age')
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Outcome', ['0', '1'])
# assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Outcome', hash_bucket_size=2)
age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80])
feat_cols = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_fn, age_bucket]
x_data = diabetes.drop('Outcome', axis=1)

x_data.head()
labels = diabetes['Outcome']

labels.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.3, random_state=101)
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)
model.train(input_fn=input_func, steps=1000)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
results = model.evaluate(eval_input_func)
results
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)
predictions = model.predict(pred_input_func)

predictions = list(predictions)
predictions[:5]
y_test.head()
print('Comparing First Element:', 'Predictions:' ,predictions[0]['class_ids'][0], '&& y_test: ',y_test[766])
print('Probability', predictions[0]['probabilities'].max())
print('Comparing Third Element:', 'Predictions:' ,predictions[2]['class_ids'][0], '&& y_test: ',y_test[42])
print('Probability', predictions[2]['probabilities'].max())
