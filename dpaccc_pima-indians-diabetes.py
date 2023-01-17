import pandas as pd
Diabetes = pd.read_csv("../input/diabetes.csv")
Diabetes.head()
Diabetes.columns
colums_to_norm = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
       'Insulin', 'BMI', 'DiabetesPedigreeFunction']
Diabetes[colums_to_norm] = Diabetes[colums_to_norm].apply(lambda x : (x - x.min()) / (x.max() - x.min()))
Diabetes.head()
import tensorflow as tf
num_preg = tf.feature_column.numeric_column('Pregnancies')
plasma_gluc = tf.feature_column.numeric_column('Glucose')
blood_press = tf.feature_column.numeric_column('BloodPressure')
tricep = tf.feature_column.numeric_column('SkinThickness')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('DiabetesPedigreeFunction')
age = tf.feature_column.numeric_column('Age')
import matplotlib.pyplot as plt
%matplotlib inline
Diabetes['Age'].hist(bins = 20)
age_bucket = tf.feature_column.bucketized_column(age, boundaries = [20, 30, 40, 50, 60, 70, 80])
feat_cols = [num_preg, plasma_gluc, blood_press, tricep, insulin, bmi, diabetes_pedigree, age_bucket]
from sklearn.model_selection import train_test_split
x_data = Diabetes.drop('Outcome', axis = 1)
x_data.head()
y_data = Diabetes['Outcome']
y_data.head()
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3)
input_func = tf.estimator.inputs.pandas_input_fn(x = X_train, y = y_train, batch_size = 16, num_epochs = 1000, shuffle = True)
model = tf.estimator.LinearClassifier(feature_columns = feat_cols, n_classes = 2)
model.train(input_fn = input_func, steps = 10000)
test_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test, y = y_test, batch_size = 16, num_epochs = 1, shuffle = False)
results = model.evaluate(test_input_func)
results
pred_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test, batch_size = 16, num_epochs = 1, shuffle = False)
predictions = model.predict(pred_input_func)
my_pred = list(predictions)
my_pred
input_func = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size = 16, num_epochs = 1000, shuffle = True)
dnn_model = tf.estimator.DNNClassifier(hidden_units = [10, 8, 4], feature_columns = feat_cols, n_classes = 2)
dnn_model.train(input_fn = input_func, steps = 10000)
test_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test, y = y_test, batch_size = 16, num_epochs = 1, shuffle = False)
dnn_model.evaluate(test_input_func)
