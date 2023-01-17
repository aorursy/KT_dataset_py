import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas.api.types as ptypes

import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
df_wisconsin = pd.read_csv('../input/data.csv')

df_wisconsin.columns = df_wisconsin.columns.str.replace('\s+', '_')  # Replacing column names by _ whereever space id found

len(df_wisconsin.columns)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

df_wisconsin_numeric = df_wisconsin.select_dtypes(include=numerics) # exclude is another keyword.

# Looking at the columns of the dataset

len(df_wisconsin_numeric.columns)
normalize_columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',

       'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave_points_worst',

       'symmetry_worst', 'fractal_dimension_worst']
x_values = df_wisconsin.drop(['diagnosis','id','Unnamed:_32'],axis=1) # Getting Predictors

y_val = df_wisconsin['diagnosis'] # getting response
x_values.head() # Examining x_values
y_val.value_counts() # Checking if the classes are balanced. Seems pretty good.
# Converting Labels to integer form. 'B' and 'M; are represented as 0,1. Since Tensorflow does not accept categorical variables in text form, we are converting to integers.

def label_numeric(label):

    if (label == 'B'):

        return(0)

    else:

        return(1)
y_val_numeric =y_val.apply(label_numeric)

y_val_numeric.value_counts()
X_train, X_test, y_train, y_test = train_test_split(x_values,y_val_numeric,test_size=0.1,random_state=1234)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = pd.DataFrame(data=scaler.transform(X_train),columns = X_train.columns,index=X_train.index)

#X_train.head()

# Have a look at the before and after scaling step to understand what transformation has been done to the data.
# Similarly, scaling is performed on test data as well.

X_test = pd.DataFrame(data=scaler.transform(X_test),columns = X_test.columns,index=X_test.index)

X_test.head()
# Now we have 30 features which will be built in tensorflow framewor. Tensorflow requires these features to be defined as feature columns.

fc_radius_mean = tf.feature_column.numeric_column('radius_mean')

fc_texture_mean = tf.feature_column.numeric_column('texture_mean')

fc_perimeter_mean = tf.feature_column.numeric_column('perimeter_mean')

fc_area_mean = tf.feature_column.numeric_column('area_mean')

fc_smoothness_mean = tf.feature_column.numeric_column('smoothness_mean')

fc_compactness_mean = tf.feature_column.numeric_column('compactness_mean')

fc_concavity_mean = tf.feature_column.numeric_column('concavity_mean')

fc_concave_points_mean = tf.feature_column.numeric_column('concave points_mean')

fc_symmetry_mean = tf.feature_column.numeric_column('symmetry_mean')

fc_fractal_dimension_mean = tf.feature_column.numeric_column('fractal_dimension_mean')

fc_radius_se = tf.feature_column.numeric_column('radius_se')

fc_texture_se = tf.feature_column.numeric_column('texture_se')

fc_perimeter_se = tf.feature_column.numeric_column('perimeter_se')

fc_area_se = tf.feature_column.numeric_column('area_se')

fc_smoothness_se = tf.feature_column.numeric_column('smoothness_se')

fc_compactness_se = tf.feature_column.numeric_column('compactness_se')

fc_concavity_se = tf.feature_column.numeric_column('concavity_se')

fc_concave_points_se = tf.feature_column.numeric_column('concave points_se')

fc_symmetry_se = tf.feature_column.numeric_column('symmetry_se')

fc_fractal_dimension_se = tf.feature_column.numeric_column('fractal_dimension_se')

fc_radius_worst = tf.feature_column.numeric_column('radius_worst')

fc_texture_worst = tf.feature_column.numeric_column('texture_worst')

fc_perimeter_worst = tf.feature_column.numeric_column('perimeter_worst')

fc_area_worst = tf.feature_column.numeric_column('area_worst')

fc_smoothness_worst = tf.feature_column.numeric_column('smoothness_worst')

fc_compactness_worst = tf.feature_column.numeric_column('compactness_worst')

fc_concavity_worst = tf.feature_column.numeric_column('concavity_worst')

fc_concave_points_worst = tf.feature_column.numeric_column('concave points_worst')

fc_symmetry_worst = tf.feature_column.numeric_column('symmetry_worst')

fc_fractal_dimension_worst = tf.feature_column.numeric_column('fractal_dimension_worst')

#feat_cols = [fc_radius_mean, ..., fc_concave_points_worst, fc_symmetry_worst]

# Efficient way of building feature columns. 

# Please notice that Categorical Columns and Numerical Columns are treated differently.

feat_cols = []

df = X_train

for col in df.columns:

  if ptypes.is_string_dtype(df[col]): #is_string_dtype is pandas function

    feat_cols.append(tf.feature_column.categorical_column_with_hash_bucket(col, 

        hash_bucket_size= len(df[col].unique())))



  elif ptypes.is_numeric_dtype(df[col]): #is_numeric_dtype is pandas function

    feat_cols.append(tf.feature_column.numeric_column(col))
print(feat_cols)
classifier = tf.estimator.DNNClassifier(

        feature_columns=feat_cols,

        # Two hidden layers of 20 nodes each.

        hidden_units=[20, 20],

        # The model must choose between 2 classes.

        n_classes=2)
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train ,batch_size=10,num_epochs=500, shuffle=True)
classifier.train(input_fn=input_func, steps=10000)
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False) # Shuffle should be set as false as we are interested in comparing with actual results.
# Prediction is done here now.

predictions = list(classifier.predict(input_fn=pred_fn))

predictions[0]
final_preds = []

for pred in predictions:

    #info = "{} {} {}".format(pred['class_ids'][0], pred['probabilities'][0] , pred['probabilities'][1])

    final_preds.append(pred['class_ids'][0])

    #final_preds.append(info)
print(classification_report(y_test,final_preds))
print(confusion_matrix(y_test,final_preds))
from sklearn.metrics import accuracy_score

accuracy_score(y_test,final_preds, normalize=True, sample_weight=None)