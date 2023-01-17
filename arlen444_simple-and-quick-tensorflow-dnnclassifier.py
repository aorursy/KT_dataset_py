import numpy as np

import pandas as pd
X_train = pd.read_csv("../input/X_train.csv")

y_train = pd.read_csv("../input/y_train.csv")

sub = pd.read_csv("../input/sample_submission.csv")
def flattenDataframe(df):

    '''

    'Flatten a dataframe

    '''

    df_new = pd.DataFrame([])

    for col in df.columns[3:]:

        df_new[col + '_mean'] = df.groupby(['series_id'])[col].mean()

        df_new[col + '_std'] = df.groupby(['series_id'])[col].std()

        df_new[col + '_var'] = df.groupby(['series_id'])[col].var()

        df_new[col + '_sem'] = df.groupby(['series_id'])[col].sem()

        df_new[col + '_max'] = df.groupby(['series_id'])[col].max()

        df_new[col + '_min'] = df.groupby(['series_id'])[col].min()

        df_new[col + '_max_to_min'] = df_new[col + '_max'] / df_new[col + '_min']

        df_new[col + '_max_minus_min'] = df_new[col + '_max'] - df_new[col + '_min']

        df_new[col + '_std_to_var'] = df_new[col + '_std'] * df_new[col + '_var']

        df_new[col + '_mean_abs_change'] = df.groupby('series_id')[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        df_new[col + '_abs_max'] = df.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))

    return df_new
X_train_flat = flattenDataframe(X_train)

X_train_flat.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train_flat)

scaled_features = scaler.fit_transform(X_train_flat)
X_train_new = pd.DataFrame(scaled_features,columns=X_train_flat.columns)

X_train_new.head()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()

y_train_new=labelencoder.fit_transform(y_train['surface'])

y_train_new=pd.Series(y_train_new)

print(labelencoder.classes_)
X_test = pd.read_csv("../input/X_test.csv")



X_test_flat= flattenDataframe(X_test)

scaler.fit(X_test_flat)

scaled_features_train = scaler.fit_transform(X_test_flat)

X_test_new = pd.DataFrame(scaled_features_train,columns=X_test_flat.columns)

X_test_new.head()
import tensorflow as tf
feat_cols = []

for key in X_train_new.keys():

    feat_cols.append(tf.feature_column.numeric_column(key=key))

feat_cols
steps=20000

n=9

Layers = [20,40,20]  

classifier = tf.estimator.DNNClassifier(

                                        hidden_units=Layers,

                                        n_classes=n,

                                        feature_columns=feat_cols)
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train_new,

                                                 y=y_train_new,

                                                batch_size=10,num_epochs=500,shuffle=True)
classifier.train(input_fn=input_func,steps=steps)
pred_fn = tf.estimator.inputs.pandas_input_fn(

      x=X_test_new,

      batch_size=10,

      num_epochs=1,

      shuffle=False)
note_predictions = list(classifier.predict(input_fn=pred_fn))
final_preds  = []

for pred in note_predictions:

    class_id=pred['class_ids'][0]

    final_preds.append(class_id)

    

set(final_preds)
df_pred= pd.DataFrame({'surface_code':final_preds})

df_pred.head()
df_pred['surface']=labelencoder.inverse_transform(df_pred['surface_code'])

sub['surface'] = df_pred['surface']

sub.to_csv("submission.csv",index=False)

sub.head()
#train_eval_result = classifier.evaluate(input_fn=input_func)

#print("Training set accuracy: {accuracy}".format(**train_eval_result))