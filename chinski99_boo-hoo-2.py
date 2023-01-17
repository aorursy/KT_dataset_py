import pandas as pd

import tensorflow as tf

from tensorflow.contrib import learn
df = pd.read_csv("../input/train.csv", index_col='id')
features = ["bone_length","rotting_flesh","hair_length","has_soul"]

X = df[features]

y = df["type"]

X['hair_soul'] = X['hair_length'] * X['has_soul']

X['hair_bone'] = X['hair_length'] * X['bone_length']

X['hair_soul_bone'] = X['hair_length'] * X['has_soul'] * X['bone_length']

features+=['hair_soul','hair_bone','hair_soul_bone']



X.head()
from sklearn.preprocessing import LabelEncoder as LE

letype = LE()

y = letype.fit_transform(y)
from sklearn.model_selection import train_test_split

# current test size = 0 to permit the usage of whole training data

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.0)
x=tf.contrib.learn.infer_real_valued_columns_from_input(X_train)

tf_clf_dnn = learn.DNNClassifier(hidden_units=[16], n_classes=3, feature_columns=x, activation_fn=tf.sigmoid)

tf_clf_dnn.fit(X_train, y_train,max_steps=5000)



from sklearn.metrics import accuracy_score as acc_s



print(acc_s(y_train,tf_clf_dnn.predict(X_train)))


test_df = pd.read_csv("../input/test.csv",index_col='id')

features = ["bone_length","rotting_flesh","hair_length","has_soul"]

X = test_df[features]

X['hair_soul'] = X['hair_length'] * X['has_soul']

X['hair_bone'] = X['hair_length'] * X['bone_length']

X['hair_soul_bone'] = X['hair_length'] * X['has_soul'] * X['bone_length']

features+=['hair_soul','hair_bone','hair_soul_bone']

 

pred = tf_clf_dnn.predict(X)

pred = letype.inverse_transform(pred)



submission_df = pd.DataFrame({'type':pred}) 

submission_df['id'] = test_df.index 

submission_df = submission_df[['id', 'type']].set_index('id') 

submission_df.to_csv('tf_pred.csv')

submission_df.head()