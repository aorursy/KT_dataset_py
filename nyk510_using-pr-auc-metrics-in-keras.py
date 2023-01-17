from keras.layers import Dense, Input

from keras.models import Model
def create_model(n_input, n_output):

    input_tensor = Input(shape=(n_input, ))

    output = Dense(1, activation='sigmoid')(input_tensor)

    return Model(input_tensor, outputs=output)
model = create_model(128, 1)
# first way. AUC class instance 



from tensorflow.keras.metrics import AUC

pr_metric = AUC(curve='PR', num_thresholds=1000) # The higher the threshold value, the more accurate it is calculated.





# second way. apply scikit learn average precision using tf.py_function 



import tensorflow as tf

from sklearn.metrics import average_precision_score



def sk_pr_auc(y_true, y_pred):

    return tf.py_function(average_precision_score, (y_true, y_pred), tf.float64)



model.compile(loss='binary_crossentropy', optimizer='adam', 

              metrics=[

                  pr_metric,

                  sk_pr_auc

              ])



from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=128, n_classes=2)



model.fit(X, y, validation_data=(X, y), epochs=10)
y_pred = model.predict(X)
from sklearn.metrics import average_precision_score
average_precision_score(y, y_pred)