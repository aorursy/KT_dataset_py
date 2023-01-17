!pip install git+https://github.com/keras-team/keras-tuner.git@1.0.2rc2 -q
!pip install autokeras==1.0.9 -q
import pandas as pd

full_train_dataframe = pd.read_csv('../input/titanic/train.csv')
full_train_dataframe.head()
test_dataframe = pd.read_csv('../input/titanic/test.csv')
test_dataframe.head()
import autokeras as ak

# Initialize the classifier, which at most tries 10 different models.
clf = ak.StructuredDataClassifier(max_trials=10, overwrite=True)
# Search and train the model with training data.
# Pass the csv path as x and the target column name as y.
clf.fit('/kaggle/input/titanic/train.csv', 'Survived')
# Predict for the testing data.
predictions = clf.predict('/kaggle/input/titanic/test.csv')
print(predictions[:10])
import numpy as np

test_dataframe = pd.read_csv('../input/titanic/test.csv')
passenger_ids = test_dataframe.pop("PassengerId")
submission = pd.DataFrame({"PassengerId": passenger_ids,
                           "Survived": np.ravel(np.round(predictions))})
submission.to_csv("submission.csv", index=False)
keras_model = clf.export_model()
keras_model.summary()
keras_model.save('saved_model')
import tensorflow as tf
# Import AutoKeras before loading the model for custom layers.
import autokeras as ak

keras_model = tf.keras.models.load_model('saved_model')
print(keras_model.predict(pd.read_csv('../input/titanic/test.csv').to_numpy().astype(np.unicode))[:10])