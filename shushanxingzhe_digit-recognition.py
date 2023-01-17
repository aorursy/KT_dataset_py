# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

labels = train_data["label"]
features = train_data[train_data.columns[1:]].values/255
test = test_data.values/255
train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"pixels": features},
        y=labels,
        batch_size=100,
        num_epochs=3,
        shuffle=True)
feature_columns = [tf.feature_column.numeric_column("pixels", shape=784)]
classifier = tf.estimator.LinearClassifier(
                feature_columns=feature_columns,
                n_classes=10
                )
classifier.train(input_fn=train_input_fn)

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'pixels': test},
        batch_size=1,
        num_epochs=1,
        shuffle=False)
predictions = classifier.predict(input_fn=predict_input_fn)
predicted_classes = [int(prediction['classes'][0]) for prediction in predictions]
index = list(range(1,28001))
pd.DataFrame({"ImageId":index,"Label":predicted_classes}).to_csv("sample_submission.csv",index=False)
import matplotlib.pyplot as plt
%matplotlib inline

for i in range(1,10):
    img = np.reshape(test[i],(28,28))
    plt.figure()
    plt.title("labeled class {}".format(predicted_classes[i]))
    plt.imshow(img, 'gray')