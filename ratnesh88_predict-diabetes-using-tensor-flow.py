import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir('/kaggle/input'))
data = pd.read_csv('../input/diabetes.csv')
data.head()
data.describe()
data.head()
#import plotly.offline as py
#py.init_notebook_mode(connected=True)
#import plotly.graph_objs as go
#import plotly.tools as tls
#import cufflinks as cf
#cf.set_config_file(offline=True, world_readable=True, theme='ggplot')

data['Outcome'].value_counts().plot(kind='bar')
data.hist(figsize=(12,12));
X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']] 
Y = data['Outcome']

print(X.columns)
data.values.astype(float)[4]
def parse_csv(line):
    example_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0]]  # sets field types
    parsed_line = tf.decode_csv(line, example_defaults)
    #print(parsed_line)
    
    # First 8 fields are features, combine into single tensor
    features = tf.reshape(parsed_line[:-1], shape=(8,))
    # Last field is the label
    label = tf.reshape(parsed_line[-1], shape=())
    return features, label
print('parsing...')
train_dataset = tf.data.TextLineDataset('/kaggle/input/diabetes.csv')
train_dataset = train_dataset.skip(1)             # skip the first header row
train_dataset = train_dataset.map(parse_csv)      # parse each row
train_dataset = train_dataset.shuffle(buffer_size=200)  # randomize
train_dataset = train_dataset.batch(140)

# View a single example entry from a batch
#iterator = train_dataset.make_one_shot_iterator()
#features, label = iterator.get_next()

features, label = tfe.Iterator(train_dataset).next()
print("example features:", features[31])
print("example label:", label[31])
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation="relu", input_shape=(8,)),  # input shape required
  tf.keras.layers.Dense(15, activation="relu"),
  tf.keras.layers.Dense(2)
])

print('Model creation')
def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

def grad(model, inputs, targets):
    with tfe.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

      # Training loop - using batches
    for x, y in tfe.Iterator(train_dataset):
        # Optimize the model
        grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=tf.train.get_or_create_global_step())

        # Track progress
        epoch_loss_avg(loss(model, x, y))  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

        # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
    
    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)

plt.show()
class_ids = ["No", "Yes"]

predict_dataset = tf.convert_to_tensor([
    [6.0   , 148.0   ,  72.0   ,  35.0   ,   0.0   ,  33.6  ,   0.627,    50.0   ,],
    [61.0   , 18.0   ,  10.0   ,  65.0   ,   0.0   ,  12.6  ,   0.927,    55.0   ,],
    [40.   , 110.0  ,  92.0   ,   0.0  ,   0.0  ,  37.6  ,   0.191, 30.0    ]
    
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    name = class_ids[class_idx]
    print("Example {} prediction: {}".format(i, name))