import pickle

import numpy as np

import pandas as pd

from types import SimpleNamespace



import tensorflow as tf

import tensorflow_hub as hub

import transformers



from tqdm.notebook import tqdm, trange

import matplotlib.pyplot as plt
tokenizer_type = transformers.DistilBertTokenizer

nlp_model_type = transformers.TFDistilBertForSequenceClassification

pretrained_model = 'distilbert-base-uncased'



input_layer_name = 'input'



num_epochs = 4

max_sequence_length = 256

batch_size = 256 # too large batch sizes cause memory errors
with open('/kaggle/input/toxicity/toxicity_data.pkl', 'rb') as f:

    data = pickle.load(f)

train = data[data.Training_evaluation_split.eq('Training')].drop(columns=['Source', 'Training_evaluation_split'])

test = data[data.Training_evaluation_split.eq('Evaluation')].drop(columns=['Source', 'Training_evaluation_split'])
train
def spacify(data):

    result = data.copy()

    result['Text'] = result.Text.map(lambda t: t.replace('\n', ' '))

    return result



train = spacify(train)

test = spacify(test)



train
plt.figure(figsize=(16, 8))

plt.title('Score distribution')

plt.xlabel('Toxicity (fraction of reviewers which labelled text as toxic)')

plt.ylabel('Number of comments')

plt.hist(train.Toxicity, bins=64)

plt.show()
train.Text[train.Toxicity == 0.5].values[:4]
train.Text[train.Toxicity == 0.75].values[:4]
def binarize(data):

    result = pd.DataFrame()

    result = result.append(data[data.Toxicity < 0.25])

    result = result.append(data[data.Toxicity > 0.75])

    result['Class'] = (result.Toxicity > 0.5).astype(int)

    result.drop(columns=['Toxicity'])

    return result
train = binarize(train)

test = binarize(test)
train[train.Class == 0]
train[train.Class == 1]
def plot_class_distribution(data):

    class_counts = [sum(data.Class == cls) for cls in [0, 1]]



    plt.figure(figsize=(8, 8))

    plt.title('Class distribution')

    plt.pie(class_counts, labels=['Not hate speech', 'Hate speech'])

    plt.show()
plot_class_distribution(train)
def balance_data(data):

    class_counts = [sum(data.Class == cls) for cls in [0, 1]]

    resampled = [data[data.Class == cls].sample(min(class_counts)) for cls in [0, 1]]

    result = pd.DataFrame()

    for r in resampled:

        result = result.append(r)

    return result
plot_class_distribution(balance_data(train))
with open('/kaggle/working/test.txt', 'w') as test_file:

    for t in balance_data(test).Text:

        test_file.write(t + '\n')
def cut_to_divisible(data, batch_size):

    num_used = len(data) // batch_size

    return data[:num_used * batch_size]
tokenizer = tokenizer_type.from_pretrained(pretrained_model)
def tokenize_data(data):

    encoded = [tokenizer.encode_plus(t, max_length=max_sequence_length, pad_to_max_length=True) for t in tqdm(data.Text)]

    input_ids = [t['input_ids'] for t in encoded]

    return np.array(input_ids)
def preprocess_data(data, balance=True):

    if balance:

        data = balance_data(data)

    data = cut_to_divisible(data, batch_size)

    ids = tokenize_data(data)

    return SimpleNamespace(ids=ids, classes=data.Class.values)



train = preprocess_data(train)

test = preprocess_data(test)
plt.figure(figsize=(16, 8))

plt.title('Distribution of lengths')

plt.xlabel('Sequence length, without padding')

plt.ylabel('Number of sequences')

plt.hist([sum(x != 0) for x in train.ids], bins=64)

plt.show()
# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
with tpu_strategy.scope():

    nlp_model = nlp_model_type.from_pretrained(pretrained_model)

    input_layer = tf.keras.layers.Input(shape=(max_sequence_length,), dtype=tf.int32, name=input_layer_name)

    output_placeholder = nlp_model(input_layer)[0]

    hate_detector = tf.keras.Model(inputs=[input_layer], outputs=[output_placeholder])

    hate_detector.summary()
with tpu_strategy.scope():

    test_prediction = hate_detector(test.ids[:1])

    print(f'Test prediction: {test_prediction}')
class SpikingSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, max_rate=1e-4, warmup_epochs=2, slope=2):

    super().__init__()

    self.max_rate = max_rate

    self.warmup_steps = int(len(train.ids) / batch_size * warmup_epochs)

    self.slope = slope

    

  def __call__(self, step):

    warmup_fraction = step / self.warmup_steps + 1e-5

    up_rate = self.max_rate * warmup_fraction ** self.slope

    down_rate = self.max_rate * warmup_fraction ** -self.slope

    return tf.minimum(up_rate, down_rate)



schedule = SpikingSchedule()



plt.figure(figsize=(16, 8))

plt.title('Learning rate schedule')

plt.xlabel('Batch number')

plt.ylabel('Learning rate')

plt.yscale('log')

plt.plot([schedule(x) for x in range(int(len(train.ids) / batch_size * num_epochs))])

for epoch_id in range(1, num_epochs):

    plt.axvline(len(train.ids) / batch_size * epoch_id)

plt.show()
with tpu_strategy.scope():

    hate_detector.compile(optimizer=tf.keras.optimizers.Adam(schedule), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
with tpu_strategy.scope():

    hate_detector.fit(

        x=train.ids,

        y=train.classes,

        batch_size=batch_size,

        epochs=num_epochs

    )
with tpu_strategy.scope():

    hate_detector.evaluate(

        x=test.ids,

        y=test.classes,

        batch_size=batch_size

    )
pickle.dump(nlp_model.config, open('/kaggle/working/config.pkl', 'wb'))

hate_detector.save_weights('/kaggle/working/weights.h5')
loaded_config = pickle.load(open('/kaggle/working/config.pkl', 'rb'))

nlp_model = nlp_model_type(loaded_config)

input_layer = tf.keras.layers.Input(shape=(max_sequence_length,), dtype=tf.int32, name=input_layer_name)

output_placeholder = nlp_model(input_layer)[0]

loaded_model = tf.keras.Model(inputs=[input_layer], outputs=[output_placeholder])

loaded_model.load_weights('/kaggle/working/weights.h5')

loaded_model.summary()
test_prediction = loaded_model(test.ids[:1])

print(f'Test prediction: {test_prediction}')
loaded_model.compile(optimizer=tf.keras.optimizers.Adam(schedule), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
loaded_model.evaluate(

    x=test.ids,

    y=test.classes,

    batch_size=batch_size

)