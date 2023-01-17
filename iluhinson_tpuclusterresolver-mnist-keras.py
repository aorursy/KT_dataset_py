import tensorflow as tf



try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy()



print("REPLICAS:", strategy.num_replicas_in_sync)

print("Job name:", tpu.get_job_name())

print("Master:", tpu.get_master())

print("Number of accelerators:", tpu.num_accelerators())
!pip install tensorflow_datasets
import tensorflow_datasets as tfds



def get_dataset(batch_size=200):

  datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True,

                             try_gcs=True)

  mnist_train, mnist_test = datasets['train'], datasets['test']



  def scale(image, label):

    image = tf.cast(image, tf.float32) # TPU supports only tf.float32, tf.int32, tf.bfloat16, and tf.bool, uint8 is NOT supported

    image /= 255.0



    return image, label



  train_dataset = mnist_train.map(scale).shuffle(10000).batch(batch_size)

  test_dataset = mnist_test.map(scale).batch(batch_size)



  return train_dataset, test_dataset
def create_model():

  return tf.keras.Sequential(

      [tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),

       tf.keras.layers.Flatten(),

       tf.keras.layers.Dense(128, activation='relu'),

       tf.keras.layers.Dense(10)])
with strategy.scope():

  model = create_model()

  model.compile(optimizer='adam',

                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

                metrics=['sparse_categorical_accuracy'])

  model.summary()

  

train_dataset, test_dataset = get_dataset()



history = model.fit(train_dataset,

          epochs=5,

          validation_data=test_dataset)