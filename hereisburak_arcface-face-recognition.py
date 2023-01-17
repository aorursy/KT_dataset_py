import csv

import tensorflow as tf



from tqdm import tqdm
tf.__version__  # make sure you are using TensorFlow 2.x
model = tf.keras.models.load_model("../input/arcface-final-lresnet50ir/arcface_final.h5")
print(f"Input shape --> {model.input_shape}\nOutput Shape --> {model.output_shape}")



"""

As you can see, model take image(s) in 112x112 and with 3 channels(RGB, not BGR)

"""
# now i will define a data loader to read pins face recognition dataset, i will be receiving this code from Liyana



class DataEngineTypical:

    def make_label_map(self):

        self.label_map = {}



        for i, class_name in enumerate(tf.io.gfile.listdir(self.main_path)):

            self.label_map[class_name] = i



        self.reverse_label_map = {v: k for k, v in self.label_map.items()}



    def path_yielder(self):

        for class_name in tf.io.gfile.listdir(self.main_path):

            if not "tfrecords" in class_name:

                for path_only in tf.io.gfile.listdir(self.main_path + class_name):

                    yield (self.main_path + class_name + "/" + path_only, self.label_map[class_name])



    def image_loader(self, image):

        image = tf.io.read_file(image)

        image = tf.io.decode_jpeg(image, channels=3)

        image = tf.image.resize(image, (112, 112), method="nearest")

        image = tf.image.random_flip_left_right(image)



        return (tf.cast(image, tf.float32) - 127.5) / 128.



    def mapper(self, path, label):

        return (self.image_loader(path), label)



    def __init__(self, main_path: str, batch_size: int = 16, buffer_size: int = 10000, epochs: int = 1,

                 reshuffle_each_iteration: bool = False, test_batch=64,

                 map_to: bool = True):

        self.main_path = main_path.rstrip("/") + "/"

        self.make_label_map()



        self.dataset_test = None

        if test_batch > 0:

            reshuffle_each_iteration = False

            print(f"[*] reshuffle_each_iteration set to False to create a appropriate test set, this may cancelled if tf.data will fixed.")



        self.dataset = tf.data.Dataset.from_generator(self.path_yielder, (tf.string, tf.int64))

        if buffer_size > 0:

            self.dataset = self.dataset.shuffle(buffer_size, reshuffle_each_iteration=reshuffle_each_iteration, seed=42)



        if map_to:

            self.dataset = self.dataset.map(self.mapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        self.dataset = self.dataset.batch(batch_size, drop_remainder=True)



        if test_batch > 0:

            self.dataset_test = self.dataset.take(int(test_batch))

            self.dataset = self.dataset.skip(int(test_batch))



        self.dataset = self.dataset.repeat(epochs)

# now i will define an Engine Class to run jobs for getting outputs from images



# this class will take data and get 512-D features through ArcFace model

class Engine:

    @staticmethod

    def flip_batch(batch):

        return batch[:, :, ::-1, :]



    def __init__(self, data_engine: DataEngineTypical):

        self.data_engine = data_engine

        self.model = tf.keras.models.load_model("../input/arcface-final-lresnet50ir/arcface_final.h5")



        tf.io.gfile.mkdir("projector_tensorboard")



    def __call__(self, flip: bool = False):

        metadata_file = open('projector_tensorboard/metadata.tsv', 'w')

        metadata_file.write('Class\tName\n')

        with open("projector_tensorboard/feature_vecs.tsv", 'w') as fw:

            csv_writer = csv.writer(fw, delimiter='\t')



            for x, y in tqdm(self.data_engine.dataset):

                outputs = self.model(x, training=False)

                if flip:

                    outputs += self.model(self.flip_batch(x), training=False)



                csv_writer.writerows(outputs.numpy())

                for label in y.numpy():

                    name = self.data_engine.reverse_label_map[label]

                    metadata_file.write(f'{label}\t{name}\n')



        metadata_file.close()
# Now we have both data loader and an engine to process that data through arcface model and save those features to a tsv file.



TDOM = DataEngineTypical(

    "../input/pins-face-recognition/105_classes_pins_dataset/",

    batch_size=64,

    epochs=1,

    buffer_size=0,

    reshuffle_each_iteration=False,

    test_batch=0

)  # TDOM for "Tensorflow Dataset Object Manager"



e = Engine(

    data_engine=TDOM

)



e()