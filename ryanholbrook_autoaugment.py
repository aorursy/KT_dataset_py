!pip install -qU tensorflow-datasets

!pip install -qU tf-models-official
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

from official.vision.image_classification import augment
ds, ds_info = tfds.load('tf_flowers',

                        split='train',

                        with_info=True,

                        data_dir='/kaggle/input/tensorflow-flowers')



fig = tfds.show_examples(ds_info=ds_info, ds=ds)
rand_augment = augment.RandAugment()



rows = 8

cols = 4

plt.figure(figsize=(15, (15 * rows) // cols))

for i, example in enumerate(ds.take(rows*cols)):

    image = rand_augment.distort(example['image'])

    plt.subplot(rows, cols, i+1)

    plt.axis('off')

    plt.imshow(image)

plt.tight_layout()

plt.show()
auto_augment = augment.AutoAugment(translate_const=1)



rows=8

cols=4

plt.figure(figsize=(15, (15 * rows) // cols))

for i, example in enumerate(ds.take(rows*cols)):

    image = auto_augment.distort(example['image'])

    plt.subplot(rows, cols, i+1)

    plt.axis('off')

    plt.imshow(image)

plt.tight_layout()

plt.show()