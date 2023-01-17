import matplotlib.pyplot as plt

import os

import inspect

import numpy as np

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator



from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.models import Sequential,load_model,model_from_json

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import utils as kutils
model = load_model("../input/models-trained-on-docknet-data/MobXDock3.hd5")
docnet_train_path = "../input/open-sprayer-images/docknet/Docknet/train"

docnet_valid_path = "../input/open-sprayer-images/docknet/Docknet/valid"
docks = [docnet_train_path + "/docks/" + fn for fn in os.listdir(docnet_train_path + "/docks")]

docks += [docnet_valid_path + "/docks/" + fn for fn in os.listdir(docnet_valid_path + "/docks")]

not_docks = [docnet_train_path + "/notdocks/" + fn for fn in os.listdir(docnet_train_path + "/notdocks")]

not_docks += [docnet_valid_path + "/notdocks/" + fn for fn in os.listdir(docnet_valid_path + "/notdocks")]
dock_df = pd.DataFrame()

dock_df['image_path'] = docks + not_docks

dock_df['weed'] = ['no' if nd else 'yes' for nd in dock_df['image_path'].str.contains('notdocks')]
not_docks = dock_df[dock_df['weed'].str.contains('no')]

docks = dock_df[dock_df['weed'].str.contains('yes')]

print(len(not_docks), len(docks))
fit_these = not_docks.sample(len(docks))

fit_these = fit_these.append(docks)

len(fit_these)
datagen=ImageDataGenerator(

    rescale=1./255.,

    width_shift_range = 0.1,

    height_shift_range = 0.1,

    rotation_range = 5,

    shear_range = 5,

    zoom_range = (0.90,1.10),

    fill_mode = "constant",

    cval = 0,

    validation_split = 0.0

    )
for image_path in dock_df.sample(8).image_path:

    ima = plt.imread(image_path)

    txfm = datagen.get_random_transform(np.shape(ima))

    imt = datagen.apply_transform(ima,txfm)

    plt.subplot(121)

    plt.imshow(ima)

    plt.subplot(122)

    plt.imshow(imt)

    plt.show()

    
train_generator = datagen.flow_from_dataframe(

    fit_these,

    x_col = 'image_path',

    y_col = 'weed',

    target_size = (256,256),

    color_mode="rgb")
cd = train_generator.class_indices

ivd = {v: k for k, v in cd.items()}

cddf = pd.DataFrame.from_dict(cd,orient='index')

cd
history = model.fit_generator(train_generator, epochs=1,

    use_multiprocessing = False,

    verbose=1,shuffle=True

    )
#plt.plot(history.history['accuracy'])

#plt.show()
#model.save("MobXDockDemo.hd5")