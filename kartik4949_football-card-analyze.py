import cv2
import numpy as np
from PIL import Image, ImageEnhance 
im = Image.open("../input/reddata/Red (1).jpg")
enhancer = ImageEnhance.Contrast(im)
enhanced_im = enhancer.enhance(4.0)
enhanced_im.save("enhanced.sample1.png")
im = Image.open("enhanced.sample1.png")
im.show()
enhanced_im.show()
im.point(enhanced_im)

import os
import numpy as np
path = '../input/cardsdata/'
label = []
xdata = []
for file,fld,fileList in os.walk(path):
    print(file)
    for fname in fileList:
        if('Red' in file):
            label.append(1)
        elif('Yellow' in file):
            label.append(0)
            
        xdata.append(fname)
            
label=np.asarray(label)  
xdata=np.asarray(xdata)
finaldata=np.column_stack((xdata,label))
import pandas as pd
df = pd.DataFrame({"id" : xdata, "label" : label})
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(featurewise_center=True,
                                  rescale=1./255,
                                  rotation_range = 20,
                                  width_shift_range = 0.1,
                                  height_shift_range = 0.1,
                                  zoom_range = 0.2,
                                  shear_range = 0.1,
                                  horizontal_flip = True,
                                  fill_mode = 'constant')
train_generator = train_datagen.flow_from_dataframe(
    dataframe = df,
    directory = "../input/cardsdata/cards/Cards/",
    x_col = "id",
    y_col = "label",
    has_ext = True,
    classes = 2,
    class_mode = "categorical",
    target_size = (160, 160),
    batch_size = 10
)
