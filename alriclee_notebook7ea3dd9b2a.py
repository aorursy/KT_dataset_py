import matplotlib.pyplot as plt

import pandas as pd

import math

import os
name = "../input/holter-samples/RRholter/RRholter2"

output = "./"
# Read ECG csv file containing RR interval data



amp = pd.read_csv(name +".csv", sep=',')

amp = amp[:2000]

amp = amp.astype(int)



amp = amp.values.flatten()

amp = amp.tolist()

amp = [i for i in amp if i !=0]



amp_1 = amp[1:]

amp = amp[:-1]
# RR interval Lorenz plot



fig = plt.figure()

ax=fig.add_axes([0,0,1,1])

ax.set_aspect(1)

ax.scatter(amp, amp_1, color='r', s=3)

plt.xlim(0,1500)

plt.ylim(0,1500)

ax.set_xlabel('RRI n (ms)')

ax.set_ylabel('RRI n+1 (ms)')

plt.title(name + 'RR Interval Lorenz Plot')

plt.grid(b=True, which='major', color='#666666', linestyle='solid')

plt.minorticks_on()

plt.grid(b=True, which='minor', color='#999999', linestyle='dotted', alpha=0.2)

#plt.savefig(output, dpi = 300, bbox_inches='tight')

plt.show()
# load Keras DL library



from keras.models import load_model

from keras.preprocessing import image

import matplotlib.pyplot as plt

import numpy as np



# Define Lorenz plot image loading function



def load_image(img_path, show=False):



    img = image.load_img(img_path, target_size=(150, 150))

    img_tensor = image.img_to_array(img)                    # (height, width, channels)

    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)

    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]



    if show:

        plt.imshow(img_tensor[0])                           

        plt.axis('off')

        plt.show()



    return img_tensor



# Load demo CNN model

model = load_model("../input/holter-samples/model_12.h5")



# Define image path

img_path = '../input/holter-samples/RRholter/RRholter2.png'  



# load a single image

new_image = load_image(img_path)
# Check prediction

pred = model.predict(new_image)



if pred[0,0] == 0:

    result = "NEGATIVE"

else:

    result = "POSITIVE"

    

print ("The AF classifcation label is "+(result))