from PIL import Image

import keras.backend as K

from keras.models import load_model

import numpy as np



model_path = "../input/boyudemo/g_model_256.h5"

image_size = 256
def load_image(fn, image_size):

    im = Image.open(fn).convert('RGB')

    

    #切割图像(截取图像中间的最大正方形，然后将大小调整至输入大小)

    if (im.size[0] >= im.size[1]):

        im = im.crop(((im.size[0] - im.size[1])//2, 0, (im.size[0] + im.size[1])//2, im.size[1]))

    else:

        im = im.crop((0, (im.size[1] - im.size[0])//2, im.size[0], (im.size[0] + im.size[1])//2))

    im = im.resize((image_size, image_size), Image.BILINEAR)

    

    #将0-255的RGB值转换到[-1,1]上的值

    arr = np.array(im)/255*2-1   

    

    return arr



def arr2image(X):

    int_X = ((X+1)/2*255).clip(0,255).astype('uint8')

    return Image.fromarray(int_X)



def generate(img, fn):

    r = fn([np.array([img])])[0]

    return arr2image(np.array(r[0]))
from IPython.display import display

K.clear_session()

model = load_model(model_path)

fn_generate = K.function([model.inputs[0]],[model.outputs[0]])



import os



for f in os.listdir("../input/boyudemo"):

    if (f[-4:] == ".jpg" or f[-4:] == ".png"):

        input_img = load_image("../input/boyudemo/"+f, image_size)

        output = generate(input_img, fn_generate)

        out = np.concatenate([np.array(input_img), np.array(output) / 255 * 2 - 1], axis = 1)

        display(arr2image(out))