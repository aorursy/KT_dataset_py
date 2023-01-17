from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import string
import os
import cv2
import numpy as np
def view(image):
    img=mpimg.imread(image)
    imgplot = plt.imshow(img)
    plt.show()
def id_generator(size=4, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
def generate(n):
    if not os.path.exists('data'):
        os.makedirs('data')
    image = ImageCaptcha()
    for _ in range(n):
        text=id_generator()
        data = image.generate(text)
        image.write(text,'data/'+text+'.png')        
generate(200)
import shutil
shutil.make_archive('images', 'zip','data')