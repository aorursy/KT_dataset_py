# Pillow Method to GreyScale
from PIL import Image
img_path = '../input/Corn_Emoji.png'
img = Image.open(img_path).convert('L')
img.save('greyscale.png')
display(img)
# matplotlib method to GreyScale
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

np_img = mpimg.imread(img_path)     
gray = rgb2gray(np_img)    
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()
# Hybrid
pix = np.array(img)
plt.imshow(pix, cmap = plt.get_cmap('gray'))
plt.show()
print(type(pix))
# Matrix Representation
import pandas as pd

df = pd.DataFrame(pix)
with pd.option_context('display.max_rows', None, 'display.max_columns', 28):
    display(df)