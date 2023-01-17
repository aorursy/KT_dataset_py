%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *
path = untar_data(URLs.BIWI_HEAD_POSE)
cal = np.genfromtxt(path/'01'/'rgb.cal', skip_footer=6); cal
fname = '09/frame_00777_rgb.jpg'
def img2txt_name(f): return path/f'{str(f)[:-7]}pose.txt'
img = open_image(path/fname)

img.show()