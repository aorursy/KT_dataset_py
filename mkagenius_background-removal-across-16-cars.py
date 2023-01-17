import cv2

from PIL import Image

import numpy as np

import matplotlib.pylab as plt



def get_image_data(image_pil, **kwargs):

    

    img = _get_image_data_pil(image_pil, **kwargs)

    return img



def _get_image_data_pil(img_pil, return_exif_md=False, return_shape_only=False):

    

    if return_shape_only:

        return img_pil.size[::-1] + (len(img_pil.getbands()),)



    img = np.asarray(img_pil)

    assert isinstance(img, np.ndarray), "Open image is not an ndarray. Image id/type : %s, %s" % (image_id, image_type)

    if not return_exif_md:

        return img

    else:

        return img, img_pil._getexif()
from PIL import Image

s = '02159e548029'

imgs = []



for i in range(1,17):

    s2 = '%02d' % i

    fname = s + '_' + s2 + '.jpg'

    imgs.append(list(Image.open('../input/train/'+fname).getdata()))





 

def closeby(a, b):

    pc = 20 * a[0] / 100.0

    if abs(a[0]-b[0]) > pc or abs(a[1]-b[1]) > pc or abs(a[2]-b[2]) > pc:

        return False

    return True

    



for i in range(1918*1280):

    flag = True

    x = imgs[0][i]

    if(i % 200000 == 0):

        print(i)

    for k in range(0, 16):

        if not closeby(x, imgs[k][i]):

            flag = False

    if flag:

        for k in range(16):

            imgs[k][i] = (0,0,0)

 

 

for j in range(16):

    im = Image.new('RGB', (1918, 1280))

    im.putdata(imgs[j])

    img_data = get_image_data(im)

    plt.figure(figsize=(20, 20))

    plt.imshow(img_data)

    #im.save('removeback_'+str('%02d' % j) + ".jpg")