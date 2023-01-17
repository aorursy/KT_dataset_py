from PIL import Image

import numpy as np



path_to_img = '../input/akensert-transform-panda-tiles/akensert_little/0005f7aaab2800f6170c399693a96917.jpeg'



imgs = np.array(Image.open(path_to_img)).reshape(-1, 256, 256, 3)



imgs.shape
Image.fromarray(imgs[0])