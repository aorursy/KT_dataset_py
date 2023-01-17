from PIL import Image

import os



n = 1

for root, dirs, files in os.walk('/kaggle/input/'):

    for file in files:

        print('IMAGE', n)

        im = Image.open('/kaggle/input/' + file)

        print('  filename:', im.filename)

        print('  size:', im.size)

        print('  dpi:', im.info.get('dpi'))

        print('  color depth:', im.mode)

        print('  compression:', im.info.get('compression'))

        n += 1


