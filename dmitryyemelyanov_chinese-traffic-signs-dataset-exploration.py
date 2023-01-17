import pandas as pd

import os



import IPython.display as display

from PIL import Image

from PIL.ImageDraw import Draw
data_dir = '/kaggle/input/chinese-traffic-signs'

df = pd.read_csv(os.path.join(data_dir, 'annotations.csv'))

df.head(5)
for index, row in df.sample(frac=1)[:5].iterrows():

    image = Image.open(os.path.join(data_dir, 'images', row.file_name))

    draw = Draw(image)

    draw.rectangle([row.x1, row.y1, row.x2, row.y2], outline='#00FF00', width=3)

    display.display(image)

    print('category:', row.category)