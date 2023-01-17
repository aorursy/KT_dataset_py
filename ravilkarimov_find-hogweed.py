import os

path = "../input/"

def list_files(startpath):

    for root, dirs, files in os.walk(startpath):

        level = root.replace(startpath, '').count(os.sep)

        indent = ' ' * 4 * (level)

        print('{}{}/'.format(indent, os.path.basename(root)))

        subindent = ' ' * 4 * (level + 1)

        for f in files:

            print('{}{}'.format(subindent, f))

list_files(path)
from IPython.display import Image

Image("../input/hogweednew/heracleum_lanantum_maxima_03.jpg")
Image("../input/hogweed-screenshot/screenshot_example.jpg")
Image("../input/hogweed-outscreenshot/screenshot_out_example.jpg")
import fiona

shape = fiona.open("../input/shapes/Borschevik.shp")

print(shape.schema)
#first feature of the shapefile

first = next(iter(shape))

print(first) # (GeoJSON format)
import shapefile as shp  # Requires the pyshp package

import matplotlib.pyplot as plt



sf = shp.Reader("../input/shapes/Borschevik.shp")



plt.figure()

for shape in sf.shapeRecords():

    x = [i[0] for i in shape.shape.points[:]]

    y = [i[1] for i in shape.shape.points[:]]

    plt.plot(x,y)

plt.show()