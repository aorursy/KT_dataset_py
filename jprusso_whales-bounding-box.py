import pandas as pd



def read_keypoints():

    keypoints = pd.read_csv("../input/humpback_fluke_keypoints/keypoints.csv", index_col=False)

    return keypoints
with open('../input/humpback_fluke_keypoints/keypoints.csv', 'rt') as f: data = f.read().split('\n')[1:-1]

len(data)
for line in data[:5]: print(line)
data = [line.split(',') for line in data]

data = [filter(None,line) for line in data]

data = [(p,[(int(float(coord[i])),int(float(coord[i+1]))) for i in range(0,len(coord)-1,2)]) for p,*coord in data]

data[0] # First row of the dataset
from PIL import Image as pil_image

from PIL.ImageDraw import Draw



def read_raw_image(p):

    return pil_image.open('../input/humpback_fluke_keypoints/' + p)



def draw_dot(draw, x, y):

    draw.ellipse(((x-5,y-5),(x+5,y+5)), fill='red', outline='red')



def draw_dots(draw, coordinates):

    for x,y in coordinates: draw_dot(draw, x, y)



filename,coordinates = data[0]

img = read_raw_image(filename)

draw = Draw(img)

draw_dots(draw, coordinates)

img
def bounding_rectangle(list):

    x0, y0 = list[0]

    x1, y1 = x0, y0

    for x,y in list[1:]:

        x0 = min(x0, x)

        y0 = min(y0, y)

        x1 = max(x1, x)

        y1 = max(y1, y)

    return x0,y0,x1,y1



box = bounding_rectangle(coordinates)

box
draw.rectangle(box, outline='red')

img