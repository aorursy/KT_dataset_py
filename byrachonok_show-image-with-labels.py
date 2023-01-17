import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
def draw_boxes(image, boxes):
    fig, ax = plt.subplots(1, figsize=(7,7))
    
    colors = {i: np.random.rand(3,) for i in range(len(boxes))}
    for i, box in enumerate(boxes):
        y, x, h, w = box[0], box[1], box[2], box[3]
        ymin, xmin, ymax, xmax = x, y, x+w, y+h
        p = Polygon(((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)),
            fc=(colors[i][0],colors[i][1],colors[i][2],0.35), 
            ec=(colors[i][0],colors[i][1],colors[i][2],0.95), lw=3)
        ax.add_patch(p)
    ax.imshow(image, cmap=plt.cm.gist_gray)
train_info_path = '../input/stage_1_train_labels.csv'
data = pd.read_csv(train_info_path)

# get labels
idx = '00c0b293-48e7-4e16-ac76-9269ba535a62'
boxes = data[data['patientId'] == idx][['x', 'y', 'width', 'height']].values.tolist()

# get image
dcm_data = pydicom.read_file('../input/stage_1_train_images/'+idx+'.dcm')
image = dcm_data.pixel_array

# draw image with labels
draw_boxes(image, boxes)
