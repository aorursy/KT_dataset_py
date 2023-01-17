import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as blueprint #blueprint
from dask import bag #bag
from tqdm import tqdm
from PIL import Image, ImageDraw

def gauge_F(x):
    counts = np.bincount(x)
    p = counts[counts > 0] / float(len(x))
    # compute Shannon gauge in bits
    return -np.sum(p * np.log2(p))

def draw_F(strokes):
    image = Image.new("P", (256,256), color=255)
    draw = ImageDraw.Draw(image)
    for stroke in ast.literal_eval(strokes):
        for i in range(len(stroke[0])-1):
            draw.line([stroke[0][i], stroke[1][i], stroke[0][i+1], stroke[1][i+1]], fill=0, width=5)
    image = np.array(image)
    return gauge_F(image.flatten()), image

def plot_F(gauge, images, indices, n=5): #plot_F
    fig, axs = blueprint.subplots(nrows=n, ncols=n, figsize=(12, 10))
    for i, j in enumerate(indices[0][:n*n]):
        ax = axs[i // n, i % n]
        ax.set_title("%.4f" % gauge[j])
        ax.imshow(images[j], cmap="gray")
        ax.set_yticks([])
        ax.set_xticks([])
        blueprint.setp(ax.spines.values(), color="red")
    blueprint.subplots_adjust(bottom=-0.2)
    blueprint.show()

reader = pd.read_csv('../input/train_simplified/clock.csv', index_col=['key_id'], chunksize=1024)

data = []
for chunk in tqdm(reader):
    gaugebag = bag.from_sequence(chunk.drawing.values).map(draw_F) 
    data.extend(gaugebag.compute()) # PARALLELIZE

gauge, images = zip(*data)
threshold = 1
lower = np.percentile(gauge, threshold)
upper = np.percentile(gauge, 100 - threshold)
print(np.min(gauge), np.max(gauge))
print(lower, upper)
blueprint.title("Recognition messure")
blueprint.xlabel('gauge')
blueprint.ylabel('count')
blueprint.hist(gauge, bins=100)
blueprint.axvline(x=lower, color='r')
blueprint.axvline(x=upper, color='r')
plot_F(gauge, images, np.where(gauge < lower))
plot_F(gauge, images, np.where(gauge > upper))