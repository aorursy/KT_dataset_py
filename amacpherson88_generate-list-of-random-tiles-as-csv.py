import os

import numpy as np

import pandas as pd



import tifffile

from tqdm import tqdm



import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go
IMAGE_PATH = "/kaggle/input/prostate-cancer-grade-assessment/train_images"



TILE_SIZE = 1024

MIN_TILES_PER_FILE = 36  # deprecated

TRIES = 400              # faster to use for loop with fixed number of tries
def maybe_get_offset(img, size=TILE_SIZE//16, threshold=.95):

    """

    Load a tile from a random location and return its position if its 

    "energy" exceeds a threshold; otherwise return None. 

    

    To be used in a loop that generates a list of offsets.

    """ 

    

    threshold = threshold * 765 * size**2

    try:

        high = (img.shape[0] - size - 1, img.shape[1] - size - 1)

        offset = np.random.randint(0, high=high, size=2)

    except ValueError:

        return None

    score = get_score(get_tile(img, offset))

    if score < threshold:

        return offset.tolist()



def get_tile(img, offset, size=TILE_SIZE//16):

    return img[offset[0]:offset[0]+size, offset[1]:offset[1]+size]



def get_score(tile):

    # Could be replaced with a more intelligent "energy" function

    return tile.sum()
# Build list of tuples (id, offset) with TRIES tries per id.

# Record the start and end index of the offsets for a fixed id.



# Threshold = .95 seems to hit about 50% of the time.



offsets = []

offset_addresses = []



count = 0



for path in tqdm(os.listdir(IMAGE_PATH)):

    

    idx = path[:-5]    # shaves off '.tiff'

    address = [idx, len(offsets)]

    with tifffile.TiffFile(os.path.join(IMAGE_PATH, path)) as handle:

        img = handle.asarray(2)    

        # Using the lowest resolution image should be good enough.

        

        for _ in range(TRIES):

            offset = maybe_get_offset(img, threshold=.95)

            if offset is not None:

                offsets.append((idx, offset))

        del(img)

        address.append(len(offsets))

        offset_addresses.append(address)

    

    count = count + 1

    if count > 100:    # for testing. Delete for full run.

        break



print(f"retrieved {len(offsets):d} offsets")
### Visualize sample tiles ###



fig = make_subplots(1,5)



sample = [offset_addresses[i] for i in np.random.randint(low=0, high=100, size=5)]

sample_id = [entry[0] for entry in sample]

sample_offsets = [offsets[np.random.randint(low=entry[1], 

                                            high=entry[2])][1:][0] for entry in sample]



for i in range(5):

    handle = tifffile.TiffFile(os.path.join(IMAGE_PATH, sample_id[i] + '.tiff'))

    img = get_tile(handle.asarray(2), sample_offsets[i])

    fig.add_trace(go.Image(z=img), 1, i + 1)



    

fig.update_layout(height=300, width=1000, title_text="Check out my tiles")

fig.show()
### Write output tables ###



import csv

offsets_flat = [[row[0], row[1][0]*16, row[1][1]*16] for row in offsets]

column_names = ["file_id", "y_offset", "x_offset"]

with open(f'offsets_{TILE_SIZE:d}.csv', 'w') as out:

    w = csv.writer(out)

    w.writerow(column_names)

    w.writerows(offsets_flat)

    

address_column_names = ["file_id", "offsets_start", "offsets_end"]

with open(f'offset_addresses_{TILE_SIZE:d}.csv', 'w') as out:

    w = csv.writer(out)

    w.writerow(address_column_names)

    w.writerows(offset_addresses)