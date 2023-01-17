# Install and imports

!pip install zstd



import json, zstd, os

import pandas as pd

HALITE_VERSION = 3
# Add the data directory

halite_path = f'/kaggle/input/connect-four-datasets/halite{HALITE_VERSION}'

halite_files = os.listdir(halite_path)

print(f'There are {len(halite_files)} training files')
# Load an hlt file >>> https://github.com/HaliteChallenge/Halite-III/blob/master/starter_kits/ml/SVM/parse.py

def load_hlt(hlt, path=halite_path):

    with open(os.path.join(path, hlt), 'rb') as f:

        data = json.loads(zstd.loads(f.read()))

    return data



sample_game = load_hlt(halite_files[0])

sample_game.keys()