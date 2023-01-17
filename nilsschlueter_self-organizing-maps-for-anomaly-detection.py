!pip install MiniSom
from IPython.display import Image

from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from matplotlib import colors
import random
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import pandas as pd

%matplotlib inline
%load_ext autoreload
# Helper Code
def draw_som():
    fig, ax = plt.subplots(figsize=(6, 6))
    w = som.get_weights().T
    w = np.moveaxis(w, 0, -1)
    ax.imshow(w / 256, origin="lower")
    plt.grid(False)
    
def draw_data(data):
    fig, ax = plt.subplots(figsize=(25, 20))
    ax.imshow([x / 256 for x in np.reshape([data], (-1, len(data), 3))], origin="lower")
    plt.axis('off')
    
def draw_datapoint(datapoint):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow([x / 256 for x in np.reshape([datapoint], (-1, len(datapoint), 3))], origin="lower")
    plt.axis('off')
test_data = [
    [245,101,101], #red
    [237,137,54], #orange
    [236,201,75], #yellow
    [72,187,120], #green
    [56,178,172], #teal
    [66,153,225], #blue
    [102,126,234], #indigo
    [159,122,234], #purple
    [237,100,166], #pink
    [0, 0, 0],
    [255, 255, 255]
]

test_data_df = pd.DataFrame({"label": ["red", "orange", "yellow", "green", 
                                     "teal", "blue", "indigo", "purple", 
                                     "pink", "black", "white"],
                             "rgb": test_data,
                             "quant_error": np.zeros(len(test_data))})

train_data = [
    [254,215,215], [254,178,178], [252,129,129], [229, 62, 62], [197, 48, 48], [155, 44, 44], [116,42,42], # red
    [254,235,200], [251,211,141], [246,173,85], [221,107,32], [192,86,33], [221,107,32], [123,52,30], #orange
    [254,252,191], [250,240,137], [246,224,94], [214,158,46], [183,121,31], [151,90,22], [116,66,16], # yellow
    [198,246,213], [154,230,180], [104,211,145], [56,161,105], [47,133,90], [39,103,73], [34,84,61], # green
    [178,245,234], [129,230,217], [79,209,197], [49,151,149], [44,122,123], [40,94,97], [35,78,82], #teal
    [190,227,248], [144,205,244], [99,179,237], [49,130,206], [43,108,176], [44,82,130], [42,67,101], # blue
    [195,218,254], [163,191,250], [127,156,245], [90,103,216], [76,81,191], [67,65,144], [60,54,107], #indigo
    [233,216,253], [214,188,250], [183,148,244], [128,90,213], [107,70,193], [85,60,154], [68,51,122], #purple
    [254,215,226], [251,182,206], [246,135,179], [213,63,140], [184,50,128], [151,38,109], [112,36,89]  # pink
]


draw_data(train_data)
som = MiniSom(10, 10, 3)
som.random_weights_init(data=train_data) # Uses random training data
draw_som()
random_data_point = random.choice(train_data)
draw_datapoint([random_data_point])
def draw_bmu_compare(datapoint, winner_coords):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow([x / 256 for x in np.reshape([datapoint], (-1, len(datapoint), 3))], origin="lower")
    axes[0].axis('off')

    w = som.get_weights().T
    w = np.moveaxis(w, 0, -1)
    axes[1].imshow(w / 256, origin="lower")
    plt.plot(winner_coords[0], winner_coords[1], 'x', markeredgecolor="black", markersize=12, markeredgewidth=2)
winner_coords = som.winner(random_data_point)
draw_bmu_compare([random_data_point], winner_coords)

print(som.get_weights()[winner_coords[0]][winner_coords[1]])
print(random_data_point)
def draw_updated_som():
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    w = som.get_weights().T
    w = np.moveaxis(w, 0, -1)
    axes[0].imshow(w / 256, origin="lower")
    axes[0].plot(winner_coords[0], winner_coords[1], 'x', markeredgecolor="black", markersize=12, markeredgewidth=2)
    
    som.update(random_data_point, winner_coords, 0, 1)
    w = som.get_weights().T
    w = np.moveaxis(w, 0, -1)
    axes[1].imshow(w / 256, origin="lower")
    axes[1].plot(winner_coords[0], winner_coords[1], 'x', markeredgecolor="black", markersize=12, markeredgewidth=2)
    
    axes[0].grid(False)
    axes[1].grid(False)
draw_updated_som()
def draw_som_and_distance_map():
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow([x / 256 for x in som.get_weights()], origin="lower")
    axes[1].pcolor(som.distance_map().T, cmap='bone_r')
    axes[0].grid(False)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
som = MiniSom(10, 10, 3, learning_rate=0.001, sigma=1.2) # learning_rate und sigma von Hand gesetzt
som.random_weights_init(data=train_data)

num_iter = 100000 # 1, 10, 25, 50, 100, 500, 1000, 10000
som.train_batch(train_data, num_iter)
draw_som_and_distance_map()
def plot_quant_errors():
    sns.set(rc={'figure.figsize':(12, 6)})
    quant_errors = [som.quantization_error([datapoint_test]) for datapoint_test in train_data]
    quant_errors_compl = pd.DataFrame({"observation": range(len(quant_errors)), "Quantization Error": quant_errors})
    sns.lineplot('observation', 'Quantization Error', data=quant_errors_compl, label="Train Data")
plot_quant_errors()
print("Topographic Error: %s" % som.topographic_error(train_data))
quant_errors = [som.quantization_error([datapoint]) for datapoint in train_data] # Berechnen des Quantization Errors für alle Trainingsdaten
threshold = 0.95 * np.amax(quant_errors) # Der Wert ist frei wählbar

sns.distplot(quant_errors, axlabel='Quantization Error')
plt.axvline(threshold, color='red')
print("The Threshold is %s" % threshold)
test_data = [
    [245,101,101], #red
    [237,137,54], #orange
    [236,201,75], #yellow
    [72,187,120], #green
    [56,178,172], #teal
    [66,153,225], #blue
    [102,126,234], #indigo
    [159,122,234], #purple
    [237,100,166], #pink
    [0, 0, 0], #black -> anomaly
    [255, 255, 255] #white -> anomaly
]

draw_data(test_data)
def draw_quant_errors():
    quant_errors_test = [som.quantization_error([datapoint_test]) for datapoint_test in test_data]
    test_data_df["quant_error"] = quant_errors_test

    sns.set(rc={'figure.figsize':(12, 6)})
    palette = ['#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2]) for rgb in test_data_df["rgb"].to_list()]
    ax = sns.scatterplot(x=range(11), y=test_data_df["quant_error"], palette=palette,
                         hue=test_data_df["label"], legend=False, s=100, edgecolor='black')
    ax.set_xticks(range(11))
    ax.set_xticklabels(test_data_df['label'])
    plt.axhline(threshold, color='r')
draw_quant_errors()
# random_test_data = random.choice(test_data)
random_test_data = [10, 10, 10]

winner_coords = som.winner(random_test_data)
draw_bmu_compare([random_test_data], winner_coords)
print("Quantization Error: %s | Threshold: %s" % (som.quantization_error([random_test_data]), threshold))
# You can test the SOM with any random colors

random_test_data = random.choice(test_data)
# random_test_data = [25, 56, 29]

winner_coords = som.winner(random_test_data)
draw_bmu_compare([random_test_data], winner_coords)
