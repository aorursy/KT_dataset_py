# load some default libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# define a PATH variables so you easily use this notebook in a different computing environment
PATH = "../input/"

# read csv ad Pandas dataframe
df_train = pd.read_csv(PATH + "train.csv")
df_train.info()
# print head of dataframe
df_train.head(5)
# put labels in list for easy printing
protein_labels = [
    'Nucleoplasm', 
    'Nuclear membrane', 
    'Nucleoli', 
    'Nucleoli fibrillar center', 
    'Nuclear speckles', 
    'Nuclear bodies', 
    'Endoplasmic reticulum', 
    'Golgi apparatus', 
    'Peroxisomes', 
    'Endosomes', 
    'Lysosomes', 
    'Intermediate filaments', 
    'Actin filaments', 
    'Focal adhesion sites', 
    'Microtubules',
    'Microtubule ends', 
    'Cytokinetic bridge', 
    'Mitotic spindle', 
    'Microtubule organizing center',
    'Centrosome', 
    'Lipid droplets',
    'Plasma membrane', 
    'Cell junctions',
    'Mitochondria', 
    'Aggresome', 
    'Cytosol', 
    'Cytoplasmic bodies',
    'Rods & rings']

# define function to print labels
def print_labels(target):
    label_ints = [int(l) for l in target.split()]
    for i in label_ints:
        print("{} - {}".format(i, protein_labels[i]))
# each datapoint consists of four images with different color 
# define function to plot those four images
def plot_protein_images(id):
    fig, axs = plt.subplots(1, 4, figsize=(16,4))

    for i, color in enumerate(['red', 'green', 'yellow', 'blue']):
        filename = "train/{}_{}.png".format(id, color)
        im = plt.imread(PATH + filename)
        axs[i].imshow(im, cmap='binary')
        axs[i].set_title(color)
plot_protein_images(df_train.Id[0])
print_labels(df_train.Target[0])
plot_protein_images(df_train.Id[1])
print_labels(df_train.Target[1])
def plot_color_protein_images(id, ax=None, figsize=(10,10)):
    # use ax argument so this function can be using to plot in a grid using axes
    if ax==None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # read all color images
    all_images = np.empty((512,512,4))
    for i, color in enumerate(['red', 'green', 'yellow', 'blue']):
        all_images[:,:,i] = plt.imread(PATH + "train/{}_{}.png".format(id, color))

    # define transformation matrix
    # note that yellow is made usign red and green
    # but you can tune this color conversion yourself
    T = np.array([[1,0,1,0],[0,1,1,0],[0,0,0,1]])
    
    # convert to rgb
    rgb_image = np.matmul(all_images.reshape(-1, 4), np.transpose(T))
    rgb_image = rgb_image.reshape(all_images.shape[0], all_images.shape[0], 3)
    rgb_image = np.clip(rgb_image, 0, 1)
    
    # plot
    ax.imshow(rgb_image)
    ax.set(xticks=[], yticks=[])
plot_color_protein_images(df_train.Id[1])
# plot color protein images with target as title
n_rows, n_cols = 4, 8
fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 10))
axs = axs.ravel()
N = n_rows * n_cols
for i in range(N):
    plot_color_protein_images(df_train.Id[i], axs[i])
    axs[i].set_title(df_train.Target[i])
# number of samples
n_datapoints = df_train.shape[0]
n_datapoints
# count protein targets
count = np.zeros(len(protein_labels))
for target in df_train.Target:
    label_ints = [int(l) for l in target.split()]
    count[label_ints] = count[label_ints] + 1

plt.figure(figsize=(14,6))
plt.bar(range(len(protein_labels)), count)
plt.ylabel('count')
plt.xticks(range(len(protein_labels)), protein_labels, rotation=-90);
# create array with target encoding
n_labels = len(protein_labels)
a_targets = np.zeros((n_datapoints, n_labels))
for i, target in enumerate(df_train.Target):
    label_ints = [int(l) for l in target.split()]
    a_targets[i, label_ints] = 1 
# calculate correlation matrix
C = np.corrcoef(a_targets, rowvar=False)
C.shape
plt.figure(figsize=(10,8))
sns.heatmap(C, xticklabels=protein_labels, yticklabels=protein_labels)
plt.title('Correlation matrix')
