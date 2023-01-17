!nvidia-smi
!cat /usr/local/cuda/version.txt
## Passing Y as input while conda asks for confirmation, we use yes command

!yes Y | conda install faiss-gpu cudatoolkit=10.0 -c pytorch
# !wget https://anaconda.org/CannyLab/tsnecuda/2.1.0/download/linux-64/tsnecuda-2.1.0-cuda100.tar.bz2

# !tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2

# !cp -r site-packages/* /opt/conda/lib/python3.6/site-packages/

# # !export LD_LIBRARY_PATH="/kaggle/working/lib/" 

# !cp /kaggle/working/lib/libfaiss.so /usr/local/cuda/lib64/
!wget https://anaconda.org/CannyLab/tsnecuda/2.1.0/download/linux-64/tsnecuda-2.1.0-cuda100.tar.bz2

!tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2 --wildcards 'lib/*'

!tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2 --wildcards 'site-packages/*'

!cp -r site-packages/* /opt/conda/lib/python3.6/site-packages/

# !export LD_LIBRARY_PATH="/kaggle/working/lib/" 

!cp /kaggle/working/lib/libfaiss.so /usr/local/cuda/lib64/
!apt search openblas

!yes Y | apt install libopenblas-dev 

# !printf '%s\n' 0 | update-alternatives --config libblas.so.3 << 0

# !apt-get install libopenblas-dev 

!
import faiss

from tsnecuda import TSNE

import pandas as pd

import numpy as np

from  sklearn.manifold import TSNE as sktsne

import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_train = df_train.head(10000)

Y = df_train[['label']]

X = df_train.drop('label', axis=1)

def plot_digit(digits):

    fig, axs = plt.subplots(1,len(digits),figsize=(2,2))

    for i, pixels in enumerate(digits):

        ax = axs[i]

        digit_data = pixels.values.reshape(28,28)

        ax.imshow(digit_data,interpolation=None, cmap='gray')

    plt.show()
plot_digit([X.iloc[0], X.iloc[20], X.iloc[201]])
tsne_model = TSNE(n_components=2, perplexity=40.0, n_iter=2000).fit_transform(X)
tsne_df = pd.DataFrame(tsne_model)

tsne_df = pd.concat([tsne_df,Y], axis=1)
sns.FacetGrid(tsne_df, hue="label" , size=6).map(plt.scatter, 0, 1).add_legend()

plt.show()