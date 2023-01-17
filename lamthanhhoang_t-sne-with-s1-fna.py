# !pip install bokeh
import pandas as pd
import numpy as np
import bokeh.plotting as bp
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from bokeh.plotting import save
from bokeh.models import HoverTool
from mpl_toolkits.mplot3d import Axes3D
# load the result created by run LDA with the S1.fna dataset 
# with num_topics = 10, alpha = 1 and beta = 0.1
data = pd.read_csv('../input/prob_S1_fna_LDA_10_1_0.1.csv')
df = pd.DataFrame(data)
df.head()
len(df)
# get the matrix from DataFrame
X_topics = df.values
X_topics
# run t-SNE to reduce to 2-D
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')

# 10-D -> 2-D
tsne_lda = tsne_model.fit_transform(X_topics)
tsne_lda.shape
tsne_lda[:, 0]
# convert the result matrix to DataFrame
result_df = pd.DataFrame({'X': tsne_lda[:, 0], 'Y': tsne_lda[:, 1]})
# add a color column into DataFrame
result_df['color'] = ['Blue']*44405 + ['Red']*51962
# plot the result using scatter plot in pandas
result_df.plot(kind='scatter', x='X', y='Y', c=result_df['color'])
# run t-SNE to reduce 3-D
tsne_model = TSNE(n_components=3, verbose=1, random_state=0, angle=.99, init='pca')

# 10-D -> 3-D
tsne_lda = tsne_model.fit_transform(X_topics)
# convert to DataFrame
result_df = pd.DataFrame({'X': tsne_lda[:, 0], 'Y': tsne_lda[:, 1], 'Z': tsne_lda[:, 2]})
# write to csv file to use in later
result_df.to_csv('t_SNE_10_3_S1_fna_LDA_10_1_0.1.csv', index=False)
# add a color column into DataFrame
result_df['color'] = ['Blue']*44405 + ['Red']*51962
# plot in 3D
threedee = plt.figure().gca(projection='3d')

threedee.scatter(result_df['X'], result_df['Y'], result_df['Z'], marker='.', alpha=0.2, color=result_df['color'])
threedee.set_xlabel('X')
threedee.set_ylabel('Y')
threedee.set_zlabel('Z')
threedee.view_init(60, 35)