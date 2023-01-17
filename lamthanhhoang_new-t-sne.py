import pandas as pd
import numpy as np
import bokeh.plotting as bp
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from bokeh.plotting import save
from bokeh.models import HoverTool
from mpl_toolkits.mplot3d import Axes3D
# load the result created by run LDA with the R4.fna dataset 
# with num_topics = 10
data = pd.read_csv('../input/prob_R4_fna_LDA_10.csv')
df = pd.DataFrame(data)
df.head()
# get the matrix from DataFrame
X_topics = df.values
X_topics
# run t-SNE to reduce 3-D
tsne_model = TSNE(n_components=3, verbose=1, random_state=0, angle=.99, init='pca')

# 10-D -> 3-D
tsne_lda = tsne_model.fit_transform(X_topics)
# convert to DataFrame
result_df = pd.DataFrame({'X': tsne_lda[:, 0], 'Y': tsne_lda[:, 1], 'Z': tsne_lda[:, 2]})
# write to csv file to use in later
result_df.to_csv('t_SNE_10_3_R4_fna_LDA_10.csv', index=False)
# add a color column into DataFrame
result_df['color'] = ['Red']*16447 + ['Blue']*18010
# plot in 3D
threedee = plt.figure().gca(projection='3d')

threedee.scatter(result_df['X'], result_df['Y'], result_df['Z'], marker='.', alpha=0.2, color=result_df['color'])
threedee.set_xlabel('X')
threedee.set_ylabel('Y')
threedee.set_zlabel('Z')
threedee.view_init(60, 35)