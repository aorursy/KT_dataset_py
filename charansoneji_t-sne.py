import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn import datasets
from sklearn import manifold
data=datasets.fetch_openml('mnist_784',version=1,return_X_y=True)
images,target=data
single_image=images[1,:].reshape(28,28)
plt.imshow(single_image,cmap='gray')
tsne = manifold.TSNE(n_components=2, random_state=42)  
transformed_data = tsne.fit_transform(images[:3000,:])
tsne_df = pd.DataFrame(  np.column_stack((transformed_data, target[:3000])),  columns=["x", "y", "targets"]  )  
tsne_df.loc[:, "targets"] = tsne_df.targets.astype(int) 
grid = sns.FacetGrid(tsne_df, hue="targets", size=8)  
grid.map(plt.scatter, "x", "y").add_legend() 
