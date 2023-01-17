import pandas as pd
import umap

digits = pd.read_csv("../input/train.csv")

embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit_transform(digits.iloc[:20000, 1:])
import matplotlib.pyplot as plt

plt.figure(figsize=(12,12))
plt.scatter(embedding[:20000, 0], embedding[:20000, 1], 
            c=digits.iloc[:20000, 0], 
            edgecolor='none', 
            alpha=0.80, 
            s=10)
plt.axis('off');