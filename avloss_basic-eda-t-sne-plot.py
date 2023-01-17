#%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("../input/train.csv")
df_test =  pd.read_csv("../input/test.csv")
print(f"Train data has {len(df)} rows and {len(df.columns)} columns")
print(f"Test data has {len(df_test)} rows")
print(f"Sicne we have {4993 - 4459} more columns than rows, there might be an overfitting issues")
df.head(4)
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))

feat_cols = [c for c in df.columns if c not in ["ID", "target"]]
flatten_values = df[feat_cols].values.flatten()

# Data to plot
labels = 'Non-zero', 'Zero'
sizes = [sum(flatten_values!=0), sum(flatten_values==0)]
colors = ['gold', 'yellowgreen']
explode = (0.4, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("Looks like most of the values are zeros")
plt.show()
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df[feat_cols].values)
%matplotlib inline
plt.figure(figsize=(20,10))
plt.title("groups of bright spots here and there but otherwise seems quite random")
plt.scatter(*tsne_results.clip(-15,15).T, c=df["target"])
plt.show()
