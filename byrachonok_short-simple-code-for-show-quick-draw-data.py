import pandas as pd
import matplotlib.pyplot as plt
import ast
df = pd.read_csv('/kaggle/input/train_simplified/owl.csv')
examples = [ast.literal_eval(e) for e in df['drawing'][:8].values]

fig, ax = plt.subplots(1,8,figsize=(20,4))
for i, example in enumerate(examples[:8]):
    for x, y in example:
        ax[i].plot(x, y, marker='.', markersize=1, lw=3)
        ax[i].invert_yaxis()
        ax[i].axis('off')
