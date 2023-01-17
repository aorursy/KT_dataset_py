!pip install pycaret
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pycaret.classification import setup, models, compare_models, predict_model
data = pd.read_csv('../input/glass/glass.csv')
data
data.info()
corr = data.corr()

plt.figure(figsize=(16, 14))
sns.heatmap(corr, annot=True, vmin=-1.0, vmax=1.0, cmap='mako')
plt.title("Correlations")
plt.show()
plt.figure(figsize=(12, 12))
plt.pie(data['Type'].value_counts(), labels=data['Type'].value_counts().index, colors=sns.color_palette('mako'))
plt.title("Class Distribution")
plt.show()
setup(
    data=data,
    target='Type',
    normalize=True,
    train_size=0.7
)
models()
best_model = compare_models()
predict_model(best_model)