import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

train_df = pd.read_csv("../input/train.csv",nrows=2000)
X = train_df.drop("label",axis=1)
print (X.values.shape)
y = train_df.label

clf = RandomForestClassifier(n_estimators=60,max_depth=5)
rfe = RFE(estimator=clf)
rfe.fit(X,y)
rank = rfe.ranking_.reshape((28,28))
sns.heatmap(rank)
plt.title("Ranking of pixels with RFE")
plt.show()