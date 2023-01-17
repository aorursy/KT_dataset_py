import pandas as pd

import numpy as np

from xgboost import XGBClassifier,plot_tree

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
df = {'Dosage':[2,8,12,18],'Gender': ['M','F','M','F'],'Effect':[0,1,1,0]}

df = pd.DataFrame(df)

df
# label encode Gender

lb = LabelEncoder()

df['Gender'] = lb.fit_transform(df['Gender'])



# dataset

X = df.drop(columns=['Effect']).values

y = df['Effect'].values



# Define parameters and fit XgBoost Model

model=XGBClassifier(max_depth=2,learning_rate=1,n_estimators=2,gamma=2,

                    min_child_weight=0,reg_alpha=0,reg_lambda=0,base_score=0.5)

model.fit(X, y)





# plot the first tree

plot_tree(model, num_trees=0)

plt.show()


