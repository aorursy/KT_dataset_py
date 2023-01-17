import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns





cereal_df = pd.read_csv('../input/cereal.csv')



sns.distplot( cereal_df["calories"] , color="skyblue", axlabel="Calories"  ,kde=False).set_title('Calories frequency')





f, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True)

sns.distplot( cereal_df["sugars"] , color="skyblue", ax=axes[0, 0] , bins=20,axlabel="Sugars" )

sns.distplot( cereal_df["fiber"] , color="olive", ax=axes[0, 1] , bins=20 ,axlabel="Fiber" )

sns.distplot( cereal_df["fat"] , color="gold", ax=axes[1, 0] , bins=20 ,axlabel="Fat" )

sns.distplot( cereal_df["protein"] , color="teal", ax=axes[1, 1] , bins=20 ,axlabel="Protein" )




