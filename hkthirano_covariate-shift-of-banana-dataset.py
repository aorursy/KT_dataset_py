# reference
# https://www.kaggle.com/hkthirano/covariate_shift_of_banana_dataset
# https://qiita.com/kenmatsu4/items/0a7a3ef71d4e8bb53da0
# https://pypi.org/project/densratio/
!pip install densratio
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from densratio import densratio
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../input/standard-classification-banana-dataset/banana.csv",sep=",")
print(df.head())
df_de = df[df['Class']==1][['At1', 'At2']]; de = scale(df_de) # as normal
df_nu = df[df['Class']==-1][['At1', 'At2']]; nu = scale(df_nu) # as anomal

sns.jointplot(x="At1", y="At2", data=df_de, kind='kde')
sns.jointplot(x="At1", y="At2", data=df_nu, kind='kde')
# Calculate the density ratio
densratio_obj = densratio(de, nu)
# de
w_hat_de = densratio_obj.compute_density_ratio(de) # Get the density ratio
anom_score_de = -np.log(w_hat_de) # Converting density ratio to anomaly score

# nu
w_hat_nu = densratio_obj.compute_density_ratio(nu)
anom_score_nu = -np.log(w_hat_nu)
# Set the 95th percentile of the density ratio of normal samples as the threshold for detecting abnormalities.
thresh = np.percentile(anom_score_de, 95)

plt.figure()
plt.scatter(
    x=list(range(len(de))),
    y=anom_score_de,
    color='b', s=5)
plt.hlines(thresh, 0, len(de), "r", linestyles='dashed')
plt.show()

plt.figure()
plt.hist(anom_score_de, color='b', bins=40)
plt.vlines(thresh, 1000, 0, "r", linestyles='dashed')
plt.show()
plt.figure()
plt.scatter(
    x=list(range(len(nu))),
    y=anom_score_nu,
    color='b', s=5)
plt.hlines(thresh, 0, len(nu), "r", linestyles='dashed')
plt.show()

plt.figure()
plt.hist(anom_score_nu, color='b', bins=40)
plt.vlines(thresh, 800, 0, "r", linestyles='dashed')
plt.show()
