import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
from sklearn.preprocessing import StandardScaler
dat = pd.read_csv("/kaggle/input/prin-for-1815/PRIN.csv")
print(dat.shape)
dat.head()
dat.info()
dat.describe()
display(dat.corr())
sns.pairplot(dat)
sc = StandardScaler()
dats = pd.DataFrame(sc.fit_transform(dat),columns=dat.columns)
display(dat.head(), dats.head())
pca = PCA()
pca.fit(dats)
feature = pca.transform(dats)
a = pd.DataFrame(pca.explained_variance_, index=["PC{}".format(x + 1) for x in range(len(dats.columns))], columns=["固有値"])
a["寄与率"] = a["固有値"] / a["固有値"].sum()
a["累積寄与率"] = ""
for i in range(len(a["寄与率"])):
    if i != 0:
        a.at[a.index[i],"累積寄与率"] = a.at[a.index[i], "寄与率"] + a.at[a.index[i-1], "累積寄与率"]
    elif i == 0:
        a.at[a.index[0], "累積寄与率"] = a.at[a.index[0], "寄与率"]
display(a)
plt.subplots(facecolor='w', figsize=(8,5)) 
plt.grid()
plt.plot(a.index, a["累積寄与率"])
plt.bar(a.index, a["寄与率"], alpha=0.4)
plt.show()
pd.DataFrame((-1)*pca.components_, columns=dat.columns[:], index=["PC{}".format(x + 1) for x in range(len(dat.columns))]).T
tmp = pd.DataFrame((-1)*feature, columns=["PC{}".format(x + 1) for x in range(len(dats.columns))]).iloc[:,0:2]
tmp2 = pd.concat([dat, tmp], axis=1)
display(tmp2.head())
fig, ax = plt.subplots(facecolor="w", figsize=(8,6))
x = np.linspace(-5, 5, 101)
y = np.linspace(-5, 5, 101)
z = np.zeros(101)
ax.set_ylabel("PC1", size=20)
ax.set_xlabel("PC2", size=20)

plt.text(1.75, 0, 'Liberal Arts→', size=18, bbox=dict(alpha=0.4, boxstyle="round",
                   ec=(1., 0., 0.),
                   fc=(1., 1., 1.),
                   ))
plt.text(-4.2, 0, '←Sience', size=18, bbox=dict(alpha=0.4, boxstyle="round",
                   ec=(1., 0., 0.),
                   fc=(1., 1., 1.),
                   ))
plt.text(0,4, '↑High', size=18,  bbox=dict(alpha=0.4, boxstyle="round",
                   ec=(1., 0., 0.),
                   fc=(1., 1., 1.),
                   ))
plt.text(0,-4., '↓Low', size=18, bbox=dict(alpha=0.4, boxstyle="round",
                   ec=(1., 0., 0.),
                   fc=(1., 1., 1.),
                   ))

plt.plot(x,z)
plt.plot(z,y)
plt.plot(0.1,0.5)
plt.scatter(tmp2["PC1"], tmp2["PC2"])
plt.show()

