import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv("../input/data.csv")
df.head()

df.shape
df.isnull().sum()
plt.figure(1)
plt.subplot(221)
df['clump_thickness'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'clump_thickness', color="red")

plt.subplot(222)
df['unif_cell_size'].value_counts(normalize=True).plot.bar(title= 'unif_cell_size',color="green")

plt.subplot(223)
df['unif_cell_shape'].value_counts(normalize=True).plot.bar(title= 'unif_cell_shape',color="pink")

plt.subplot(224)
df['marg_adesion'].value_counts(normalize=True).plot.bar(title= 'marg_adesion',color="blue")



plt.show()
plt.figure(1)
plt.subplot(221)
df['single_epith_cell_size'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'single_epith_cell_size', color="red")

plt.subplot(222)
df['bare_nuclei'].value_counts(normalize=True).plot.bar(title= 'bare_nuclei',color="green")

plt.subplot(223)
df['bland_chrom'].value_counts(normalize=True).plot.bar(title= 'bland_chrom',color="pink")

plt.subplot(224)
df['norm_nuclei'].value_counts(normalize=True).plot.bar(title= 'norm_nuclei',color="blue")



plt.show()
plt.figure(1)
plt.subplot(221)
df['mitoses'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'mitoses', color="red")

plt.subplot(222)
df['class'].value_counts(normalize=True).plot.bar(title= 'class',color="green")

plt.show()
df["bare_nuclei"].value_counts()
df['bare_nuclei'].replace(('?'), (1),inplace=True)

df=df.drop("id",axis=1)
df.head()
X=df.drop("class",axis=1)
y=df["class"]
X.head()
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
pred_test = model.predict(X_test)
score = accuracy_score(y_test,pred_test)
print('accuracy_score',score)