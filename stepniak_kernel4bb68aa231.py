import pandas as pd

import numpy as np



df = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
from sklearn.preprocessing import StandardScaler

features = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']# Separating out the features

x = df.loc[:, features].values# Separating out the target

y = df.loc[:,['Outcome']].values# Standardizing the features

x = StandardScaler().fit_transform(x)
x
from sklearn.decomposition import PCA

pca = PCA(n_components=8)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents, columns=['pc'+str(i) for i in range(8)])
finalDf = pd.concat([principalDf, df[['Outcome']]], axis = 1)

finalDf
from matplotlib import pyplot as plt

plt.plot(pca.explained_variance_ratio_.cumsum())
fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)

targets = [0, 1]

colors = ['r', 'g']

for target, color in zip(targets,colors):

    indicesToKeep = finalDf['Outcome'] == target

    ax.scatter(finalDf.loc[indicesToKeep, 'pc1']

               , finalDf.loc[indicesToKeep, 'pc2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()