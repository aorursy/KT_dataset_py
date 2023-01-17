# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns
data = pd.read_csv("/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv")

data.head()
data.columns = ["mean_intrgt", "std_intrgt", "kurtosis_intrgt", "skew_intrgt", "mean_dmsnr", "std_dmsnr", "kurtosis_dmsnr", "skew_dmsnr", "class"]
sns.kdeplot(data[data["class"] == 0]["mean_intrgt"], label = "Class 0")

sns.kdeplot(data[data["class"] == 1]["mean_intrgt"], label = "Class 1")

plt.title("Mean of the integrated profile")

plt.show()
# DM-SNR curve

sns.kdeplot(data[data["class"] == 0]["mean_dmsnr"], label = "Class 0")

sns.kdeplot(data[data["class"] == 1]["mean_dmsnr"], label = "Class 1")

plt.title("Mean of the DM-SNR curve")

plt.show()
sns.kdeplot(data[data["class"] == 0]["std_intrgt"], label = "Class 0")

sns.kdeplot(data[data["class"] == 1]["std_intrgt"], label = "Class 1")

plt.title("Std. Dev. of the integrated profile")

plt.show()
# DM-SNR curve

sns.kdeplot(data[data["class"] == 0]["std_dmsnr"], label = "Class 0")

sns.kdeplot(data[data["class"] == 1]["std_dmsnr"], label = "Class 1")

plt.title("Std. Dev. of the DM-SNR curve")

plt.show()
sns.kdeplot(data[data["class"] == 0]["kurtosis_intrgt"], label = "Class 0")

sns.kdeplot(data[data["class"] == 1]["kurtosis_intrgt"], label = "Class 1")

plt.title("Excess Kurtosis of Integrated Profile")

plt.show()
sns.kdeplot(data[data["class"] == 0]["kurtosis_dmsnr"], label = "Class 0")

sns.kdeplot(data[data["class"] == 1]["kurtosis_dmsnr"], label = "Class 1")

plt.title("Excess Kurtosis of DM-SNR Curve")

plt.show()
sns.kdeplot(data[data["class"] == 0]["skew_intrgt"], label = "Class 0")

sns.kdeplot(data[data["class"] == 1]["skew_intrgt"], label = "Class 1")

plt.title("Skewness of Integrated Profile")

plt.show()
sns.kdeplot(data[data["class"] == 0]["skew_dmsnr"], label = "Class 0")

sns.kdeplot(data[data["class"] == 1]["skew_dmsnr"], label = "Class 1")

plt.title("Skewness of the DM-SNR Curve")

plt.show()
sns.countplot(data["class"])
from sklearn.decomposition import PCA 

pca = PCA(n_components=2)

pca_data = pca.fit_transform(data.iloc[:, :-1])



pca_data = pd.DataFrame(pca_data)

pca_data["class"] = data["class"]
plt.scatter(pca_data[pca_data["class"] == 1].iloc[:, 0], pca_data[pca_data["class"] == 1].iloc[:, 1], color = "green",label = "Class 1" )

plt.scatter(pca_data[pca_data["class"] == 0].iloc[:, 0], pca_data[pca_data["class"] == 0].iloc[:, 1], color = "red", label = "Class 0" )

plt.legend()

plt.show()
plt.scatter(pca_data[pca_data["class"] == 0].iloc[:, 0], pca_data[pca_data["class"] == 0].iloc[:, 1], color = "red", label = "Class 0" )

plt.scatter(pca_data[pca_data["class"] == 1].iloc[:, 0], pca_data[pca_data["class"] == 1].iloc[:, 1], color = "blue", alpha= .2,label = "Class 1" )

plt.legend()

plt.show()
sns.pairplot(data)
from scipy.stats import pearsonr

for i in data.columns[:-1]:

    for j in data.columns[:-1]:

        corr = pearsonr(data[i], data[j])[0]

        if corr > .7 or corr < -.7:

            print(i, " ", j, pearsonr(data[i], data[j])[0])
X = data.iloc[:, :-1]

y = data.iloc[:, -1]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, shuffle = True, random_state  = 8)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = .1, shuffle = True, random_state  = 8)
from sklearn.linear_model import LogisticRegression



model = LogisticRegression()

model.fit(X_train, y_train)
preds = model.predict(X_val)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_val, preds)

sns.heatmap(confusion_matrix(y_val, preds), annot = True, cmap="YlGnBu")
test_preds = model.predict(X_test)

confusion_matrix(y_test, test_preds)
print("Accuracy: ", (4846+ 404)/(4846+ 404 + 95 + 25))
precision = (4846)/(4846+ 25)

recall = (4846)/(4846+ 95)

print("Precision: ",precision)

print("Recall: ", recall)

print("f1 score: ", (2*precision*recall)/(precision + recall))