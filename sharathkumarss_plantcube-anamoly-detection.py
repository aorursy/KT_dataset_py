# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
from scipy.stats import pearsonr,spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
!pip install pyod
from pyod.models.knn import KNN
import matplotlib
plant_data  = pd.read_csv(r"../input/simulated_plantcube_archive.csv")
plant_data.head()
plant_data.shape
correlation_matrix = plant_data.corr()
print("Pearson corelation of variable{}".format(correlation_matrix))
#check obeservations which has null values
nan_values = plant_data.isnull().sum(axis=0)
print('missing data:\n {}'.format(nan_values))
cleaned_data = plant_data.dropna(axis=1)
cleaned_data.head()
cleaned_data.shape
cleaned_data.rename(columns={"Unnamed: 0": "unnamed"}, inplace=True)
len(set(cleaned_data["Cube ID"]))
cnts = Counter(cleaned_data["Cube ID"])
cnts
sns.set(font_scale = 1.5)
plt.figure(figsize=(15,8))

ax= sns.barplot(list(cnts.keys()), list(cnts.values()))

plt.title("Cube IDs in each category", fontsize=22)
plt.ylabel('Number of cube ids', fontsize=18)
plt.xlabel('Cube id Type ', fontsize=18)

#adding the text labels
rects = ax.patches
labels = cnts.values()
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=0)

plt.show()
pear_corr, _ = pearsonr(cleaned_data["unnamed"], cleaned_data["Cube ID"])
spear_corr, _ = spearmanr(cleaned_data["unnamed"], cleaned_data["Cube ID"])
print('Pearsons correlation: %.3f' % pear_corr)
print('Spearman correlation: %.3f' % spear_corr)
#cleaned_data["Timestamp"] = pd.to_datetime(cleaned_data["Timestamp"])
cleaned_data = cleaned_data.sort_values(by="Timestamp")

cleaned_data
cleaned_data.tail()
plt.scatter(cleaned_data["unnamed"], cleaned_data["Cube ID"])
plt.xlabel("unnamed")
plt.ylabel("cube id")
plt.show()
plt.scatter(range(cleaned_data.shape[0]), np.sort(cleaned_data["unnamed"].values))
plt.xlabel('index')
plt.ylabel('unnamed')
plt.title("unnamed distribution")
sns.despine()
sns.distplot(cleaned_data["unnamed"])
plt.title("Distribution of unnamed")
sns.despine()
plt.scatter(range(cleaned_data.shape[0]), np.sort(cleaned_data["Cube ID"].values))
plt.xlabel('index')
plt.ylabel('Cube id')
plt.title("Cube id distribution")
sns.despine()
sns.distplot(cleaned_data["Cube ID"])
plt.title("Distribution of cube id")
sns.despine()
x_knn = cleaned_data[["unnamed","Cube ID"]]
outliers_fraction = 0.2
clf = KNN(contamination=outliers_fraction)
clf.fit(x_knn)
scores_pred = clf.decision_function(x_knn) * -1
y_pred = clf.predict(x_knn)
n_inliers = len(y_pred) - np.count_nonzero(y_pred)
n_outliers = np.count_nonzero(y_pred == 1)

df1 = cleaned_data
df1['outlier'] = y_pred.tolist()
            
print('OUTLIERS: ',n_outliers,'INLIERS: ',n_inliers)

df1[df1["outlier"]==1]
X_train, X_test, y_train, y_test = train_test_split(cleaned_data["unnamed"], cleaned_data["Cube ID"], test_size=0.2) 
X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
X_train.shape
X_test.shape
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=11)

# Train the model using the training sets
model.fit(X_train,y_train)

#Predict Output
predicted= model.predict(X_test) 
print(predicted)
from sklearn import metrics
# Model Accuracy
print("Accuracy on test data:",metrics.accuracy_score(y_test, predicted))
comparison = list(zip(y_test, predicted))
comparison