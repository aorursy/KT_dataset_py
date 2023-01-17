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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')

from sklearn.metrics import confusion_matrix

from sklearn import metrics
df = pd.read_csv('/kaggle/input/wine-customer-segmentation/Wine.csv')

df.head(10)
df.info()
plt.figure(figsize=(12,8))

sns.heatmap(df.describe()[1:].transpose(),

            annot=True,linecolor="w",

            linewidth=2,cmap=sns.color_palette("Set1"))

plt.title("Data summary")

plt.show()
cor_mat= df[:].corr()

mask = np.array(cor_mat)

mask[np.tril_indices_from(mask)] = False

fig=plt.gcf()

fig.set_size_inches(30,12)

sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)
corr=df.corr()

corr.sort_values(by=["Customer_Segment"],ascending=False).iloc[0].sort_values(ascending=False)
print('Goup 1:',len(df[df.Customer_Segment == 1]))

print('Group 2:',len(df[df.Customer_Segment == 2]))

print('Group 3:',len(df[df.Customer_Segment == 3]))
plt.rcParams['figure.figsize'] = (20, 10)

size = [59, 71, 48]

colors = ['mediumseagreen', 'c', 'gold']

labels = "Group A", "Group B", "Group C"

explode = [0, 0, 0.1]

plt.subplot(1, 2, 1)

plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')

#plt.title('Different Visitors', fontsize = 20)

plt.axis('off')

plt.legend()
plt.figure(figsize=(10,9))

sns.scatterplot(x='Ash_Alcanity',y='Color_Intensity',data=df,palette='Set1', hue = 'Customer_Segment');
X = df.drop('Customer_Segment',axis=1).values

y = df['Customer_Segment'].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.decomposition import PCA

pca = PCA(n_components = None)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

explained_variance
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

explained_variance
from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train, y_train)
#prediction the test set result

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test,y_pred)

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cm), annot=True, cmap="BuPu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from matplotlib.colors import ListedColormap

plt.figure(figsize=(12,8))

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)

plt.title('Logistic Regression (Training set)')

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.legend()

plt.show()
from matplotlib.colors import ListedColormap

plt.figure(figsize=(12,8))

X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)

plt.title('Logistic Regression (Test set)')

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.legend()

plt.show()