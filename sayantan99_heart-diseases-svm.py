# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd # pandas is used to load and manipulate data and for One-Hot Encoding

import numpy as np # numpy is used to calculate the mean and standard deviation

import matplotlib.pyplot as plt # matplotlib is for drawing graphs

import matplotlib.colors as colors

from sklearn.model_selection import train_test_split # split  data into training and testing sets

from sklearn.preprocessing import scale # scale and center data

from sklearn.svm import SVC # this will make a support vector machine for classificaiton

from sklearn.model_selection import GridSearchCV # this will do cross validation

from sklearn.metrics import confusion_matrix # this creates a confusion matrix

from sklearn.metrics import plot_confusion_matrix # draws a confusion matrix

from sklearn.decomposition import PCA # to perform PCA to plot the data

import seaborn as sns

#sns.set(style="darkgrid")
heart=pd.read_csv('/kaggle/input/processed.cleveland.data',header=None)
heart.head()
heart.columns = ['age',

              'sex',

              'cp',

              'restbp',

              'chol',

              'fbs',

              'restecg',

              'thalach',

              'exang',

              'oldpeak',

              'slope',

              'ca',

              'thal',

              'target']

heart.head()
heart.target.value_counts()

sns.countplot(x="target",data=heart)

plt.show()

sns.countplot(x="target",data=heart,palette="Set3") # paletter="brw"

plt.show()
sns.countplot(x="target", data=heart,

             facecolor=(0,0,0,0),

             linewidth=3,

             edgecolor=sns.color_palette("dark",3))
countNoDisease=len(heart[heart.target == 0])

countHaveDisease=len(heart[heart.target != 0])

print(f"Percentage of patients with Heart Disease {countNoDisease/(len(heart.target))*100}")

print(countNoDisease)

print(len(heart.target))
heart['thal'].unique()
heart['ca'].unique()
len(heart.loc[(heart['ca'] == '?') | (heart['thal'] == '?')])
heart_with_no_missing = heart.loc[(heart['ca'] != '?') & (heart['thal'] != '?')]

final=heart_with_no_missing

print(len(final))

print(len(heart))
x=final.drop('target',axis=1).copy() # using copy() ensures that the original data is unchanged.

x.head()
y=final['target'].copy()

y.head()
pd.get_dummies(x, columns=['cp']).head()
X_encoded = pd.get_dummies(x, columns=['cp',

                                       'restecg',

                                       'slope', 

                                       'thal'])

X_encoded.head()
y_not_zero=y >0

y[y_not_zero]=1

y.unique()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale # scale and center data

from sklearn.svm import SVC # this will make a support vector machine for classificaiton

from sklearn.model_selection import GridSearchCV # this will do cross validation

from sklearn.metrics import confusion_matrix # this creates a confusion matrix

from sklearn import metrics

from sklearn.decomposition import PCA # to perform PCA to plot the data



x_train , x_test, y_train , y_test = train_test_split(x,y, random_state=42)

x_train_scaled=scale(x_train)

x_test_scaled=scale(x_test)
clf_svm = SVC(random_state=42)

clf_svm.fit(x_train_scaled, y_train)
c=metrics.plot_confusion_matrix(clf_svm, 

                      x_test_scaled, 

                      y_test, 

                      display_labels=["Does not have HD", "Has HD"])
param_grid=[{'C':[1,10,100,1000],

            'gamma':[0.001,0.0001],

            'kernel':['rbf']},

           ]

optimal_params=GridSearchCV(SVC(),

                           param_grid,

                           cv=5,

                           verbose=0) # To check what GridSeaarchCV is doing we can set the verbose to 2

optimal_params.fit(x_train_scaled,y_train)

print(optimal_params.best_params_)
pca=PCA() # PCA centers the data but doesnot scale it

x_train_pca=pca.fit_transform(x_train_scaled)



per_var=np.round(pca.explained_variance_ratio_*100, decimals=1)

labels=['PC'+ str(x) for x in range(1,len(per_var)+1)]



plt.bar(x=range(1,len(per_var)+1), height=per_var ,tick_label=labels)



plt.ylabel('Percentage of Explained Variance')

plt.xlabel('Principal Component')

plt.title('scree Plot')

plt.show()
pc1=x_train_pca[:,0]

pc2=x_train_pca[:,1]

clf_svm.fit(np.column_stack((pc1, pc2)), y_train)



## Now create a matrix of points that we can use to show

## the decision regions.

## The matrix will be a little bit larger than the

## transformed PCA points so that we can plot all of

## the PCA points on it without them being on the edge

x_min = pc1.min() - 1

x_max = pc1.max() + 1



y_min = pc2.min() - 1

y_max = pc2.max() + 1



xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=0.1),

                     np.arange(start=y_min, stop=y_max, step=0.1))



## now we will classify every point in that 

## matrix with the SVM. Points on one side of the 

## classification boundary will get 0, and points on the other

## side will get 1.

Z = clf_svm.predict(np.column_stack((xx.ravel(), yy.ravel())))

## Right now, Z is just a long array of lots of 0s and 1s, which

## reflect how each point in the mesh was classified.

## We use reshape() so that each classification (0 or 1) corresponds

## to a specific point in the matrix.

Z = Z.reshape(xx.shape)



fig, ax = plt.subplots(figsize=(10,10))

## now we will use contourf() to draw a filled contour plot

## using the matrix values and classifications. 

## The contours will be filled according to the 

## predicted classifications (0s and 1s) in Z

ax.contourf(xx, yy, Z, alpha=0.1)



## now create custom colors for the actual data points

cmap = colors.ListedColormap(['#e41a1c', '#4daf4a'])

## now darw the actual data points - these will

## be colored by their known (not predcited) classifications

## NOTE: setting alpha=0.7 lets us see if we are covering up a point 

scatter = ax.scatter(pc1, pc2, c=y_train, 

               cmap=cmap, 

               s=100, 

               edgecolors='k', ## 'k' = black

               alpha=0.7)



## now create a legend

legend = ax.legend(scatter.legend_elements()[0], 

                   scatter.legend_elements()[1],

                    loc="upper right")

legend.get_texts()[0].set_text("No HD")

legend.get_texts()[1].set_text("Yes HD")



## now add axis labels and titles

ax.set_ylabel('PC2')

ax.set_xlabel('PC1')

ax.set_title('Decison surface using the PCA transformed/projected features')

# plt.savefig('svm.png')

plt.show()