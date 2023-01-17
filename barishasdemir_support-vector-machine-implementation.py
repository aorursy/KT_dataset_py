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
import matplotlib.pyplot as plt

import seaborn as sns



# import Support Vector Classifier

from sklearn.svm import SVC



# for splitting the data into train and test sets

from sklearn.model_selection import train_test_split



# to evaluate the model

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score



# I will keep the resulting plots

%matplotlib inline



# Enable Jupyter Notebook's intellisense

%config IPCompleter.greedy=True
# Load the data set

breast_cancer = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")



# Display first 5 rows of the DataFrame

display(breast_cancer.head())



# Display the statistics

display(breast_cancer.describe())



# Print info

print(breast_cancer.info())

# Drop the columns

breast_cancer.drop(["Unnamed: 32","id"], axis=1, inplace=True)
# Print the counts

print(breast_cancer["diagnosis"].value_counts())



# Visualize the counts

sns.countplot(breast_cancer["diagnosis"])

plt.show()
# Visualize some 2-D features to see patterns

def make_scatterplot(x,y):

    sns.scatterplot(x,y,data=breast_cancer,hue='diagnosis')

    plt.title(y + " vs " + x)

    plt.show()

    

make_scatterplot('radius_mean', 'texture_mean')

make_scatterplot('perimeter_mean', 'area_mean')

make_scatterplot('smoothness_mean', 'smoothness_se')

make_scatterplot('concavity_mean', 'compactness_mean')

make_scatterplot('fractal_dimension_mean', 'perimeter_se')

make_scatterplot('symmetry_worst', 'concave points_worst')
# Print the correlation matrix

print(breast_cancer.corr())
# Visualize with a heatmap

figure, ax = plt.subplots(figsize=(20,20))

mask = np.triu(np.ones_like(breast_cancer.corr(), dtype=np.bool))

sns.heatmap(breast_cancer.corr(), mask=mask, annot=True)

plt.show()
# Plot the histograms

def plot_histogram(column):

    sns.distplot(breast_cancer[column])

    plt.title(column)

    plt.show()





plot_histogram("radius_mean")

plot_histogram("texture_mean")

plot_histogram("perimeter_mean")

plot_histogram("area_mean")

plot_histogram("smoothness_mean")

plot_histogram("compactness_mean")

plot_histogram("concavity_mean")

plot_histogram("concave points_mean")

plot_histogram("symmetry_mean")

plot_histogram("fractal_dimension_mean")

# import TSNE

from sklearn.manifold import TSNE



# fit and transform the TSNE model

tsne = TSNE(learning_rate =  50)

tsne_f = tsne.fit_transform(breast_cancer.drop("diagnosis", axis=1))



# Create a new DataFrame to store reduced features

df = pd.DataFrame({'x':tsne_f[:,0],'y':tsne_f[:,1]})



print("Before:",breast_cancer.shape)

print("After",df.shape)



display(df.head())

sns.scatterplot(x='x', y='y', hue=breast_cancer['diagnosis'],data=df)

plt.title("After Dimensionality Reduction")

plt.show()
# Get features and the target

X = breast_cancer.drop("diagnosis", axis=1)

y = breast_cancer["diagnosis"]
# Split the data as 30% test and 80% training sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)
# Initialize the Support vector classifier

svc = SVC(kernel="linear")



# Fit the SVC with training sets

svc.fit(X_train, y_train)



scores = cross_val_score(svc, X_train, y_train, cv=10, scoring='f1_macro')



print(scores)

# Make predictions on test set

y_pred = svc.predict(X_test)



acc = accuracy_score(y_test, y_pred)

print("Accuracy:",acc)



print("\n Classification Report\n")

print(classification_report(y_test, y_pred))