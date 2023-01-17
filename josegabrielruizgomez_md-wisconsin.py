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
# Third party

from sklearn.dummy import DummyClassifier

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import KBinsDiscretizer

from sklearn.tree import DecisionTreeClassifier



import seaborn as sns

sns.set()

from sklearn.impute import SimpleImputer



#IMPORTANTE AÑADIRLO EN NUESTRO LOCAL

# Local application

import miner_a_de_datos_an_lisis_exploratorio_utilidad as utils
#Fijación de la semilla 

seed = 27912
filepath = "../input/breast-cancer-wisconsin-data/data.csv"



index = "id"

target = "diagnosis"



data = utils.load_data(filepath, index, target)
(X, y) = utils.divide_dataset(data, target="diagnosis")
train_size = 0.7



(X_train, X_test, y_train, y_test) = train_test_split(X, y,

                                                     stratify=y, 

                                                     random_state=seed,

                                                     train_size=train_size)

data_train = utils.join_dataset(X_train, y_train)
data_train
data_test = utils.join_dataset(X_test, y_test)
print(X_train.shape)

print(y_train.shape)
data_train.info(memory_usage=False)
y_train.cat.categories
utils.plot_histogram(X_train)
utils.plot_barplot(data)
data_train.describe(include="category")
data_train.describe(include="number")
means_data = data_train[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean", "concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","diagnosis"]]

SE_data = data_train[["radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se", "concavity_se","concave points_se","symmetry_se","fractal_dimension_se","diagnosis"]]

worst_data = data_train[["radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst", "concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst","diagnosis"]]
sns.heatmap(means_data.corr(), annot=True)

sns.heatmap(SE_data.corr(), annot=True)
sns.heatmap(worst_data.corr(), annot=True)
sp = utils.plot_pairplot(means_data, target="diagnosis")

sp.update_layout(width=1400, height=1400, hovermode='closest')

sp.show()
sp = utils.plot_pairplot(SE_data, target="diagnosis")

sp.update_layout(width=1400, height=1400, hovermode='closest')

sp.show()
sp = utils.plot_pairplot(worst_data, target="diagnosis")

sp.update_layout(width=1400, height=1400, hovermode='closest')

sp.show()
X_train.sample(100, random_state=seed)
#imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')

#imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')

imputer = SimpleImputer(missing_values=np.NaN, strategy='median')
discretizer = KBinsDiscretizer(n_bins=2, strategy="kmeans")
zero_r_model = make_pipeline(imputer, DummyClassifier(strategy="most_frequent"))
tree_model = make_pipeline(imputer, DecisionTreeClassifier(random_state=seed))
discretize_tree_model = make_pipeline(imputer, discretizer, DecisionTreeClassifier(random_state=seed))
utils.evaluate(zero_r_model,

              X_train, X_test,

              y_train, y_test)
utils.evaluate(tree_model,

              X_train, X_test,

              y_train, y_test)
utils.evaluate(discretize_tree_model,

               X_train, X_test,

               y_train, y_test)