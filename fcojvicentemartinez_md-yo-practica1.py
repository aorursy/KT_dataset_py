# Third party

from sklearn.dummy import DummyClassifier

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import KBinsDiscretizer

from sklearn.tree import DecisionTreeClassifier



import numpy as np

import seaborn as sns

sns.set()

from sklearn.impute import SimpleImputer



# Local application

import miner_a_de_datos_an_lisis_exploratorio_utilidad as utils
seed = 27912
filepath = "../input/pima-indians-diabetes-database/diabetes.csv"



index = False

target = "Outcome"



data = utils.load_data(filepath, index, target)
data.head(5)
(X, y) = utils.divide_dataset(data, target="Outcome")
X.sample(5, random_state=seed)
y.sample(5, random_state=seed)
train_size = 0.7



(X_train, X_test, y_train, y_test) = train_test_split(X, y,

                                                      shuffle=True,

                                                      stratify=y,

                                                      random_state=seed,

                                                      train_size=train_size)
X_train.sample(5, random_state=seed)
X_test.sample(5, random_state=seed)
y_train.sample(5, random_state=seed)
y_test.sample(5, random_state=seed)
data_train = utils.join_dataset(X_train, y_train)

data_test = utils.join_dataset(X_test, y_test)
data_train.sample(5, random_state=seed)
data_test.sample(5, random_state=seed)
data_train.shape
data_train.info(memory_usage=False)
y_train.cat.categories
utils.plot_histogram(data_train)
utils.plot_barplot(data_train)
sp = utils.plot_pairplot(data_train, target="Outcome")

sp.update_layout(width=1400, height=1400, hovermode='closest')

sp.show()
sns.heatmap(data_train.corr(), annot=True)
data_train.hist(figsize=(20,20))
X_train.sample(5, random_state=seed)
X_train[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

X_test[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
X_train.sample(5, random_state=seed)
imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
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