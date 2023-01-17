# Third party

from sklearn.dummy import DummyClassifier

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import KBinsDiscretizer

from sklearn.tree import DecisionTreeClassifier



# Local application

import miner_a_de_datos_an_lisis_exploratorio_utilidad as utils
seed = 27912
filepath = "../input/iris/Iris.csv"



index = "Id"

target = "Species"



data = utils.load_data(filepath, index, target)
data.head(5)
data.sample(5, random_state=seed)
(X, y) = utils.divide_dataset(data, target="Species")
X.sample(5, random_state=seed)
y.sample(5, random_state=seed)
train_size = 0.7



(X_train, X_test, y_train, y_test) = train_test_split(X, y,

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
utils.plot_pairplot(data_train, target="Species")
data_train.describe(include="number")
data_train.describe(include="category")
discretizer = KBinsDiscretizer(n_bins=3, strategy="uniform")
zero_r_model = DummyClassifier(strategy="most_frequent")
tree_model = DecisionTreeClassifier(random_state=seed)
discretize_tree_model = make_pipeline(discretizer, tree_model)
utils.evaluate(zero_r_model,

               X_train, X_test,

               y_train, y_test)
utils.evaluate(tree_model,

               X_train, X_test,

               y_train, y_test)
utils.evaluate(discretize_tree_model,

               X_train, X_test,

               y_train, y_test)