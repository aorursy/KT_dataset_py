import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
data = pd.read_csv('../input/pulsar_stars.csv')
data.isnull().any()
y_data = data['target_class']
data.columns
index_list = [' Mean of the integrated profile',

       ' Standard deviation of the integrated profile',

       ' Excess kurtosis of the integrated profile',

       ' Skewness of the integrated profile', ' Mean of the DM-SNR curve',

       ' Standard deviation of the DM-SNR curve',

       ' Excess kurtosis of the DM-SNR curve', ' Skewness of the DM-SNR curve']

x_data = data[index_list]
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=324)
tree_clf = DecisionTreeClassifier(max_leaf_nodes = 10, random_state=0)
tree_clf.fit(x_train,y_train)
predictions = tree_clf.predict(x_test)
accuracy_score(predictions, y_test)