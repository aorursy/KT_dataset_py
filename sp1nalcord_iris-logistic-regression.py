import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.metrics import plot_confusion_matrix

import warnings  

warnings.filterwarnings('ignore')

data = pd.read_csv('../input/iris/Iris.csv')

data.head()
# Check for null values

data.isnull().sum()
count = data.Species.value_counts()

sns.countplot(x='Species', data=data)

plt.show()
del data['Id']

x = data.iloc[:, :-1]

y = data.iloc[:, -1]



sns.pairplot(data, hue="Species", kind="reg")

plt.show()

density = sns.PairGrid(data, hue="Species")

density = density.map_diag(sns.kdeplot, lw=3, shade=True)

density = density.map_offdiag(sns.kdeplot, lw=1, legend=True)

plt.show()
sns.heatmap(data.corr(), annot=True).set_title("Correlation of attributes (petal length,width and sepal "

                                                        "length,width) among Iris species")

plt.show()
# Splitting into test and training.

x_train, x_test, y_train, y_test = train_test_split(x, y,

                                                    test_size=0.2,

                                                    random_state=40)



# Model.

model = LogisticRegression()

model.fit(x_train, y_train)

# Predicting the test data.

predictions = model.predict(x_test)

# Checking performance of the model.

report = classification_report(y_test, predictions)

score = accuracy_score(y_test, predictions)

print(f'Prediction Score: {predictions}\n\nClassification Report:{report}\nAccuracy Score: {score}')
plot_confusion_matrix(model, x_test, y_test)

plt.show()