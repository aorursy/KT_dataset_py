import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_path = '/kaggle/input/titanic/train.csv'

train_data = pd.read_csv(train_path)

train_data
train_data['Pclass'].value_counts()
# The following function and plot are largely based on matplotlib's documentation.

# You can refer to it in the following link

# https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py



def autolabel(rects, ax):

    for rect in rects:

        height = rect.get_height()

        ax.annotate(f"{height}",

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),

                    textcoords="offset points",

                    ha='center', va='bottom')
classes = [train_data[train_data['Pclass'] == 1],

           train_data[train_data['Pclass'] == 2],

           train_data[train_data['Pclass'] == 3]]

class_labels = ['C1', 'C2', 'C3']



class_survided_count = [np.sum(_class['Survived']==True) for _class in classes]

class_not_survived_count = [np.sum(_class['Survived']==False) for _class in classes]



class_x_axis = np.arange(len(class_labels))

width = 0.40



class_fig, class_ax = plt.subplots()

class_survived_rects = class_ax.bar(x_axis - width/2, class_survided_count,

                        width, label='Surviving passangers')

class_not_survived_rects = class_ax.bar(x_axis + width/2, class_not_survived_count,

                            width, label='Non surviving passangers')

class_ax.set_ylabel('Survived')

class_ax.set_title('Amount of passangers by class')

class_ax.set_xticks(class_x_axis)

class_ax.set_xticklabels(class_labels)

class_ax.legend()



autolabel(class_survived_rects, class_ax)

autolabel(class_not_survived_rects, class_ax)



class_fig.tight_layout()



plt.show()
x_fare = train_data['Survived']

y_fare = train_data['Fare']



plt.scatter(x_fare, y_fare, alpha=0.3)

plt.show()
train_data['Embarked'].value_counts()
ports = [train_data[train_data['Embarked'] == 'S'],

           train_data[train_data['Embarked'] == 'C'],

           train_data[train_data['Embarked'] == 'Q']]

embarked_labels = ['Southampton', 'Cherbourg', 'Queenstown']



embarked_survided_count = [np.sum(port['Survived']==True) for port in ports]

embarked_not_survived_count = [np.sum(port['Survived']==False) for port in ports]



embarked_x_axis = np.arange(len(embarked_labels))

# width = 0.40 -- already declared on the previous bar chart



embarked_fig, embarked_ax = plt.subplots()

embarked_survived_rects = embarked_ax.bar(embarked_x_axis - width/2, embarked_survided_count,

                        width, label='Surviving passangers')

embarked_not_survived_rects = embarked_ax.bar(embarked_x_axis + width/2, embarked_not_survived_count,

                            width, label='Non surviving passangers')

embarked_ax.set_ylabel('Survived')

embarked_ax.set_title('Amount of passangers by embarking location')

embarked_ax.set_xticks(embarked_x_axis)

embarked_ax.set_xticklabels(embarked_labels)

embarked_ax.legend()



# def autolabel(rects) -- already declared on the previous bar chart



autolabel(embarked_survived_rects, embarked_ax)

autolabel(embarked_not_survived_rects, embarked_ax)



fig.tight_layout()



plt.show()
no_na_data = train_data.dropna()
features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']



filtered_train_data = no_na_data[features]
def one_hot_encode_feature(df, feature):

    feature_df = pd.get_dummies(df[feature])

    df.drop([feature], axis=1, inplace=True)

    return df.join(feature_df)
encoded_sex = one_hot_encode_feature(filtered_train_data, 'Sex')

encoded_sex
encoded_embarked = one_hot_encode_feature(encoded_sex, 'Embarked')

encoded_embarked
train_X = encoded_embarked.drop('Survived', axis=1)

train_X
train_y = encoded_embarked['Survived']

train_y
model = LogisticRegression(random_state=42)

model.fit(train_X, train_y)
test_path = '/kaggle/input/titanic/test.csv'

test_data = pd.read_csv(test_path)

test_data
test_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

filtered_test = test_data[test_features]

filtered_test
filled_test = filtered_test.fillna(0)

filled_test
encoded_sex_test = one_hot_encode_feature(filled_test, 'Sex')

encoded_sex_test
test_X = one_hot_encode_feature(encoded_sex_test, 'Embarked')

test_X
predictions = model.predict(test_X)

predictions
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})

output
output.to_csv('submission.csv', index=False)