# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
FILEPATH = '../input/Toronto_apartment_rentals_2018.csv'
data = pd.read_csv(FILEPATH)
data.head()
df.info()
# get the number of missing data points per column

missing_values_count = data.isnull().sum()



# missing points in the first 10 

missing_values_count[0:10]
def get_space(pre_content, total_space_count = 30):



    current_space_count = total_space_count - len(pre_content)

    

    return pre_content + (" " * current_space_count)
def show_missing_percentage(current_df):

    

    total_cells = np.product(current_df.shape)

    total_missing = missing_values_count.sum()

    

    total_space_count = 20



    print(get_space("Total cells", total_space_count)+": {}".format(total_cells))

    print(get_space("Total missing cells", total_space_count)+": {}".format(total_missing))



    missing_percentage = (total_missing / total_cells)



    print(get_space("Missing Percentage", total_space_count)+": {:.2%}".format(missing_percentage))
show_missing_percentage(df)
data = data.dropna(axis=0)
data.Bathroom = data.Bathroom.astype(float).round().astype(int)
data['Price'] = data['Price'].str.replace(',', '')

data['Price'] = data['Price'].str.replace('$', '')

data['Price'] = data['Price'].astype(float).round().astype(int)
df = data
y = data.Price
features = ['Bedroom', 'Bathroom', 'Den', 'Lat', 'Long']
X = data[features]
# Building your model

# define model

model = DecisionTreeRegressor(random_state=1)
# Fit model

model.fit(X, y)
print('Making predictions for the following 5 houses')

print(X.head())



print('The predictions are')

print(model.predict(X.head()))
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size = 0.67, random_state = 34)
# Train and Test dataset size details

print("Train_x Shape :: ", train_X.shape)

print("Train_y Shape :: ", train_y.shape)

print("Test_x Shape :: ", test_X.shape)

print("Test_y Shape :: ", test_y.shape)
clf = model.fit(X, y)
y_predicted = clf.predict(test_X)

    

# print('original values')

# print(test_y)

# print('predicted')

# print(y_predicted)
print("accuracy", model.score(X, y) * 100)
from sklearn.neural_network import MLPClassifier



mlpc_model = MLPClassifier().fit(train_X, train_y)
from sklearn.linear_model import LinearRegression



lir_model = LinearRegression().fit(X, y)



# lir_model
from sklearn.ensemble import RandomForestClassifier



rf_model = RandomForestClassifier().fit(train_X, train_y)



# rf_model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()



knn_model = knn.fit(train_X, train_y)



# knn_model
from sklearn.linear_model import LogisticRegression



lor = LogisticRegression(solver = "liblinear")

lor_model = lor.fit(X,y)



# lor_model
from sklearn.tree import DecisionTreeClassifier



dt_model = DecisionTreeClassifier()

dt_model = dt_model.fit(train_X, train_y)



# dt_model
# from sklearn.svm import SVC



# svm_model = SVC(kernel = "linear").fit(train_X, train_y)



# svm_model
from sklearn.naive_bayes import GaussianNB



ganb = GaussianNB()

ganb_model = ganb.fit(train_X, train_y)



ganb_model
models = [

    mlpc_model,

    rf_model,

    knn_model,

    lor_model,

    dt_model,

#     svm_model,

    ganb_model

]



best_model_accuracy = 0

best_model = None



for model in models:

    

    model_name = model.__class__.__name__

    

    y_pred = model.predict(test_X)

    accuracy = accuracy_score(test_y, y_pred)

    

    print("-" * 30)

    print(model_name + ": " )

    

    if(accuracy > best_model_accuracy):

        best_model_accuracy = accuracy

        best_model = model_name

    

    print("Accuracy: {:.2%}".format(accuracy))
print("Best Model : {}".format(best_model))

print("Best Model Accuracy : {:.2%}".format(best_model_accuracy))