import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import metrics



from sklearn.preprocessing import Imputer



from sklearn.tree import DecisionTreeClassifier







colnames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']



pima_df = pd.read_csv("../input/pima-indians-dataset/pima-indians-diabetes.csv", names= colnames)







pima_df[['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']] = pima_df[['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']].replace(0, np.NaN)



pima_df.head(50)
import os

os.listdir('../input/pima-indians-dataset')
import seaborn as sns



sns.pairplot(pima_df , hue='class' , diag_kind = 'kde')
# split dataset into inputs and outputs

values = pima_df.values

X = values[:,0:8]

y = values[:,8]

# fill missing values with mean column values

imputer = Imputer()

#transformed_X = imputer.fit_transform(X)



#Try following -



transformed_x = Imputer(missing_values='NaN', strategy='median', axis=0)  # override default behavior to replace Nan with Median



test_size = 0.30 # taking 70:30 training and test set

seed = 7  # Random numbmer seeding for reapeatability of the code

X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=test_size, random_state=seed)







dt_model = DecisionTreeClassifier(criterion = 'entropy' )

dt_model.fit(X_train, y_train)
dt_model.score(X_test , y_test)
y_predict = dt_model.predict(X_test)

print(metrics.confusion_matrix(y_test, y_predict))
dt_model = DecisionTreeClassifier(criterion = 'entropy',  max_depth = 8)

dt_model.fit(X_train, y_train)
dt_model.score(X_test , y_test)
y_predict = dt_model.predict(X_test)

print(metrics.confusion_matrix(y_test, y_predict))