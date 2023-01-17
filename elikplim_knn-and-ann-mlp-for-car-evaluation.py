import numpy

import pandas

from keras.utils import np_utils

import matplotlib.pyplot as plt

from keras.utils import to_categorical





from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score







from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.constraints import maxnorm







# fix random seed for reproducibility

seed = 7

numpy.random.seed(seed)
# load dataset

dataframe = pandas.read_csv(r"../input/car_evaluation.csv")



# Assign names to Columns

dataframe.columns = ['buying','maint','doors','persons','lug_boot','safety','classes']



# Encode Data

dataframe.buying.replace(('vhigh','high','med','low'),(1,2,3,4), inplace=True)

dataframe.maint.replace(('vhigh','high','med','low'),(1,2,3,4), inplace=True)

dataframe.doors.replace(('2','3','4','5more'),(1,2,3,4), inplace=True)

dataframe.persons.replace(('2','4','more'),(1,2,3), inplace=True)

dataframe.lug_boot.replace(('small','med','big'),(1,2,3), inplace=True)

dataframe.safety.replace(('low','med','high'),(1,2,3), inplace=True)

dataframe.classes.replace(('unacc','acc','good','vgood'),(1,2,3,4), inplace=True)
print("dataframe.head: ", dataframe.head())
print("dataframe.describe: ", dataframe.describe())
plt.hist((dataframe.classes))

dataframe.hist()
dataset = dataframe.values





X = dataset[:,0:6]

Y = numpy.asarray(dataset[:,6], dtype="S6")





# Split Data to Train and Test

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2)





# create model

knn = KNeighborsClassifier()



knn.fit(X_Train, Y_Train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=2, weights='uniform')



predictions = knn.predict(X_Test)



score = accuracy_score(Y_Test, predictions)

print(score)

# create model

model = Sequential()

model.add(Dense(25, input_dim=6, init='uniform', activation='relu'))

model.add(Dense(30, init='uniform', activation='relu', kernel_constraint=maxnorm(3)))

model.add(Dropout(0.2))

model.add(Dense(10, init='uniform', activation='relu'))

model.add(Dense(1, init='uniform', activation='relu'))



# Compile model

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])



# Fit the model

model.fit(X_Train, Y_Train, epochs=600, batch_size=10)



# Evaluate the model

scores = model.evaluate(X_Test, Y_Test)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
