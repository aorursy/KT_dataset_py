import pandas as pd

main_file_path = '../input/Iris.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)
# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
#select Target
f = {'Iris-virginica': 111, 'Iris-setosa': 222, 'Iris-versicolor': 333}
y = pd.Series([f[specie] for specie in data.Species if specie in f])

#select Predictors
predictors = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalLengthCm']
X = data[predictors]
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
#select Model
iris_model = DecisionTreeRegressor()

#split test and train data
train_X, val_X, train_y, val_y = train_test_split(X, y,test_size=0.30,random_state = 0)
print ("Training Data: ", train_X.shape)
print ("\n")
print ("Test Data: ", val_X.shape)
#train your model
iris_model.fit(train_X,train_y)
val_predictions = iris_model.predict(val_X)

#print(mean_absolute_error(val_y, val_predictions))

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
#print(confusion_matrix(val_y, val_predictions))
print('\n')
print(classification_report(val_y, val_predictions))
print('\n')
print('Accuracy score is: ', accuracy_score(val_y, val_predictions))

