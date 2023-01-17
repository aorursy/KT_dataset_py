# Load data and preview data
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = pd.read_csv("../input/census.csv")
data.head()
def convert_income(income):
    if (income == ' >50K'):
        return 1
    else:
        return 0

# convert categorical value to numerical value
data['c#income'] = data['c#income'].apply(convert_income)

# preview info about the dependent variable
data['c#income'].unique()
# define variables to X axis
data_x = data.drop('c#income', axis=1) # all columns except column "income"
data_x.head()
# define variables to Y axis
data_y = data['c#income'] 
type(data_y)

# preview the first items to identify which columns are categorical
#   in the results, we can verify that we have 8 categorical columns 
#   (workclass, final-weight, education, marital-status, occupation, relationship, race, sex and native-country)
data_x.head()
categorical_columns = ["workclass","education","marital-status","occupation","relationship","race","sex", "inative-country"]
# We will use the LabelEncoder to transform (encode) categorical columns into numeric columns
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for c in categorical_columns:
    data_x[c] = label_encoder.fit_transform(data_x[c].values)
# preview the first item with categorical columns encoded
data_x.head()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_x_scaled = pd.DataFrame(scaler.fit_transform(data_x), columns=data_x.columns)
print(type(data_x_scaled))
data_x_scaled.head()
from sklearn.model_selection import train_test_split
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x_scaled, data_y, test_size = 0.3) # 70% to train and 30% to test

print('Items to train: ' + str(data_x_train.shape))
print('Items to test: ' + str(data_x_test.shape))

dependent_columns_name = list(data_x.columns)
import tensorflow as tf
tf_x_columns = [tf.feature_column.numeric_column(key=c) for c in dependent_columns_name]
tf_x_columns
print(data_x_train.shape)
print(type(data_y_train))


data_x.head()
train_function = tf.estimator.inputs.pandas_input_fn(x = data_x_train, y = data_y_train, batch_size=32, num_epochs=None, shuffle=True)
classifier = tf.estimator.LinearClassifier(feature_columns=tf_x_columns)
classifier.train(input_fn=train_function, steps=10000)

# after that, our classifier are trained
predict_function = tf.estimator.inputs.pandas_input_fn(x = data_x_test, batch_size = 32, shuffle = False)
predictions = classifier.predict(input_fn=predict_function)
predictions_result = []
for p in predictions:
    predictions_result.append(p['class_ids'])
    
# using our trained classifier, we predict the class for unknown data (data_x_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(data_y_test, predictions_result)
score

# using our predictions, we evaluate how good is our classifier
# we got a good 82% score