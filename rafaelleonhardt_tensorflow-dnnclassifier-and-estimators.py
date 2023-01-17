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

data_x.head()
import tensorflow as tf

def create_bucket_categorical(column_name):
    return tf.feature_column.categorical_column_with_hash_bucket(key = column_name, hash_bucket_size=100)

workclass = create_bucket_categorical('workclass')
education = create_bucket_categorical('education')
marital_status = create_bucket_categorical('marital-status')
occupation = create_bucket_categorical('occupation')
relationship = create_bucket_categorical('relationship')
race = create_bucket_categorical('race')
country = create_bucket_categorical('inative-country')
sex = tf.feature_column.categorical_column_with_vocabulary_list(key='sex', vocabulary_list=[' Male', ' Female'])
def create_embedded(column, column_dimension):
    return tf.feature_column.embedding_column(column, dimension=column_dimension)

workclass_embedded = tf.feature_column.embedding_column(workclass, len(data['education'].unique()))
education_embedded = create_embedded(education, len(data['education'].unique()))
marital_status_embedded = create_embedded(marital_status, len(data['marital-status'].unique()))
occupation_embedded = create_embedded(occupation, len(data['occupation'].unique()))
relationship_embedded = create_embedded(relationship, len(data['relationship'].unique()))
race_embedded = create_embedded(race, len(data['race'].unique()))
country_embedded = create_embedded(country, len(data['inative-country'].unique()))
sex_embedded = create_embedded(sex, len(data['sex'].unique()))
def create_numerical(column_name):
    return tf.feature_column.numeric_column(key=column_name)

age = create_numerical('age')
final_weight = create_numerical('final-weight')
education_num = create_numerical('education-num')
capital_gain = create_numerical('capital-gain')
capital_loos = create_numerical('capital-loos')
hour = create_numerical('hour-per-week')
columns = [age, workclass_embedded, final_weight, education_embedded, education_num, marital_status_embedded, 
           occupation_embedded, relationship_embedded, race_embedded, sex_embedded, capital_gain, capital_loos, hour, country_embedded]
from sklearn.model_selection import train_test_split
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y, test_size = 0.3) # 70% to train and 30% to test

print('Items to train: ' + str(data_x_train.shape))
print('Items to test: ' + str(data_x_test.shape))

print(data_x_train.shape)
print(type(data_y_train))


data_x.head()
train_function = tf.estimator.inputs.pandas_input_fn(x = data_x_train, y = data_y_train, batch_size=32, num_epochs=None, shuffle=True)
classifier = tf.estimator.DNNClassifier(hidden_units=[8,8], feature_columns=columns, n_classes=2)
classifier.train(input_fn=train_function, steps=10000)

# after that, our classifier are trained
predict_function = tf.estimator.inputs.pandas_input_fn(x = data_x_test, batch_size = 32, num_epochs=1, shuffle = False)
predictions = classifier.predict(input_fn=predict_function)
predictions_result = []
for p in predictions:
    predictions_result.append(p['class_ids'])

    # using our trained classifier, we predict the class for unknown data (data_x_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(data_y_test, predictions_result)
score

# using our predictions, we evaluate how good is our classifier
# we got a 76% score