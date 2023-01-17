import pandas as pd 

from sklearn.metrics import accuracy_score 

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv("../input/daily_weather.csv")

data.head()
# Let's look at the columns in the dataset 

data.columns
data.isnull().any().any()
data.isnull().sum() 
# Print the rows with missing values 

data[data.isnull().any(axis = 1)]
# We do not need to number the rows as Pandas provides its's own indexing 

del data['number']

data.columns
before_rows = data.shape[0]

data = data.dropna()

after_rows = data.shape[0]
print("The number of dropped rows are {}".format(before_rows - after_rows))
clean_data = data.copy() # New data frame to avoid confusion 

clean_data['high_humidity_label'] = (clean_data['relative_humidity_3pm'] > 24.99) * 1

print(clean_data['high_humidity_label'])
y = clean_data[['high_humidity_label']].copy()

y
clean_data['relative_humidity_3pm'].head()
y.head()
time = '9am'

features = list(clean_data.columns[clean_data.columns.str.contains(time)])



# we do not need relative humidity at 9am 

features.remove('relative_humidity_9am')



features
# Make the data of these features as X

X = clean_data[features].copy()

#X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 324)
# type(X_train)

# type(X_test)

# type(y_train)

# type(y_test)

# X_train.head()

# #y_train.describe()
y_train.describe()

X_train.describe()
humidity_classifier = DecisionTreeClassifier(max_leaf_nodes = 10, random_state = 0)

humidity_classifier.fit(X_train, y_train)
type(humidity_classifier)
predictions = humidity_classifier.predict(X_test)

type(predictions)
predictions[:10]

#predictions[:len(predictions)]
y_test[['high_humidity_label']][:10]

accuracy_score(y_test, y_pred = predictions)
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred = predictions)