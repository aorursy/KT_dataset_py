import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
df.head()
df.describe()
columns = df.columns.tolist()
features = columns[1: -1]
target = columns[-1]
print(features, target)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], 
                                                    test_size=0.33, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
predict = model.predict(X_test)
# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(y_test, predict))
# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score(y_test, predict))

