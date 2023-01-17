import numpy as np

import pandas as pd



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LinearRegression
data = pd.read_csv("../input/world-university-rankings/cwurData.csv")
data
data.drop('institution', axis=1, inplace=True)
data.drop('year', axis=1, inplace=True)
np.sum(data.isnull())
data.drop('broad_impact', axis=1, inplace=True)
encoder = LabelEncoder()



data['country'] = encoder.fit_transform(data['country'])

country_mappings = {index: label for index, label in enumerate(encoder.classes_)}
data
y = data['world_rank']

X = data.drop('world_rank', axis=1)
scaler = StandardScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
model = LinearRegression()

model.fit(X_train, y_train)
print(f"Model R^2: {model.score(X_test, y_test)}")