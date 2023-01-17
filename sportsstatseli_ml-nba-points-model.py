from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

import pandas as pd

NBA = pd.read_csv("../input/seasons-1318/NBA.csv")
y = NBA.teamPTS



NBA_features = ['teamAST', 'teamTO', 'teamSTL', 'teamBLK', 'teamFGM', 'pace', 'poss', 'team2P%', 'team3P%', 'teamFTM', 'teamORB' ]

X = NBA[NBA_features]

X.describe()
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
NBA_model = DecisionTreeRegressor(random_state=1)

NBA_model.fit(train_X, train_y)

NBA_pred = NBA_model.predict(val_X)
print("The actual results")

print(y.head(20))

print("The predictions are")

print(NBA_pred[:20])
mean_absolute_error(val_y, NBA_pred)
from sklearn.ensemble import RandomForestRegressor
NBA_model_2 = RandomForestRegressor(random_state=1)

NBA_model_2.fit(train_X, train_y)

NBA_pred_2 = NBA_model_2.predict(val_X)

print("The actual results")

print(y.head(20))

print("The predictions are")

print(NBA_pred_2[:20])
print(mean_absolute_error(val_y, NBA_pred_2))