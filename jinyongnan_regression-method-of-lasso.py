import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
# Read the data
csv = pd.read_csv('/kaggle/input/boston-housing-dataset/HousingData.csv')
csv.head(10)
# Get columns contain nan values
nan_cols = []
for col in csv.columns[:-1]:
    if csv[col].hasnans:
        nan_cols.append(col)
print('Columns contain NAN values:',nan_cols)
# Fill each nan column with its median value
for col in nan_cols:
    csv[col].fillna(csv[col].median(),inplace=True)
csv.head(10)
# Get X and y
data = csv.values
X,y = data[:,:-1],data[:,-1]
# define model
model = Lasso()
# Set trainging method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=7)
# Set Grid for GridSearchCV
grid = {'alpha':[1, 0.1, 0.01, 0.001]}
# Define GridSearchCV
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)# -1 means using all processors
# Train the model
results = search.fit(X, y)
# Check out fit results
print("Best mae is: ",-1*results.best_score_)# the neg_mae has negative mae
print("Best alpha is: ",results.best_params_)
# Final model
model = Lasso(alpha=0.01)
model.fit(X,y)
# Print coefficients
model.coef_
