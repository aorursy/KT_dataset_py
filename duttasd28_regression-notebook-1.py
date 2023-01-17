import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# This notebook was made on Kaggle. It might not work if input path is different. 

## Please change the _dataPath variable as per needs

### Change this if needed, can use "./scores.csv"



_dataPath = "../input/linear-regression-simple/scores.csv"   



data = pd.read_csv(_dataPath)
data.head()
# Check for null or missing values

data.isnull().any()
sns.set(style = 'whitegrid')

sns.pairplot(data, corner =True, kind = 'reg', height = 5);
from scipy.stats import pearsonr



statistics = pearsonr(data['Hours'], data['Scores'])

statistics
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
linearModel = LinearRegression()



## Reshaping is required for sklearn models.

## -1 indicates that sklearn should automatically calculate this value from the dataset



linearModel.fit(data['Hours'].values.reshape(-1, 1),

                data['Scores'].values.reshape(-1, 1));
linearModel.coef_, linearModel.intercept_
## Score of the values

r2_score(data['Scores'].values.reshape(-1, 1),

         linearModel.predict(data['Hours'].values.reshape(-1, 1)))
preds = linearModel.predict(data['Hours'].values.reshape(-1, 1))

plt.plot(data['Hours'],preds,'g')

plt.scatter(data['Hours'], data['Scores']);
plt.plot(data['Scores'] - preds.flatten(), 'mo');

plt.plot(np.zeros(25));
# Get the intercept and slope and make predictions

linearModel.predict(np.array([[9.25]]))  ## The square brackets are to convert the data into a 1x1 matrix