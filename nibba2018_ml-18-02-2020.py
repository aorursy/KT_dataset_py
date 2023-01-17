import numpy as np

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import seaborn as sns
appendix_data = pd.read_csv("../input/apndcts/apndcts.csv")

appendix_data.head()
data_y = appendix_data['class']

data_X = appendix_data.drop(['class'], axis=1)
data_X.head()
data_y.head()
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size = 0.3)
logistic = LogisticRegression()

logistic.fit(train_X, train_y)
logistic.score(test_X, test_y)
x = np.array([1,2,4,3,5])

y = np.array([1,3,3,2,5])



mean_x = np.mean(x)

mean_y = np.mean(y)



slope = np.sum((x - mean_x)*(y - mean_y))/np.sum((x - mean_x)**2)



constant = mean_y - slope*mean_x



prediction = slope*x + constant



rmse = np.sqrt(np.sum((prediction - y)**2)/x.size)



r_squared = np.sum((prediction - mean_y)**2)/np.sum((y - mean_y)**2)



print(f'Slope:      {slope}')

print(f'Constant:   {constant}')

print(f'Prediction: {prediction}')

print(f'RMSE:       {rmse}')

print(f'R Squared:  {r_squared}')
sns.regplot(x, y)