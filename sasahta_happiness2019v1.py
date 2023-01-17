import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
df = pd.read_csv('/kaggle/input/world-happiness/2019.csv')
df.head()
correlations = df.corr()
sns.heatmap(correlations, annot = False, cmap="YlGnBu") # set annot to True to see correlation values
from sklearn.linear_model import LinearRegression
x = np.array(df['GDP per capita']).reshape((-1, 1)) # reshape x to a 156, 1 matrix
y = np.array(df['Score']) # do not need to reshape y; y.shape = 1, 156
model = LinearRegression() # create an instance of the linear regression class. This model does not do anything yet
model.fit(x, y)
print('slope:', model.coef_)
print('Intercept', model.intercept_)
y_pred = model.predict(x)
# lets see what our line looks like
sns.scatterplot(x = 'GDP per capita', y = 'Score', data = df)
sns.lineplot(df['GDP per capita'], y_pred, color='red')
x = df['GDP per capita']
theta_hat = (x.T @ y) / (x.T @ x) # analytical computation of theta
y_hat = theta_hat * x # y predicted by theta_hat

sns.scatterplot(x = 'GDP per capita', y = 'Score', data = df)
sns.lineplot(df['GDP per capita'], y_hat, color='red')

# predefined class works better; this method didn't account for an intercept, hence the poorfit