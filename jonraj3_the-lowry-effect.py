import requests
import pandas as pd
from bs4 import BeautifulSoup
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import math

#Current Year, 2020
#url = 'https://www.espn.com/nba/player/gamelog/_/id/3012/kyle-lowry'

#2019
url = 'https://www.espn.com/nba/player/gamelog/_/id/3012/type/nba/year/2019'

#2018
#url = 'https://www.espn.com/nba/player/gamelog/_/id/3012/type/nba/year/2018'


soup = BeautifulSoup(requests.get(url).content, 'html.parser')
columns = ['Date','OPP','Result','MIN','FG','FG%','3PT','3P%','FT','FT%','REB','AST','BLK','STL','PF','TO', 'PTS']

all_data = []
for row in soup.select('.Table__TR'):
    tds = [td.get_text(strip=True, separator=' ') for td in row.select('.Table__TD')]
    if len(tds) != 17:
        continue
    all_data.append(tds)

df = pd.DataFrame(all_data, columns=columns)



df.info()
df.dtypes
df.PTS = pd.to_numeric(df.PTS)
df.AST = pd.to_numeric(df.AST)
df.MIN = pd.to_numeric(df.MIN)
df.head()

sns.lmplot(x="MIN", y="PTS", data=df)
plt.show()
sns.lmplot(x="MIN", y="AST", data=df)
plt.show()
print('Kyle Lowry Mean Points is %.3f' % df['PTS'].mean())
X = df[['MIN']]
Y = df[['PTS']]
#df = pd.get_dummies(df)

# Build linear regression model
lr_model = LinearRegression(fit_intercept=True, normalize=False)
lr_model.fit(X, Y)
sc = lr_model.score(X, Y)
print('R2 score: %.3f' % sc)
y_pred = lr_model.predict(X)
rmse = math.sqrt(mean_squared_error(Y, y_pred))
print('Root Mean Square Error is %.3f. Data STD is %.3f' % (rmse, Y.std()))
# Calculate Pearson correlation coefficient between the two variables
corr, _ = pearsonr(df['PTS'], df['AST'])
print('Pearson correlation coefficient: %.3f' % corr)
df.columns
y = df.PTS
lowry_features = ['MIN','FG%','3P%','FT%','TO']
X = df[lowry_features]
X.describe()
X.head()
lowry_model = DecisionTreeRegressor(random_state=1)
lowry_model.fit(X, y)
print("Making predictions for the following 5 stat lines:")
print(X.head())
print("The predictions are")
print(lowry_model.predict(X.head()))
y.head()
predicted_points = lowry_model.predict(X)
mean_absolute_error(y, predicted_points)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
lowry_model = DecisionTreeRegressor()
lowry_model.fit(train_X, train_y)
pts_predictions = lowry_model.predict(val_X)
print(mean_absolute_error(val_y, pts_predictions))
