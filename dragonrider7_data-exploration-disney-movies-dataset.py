import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#Read the dataset
df=pd.read_csv('/kaggle/input/disney-movies/disney_movies.csv', parse_dates=['release_date'], index_col=['release_date'])
df.head()
#Info
display(df.shape)
display(df.info())
df.describe()
earnings_genre = df[['total_gross', 'inflation_adjusted_gross', 'genre']].groupby(['genre']).mean().sort_values(by='inflation_adjusted_gross',ascending=False)
display(earnings_genre)
display(earnings_genre.plot(kind='bar'))
plot1 = sns.swarmplot(x = "genre", y = "total_gross", data = df, size = 5)

plot2 = sns.swarmplot(x = 'genre', y = "inflation_adjusted_gross", data = df, size = 5)
#Effect of mpaa_rating on earnings
earnings_rating = df[['mpaa_rating', 'total_gross', 'inflation_adjusted_gross']].groupby(['mpaa_rating']).mean().sort_values(by='inflation_adjusted_gross',ascending=False)
display(earnings_rating)
display(earnings_rating.plot(kind='bar'))
plot3 = sns.swarmplot(x = "mpaa_rating", y = "total_gross", data = df, size = 5)
plot3

plot4 = sns.swarmplot(x='mpaa_rating', y='inflation_adjusted_gross', data=df, size=5)
plot4
df.index
dates = df.index.values
inflation_adjusted = df['inflation_adjusted_gross']
gross = df['total_gross']
plt.rcParams['figure.figsize']=[16,8]
print(plt.scatter(x=dates, y=inflation_adjusted, color='green', alpha=0.9))
print(plt.scatter(x=dates, y=gross, color='lightblue', alpha = 0.5))
sns.lineplot(x=dates, y=inflation_adjusted)
sns.lineplot(x=dates, y=gross, alpha = 0.6)
x = df.reset_index()
y = x.inflation_adjusted_gross
x = x.drop(['total_gross','movie_title','inflation_adjusted_gross'], axis = 1)
x.head()
g = pd.get_dummies(x.genre)
g.head()
r = pd.get_dummies(x.mpaa_rating)
r.head()
x = pd.concat([x,g,r], axis = 1)
x.head()
x = x.drop(['genre','Western','R','mpaa_rating','release_date'], axis=1)
x.head()
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
xTrain,xTest,yTrain,yTest=tts(x,y,test_size=0.3)
Linreg=LinearRegression()
Linreg.fit(xTrain,yTrain)
y_pred=Linreg.predict(xTest)
p = pd.DataFrame(y_pred, columns=['Actual'])
p1 = np.asarray(yTest)
p2 = pd.DataFrame(p1, columns=['Pred'])
cm = pd.concat([p,p2], axis=1)
cm
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
#Linear Regression Training MSE
print((mean_squared_error(np.square(Linreg.predict(xTrain)),np.square(yTrain))))
#Linear Regression Testing MSE 
print((mean_squared_error(np.square(Linreg.predict(xTest)),np.square(yTest))))
#Linear Regression Mean Absolute Error Training
print((mean_absolute_error(np.square(Linreg.predict(xTrain)),np.square(yTrain))))
#Linear Regression Mean Absolute Error Testing
print((mean_absolute_error(np.square(Linreg.predict(xTest)),np.square(yTest))))
sns.lineplot(x=cm.index.values, y=cm.Actual, color='purple')
sns.lineplot(x=cm.index.values, y=cm.Pred, alpha = 0.6)