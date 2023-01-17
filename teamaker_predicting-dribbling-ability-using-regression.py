import pandas as pd

df = pd.read_csv('../input/PlayerAttributeData.csv')

df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

cols = ['Agility', 'Dribbling', 'Composure', 'Balance', 'Ball control']

df_personal = pd.read_csv('../input/PlayerPersonalData.csv')

df_personal.head()

df = pd.merge(df, df_personal, left_index=True, right_index=True, how='inner')

df = df[cols+['Name']]

df = df.dropna()

df = df.set_index('Name')

print (df.shape)

df.head()
import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = (20, 12)

f, axarr = plt.subplots(2,2)

for (col, (i, ax)) in zip(cols, enumerate(axarr.flatten())):

    ax.hist(df[col], bins=20)

    ax.set_title(col)

f.subplots_adjust(hspace=0.5)

plt.show()
def percent_values(series):

    mean = series.mean()

    std = series.std()

    sorted_vals = series.sort_values().tolist()

    low = min(range(len(sorted_vals)), key=lambda i: abs(sorted_vals[i]-(mean-std)))

    high = min(range(len(sorted_vals)), key=lambda i: abs(sorted_vals[i]-(mean+std)))

    return (high-low)/float(len(sorted_vals))



for col in cols:

    print ("percentage in {} is {}%".format(col, 100*percent_values(df[col])))
from pandas.plotting import scatter_matrix

scatter_matrix(df)
df.corr()
from sklearn import linear_model

regr = linear_model.LinearRegression()

x = df.drop(['Dribbling'], axis=1)

y = df['Dribbling'].reshape(-1,1)

regr.fit(x, y)
print ("r squared error : {}".format(regr.score(x,y)))

from sklearn.metrics import mean_squared_error

y_pred = regr.predict(x)

print ("mean squared error : {}".format(mean_squared_error(y, y_pred)))
from yellowbrick.regressor.residuals import ResidualsPlot

visualizer = ResidualsPlot(regr)

visualizer.fit(x,y)

g = visualizer.poof() 

g