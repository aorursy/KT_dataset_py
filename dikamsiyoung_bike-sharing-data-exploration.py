!conda install -y gdown
import gdown
_ = gdown.download('https://drive.google.com/uc?id=1oS9HJDQhTcKb7lkrwf9ydbACMF8KGzdL', 'day.csv', quiet=False)
_ = gdown.download('https://drive.google.com/uc?id=1X9aCg-Pn6Kb7bVZQTGMsICxKig8qAo-h', 'hour.csv', quiet=False)
import pandas as pd
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)
df_hour = pd.read_csv('hour.csv')
df_hour.head()
df_day = pd.read_csv('day.csv')
df_day.head()
df_hour.isna().sum()
df_day.isna().sum()
df_hour.dtypes
df_day.dtypes
import seaborn as sns
sns.set(style="whitegrid")

import matplotlib.pyplot as plt
%matplotlib inline
g = sns.catplot(x="mnth", y="cnt", col="yr", data=df_day, legend=True, hue='yr', kind='box')
_ = g.fig.suptitle("Bike Rides in 2011 | 2012", y=1.08)
df_hour.corr()
df_day.corr()
fig, axs = plt.subplots(ncols=2, figsize=(18, 6))
g = sns.heatmap(df_hour.corr(), ax=axs[0], )
h = sns.heatmap(df_day.corr(), ax=axs[1])

_ = g.set_title('Correlation for Hours')
_ = h.set_title('Correlation for Day')
g = sns.catplot(x="mnth", y="temp", col="yr", data=df_day, legend=True, hue='yr', kind='box')
_ = g.fig.suptitle("Bike Rides in 2011 | 2012", y=1.08)
fig, axs = plt.subplots(ncols=2, figsize=(20,6))
_ = sns.boxplot(x=df_day['yr'], y=df_day['casual'], ax=axs[0])
_ = sns.boxplot(x=df_day.yr, y=df_day.registered, ax=axs[1])
def scatter(x,y, **kwargs):
    sns.scatterplot(x, y)

g = sns.FacetGrid(df_hour, col='weekday', row='yr', hue='yr', aspect=1)
_ = g.map(scatter, "hr", "cnt")
_ = g.add_legend()
_ = g.fig.suptitle("Bike Rides all Week", y=1.08)
_ = g.set_titles(row_template = 'yr {row_name}', col_template = 'day {col_name}')
plt.subplots_adjust(hspace=0.4, wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.FacetGrid(df_hour, col='weekday', aspect=1,)
_ = g.map(scatter, "hr", "casual")
_ = g.add_legend()
_ = g.fig.suptitle("Casual Rides per Week", y=1.08)
_ = g.fig.set_size_inches(15,5)
g = sns.FacetGrid(df_hour, col='weekday', aspect=1)
_ = g.map(scatter, "hr", "registered")
_ = g.add_legend()
_ = g.fig.suptitle("Registered Rides per Week", y=1.08)
_ = g.fig.set_size_inches(15,5)
g = sns.FacetGrid(df_hour[(df_hour['mnth']==12)], col='weekday', hue='yr', aspect=1)
_ = g.map(scatter, "hr", "cnt")
_ = g.add_legend()
_ = g.fig.suptitle("Bike Rides in other Months", y=1.08)
plt.subplots_adjust(hspace=0.4, wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.catplot(x='weekday', y='temp', col='yr', data=df_hour[df_hour['mnth']==12],kind='box', hue='yr')
plt.subplots_adjust(wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.FacetGrid(df_hour[(df_hour['mnth']==1)], col='weekday', hue='yr', aspect=1)
_ = g.map(scatter, "hr", "cnt")
_ = g.add_legend()
_ = g.fig.suptitle("Bike Rides in other Months", y=1.08)
plt.subplots_adjust(hspace=0.4, wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.catplot(x='weekday', y='temp', col='yr', data=df_hour[df_hour['mnth']==1],kind='box', hue='yr')
plt.subplots_adjust(wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.FacetGrid(df_hour[(df_hour['mnth']==2)], col='weekday', hue='yr', aspect=1)
_ = g.map(scatter, "hr", "cnt")
_ = g.add_legend()
_ = g.fig.suptitle("Bike Rides in other Months", y=1.08)
plt.subplots_adjust(hspace=0.4, wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.catplot(x='weekday', y='temp', col='yr', data=df_hour[df_hour['mnth']==2],kind='box', hue='yr')
plt.subplots_adjust(wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.FacetGrid(df_hour[(df_hour['mnth']==3)], col='weekday', hue='yr', aspect=1)
_ = g.map(scatter, "hr", "cnt")
_ = g.add_legend()
_ = g.fig.suptitle("Bike Rides in March", y=1.08)
plt.subplots_adjust(hspace=0.4, wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.catplot(x='weekday', y='temp', col='yr', data=df_hour[df_hour['mnth']==3],kind='box', hue='yr')
plt.subplots_adjust(wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.FacetGrid(df_hour[(df_hour['mnth']==4)], col='weekday', hue='yr', aspect=1)
_ = g.map(scatter, "hr", "cnt")
_ = g.add_legend()
_ = g.fig.suptitle("Bike Rides in April", y=1.08)
plt.subplots_adjust(hspace=0.4, wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.FacetGrid(df_hour[(df_hour['mnth']==4)&(df_hour.dteday == '2012-04-22')], col='weekday', hue='yr', aspect=1)
_ = g.map(scatter, "hr", "cnt")
_ = g.add_legend()
_ = g.fig.suptitle("Bike Rides on 22nd April", y=1.08)
_ = g.set_titles(row_template = 'yr {row_name}', col_template = 'day {col_name}')
plt.subplots_adjust(hspace=0.4, wspace=0.1)
_ = g.fig.set_size_inches(10,5)
g = sns.catplot(x='weekday', y='temp', col='yr', data=df_hour[df_hour['mnth']==3],kind='box', hue='yr')
plt.subplots_adjust(wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.FacetGrid(df_hour[(df_hour['mnth']==5)], col='weekday', hue='yr', aspect=1)
_ = g.map(scatter, "hr", "cnt")
_ = g.add_legend()
_ = g.fig.suptitle("Bike Rides in May", y=1.08)
_ = g.set_titles(row_template = 'yr {row_name}', col_template = 'day {col_name}')
plt.subplots_adjust(hspace=0.4, wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.FacetGrid(df_hour[(df_hour['mnth']==5)&(df_hour.dteday == '2012-05-28')], col='weekday', hue='yr', aspect=1)
_ = g.map(scatter, "hr", "cnt")
_ = g.add_legend()
_ = g.fig.suptitle("Bike Rides on 28th May", y=1.08)
_ = g.set_titles(row_template = 'yr {row_name}', col_template = 'day {col_name}')
plt.subplots_adjust(hspace=0.4, wspace=0.1)
_ = g.fig.set_size_inches(10,5)
g = sns.catplot(x='weekday', y='temp', col='yr', data=df_hour[df_hour['mnth']==5],kind='box', hue='yr')
plt.subplots_adjust(wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.FacetGrid(df_hour[(df_hour['mnth']==6)], col='weekday', hue='yr', aspect=1)
_ = g.map(scatter, "hr", "cnt")
_ = g.add_legend()
_ = g.fig.suptitle("Bike Rides in June", y=1.08)
plt.subplots_adjust(hspace=0.4, wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.catplot(x='weekday', y='temp', col='yr', data=df_hour[df_hour['mnth']==6],kind='box', hue='yr')
plt.subplots_adjust(wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.FacetGrid(df_hour[(df_hour['mnth']==7)], col='weekday', hue='yr', aspect=1)
_ = g.map(scatter, "hr", "cnt")
_ = g.add_legend()
_ = g.fig.suptitle("Bike Rides in July", y=1.08)
plt.subplots_adjust(hspace=0.4, wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.catplot(x='weekday', y='temp', col='yr', data=df_hour[df_hour['mnth']==7],kind='box', hue='yr')
plt.subplots_adjust(wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.FacetGrid(df_hour[(df_hour['mnth']==8)], col='weekday', hue='yr', aspect=1)
_ = g.map(scatter, "hr", "cnt")
_ = g.add_legend()
_ = g.fig.suptitle("Bike Rides in August", y=1.08)
plt.subplots_adjust(hspace=0.4, wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.catplot(x='weekday', y='temp', col='yr', data=df_hour[df_hour['mnth']==8],kind='box', hue='yr')
plt.subplots_adjust(wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.FacetGrid(df_hour[(df_hour['mnth']==9)], col='weekday', hue='yr', aspect=1)
_ = g.map(scatter, "hr", "cnt")
_ = g.add_legend()
_ = g.fig.suptitle("Bike Rides in September", y=1.08)
plt.subplots_adjust(hspace=0.4, wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.catplot(x='weekday', y='temp', col='yr', data=df_hour[df_hour['mnth']==9],kind='box', hue='yr')
plt.subplots_adjust(wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.FacetGrid(df_hour[(df_hour['mnth']==10)], col='weekday', hue='yr', aspect=1)
_ = g.map(scatter, "hr", "cnt")
_ = g.add_legend()
_ = g.fig.suptitle("Bike Rides in October", y=1.08)
plt.subplots_adjust(hspace=0.4, wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.catplot(x='weekday', y='temp', col='yr', data=df_hour[df_hour['mnth']==10],kind='box', hue='yr')
plt.subplots_adjust(wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.FacetGrid(df_hour[(df_hour['mnth']==11)], col='weekday', hue='yr', aspect=1)
_ = g.map(scatter, "hr", "cnt")
_ = g.add_legend()
_ = g.fig.suptitle("Bike Rides in November", y=1.08)
plt.subplots_adjust(hspace=0.4, wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.catplot(x='weekday', y='temp', col='yr', data=df_hour[df_hour['mnth']==11],kind='box', hue='yr')
plt.subplots_adjust(wspace=0.1)
_ = g.fig.set_size_inches(15,5)
g = sns.FacetGrid(df_hour, col='weekday', row='weathersit', aspect=1, margin_titles=True)
_ = g.map(scatter, "hr", "cnt")
_ = g.fig.suptitle("Bike Rides per Week with Respec to Weather", y=1.08)
_ = g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
plt.subplots_adjust(hspace=0.4, wspace=0.1)
_ = g.fig.set_size_inches(15,5,)
X = df_day.drop(columns=['cnt', 'casual', 'registered', 'dteday'])
y = df_day['cnt']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
_ = regression.fit(X_train, y_train)
y_pred = regression.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.head(10)
df.head(50).plot(kind='bar', figsize=(20,8))
from sklearn import metrics
metrics.mean_absolute_error(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)
import numpy as np
np.sqrt(metrics.mean_squared_error(y_test, y_pred))
X1 = df_day.drop(columns=['weathersit', 'dteday', 'yr', 'instant'])
y1 = df_day['weathersit']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25, random_state=1)
from sklearn import tree
dsTree = tree.DecisionTreeClassifier()
_ = dsTree.fit(X1_train, y1_train)
from sklearn import ensemble
rfClf = ensemble.RandomForestClassifier()
_ = rfClf.fit(X1_train, y1_train)
gbClf = ensemble.GradientBoostingClassifier()
_ = gbClf.fit(X1_train, y1_train)
ds_pred = dsTree.predict(X1_test)
ds_accuracy = metrics.accuracy_score(y1_test, ds_pred)
ds_accuracy
rf_pred = rfClf.predict(X1_test)
rf_accuracy = metrics.accuracy_score(y1_test, rf_pred)
rf_accuracy
gb_pred = gbClf.predict(X1_test)
gb_accuracy = metrics.accuracy_score(y1_test, rf_pred)
gb_accuracy
from sklearn.metrics import classification_report
print('.....Decision Tree Classification Report......')
print(classification_report(y1_test, ds_pred))
print('.....Random Forest Classification Report......')
print(classification_report(y1_test, rf_pred))
print('.....Gradient Boosting Classification Report......')
print(classification_report(y1_test, gb_pred))