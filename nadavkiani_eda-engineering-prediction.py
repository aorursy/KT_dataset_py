import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/zomato.csv', encoding='iso-8859-1')
df.head(2)
df.columns
df.shape
df.info()
df.describe()
sns.set(rc={'figure.figsize':(9,7)})

sns.countplot(x='Has Table booking',data=df,palette='viridis')
sns.countplot(x='Has Online delivery',data=df,palette='viridis',order=['Yes','No'])
sns.countplot(x='Is delivering now',data=df,palette='viridis',order=['Yes','No'])
df['Is delivering now'].value_counts()
df['Switch to order menu'].value_counts()
sns.countplot(x='Price range',data=df,palette='viridis')
sns.countplot(x='Rating text',data=df,palette='viridis')
sns.countplot(x='Rating color',data=df,palette='viridis')
sns.distplot(df['Aggregate rating'], hist=True,kde=False,bins=20,color = 'blue',hist_kws={'edgecolor':'black'})
df['Currency'].unique()
df['new cost'] = 0
df['Currency'].unique()
d = {'Botswana Pula(P)':0.095, 'Brazilian Real(R$)':0.266,'Dollar($)':1,'Emirati Diram(AED)':0.272,

    'Indian Rupees(Rs.)':0.014,'Indonesian Rupiah(IDR)':0.00007,'NewZealand($)':0.688,'Pounds(\x8c£)':1.314,

    'Qatari Rial(QR)':0.274,'Rand(R)':0.072,'Sri Lankan Rupee(LKR)':0.0055,'Turkish Lira(TL)':0.188}



df['new cost'] = df['Average Cost for two'] * df['Currency'].map(d) 
df.head(2)
sns.heatmap(data=df.corr(),cmap='coolwarm',annot=True)
df['new Rating'] = 0
mask1 = (df['Aggregate rating'] < 1)

mask2 = (df['Aggregate rating'] >= 1) & (df['Aggregate rating'] < 2)

mask3 = (df['Aggregate rating'] >= 2) &(df['Aggregate rating'] < 3)

mask4 = (df['Aggregate rating'] >= 3) & (df['Aggregate rating'] < 4)

mask5 = (df['Aggregate rating'] >= 4)



df['new Rating'] = df['new Rating'].mask(mask1, 'Low')

df['new Rating'] = df['new Rating'].mask(mask2, 'Medium -')

df['new Rating'] = df['new Rating'].mask(mask3, 'Medium')

df['new Rating'] = df['new Rating'].mask(mask4, 'Medium +')

df['new Rating'] = df['new Rating'].mask(mask5, 'High')
sns.set(rc={'figure.figsize':(18,6)})

sns.countplot(data=df,x='new Rating',order=['Low','Medium -','Medium','Medium +','High'])
sns.set(rc={'figure.figsize':(18,6)})

sns.scatterplot(data=df,x='Aggregate rating',y='Votes')

plt.ylim(0,1000)

plt.xlim(1,5)
sns.countplot(data=df,x='Aggregate rating',hue='Has Table booking',palette='viridis')
sns.countplot(data=df,x='Aggregate rating',hue='Has Online delivery',palette='viridis')
df.head(2)
new_df = df[['Has Table booking','Has Online delivery','Price range','Rating text','Votes','new cost','Aggregate rating']]

new_df.head()
new_df = pd.get_dummies(new_df, columns=['Has Table booking','Has Online delivery','Price range','Rating text'])
new_df.head()
X = new_df.drop(['Aggregate rating'], axis=1)

y = new_df['Aggregate rating']
from sklearn import model_selection

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# implementation of Linear Regression model using scikit-learn and K-fold for stable model

from sklearn.linear_model import LinearRegression

kfold = model_selection.KFold(n_splits=10)

lr = LinearRegression()

scoring = 'r2'

results = model_selection.cross_val_score(lr, X, y, cv=kfold, scoring=scoring)

lr.fit(X_train,y_train)

lr_predictions = lr.predict(X_test)

print('Coefficients: \n', lr.coef_,'\n')

print(results)

print(results.sum()/10)
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, lr_predictions))

print('MSE:', metrics.mean_squared_error(y_test, lr_predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lr_predictions)))
from sklearn.metrics import r2_score

print("R_square score: ", r2_score(y_test,lr_predictions))
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state = 42)

dtr.fit(X_train,y_train)

dtr_predictions = dtr.predict(X_test) 

results = model_selection.cross_val_score(dtr, X, y, cv=kfold, scoring='r2')

print(results)

print(results.sum()/10)



# R^2 Score

print("R_square score: ", r2_score(y_test,dtr_predictions))
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators = 100)

rfr.fit(X_train,y_train)

rfr_predicitions = rfr.predict(X_test) 

results = model_selection.cross_val_score(dtr, X, y, cv=kfold, scoring='r2')

print(results)

print(results.sum()/10)



# R^2 Score

print("R_square score: ", r2_score(y_test,rfr_predicitions))
from sklearn import ensemble

clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,

          learning_rate = 0.1, loss = 'ls')

clf.fit(X_train, y_train)

clf_predicitions = clf.predict(X_test) 

results = model_selection.cross_val_score(dtr, X, y, cv=kfold, scoring='r2')

print(results)

print(results.sum()/10)

print("R_square score: ", r2_score(y_test,clf_predicitions))
y = np.array([r2_score(y_test,lr_predictions),r2_score(y_test,dtr_predictions),r2_score(y_test,rfr_predicitions),

           r2_score(y_test,clf_predicitions)])

x = ["LinearRegression","RandomForest","DecisionTree","Grdient Boost"]

plt.bar(x,y)

plt.title("Comparison of Regression Algorithms")

plt.ylabel("r2_score")

plt.show()