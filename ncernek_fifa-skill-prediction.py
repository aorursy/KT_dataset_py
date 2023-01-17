import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
players = pd.read_csv('../input/fifa18/full_filter_age_adjusted.csv',index_col='index')
players.shape


#remove the pluses and minuses from these numbers
import re

skill_columns = ['Acceleration',
       'Aggression', 'Agility', 'Balance', 'Ball control', 'Composure',
       'Crossing', 'Curve', 'Dribbling', 'Finishing', 'Free kick accuracy',
       'GK diving', 'GK handling', 'GK kicking', 'GK positioning',
       'GK reflexes', 'Heading accuracy', 'Interceptions', 'Jumping',
       'Long passing', 'Long shots', 'Marking', 'Penalties', 'Positioning',
       'Reactions', 'Short passing', 'Shot power', 'Sliding tackle',
       'Sprint speed', 'Stamina', 'Standing tackle', 'Strength', 'Vision',
       'Volleys']

def stripValue(value):
    if isinstance(value, str):
        return int(re.findall('\d+',value)[0])
    else:
        return None
    
# remove missing data
players.dropna(inplace=True)

for column in skill_columns:
    players[column] = players[column].apply(stripValue)
    
# remove missing data
players.dropna(inplace=True)

# create plotting objects
fig, ax = plt.subplots(figsize=(15,15))

for column in skill_columns:
    
    sns.distplot(
        players[column],
        bins=50, 
        axlabel=False,
        hist=False,
        kde_kws={"label": column},
        ax=ax,
    )
    
# these distributions are reasonable.
# most of them are skewed left, centered around 60
# many of them have a cluster of low skill
# explore correlations across all skills
correlations = players[['Overall','Age'] + skill_columns].corr()

fig, ax = plt.subplots(figsize=(20,20)) 
sns.heatmap(correlations,annot=False,cmap='coolwarm', linewidths=.5, ax=ax)
# explore correlations to overall and age
correlations = correlations.loc[skill_columns,['Overall','Age']]
correlations.sort_values('Overall',ascending=False)


# 1 predictor linear model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

y = players['Overall'].values.reshape(-1,1)
X = players['Reactions'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


lm = LinearRegression()
lm.fit(X_train,y_train)

coefficients = pd.DataFrame(lm.coef_,['Reactions'])
coefficients.columns = ['Coefficient']
display(coefficients)
#compare predicted to actual values
predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
# evaluate  model performance by calculating the residual sum of squares 
# and the explained variance score (R^2).
from sklearn import metrics

eval_1 = {
    'MAE': metrics.mean_absolute_error(y_test, predictions),
    'MSE': metrics.mean_squared_error(y_test, predictions),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, predictions))
}
eval_1
#make sure residuals are normally distributed and not too big
sns.distplot((y_test-predictions),bins=50);
y = players['Overall']#.values.reshape(-1,1)
X = players[['Reactions','Composure']]#.values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


lm = LinearRegression()
lm.fit(X_train,y_train)

coefficients = pd.DataFrame(lm.coef_,X.columns)
coefficients.columns = ['Coefficient']
display(coefficients)
#compare predicted to actual values
predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
eval_2 = {
    'MAE': metrics.mean_absolute_error(y_test, predictions),
    'MSE': metrics.mean_squared_error(y_test, predictions),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, predictions))
}
eval_2
#make sure residuals are normally distributed and not too big
sns.distplot((y_test-predictions),bins=50);
y = players['Overall']#.values.reshape(-1,1)
X = players[['Reactions','Composure','Short passing']]#.values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


lm = LinearRegression()
lm.fit(X_train,y_train)

coefficients = pd.DataFrame(lm.coef_,X.columns)
coefficients.columns = ['Coefficient']
display(coefficients)
#compare predicted to actual values
predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
eval_3= {
    'MAE': metrics.mean_absolute_error(y_test, predictions),
    'MSE': metrics.mean_squared_error(y_test, predictions),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, predictions))
}
eval_3
#make sure residuals are normally distributed and not too big
sns.distplot((y_test-predictions),bins=50);
y = players['Overall']
X = players[skill_columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm = LinearRegression()
lm.fit(X_train,y_train)

coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coefficient']
coeffecients.sort_values(by='Coefficient',ascending=False)
#compare predicted to actual values
predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
eval_all = {
    'MAE': metrics.mean_absolute_error(y_test, predictions),
    'MSE': metrics.mean_squared_error(y_test, predictions),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, predictions))
}
eval_all
#make sure residuals are normally distributed and not too big
sns.distplot((y_test-predictions),bins=50);
print('eval_1',eval_1)
print('eval_2',eval_2)
print('eval_3',eval_3)
print('eval_all',eval_all)
# We see that adding the second predictor nets positive.
diff = eval_1['RMSE'] - eval_2['RMSE']
value = diff/0.1 * 1000000
print(value)
value > 2500000
# adding a third predictor nets negative
diff = eval_2['RMSE'] - eval_3['RMSE']
value = diff/0.1 * 1000000
print(value)
value > 2500000