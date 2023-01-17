# Import libraries



import pandas as pd # For data manipulation

import numpy as np # For various mathematical computations, array manipulations

import matplotlib.pyplot as plt # Visualization

import seaborn as sns #Visualization
# Ignore warnings



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/PGA_Data_Historical.csv')

df.head(10)
df[df['Player Name'] == 'Tiger Woods'][df['Statistic'] == 'SG: Putting']
df[df['Player Name'] == 'Tiger Woods'][df['Statistic'] == 'Official World Golf Ranking']
keep_vars = [

    'SG: Off-the-Tee - (AVERAGE)',

    'SG: Approach-the-Green - (AVERAGE)',

    'SG: Around-the-Green - (AVERAGE)',

    'SG: Putting - (AVERAGE)',

    'Official World Golf Ranking - (AVG POINTS)'

]
data = df[df['Variable'].apply(lambda x: x in keep_vars)] # Define our main dataframe to work with as data
data['Value'] = data['Value'].astype('float64') # All values are given as strings, change to float64
data = pd.pivot_table(data=data, index=['Player Name', 'Season'], columns=['Statistic'], values='Value')
# Reorder and rename OWGR column



data['OWGR Avg Pts'] = data['Official World Golf Ranking'] # Put this column at the end

data = data.drop('Official World Golf Ranking', axis=1)
data.head(15)
data = data.dropna() # Drop NaN's
sns.pairplot(data=data)

plt.show()
# Import linear regression libraries



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



# Split features and target



X = data.iloc[: ,:-1]

y = data.iloc[:,-1:]



# Train test split



X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=12)

# Fit the model



lr = LinearRegression()

lr.fit(X_train, y_train)
print('training set r^2 score = ' + str(lr.score(X_train, y_train)))



print('testing set r^2 score = ' + str(lr.score(X_test, y_test)))
# Create list of predictions



lin_reg_preds = pd.DataFrame(lr.predict(X), columns=['OWGR Avg Pts - Prediction'], index=y.index)
# Plot and compare with the line y=x



sns.scatterplot(data=pd.concat([y,lin_reg_preds], axis=1), x='OWGR Avg Pts', y='OWGR Avg Pts - Prediction', label='Pred vs True') # scatter plot of prediction vs true

sns.lineplot(np.linspace(0,15,20),np.linspace(0,15,20), color='r', label='y=x') # plot the line y=x

plt.show()
# Import polynomial libraries



from sklearn.preprocessing import PolynomialFeatures



# Create polynomial features



degree = 2 # Start with 2

poly = PolynomialFeatures(degree, include_bias=False)



X_poly = poly.fit_transform(X) # No longer a pandas dataframe

y_poly = y # Still a pandas dataframe



X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(X_poly, y_poly, random_state=12)
# Fit the model



lr_poly = LinearRegression()

lr_poly.fit(X_poly_train, y_poly_train)
print('training set r^2 score = ' + str(lr_poly.score(X_poly_train, y_poly_train)))



print('testing set r^2 score = ' + str(lr_poly.score(X_poly_test, y_poly_test)))
# Create list of predictions

lin_poly_reg_preds = pd.DataFrame(lr_poly.predict(X_poly), columns=['OWGR Avg Pts - Poly_Prediction'], index=y.index) 



# Plot and compare with the line y=x

sns.scatterplot(data=pd.concat([y_poly,lin_poly_reg_preds], axis=1), x='OWGR Avg Pts', y='OWGR Avg Pts - Poly_Prediction', label='Pred vs True') # scatter plot of prediction vs true

sns.lineplot(np.linspace(0,15,20),np.linspace(0,15,20), color='r', label='y=x') # plot the line y=x

plt.show()