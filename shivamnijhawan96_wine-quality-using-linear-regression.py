import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import train_test_split

wine_data = pd.read_csv('../input/winequality-red.csv')

wine_data.head()
wine_data.describe() 
correlation = wine_data.corr()



import matplotlib.pyplot as plt

fig = plt.subplots(figsize=(10,10))

sns.heatmap(correlation,vmax=1,square=True,annot=True,cmap='Blues')
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',

       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',

       'pH', 'sulphates', 'alcohol']

x = wine_data[features]

y = wine_data['quality']

#plotting features vs quality

sns.pairplot(wine_data,x_vars=features,y_vars='quality',kind='reg',size=7,aspect=0.5)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=3)
regressor = LinearRegression()

regressor.fit(x_train,y_train)
accuracy = regressor.score(x_test, y_test)

"Accuracy: {}%".format(int(round(accuracy * 100)))