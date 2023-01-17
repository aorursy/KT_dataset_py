import numpy as np

import pandas as pd



dataset = pd.read_csv("../input/train.csv")

testset = pd.read_csv("../input/test.csv")

dataset.head()
import matplotlib.pyplot as plt

import seaborn as sns

#TODO

sns.regplot(x = "Total Volume",y = "AveragePrice",data = dataset, order = 3)

plt.show()
#Feature Scaling

from sklearn.preprocessing import MinMaxScaler

m = MinMaxScaler()

m.fit_transform(dataset)

# X = dataset.iloc[:,0:11]  #independent columns

# y = dataset.iloc[:,-1] 

y = dataset['AveragePrice']

y = y.astype(float)

#Feature Engineering

# X = pd.get_dummies(columns=['year'],data=dataset)

X = dataset.drop(columns = ['id', 'AveragePrice', 'Total Bags', 'Total Volume'])

# totalvits = (dataset['4046'] * dataset['4225'] * dataset['4770'])*dataset['Total_Bags']

# X['totalvits'] = totalvits

bagspvol = dataset['Total Bags']/dataset['Total Volume']

X['bagspvol'] = bagspvol

X_test = testset.drop(columns = ['id', 'Total Bags', 'Total Volume'])

bagspvol = testset['Total Bags']/testset['Total Volume']

X_test['bagspvol'] = bagspvol
# Compute the correlation matrix

corr = dataset.corr(method='pearson')



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)



plt.show()
# # splitting data

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0,shuffle = True)
#Metrics

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error



def performance_metrics(y_true,y_pred):

    rmse = mean_squared_error(y_true,y_pred)

    r2 = r2_score(y_true,y_pred)

    

    return rmse, r2
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 500, min_samples_split = 2, min_samples_leaf = 1, max_features = 'sqrt', max_depth = None, bootstrap = False)

rf.fit(X, y)

y_pred = rf.predict(X_test)

print(y_pred)

# rmse,r2 = performance_metrics(y_test,y_pred)



# print("Root mean squared error:{} \nR2-score:{} ".format(rmse,r2))
answer = {"id" : testset["id"], "AveragePrice" : y_pred}

ans = pd.DataFrame(answer, columns = ["id", "AveragePrice"])

ans.to_csv("answer.csv", index = False)

ans