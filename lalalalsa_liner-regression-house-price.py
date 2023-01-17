import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import os
print(os.listdir("../input"))

class housePrice:
    def readData(self):
        data = pd.read_csv('../input/kc_house_data.csv')
        return data
    def ranking(self,ranks, names, order=1):
        minmax = MinMaxScaler()
        ranks = minmax.fit_transform(order * np.array([ranks]).T).T[0]
        ranks = map(lambda x: round(x, 2), ranks)
        return dict(zip(names, ranks))
    def linerRegression(self):
        data = self.readData()
        dataset = data.drop(['id', 'date'], axis=1)
        X = dataset.iloc[:, 1:].values
        y = dataset.iloc[:, 0].values
        colnames = dataset.columns
        model = LinearRegression()
        model.fit(X, y)
        accuracy = model.score(X, y)
        y_pred = model.predict(X)
        return model,y_pred,y,accuracy

if __name__ == '__main__':
    housePrice = housePrice()
    model,y_pred,y,accuracy= housePrice.linerRegression()
    print('features weight of liner regression')
    colnames = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
    feature_weight = housePrice.ranking(np.abs(model.coef_), colnames)
    meanplot = pd.DataFrame(list(feature_weight.items()), columns=['Feature', 'Mean Ranking'])
    print(meanplot)
sns.factorplot(x="Mean Ranking", y="Feature", data=meanplot, kind="bar",size=14, aspect=1.9, palette='coolwarm')
plt.show()
# compare pred values with true values
accuracy = int(accuracy*100)
print('the model accuracy is %s%%'%accuracy)
plt.figure()
plt.plot(range(len(y_pred[0:50])), y_pred[0:50], 'r', label="predict")
plt.plot(range(len(y_pred[0:50])), y[0:50], 'b', label="true value")
plt.legend(loc="upper right")
plt.xlabel("the number of sales")
plt.ylabel('value of sales')
plt.show()