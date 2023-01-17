import pandas as pd

nyc_pumpkins = pd.read_csv("../input/new-york_9-24-2016_9-30-2017.csv")

cat_map = {

    'sml': 0,

    'med': 1,

    'med-lge': 2,

    'lge': 3,

    'xlge': 4,

    'exjbo': 5

}

nyc_pumpkins = nyc_pumpkins.assign(

    size=nyc_pumpkins['Item Size'].map(cat_map),

    price=nyc_pumpkins['High Price'] + nyc_pumpkins['Low Price'] / 2,

    size_class=(nyc_pumpkins['Item Size'].map(cat_map) >= 2).astype(int)

)

nyc_pumpkins = nyc_pumpkins.drop([c for c in nyc_pumpkins.columns if c not in ['size', 'price', 'size_class']], 

                                 axis='columns')

nyc_pumpkins = nyc_pumpkins.dropna()
import seaborn as sns

sns.regplot('size', 'price', data=nyc_pumpkins)
import numpy as np



class LinearRegression:

    def fit(self, X, y):

        self.betas = np.linalg.inv(X.T @ X) @ X.T @ y

        

    def predict(self, X):

        return X @ self.betas
prices = nyc_pumpkins.values[:, :1]

sizes = nyc_pumpkins.values[:, 1:2]



clf = LinearRegression()

clf.fit(prices, sizes)

predicted_sizes = np.round(clf.predict(prices))
outcome = pd.DataFrame(

    [sizes[:, 0], predicted_sizes[:, 0]]

).T.rename(columns={0: 'size', 1: 'predicted_size'})

outcome.head(10)
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



(outcome['size'] - outcome.predicted_size).value_counts().sort_index().plot.bar(

    title='$y - \hat{y}$'

)
pd.Series(

    np.abs((outcome['size'] - outcome.predicted_size).values) <= 1

).value_counts().plot.bar(title='Accuracy Within 1 Class')