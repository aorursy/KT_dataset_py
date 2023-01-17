import numpy as np, pandas as pd, seaborn as sns

from sklearn import linear_model

import matplotlib.pyplot as plt

import math
train_df = pd.DataFrame.from_csv("../input/train.csv")

print(train_df.columns)

train_df.head()
sns.distplot(train_df['SalePrice'], kde=True, rug=True)
numericalFeatures = ['YrSold', 'TotalBsmtSF', 'GrLivArea', 'SalePrice']



sns.pairplot(train_df[numericalFeatures], size = 2.5)

plt.show()
categoricalFeats = ['Street', 'Utilities', 'SaleCondition', 'BldgType', 'Neighborhood', 'CentralAir']





for feat in categoricalFeats:

    train_df.sort_values(by=[feat], axis=0, inplace=True)

    sns.barplot(x=train_df[feat], y=train_df['SalePrice'])

    plt.show()
def residualPlotterSingleFeature(model, train_df, numericalFeatures):

    for feat in numericalFeatures:

        model  = model.fit(np.array(train_df[feat]).reshape(-1,1)[:1000], list(train_df["SalePrice"])[:1000])



        actuals = list(train_df["SalePrice"])[1000:]

        predictions = model.predict(np.array(train_df[feat]).reshape(-1,1)[1000:])

        residuals = [float(abs(predictions[i] - actuals[i])) for i in range(len(predictions))]

        

        print("Absolute mean error : %f" %(sum(residuals)/len(residuals)))

        

        sq_err = [item**2 for item in residuals]

        print("Root mean squared error : %f" %math.sqrt(sum(sq_err)/len(sq_err)))

        

        df = pd.DataFrame({"residuals":residuals, "actuals":actuals})



        df.plot.scatter(x = "actuals", y="residuals")

        plt.title('residual plot when trained using %s' %feat)

        plt.show()

model = linear_model.LinearRegression(fit_intercept=True)

residualPlotterSingleFeature(model, train_df, numericalFeatures)
model = linear_model.Ridge(fit_intercept=True)

residualPlotterSingleFeature(model, train_df, numericalFeatures)
model = linear_model.Lasso(fit_intercept=True)

residualPlotterSingleFeature(model, train_df, numericalFeatures)