#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#sklearn libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
def import_data ():
    df = pd.read_csv("../input/carprice.csv")
    df.columns = ["id", "type","minprice", "price", "maxprice", "rangeprice", "range", "gmp100", "speed_city", "speed_highway"]
    del df["rangeprice"]
    del df["id"]
    display (df.head())
    display(df.info())
    X = df["speed_highway"].values.reshape(-1,1)
    y = df["price"].values.reshape(-1,1)
    return (df, X, y)

df, X, y = import_data ()
display(pd.DataFrame(X).head())
display(pd.DataFrame(y).head())
display(df.describe())
def learn_linear_reg (X, y):
    reg = LinearRegression ()
    reg.fit(X, y)
    mse = mean_squared_error(y.ravel(), reg.predict(X))
    return reg, mse
def plot_poly_linear_reg (reg, transformer, df, mse, degree):
    plt.scatter (df.speed_city, df.price, color='blue', label='city speed')
    #plt.scatter (df.speed_highway, df.price, color='red', label='highway speed')
    plt.xlabel("speed")
    plt.ylabel("price")
    plt.legend()
    plt.grid(linestyle=":")
    plt.title ("price by speeds MSE={}".format(mse))
    xs = np.arange(min(df.speed_city)-10,max(df.speed_city)+10, 1).reshape(-1,1)
    preds = reg.predict(transformer.fit_transform(xs))
    plt.plot (xs, preds, color='red', label='fit line - Polynomial degree {}'.format(degree))
    plt.legend()
    plt.show()

def learn_and_plot_with_degree (degree):
    from sklearn.preprocessing import PolynomialFeatures
    degrees = []
    mses = []
    for d in range(1, degree):
        transformer = PolynomialFeatures(degree = d)
        x_poly = transformer.fit_transform(X)
        reg,mse = learn_linear_reg(x_poly,y)
        plot_poly_linear_reg(reg, transformer, df, mse, d)
        degrees.append(d)
        mses.append(mse)

    summary = pd.DataFrame({"degree" : degrees, "mse" : mses})
    return summary

summary = learn_and_plot_with_degree(25)
summary.plot("degree", "mse", title="MSE by degrees")
plt.grid(linestyle=":")
plt.xticks(np.arange(0,26))
plt.ylabel("mse")
plt.show()