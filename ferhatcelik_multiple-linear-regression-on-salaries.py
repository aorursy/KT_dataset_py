import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sympy as sym
sym.init_printing(use_latex=True, pretty_print=True)
def import_data ():
    df = pd.read_csv("../input/salaries.csv",sep=",")
    x0 = df.experience.values.reshape(-1,1)
    x1 = df.age.values.reshape(-1,1)
    x = np.concatenate([x0, x1], axis=1)
    y = df.salary.values.reshape(-1, 1)
    return (df,x,y)

df, X, y = import_data ()
display(df.head())
display(df.info())    
display(pd.DataFrame(np.concatenate([X, y], axis=1)))
def print_formulas ():
    # linear function
    # y = b0 + b1.x1 + b2.x2 + ... + bn.xn
    b0,b1,b2,salary,experience,age= sym.symbols("b0,b1,b2,salary,experience,age")
    simple_linear_reg = b0 + (b1 * experience)
    multi_linear_reg = b0 + (b1 * experience) + (b2 * age)

    # linear regression functions
    display(simple_linear_reg)
    display(multi_linear_reg)
    
print_formulas()
def learn (x,y):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(x, y)
    print ("intercept=", reg.intercept_)
    print ("coeffs   =", reg.coef_)
    return reg

reg = learn(X,y)
def plot (title, x,y):
    plt.scatter(x,y)
    plt.xticks(np.arange(min(x), max(x)+1, 1))
    plt.grid(linestyle=":")
    plt.xlabel("x {}".format(title))
    plt.ylabel("y {}".format("salary"))
    plt.show()
    
experiences = X[:, 0]
ages = X[:, 1]
plot ("experience (year)", experiences,y)
plot ("age", ages,y)

def plot_heatmap ():
    x0s = []
    x1s = []
    ys = []
    for i in np.linspace(0, 25, 26):
        for j in np.linspace(20, 50, 31):
            x0s.append(i)
            x1s.append(j)
            prediction = reg.predict([[i, j]]).ravel()[0]
            ys.append(prediction)
    dfPredictions = pd.DataFrame({"exp" : x0s, "age" : x1s, "salary" : ys})
    dfPivot = dfPredictions.pivot(index= "exp", columns="age", values="salary")

    ax = plt.figure(figsize=(20,10))
    sns.heatmap(dfPivot, annot=False, fmt="0.2f", center=5000, linewidths=1, linecolor='black')
    plt.title("Salaries by Experience and Age")
    plt.xticks(rotation=90)
    plt.xlabel("x0 - Experience")
    plt.ylabel("x1 - Age")
    plt.show()

plot_heatmap()
