import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import sklearn.preprocessing

import sklearn.pipeline

import sklearn.linear_model



# read .csv file and return a pandas DataFrame object

s = pd.read_csv("../input/lab1.csv")

s.describe()

x, y= s.x, s.y

plt.scatter(x, y)
def fitridge(s, alpha):

    poly =  sklearn.preprocessing.PolynomialFeatures(10, include_bias=False)

    scale = sklearn.preprocessing.StandardScaler()

    ridge = sklearn.linear_model.Ridge(alpha=alpha)

    model = sklearn.pipeline.make_pipeline(poly, scale, ridge)

    model.fit(s[["x"]], s.y)

    return model
def plotmodel(modelpredict, plotrange, **kwargs):

    x = np.linspace(plotrange[0], plotrange[1], 200)

    plt.plot(x,modelpredict(np.expand_dims(x, -1)), **kwargs)

    plt.ylim((-35,30))



plotrange = (0, 15)

model_10 = fitridge(s, 0.01)

plotmodel(model_10.predict, [0,15])

plt.scatter(x, y, color='red')
model_under = fitridge(s, 0)

plotmodel(model_under.predict, [0,15])



model_hover = fitridge(s, 1000)

plotmodel(model_hover.predict, [0,15])



plotmodel(model_10.predict, [0,15])



plt.scatter(x, y, color='red')
def risk_emp(h,s):

    y = s.y

    y_ = h.predict(s[["x"]])

    return sum((y_ - y)**2)/len(y_)
ts, vs = s[:40], s[40:]

re_ts = []

re_vs = []

alphas = np.array([0.00001, 0.01, 0.1, 1, 1000])

for alpha in alphas:

    model_ts = fitridge(ts, alpha)

    re_ts.append(risk_emp(model_ts, ts))

    re_vs.append(risk_emp(model_ts, vs))



plt.plot(np.log10(alphas), re_ts)

plt.plot(np.log10(alphas), re_vs)
def truemodel(x):

    p = [1, 2, -1.3, 0.1, -0.001]

    res = 0

    # Horner's method to compute efficiently values of a polynom: a0 + x (a1 +x ( a2 + x ...)))

    for pi in reversed(p):

        res = pi + x * res

    return res





# generatey(x) is a function drawing random values from the unknown distribution Y|X=x.

# It generates an examples set of Y for a given X=x.

# This kind of examples set is not available in practical application.

def generatey(x):

    e = 10

    return truemodel(x) + e * np.random.randn(*x.shape)





# generate(n) is a function drawing random values from the unknown distribution (X,Y).

# It generates a set of n examples.

# This kind of examples set is what you have in practical application.

def generate(n):

  a = 0

  b = 15

  x = a + (b - a) * np.random.rand(n) # np.random.rand(d0, d1, ...) generates d0 * d1 * ... values drawn in U(0,1).

  y = generatey(x)

  return pd.DataFrame({"x":x, "y":y})

def compute_hx0s(x0,fit,k,n):

    result = []

    for i in range(k):

        ts = generate(n)

        hs = fit(ts)

        result.append(hs.predict(np.array([[x0]]))[0])

    return np.array(result)
x0, k, n = 7.5, 300, 40

print(np.var(compute_hx0s(x0,lambda s: fitridge(s, 1e-10),k,n)))
y = generatey(np.repeat(x0, k))

y_ = compute_hx0s(x0,lambda s: fitridge(s, 1e-10),k,n)

print(np.mean((y - y_)**2))
biases = []

variances = []

alphas = np.array([0.00001, 0.01,0.05, 0.1, 0.5, 1, 5, 10, 50, 1000])

for alpha in alphas:

    y = generatey(np.repeat(x0, k))

    y_ = compute_hx0s(x0,lambda s: fitridge(s, alpha),k,n)

    

    biases.append(np.mean((y - y_)**2))

    variances.append(np.var(y_))



plt.plot(np.log10(alphas), biases)

plt.plot(np.log10(alphas), variances)