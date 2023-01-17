import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


np.random.seed(1)

# 500 random integers between 0 to 50 
x = np.random.randint(0,50,1000)

# Positive Correlation
y = x + np.random.normal(0,10,1000)

np.corrcoef(x,y)




# plotting the positive correlation
import matplotlib
import matplotlib.pyplot as plt

plt.scatter(x,y)
plt.show()
# 500 random integers between 0 to 50 
a = np.random.randint(0,50,1000)

# Positive Correlation
b = 100 - a + np.random.normal(0,10,1000)

np.corrcoef(a,b)


# plotting the positive correlation
import matplotlib
import matplotlib.pyplot as plt

plt.scatter(a,b)
plt.show()

# 500 random integers between 0 to 50 
p = np.random.randint(0,50,1000)

# Positive Correlation
q = np.random.normal(0,10,1000)

np.corrcoef(p,q)


# plotting the positive correlation
import matplotlib
import matplotlib.pyplot as plt

plt.scatter(p,q)
plt.show()
df = pd.DataFrame({'a':np.random.randint(0,50,1000)})
df['b'] = df['a'] + np.random.normal(0,10,1000)
df['c'] = 100 - df['a'] + np.random.normal(0,10,1000)
df['d'] = np.random.normal(0,10,1000)

df.corr()
pd.scatter_matrix(df)
plt.show()