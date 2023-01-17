# Let's bring in pandas
import pandas as pd
from pandas import DataFrame
# Let's create a series
# A series is a one dimensional array like object. 

s = pd.Series([3,5,5,9,6])
s
s.head()
#ufo = pd.read_table('http://bit.ly/uforeports', sep=',')
ufo = pd.read_table('../input/qwer.txt', sep=',')
ufo.head(10)
ufo['State']
ufo['Location'] = ufo.City + ', ' + ufo.State
ufo.head()
ufo.describe()
ufo.shape
ufo.dtypes
ufo.columns
#ufo = pd.read_table('http://bit.ly/uforeports', sep=',')
ufo.head() 
ufo.drop('Colors Reported', axis=1, inplace=True)
ufo.head()
ufo.drop(['State','Time'], axis=1, inplace=True)
ufo.head()
#ufo = pd.read_table('http://bit.ly/uforeports', sep=',')
ufo = pd.read_table('../input/qwer.txt', sep=',')
ufo.head()
ufo.State.sort_values(ascending=False).head(25)
ufo.sort_values('City').head()
ufo.sort_values(['City','State']).head(25)
import numpy as np
x = np.array([1,2,3,4,5])
x
type(x)
x.ndim
x.shape
len(x)
x.size
x.dtype
x
x[0,]
x[4,]
x[:3]
np.save('x', a)
np.load("x.npy")
from sklearn import datasets
from sklearn import metrics
from sklearn.svm import SVC
ds = datasets.load_iris()
model = SVC()
model.fit(ds.data, ds.target)
print(model)
expected = ds.target
predicted = model.predict(ds.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
# Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np

# Prepare the data
x = np.linspace(0, 10, 100)

# Plot the data
plt.plot(x, x, label='linear')

# Add a legend
plt.legend()

# Show the plot
plt.show()
import matplotlib.pyplot as plt

X = [590,540,740,130,810,300,320,230,470,620,770,250]
Y = [32,36,39,52,61,72,77,75,68,57,48,48]

plt.scatter(X,Y)
plt.show()
import matplotlib.pyplot as plt

X = [590,540,740,130,810,300,320,230,470,620,770,250]
Y = [32,36,39,52,61,72,77,75,68,57,48,48]

plt.scatter(X,Y)
plt.title('Relationship Between Temperature and Mountain Dew Sales')
plt.show()
import matplotlib.pyplot as plt

X = [590,540,740,130,810,300,320,230,470,620,770,250]
Y = [32,36,39,52,61,72,77,75,68,57,48,48]

plt.scatter(X,Y)
plt.title('Relationship Between Temperature and Mountain Dew Sales')
plt.xlabel('Cans of Mountain Dew Sold')
plt.ylabel('Temperature in Fahrenheit')
plt.scatter(X, Y, s=80, c='green', marker='X')
plt.show()
import numpy as np
import pylab as pl
# pylab is a module in matplotlib that gets installed alongside matplotlib
# make an array of random numbers with a gaussian distribution with
# mean = 5.0
# rms = 3.0
# number of points = 1000
data = np.random.normal(5.0, 3.0, 1000)
# make a histogram of the data array
pl.hist(data)
# make plot labels
pl.xlabel('data')
pl.show()
from nltk.tokenize import sent_tokenize
sents = "Thanks for taking my courses. NLTK rocks!"
sent_tokenize(sents)
# Let's tokenize some words. 
from nltk.tokenize import word_tokenize
word_tokenize("I like Mikes courses.")
from nltk.corpus import stopwords
stopwords.words("english")