import pandas_datareader as pdr
import matplotlib.pyplot as plt
import pandas as pd
import datetime 
#from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor # Our DEcision Tree classifier
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
# This line is necessary for the plot to appear in a Jupyter notebook
%matplotlib inline
# Control the default size of figures in this Jupyter notebook
%pylab inline
pylab.rcParams['figure.figsize'] = (14, 9)   # Change the size of plots
start_date = datetime.datetime(2007, 1, 1)
end_date = datetime.datetime(2019, 12, 7)
stock_data = pd.read_csv("../input/amazon/AMZN.csv") # load data from csv
print(stock_data)
#Now with Panas we have to convert this data into Dataframe
dataset = pd.DataFrame(stock_data)
dataset.head()
##Now we convert into csv
dataset.to_csv('TCS.csv')
## We have to read our CSV
data = pd.read_csv('TCS.csv')
data.head()
#Let's check NULL values
data.isnull().sum()

#Now see some correalations between data
import seaborn as sns
plt.figure(1 , figsize = (17 , 8))
cor = sns.heatmap(data.corr(), annot = True)

#Let's select our features
#lazem yb2a a5r 7aga (volume) lma kont bst5dm adj close kan bytl3 error index
#Now we have to divide data in Dependent and Independent variable


#We can see Date column in useul for our prediction but for simplicity we have to remove it because date format is not proper

#Now we have to split data in training and testing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
#Now we have to predict open price so this column is out dependent variable because open price depend on High,Low,Close,Last,Turnover etc..
x = data.loc[:,'High':'Volume']
y = data.loc[:,'Open']
print(x)
print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)
#Let's fit our DecisionTree Model
Classifier = DecisionTreeRegressor()
#Let's Fit our Data
Classifier.fit(x_train,y_train)
y_pred = Classifier.predict(x_test)
print("Y_Pred:",y_pred)
print("Y_Test:",y_test)
#Let's make a prediction on random day Data
#test = [[46.50 ,43.10 ,44.40, 44.45, 13889470.0]]
#prediction = Classifier.predict(test)
#prediction

#evaluate the model
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
rsquared = r2_score(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rsquared:",rsquared)
print("rmse:",rmse)
