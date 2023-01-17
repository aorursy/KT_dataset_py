# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read house_data and take a look at data contents

data = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')

data.head(5)
data.shape
# some stats

data.describe()
# data preprocessing

# check missing data for each columns

data.apply(lambda col:sum(col.isnull())/col.size)
# check duplicates

data[data.duplicated()]
# correlation between varibales

corrmat = data.corr() 

  

f, ax = plt.subplots(figsize =(6, 6)) 

sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 



# sqft_living has strong correlation with grade,price, bathrooms,bedrooms

# Meanwhile, grade, price, bathrooms, bedrooms also has some positive correlation with each other

# In order to prevent multicollinearity, we shouldn't use them together as indenpendent variable
# data visualization helps to understand the relationship between variables

sns.scatterplot(x='bedrooms',y='sqft_living',data = data)
sns.scatterplot(x='bathrooms',y='sqft_living',data = data)
# find the data trend line

sns.regplot(x='bathrooms',y='sqft_living',data = data,line_kws={"color":"r","alpha":0.7,"lw":2})
sns.scatterplot(x='price',y='sqft_living',data = data, sizes=(1000,10))
sns.regplot(x='price',y='sqft_living',data = data, line_kws={"color":"r","alpha":0.7,"lw":3})
sns.regplot(x='grade',y='sqft_living',data = data,line_kws={"color":"r","alpha":0.7,"lw":3})
# X -> sqft_above 

X = data[['sqft_above']]

y = data['sqft_living']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=100)

print(X_train.shape)

print(X_test.shape)



from sklearn.linear_model import LinearRegression

model = LinearRegression()

output_model = model.fit(X_train, y_train)

R_sq = model.score(X_train, y_train)

print("R-Square: ", R_sq)

print("Intercept: ", model.intercept_)

print("Coeficient: ", model.coef_)
# X -> sqft_living15

X = data[['sqft_living15']]

y = data['sqft_living']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=100)

print(X_train.shape)

print(X_test.shape)

model = LinearRegression()

output_model = model.fit(X_train, y_train)

R_sq = model.score(X_train, y_train)

print("R-Square: ", R_sq)

print("Intercept: ", model.intercept_)

print("Coeficient: ", model.coef_)
X = data[['price']]

y = data['sqft_living']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=100)

print(X_train.shape)

print(X_test.shape)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

output_model = model.fit(X_train, y_train)

R_sq = model.score(X_train, y_train)

print("R-Square: ", R_sq)

print("Intercept: ", model.intercept_)

print("Coeficient: ", model.coef_)
# X -> bathrooms 

X = data[['bathrooms']]

y = data['sqft_living']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=100)

print(X_train.shape)

print(X_test.shape)



model = LinearRegression()

output_model = model.fit(X_train, y_train)

R_sq = model.score(X_train, y_train)

print("R-Square: ", R_sq)

print("Intercept: ", model.intercept_)

print("Coeficient: ", model.coef_)
# X -> grade 

X = data[['grade']]

y = data['sqft_living']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=100)

print(X_train.shape)

print(X_test.shape)



model = LinearRegression()

output_model = model.fit(X_train, y_train)

R_sq = model.score(X_train, y_train)

print("R-Square: ", R_sq)

print("Intercept: ", model.intercept_)

print("Coeficient: ", model.coef_)
# X --> price&bathrooms

X = data[['price','bathrooms']]

y = data['sqft_living']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=100)

print(X_train.shape)

print(X_test.shape)



model = LinearRegression()

output_model = model.fit(X_train, y_train)

R_sq = model.score(X_train, y_train)

print("R-Square: ", R_sq)

print("Intercept: ", model.intercept_)

print("Coeficient: ", model.coef_)
# X --> grade&bathrooms

X = data[['bathrooms','grade']]

y = data['sqft_living']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=100)

print(X_train.shape)

print(X_test.shape)



model = LinearRegression()

output_model = model.fit(X_train, y_train)

R_sq = model.score(X_train, y_train)

print("R-Square: ", R_sq)

print("Intercept: ", model.intercept_)

print("Coeficient: ", model.coef_)
# X --> 'sqft_above','price'

X = data[['sqft_above','price']]

y = data['sqft_living']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=100)

print(X_train.shape)

print(X_test.shape)



model = LinearRegression()

output_model = model.fit(X_train, y_train)

R_sq = model.score(X_train, y_train)

print("R-Square: ", R_sq)

print("Intercept: ", model.intercept_)

print("Coeficient: ", model.coef_)
# X --> 'sqft_above','price','bathrooms','bedrooms'

X = data[['sqft_above','price','bathrooms','bedrooms']]

y = data['sqft_living']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=100)

print(X_train.shape)

print(X_test.shape)



model = LinearRegression()

output_model = model.fit(X_train, y_train)

R_sq = model.score(X_train, y_train)

print("R-Square: ", R_sq)

print("Intercept: ", model.intercept_)

print("Coeficient: ", model.coef_)


X = data[['sqft_above','price','bathrooms','bedrooms','grade','floors','sqft_living15']]

y = data['sqft_living']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=100)

print(X_train.shape)

print(X_test.shape)



model = LinearRegression()

output_model = model.fit(X_train, y_train)

R_sq = model.score(X_train, y_train)

print("R-Square: ", R_sq)

print("Intercept: ", model.intercept_)

print("Coeficient: ", model.coef_)
cg = sns.clustermap(corrmat, cmap ="YlGnBu", linewidths = 0.1); 

plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0)  

cg 
k = 10 

  

cols = corrmat.nlargest(k, 'sqft_living')['sqft_living'].index 

  

cm = np.corrcoef(data[cols].values.T) 

f, ax = plt.subplots(figsize =(12, 10)) 

  

sns.heatmap(cm, ax = ax, cmap ="YlGnBu", 

            linewidths = 0.1, yticklabels = cols.values,  

                              xticklabels = cols.values) 
X = data[['sqft_above','grade','sqft_living15','bathrooms','bedrooms']]

y = data['sqft_living']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=100)

print(X_train.shape)

print(X_test.shape)



model = LinearRegression()

output_model = model.fit(X_train, y_train)

R_sq = model.score(X_train, y_train)

print("R-Square: ", R_sq)

print("Intercept: ", model.intercept_)

print("Coeficient: ", model.coef_)


X = data[['sqft_above','price','bathrooms','bedrooms','grade','sqft_basement','sqft_living15']]

y = data['sqft_living']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=100)

print(X_train.shape)

print(X_test.shape)



model = LinearRegression()

output_model = model.fit(X_train, y_train)

R_sq = model.score(X_train, y_train)

print("R-Square: ", R_sq)

print("Intercept: ", model.intercept_)

print("Coeficient: ", model.coef_)
# find sqft_above and sqft_basement are great pair of independent variables

X = data[['sqft_above','sqft_basement']]

y = data['sqft_living']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=100)

print(X_train.shape)

print(X_test.shape)



model = LinearRegression()

output_model = model.fit(X_train, y_train)

R_sq = model.score(X_train, y_train)

print("R-Square: ", R_sq)

print("Intercept: ", model.intercept_)

print("Coeficient: ", model.coef_)
# model testing

y_predict = model.predict(X_test)

#y_predict



#predict result visualization

from matplotlib import rcParams

rcParams['font.sans-serif'] = 'SimHei'

fig = plt.figure(figsize=(20,6)) ##设定空白画布，并制定大小

plt.plot(range(y_test.shape[0]),y_test,color="blue", linewidth=1.5, linestyle="-")

plt.plot(range(y_test.shape[0]),y_predict,color="red", linewidth=1.5, linestyle="-.")

plt.legend(['Actual','Predicted'])

plt.show() 
#model assessment

from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,r2_score

print("Mean Squared Error: %.2f" % mean_squared_error(y_test,y_predict))

print("Mean Absolute Error: %.2f" % mean_absolute_error(y_test,y_predict))

print("Explained Variance Score: %.2f" % explained_variance_score(y_test,y_predict))

print("R2 Score: %.2f" % r2_score(y_test,y_predict))
