import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.model_selection import train_test_split
df = pd.read_csv("/kaggle/input/fish-market/Fish.csv")

print('Shape of dataset= ', df.shape) # To get no of rows and columns

df.head(5) # head(n) returns first n records only. Can also use sample(n) for random n records.
df.rename(columns={'Length1':'VerticalLen','Length2':'DiagonalLen','Length3':'CrossLen'},inplace = True) # 'inplace= true' to make change in current dataframe

df.sample(5) # Display random 5 records
df.info()
# isna() will return 'True' is value is 'None' or 'numpy.NaN'

# Characters such as empty strings '' or 'numpy.inf' are not considered NA values (unless you set pandas.options.mode.use_inf_as_na = True)

# you can also use df.isnull()

df.isna().sum() # Get sum of all Nan values from each column

#df.isna().values.any()  
df.Species.value_counts()
df_sp = df.Species.value_counts()

df_sp = pd.DataFrame(df_sp)

df_sp.T 

# Note: Just like matrices. 'dataframe.T' will Transpose index and columns

# I am using it just for saving vertical space and making notbook more readable
sns.barplot(x= df_sp.index, y = df_sp.Species) # df_sp.index will returns row labels of dataframe

plt.xlabel('Species')

plt.ylabel('Count of Species')

plt.rcParams["figure.figsize"] = (10,6)

plt.title('Fish Count Based On Species')

plt.show()
df[df.Weight <= 0]
df1 = df.drop([40])

print('New dimension of dataset is= ', df1.shape)

df1.head(5)
df1.corr()
plt.rcParams["figure.figsize"] = (10,6) # Custom figure size in inches

sns.heatmap(df1.corr(), annot =True)

plt.title('Correlation Matrix')
df2 = df1.drop(['VerticalLen', 'DiagonalLen', 'CrossLen'], axis =1) # Can also use axis = 'columns'

print('New dimension of dataset is= ', df2.shape)

df2.head()
sns.pairplot(df2, kind = 'scatter', hue = 'Species')
sns.boxplot(x=df2['Weight'])

plt.title('Outlier Detection based on Weight')
def outlier_detection(dataframe):

  Q1 = dataframe.quantile(0.25)

  Q3 = dataframe.quantile(0.75)

  IQR = Q3 - Q1

  upper_end = Q3 + 1.5 * IQR

  lower_end = Q1 - 1.5 * IQR 

  outlier = dataframe[(dataframe > upper_end) | (dataframe < lower_end)]

  return outlier
outlier_detection(df2['Weight'])
sns.boxplot(x =df2['Height'])

plt.title('Outlier Detection based on Height')
sns.boxplot(x = df2['Width'])

plt.title('Outlier Detection based on Width')
df3 = df2.drop([142,143,144])

df3.shape
df3.describe().T
#X = df3.iloc[:,[2,3]] # Select columns using column index

X = df3[['Height','Width']] # Select columns using column name

X.head()
#y = df3.iloc[:,[1]] # Select columns using column index

y = df3[['Weight']]

y.head(5)
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state = 42) 

# Use paramter 'random_state=1' if you want keep results same everytime you execute above code

print('X_train dimension= ', X_train.shape)

print('X_test dimension= ', X_test.shape)

print('y_train dimension= ', y_train.shape)

print('y_train dimension= ', y_test.shape)
model = linear_model.LinearRegression()

model.fit(X_train,y_train)
print('coef= ', model.coef_) # Since we have two features(Height and Width), there will be 2 coef

print('intercept= ', model.intercept_)

print('score= ', model.score(X_test,y_test))
predictedWeight = pd.DataFrame(model.predict(X_test), columns=['Predicted Weight']) # Create new dataframe of column'Predicted Weight'

actualWeight = pd.DataFrame(y_test)

actualWeight = actualWeight.reset_index(drop=True) # Drop the index so that we can concat it, to create new dataframe

df_actual_vs_predicted = pd.concat([actualWeight,predictedWeight],axis =1)

df_actual_vs_predicted.T
plt.scatter(y_test, model.predict(X_test))

plt.xlabel('Weight From Test Data')

plt.ylabel('Weight Predicted By Model')

plt.rcParams["figure.figsize"] = (10,6) # Custom figure size in inches

plt.title("Weight From test Data Vs Weight Predicted By Model")
plt.scatter(X_test['Height'], y_test, color='red', label = 'Actual Weight')

plt.scatter(X_test['Height'], model.predict(X_test), color='green', label = 'Prdicted Weight')

plt.xlabel('Height')

plt.ylabel('Weight')

plt.rcParams["figure.figsize"] = (10,6) # Custom figure size in inches

plt.title('Actual Vs Predicted Weight for Test Data')

plt.legend()

plt.show()
plt.scatter(X_test['Width'], y_test, color='red', label = 'Actual Weight')

plt.scatter(X_test['Width'], model.predict(X_test), color='green', label = 'Prdicted Weight')

plt.xlabel('Width')

plt.ylabel('Weight')

plt.rcParams["figure.figsize"] = (10,6) # Custom figure size in inches

plt.title('Actual Vs Predicted Weight for Test Data')

plt.legend()

plt.show()
sns.distplot((y_test-model.predict(X_test)))

plt.rcParams["figure.figsize"] = (10,6) # Custom figure size in inches

plt.title("Histogram of Residuals")