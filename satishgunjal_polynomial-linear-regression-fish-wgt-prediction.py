import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.preprocessing import PolynomialFeatures 
df = pd.read_csv('https://raw.githubusercontent.com/satishgunjal/datasets/master/Fish.csv')
print('Dimension of dataset= ', df.shape)
df.head(5) # head(n) returns first n records only. Can also use sample(n) for random n records.
df1 = df.rename(columns={'Length1':'VerticalLen','Length2':'DiagonalLen','Length3':'CrossLen'})
df1.sample(5) # Display random 5 records
df1.info()
df1.corr()
plt.rcParams["figure.figsize"] = (10,6) # Custom figure size in inches
sns.heatmap(df1.corr(), annot =True)
plt.title('Correlation Matrix')
df2 = df1.drop(['VerticalLen', 'DiagonalLen', 'CrossLen'], axis =1) # Can also use axis = 'columns'
print('New dimension of dataset is= ', df2.shape)
df2.head(3)
sns.pairplot(df2, kind = 'scatter', hue = 'Species')
def outlier_detection(dataframe):
  """
  Find the outlier in given dataset. To get the index fo the outlier data, please input single column dataframe

  Input Parameters
  ----------------
  dataframe : single column dataframe
  
  Output Parameters
  -----------------
  outlier : Index of outlier training examples.
  """
  Q1 = dataframe.quantile(0.25)
  Q3 = dataframe.quantile(0.75)
  IQR = Q3 - Q1
  upper_end = Q3 + 1.5 * IQR
  lower_end = Q1 - 1.5 * IQR 
  outlier = dataframe[(dataframe > upper_end) | (dataframe < lower_end)]
  return outlier
sns.boxplot(data= df2['Weight'] )
plt.title('Outlier Detection Based on Weight')
# 'Species' column contains categorical values, so using list slicing to iterate over all the columns except first one
for column in df2.columns[1:]: 
    print('\nOutliers in column "%s" ' % column)
    outlier = outlier_detection(df2[column])
    print(outlier)
#Lets create temp dataframe without 'Weight' feature for plotting the boxplot
df_temp = df2.drop(['Weight'], axis = 'columns')
sns.boxplot(data= df_temp[df_temp.Species == 'Perch'] )
plt.title('Outlier Detection For Pearch Species')
df_Perch = df2[df2.Species == 'Perch']
for column in df_Perch.columns[1:]: 
    print('\nOutliers in column "%s" ' % column)
    outlier = outlier_detection(df_Perch[column])
    print(outlier)
sns.boxplot(data= df_temp[df_temp.Species == 'Bream'] )
plt.title('Outlier Detection For Bream Species')
df_Bream = df2[df2.Species == 'Bream']
for column in df_Bream.columns[1:]: 
    print('\nOutliers in column "%s" ' % column)
    outlier = outlier_detection(df_Bream[column])
    print(outlier)
sns.boxplot(data= df_temp[df_temp.Species == 'Roach'] )
plt.title('Outlier Detection For Roach Species')
df_Roach = df2[df2.Species == 'Roach']
for column in df_Roach.columns[1:]: 
    print('\nOutliers in column "%s" ' % column)
    outlier = outlier_detection(df_Roach[column])
    print(outlier)
sns.boxplot(data= df_temp[df_temp.Species == 'Pike'] )
plt.title('Outlier Detection For Pike Species')
df_Pike = df2[df2.Species == 'Pike']
for column in df_Pike.columns[1:]: 
    print('\nOutliers in column "%s" ' % column)
    outlier = outlier_detection(df_Pike[column])
    print(outlier)
sns.boxplot(data= df_temp[df_temp.Species == 'Smelt'] )
plt.title('Outlier Detection For Smelt Species')
df_Smelt = df2[df2.Species == 'Smelt']
for column in df_Smelt.columns[1:]: 
    print('\nOutliers in column "%s" ' % column)
    outlier = outlier_detection(df_Smelt[column])
    print(outlier)
sns.boxplot(data= df_temp[df_temp.Species == 'Parkki'] )
plt.title('Outlier Detection For Parkki Species')
df_Parkki = df2[df2.Species == 'Parkki']
for column in df_Parkki.columns[1:]: 
    print('\nOutliers in column "%s" ' % column)
    outlier = outlier_detection(df_Parkki[column])
    print(outlier)
sns.boxplot(data= df_temp[df_temp.Species == 'Whitefish'] )
plt.title('Outlier Detection For Whitefish Species')
df_Whitefish = df2[df2.Species == 'Whitefish']
for column in df_Whitefish.columns[1:]: 
    print('\nOutliers in column "%s" ' % column)
    outlier = outlier_detection(df_Whitefish[column])
    print(outlier)
df3 = df2.drop([35,54,157,158])
df3.shape
df3.isna().sum()
df3[df3.Weight <= 0]
df4 = df3.drop([40])
df4.shape
dummies_species = pd.get_dummies(df4.Species) # store the dummy variables in 'dummies_species' dataframe
dummies_species.head(3) # To do get individual dummy value
df5 = pd.concat([df4, dummies_species],axis = 'columns')
df5.head(3)
df6 = df5.drop(['Species','Whitefish'], axis = 'columns')
df6.head(3)
X = df6[['Height', 'Width', 'Bream', 'Parkki' ,'Perch', 'Pike', 'Roach', 'Smelt']] # Or can use df6.iloc[:,[1,2,3,4,5,6,7,8]]
y = df6[['Weight']]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
print('X_train dimension= ', X_train.shape)
print('X_test dimension= ', X_test.shape)
print('y_train dimension= ', y_train.shape)
print('y_train dimension= ', y_test.shape)
def polynomial_plot(feature, label):
  # Create 1D array. We can use 'squeeze' function to reduce the 2D array to 1D array
  x_coordinates = feature
  y_coordinates = np.squeeze(label)

 # Contruct first degree polynomial function
  linear_func = np.poly1d(np.polyfit(x_coordinates, y_coordinates, 1))
  # Contruct second degree polynomial function
  quadratic_func = np.poly1d(np.polyfit(x_coordinates, y_coordinates, 2))
 
  # Generate evenly spaced values
  values = np.linspace(x_coordinates.min(), x_coordinates.max(), len(x_coordinates))

  plt.scatter(x_coordinates,y_coordinates, color='blue')  
  plt.plot(values, linear_func(values), color='cyan', linestyle='dashed', label='Linear Function')
  plt.plot(values, quadratic_func(values), color='red', label='Quadratic Function')
  plt.xlabel('%s From Test Data'%(feature.name))
  plt.ylabel('Weight')
  plt.rcParams["figure.figsize"] = (10,6) # Custom figure size in inches
  plt.legend()
  plt.title("Linear Vs Quadratic Function For Feature %s" % (feature.name))
  plt.show()  
polynomial_plot(X_train.Width, y_train)
polynomial_plot(X_train.Height, y_train)
poly = PolynomialFeatures(degree = 2) 
X_poly = poly.fit_transform(X_train) 
poly.fit(X_poly, y_train) 
lm = linear_model.LinearRegression() 
lm.fit(X_poly, y_train) 
predictions = lm.predict(poly.fit_transform(X_test))
print('r2_score= ', metrics.r2_score(y_test, predictions))
predictedWeight = pd.DataFrame(predictions, columns=['Predicted Weight']) # Create new dataframe of column'Predicted Weight'
actualWeight = pd.DataFrame(y_test)
actualWeight = actualWeight.reset_index(drop=True) # Drop the index so that we can concat it, to create new dataframe
df_actual_vs_predicted = pd.concat([actualWeight,predictedWeight],axis =1)
df_actual_vs_predicted.T
plt.scatter(y_test, predictions)
plt.xlabel('Weight From Test Data')
plt.ylabel('Weight Predicted By Model')
plt.rcParams["figure.figsize"] = (10,6) # Custom figure size in inches
plt.title("Weight From test Data Vs Weight Predicted By Model")
sns.distplot((y_test-predictions))
plt.rcParams["figure.figsize"] = (10,6) # Custom figure size in inches
plt.title("Histogram of Residuals")