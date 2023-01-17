#Import pandas for dataset manupilation and matplotlib and seaborn for visualization
import pandas as pd  
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
#Import functions for Dataset Split, Model and Evaluation Metrics
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#Read the CSV File into a DataFrame Object 
df=pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv') #Read the Dataset CSV File to a dataframe object
df.shape # To view the shape of our dataset (1599 rows and 12 columns)
df.head() #Display first 5 rows of the Dataset
df.tail() #Display last 5 rows of the Dataset
#Display detailed information such as count, max,min,etc. 
df.describe()
df.isnull().any() #To check if any column has null values. Column returns False if no null values
# Create X attributes and Y labels from dataframe object
X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values
y = df['quality'].values
#Get Correlation Matrix of the dataset attributes
corr=df.corr()
#Display the correlation matrix as a heatmap
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
print("""
# Exactly –1. A perfect downhill (negative) linear relationship

# –0.70. A strong downhill (negative) linear relationship

# –0.50. A moderate downhill (negative) relationship

# –0.25. A weak downhill (negative) linear relationship

# 0. No linear relationship


# +0.25. A weak uphill (positive) linear relationship

# +0.50. A moderate uphill (positive) relationship

# +0.70. A strong uphill (positive) linear relationship

# Exactly +1. A perfect uphill (positive) linear relationship
""")
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(df['quality'])
print("#Qualities most of the time are 5 or 6")
sns.pairplot(data=df,kind='scatter',diag_kind='kde') #Shows relationships among all pairs of features
# Create the training and test sets using 0.2 as test size (i.e 80% of data for training rest 20% for model testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  #Create the Linear Regressor Object and fit the training set to the regressor object
regressor.fit(X_train, y_train)
Coefficients=pd.DataFrame(data = regressor.coef_,index = df.columns[:11]) # Create a dataframe object containing the attributes 
# and corresponding regressor coefficients
print(Coefficients) # Display the coefficients
y_pred = regressor.predict(X_test) #Run the model and predict on the test set
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) #View first 25 predictions against actual values
df1 = df.head(25)
print(df1)
#Plot the predictions vs actual values
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title('Predictions vs Real Values')
plt.show()
# Print the Mean Absolute Error, Mean Squared Error and Root Mean Squared Errors of the Regression Model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#Print the Final R2 Score of the Regression Model
print("R2 score =", round(metrics.r2_score(y_test, y_pred), 2))