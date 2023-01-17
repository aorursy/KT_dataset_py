%matplotlib inline





import numpy as np              

import pandas as pd
mtcars = pd.read_csv("../input/mtcars/mtcars.csv")



print (mtcars.head() )



mtcars.index = mtcars.model       # Set row index to car model

del mtcars["model"]               # Drop car name column



colmeans = mtcars.sum()/mtcars.shape[0]  # Get column means



colmeans
centered = mtcars-colmeans



centered.describe()
 # Get column standard deviations

column_deviations = mtcars.std(axis=0)  



centered_and_scaled = centered/column_deviations 



centered_and_scaled.describe()
from sklearn import preprocessing
scaled_data = preprocessing.scale(mtcars)  # Scale the data*

 

scaled_cars = pd.DataFrame(scaled_data,    # Remake the DataFrame

                           index=mtcars.index,

                           columns=mtcars.columns)



scaled_cars.describe()
# Generate normal data*

normally_distributed = np.random.normal(size=10000)  



normally_distributed = pd.DataFrame(normally_distributed) # Convert to DF



normally_distributed.hist(figsize=(8,8),            # Plot histogram

                          bins=30);         
skewed = np.random.exponential(scale=2,      # Generate skewed data

                               size= 10000)  



skewed = pd.DataFrame(skewed)                # Convert to DF



skewed.hist(figsize=(8,8),                   # Plot histogram

            bins=50);                               
# Get the square root of data points*

sqrt_transformed = skewed.apply(np.sqrt) 



sqrt_transformed.hist(figsize=(8,8),     # Plot histogram

                 bins=50);        
log_transformed = (skewed+1).apply(np.log)   # Get the log of the data



log_transformed.hist(figsize = (8,8),          # Plot histogram

                 bins=50);       
mtcars.iloc[:,0:6].corr()   # Check the pairwise correlations of 6 variables
from pandas.tools.plotting import scatter_matrix



scatter_matrix(mtcars.iloc[:,0:6], # Make a scatter matrix of 6 columns

               figsize=(10, 10),   # Set plot size

               diagonal='kde');    # Show distribution estimates on diagonal
from sklearn.preprocessing import Imputer



# The following line sets a few mpg values to None

mtcars["mpg"] = np.where(mtcars["mpg"]>22, None, mtcars["mpg"])



mtcars["mpg"]       # Confirm that missing values were added
imp = Imputer(missing_values='NaN',  # Create imputation model

              strategy='mean',       # Use mean imputation

              axis=0)                # Impute by column



imputed_cars = imp.fit_transform(mtcars)   # Use imputation model to get values



imputed_cars = pd.DataFrame(imputed_cars,  # Remake DataFrame with new values

                           index=mtcars.index,

                           columns = mtcars.columns)



imputed_cars.head(10)