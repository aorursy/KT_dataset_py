%matplotlib inline



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
mtcars = pd.read_csv("../input/mtcars/mtcars.csv")

mtcars =  mtcars.rename(columns={'Unnamed: 0': 'model'})

mtcars.index = mtcars.model

del mtcars["model"]



mtcars.head()
mtcars.mean()                 # Get the mean of each column
mtcars.mean(axis=1)           # Get the mean of each row
mtcars.median()                 # Get the median of each column
norm_data = pd.DataFrame(np.random.normal(size=100000))



norm_data.plot(kind="density",

              figsize=(10,10));





plt.vlines(norm_data.mean(),     # Plot black line at mean

           ymin=0, 

           ymax=0.4,

           linewidth=5.0);



plt.vlines(norm_data.median(),   # Plot red line at median

           ymin=0, 

           ymax=0.4, 

           linewidth=2.0,

           color="red");
skewed_data = pd.DataFrame(np.random.exponential(size=100000))



skewed_data.plot(kind="density",

              figsize=(10,10),

              xlim=(-1,5));





plt.vlines(skewed_data.mean(),     # Plot black line at mean

           ymin=0, 

           ymax=0.8,

           linewidth=5.0);



plt.vlines(skewed_data.median(),   # Plot red line at median

           ymin=0, 

           ymax=0.8, 

           linewidth=2.0,

           color="red");
norm_data = np.random.normal(size=50)

outliers = np.random.normal(15, size=3)

combined_data = pd.DataFrame(np.concatenate((norm_data, outliers), axis=0))



combined_data.plot(kind="density",

              figsize=(10,10),

              xlim=(-5,20));





plt.vlines(combined_data.mean(),     # Plot black line at mean

           ymin=0, 

           ymax=0.2,

           linewidth=5.0);



plt.vlines(combined_data.median(),   # Plot red line at median

           ymin=0, 

           ymax=0.2, 

           linewidth=2.0,

           color="red");
mtcars.mode()
max(mtcars["mpg"]) - min(mtcars["mpg"])
five_num = [mtcars["mpg"].quantile(0),   

            mtcars["mpg"].quantile(0.25),

            mtcars["mpg"].quantile(0.50),

            mtcars["mpg"].quantile(0.75),

            mtcars["mpg"].quantile(1)]



five_num
mtcars["mpg"].describe()
mtcars["mpg"].quantile(0.75) - mtcars["mpg"].quantile(0.25)
mtcars.boxplot(column="mpg",

               return_type='axes',

               figsize=(8,8))



plt.text(x=0.74, y=22.25, s="3rd Quartile")

plt.text(x=0.8, y=18.75, s="Median")

plt.text(x=0.75, y=15.5, s="1st Quartile")

plt.text(x=0.9, y=10, s="Min")

plt.text(x=0.9, y=33.5, s="Max")

plt.text(x=0.7, y=19.5, s="IQR", rotation=90, size=25);
mtcars["mpg"].var()
mtcars["mpg"].std()
abs_median_devs = abs(mtcars["mpg"] - mtcars["mpg"].median())



abs_median_devs.median() * 1.4826
mtcars["mpg"].skew()  # Check skewness
mtcars["mpg"].kurt()  # Check kurtosis
norm_data = np.random.normal(size=100000)

skewed_data = np.concatenate((np.random.normal(size=35000)+2, 

                             np.random.exponential(size=65000)), 

                             axis=0)

uniform_data = np.random.uniform(0,2, size=100000)

peaked_data = np.concatenate((np.random.exponential(size=50000),

                             np.random.exponential(size=50000)*(-1)),

                             axis=0)



data_df = pd.DataFrame({"norm":norm_data,

                       "skewed":skewed_data,

                       "uniform":uniform_data,

                       "peaked":peaked_data})
data_df.plot(kind="density",

            figsize=(10,10),

            xlim=(-5,5));
data_df.skew()
data_df.kurt()