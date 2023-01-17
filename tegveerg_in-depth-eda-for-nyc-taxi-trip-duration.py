import pandas as pd             #pandas for using dataframe and reading csv file(s)
import numpy as np              #numpy for vector operations and basic maths
import matplotlib.pyplot as plt #for plotting
%matplotlib inline              
import seaborn as sns           #for making plots
from haversine import haversine #for working with latitudinal and longitudinal data points
import math                     #for basic math operations
import warnings
from pandas.plotting import parallel_coordinates #for multivariate plots
warnings.filterwarnings('ignore') #ignore deprecation warnings
#importing data
data = pd.read_csv('nyc_taxi_trip_duration.csv')
#first 20 instances using "head()" function
data.head(20)
#last 20 instances using "tail()" function
data.tail(20)
#finding out the shape of the data using "shape" variable: Output (rows, columns)
data.shape
#Printing all the columns present in data
data.columns
#Checking for NaN values present in data
data.isna().sum()
#Checking for Null values present in data
data.isnull().sum()
# A closer look at the data types present in the data
data.dtypes
#Identifying variables with integer datatype
data.dtypes[data.dtypes == 'int64']
#Converting vendor_id to category datatype
data['vendor_id'] = data['vendor_id'].astype('category')
data.dtypes
#Converting passenger_count to category datatype
data['passenger_count'] = data['passenger_count'].astype('category')
data.dtypes
#Identifying variables with object datatype
data.dtypes[data.dtypes == 'object']
#Converting the object data type variables to their respective datatype
data['id'] = data['id'].astype('category')
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
data['dropoff_datetime'] = pd.to_datetime(data['dropoff_datetime'])
data['store_and_fwd_flag'] = data['store_and_fwd_flag'].astype('category')
#Modifying values of the 'id' variable by removing the redundant 'id' part present in all values
def modify_id(x):
    return x[2:]
data['id'] = data['id'].apply(modify_id)
data.head(10)
data.tail(10)
#Checking
data['id'] = data['id'].astype('category')
data.dtypes
# Identifying variables with float datatype
data.dtypes[data.dtypes == 'float64']
#Next, for convenience, we shall add a new column for trip_duration_minutes
data['trip_duration_minutes'] = data['trip_duration'].apply(lambda x: x/60)
#We shall also make a feature for distance of the trip in kilometers (km)
def calc_distance(data):
    pickup = (data['pickup_latitude'], data['pickup_longitude'])
    drop = (data['dropoff_latitude'], data['dropoff_longitude'])
    return haversine(pickup, drop)
data['distance'] = data.apply(lambda x: calc_distance(x), axis = 1)
#And lastly, we shall make a feature for the average speed of the trip in km/hr
data['speed'] = (data.distance/(data.trip_duration/3600))
data.head()
#Obtain day names for each value
data.pickup_datetime.apply(lambda x: x.day_name())
#Obtain month names for each value
data.pickup_datetime.apply(lambda x: x.month_name())
data.head()
# create time based features for pickup_datetime 
data['pickup_datetime_moy'] = data.pickup_datetime.dt.month
data['pickup_datetime_hour'] = data.pickup_datetime.dt.hour
# create more features for pickup_datetime 
data['pickup_datetime_woy'] = data.pickup_datetime.dt.weekofyear
data['pickup_datetime_dow'] = data.pickup_datetime.dt.dayofweek
data['pickup_datetime_doy'] = data.pickup_datetime.dt.dayofyear
# create time based features for dropoff_datetime 
data['dropoff_datetime_moy'] = data.dropoff_datetime.dt.month
data['dropoff_datetime_hour'] = data.dropoff_datetime.dt.hour
data['dropoff_datetime_woy'] = data.dropoff_datetime.dt.weekofyear
data['dropoff_datetime_dow'] = data.dropoff_datetime.dt.dayofweek
data['dropoff_datetime_doy'] = data.dropoff_datetime.dt.dayofyear
data.head()
data.tail()
data.dtypes
data.describe()
# Numerical datatypes
data.select_dtypes(include=['int64','float64','Int64']).dtypes
# segregating variables into groups
pickup_dropoff_location = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']
trip_details = ['trip_duration','trip_duration_minutes', 'distance', 'speed']
pickup_dropoff_time = ['pickup_datetime_moy', 'dropoff_datetime_moy', 'pickup_datetime_hour', 'dropoff_datetime_hour', 'pickup_datetime_woy', 'dropoff_datetime_woy', 'pickup_datetime_dow', 'dropoff_datetime_dow', 'pickup_datetime_doy', 'dropoff_datetime_doy']
# custom function for easy and efficient analysis of numerical univariate

def UVA_numeric(data, var_group):
  '''
  Univariate_Analysis_numeric
  takes a group of variables (INTEGER and FLOAT) and plot/print all the descriptives and properties along with KDE.

  Runs a loop: calculate all the descriptives of i(th) variable and plot/print it
  '''

  size = len(var_group)
  plt.figure(figsize = (7*size,3), dpi = 100)
  
  #looping for each variable
  for j,i in enumerate(var_group):
    
    # calculating descriptives of variable
    mini = data[i].min()
    maxi = data[i].max()
    ran = data[i].max()-data[i].min()
    mean = data[i].mean()
    median = data[i].median()
    mode = data[i].mode()
    st_dev = data[i].std()
    skew = data[i].skew()
    kurt = data[i].kurtosis()

    # calculating points of standard deviation
    points = mean-st_dev, mean+st_dev

    #Plotting the variable with every information
    plt.subplot(1,size,j+1)
    sns.kdeplot(data[i], shade=True)
    sns.lineplot(points, [0,0], color = 'black', label = "std_dev")
    sns.scatterplot([mini,maxi], [0,0], color = 'orange', label = "min/max")
    sns.scatterplot([mean], [0], color = 'red', label = "mean")
    sns.scatterplot([median], [0], color = 'blue', label = "median")
    sns.scatterplot([mode], [0], color = 'green', label = "mode")
    plt.xlabel('{}'.format(i), fontsize = 20)
    plt.ylabel('density')
    plt.title('std_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}'.format((round(points[0],2),round(points[1],2)),
                                                                                                   round(kurt,2),
                                                                                                   round(skew,2),
                                                                                                   (round(mini,2),round(maxi,2),round(ran,2)),
                                                                                                   round(mean,2),
                                                                                                   round(median,2)))
                                                                                           
UVA_numeric(data,pickup_dropoff_location)
# copying pickup_dropoff_location
pdl_data = data[pickup_dropoff_location]

# filtering using loc
pdl_data = pdl_data.loc[(pdl_data.pickup_longitude > -74.2) & (pdl_data.pickup_longitude < -73.7)]
pdl_data = pdl_data.loc[(pdl_data.pickup_latitude > 40.5) & (pdl_data.pickup_latitude < 40.95)]
pdl_data = pdl_data.loc[(pdl_data.dropoff_longitude > -74.2) & (pdl_data.dropoff_longitude < -73.7)]
pdl_data = pdl_data.loc[(pdl_data.dropoff_latitude > 40.5) & (pdl_data.dropoff_latitude < 40.95)]

# checking how many points are removed
len(data), len(pdl_data)
UVA_numeric(pdl_data,pickup_dropoff_location)
UVA_numeric(data,trip_details)
data.distance[data.distance == 0 ].count()
# copying trip_details
td_data = data[trip_details]

# filtering all trip_details variables using loc
td_data = td_data.loc[(td_data.trip_duration < 7200)]
td_data = td_data.loc[(td_data.trip_duration_minutes < 120)]
td_data = td_data.loc[(td_data.distance < 150)]
td_data = td_data.loc[(td_data.speed < 105)]

# checking how many points are removed
len(data), len(td_data)
UVA_numeric(td_data,trip_details)
UVA_numeric(data,pickup_dropoff_time) 
#Taking into consideration the points mentioned about week of year, we shall make a new column for only year and confirm that only the year 2016 is present
data['pickup_datetime_year'] = data.pickup_datetime.dt.year
data['pickup_datetime_year'].describe()
#dropping column pickup_datetime_year
data.drop('pickup_datetime_year', axis=1, inplace=True)
data.head()
data.dtypes
#change the pickup_datetime_woy and dropoff_datetime_woy values of 53 to 1
data.loc[(data.pickup_datetime_woy == 53),'pickup_datetime_woy'] = 1
data.loc[(data.dropoff_datetime_woy == 53),'dropoff_datetime_woy'] = 1
data.head()
#confirming that pickup_datetime_woy does not have values of 53 anymore
data[data.pickup_datetime_woy == 53]
#confirming that dropoff_datetime_woy does not have values of 53 anymore
data[data.dropoff_datetime_woy == 53]
#Plotting the distributions again to see the effect of replacing incorrect values
UVA_numeric(data,pickup_dropoff_time) 
data.select_dtypes(exclude=['int64','float64','Int64']).dtypes
# Custom function for easy visualisation of Categorical Variables
def UVA_variable(data, var):

  '''
  Univariate_Analysis_categorical
  takes a categorical variable and plots/prints all the value_counts and a barplot.
  '''
  # setting figure_size
  size = len(var)
  plt.figure(figsize = (7*size,5), dpi = 100)

  # for every variable
  for j,i in enumerate(var):
    norm_count = data[i].value_counts(normalize = True)
    n_uni = data[i].nunique()

  #Plotting the variable with every information
    plt.subplot(1,size,j+1)
    sns.barplot(norm_count, norm_count.index , order = norm_count.index)
    plt.xlabel('fraction/percent', fontsize = 20)
    plt.ylabel('{}'.format(i), fontsize = 20)
    plt.title('n_uniques = {} \n value counts \n {};'.format(n_uni,norm_count))
UVA_variable(data, ['passenger_count'])
UVA_variable(data, ['vendor_id'])
UVA_variable(data, ['store_and_fwd_flag'])
data.isna().sum()
# custom function for easy outlier analysis

def UVA_outlier(data, var_group, include_outlier = True):
  '''
  Univariate_Analysis_outlier:
  takes a group of variables (INTEGER and FLOAT) and plot/print boplot and descriptives
  Runs a loop: calculate all the descriptives of i(th) variable and plot/print it 

  data : dataframe from which to plot from\n
  var_group : {list} type Group of Continuous variables
  include_outlier : {bool} whether to include outliers or not, default = True
  '''

  size = len(var_group)
  plt.figure(figsize = (10*size,8), dpi = 200)
  
  #looping for each variable
  for j,i in enumerate(var_group):
    
    # calculating descriptives of variable
    quant25 = data[i].quantile(0.25)
    quant75 = data[i].quantile(0.75)
    IQR = quant75 - quant25
    med = data[i].median()
    whis_low = med-(1.5*IQR)
    whis_high = med+(1.5*IQR)

    # Calculating Number of Outliers
    outlier_high = len(data[i][data[i]>whis_high])
    outlier_low = len(data[i][data[i]<whis_low])

    if include_outlier == True:
      print(include_outlier)
      #Plotting the variable with every information
      plt.subplot(1,size,j+1)
      sns.boxplot(data[i], orient="v")
      plt.ylabel('{}'.format(i))
      plt.title('With Outliers\nOutlier (low/high) = {} \n'.format((outlier_low,outlier_high)))
                                                                                                   
      
    else:
      # replacing outliers with max/min whisker
      data2 = data[var_group][:]
      data2[i][data2[i]>whis_high] = whis_high+1
      data2[i][data2[i]<whis_low] = whis_low-1
      
      quant25 = data2[i].quantile(0.25)
      quant75 = data2[i].quantile(0.75)
      IQR = quant75 - quant25
      med = data2[i].median()
      whis_low = med-(1.5*IQR)
      whis_high = med+(1.5*IQR)
      outlier_high = len(data2[i][data2[i]>whis_high])
      outlier_low = len(data2[i][data2[i]<whis_low])
    
      # plotting without outliers
      plt.subplot(1,size,j+1)
      sns.boxplot(data2[i], orient="v")
      plt.ylabel('{}'.format(i))
      plt.title('Without Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n'.format(round(IQR,2),
                                                                                                   round(med,2),
                                                                                                   (round(quant25,2),round(quant75,2))
                                                                                                   ))
#changing passenger_count to integer datatype only for this section to work with the function 
data['passenger_count'] = data['passenger_count'].astype('int64')
UVA_outlier(data, ['passenger_count'],) 
UVA_outlier(data, ['passenger_count'], include_outlier=False)
#revert back to categorical
data['passenger_count'] = data['passenger_count'].astype('category')
UVA_outlier(data, pickup_dropoff_location,)
UVA_outlier(data, pickup_dropoff_location, include_outlier=False)
UVA_outlier(data, trip_details,)
UVA_outlier(data, trip_details, include_outlier=False) 
UVA_outlier(data, pickup_dropoff_time,)
UVA_outlier(data, pickup_dropoff_time, include_outlier=False) 
# isolating numerical datatypes
numerical = data.select_dtypes(include=['int64','float64','Int64'])[:]
numerical.dtypes
# calculating correlation
correlation = numerical.dropna().corr()
correlation
# plotting heatmap using Pearson Coeff, Kendall's Tau, and Spearman Coeff for all numerical variables
plt.figure(figsize=(36,6), dpi=140)
for j,i in enumerate(['pearson','kendall','spearman']):
  plt.subplot(1,3,j+1)
  correlation = numerical.dropna().corr(method=i)
  sns.heatmap(correlation, linewidth = 2)
  plt.title(i, fontsize=18)
# Grouping variables
pickup_dropoff_location = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']
trip_details = ['trip_duration','trip_duration_minutes', 'distance', 'speed']
pickup_dropoff_time = ['pickup_datetime_moy', 'dropoff_datetime_moy', 'pickup_datetime_hour', 'dropoff_datetime_hour', 'pickup_datetime_woy', 'dropoff_datetime_woy', 'pickup_datetime_dow', 'dropoff_datetime_dow', 'pickup_datetime_doy', 'dropoff_datetime_doy']
# scatter plot for pickup_dropoff_location variables
plt.figure(dpi=140)
sns.pairplot(numerical[pickup_dropoff_location])
#taking log of every value and dividing by 100,000 to negate outliers
var = []
var.extend(pickup_dropoff_location)
var.extend(trip_details)
var.extend(pickup_dropoff_time)
for column in var:
  mini=1
  if numerical[column].min()<0:
    mini =  abs(numerical[column].min()) + 1
  
  numerical[column] = [i+mini for i in numerical[column]]
  numerical[column] = numerical[column].map(lambda x : np.log(x)/100000)
# scatter plot for pickup_dropoff_location variables
plt.figure(dpi=140)
sns.pairplot(numerical[pickup_dropoff_location])
# scatter plot for trip_details variables
plt.figure(dpi=140)
sns.pairplot(numerical[trip_details])
#if we remember from Univariate Analysis, we found that there were approx. 3000 observations with distance=0. We shall use a scatter plot to analyze the relation between this distance value and trip_duration_minutes
filtered_dist = data.loc[(data.distance == 0) & (data.trip_duration_minutes < 120), ['distance','trip_duration_minutes']]
plt.scatter(filtered_dist.trip_duration_minutes, filtered_dist.distance , s=1, alpha=0.5)
plt.ylabel('Distance in km')
plt.xlabel('Trip Duration in Minutes')
plt.show()
data.distance.mean()
#if we also remember from Univariate Analysis, we found that there were approx. 3000 observations with distance=0. We shall use a scatter plot to analyze the relation between this distance value and trip_duration_minutes
filtered_dist = data.loc[(data.distance == 0) & (data.trip_duration_minutes < 120), ['distance','trip_duration_minutes']]
plt.scatter(filtered_dist.trip_duration_minutes, filtered_dist.distance , s=1, alpha=0.5)
plt.ylabel('Distance in km')
plt.xlabel('Trip Duration in Minutes')
plt.show()
# scatter plot for pickup_dropoff_time variables
plt.figure(dpi=140)
sns.pairplot(numerical[pickup_dropoff_time])
def TwoSampZ(X1, X2, sigma1, sigma2, N1, N2):
  '''
  takes mean, standard deviation, and number of observations and returns p-value calculated for 2-sampled Z-Test
  '''
  from numpy import sqrt, abs, round
  from scipy.stats import norm
  ovr_sigma = sqrt(sigma1**2/N1 + sigma2**2/N2)
  z = (X1 - X2)/ovr_sigma
  pval = 2*(1 - norm.cdf(abs(z)))
  return pval
def TwoSampT(X1, X2, sd1, sd2, n1, n2):
  '''
  takes mean, standard deviation, and number of observations and returns p-value calculated for 2-sample T-Test
  '''
  from numpy import sqrt, abs, round
  from scipy.stats import t as t_dist
  ovr_sd = sqrt(sd1**2/n1 + sd2**2/n2)
  t = (X1 - X2)/ovr_sd
  df = n1+n2-2
  pval = 2*(1 - t_dist.cdf(abs(t),df))
  return pval
def Bivariate_cont_cat_hypoth1(data, cont, cat):
  #creating 2 samples, passenger_counts <= 2 (x1) and 2 < passenger_counts < 7 (x2)
  x1 = data[cont][data[cat].isin([0,1,2])][:]
  x2 = data[cont][data[cat].isin([3,4,5,6])][:]
                  
  #calculating descriptives
  n1, n2 = x1.shape[0], x2.shape[0]
  m1, m2 = x1.mean(), x2.mean()
  std1, std2 = x1.std(), x2.std()
  
  #calculating p-values
  t_p_val = TwoSampT(m1, m2, std1, std2, n1, n2)
  z_p_val = TwoSampZ(m1, m2, std1, std2, n1, n2)

  #table
  table = pd.pivot_table(data=data, values=cont, columns=cat, aggfunc = np.mean)

  #plotting
  plt.figure(figsize = (15,6), dpi=140)
  
  #barplot
  plt.subplot(1,2,1)
  sns.barplot(['passenger_counts <= 2','2 < passenger_counts < 7'], [m1, m2])
  plt.ylabel('mean {}'.format(cont))
  plt.xlabel(cat)
  plt.title('t-test p-value = {} \n z-test p-value = {}\n {}'.format(t_p_val,
                                                                z_p_val,
                                                                table))

  # boxplot
  plt.subplot(1,2,2)
  sns.boxplot(x=cat, y=cont, data=data)
  plt.title('categorical boxplot')
  
Bivariate_cont_cat_hypoth1(data, 'trip_duration_minutes', 'passenger_count') 
def Bivariate_cont_cat(data, cont, cat, category):
  #creating 2 samples
  x1 = data[cont][data[cat]==category][:]
  x2 = data[cont][~(data[cat]==category)][:]
  
  #calculating descriptives
  n1, n2 = x1.shape[0], x2.shape[0]
  m1, m2 = x1.mean(), x2.mean()
  std1, std2 = x1.std(), x2.mean()
  
  #calculating p-values
  t_p_val = TwoSampT(m1, m2, std1, std2, n1, n2)
  z_p_val = TwoSampZ(m1, m2, std1, std2, n1, n2)

  #table
  table = pd.pivot_table(data=data, values=cont, columns=cat, aggfunc = np.mean)

  #plotting
  plt.figure(figsize = (15,6), dpi=140)
  
  #barplot
  plt.subplot(1,2,1)
  sns.barplot([str(category),'2'.format(category)], [m1, m2])
  plt.ylabel('mean {}'.format(cont))
  plt.xlabel(cat)
  plt.title('t-test p-value = {} \n z-test p-value = {}\n {}'.format(t_p_val,
                                                                z_p_val,
                                                                table))

  # boxplot
  plt.subplot(1,2,2)
  sns.boxplot(x=cat, y=cont, data=data)
  plt.title('categorical boxplot')
  
Bivariate_cont_cat(data, 'trip_duration_minutes', 'vendor_id', 1)
def Bivariate_cont_cat(data, cont, cat, category):
  #creating 2 samples
  x1 = data[cont][data[cat]==category][:]
  x2 = data[cont][~(data[cat]==category)][:]
  
  #calculating descriptives
  n1, n2 = x1.shape[0], x2.shape[0]
  m1, m2 = x1.mean(), x2.mean()
  std1, std2 = x1.std(), x2.mean()
  
  #calculating p-values
  t_p_val = TwoSampT(m1, m2, std1, std2, n1, n2)
  z_p_val = TwoSampZ(m1, m2, std1, std2, n1, n2)

  #table
  table = pd.pivot_table(data=data, values=cont, columns=cat, aggfunc = np.mean)

  #plotting
  plt.figure(figsize = (15,6), dpi=140)
  
  #barplot
  plt.subplot(1,2,1)
  sns.barplot([str(category),'N'.format(category)], [m1, m2])
  plt.ylabel('mean {}'.format(cont))
  plt.xlabel(cat)
  plt.title('t-test p-value = {} \n z-test p-value = {}\n {}'.format(t_p_val,
                                                                z_p_val,
                                                                table))

  # boxplot
  plt.subplot(1,2,2)
  sns.boxplot(x=cat, y=cont, data=data)
  plt.title('categorical boxplot')
Bivariate_cont_cat(data, 'trip_duration_minutes', 'store_and_fwd_flag', 'Y')
Bivariate_cont_cat_hypoth1(data, 'distance', 'passenger_count') 
def Bivariate_cont_cat(data, cont, cat, category):
  #creating 2 samples
  x1 = data[cont][data[cat]==category][:]
  x2 = data[cont][~(data[cat]==category)][:]
  
  #calculating descriptives
  n1, n2 = x1.shape[0], x2.shape[0]
  m1, m2 = x1.mean(), x2.mean()
  std1, std2 = x1.std(), x2.mean()
  
  #calculating p-values
  t_p_val = TwoSampT(m1, m2, std1, std2, n1, n2)
  z_p_val = TwoSampZ(m1, m2, std1, std2, n1, n2)

  #table
  table = pd.pivot_table(data=data, values=cont, columns=cat, aggfunc = np.mean)

  #plotting
  plt.figure(figsize = (15,6), dpi=140)
  
  #barplot
  plt.subplot(1,2,1)
  sns.barplot([str(category),'2'.format(category)], [m1, m2])
  plt.ylabel('mean {}'.format(cont))
  plt.xlabel(cat)
  plt.title('t-test p-value = {} \n z-test p-value = {}\n {}'.format(t_p_val,
                                                                z_p_val,
                                                                table))

  # boxplot
  plt.subplot(1,2,2)
  sns.boxplot(x=cat, y=cont, data=data)
  plt.title('categorical boxplot')
Bivariate_cont_cat(data, 'distance', 'vendor_id', 1)
def Bivariate_cont_cat(data, cont, cat, category):
  #creating 2 samples
  x1 = data[cont][data[cat]==category][:]
  x2 = data[cont][~(data[cat]==category)][:]
  
  #calculating descriptives
  n1, n2 = x1.shape[0], x2.shape[0]
  m1, m2 = x1.mean(), x2.mean()
  std1, std2 = x1.std(), x2.mean()
  
  #calculating p-values
  t_p_val = TwoSampT(m1, m2, std1, std2, n1, n2)
  z_p_val = TwoSampZ(m1, m2, std1, std2, n1, n2)

  #table
  table = pd.pivot_table(data=data, values=cont, columns=cat, aggfunc = np.mean)

  #plotting
  plt.figure(figsize = (15,6), dpi=140)
  
  #barplot
  plt.subplot(1,2,1)
  sns.barplot([str(category),'N'.format(category)], [m1, m2])
  plt.ylabel('mean {}'.format(cont))
  plt.xlabel(cat)
  plt.title('t-test p-value = {} \n z-test p-value = {}\n {}'.format(t_p_val,
                                                                z_p_val,
                                                                table))

  # boxplot
  plt.subplot(1,2,2)
  sns.boxplot(x=cat, y=cont, data=data)
  plt.title('categorical boxplot')
Bivariate_cont_cat(data, 'distance', 'store_and_fwd_flag', 'Y')
data.dtypes[data.dtypes == 'category']
def BVA_categorical_plot(data, tar, cat):
  '''
  take data and two categorical variables,
  calculates the chi2 significance between the two variables 
  and prints the result with countplot & CrossTab
  '''
  #isolating the variables
  data = data[[cat,tar]][:]

  #forming a crosstab
  table = pd.crosstab(data[tar],data[cat],)
  f_obs = np.array([table.iloc[0][:].values,
                    table.iloc[1][:].values])

  #performing chi2 test
  from scipy.stats import chi2_contingency
  chi, p, dof, expected = chi2_contingency(f_obs)
  
  #checking whether results are significant
  if p<0.05:
    sig = True
  else:
    sig = False

  #plotting grouped plot
  sns.countplot(x=cat, hue=tar, data=data)
  plt.title("p-value = {}\n difference significant? = {}\n".format(round(p,8),sig))

  #plotting percent stacked bar plot
  #sns.catplot(ax, kind='stacked')
  ax1 = data.groupby(cat)[tar].value_counts(normalize=True).unstack()
  ax1.plot(kind='bar', stacked='True',title=str(ax1))
  int_level = data[cat].value_counts()
# converting passenger_count to integer first, segregating customers into segments, removing passenger_count_group values of 'str'
data['passenger_count'] = data['passenger_count'].astype('int64')
vendor = data[['passenger_count','vendor_id']][:]
vendor['passenger_count_group'] = 'str'
vendor['passenger_count_group'][vendor['passenger_count']<=2] = 'low passenger count'
vendor['passenger_count_group'][(vendor['passenger_count']>2) & (vendor['passenger_count']<7)] = 'high passenger count'
vendor = vendor[vendor.passenger_count_group != 'str']

BVA_categorical_plot(vendor, 'vendor_id', 'passenger_count_group')
# converting passenger_count to integer first, segregating customers into segments, removing passenger_count_group values of 'str'
data['passenger_count'] = data['passenger_count'].astype('int64')
store_forward = data[['passenger_count','store_and_fwd_flag']][:]
store_forward['passenger_count_group'] = 'str'
store_forward['passenger_count_group'][store_forward['passenger_count']<=2] = 'low passenger count'
store_forward['passenger_count_group'][(store_forward['passenger_count']>2) & (store_forward['passenger_count']<7)] = 'high passenger count'
store_forward = store_forward[store_forward.passenger_count_group != 'str']

BVA_categorical_plot(store_forward, 'store_and_fwd_flag', 'passenger_count_group')
BVA_categorical_plot(data, 'vendor_id', 'store_and_fwd_flag')
def Grouped_Box_Plot(data, cont, cat1, cat2):
    # boxplot
    sns.boxplot(x=cat1, y=cont, hue=cat2, data=data, orient='v')
    plt.title('Boxplot')
data['trip_duration_mins_log'] = np.log(data['trip_duration_minutes'].astype('float'))
Grouped_Box_Plot(data,'trip_duration_mins_log', 'passenger_count', 'vendor_id')
Grouped_Box_Plot(data,'trip_duration_mins_log', 'store_and_fwd_flag', 'vendor_id')
Grouped_Box_Plot(data,'trip_duration_mins_log', 'passenger_count', 'store_and_fwd_flag')
data.head()
import folium
def show_fmaps(data, path=1):
    """function to generate map and add the pick up and drop coordinates
    1. Path = 1 : Join pickup (blue) and drop(red) using a straight line
    """
    map_1 = folium.Map(location=[40.8, -74.2], zoom_start=9,tiles='Stamen Toner') # manually added centre
    data.sample(frac=1)
    data_reduced = data.iloc[1:2000]
    for i in range(data_reduced.shape[0]):
        pick_long = data.loc[data.index ==i]['pickup_longitude'].values[0]
        pick_lat = data.loc[data.index ==i]['pickup_latitude'].values[0]
        dest_long = data.loc[data.index ==i]['dropoff_longitude'].values[0]
        dest_lat = data.loc[data.index ==i]['dropoff_latitude'].values[0]
        folium.Marker([pick_lat, pick_long], icon=folium.Icon(color='green',icon='play')).add_to(map_1)
        folium.Marker([dest_lat, dest_long], icon=folium.Icon(color='red',icon='stop')).add_to(map_1)
    return map_1
osm = show_fmaps(data, path=1)
osm
#make a new copy of our data
data_cleaned = data.copy()
data_cleaned.head()
#we shall drop those rows that have passenger_count values of 0,7, and 9. We first change datatype to integer for convenience and then convert it back to category after removal
data_cleaned['passenger_count'] = data_cleaned['passenger_count'].astype('int64')
data_cleaned=data_cleaned[data_cleaned.passenger_count<=6]
data_cleaned=data_cleaned[data_cleaned.passenger_count!=0]
data_cleaned['passenger_count'] = data_cleaned['passenger_count'].astype('category')
data_cleaned['passenger_count'].value_counts()
data_cleaned.shape
#we shall drop those rows that have passenger_count values of 0,7, and 9.
data_cleaned=data_cleaned[data_cleaned.pickup_latitude>=40.6]
data_cleaned=data_cleaned[data_cleaned.pickup_latitude<=40.95]
data_cleaned=data_cleaned[data_cleaned.dropoff_latitude>=40.6]
data_cleaned=data_cleaned[data_cleaned.dropoff_latitude<=40.95]
data_cleaned=data_cleaned[data_cleaned.pickup_longitude>=-74.2]
data_cleaned=data_cleaned[data_cleaned.pickup_longitude<=-73.70]
data_cleaned=data_cleaned[data_cleaned.dropoff_longitude>=-74.2]
data_cleaned=data_cleaned[data_cleaned.dropoff_longitude<=-73.70]
data_cleaned.shape
#Looking at the distributions of the above variables
UVA_numeric(data_cleaned,pickup_dropoff_location)
data_cleaned.distance[data_cleaned.distance == 0].count()
data_cleaned['distance'].loc[data_cleaned['distance']==0] = data.distance.mean()
data_cleaned['distance'].loc[data_cleaned['distance']==0].count()
data_cleaned.shape
#checking how many outlier values will be removed for trip_duration > 3600
data_cleaned.trip_duration[data_cleaned.trip_duration > 3600 ].count()
#removing the outliers and cleaning the dataset
data_cleaned=data_cleaned[data_cleaned.trip_duration<=3600]
#removing trip_duration_mins_log feature as it was used only for data visualization and will not be needed anymore
data_cleaned.drop(["trip_duration_mins_log"], axis = 1, inplace = True)
UVA_numeric(data_cleaned,trip_details)
data_cleaned.trip_duration
data_cleaned.shape
data_cleaned.speed.describe()
#next we shall get rid of outliers in the speed column, i.e speed > 55 and replace speed values of 0 with mean value of 14.39 km/hr
data_cleaned.speed[data_cleaned.speed > 55].count()
data_cleaned=data_cleaned[data_cleaned.speed<=55]
data_cleaned['speed'].loc[data_cleaned['speed']==0] = data.speed.mean()
UVA_numeric(data_cleaned,trip_details)
data_cleaned['speed'].loc[data_cleaned['speed']==0].count()
(len(data)), (len(data_cleaned))
data_cleaned=data_cleaned[data_cleaned.distance<=30]
UVA_numeric(data_cleaned,trip_details)
#removing trip_duration_minutes feature too as it was used only for data visualization and will not be needed anymore
data_cleaned.drop(["trip_duration_minutes"], axis = 1, inplace = True)
(len(data)), (len(data_cleaned))
data_cleaned.describe()
data_cleaned.dtypes
data_cleaned.shape

