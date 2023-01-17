# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df_test_file="../input/test_AV3.csv"
df_train_file="../input/train_AV3.csv"
df_train=pd.read_csv(df_train_file)
df_train

df_train.describe()
df_train.describe(include=['object'])

df_train.isnull().sum()
#dropping missing values of colummn Credit_History
df_train.dropna(subset=['Credit_History'],how='any',inplace=True)

 #Checking
df_train.Credit_History.isnull().sum()
df_train.shape
# mean 'ApplicantIncome'
df_train['ApplicantIncome'].mean()
# mean 'Loan Amount' 
df_train['LoanAmount'].mean()

# mean 'CoapplicantIncome'
df_train['CoapplicantIncome'].mean()
# mean 'Loan_Amount_Term'
df_train['Loan_Amount_Term'].mean()

# median 'ApplicantIncome'
df_train['ApplicantIncome'].median()
# median 'Loan Amount' 
LA_median=df_train['LoanAmount'].median()
print(LA_median)
# median 'CoapplicantIncome'
df_train['CoapplicantIncome'].median()
# median 'Loan_Amount_Term'
df_train['Loan_Amount_Term'].median()
# mode 'Gender'
df_train['Gender'].mode()
# mode 'Married'
df_train['Married'].mode()
# mode 'Dependents'
df_train['Dependents'].mode()
# mode 'Education'
df_train['Education'].mode()
# mode 'Self_Employed'
df_train['Self_Employed'].mode()
# mode 'ApplicantIncome'
df_train['ApplicantIncome'].mode()
# mode 'Loan Amount' 
df_train['LoanAmount'].mode()
# mode 'CoapplicantIncome'
df_train['CoapplicantIncome'].mode()
# mode 'Loan_Amount_Term'
df_train['Loan_Amount_Term'].mode()
# mode 'Property_Area'
df_train['Property_Area'].mode()
# mode 'Loan_Status'
df_train['Loan_Status'].mode()
# dropping numeric columns
df=df_train.drop(['Loan_Amount_Term','LoanAmount'],axis=1)
#filling missing ojects with mode of respective columns
df=df.fillna(df.mode().iloc[0])
# meandf here is going to be Dataframe in which we replace missing numeric values with the mean of their respective columns
meandf=df
meandf['loan_amount']=df_train.LoanAmount.fillna(value=df_train.LoanAmount.mean())
meandf['loan_amount_term']=df_train.Loan_Amount_Term.fillna(value=df_train.Loan_Amount_Term.mean())
# boxplot
meandf.boxplot(column=['ApplicantIncome','CoapplicantIncome','loan_amount','loan_amount_term'],figsize=(10,10))
from matplotlib import pyplot as plt
from matplotlib import style
# histogram plot
plt.hist(meandf['loan_amount'],histtype='bar',rwidth=0.9)
plt.title('Distribution of Loan Amount')
plt.ylabel('Number of people')
plt.xlabel('LoanAmount')



# scatter plot
plt.title('Dependency of loan Amount on Applicant income')
plt.scatter(meandf['loan_amount'],meandf['ApplicantIncome'],color='red',s=25,marker='o',label='Applicant')
plt.scatter(meandf['loan_amount'],meandf['CoapplicantIncome'],color='yellow',s=25,marker='x',label='Coapplicant')
plt.xlabel('Amount of loan taken')
plt.ylabel('Total annual income')
plt.legend
plt.grid(True,color='k')

cols=['ApplicantIncome','CoapplicantIncome','loan_amount','loan_amount_term']
for col in cols:
    q1=meandf[col].describe()[4]#25%
    q3=meandf[col].describe()[6]#75%
    inter_range=q3-q1
    A=q1-1.5*inter_range #lower outlier range
    B=q3+q1-1.5*inter_range#upper outlier range
    print("no of ouliers in column-"+str(col))
    print(str(meandf[(meandf[col]<A)+(meandf[col]>B)][col].count()))


col_series = meandf['ApplicantIncome']
a1=meandf['ApplicantIncome'].describe()[4]#25%
a2=meandf['ApplicantIncome'].describe()[5]#50%
a3=meandf['ApplicantIncome'].describe()[6]#75%
x1=col_series<a1
x2=(col_series>a1)&(col_series<a2)
x3=(col_series<a3)&(col_series>a2)
x4=col_series>a3
col_series[x1]='Lower Class'
col_series[x2]='Lower Middle Class'
col_series[x3]='Upper Middle Class'
col_series[x4]='Upper Class'


# replacing ApplicantIncome column with  col_series with  (Lower Class|Lower Middle Class|Upper Middle Class|Upper Class) 
meandf['ApplicantIncome']=col_series

a=len(meandf.ApplicantIncome[(meandf.Education=='Graduate')&(meandf.ApplicantIncome=='Lower Class')])



b=len(meandf.ApplicantIncome[(meandf.Education=='Not Graduate')&(meandf.ApplicantIncome=='Lower Class')])
    

c=len(meandf.ApplicantIncome[(meandf.Education=='Graduate')&(meandf.ApplicantIncome=='Lower Middle Class')])
d=len(meandf.ApplicantIncome[(meandf.Education=='Not Graduate')&(meandf.ApplicantIncome=='Lower Middle Class')])
e=len(meandf.ApplicantIncome[(meandf.Education=='Graduate')&(meandf.ApplicantIncome=='Upper Middle Class')])
f=len(meandf.ApplicantIncome[(meandf.Education=='Not Graduate')&(meandf.ApplicantIncome=='Upper Middle Class')])
g=len(meandf.ApplicantIncome[(meandf.Education=='Graduate')&(meandf.ApplicantIncome=='Upper Class')])
h=len(meandf.ApplicantIncome[(meandf.Education=='Not Graduate')&(meandf.ApplicantIncome=='Upper Class')])

slices=[a,b,c,d,e,f,g,h]
activities=['Lower Class','Lower Class','Lower Middle Class','Lower Middle Class','Upper Middle Class','Upper Middle Class','Upper Class','Upper Class']
cols=['c','r','c','r','c','r','c','r']
plt.pie(slices,labels=activities,colors=cols,startangle=90,shadow=True,radius=2.5,explode=(0,0.1,0,0.1,0,0.1,0,0.1),autopct='%1.1f%%')

plt.title('pie plot')
plt.show()

# mediandf here is going to be Dataframe in which we replace missing numeric values with the median of their respective columns
df=df.drop(['loan_amount_term','loan_amount'],axis=1)
mediandf=df
mediandf['Loan_amount']=df_train.LoanAmount.fillna(value=df_train.LoanAmount.median())
mediandf['Loan_amount_term']=df_train.Loan_Amount_Term.fillna(value=df_train.Loan_Amount_Term.median())

mediandf.boxplot(column=['CoapplicantIncome'],figsize=(8,8))

mediandf.boxplot(['Loan_amount'])
# histogram plot
X=mediandf.Loan_amount

plt.hist(X,histtype='bar',rwidth=0.9)
plt.title('Applicants v/s Loan Amount')
plt.ylabel('Loan Amount')
plt.xlabel('number of Applicants')
cols=['ApplicantIncome','CoapplicantIncome','Loan_amount','Loan_amount_term']
for col in cols:
    q1=mediandf[col].describe()[4]#25%
    q3=mediandf[col].describe()[6]#75%
    inter_range=q3-q1
    A=q1-1.5*inter_range #lower outlier range
    B=q3+q1-1.5*inter_range#upper outlier range
    print("no of ouliers in column-"+str(col))
    print(str(mediandf[(mediandf[col]<A)+(mediandf[col]>B)][col].count()))
col_series = mediandf['ApplicantIncome']
a1=mediandf['ApplicantIncome'].describe()[4]#25%
a2=mediandf['ApplicantIncome'].describe()[5]#50%
a3=mediandf['ApplicantIncome'].describe()[6]#75%
x1=col_series<a1
x2=(col_series>a1)&(col_series<a2)
x3=(col_series<a3)&(col_series>a2)
x4=col_series>a3

col_series[x1]='Lower Class'
col_series[x2]='Lower Middle Class'
col_series[x3]='Upper Middle Class'
col_series[x4]='Upper Class'

# replacing ApplicantIncome column with  col_series with  (Lower Class|Lower Middle Class|Upper Middle Class|Upper Class) 
mediandf['ApplicantIncome']=col_series

mediandf
mediandf.boxplot(figsize=(10,10))
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import hmean
from scipy.spatial.distance import cdist
from scipy import stats
import numbers


def weighted_hamming(data):
    """ Compute weighted hamming distance on categorical variables. For one variable, it is equal to 1 if
        the values between point A and point B are different, else it is equal the relative frequency of the
        distribution of the value across the variable. For multiple variables, the harmonic mean is computed
        up to a constant factor.

        @params:
            - data = a pandas data frame of categorical variables

        @returns:
            - distance_matrix = a distance matrix with pairwise distance for all attributes
    """
    categories_dist = []
    
    for category in data:
        X = pd.get_dummies(data[category])
        X_mean = X * X.mean()
        X_dot = X_mean.dot(X.transpose())
        X_np = np.asarray(X_dot.replace(0,1,inplace=False))
        categories_dist.append(X_np)
    categories_dist = np.array(categories_dist)
    distances = hmean(categories_dist, axis=0)
    return distances


def distance_matrix(data, numeric_distance = "euclidean", categorical_distance = "jaccard"):
    """ Compute the pairwise distance attribute by attribute in order to account for different variables type:
        - Continuous
        - Categorical
        For ordinal values, provide a numerical representation taking the order into account.
        Categorical variables are transformed into a set of binary ones.
        If both continuous and categorical distance are provided, a Gower-like distance is computed and the numeric
        variables are all normalized in the process.
        If there are missing values, the mean is computed for numerical attributes and the mode for categorical ones.
        
        Note: If weighted-hamming distance is chosen, the computation time increases a lot since it is not coded in C 
        like other distance metrics provided by scipy.

        @params:
            - data                  = pandas dataframe to compute distances on.
            - numeric_distances     = the metric to apply to continuous attributes.
                                      "euclidean" and "cityblock" available.
                                      Default = "euclidean"
            - categorical_distances = the metric to apply to binary attributes.
                                      "jaccard", "hamming", "weighted-hamming" and "euclidean"
                                      available. Default = "jaccard"

        @returns:
            - the distance matrix
    """
    possible_continuous_distances = ["euclidean", "cityblock"]
    possible_binary_distances = ["euclidean", "jaccard", "hamming", "weighted-hamming"]
    number_of_variables = data.shape[1]
    number_of_observations = data.shape[0]

    # Get the type of each attribute (Numeric or categorical)
    is_numeric = [all(isinstance(n, numbers.Number) for n in data.iloc[:, i]) for i, x in enumerate(data)]
    is_all_numeric = sum(is_numeric) == len(is_numeric)
    is_all_categorical = sum(is_numeric) == 0
    is_mixed_type = not is_all_categorical and not is_all_numeric

    # Check the content of the distances parameter
    if numeric_distance not in possible_continuous_distances:
        print ("The continuous distance " + numeric_distance + " is not supported.")
        return None
    elif categorical_distance not in possible_binary_distances:
        print ("The binary distance " + categorical_distance + " is not supported.")
        return None

    # Separate the data frame into categorical and numeric attributes and normalize numeric data
    if is_mixed_type:
        number_of_numeric_var = sum(is_numeric)
        number_of_categorical_var = number_of_variables - number_of_numeric_var
        data_numeric = data.iloc[:, is_numeric]
        data_numeric = (data_numeric - data_numeric.mean()) / (data_numeric.max() - data_numeric.min())
        data_categorical = data.iloc[:, [not x for x in is_numeric]]

    # Replace missing values with column mean for numeric values and mode for categorical ones. With the mode, it
    # triggers a warning: "SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame"
    # but the value are properly replaced
    if is_mixed_type:
        data_numeric.fillna(data_numeric.mean(), inplace=True)
        for x in data_categorical:
            data_categorical[x].fillna(data_categorical[x].mode()[0], inplace=True)
    elif is_all_numeric:
        data.fillna(data.mean(), inplace=True)
    else:
        for x in data:
            data[x].fillna(data[x].mode()[0], inplace=True)

    # "Dummifies" categorical variables in place
    if not is_all_numeric and not (categorical_distance == 'hamming' or categorical_distance == 'weighted-hamming'):
        if is_mixed_type:
            data_categorical = pd.get_dummies(data_categorical)
        else:
            data = pd.get_dummies(data)
    elif not is_all_numeric and categorical_distance == 'hamming':
        if is_mixed_type:
            data_categorical = pd.DataFrame([pd.factorize(data_categorical[x])[0] for x in data_categorical]).transpose()
        else:
            data = pd.DataFrame([pd.factorize(data[x])[0] for x in data]).transpose()

    if is_all_numeric:
        result_matrix = cdist(data, data, metric=numeric_distance)
    elif is_all_categorical:
        if categorical_distance == "weighted-hamming":
            result_matrix = weighted_hamming(data)
        else:
            result_matrix = cdist(data, data, metric=categorical_distance)
    else:
        result_numeric = cdist(data_numeric, data_numeric, metric=numeric_distance)
        if categorical_distance == "weighted-hamming":
            result_categorical = weighted_hamming(data_categorical)
        else:
            result_categorical = cdist(data_categorical, data_categorical, metric=categorical_distance)
        result_matrix = np.array([[1.0*(result_numeric[i, j] * number_of_numeric_var + result_categorical[i, j] *
                               number_of_categorical_var) / number_of_variables for j in range(number_of_observations)] for i in range(number_of_observations)])

    # Fill the diagonal with NaN values
    np.fill_diagonal(result_matrix, np.nan)

    return pd.DataFrame(result_matrix)


def knn_impute(target, attributes, k_neighbors, aggregation_method="mean", numeric_distance="euclidean",
               categorical_distance="jaccard", missing_neighbors_threshold = 0.5):
    """ Replace the missing values within the target variable based on its k nearest neighbors identified with the
        attributes variables. If more than 50% of its neighbors are also missing values, the value is not modified and
        remains missing. If there is a problem in the parameters provided, returns None.
        If to many neighbors also have missing values, leave the missing value of interest unchanged.

        @params:
            - target                        = a vector of n values with missing values that you want to impute. The length has
                                              to be at least n = 3.
            - attributes                    = a data frame of attributes with n rows to match the target variable
            - k_neighbors                   = the number of neighbors to look at to impute the missing values. It has to be a
                                              value between 1 and n.
            - aggregation_method            = how to aggregate the values from the nearest neighbors (mean, median, mode)
                                              Default = "mean"
            - numeric_distances             = the metric to apply to continuous attributes.
                                              "euclidean" and "cityblock" available.
                                              Default = "euclidean"
            - categorical_distances         = the metric to apply to binary attributes.
                                              "jaccard", "hamming", "weighted-hamming" and "euclidean"
                                              available. Default = "jaccard"
            - missing_neighbors_threshold   = minimum of neighbors among the k ones that are not also missing to infer
                                              the correct value. Default = 0.5

        @returns:
            target_completed        = the vector of target values with missing value replaced. If there is a problem
                                      in the parameters, return None
    """

    # Get useful variables
    possible_aggregation_method = ["mean", "median", "mode"]
    number_observations = len(target)
    is_target_numeric = all(isinstance(n, numbers.Number) for n in target)

    # Check for possible errors
    if number_observations < 3:
        print ("Not enough observations.")
        return None
    if attributes.shape[0] != number_observations:
        print ("The number of observations in the attributes variable is not matching the target variable length.")
        return None
    if k_neighbors > number_observations or k_neighbors < 1:
        print ("The range of the number of neighbors is incorrect.")
        return None
    if aggregation_method not in possible_aggregation_method:
        print ("The aggregation method is incorrect.")
        return None
    if not is_target_numeric and aggregation_method != "mode":
        print ("The only method allowed for categorical target variable is the mode.")
        return None

    # Make sure the data are in the right format
    target = pd.DataFrame(target)
    attributes = pd.DataFrame(attributes)

    # Get the distance matrix and check whether no error was triggered when computing it
    distances = distance_matrix(attributes, numeric_distance, categorical_distance)
    if distances is None:
        return None

    # Get the closest points and compute the correct aggregation method
    for i, value in enumerate(target.iloc[:, 0]):
        if pd.isnull(value):
            order = distances.iloc[i,:].values.argsort()[:k_neighbors]
            closest_to_target = target.iloc[order, :]
            missing_neighbors = [x for x  in closest_to_target.isnull().iloc[:, 0]]
            # Compute the right aggregation method if at least more than 50% of the closest neighbors are not missing
            if sum(missing_neighbors) >= missing_neighbors_threshold * k_neighbors:
                continue
            elif aggregation_method == "mean":
                target.iloc[i] = np.ma.mean(np.ma.masked_array(closest_to_target,np.isnan(closest_to_target)))
            elif aggregation_method == "median":
                target.iloc[i] = np.ma.median(np.ma.masked_array(closest_to_target,np.isnan(closest_to_target)))
            else:
                target.iloc[i] = stats.mode(closest_to_target, nan_policy='omit')[0][0]

    return target


knn_impute(target=df_train['LoanAmount'], attributes=df_train.drop(['LoanAmount', 'ApplicantIncome'], 1),
                                    aggregation_method="median", k_neighbors=10, numeric_distance='euclidean',
                                    categorical_distance='hamming', missing_neighbors_threshold=0.8)