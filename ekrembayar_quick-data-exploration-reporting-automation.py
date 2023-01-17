import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt



# Configuration

import warnings

warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)



pd.set_option('display.max_columns', None)

# Nümerik Değerlerin Düzgün Gösterimi

pd.options.display.float_format = '{:.2f}'.format
df = pd.read_csv("../input/home-data-for-ml-course/train.csv")

df.head()
df.shape
def cat_summary(data, cat_length, plot = False):



    print("# How many classes are there in variables?")

    print("----------------------------------------------------------------------------------")

    

    cat_names_less = [col for col in data.columns if len(data[col].unique()) < cat_length]

    cat_names_more = [col for col in data.columns if len(data[col].unique()) >= cat_length]

    print('Number of Classes <', str(cat_length) + ":", cat_names_less)

    print('Number of Classes >=', str(cat_length) + ":", cat_names_more, "\n\n")

    

    print("# Which variables are object variables that number of classes is more than", str(cat_length) + "?")

    print("----------------------------------------------------------------------------------")

    object_variables = [col for col in cat_names_more if data[col].dtype == "O"]

    more = [col for col in object_variables if len(data[col].unique()) >= cat_length]

    print('Number of Classes >', str(cat_length) + ":", more, "\n\n")

    

    print("# Number of Unique Classes")

    print("----------------------------------------------------------------------------------")

    for  i in data.columns:

        print(i, "(",str(data[i].dtypes),"):",data[i].nunique())

    print("\n\n")

    

    

    if plot:

        print("# Stats & Visualization")

    else:

        print("# Stats")

    print("-----------------------------------------------------------------------------------")

    

    for col in cat_names_less:

        print(pd.DataFrame({col: data[col].value_counts(),

                           "Ratio": 100 * data[col].value_counts()/ len (data)}), end = "\n\n\n")

        

        if plot:

            sns.countplot(x = col, data = data)

            plt.show()

        print("#------------------------------------------------------------------------------ \n")

          

    





    

cat_summary(df, cat_length = 10, plot = True)
def num_plot(data, cat_length = 10, remove = ["Id"], hist_bins = 10, figsize = (20,4)):

    

    num_cols = [col for col in data.columns if df[col].dtypes != "O" 

                and len(data[col].unique()) >= cat_length]

    

    if len(remove) > 0:

        num_cols = list(set(num_cols).difference(remove))

        

    print("# Histogram & Boxplot \n")    

    for i in num_cols:

        fig, axes = plt.subplots(1, 3, figsize = figsize)

        data.hist(str(i), bins = hist_bins, ax=axes[0])

        data.boxplot(str(i),  ax=axes[1], vert=False);

        try: 

            sns.kdeplot(np.array(data[str(i)]))

        except: ValueError

        

        axes[1].set_yticklabels([])

        axes[1].set_yticks([])

        axes[0].set_title(i + " | Histogram")

        axes[1].set_title(i + " | Boxplot")

        axes[2].set_title(i + " | Density")

        plt.show()

        

        

num_plot(df, cat_length = 10, remove = ["Id"], hist_bins = 10, figsize = (20,4))
def relationship(data, cat_length = 10, target = "SalePrice",remove = ["Id", "SalePrice"], figsize = (15,4)):

    

    num_cols = [col for col in data.columns if data[col].dtypes != "O" 

                and len(data[col].unique()) >= cat_length]



    if len(remove) > 0:

        num_cols = list(set(num_cols).difference(remove + [target]))

    

    for i in num_cols:

        

        correlation = data[[target, i]].corr().loc[target][1]

        

        plt.figure()

        sns.scatterplot(x=i, y=target, data=data)

        plt.title("Correlation: " + str(correlation)[:5])

        plt.show();

        

        

relationship(df, cat_length = 10, target = "SalePrice",remove = ["Id"], figsize = (15,4))
def outlier_detection(data, analysis_type = "detect", cat_length = 10, remove = ["PassengerId"], plot = False, method = "nothing"):

    num_names = [col for col in df.columns if len(data[col].unique()) > 10

             and df[col].dtypes != 'O']

    if len(remove) > 0:

        num_names = list(set(num_names).difference(remove))

        

    # Outlier Threshold belirlenir

    def outlier_thresholds(dataframe, variable):

        quartile1 = dataframe[variable].quantile(0.25)

        quartile3 = dataframe[variable].quantile(0.75)

        interquantile_range = quartile3 - quartile1

        up_limit = quartile3 + 1.5 * interquantile_range

        low_limit = quartile1 - 1.5 * interquantile_range

        return low_limit, up_limit 

    

    # Remove Outliers

    def remove_outliers(dataframe, variable):

        low_limit, up_limit = outlier_thresholds(dataframe, variable)

        df_without_outliers = dataframe[~((dataframe[variable] < low_limit) | (dataframe[variable] > up_limit))]

        return df_without_outliers

    # Replace with Thresholds

    def replace_with_thresholds(dataframe, variable):

        low_limit, up_limit = outlier_thresholds(dataframe, variable)

        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit

        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

    

    

    if analysis_type == "detect":

        

        variable_names = []

        lowerUpper = []

        print("# How many outlier values are there?")

        print("--------------------------------------------------------------------")

        for col in num_names:

            low_limit, up_limit = outlier_thresholds(data, col)

        

            lowerUpper.append((low_limit, up_limit))    

        

            if data[(data[col] > up_limit) | (data[col] < low_limit)].any(axis=None):

                number_of_outliers = data[(data[col] > up_limit) | (data[col] < low_limit)].shape[0]

                variable_names.append(col)

            

                print(col, ":", number_of_outliers)

            

            

        print("\n")

        print("Outlier Variables:", variable_names, "\n\n")

    

        print("# Lower & Upper Thresholds (IQR)")

        print("----------------------------------------------------------------------")

        print(pd.DataFrame({"Variable":num_names, "Lower, Upper":lowerUpper}), "\n\n")

    

        if plot:

            print("# Boxplot")

            print("------------------------------------------------------------------")

            for col in num_names:

                plt.figure(figsize = (15,4))

                sns.boxplot(x=data[col])

                plt.title(col + " Boxplot")

                plt.show()

            

            

    if analysis_type == "analyze":  

        if method == "remove":

            for col in num_names:

                data = remove_outliers(data, col)           

        elif method == "replace":

            for col in num_names:

                replace_with_thresholds(data,col)

    

        return data

       

    

        

outlier_detection(df, analysis_type = "detect", cat_length = 10, remove = ["Id"], plot = True)
# Remove

outlier_detection(df, analysis_type = "analyze", method = "remove",

                  cat_length = 10, remove = ["Id"], plot = False).shape
# Replace

outlier_detection(df, analysis_type = "analyze", method = "replace",

                  cat_length = 10, remove = ["Id"], plot = False).shape
def quick_missing_imp(data, num_method = "mean", cat_length = 10):

    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]

    

    print("# BEFORE")

    print(data[variables_with_na].isnull().sum(), "\n\n")

    

    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

        

    if num_method == "mean":

        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

    elif num_method == "median":

        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

       

    print("# AFTER \n Imputation method is '" + num_method.upper() + "' for numeric variables! \n")

    print(data[variables_with_na].isnull().sum(), "\n\n")

        

    return data

        

imputed = quick_missing_imp(df, num_method = "mean", cat_length = 10)

imputed.head(3)