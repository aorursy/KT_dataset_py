import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn
apps = pd.read_csv(r"../input/google-play-store-apps/googleplaystore.csv")
print(apps.head())

print(apps.tail())

apps.info()
apps['Rating'] = apps.Rating.fillna(0) # removing nulls in Rating

#apps = apps.drop(index=9148) # removing nulls in Type

apps['Content Rating'] = apps['Content Rating'].fillna("0") # removing nulls in COntent Rating

apps['Current Ver'] = apps['Current Ver'].fillna("1.0.0")# removing nulls in Current Ver

apps['Android Ver'] = apps['Android Ver'].fillna("Varies with device") # removing nulls in Anroid Ver
apps['Reviews'] = apps.Reviews.str.replace("M","000000")

apps['Reviews'] = pd.to_numeric(apps.Reviews)



apps['Size'] = apps['Size'].str.replace("Varies with device","0")

apps['Size'] = apps['Size'].str.replace("M","")

apps['Size'] = apps['Size'].str.replace("k","000")

apps['Size'] = apps['Size'].str.replace(",","")

apps['Size'] = apps['Size'].str.replace("+","")

apps['Size'] = pd.to_numeric(apps['Size'])



apps['Installs'] = apps['Installs'].str.replace(",","")

apps['Installs'] = apps['Installs'].str.replace("+","")

apps['Installs'] = apps.Installs.str.replace("Free","")

apps['Installs'] = pd.to_numeric(apps.Installs)



apps['Price'] = apps.Price.str.replace("Everyone","")

apps['Price'] = apps.Price.str.replace("$","")

apps['Price'] = pd.to_numeric(apps['Price'])



apps['Content Rating'] = apps['Content Rating'].str.replace("Unrated","Everyone")

apps['Content Rating'] = apps['Content Rating'].str.replace("0","Everyone")

apps =  apps.drop_duplicates(subset="App").reset_index()
# credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df

reduce_mem_usage(apps)
apps['Rating'] = apps.Rating.astype("category")