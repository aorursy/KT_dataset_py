import pandas as pd

# Read the dataset
df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.info()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# transform_categorical
#   - Function to automatically transform categorical features.
# input  : 
#   - df_input (dataframe) : dataset
#   - exclude (list) : list of feature name that will be excluded from transformation
# output : 
#   - df (dataframe) : numerical-encoded dataset

def transform_categorical(df_input, exclude=[]) :
    df = df_input.copy()
    # Dictionary to put the label encoder & one hot encoder
    l_encoderDict = dict()
    o_encoderDict = dict()
    
    # Iterate column in dataframe
    columns = list(df)
    for ori_column in columns :
        # Get one sample of the data to determine data type
        sample = df[ori_column][0]
        # Check whether the data is str or subtype of str
        if isinstance(sample, str) and ori_column not in exclude:
            # Initialize new column name
            new_column = ori_column + '_encoded'
            # Label encoding
            l_encoderDict[ori_column] = LabelEncoder()
            df[new_column] = l_encoderDict[ori_column].fit_transform(df[ori_column])
            # Delete old column
            df.drop(columns=[ori_column], inplace=True)
            # Check whether the numerical values are not binary
            if len(l_encoderDict[ori_column].classes_) > 2 :
                # One hot encoding
                o_encoderDict[ori_column] = OneHotEncoder()
                oneHot_temp = df[new_column].values.reshape(-1,1)
                oneHot_array = o_encoderDict[ori_column].fit_transform(oneHot_temp).toarray()
                # Initializa new column name
                oneHot_columns = [ori_column + '_' + str(j) for j in range(oneHot_array.shape[1])]
                # Convert one hot encoding array to dataframe
                dfOneHot = pd.DataFrame(oneHot_array, columns=oneHot_columns)
                # Add one hot encoding to existing dataframe
                df = pd.concat([df, dfOneHot], axis=1)
                # Delete old column
                df.drop(columns=[new_column], inplace=True)
    return df

# Use of the subroutine on Telco dataset
df_encoded = transform_categorical(df, exclude=['customerID', 'TotalCharges'])
# Original dataset
df.head(5)
# Encoded dataset
df_encoded.head(5)
df_encoded.describe()