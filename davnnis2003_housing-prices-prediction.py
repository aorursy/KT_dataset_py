import pandas as pd



#First to load the csv file into the runtime

df_train = pd.read_csv('../input/train.csv')

df_train.head()
# Have a quick & basic understanding of the dataset

print ('Columns in the dataset:')

print (df_train.columns)

print ('')

print ('Shape of the dataset: ')

print (df_train.shape)

print ('')

print ('Quick stats of the prediction targe')

print (df_train['SalePrice'].describe())
import seaborn as sns



#Visualization of the distribtion of the prediction target

sns.kdeplot(df_train['SalePrice'])
import pandas as pd



df_test = pd.read_csv('../input/test.csv')

df_test.head()
# Have a quick & basic understanding of the Testing dataset

print ('Columns in the dataset:')

print (df_test.columns)

print ('')

print ('Shape of the dataset: ')

print (df_test.shape)
print ('Training dataset with NULLs:')

print (df_train.isnull()

        .sum()

        .sort_values(ascending = False)

        .loc[lambda x : x >0]) # only get those columns with NULL values

print ('')

print ('Test dataset with NULLs:')

print (df_test.isnull()

        .sum()

        .sort_values(ascending = False)

        .loc[lambda x : x >0]) # only get those columns with NULL values
df_train_cleaned = df_train.fillna('None')

df_train_cleaned.head()
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier





y = df_train_cleaned['SalePrice'] #Prediction Target

feature_names = [i for i in df_train_cleaned.columns if df_train_cleaned[i].dtype in [np.int64]]

X = df_train_cleaned[feature_names].drop(columns=['Id','SalePrice'])

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)



print ('Traning completed')
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = val_X.columns.tolist())
from sklearn.metrics import mean_absolute_error

# Train the model with the Features with high importance



y = df_train_cleaned['SalePrice'] #Prediction Target

feature_names = ['FullBath',

                'MoSold',

                'HalfBath',

                'OpenPorchSF',

                'BsmtFinSF1',

                'GarageArea',

                'GarageCars']





X = df_train_cleaned[feature_names]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)



print ('MAE: ' + str(mean_absolute_error(val_y, my_model.predict(val_X))))
# Transform the results into desired format before exporting

result = my_model.predict(val_X)

df_result = pd.Series(result)

df_result = pd.concat([df_result, df_train['Id']], axis=1)

df_result.columns = ['SalePrice','Id']

df_result = (df_result[['Id', 'SalePrice']]

             .set_index('Id')

             .drop([1460])

             .fillna(0)

            )



df_result.head()
# Export the result as a csv file

print('result shape: ' + str(df_result.shape))

print('Exporting...')

df_result.to_csv(path_or_buf='submission.csv', index=False)

print('Exporting Complete!')