import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# set up dataset
number_classes = 7
train_df = pd.read_csv('../input/train.csv')

# lets take a look...
train_df.head()
## normalise dataset
# def normalise_df(df):
#     df_mean = df.mean()
#     df_std = df.std()    
#     df_norm = (df - df_mean) / (df_std)
#     return df_norm, df_mean, df_std

## define columsn to normalise
# cols_to_norm = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
#                 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
#                 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
#                 'Horizontal_Distance_To_Fire_Points']

# norm_df, df_mean, df_std = normalise_df(train_df[cols_to_norm])

## replace columns with normalised versions
# train_df = train_df.drop(cols_to_norm, axis=1)
# train_df = pd.concat([norm_df,train_df], axis=1)

## removed due to un-normalised dataset actually being more accurate!
X = train_df.drop(['Id', 'Cover_Type'], axis=1)
Y = train_df[['Cover_Type']].values
Y = Y.reshape(len(Y))

Xt, Xv, Yt, Yv = train_test_split(X, Y, test_size=0.20)
Xt.head()
Yt.shape
# initialise and train the KNN classifier
knn = KNeighborsClassifier(number_classes, n_jobs=-1)
knn.fit(Xt, Yt)
## evaluate the accuracy
score = knn.score(Xv, Yv)
print('validation score: {}'.format(score))
# create the test dataset
test_df = pd.read_csv('../input/test.csv')
ID_test = test_df['Id'].values
X_test = test_df.drop(['Id'], axis=1)
# use KNN to make a prediction on the test set
y_pred = knn.predict(X_test)

print(min(y_pred))
print(max(y_pred))
sub = pd.DataFrame()
sub['Id'] = ID_test
sub['Cover_Type'] = y_pred
sub.to_csv('my_submission.csv', index=False)
print('good luck!')