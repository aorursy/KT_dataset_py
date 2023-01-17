import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn import preprocessing



data = pd.read_csv('../input/train.csv')



pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 10000)



def check_all(df):

    print(df.describe())

    check_na(df)

    plot_hist(df)



def plot_hist(df):

    df.plot.hist(bins=10)

    plt.show()

    plt.close()



def check_na(df):

    #print("there are NA values --> " + str(df.isnull().values.any()), "in column", df.columns)

    if df.isnull().values.any() == True:

        return True

    else:

        return False



def corr_matrix(df, size):

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(size, size))

    ax.matshow(corr, cmap='hot')

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.show()

    plt.close()



def binn(df,num_bins):

    index = (df.max() - df.min())/num_bins

    bin = [0]

    counter_index = df.min()

    x = 0

    while x < num_bins+1:

        bin.append(counter_index)

        counter_index = counter_index + index

        x=x+1

    bin.append(1000000000)



    group_names = []

    y=0

    while y<len(bin)-1:

        group_names.append(y)

        y = y+1



    return bin, group_names



def fillMedian(df):

    return df.fillna(df.median())



def cat_na(df):

    return df.fillna('N/A')



def fillMostCommonCat(df):

    df_common = df.value_counts().index[0]

    return df.fillna(df_common)



def process_data(data):

    #check for NA

    NA_columns = []

    for x in data:

        if check_na(data[x]) == True:

            NA_columns.append(x)



    for x in NA_columns:

        perc_nan = round((data[x].isna().sum()/len(data))*100)

        #print(x, perc_nan, '% NAN')

        #print(data[x].head())



    Median_NA = ['LotFrontage','MasVnrArea','GarageYrBlt']



    Categorize_NA = ['Alley','FireplaceQu', 'Fence', 'MiscFeature']



    Random_Cat_NA = ['MasVnrType','BsmtQual', 'BsmtCond','BsmtExposure','BsmtFinType1',

                       'BsmtFinType2','Electrical','GarageType','GarageQual','GarageCond','GarageFinish']



    dropping_column = ["Id", 'PoolQC']



    for x in Median_NA:

        data[x] = fillMedian(data[x])



    for x in Categorize_NA:

        data[x] = cat_na(data[x])



    for x in Random_Cat_NA:

        data[x] = fillMostCommonCat(data[x])



    data = data.drop(dropping_column, axis=1)



    #convert categorized data into numeric data

    for x in data:

        if data[x].dtypes == 'object':

           data[x] = pd.Categorical(data[x])

           data[x] = data[x].cat.codes

    return data



y = data.SalePrice

data = data.drop('SalePrice', axis=1)

print(data.head())

data = process_data(data)



low_corr_drop = []

#drop the columns with low correlation between saleprice

for x in data:

    corr_withy = int(data[x].corr(y)*100)

    if corr_withy < 5 and corr_withy > -5:

        low_corr_drop.append(x)

data = data.drop(low_corr_drop, axis=1)

d_columns_preTrain = data.columns



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data,y,test_size=0.1)

from sklearn import preprocessing

min_max_optimization = preprocessing.MinMaxScaler()

X_test = min_max_optimization.fit_transform(X_test)

X_train = min_max_optimization.fit_transform(X_train)



from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.optimizers import SGD



model_choice = 1



if model_choice == 1:

    model = Sequential()

    model.add(Dense(65, input_dim=65, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model

    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')

    H = model.fit(X_train,y_train,nb_epoch=1130)

    model.save('Model_final2')
from keras.models import load_model

model = load_model('Model_final2')



a = model.predict(X_test)

a = pd.DataFrame(a, columns=['predicted'])

a = a.reset_index(drop=True)

y_test = y_test.reset_index(drop=True)

final = pd.concat([a, y_test], axis=1)

final['diff'] = final['predicted'] - final['SalePrice']

score = round((abs(final['diff'])).mean())

score = "{:,}".format(score)

print(final)

print(score)
final_sub = pd.read_csv('../input/test.csv')

finalID = final_sub.Id

final_sub = process_data(final_sub)

drop1 = []

for x in final_sub:

    if x not in d_columns_preTrain:

        drop1.append(x)

final_sub = final_sub.drop(drop1,axis=1)

final_sub = min_max_optimization.fit_transform(final_sub)

a = model.predict(final_sub)

a = pd.DataFrame(a, columns = ['SalePrice'])



my_submission = pd.concat([finalID, a], axis=1)

my_submission.to_csv('submission.csv', index=False)
