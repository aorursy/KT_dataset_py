import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from math import sqrt
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#Print all rows and columns. Dont hide any

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)



train.head(3)
print(train.shape, test.shape)
train.describe()
#Checking missing values

print('Is the training data contains any missing values? ' + str(train.isnull().any().any()) + '\n'

     + 'Is the testing data contains any missing values? ' + str(test.isnull().any().any()))
train_list_name = list(train.columns.values)

train_list_name.pop() #pop out the loss column

test_list_name = list(test.columns.values)

print('Are the columns identical to each other for both train & test dataset? ' + str(train_list_name == test_list_name))
#Check column values for each categorical columns

def showunique(df):

    list_name = list(train.columns.values)

    for i, col_name in enumerate(list_name):

        if col_name[:3] == 'cat':

            print(df.groupby('cat' + str(i))['id'].nunique())
showunique(train)
#Separate the dataset, starting from column index 117

train_cont = train.iloc[:, 117:]

test_cont = test.iloc[:, 117:]
train_cont.head(3)
test_cont.head(3)
#Checking the skewness of the remaining dataset, the ones close to 0 are less skewed data

print(train_cont.skew())
#Heatmap to check the correlation

cor = train_cont.corr()

f, ax = plt.subplots(figsize = (12, 8))

sns.heatmap(cor, vmax = 0.9, annot = True, square = True, fmt = '.2f')
#Let us apply PCA

from sklearn.decomposition import PCA

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler



loss = train_cont.loc[:, train_cont.columns == 'loss']

train_cont_phase = train_cont.loc[:, train_cont.columns != 'loss']



scaled_train_cont = StandardScaler().fit_transform(train_cont_phase)

scaled_test_cont = StandardScaler().fit_transform(test_cont)
#Do the PCA

pca = PCA()

pca.fit(scaled_train_cont)

pca.data = pca.transform(scaled_train_cont)
#Percentage variance of each pca component stands for

per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1)

#Create labels for the scree plot

labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
#Plot the data

plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label = labels)

plt.ylabel('percentage of Explained Variance')

plt.xlabel('Principle Component')

plt.title('Scree plot')

plt.show()
variance = 0

count = 0

for i in pca.explained_variance_ratio_:

    if variance <= 95:

        variance += i * 100

        count+=1

print(str(np.round(variance, 2)) + '% of the variance is explained by ' + str(count) + ' of Principle Components')
#Extract the PC1 through PC9 information

train_append = pd.DataFrame(data=pca.data[:,:9], columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'

                                                           , 'PC8', 'PC9'])
train_append.head(3)
#Glue the PC data back to the training dataset

new_train = pd.concat((train.iloc[:, :117], train_append), axis = 1)

new_train.head(3)
#Now performing the same action for the testing dataset

pca.fit(scaled_test_cont)

pca.data = pca.transform(scaled_test_cont)

test_append = pd.DataFrame(data=pca.data[:,:9], columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'

                                                           , 'PC8', 'PC9'])

new_test = pd.concat((test.iloc[:, :117], test_append), axis = 1)

new_test.head(3)
#Check the distributions of the catevariables:

# Count of each label in each category



#names of all the columns

cols = new_train.columns



#Plot count plot for all attributes in a 29x4 grid

n_cols = 4

n_rows = 29

for i in range(n_rows):

    fg,ax = plt.subplots(nrows=1,ncols=n_cols,sharey=True,figsize=(12, 8))

    for j in range(n_cols):

        sns.countplot(x=cols[i*n_cols+j+1], data=new_train, ax=ax[j])
#Show dominanting percentage less than 2%, and drop them during the process

def show_and_drop_percentage(df, df2):

    for i in range(1, 117):

        A = df['cat' + str(i)].value_counts()

        per = sum(A[1:]) / sum(A) * 100

        if per < 2:

            print('cat' + str(i) + ': ' + 'Dominating percentage is: ' + str(np.round(per, 2)) + '%')

            df = df.drop(['cat' + str(i)], axis = 1)

            df2 = df2.drop(['cat' + str(i)], axis = 1)

    print('-' * 80 + '\n')

    print('Cleaning complete for columns cat1 to cat 116, The above categories had been dropped\n')

    return df, df2
#Operate on the training set

removed_train, removed_test = show_and_drop_percentage(new_train, new_test)
removed_train.head(3)
removed_test.head(3)
#Check if the same procedure was done on the train & test columns

any(removed_train.columns == removed_test.columns)
#Remove the id columns

removed_train = removed_train.iloc[:, 1:]

removed_test = removed_test.iloc[:, 1:]
removed_train.head(3)
#range of features considered

split = 78



#Grab out the categorical variables

cat_train = removed_train.iloc[:, :split]

cat_test = removed_test.iloc[:, :split]



#List the column names

cols = cat_train.columns



#Variable to hold the list of variables for an attribute in the train and test data

labels = []



for i in range(0,split):

    train = cat_train[cols[i]].unique()

    test = cat_test[cols[i]].unique()

    labels.append(list(set(train) | set(test))) 
#One hot encode all categorical attributes 



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder





cats = []

for i in range(0, split):

    #Label encode

    label_encoder = LabelEncoder()

    label_encoder.fit(labels[i])

    feature = label_encoder.transform(cat_train.iloc[:,i])

    feature = feature.reshape(cat_train.shape[0], 1)

    #One hot encode

    onehot_encoder = OneHotEncoder(sparse=False,categories= [range(len(labels[i]))])

    feature = onehot_encoder.fit_transform(feature)

    cats.append(feature)

    

# Make a 2D array from a list of 1D arrays

encoded_cats = np.column_stack(cats)



# Print the shape of the encoded data

print(encoded_cats.shape)



#Concatenate encoded attributes with continuous attributes

train_encoded = np.concatenate((encoded_cats,removed_train.iloc[:,split:].values),axis=1)



#Transfer it back into pandas dataframe

train_encoded = pd.DataFrame(data=train_encoded)

train_encoded.head(3)
#One hot encode all categorical attributes

cats = []

for i in range(0, split):

    #Label encode

    label_encoder = LabelEncoder()

    label_encoder.fit(labels[i])

    feature = label_encoder.transform(cat_test.iloc[:,i])

    feature = feature.reshape(cat_test.shape[0], 1)

    #One hot encode

    onehot_encoder = OneHotEncoder(sparse=False,categories= [range(len(labels[i]))])

    feature = onehot_encoder.fit_transform(feature)

    cats.append(feature)

    

# Make a 2D array from a list of 1D arrays

encoded_cats = np.column_stack(cats)



# Print the shape of the encoded data

print(encoded_cats.shape)



#Concatenate encoded attributes with continuous attributes

test_encoded = np.concatenate((encoded_cats,removed_test.iloc[:,split:].values),axis=1)



test_encoded = pd.DataFrame(data=test_encoded)

test_encoded.head(3)
#First of all, we do train test split of our dataset

from sklearn.model_selection import train_test_split



#Set our random seed to ensure productive result

seed = 2019



X_train, X_test, y_train, y_test = train_test_split(

     train_encoded, loss, test_size=0.25, random_state=seed)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train) 

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(X_train_scaled, y_train)



#Calculating the MAE

lin_pred = lin_reg.predict(X_test_scaled)

lin_result = mean_absolute_error(y_test, lin_pred)

lin_result
from sklearn.linear_model import ElasticNet



ela = ElasticNet(random_state=seed)

ela.fit(X_train_scaled, y_train)



ela_pred = ela.predict(X_test_scaled)

ela_result = mean_absolute_error(y_test, ela_pred)

ela_result
#Make predictions using the model

#Write it to the file

Test_scaled = scaler.transform(test_encoded)



predictions = ela.predict(Test_scaled)



pd.DataFrame(predictions, columns = ['loss']).to_csv('submission.csv')
from sklearn.linear_model import SGDRegressor



sgd = SGDRegressor(max_iter = 1500, eta0=1e-14,

                  learning_rate = 'adaptive',

                  penalty = 'elasticnet')

sgd.fit(X_train_scaled, y_train)



sgd_pred = sgd.predict(X_test_scaled)

sgd_result = mean_absolute_error(y_test, sgd_pred)

sgd_result
sgd_pred