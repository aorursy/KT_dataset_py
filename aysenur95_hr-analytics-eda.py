# import packages

import csv

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import seaborn as sns
#read data

train_features_data = pd.read_csv('../input/hr-dataset/train_LZdllcl.csv')

test_features_data = pd.read_csv('../input/hr-dataset/test_2umaH9m.csv')
train_features_data.head()
test_features_data.head()
train_features_data['education'].value_counts().plot(kind='barh')
train_features_data['education'].value_counts()
sns.set_style("dark")

sns.countplot(x="education", data=train_features_data, palette=sns.color_palette("husl", 8), saturation=10)
sns.set_style("dark")

sns.countplot(x="education", data=train_features_data, palette=sns.color_palette("husl", 8), saturation=10, hue="is_promoted")
sns.set(rc={"font.style":"normal",

            "text.color":"black",

            "xtick.color":"black",

            "ytick.color":"black",

            "axes.labelcolor":"black",

            "axes.grid":False,

            'axes.labelsize':30,

            'figure.figsize':(12.0, 6),

            'xtick.labelsize':25,

            'ytick.labelsize':20})



sns.set(style="white",font_scale=1)





sns.set_style("dark")

sns.countplot(x="department", data=train_features_data, palette=sns.color_palette("husl", 8), 

              saturation=10, edgecolor=(0,0,0), linewidth=2)
sns.set(rc={"font.style":"normal",

            "text.color":"black",

            "xtick.color":"black",

            "ytick.color":"black",

            "axes.labelcolor":"black",

            "axes.grid":False,

            'axes.labelsize':30,

            'figure.figsize':(12, 6),

            'xtick.labelsize':25,

            'ytick.labelsize':20})





sns.set(style="white",font_scale=1)



sns.set_style("dark")

sns.countplot(x="department", data=train_features_data, palette=sns.color_palette("husl", 8), 

              saturation=10, edgecolor=(0,0,0), linewidth=2, hue="is_promoted")
train_features_data["department"].unique()
train_features_data["department"].value_counts()
# library

import matplotlib.pyplot as plt

from palettable.colorbrewer.qualitative import Pastel1_7



# create data

names=list(train_features_data["department"].unique())

sizes=[train_features_data["department"].value_counts()[unique_class]*100/len(train_features_data["department"]) for unique_class in names]

colors = Pastel1_7.hex_colors

explode = (0, 0, 0, 0, 0, 0, 0, 0, 0)  # explode a slice if required



plt.pie(sizes, explode=explode, labels=names, colors=colors,

        autopct='%1.1f%%', shadow=True)

        

#draw a circle at the center of pie to make it look like a donut

centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)

fig = plt.gcf()

fig.gca().add_artist(centre_circle)





# Set aspect ratio to be equal so that pie is drawn as a circle.

plt.axis('equal')

plt.show()



train_features_data.count()
train_features_data.info()
train_features_data.describe()
print("   # of unique values for each column")

print("***************************************")

for column in train_features_data.columns:

    print(f"{column} --> {train_features_data[column].nunique()}")

    print("-------------------------")
train_features_data.drop(['employee_id'], axis="columns", inplace=True)
train_features_data.head()
#percentage of the missing values 

total = train_features_data.isnull().sum().sort_values(ascending=False)



percent_1 = train_features_data.isnull().sum() / train_features_data.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data_df = pd.concat([total, percent_2], axis=1, keys=['Total','%'])



missing_data_df
cat_fetaures_col = []

for column in train_features_data.columns:

    if train_features_data[column].dtype == object:

        cat_fetaures_col.append(column)

        print(f"{column} : {train_features_data[column].unique()}")

        print(train_features_data[column].value_counts())

        print("-------------------------------------------")

#numeric-cat ==> discrete

disc_feature_col = []

for column in train_features_data.columns:

    if train_features_data[column].dtypes != object and train_features_data[column].nunique() <= 30:

        print(f"{column} : {train_features_data[column].unique()}")

        print(train_features_data[column].value_counts())

        disc_feature_col.append(column)

        print("-------------------------------------------")

        

disc_feature_col.remove('is_promoted')
print(train_features_data['education'].value_counts())

print("-------------")

print(train_features_data['no_of_trainings'].value_counts())

cont_feature_col=[]

for column in train_features_data.columns:

    if train_features_data[column].dtypes != object and train_features_data[column].nunique() > 30:

        print(f"{column} : Minimum: {train_features_data[column].min()}, Maximum: {train_features_data[column].max()}")

        cont_feature_col.append(column)

        print("-------------------------------------------")
#our dataset there are missing values for "education" and "previous_year_rating" cols.

train_features_data.isnull().sum()
#eliminate null values(fill with mode of that column)



for column in train_features_data.columns:

    train_features_data[column].fillna(train_features_data[column].mode()[0], inplace=True)

print(train_features_data['education'].mode()[0])

print(train_features_data['previous_year_rating'].mode()[0])
#here is there are no missing values in our dataset anymore!!!

train_features_data.isnull().sum()
train_features_data.info()
#there is no "NaN" values anymore

print(train_features_data['previous_year_rating'].unique())

print(train_features_data['education'].unique())
#for numeric-cat (discrete)



import matplotlib



plt.figure(figsize=(32, 32))

matplotlib.rc('axes', titlesize=24)#cols size



for i, column in enumerate(disc_feature_col, 1):

    plt.subplot(4, 4, i)

    train_features_data[train_features_data["is_promoted"] == 0][column].hist(bins=20, color='pink', label='is_promoted = NO', alpha=1)

    train_features_data[train_features_data["is_promoted"] == 1][column].hist(bins=20, color='tomato', label='is_promoted = YES', alpha=.8)

    plt.legend(fontsize='medium')#is_promoted size

    plt.title(column)
#for string-cat

import matplotlib



matplotlib.rc('xtick', labelsize=15) 

matplotlib.rc('ytick', labelsize=20)

matplotlib.rc('axes', titlesize=24)#cols size



plt.rcParams['figure.autolayout'] = True



plt.figure(figsize=(30, 20))



for i, column in enumerate(cat_fetaures_col, 1):

    plt.subplot(3, 3, i)

    train_features_data[train_features_data["is_promoted"] == 0][column].hist(bins=35, color='plum', label='is_promoted = NO', alpha=.8)

    train_features_data[train_features_data["is_promoted"] == 1][column].hist(bins=35, color='indigo', label='is_promoted = YES', alpha=1)

    plt.legend(fontsize='large')#is_promoted size

    plt.title(column)

    plt.xticks(rotation=45)



#for cont --> scatterplot matrix



#cont_feature_col



sns.set(style="ticks")



sns.pairplot(train_features_data[cont_feature_col + ['is_promoted']], hue='is_promoted', palette="husl", corner=True)
#iki eksen(variable) kullanarak target label'ı (is_promoted) tahmin edebilir miyiz?

#yeşil-kırmızı üst üste biniyorsa ayrım yoktur, o iki feature kullanmak yeterli değil, daha fazla feature lazım

#iki feature'ın birbiriyle correlation'larına da bakabiliriz; hiçbiri "line" olmadığı için weak correlation 
#outlier analysis using box-plot(continuos data can have outliers(aykırı değerler))



sns.set(style="whitegrid",font_scale=1)

plt.figure(figsize=(5,7))

sns.boxplot(data=train_features_data[cont_feature_col])

#sns.swarmplot(data=train_features_data[cont_feature_col], color=".25")

plt.xticks(rotation=90)

plt.title("Box plot ")

plt.show()
# find the IQR

q1 = train_features_data[cont_feature_col].quantile(.25)

q3 = train_features_data[cont_feature_col].quantile(.75)

IQR = q3-q1



print(IQR)



outliers_df = np.logical_or((train_features_data[cont_feature_col] < (q1 - 1.5 * IQR)), (train_features_data[cont_feature_col] > (q3 + 1.5 * IQR))) 
outlier_list=[]

total_outlier=[]

for col in list(outliers_df.columns):

    try:

        total_outlier.append(outliers_df[col].value_counts()[True])

        outlier_list.append((outliers_df[col].value_counts()[True] / outliers_df[col].value_counts().sum())*100)

    except:

        outlier_list.append(0)

        total_outlier.append(0)

        

        

outlier_list



outlier_df=pd.DataFrame(zip(list(outliers_df.columns), total_outlier,outlier_list), columns=['name', 'total', 'outlier(%)'])
outlier_df.set_index('name', inplace=True)

#del outlier_df.index.name

outlier_df
#encode ediyoruzzz!!!



#encoding categorical features (str-->float)



from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()



enc.fit(train_features_data)

train_features_data_arr=enc.transform(train_features_data)



col_names_list=train_features_data.columns

encoded_categorical_df=pd.DataFrame(train_features_data_arr, columns=col_names_list)



#heatmap for correlation coefficient(object tiplerin corr hesaplanamaz, önce encode etmek lazım, encode ettik)



# calculate correlation

train_corr = encoded_categorical_df.corr()



# correlation matrix

sns.set(font_scale=1)

plt.figure(figsize=(14,10))

sns.heatmap(train_corr, annot=True, fmt=".4f",vmin=-1, vmax=1, linewidths=.5, cmap = sns.color_palette("BrBG", 100))

#plt.yticks(rotation=0)

plt.show()
#feature importance using corr

encoded_categorical_df.drop('is_promoted', axis=1).corrwith(encoded_categorical_df.is_promoted).plot(kind='barh', figsize=(7, 5), color='skyblue', title="is_promoted vs all features")
#parallel coordinates



import plotly.express as px



from plotly.offline import init_notebook_mode, iplot

from plotly.graph_objs import *



init_notebook_mode(connected=True)

fig = px.parallel_coordinates(encoded_categorical_df, color="is_promoted",

                             color_continuous_scale=px.colors.diverging.Tealrose)

fig.show()

#check types

encoded_categorical_df.info()
# split df to X and Y

from sklearn.model_selection import train_test_split



y = encoded_categorical_df.loc[:, 'is_promoted'].values

X = encoded_categorical_df.drop('is_promoted', axis=1)



# split data into 80-20 for training set / test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)



binary_cols = [col for col in list(encoded_categorical_df.columns) if encoded_categorical_df[col].nunique() <= 2] 

binary_cols.remove('is_promoted')



non_binary_cols = [col for col in list(encoded_categorical_df.columns) if encoded_categorical_df[col].nunique() > 2]
#normalization(make all values bet. 0-1)





from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train[non_binary_cols])



X_train_normalized_arr=scaler.transform(X_train[non_binary_cols])

X_train_normalized_df=pd.DataFrame(X_train_normalized_arr, columns=non_binary_cols)



X_test_normalized_arr=scaler.transform(X_test[non_binary_cols])

X_test_normalized_df=pd.DataFrame(X_test_normalized_arr, columns=non_binary_cols)
X_train_normalized_df.head()
X_test_normalized_df.head()
X_train_binary_cols_df = X_train[binary_cols]

X_train_binary_cols_df.reset_index(inplace=True, drop=True)



X_train_final_df = pd.concat([X_train_binary_cols_df,X_train_normalized_df], axis=1)



X_train_final_df.head()
X_test_binary_cols_df = X_test[binary_cols]

X_test_binary_cols_df.reset_index(inplace=True, drop=True)



X_test_final_df = pd.concat([X_test_binary_cols_df,X_test_normalized_df], axis=1)



X_test_final_df.head()
print(len(X_test_final_df)) 

print(len(X_train_final_df))
#feature importances



from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators = 500, max_depth=13)

rf_clf.fit(X_train_final_df, y_train)

rf_y_pred = rf_clf.predict(X_test_final_df)



pd.Series(rf_clf.feature_importances_, index = X_train_final_df.columns).nlargest(13).plot(kind = 'pie',

                                                                               figsize = (7, 7),

                                                                              title = 'Feature importance from RandomForest', colormap='magma')
from xgboost import XGBClassifier

xgb_clf = XGBClassifier(max_depth=13, learning_rate=1e-4,n_estimators=500)

xgb_clf.fit(X_train_final_df, y_train)

xgb_y_pred = xgb_clf.predict(X_test_final_df)



pd.Series(xgb_clf.feature_importances_, index = X_train_final_df.columns).nlargest(13).plot(kind = 'barh',figsize = (7, 5),

                                                                                          title = 'Feature importance from XGBoost', fontsize=10, colormap='bone')
"""

from pandas_profiling import ProfileReport

prof = ProfileReport(train_features_data)

prof.to_file(output_file='output.html')

"""