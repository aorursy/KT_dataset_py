#import the necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score,confusion_matrix

from scipy.stats import zscore

from sklearn.model_selection import train_test_split
#load the csv file and make the data frame

vehicle_df = pd.read_csv('/kaggle/input/vehicle/vehicle.csv')
#display the first 5 rows of dataframe

vehicle_df.head()
print("The dataframe has {} rows and {} columns".format(vehicle_df.shape[0],vehicle_df.shape[1]))
#display the information of dataframe

vehicle_df.info()
#display in each column how many null values are there

vehicle_df.apply(lambda x: sum(x.isnull()))
#display 5 point summary of dataframe

vehicle_df.describe().transpose()
sns.pairplot(vehicle_df,diag_kind='kde')

plt.show()
#copy the dataframe to another dataframe and drop null/missing values from the newly created dataframe

new_vehicle_df = vehicle_df.copy()
#display the first 5 rows of new dataframe

new_vehicle_df.head()
#display the shape of dataframe

print("Shape of newly created dataframe:",new_vehicle_df.shape)
#drop the null vaues from the new dataframe

new_vehicle_df.dropna(axis=0,inplace=True)
#now we will see what is the shape of dataframe

print("After dropping missing values shape of dataframe:",new_vehicle_df.shape)
#display 5 point summary of new dataframe

new_vehicle_df.describe().transpose()
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(new_vehicle_df['compactness'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(new_vehicle_df['compactness'],ax=ax2)

ax2.set_title("Box Plot")
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(new_vehicle_df['circularity'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(new_vehicle_df['circularity'],ax=ax2)

ax2.set_title("Box Plot")
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(new_vehicle_df['distance_circularity'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(new_vehicle_df['distance_circularity'],ax=ax2)

ax2.set_title("Box Plot")
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(new_vehicle_df['radius_ratio'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(new_vehicle_df['radius_ratio'],ax=ax2)

ax2.set_title("Box Plot")
#check how many outliers are there in radius_ratio column

q1 = np.quantile(new_vehicle_df['radius_ratio'],0.25)

q2 = np.quantile(new_vehicle_df['radius_ratio'],0.50)

q3 = np.quantile(new_vehicle_df['radius_ratio'],0.75)

IQR = q3-q1

print("Quartie1::",q1)

print("Quartie2::",q2)

print("Quartie3::",q3)

print("Inter Quartie Range::",IQR)

#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR

print("radius_ratio above",new_vehicle_df['radius_ratio'].quantile(0.75)+(1.5 * IQR),"are outliers")

print("The Outliers in radius_ratio column are",new_vehicle_df[new_vehicle_df['radius_ratio']>276]['radius_ratio'].shape[0])
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(new_vehicle_df['pr.axis_aspect_ratio'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(new_vehicle_df['pr.axis_aspect_ratio'],ax=ax2)

ax2.set_title("Box Plot")
#check how many outliers are there in pr.axis_aspect_ratio column

q1 = np.quantile(new_vehicle_df['pr.axis_aspect_ratio'],0.25)

q2 = np.quantile(new_vehicle_df['pr.axis_aspect_ratio'],0.50)

q3 = np.quantile(new_vehicle_df['pr.axis_aspect_ratio'],0.75)

IQR = q3-q1

print("Quartie1::",q1)

print("Quartie2::",q2)

print("Quartie3::",q3)

print("Inter Quartie Range::",IQR)

#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR

print("pr.axis_aspect_ratio above",new_vehicle_df['pr.axis_aspect_ratio'].quantile(0.75)+(1.5 * IQR),"are outliers")

print("The Outliers in pr.axis_aspect_ratio column are",new_vehicle_df[new_vehicle_df['pr.axis_aspect_ratio']>77]['pr.axis_aspect_ratio'].shape[0])
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(new_vehicle_df['max.length_aspect_ratio'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(new_vehicle_df['max.length_aspect_ratio'],ax=ax2)

ax2.set_title("Box Plot")
#check how many outliers are there in pr.axis_aspect_ratio column

q1 = np.quantile(new_vehicle_df['max.length_aspect_ratio'],0.25)

q2 = np.quantile(new_vehicle_df['max.length_aspect_ratio'],0.50)

q3 = np.quantile(new_vehicle_df['max.length_aspect_ratio'],0.75)

IQR = q3-q1

print("Quartie1::",q1)

print("Quartie2::",q2)

print("Quartie3::",q3)

print("Inter Quartie Range::",IQR)

#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR

print("max.length_aspect_ratio above",new_vehicle_df['max.length_aspect_ratio'].quantile(0.75)+(1.5 * IQR),"are outliers")

print("max.length_aspect_ratio below",new_vehicle_df['max.length_aspect_ratio'].quantile(0.25)-(1.5 * IQR),"are outliers")

print("The above Outliers in max.length_aspect_ratio column are",new_vehicle_df[new_vehicle_df['max.length_aspect_ratio']>14.5]['max.length_aspect_ratio'].shape[0])

print("The below Outliers in max.length_aspect_ratio column are",new_vehicle_df[new_vehicle_df['max.length_aspect_ratio']<2.5]['max.length_aspect_ratio'].shape[0])
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(new_vehicle_df['scatter_ratio'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(new_vehicle_df['scatter_ratio'],ax=ax2)

ax2.set_title("Box Plot")
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(new_vehicle_df['elongatedness'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(new_vehicle_df['elongatedness'],ax=ax2)

ax2.set_title("Box Plot")
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(new_vehicle_df['pr.axis_rectangularity'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(new_vehicle_df['pr.axis_rectangularity'],ax=ax2)

ax2.set_title("Box Plot")
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(new_vehicle_df['max.length_rectangularity'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(new_vehicle_df['max.length_rectangularity'],ax=ax2)

ax2.set_title("Box Plot")
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(new_vehicle_df['scaled_variance'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(new_vehicle_df['scaled_variance'],ax=ax2)

ax2.set_title("Box Plot")
#check how many outliers are there in scaled_variance column

q1 = np.quantile(new_vehicle_df['scaled_variance'],0.25)

q2 = np.quantile(new_vehicle_df['scaled_variance'],0.50)

q3 = np.quantile(new_vehicle_df['scaled_variance'],0.75)

IQR = q3-q1

print("Quartie1::",q1)

print("Quartie2::",q2)

print("Quartie3::",q3)

print("Inter Quartie Range::",IQR)

#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR

print("scaled_variance above",new_vehicle_df['scaled_variance'].quantile(0.75)+(1.5 * IQR),"are outliers")

print("The Outliers in scaled_variance column are",new_vehicle_df[new_vehicle_df['scaled_variance']>292]['scaled_variance'].shape[0])
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(new_vehicle_df['scaled_variance.1'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(new_vehicle_df['scaled_variance.1'],ax=ax2)

ax2.set_title("Box Plot")
#check how many outliers are there in scaled_variance.1 column

q1 = np.quantile(new_vehicle_df['scaled_variance.1'],0.25)

q2 = np.quantile(new_vehicle_df['scaled_variance.1'],0.50)

q3 = np.quantile(new_vehicle_df['scaled_variance.1'],0.75)

IQR = q3-q1

print("Quartie1::",q1)

print("Quartie2::",q2)

print("Quartie3::",q3)

print("Inter Quartie Range::",IQR)

#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR

print("scaled_variance.1 above",new_vehicle_df['scaled_variance.1'].quantile(0.75)+(1.5 * IQR),"are outliers")

print("The Outliers in scaled_variance.1 column are",new_vehicle_df[new_vehicle_df['scaled_variance.1']>988]['scaled_variance.1'].shape[0])
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(new_vehicle_df['scaled_radius_of_gyration'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(new_vehicle_df['scaled_radius_of_gyration'],ax=ax2)

ax2.set_title("Box Plot")
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(new_vehicle_df['scaled_radius_of_gyration.1'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(new_vehicle_df['scaled_radius_of_gyration.1'],ax=ax2)

ax2.set_title("Box Plot")
#check how many outliers are there in scaled_radius_of_gyration.1 column

q1 = np.quantile(new_vehicle_df['scaled_radius_of_gyration.1'],0.25)

q2 = np.quantile(new_vehicle_df['scaled_radius_of_gyration.1'],0.50)

q3 = np.quantile(new_vehicle_df['scaled_radius_of_gyration.1'],0.75)

IQR = q3-q1

print("Quartie1::",q1)

print("Quartie2::",q2)

print("Quartie3::",q3)

print("Inter Quartie Range::",IQR)

#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR

print("scaled_radius_of_gyration.1 above",new_vehicle_df['scaled_radius_of_gyration.1'].quantile(0.75)+(1.5 * IQR),"are outliers")

print("The Outliers in scaled_radius_of_gyration.1 column are",new_vehicle_df[new_vehicle_df['scaled_radius_of_gyration.1']>87]['scaled_radius_of_gyration.1'].shape[0])
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(new_vehicle_df['skewness_about'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(new_vehicle_df['skewness_about'],ax=ax2)

ax2.set_title("Box Plot")
#check how many outliers are there in skewness_about column

q1 = np.quantile(new_vehicle_df['skewness_about'],0.25)

q2 = np.quantile(new_vehicle_df['skewness_about'],0.50)

q3 = np.quantile(new_vehicle_df['skewness_about'],0.75)

IQR = q3-q1

print("Quartie1::",q1)

print("Quartie2::",q2)

print("Quartie3::",q3)

print("Inter Quartie Range::",IQR)

#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR

print("skewness_about above",new_vehicle_df['skewness_about'].quantile(0.75)+(1.5 * IQR),"are outliers")

print("The Outliers in skewness_about column are",new_vehicle_df[new_vehicle_df['skewness_about']>19.5]['skewness_about'].shape[0])
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(new_vehicle_df['skewness_about.1'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(new_vehicle_df['skewness_about.1'],ax=ax2)

ax2.set_title("Box Plot")
#check how many outliers are there in skewness_about.1 column

q1 = np.quantile(new_vehicle_df['skewness_about.1'],0.25)

q2 = np.quantile(new_vehicle_df['skewness_about.1'],0.50)

q3 = np.quantile(new_vehicle_df['skewness_about.1'],0.75)

IQR = q3-q1

print("Quartie1::",q1)

print("Quartie2::",q2)

print("Quartie3::",q3)

print("Inter Quartie Range::",IQR)

#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR

print("skewness_about.1 above",new_vehicle_df['skewness_about.1'].quantile(0.75)+(1.5 * IQR),"are outliers")

print("The Outliers in skewness_about.1 column are",new_vehicle_df[new_vehicle_df['skewness_about.1']>38.5]['skewness_about.1'].shape[0])
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(new_vehicle_df['skewness_about.2'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(new_vehicle_df['skewness_about.2'],ax=ax2)

ax2.set_title("Box Plot")
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(new_vehicle_df['hollows_ratio'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(new_vehicle_df['hollows_ratio'],ax=ax2)

ax2.set_title("Box Plot")
#display how many are car,bus,van. 

new_vehicle_df['class'].value_counts()
sns.countplot(new_vehicle_df['class'])

plt.show()
#radius_ratio column outliers

new_vehicle_df.drop(new_vehicle_df[new_vehicle_df['radius_ratio']>276].index,axis=0,inplace=True)
#pr.axis_aspect_ratio column outliers

new_vehicle_df.drop(new_vehicle_df[new_vehicle_df['pr.axis_aspect_ratio']>77].index,axis=0,inplace=True)
#max.length_aspect_ratio column outliers

new_vehicle_df.drop(new_vehicle_df[new_vehicle_df['max.length_aspect_ratio']>14.5].index,axis=0,inplace=True)

new_vehicle_df.drop(new_vehicle_df[new_vehicle_df['max.length_aspect_ratio']<2.5].index,axis=0,inplace=True)
#scaled_variance column outliers

new_vehicle_df[new_vehicle_df['scaled_variance']>292]
#scaled_variance.1 column outliers

new_vehicle_df.drop(new_vehicle_df[new_vehicle_df['scaled_variance.1']>988].index,axis=0,inplace=True)
#scaled_radius_of_gyration.1 column outliers

new_vehicle_df.drop(new_vehicle_df[new_vehicle_df['scaled_radius_of_gyration.1']>87].index,axis=0,inplace=True)
#skewness_about column outliers

new_vehicle_df.drop(new_vehicle_df[new_vehicle_df['skewness_about']>19.5].index,axis=0,inplace=True)
#skewness_about.1 column outliers

new_vehicle_df.drop(new_vehicle_df[new_vehicle_df['skewness_about.1']>38.5].index,axis=0,inplace=True)
#now what is the shape of dataframe

print("after removing outliers shape of dataframe:",new_vehicle_df.shape)
#find the correlation between independent variables

plt.figure(figsize=(20,5))

sns.heatmap(new_vehicle_df.corr(),annot=True)

plt.show()
#now separate the dataframe into dependent and independent variables

new_vehicle_df_independent_attr = new_vehicle_df.drop('class',axis=1)

new_vehicle_df_dependent_attr = new_vehicle_df['class']

print("shape of new_vehicle_df_independent_attr::",new_vehicle_df_independent_attr.shape)

print("shape of new_vehicle_df_dependent_attr::",new_vehicle_df_dependent_attr.shape)
#now sclaed the independent attribute and replace the dependent attr value with number

new_vehicle_df_independent_attr_scaled = new_vehicle_df_independent_attr.apply(zscore)

new_vehicle_df_dependent_attr.replace({'car':0,'bus':1,'van':2},inplace=True)
#make the covariance matrix and we have 18 independent features so aur covariance matrix is 18*18 matrix

cov_matrix = np.cov(new_vehicle_df_independent_attr_scaled,rowvar=False)

print("cov_matrix shape:",cov_matrix.shape)

print("Covariance_matrix",cov_matrix)
#now with the help of above covariance matrix we will find eigen value and eigen vectors

pca_to_learn_variance = PCA(n_components=18)

pca_to_learn_variance.fit(new_vehicle_df_independent_attr_scaled)
#display explained variance ratio

pca_to_learn_variance.explained_variance_ratio_
#display explained variance

pca_to_learn_variance.explained_variance_
#display principal components

pca_to_learn_variance.components_
plt.bar(list(range(1,19)),pca_to_learn_variance.explained_variance_ratio_)

plt.xlabel("eigen value/components")

plt.ylabel("variation explained")

plt.show()
plt.step(list(range(1,19)),np.cumsum(pca_to_learn_variance.explained_variance_ratio_))

plt.xlabel("eigen value/components")

plt.ylabel("cummalative of variation explained")

plt.show()
#use first 8 principal components

pca_eight_components = PCA(n_components=8)

pca_eight_components.fit(new_vehicle_df_independent_attr_scaled)
#transform the raw data which is in 18 dimension into 8 new dimension with pca

new_vehicle_df_pca_independent_attr = pca_eight_components.transform(new_vehicle_df_independent_attr_scaled)
#display the shape of new_vehicle_df_pca_independent_attr

new_vehicle_df_pca_independent_attr.shape
#now split the data into 80:20 ratio

rawdata_X_train,rawdata_X_test,rawdata_y_train,rawdata_y_test = train_test_split(new_vehicle_df_independent_attr_scaled,new_vehicle_df_dependent_attr,test_size=0.20,random_state=1)

pca_X_train,pca_X_test,pca_y_train,pca_y_test = train_test_split(new_vehicle_df_pca_independent_attr,new_vehicle_df_dependent_attr,test_size=0.20,random_state=1)
print("shape of rawdata_X_train",rawdata_X_train.shape)

print("shape of rawdata_y_train",rawdata_y_train.shape)

print("shape of rawdata_X_test",rawdata_X_test.shape)

print("shape of rawdata_y_test",rawdata_y_test.shape)

print("--------------------------------------------")

print("shape of pca_X_train",pca_X_train.shape)

print("shape of pca_y_train",pca_y_train.shape)

print("shape of pca_X_test",pca_X_test.shape)

print("shape of pca_y_test",pca_y_test.shape)
#now we will train the model with both raw data and pca data with new dimension

svc = SVC() #instantiate the object
#fit the model on raw data

svc.fit(rawdata_X_train,rawdata_y_train)
#predict the y value

rawdata_y_predict = svc.predict(rawdata_X_test)
#now fit the model on pca data with new dimension

svc.fit(pca_X_train,pca_y_train)
#predict the y value

pca_y_predict = svc.predict(pca_X_test)
#display accuracy score of both models

print("Accuracy score with raw data(18 dimension)",accuracy_score(rawdata_y_test,rawdata_y_predict))

print("Accuracy score with pca data(8 dimension)",accuracy_score(pca_y_test,pca_y_predict))
#display confusion matrix of both models

print("Confusion matrix with raw data(18 dimension)\n",confusion_matrix(rawdata_y_test,rawdata_y_predict))

print("Confusion matrix with pca data(8 dimension)\n",confusion_matrix(pca_y_test,pca_y_predict))
#drop the columns

new_vehicle_df_independent_attr_scaled.drop(['max.length_rectangularity','scaled_radius_of_gyration','skewness_about.2','scatter_ratio','elongatedness','pr.axis_rectangularity','scaled_variance','scaled_variance.1'],axis=1,inplace=True)
#display the shape of new dataframe

new_vehicle_df_independent_attr_scaled.shape
dropcolumn_X_train,dropcolumn_X_test,dropcolumn_y_train,dropcolumn_y_test = train_test_split(new_vehicle_df_independent_attr_scaled,new_vehicle_df_dependent_attr,test_size=0.20,random_state=1)
print("shape of dropcolumn_X_train",dropcolumn_X_train.shape)

print("shape of dropcolumn_y_train",dropcolumn_y_train.shape)

print("shape of dropcolumn_X_test",dropcolumn_X_test.shape)

print("shape of dropcolumn_y_test",dropcolumn_y_test.shape)
#fit the model on dropcolumn_X_train,dropcolumn_y_train

svc.fit(dropcolumn_X_train,dropcolumn_y_train)
#predict the y value

dropcolumn_y_predict = svc.predict(dropcolumn_X_test)
#display the accuracy score and confusion matrix

print("Accuracy score with dropcolumn data(10 dimension)",accuracy_score(dropcolumn_y_test,dropcolumn_y_predict))

print("Confusion matrix with dropcolumn data(10 dimension)\n",confusion_matrix(dropcolumn_y_test,dropcolumn_y_predict))
#create a new dataframe

impute_vehicle_df = vehicle_df.copy()
#display the first 5 rows of dataframe

impute_vehicle_df.head()
#display the shape of dataframe

impute_vehicle_df.shape
#display the information of dataframe

impute_vehicle_df.info()
#display 5 point summary

impute_vehicle_df.describe().transpose()
impute_vehicle_df.fillna(impute_vehicle_df.median(),axis=0,inplace=True)
#display the info of dataframe

impute_vehicle_df.info()
#display 5 point summary after imputation 

impute_vehicle_df.describe().transpose()
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(impute_vehicle_df['compactness'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(impute_vehicle_df['compactness'],ax=ax2)

ax2.set_title("Box Plot")
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(impute_vehicle_df['circularity'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(impute_vehicle_df['circularity'],ax=ax2)

ax2.set_title("Box Plot")
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(impute_vehicle_df['distance_circularity'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(impute_vehicle_df['distance_circularity'],ax=ax2)

ax2.set_title("Box Plot")
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(impute_vehicle_df['radius_ratio'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(impute_vehicle_df['radius_ratio'],ax=ax2)

ax2.set_title("Box Plot")
#check how many outliers are there in radius_ratio column

q1 = np.quantile(impute_vehicle_df['radius_ratio'],0.25)

q2 = np.quantile(impute_vehicle_df['radius_ratio'],0.50)

q3 = np.quantile(impute_vehicle_df['radius_ratio'],0.75)

IQR = q3-q1

print("Quartie1::",q1)

print("Quartie2::",q2)

print("Quartie3::",q3)

print("Inter Quartie Range::",IQR)

#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR

print("radius_ratio above",impute_vehicle_df['radius_ratio'].quantile(0.75)+(1.5 * IQR),"are outliers")

print("The Outliers in radius_ratio column are",impute_vehicle_df[impute_vehicle_df['radius_ratio']>276]['radius_ratio'].shape[0])
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(impute_vehicle_df['pr.axis_aspect_ratio'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(impute_vehicle_df['pr.axis_aspect_ratio'],ax=ax2)

ax2.set_title("Box Plot")
#check how many outliers are there in pr.axis_aspect_ratio column

q1 = np.quantile(impute_vehicle_df['pr.axis_aspect_ratio'],0.25)

q2 = np.quantile(impute_vehicle_df['pr.axis_aspect_ratio'],0.50)

q3 = np.quantile(impute_vehicle_df['pr.axis_aspect_ratio'],0.75)

IQR = q3-q1

print("Quartie1::",q1)

print("Quartie2::",q2)

print("Quartie3::",q3)

print("Inter Quartie Range::",IQR)

#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR

print("pr.axis_aspect_ratio above",impute_vehicle_df['pr.axis_aspect_ratio'].quantile(0.75)+(1.5 * IQR),"are outliers")

print("The Outliers in pr.axis_aspect_ratio column are",impute_vehicle_df[impute_vehicle_df['pr.axis_aspect_ratio']>77]['pr.axis_aspect_ratio'].shape[0])
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(impute_vehicle_df['max.length_aspect_ratio'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(impute_vehicle_df['max.length_aspect_ratio'],ax=ax2)

ax2.set_title("Box Plot")
#check how many outliers are there in max.length_aspect_ratio column

q1 = np.quantile(impute_vehicle_df['max.length_aspect_ratio'],0.25)

q2 = np.quantile(impute_vehicle_df['max.length_aspect_ratio'],0.50)

q3 = np.quantile(impute_vehicle_df['max.length_aspect_ratio'],0.75)

IQR = q3-q1

print("Quartie1::",q1)

print("Quartie2::",q2)

print("Quartie3::",q3)

print("Inter Quartie Range::",IQR)

#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR

print("max.length_aspect_ratio above",impute_vehicle_df['max.length_aspect_ratio'].quantile(0.75)+(1.5 * IQR),"are outliers")

print("max.length_aspect_ratio below",impute_vehicle_df['max.length_aspect_ratio'].quantile(0.25)-(1.5 * IQR),"are outliers")

print("The above Outliers in max.length_aspect_ratio column are",impute_vehicle_df[impute_vehicle_df['max.length_aspect_ratio']>14.5]['max.length_aspect_ratio'].shape[0])

print("The below Outliers in max.length_aspect_ratio column are",impute_vehicle_df[impute_vehicle_df['max.length_aspect_ratio']<2.5]['max.length_aspect_ratio'].shape[0])
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(impute_vehicle_df['scatter_ratio'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(impute_vehicle_df['scatter_ratio'],ax=ax2)

ax2.set_title("Box Plot")
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(impute_vehicle_df['elongatedness'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(impute_vehicle_df['elongatedness'],ax=ax2)

ax2.set_title("Box Plot")
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(impute_vehicle_df['pr.axis_rectangularity'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(impute_vehicle_df['pr.axis_rectangularity'],ax=ax2)

ax2.set_title("Box Plot")
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(impute_vehicle_df['max.length_rectangularity'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(impute_vehicle_df['max.length_rectangularity'],ax=ax2)

ax2.set_title("Box Plot")
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(impute_vehicle_df['scaled_variance'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(impute_vehicle_df['scaled_variance'],ax=ax2)

ax2.set_title("Box Plot")
#check how many outliers are there in scaled_variance column

q1 = np.quantile(impute_vehicle_df['scaled_variance'],0.25)

q2 = np.quantile(impute_vehicle_df['scaled_variance'],0.50)

q3 = np.quantile(impute_vehicle_df['scaled_variance'],0.75)

IQR = q3-q1

print("Quartie1::",q1)

print("Quartie2::",q2)

print("Quartie3::",q3)

print("Inter Quartie Range::",IQR)

#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR

print("scaled_variance above",impute_vehicle_df['scaled_variance'].quantile(0.75)+(1.5 * IQR),"are outliers")

print("The Outliers in scaled_variance column are",impute_vehicle_df[impute_vehicle_df['scaled_variance']>292]['scaled_variance'].shape[0])
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(impute_vehicle_df['scaled_variance.1'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(impute_vehicle_df['scaled_variance.1'],ax=ax2)

ax2.set_title("Box Plot")
#check how many outliers are there in scaled_variance.1 column

q1 = np.quantile(impute_vehicle_df['scaled_variance.1'],0.25)

q2 = np.quantile(impute_vehicle_df['scaled_variance.1'],0.50)

q3 = np.quantile(impute_vehicle_df['scaled_variance.1'],0.75)

IQR = q3-q1

print("Quartie1::",q1)

print("Quartie2::",q2)

print("Quartie3::",q3)

print("Inter Quartie Range::",IQR)

#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR

print("scaled_variance.1 above",impute_vehicle_df['scaled_variance.1'].quantile(0.75)+(1.5 * IQR),"are outliers")

print("The Outliers in scaled_variance.1 column are",impute_vehicle_df[impute_vehicle_df['scaled_variance.1']>989.5]['scaled_variance.1'].shape[0])
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(impute_vehicle_df['scaled_radius_of_gyration'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(impute_vehicle_df['scaled_radius_of_gyration'],ax=ax2)

ax2.set_title("Box Plot")
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(impute_vehicle_df['scaled_radius_of_gyration.1'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(impute_vehicle_df['scaled_radius_of_gyration.1'],ax=ax2)

ax2.set_title("Box Plot")
#check how many outliers are there in scaled_radius_of_gyration.1 column

q1 = np.quantile(impute_vehicle_df['scaled_radius_of_gyration.1'],0.25)

q2 = np.quantile(impute_vehicle_df['scaled_radius_of_gyration.1'],0.50)

q3 = np.quantile(impute_vehicle_df['scaled_radius_of_gyration.1'],0.75)

IQR = q3-q1

print("Quartie1::",q1)

print("Quartie2::",q2)

print("Quartie3::",q3)

print("Inter Quartie Range::",IQR)

#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR

print("scaled_radius_of_gyration.1 above",impute_vehicle_df['scaled_radius_of_gyration.1'].quantile(0.75)+(1.5 * IQR),"are outliers")

print("The Outliers in scaled_radius_of_gyration.1 column are",impute_vehicle_df[impute_vehicle_df['scaled_radius_of_gyration.1']>87]['scaled_radius_of_gyration.1'].shape[0])
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(impute_vehicle_df['skewness_about'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(impute_vehicle_df['skewness_about'],ax=ax2)

ax2.set_title("Box Plot")
#check how many outliers are there in skewness_about column

q1 = np.quantile(impute_vehicle_df['skewness_about'],0.25)

q2 = np.quantile(impute_vehicle_df['skewness_about'],0.50)

q3 = np.quantile(impute_vehicle_df['skewness_about'],0.75)

IQR = q3-q1

print("Quartie1::",q1)

print("Quartie2::",q2)

print("Quartie3::",q3)

print("Inter Quartie Range::",IQR)

#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR

print("skewness_about above",impute_vehicle_df['skewness_about'].quantile(0.75)+(1.5 * IQR),"are outliers")

print("The Outliers in skewness_about column are",impute_vehicle_df[impute_vehicle_df['skewness_about']>19.5]['skewness_about'].shape[0])
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(impute_vehicle_df['skewness_about.1'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(impute_vehicle_df['skewness_about.1'],ax=ax2)

ax2.set_title("Box Plot")
#check how many outliers are there in skewness_about.1 column

q1 = np.quantile(impute_vehicle_df['skewness_about.1'],0.25)

q2 = np.quantile(impute_vehicle_df['skewness_about.1'],0.50)

q3 = np.quantile(impute_vehicle_df['skewness_about.1'],0.75)

IQR = q3-q1

print("Quartie1::",q1)

print("Quartie2::",q2)

print("Quartie3::",q3)

print("Inter Quartie Range::",IQR)

#outliers = q3 + 1.5*IQR, q1 - 1.5*IQR

print("skewness_about.1 above",impute_vehicle_df['skewness_about.1'].quantile(0.75)+(1.5 * IQR),"are outliers")

print("The Outliers in skewness_about.1 column are",impute_vehicle_df[impute_vehicle_df['skewness_about.1']>40]['skewness_about.1'].shape[0])
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(impute_vehicle_df['skewness_about.2'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(impute_vehicle_df['skewness_about.2'],ax=ax2)

ax2.set_title("Box Plot")
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

fig.set_size_inches(20,4)

sns.distplot(impute_vehicle_df['hollows_ratio'],ax=ax1)

ax1.set_title("Distribution Plot")



sns.boxplot(impute_vehicle_df['hollows_ratio'],ax=ax2)

ax2.set_title("Box Plot")
impute_vehicle_df['class'].value_counts()
sns.countplot(impute_vehicle_df['class'])

plt.show()
#radius_ratio column outliers

impute_vehicle_df.drop(impute_vehicle_df[impute_vehicle_df['radius_ratio']>276].index,axis=0,inplace=True)
#pr.axis_aspect_ratio column outliers

impute_vehicle_df.drop(impute_vehicle_df[impute_vehicle_df['pr.axis_aspect_ratio']>77].index,axis=0,inplace=True)
#max.length_aspect_ratio column outliers

impute_vehicle_df.drop(impute_vehicle_df[impute_vehicle_df['max.length_aspect_ratio']>14.5].index,axis=0,inplace=True)

impute_vehicle_df.drop(impute_vehicle_df[impute_vehicle_df['max.length_aspect_ratio']<2.5].index,axis=0,inplace=True)
#scaled_variance column outliers

impute_vehicle_df[impute_vehicle_df['scaled_variance']>292]
#scaled_variance.1 column outliers

impute_vehicle_df.drop(impute_vehicle_df[impute_vehicle_df['scaled_variance.1']>989.5].index,axis=0,inplace=True)
#scaled_radius_of_gyration.1 column outliers

impute_vehicle_df.drop(impute_vehicle_df[impute_vehicle_df['scaled_radius_of_gyration.1']>87].index,axis=0,inplace=True)
#skewness_about column outliers

impute_vehicle_df.drop(impute_vehicle_df[impute_vehicle_df['skewness_about']>19.5].index,axis=0,inplace=True)
#skewness_about.1 column outliers

impute_vehicle_df.drop(impute_vehicle_df[impute_vehicle_df['skewness_about.1']>40].index,axis=0,inplace=True)
#display the shape of data frame

print("after fixing outliers shape of dataframe:",impute_vehicle_df.shape)
plt.figure(figsize=(20,4))

sns.heatmap(impute_vehicle_df.corr(),annot=True)

plt.show()
#now separate the dataframe into dependent and independent variables

impute_vehicle_df_independent_attr = impute_vehicle_df.drop('class',axis=1)

impute_vehicle_df_dependent_attr = impute_vehicle_df['class']

print("shape of impute_vehicle_df_independent_attr::",impute_vehicle_df_independent_attr.shape)

print("shape of impute_vehicle_df_dependent_attr::",impute_vehicle_df_dependent_attr.shape)
#now sclaed the independent attribute and replace the dependent attr value with number

impute_vehicle_df_independent_attr_scaled = impute_vehicle_df_independent_attr.apply(zscore)

impute_vehicle_df_dependent_attr.replace({'car':0,'bus':1,'van':2},inplace=True)
#make the covariance matrix and we have 18 independent features so aur covariance matrix is 18*18 matrix

impute_cov_matrix = np.cov(impute_vehicle_df_independent_attr_scaled,rowvar=False)

print("Impute cov_matrix shape:",impute_cov_matrix.shape)

print("Impute Covariance_matrix",impute_cov_matrix)
#now with the help of above covariance matrix we will find eigen value and eigen vectors

impute_pca_to_learn_variance = PCA(n_components=18)

impute_pca_to_learn_variance.fit(impute_vehicle_df_independent_attr_scaled)
#display explained variance ratio

impute_pca_to_learn_variance.explained_variance_ratio_
#display explained variance

impute_pca_to_learn_variance.explained_variance_
#display principal components

impute_pca_to_learn_variance.components_
plt.bar(list(range(1,19)),impute_pca_to_learn_variance.explained_variance_ratio_)

plt.xlabel("eigen value/components")

plt.ylabel("variation explained")

plt.show()
plt.step(list(range(1,19)),np.cumsum(impute_pca_to_learn_variance.explained_variance_ratio_))

plt.xlabel("eigen value/components")

plt.ylabel("cummalative of variation explained")

plt.show()
#use first 8 principal components

impute_pca_eight_components = PCA(n_components=8)

impute_pca_eight_components.fit(impute_vehicle_df_independent_attr_scaled)
#transform the impute raw data which is in 18 dimension into 8 new dimension with pca

impute_vehicle_df_pca_independent_attr = impute_pca_eight_components.transform(impute_vehicle_df_independent_attr_scaled)
#display the shape of new_vehicle_df_pca_independent_attr

impute_vehicle_df_pca_independent_attr.shape
#now split the data into 80:20 ratio

impute_rawdata_X_train,impute_rawdata_X_test,impute_rawdata_y_train,impute_rawdata_y_test = train_test_split(impute_vehicle_df_independent_attr_scaled,impute_vehicle_df_dependent_attr,test_size=0.20,random_state=1)

impute_pca_X_train,impute_pca_X_test,impute_pca_y_train,impute_pca_y_test = train_test_split(impute_vehicle_df_pca_independent_attr,impute_vehicle_df_dependent_attr,test_size=0.20,random_state=1)
print("shape of impute_rawdata_X_train",impute_rawdata_X_train.shape)

print("shape of impute_rawdata_y_train",impute_rawdata_y_train.shape)

print("shape of impute_rawdata_X_test",impute_rawdata_X_test.shape)

print("shape of impute_rawdata_y_test",impute_rawdata_y_test.shape)

print("--------------------------------------------")

print("shape of impute_pca_X_train",impute_pca_X_train.shape)

print("shape of impute_pca_y_train",impute_pca_y_train.shape)

print("shape of impute_pca_X_test",impute_pca_X_test.shape)

print("shape of impute_pca_y_test",impute_pca_y_test.shape)
#fit the model on impute raw data

svc.fit(impute_rawdata_X_train,impute_rawdata_y_train)
#predict the y value

impute_rawdata_y_predict = svc.predict(impute_rawdata_X_test)
#now fit the model on pca data with new dimension

svc.fit(impute_pca_X_train,impute_pca_y_train)
#predict the y value

impute_pca_y_predict = svc.predict(impute_pca_X_test)
#display accuracy score of both models

print("Accuracy score with impute raw data(18 dimension)",accuracy_score(impute_rawdata_y_test,impute_rawdata_y_predict))

print("Accuracy score with impute pca data(8 dimension)",accuracy_score(impute_pca_y_test,impute_pca_y_predict))
#display confusion matrix of both models

print("Confusion matrix with impute raw data(18 dimension)\n",confusion_matrix(impute_rawdata_y_test,impute_rawdata_y_predict))

print("Confusion matrix with impute pca data(8 dimension)\n",confusion_matrix(impute_pca_y_test,impute_pca_y_predict))
#drop the columns

impute_vehicle_df_independent_attr_scaled.drop(['max.length_rectangularity','scaled_radius_of_gyration','skewness_about.2','scatter_ratio','elongatedness','pr.axis_rectangularity','scaled_variance','scaled_variance.1'],axis=1,inplace=True)
#display the shape of new dataframe

impute_vehicle_df_independent_attr_scaled.shape
impute_dropcolumn_X_train,impute_dropcolumn_X_test,impute_dropcolumn_y_train,impute_dropcolumn_y_test = train_test_split(impute_vehicle_df_independent_attr_scaled,impute_vehicle_df_dependent_attr,test_size=0.20,random_state=1)
print("shape of impute_dropcolumn_X_train",impute_dropcolumn_X_train.shape)

print("shape of impute_dropcolumn_y_train",impute_dropcolumn_y_train.shape)

print("shape of impute_dropcolumn_X_test",impute_dropcolumn_X_test.shape)

print("shape of impute_dropcolumn_y_test",impute_dropcolumn_y_test.shape)
#fit the model on dropcolumn_X_train,dropcolumn_y_train

svc.fit(impute_dropcolumn_X_train,impute_dropcolumn_y_train)
#predict the y value

impute_dropcolumn_y_predict = svc.predict(impute_dropcolumn_X_test)
#display the accuracy score and confusion matrix

print("Accuracy score with impute dropcolumn data(10 dimension)",accuracy_score(impute_dropcolumn_y_test,impute_dropcolumn_y_predict))

print("Confusion matrix with impute dropcolumn data(10 dimension)\n",confusion_matrix(impute_dropcolumn_y_test,impute_dropcolumn_y_predict))