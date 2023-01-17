# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#Importing libraries 

import numpy as np # linear algebra

import pandas as pd # data processing

import os

import matplotlib.pyplot as plt#visualization

from matplotlib import cm as cm

%matplotlib inline

import seaborn as sns

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import VarianceThreshold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score,roc_curve,scorer,auc,f1_score,precision_score,recall_score

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Importing datasets from the defined path

telecom_data = pd.read_excel('../input/Customer_Churn.xlsx')



telecom_data.head()
telecom_data.describe()
#Display the shape of the dataframe to figure out the no. of rows and columns.

print ("Dimension   : " ,telecom_data.shape)

print ("\nFeatures : \n" ,telecom_data.columns.tolist())

print ("\nMissing values :  ",telecom_data.isnull().sum().values.sum())

print ("\nUnique values :  \n",telecom_data.nunique())
sns.heatmap(telecom_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Dropping null values from total charges

telecom_data['TotalCharges'] = telecom_data["TotalCharges"].replace(" ",np.nan)

telecom_data = telecom_data[telecom_data["TotalCharges"].notnull()]

telecom_data = telecom_data.reset_index()[telecom_data.columns]



#Data manipulation for columns

# 1.Changing the value "No internet service" to "No" for columns in cols 

# 2.Changing Boolean value for "SeniorCitizen" 

cols = ['OnlineSecurity','OnlineBackup', 'DeviceProtection',

                'TechSupport','StreamingTV', 'StreamingMovies']



for i in cols : 

    telecom_data[i]  = telecom_data[i].replace({'No internet service' : 'No'})

    

telecom_data["MultipleLines"] = telecom_data["MultipleLines"].replace({'No phone service' : 'No'})  

telecom_data["SeniorCitizen"] = telecom_data["SeniorCitizen"].replace({1:"Yes",0:"No"})



churn     = telecom_data[telecom_data["Churn"] == "Yes"]

not_churn = telecom_data[telecom_data["Churn"] == "No"]



telecom_data
#Checking ratio of Churn to Non Churn in pie chart

fig, ax = plt.subplots()

lab = telecom_data["Churn"].value_counts().keys().tolist()

sizes= telecom_data["Churn"].value_counts().values.tolist()

ax.pie(sizes, labels=lab, autopct='%1.1f%%', shadow=True)

ax.axis('equal')

plt.show()
#count of online services availed is identified by creating a new column "Count_OnlineServices" and then visualize those number of customers against it.

plt.figure(figsize=(12,6))

telecom_data['Count_OnlineServices'] = (telecom_data[['OnlineSecurity', 'DeviceProtection', 'StreamingMovies', 'TechSupport',

       'StreamingTV', 'OnlineBackup']] == 'Yes').sum(axis=1)



ax = sns.countplot(x='Count_OnlineServices', hue='Churn' , data=telecom_data)

ax.set_title('Number of Services Availed Vs Churn', fontsize=20)

ax.set_ylabel('Number of Customers', fontsize=15)

ax.set_xlabel('Number of Online Services', fontsize=15)
#Calculating Average Monthly charged for those customer with respect to their number online services

average_monthly_data = telecom_data.groupby('Count_OnlineServices', as_index=False)[['MonthlyCharges']].mean()



plt.figure(figsize=(12,6))

ax = sns.barplot(y='MonthlyCharges', x='Count_OnlineServices', data=average_monthly_data)

ax.set_xlabel('Number of Online Services Availed', fontsize=15)

ax.set_ylabel('Average Monthly Charges',  fontsize=15)

ax.set_title('Avg Monthly Charges vs Number of Services', fontsize=20)
#Visualizing Percentage of Churn based on their tenure(in months)

tenure_month = telecom_data.replace('Yes', 1).replace('No', 0).groupby('tenure', as_index=False)[['Churn']].mean()





plt.figure(figsize=(20,6))



ax = sns.barplot(x='tenure', y='Churn', data = tenure_month)

ax.set_title('Churn Percentage Over Tenure months', fontsize=20)

ax.set_ylabel('Percentage of Churn', fontsize = 15)

ax.set_xlabel('Tenure in Months', fontsize = 15)

#Checking whether any particualr type of Internet service has any impact 

sns.catplot(x="InternetService", y="tenure",hue="Churn",kind="box",data=telecom_data);


plt.figure(figsize=(12,6))

ax = sns.countplot(x="Churn", hue="InternetService", data=telecom_data);

ax.set_ylabel('Number of Customers', fontsize = 15)

ax.set_xlabel('Churn', fontsize = 15)

ax.set_title('Churn By Internet Service Type', fontsize=20)
#To assess the strength and direction of the linear relationships between pairs of variables using Heat map

def corelation(telecom_data):

    # Create Correlation df

    corr = telecom_data.corr()

    # Plot figsize

    fig, ax = plt.subplots(figsize=(16, 16))

    # Generate Color Map

    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    # Drop self-correlations

    dropSelf = np.zeros_like(corr)

    dropSelf[np.triu_indices_from(dropSelf)] = True# Generate Color Map

    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    # Generate Heat Map, allow annotations and place floats in map

    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")

    # Apply xticks

    plt.xticks(range(len(corr.columns)), corr.columns);

    # Apply yticks

    plt.yticks(range(len(corr.columns)), corr.columns)

    # show plot

    plt.show()



corelation(telecom_data)
#Churn By Count against Tenure 

fig, ax = plt.subplots(figsize=(20,10))

sns.catplot(ax=ax ,x="tenure", hue="Churn", kind="count", data=telecom_data);


plt.figure(figsize=(20,6))

average_totalcharge_data = telecom_data.groupby('tenure', as_index=False)[['TotalCharges']].mean()



ax = sns.barplot(y='TotalCharges', x='tenure', data=average_totalcharge_data)

ax.set_xlabel('Number of Online Services Availed', fontsize=15)

ax.set_ylabel('Average Monthly Charges',  fontsize=15)

ax.set_title('Avg Monthly Charges vs Number of Services', fontsize=20)
telecom_data_copy = telecom_data.copy()



Id_col     = ['customerID']

#Target columns

target_col = ["Churn"]

#telecom_data_copy = telecom_data_copy.drop('Churn',axis=1)

#categorical columns

cat_cols   = telecom_data_copy.nunique()[telecom_data_copy.nunique() < 6].keys().tolist()

cat_cols   = [x for x in cat_cols if x not in target_col]

#numerical columns

num_cols   = [x for x in telecom_data_copy.columns if x not in cat_cols + target_col + Id_col]

#Binary columns with 2 values

bin_cols   = telecom_data_copy.nunique()[telecom_data_copy.nunique() == 2].keys().tolist()

#Columns more than 2 values

multi_cols = [i for i in cat_cols if i not in bin_cols]

telecom_data_copy
print ("Dimension   : " ,telecom_data_copy.shape)

print ("\nFeatures : \n" ,telecom_data_copy.columns.tolist())

print ("\nMissing values :  ",telecom_data_copy.isnull().sum().values.sum())

print ("\nUnique values :  \n",telecom_data_copy.nunique())

print ("\nCategorical columns : \n",cat_cols)

print ("\nNumerical columns : \n",num_cols)

print ("\nBinary valued columns : \n",bin_cols)

print ("\nMultivalued columns : \n",multi_cols)
#Duplicating columns for multi value columns

telecom_data_copy = pd.get_dummies(data = telecom_data_copy,columns = multi_cols )
#label encoder for binary valud columns

le = LabelEncoder()

for i in bin_cols :

    telecom_data_copy[i] = le.fit_transform(telecom_data_copy[i])



telecom_data_copy.shape
#Chi-Square test is a statistical method to determine if two categorical variables have a significant correlation between them.

chi2_selector = SelectKBest(chi2, k=2)

df_chi = telecom_data_copy.copy()

#Drop values for non negative columns and id column

df_chi = df_chi.drop(num_cols + Id_col ,axis=1)



df_chi
X_kbest = chi2_selector.fit_transform(df_chi, telecom_data_copy[target_col])

X_kbest.shape
#Scaling Numerical columns

std = StandardScaler()

scaled = std.fit_transform(telecom_data_copy[num_cols])

scaled = pd.DataFrame(scaled,columns=num_cols)

telecom_data_copy = telecom_data_copy.drop(columns = num_cols,axis = 1)

telecom_data_copy = telecom_data_copy.merge(scaled,left_index=True,right_index=True,how = "left")

telecom_data_copy
#To get better interrelation between features after scaling

corelation(telecom_data_copy)
# To convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components.





#Dataframe without chi square 

prechi2_data = telecom_data_copy[[i for i in telecom_data_copy.columns if i not in Id_col + target_col]]

# Adding numeriacal columns which were dropped during chi squared statistics implementation

dropped_column = ['tenure','Count_OnlineServices','MonthlyCharges']

droppedcolumnDF = telecom_data_copy[[i for i in telecom_data_copy.columns if i in dropped_column]]

chi2dataframe = pd.DataFrame(X_kbest)

#Dataframe with chi square

postchi2_data = pd.concat([chi2dataframe, droppedcolumnDF], axis=1)



#Sample feature and Target feature for PCA

X = postchi2_data

Y = telecom_data_copy[target_col]

pca = PCA().fit(X)
#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Dataset Explained Variance')

plt.show()
#PCA implentation with 2 component

pca = PCA(n_components=2)

principal_components = pca.fit_transform(X)

pca_data = pd.DataFrame(data = principal_components

             , columns = ['principal component 1', 'principal component 2'])



#Final dataframe for model implementation

finalDf = pd.concat([pca_data, telecom_data_copy[['Churn']]], axis = 1)

finalDf.shape

finalDf.columns
finalDF_X = finalDf.drop("Churn",axis=1)

finalDF_Y = finalDf["Churn"]
fig = plt.figure(figsize = (15,15))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)

finalDf["Churn"] = finalDf["Churn"].replace({1:"Churn",0:"Not Churn"})

targets = ["Churn","Not Churn"]

colors = ['r', 'b']

for target, color in zip(targets,colors):

    indicesToKeep = finalDf['Churn'] == target

    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']

               , finalDf.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()
#The explained variance tells you how much information (variance) can be attributed to each of the principal components

pca.explained_variance_ratio_
X_train, X_test, Y_train, Y_test = train_test_split(finalDF_X, finalDF_Y, test_size=0.30, random_state=0)

#KNN algorithm used for both classification and regression problems. KNN algorithm based on feature similarity approach.

knn = KNeighborsClassifier(n_neighbors= 3)

 

knn.fit(X_train, Y_train)
prediction = knn.predict(X_test)
# Calculate the fpr and tpr for all thresholds of the classification

# Predict the class labels for the provided data

f1_score = round(f1_score(Y_test, prediction), 2)

recall_score = round(recall_score(Y_test, prediction), 2)

fpr, tpr, thresholds = roc_curve(Y_test, prediction)

roc_auc = auc(fpr, tpr)



print(' ')

print("Sensitivity/Recall : {recall_score}".format(recall_score = recall_score))

print(' ')

print("F1 Score : {f1_score}".format(f1_score = f1_score))

print(' ')

print("AUC-ROC score : ", roc_auc_score(Y_test, prediction))

print(' ')

plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()

print(' ')

print("Confusion Matrix : ")

print(confusion_matrix(Y_test, prediction))

print(' ')

print("Classification Report : ")

print(classification_report(Y_test, prediction))

print(' ')

print("Accuracy : ", round(accuracy_score (Y_test, prediction)*100,2), "%")