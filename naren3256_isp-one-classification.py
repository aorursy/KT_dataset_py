# import required modules

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import Image

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier,export_graphviz

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report,precision_score,f1_score
# import the data

data = pd.read_csv("../input/ISP_One_Attrition_Data_file.csv")
# First few rows of the data

data.head()
# Remove duplicates

data.drop_duplicates(subset=None, keep='first', inplace=True)

print("shape of the data after removing the duplicates", data.shape)
# shape of the data

print("Number of rows :",data.shape[0])

print("Number of columns :",data.shape[1])

print("===============================================")

# check if there are any missing values in each attribute

print(data.isnull().any())

print("===============================================")

# check the number of samples and their data type

data.info()

# Correlation matrix using standard pearson correlation --> Check for linear dependancy

plt.figure(figsize = (20,10))

sns.heatmap(data.corr(method = "pearson"),annot = True,cmap="coolwarm")
# Removing the linearly dependant variable(income)

cols = ["expenditure","months_on_network","Num_complaints","number_plan_changes","relocated","monthly_bill","technical_issues_per_month","Speed_test_result"]

# statistics

plt.figure(figsize=(16,8))

sns.heatmap(data.describe(),fmt = ".2f",cmap = "coolwarm",annot = True)
# removing negative values in the months_on_network attribute

data = data[data.months_on_network>0]

print("Number of rows and columns after removing negative values :",data.shape)
# outlier analysis

def plots(df,attrib):

    # this function takes two arguments (dataframe and attribute of interest)

    

    # define the figure size

    plt.figure(figsize = (16,4))

    

    # histogram

    plt.subplot(1,2,1)

    sns.distplot(df[attrib],bins = 50)

    plt.title("Histogram")

    

    # boxplot

    plt.subplot(1,2,2)

    sns.boxplot(y = df[attrib])

    plt.title("Boxplot")

    plt.show()

plots(data,"expenditure")

plots(data,"months_on_network")

plots(data,"Speed_test_result")
def find_skewed_boundaries(df,attrib,distance):

    # distance is the attribute required to estimate the amount of data loss during the outlier trimming using IQR proximity measure

    

    IQR = df[attrib].quantile(0.75)-df[attrib].quantile(0.25)

    lower_boundary = df[attrib].quantile(0.25) - (IQR * distance)

    upper_boundary = df[attrib].quantile(0.75) + (IQR * distance)

    

    return upper_boundary,lower_boundary
# Find the limits for expenditure attribute

exp_upper_lim,exp_lower_lim = find_skewed_boundaries(data,"expenditure",1.5)

print("Upper and lower limits of expenditure attribute :",exp_upper_lim,exp_lower_lim)



# Find the limits for months_on_network attribute

months_upper,months_lower = find_skewed_boundaries(data,"months_on_network",1.5)

print("Upper and lower limits of months on network attribute :", months_upper,months_lower)



# Find the limits for speed_test results

speed_upper,speed_lower = find_skewed_boundaries(data,"Speed_test_result",1.5)

print("Upper and lower limits of speed_test_result attribute : ",speed_upper,speed_lower)
# Extract the outliers from each attribute

outliers_expen = np.where(data["expenditure"] > exp_upper_lim,True,

                         np.where(data["expenditure"] < exp_lower_lim,True, False))



outliers_months = np.where(data["months_on_network"] > months_upper,True,

                         np.where(data["months_on_network"] < months_lower,True, False))



outliers_speed = np.where(data["Speed_test_result"] > speed_upper,True,

                         np.where(data["Speed_test_result"] < speed_lower,True, False))



# trim the dataset

data_trimmed = data.loc[~(outliers_expen+outliers_months+outliers_speed),]

print("Data size before and after the outlier removal : " ,data.shape , data_trimmed.shape)
sns.countplot(data["active_cust"])

data["active_cust"].value_counts()


# Normalization

# Independant attributes or attribute vector

cols = ["expenditure","months_on_network","Num_complaints","number_plan_changes","relocated","monthly_bill","technical_issues_per_month","Speed_test_result"]

X = data_trimmed[cols].values

# dependant variable or the Class label

y = data_trimmed["active_cust"].values





scale = StandardScaler()

X = scale.fit_transform(X)

print(X[0:2])

# train and test split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

model1 = DecisionTreeClassifier(criterion="gini")

#train the model

model1.fit(X_train,y_train)

#prediction for test data

y_Pred = model1.predict(X_test)

print("y_pred has the predicted values for x_test")
# Model Evaluation using confusion matrix

cm = confusion_matrix(y_test,y_Pred)

print("Accuracy of the model : ",accuracy_score(y_test,y_Pred))



# Confusion matrix

plt.figure(figsize = (6,6))

plt.subplot(2,1,1)

plt.title("Confusion matrix : DecisionTreeClassifier")

sns.heatmap(cm,annot = True,cmap="BuPu",fmt=".2f")



# Countplot

plt.figure(figsize = (6,6))

plt.subplot(2,1,2)

plt.title("No. of negative and positive classes in test data")

# Countplot to check the class imbalance

sns.countplot(y_test)



# Performance metrics

sensitivity = (9754/(9754+1303))

print("True positive rate/sensitivity :",sensitivity)

specificity = (6935/(1345+6935))

print("True negative rate/specificity :",specificity)

precision = precision_score(y_test,y_Pred)

print("precision_score :", precision)

F1_score = f1_score(y_test,y_Pred)

print("f1 score :",F1_score)

print("====================================================")

print(classification_report(y_test,y_Pred))

# Model implementation : RandomForestClassifier

model = RandomForestClassifier(n_estimators = 20, criterion= "gini")

#train

model.fit(X_train,y_train)

y_Pred = model.predict(X_test)



#Evaluation of the Random forest classifier 

cm = confusion_matrix(y_test,y_Pred)

print("Accuracy of the model: ",accuracy_score(y_test,y_Pred))

plt.figure(figsize = (6,4))

plt.title("Confusion matrix : RandomForestClassifier")

sns.heatmap(cm,annot = True,cmap="Greens",fmt=".2f")





# Model evaluation

print(classification_report(y_test,y_Pred))
