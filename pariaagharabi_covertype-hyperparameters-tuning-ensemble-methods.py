from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Import the necessary packages

import numpy as np

import pandas as pd



import warnings

warnings.simplefilter(action ="ignore")



from collections import Counter



# Data visualization

import matplotlib.pyplot as plt

import seaborn as sns

from scikitplot.plotters import plot_learning_curve

from mlxtend.plotting import plot_learning_curves

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

from yellowbrick.model_selection import FeatureImportances



# Algorithms

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn import metrics

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import mean_squared_error
# Load Dataset

train = pd.read_csv('../input/forest-cover-type-prediction/train.csv')

test  = pd.read_csv('../input/forest-cover-type-prediction/test.csv')
train.head()
train.columns.values
test.head()
test.columns.values
# Analyse statically insight of train data

train.describe()
# Analyse statically insight of test data

test.describe()
train.info()
test.info()
print(f"The train data size: {train.shape}")

print(f"The test data size: {test.shape}")
diff_train_test = set(train.columns) - set(test.columns)

diff_train_test
train["Cover_Type"].describe()
plt.figure(figsize=(22,6), dpi= 80)

ax = sns.countplot(y=train["Cover_Type"], hue="Cover_Type", data=train)
numeric_data=train.select_dtypes(exclude="object")

numeric_corr=numeric_data.corr()

f,ax=plt.subplots(figsize=(19,1))

sns.heatmap(numeric_corr.sort_values(by=["Cover_Type"], ascending=False).head(1), cmap="Greens")

plt.title("Numerical features correlation with the Cover_Type", weight="bold", fontsize=18, color="darkgreen")

plt.yticks(weight="bold", color="darkgreen", rotation=0)



plt.show()
Num_feature=numeric_corr["Cover_Type"].sort_values(ascending=False).head(20).to_frame()



cm = sns.light_palette("forestgreen", as_cmap=True)



style = Num_feature.style.background_gradient(cmap=cm)

style
# Draw Plot

plt.figure(figsize=(16,10), dpi= 80)

sns.kdeplot(train.loc[train["Cover_Type"] == 1, "Elevation"], shade=True, color="#4169E1", label="Cover=1", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 2, "Elevation"], shade=True, color="#FF8C00", label="Cover=2", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 3, "Elevation"], shade=True, color="#FF4500", label="Cover=3", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 4, "Elevation"], shade=True, color="#BDB76B", label="Cover=4", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 5, "Elevation"], shade=True, color="#8B4513", label="Cover=5", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 6, "Elevation"], shade=True, color="#9400D3", label="Cover=6", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 7, "Elevation"], shade=True, color="#006400", label="Cover=7", alpha=.7)



# Decoration

plt.title("The distribution of the attribute Elevation", color="#006400", fontsize=22)

plt.legend()

plt.show()
g = sns.catplot(x="Elevation", hue="Cover_Type", col="Cover_Type",

                data=train, kind="count",

                height=4, aspect=.7);
# Draw Plot

plt.figure(figsize=(15,8))

sns.boxplot(x="Cover_Type", y="Elevation", data=train, hue="Cover_Type")



# Decoration

plt.title("Box Plot of Elevation by Cover_Type", fontsize=22, color="#006400")

plt.legend(title="Cover")

plt.show()
# Draw Plot

plt.figure(figsize=(16,10), dpi= 80)

sns.kdeplot(train.loc[train["Cover_Type"] == 1, "Aspect"], shade=True, color="#4169E1", label="Cover=1", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 2, "Aspect"], shade=True, color="#FF8C00", label="Cover=2", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 3, "Aspect"], shade=True, color="#FF4500", label="Cover=3", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 4, "Aspect"], shade=True, color="#BDB76B", label="Cover=4", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 5, "Aspect"], shade=True, color="#8B4513", label="Cover=5", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 6, "Aspect"], shade=True, color="#9400D3", label="Cover=6", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 7, "Aspect"], shade=True, color="#006400", label="Cover=7", alpha=.7)



# Decoration

plt.title("The distribution of the attribute Aspect", color="#006400", fontsize=22)

plt.legend()

plt.show()
g = sns.catplot(x="Aspect", hue="Cover_Type", col="Cover_Type",

                data=train, kind="count",

                height=4, aspect=.7);
# Draw Plot

plt.figure(figsize=(15,8))

sns.boxplot(x="Cover_Type", y="Aspect", data=train, hue="Cover_Type")



# Decoration

plt.title("Box Plot of Aspect by Cover_Type", fontsize=22, color="#006400")

plt.legend(title="Cover")

plt.show()
# Draw Plot

plt.figure(figsize=(16,10), dpi= 80)

sns.kdeplot(train.loc[train["Cover_Type"] == 1, "Slope"], shade=True, color="#4169E1", label="Cover=1", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 2, "Slope"], shade=True, color="#FF8C00", label="Cover=2", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 3, "Slope"], shade=True, color="#FF4500", label="Cover=3", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 4, "Slope"], shade=True, color="#BDB76B", label="Cover=4", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 5, "Slope"], shade=True, color="#8B4513", label="Cover=5", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 6, "Slope"], shade=True, color="#9400D3", label="Cover=6", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 7, "Slope"], shade=True, color="#006400", label="Cover=7", alpha=.7)



# Decoration

plt.title("The distribution of the attribute Slope", color="#006400", fontsize=22)

plt.legend()

plt.show()
g = sns.catplot(x="Slope", hue="Cover_Type", col="Cover_Type",

                data=train, kind="count",

                height=4, aspect=.7);
# Draw Plot

plt.figure(figsize=(15,8))

sns.boxplot(x="Cover_Type", y="Slope", data=train, hue="Cover_Type")



# Decoration

plt.title("Box Plot of Slope by Cover_Type", fontsize=22, color="#006400")

plt.legend(title="Cover")

plt.show()
# Draw Plot

plt.figure(figsize=(16,10), dpi= 80)

sns.kdeplot(train.loc[train["Cover_Type"] == 1, "Horizontal_Distance_To_Hydrology"], shade=True, color="#4169E1", label="Cover=1", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 2, "Horizontal_Distance_To_Hydrology"], shade=True, color="#FF8C00", label="Cover=2", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 3, "Horizontal_Distance_To_Hydrology"], shade=True, color="#FF4500", label="Cover=3", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 4, "Horizontal_Distance_To_Hydrology"], shade=True, color="#BDB76B", label="Cover=4", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 5, "Horizontal_Distance_To_Hydrology"], shade=True, color="#8B4513", label="Cover=5", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 6, "Horizontal_Distance_To_Hydrology"], shade=True, color="#9400D3", label="Cover=6", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 7, "Horizontal_Distance_To_Hydrology"], shade=True, color="#006400", label="Cover=7", alpha=.7)



# Decoration

plt.title("The distribution of the attribute Horizontal_Distance_To_Hydrology", color="#006400", fontsize=22)

plt.legend()

plt.show()
g = sns.catplot(x="Horizontal_Distance_To_Hydrology", hue="Cover_Type", col="Cover_Type",

                data=train, kind="count",

                height=4, aspect=.7);
# Draw Plot

plt.figure(figsize=(15,8))

sns.boxplot(x="Cover_Type", y="Horizontal_Distance_To_Hydrology", data=train, hue="Cover_Type")



# Decoration

plt.title("Box Plot of Horizontal_Distance_To_Hydrology by Cover_Type", fontsize=22, color="#006400")

plt.legend(title="Cover")

plt.show()
# Draw Plot

plt.figure(figsize=(16,10), dpi= 80)

sns.kdeplot(train.loc[train["Cover_Type"] == 1, "Vertical_Distance_To_Hydrology"], shade=True, color="#4169E1", label="Cover=1", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 2, "Vertical_Distance_To_Hydrology"], shade=True, color="#FF8C00", label="Cover=2", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 3, "Vertical_Distance_To_Hydrology"], shade=True, color="#FF4500", label="Cover=3", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 4, "Vertical_Distance_To_Hydrology"], shade=True, color="#BDB76B", label="Cover=4", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 5, "Vertical_Distance_To_Hydrology"], shade=True, color="#8B4513", label="Cover=5", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 6, "Vertical_Distance_To_Hydrology"], shade=True, color="#9400D3", label="Cover=6", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 7, "Vertical_Distance_To_Hydrology"], shade=True, color="#006400", label="Cover=7", alpha=.7)



# Decoration

plt.title("The distribution of the attribute Vertical_Distance_To_Hydrology", color="#006400", fontsize=22)

plt.legend()

plt.show()
# Draw Plot

plt.figure(figsize=(15,8))

sns.boxplot(x="Cover_Type", y="Vertical_Distance_To_Hydrology", data=train, hue="Cover_Type")



# Decoration

plt.title("Box Plot of Vertical_Distance_To_Hydrology by Cover_Type", fontsize=22, color="#006400")

plt.legend(title="Cover")

plt.show()
# Draw Plot

plt.figure(figsize=(16,10), dpi= 80)

sns.kdeplot(train.loc[train["Cover_Type"] == 1, "Horizontal_Distance_To_Roadways"], shade=True, color="#4169E1", label="Cover=1", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 2, "Horizontal_Distance_To_Roadways"], shade=True, color="#FF8C00", label="Cover=2", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 3, "Horizontal_Distance_To_Roadways"], shade=True, color="#FF4500", label="Cover=3", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 4, "Horizontal_Distance_To_Roadways"], shade=True, color="#BDB76B", label="Cover=4", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 5, "Horizontal_Distance_To_Roadways"], shade=True, color="#8B4513", label="Cover=5", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 6, "Horizontal_Distance_To_Roadways"], shade=True, color="#9400D3", label="Cover=6", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 7, "Horizontal_Distance_To_Roadways"], shade=True, color="#006400", label="Cover=7", alpha=.7)



# Decoration

plt.title("The distribution of the attribute Horizontal_Distance_To_Roadways", color="#006400", fontsize=22)

plt.legend()

plt.show()
# Draw Plot

plt.figure(figsize=(15,8))

sns.boxplot(x="Cover_Type", y="Horizontal_Distance_To_Roadways", data=train, hue="Cover_Type")



# Decoration

plt.title("Box Plot of Horizontal_Distance_To_Roadways by Cover_Type", fontsize=22, color="#006400")

plt.legend(title="Cover")

plt.show()
# Draw Plot

plt.figure(figsize=(16,10), dpi= 80)

sns.kdeplot(train.loc[train["Cover_Type"] == 1, "Hillshade_9am"], shade=True, color="#4169E1", label="Cover=1", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 2, "Hillshade_9am"], shade=True, color="#FF8C00", label="Cover=2", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 3, "Hillshade_9am"], shade=True, color="#FF4500", label="Cover=3", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 4, "Hillshade_9am"], shade=True, color="#BDB76B", label="Cover=4", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 5, "Hillshade_9am"], shade=True, color="#8B4513", label="Cover=5", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 6, "Hillshade_9am"], shade=True, color="#9400D3", label="Cover=6", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 7, "Hillshade_9am"], shade=True, color="#006400", label="Cover=7", alpha=.7)



# Decoration

plt.title("The distribution of the attribute Hillshade_9am", color="#006400", fontsize=22)

plt.legend()

plt.show()
# Draw Plot

plt.figure(figsize=(15,8))

sns.boxplot(x="Cover_Type", y="Hillshade_9am", data=train, hue="Cover_Type")



# Decoration

plt.title("Box Plot of Hillshade_9am by Cover_Type", fontsize=22, color="#006400")

plt.legend(title="Cover")

plt.show()
# Draw Plot

plt.figure(figsize=(16,10), dpi= 80)

sns.kdeplot(train.loc[train["Cover_Type"] == 1, "Hillshade_Noon"], shade=True, color="#4169E1", label="Cover=1", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 2, "Hillshade_Noon"], shade=True, color="#FF8C00", label="Cover=2", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 3, "Hillshade_Noon"], shade=True, color="#FF4500", label="Cover=3", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 4, "Hillshade_Noon"], shade=True, color="#BDB76B", label="Cover=4", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 5, "Hillshade_Noon"], shade=True, color="#8B4513", label="Cover=5", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 6, "Hillshade_Noon"], shade=True, color="#9400D3", label="Cover=6", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 7, "Hillshade_Noon"], shade=True, color="#006400", label="Cover=7", alpha=.7)



# Decoration

plt.title("The distribution of the attribute Hillshade_Noon", color="#006400", fontsize=22)

plt.legend()

plt.show()
# Draw Plot

plt.figure(figsize=(15,8))

sns.boxplot(x="Cover_Type", y="Hillshade_Noon", data=train, hue="Cover_Type")



# Decoration

plt.title("Box Plot of Hillshade_Noon by Cover_Type", fontsize=22, color="#006400")

plt.legend(title="Cover")

plt.show()
# Draw Plot

plt.figure(figsize=(16,10), dpi= 80)

sns.kdeplot(train.loc[train["Cover_Type"] == 1, "Hillshade_3pm"], shade=True, color="#4169E1", label="Cover=1", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 2, "Hillshade_3pm"], shade=True, color="#FF8C00", label="Cover=2", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 3, "Hillshade_3pm"], shade=True, color="#FF4500", label="Cover=3", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 4, "Hillshade_3pm"], shade=True, color="#BDB76B", label="Cover=4", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 5, "Hillshade_3pm"], shade=True, color="#8B4513", label="Cover=5", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 6, "Hillshade_3pm"], shade=True, color="#9400D3", label="Cover=6", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 7, "Hillshade_3pm"], shade=True, color="#006400", label="Cover=7", alpha=.7)



# Decoration

plt.title("The distribution of the attribute Hillshade_3pm", color="#006400", fontsize=22)

plt.legend()

plt.show()
# Draw Plot

plt.figure(figsize=(15,8))

sns.boxplot(x="Cover_Type", y="Hillshade_3pm", data=train, hue="Cover_Type")



# Decoration

plt.title("Box Plot of Hillshade_3pm by Cover_Type", fontsize=22, color="#006400")

plt.legend(title="Cover")

plt.show()
# Draw Plot

plt.figure(figsize=(16,10), dpi= 80)

sns.kdeplot(train.loc[train["Cover_Type"] == 1, "Horizontal_Distance_To_Fire_Points"], shade=True, color="#4169E1", label="Cover=1", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 2, "Horizontal_Distance_To_Fire_Points"], shade=True, color="#FF8C00", label="Cover=2", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 3, "Horizontal_Distance_To_Fire_Points"], shade=True, color="#FF4500", label="Cover=3", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 4, "Horizontal_Distance_To_Fire_Points"], shade=True, color="#BDB76B", label="Cover=4", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 5, "Horizontal_Distance_To_Fire_Points"], shade=True, color="#8B4513", label="Cover=5", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 6, "Horizontal_Distance_To_Fire_Points"], shade=True, color="#9400D3", label="Cover=6", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 7, "Horizontal_Distance_To_Fire_Points"], shade=True, color="#006400", label="Cover=7", alpha=.7)



# Decoration

plt.title("The distribution of the attribute Horizontal_Distance_To_Fire_Points", color="#006400", fontsize=22)

plt.legend()

plt.show()
# Draw Plot

plt.figure(figsize=(15,8))

sns.boxplot(x="Cover_Type", y="Horizontal_Distance_To_Fire_Points", data=train, hue="Cover_Type")



# Decoration

plt.title("Box Plot of Horizontal_Distance_To_Fire_Points by Cover_Type", fontsize=22, color="#006400")

plt.legend(title="Cover")

plt.show()
# Draw Plot

plt.figure(figsize=(16,10), dpi= 80)

sns.kdeplot(train.loc[train["Cover_Type"] == 1, "Wilderness_Area1"], shade=True, color="#4169E1", label="Cover=1", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 2, "Wilderness_Area1"], shade=True, color="#FF8C00", label="Cover=2", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 3, "Wilderness_Area1"], shade=True, color="#FF4500", label="Cover=3", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 4, "Wilderness_Area1"], shade=True, color="#BDB76B", label="Cover=4", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 5, "Wilderness_Area1"], shade=True, color="#8B4513", label="Cover=5", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 6, "Wilderness_Area1"], shade=True, color="#9400D3", label="Cover=6", alpha=.7)

sns.kdeplot(train.loc[train["Cover_Type"] == 7, "Wilderness_Area1"], shade=True, color="#006400", label="Cover=7", alpha=.7)



# Decoration

plt.title("The distribution of the attribute Wilderness_Area1", color="#006400", fontsize=22)

plt.legend()

plt.show()
cols = train.columns



#number of rows=r , number of columns=c

r,c = train.shape



#Create a new dataframe with r rows, one column for each encoded category, and target in the end

data = pd.DataFrame(index=np.arange(0, r),columns=['Wilderness_Area','Soil_Type','Cover_Type'])



#Make an entry in 'data' for each r as category_id, target value

for i in range(0,r):

    w=0;

    s=0;

    # Category1 range

    for j in range(10,14):

        if (train.iloc[i,j] == 1):

            w=j-9  #category class

            break

    # Category2 range        

    for k in range(14,54):

        if (train.iloc[i,k] == 1):

            s=k-13 #category class

            break

    #Make an entry in 'data' for each r as category_id, target value        

    data.iloc[i]=[w,s,train.iloc[i,c-1]]



#Plot for Category1    

sns.countplot(x="Wilderness_Area", hue="Cover_Type", data=data)

plt.show()

#Plot for Category2

plt.rc("figure", figsize=(25, 10))

sns.countplot(x="Soil_Type", hue="Cover_Type", data=data)

plt.show()
ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train["Cover_Type"].to_frame()



#Combine train and test sets

concat_data = pd.concat((train, test), sort=False).reset_index(drop=True)

#Drop the target "Cover_Type" and Id columns

concat_data.drop(["Cover_Type"], axis=1, inplace=True)

concat_data.drop(["Id"], axis=1, inplace=True)

print("Total size is :",concat_data.shape)
concat_data.head()
concat_data.tail()
concat_data.info()
# Count the null columns

null_columns = concat_data.columns[concat_data.isnull().any()]

concat_data[null_columns].isnull().sum()
numeric_features = concat_data.select_dtypes(include=[np.number])

categoricals = concat_data.select_dtypes(exclude=[np.number])



print(f"Numerical features: {numeric_features.shape}")

print(f"Categorical features: {categoricals.shape}")
concat_data.columns
# we split the combined dataset to the original train and test sets

TrainData = concat_data[:ntrain] 

TestData = concat_data[ntrain:]
TrainData.shape, TestData.shape
TrainData.info()
TestData.info()
target = train[["Cover_Type"]]
print("We make sure that both train and target sets have the same row number:")

print(f"Train: {TrainData.shape[0]} rows")

print(f"Target: {target.shape[0]} rows")
# Remove any duplicated column names

concat_data = concat_data.loc[:,~concat_data.columns.duplicated()]
x = TrainData

y = np.array(target)
from sklearn.model_selection import train_test_split

# Split the data set into train and test sets 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
scaler = RobustScaler()



# transform "x_train"

x_train = scaler.fit_transform(x_train)

# transform "x_test"

x_test = scaler.transform(x_test)

#Transform the test set

X_test= scaler.transform(TestData)
# Baseline model of Logistic Regression with default parameters:



logistic_regression = linear_model.LogisticRegression()

logistic_regression_mod = logistic_regression.fit(x_train, y_train)

print(f"Baseline Logistic Regression: {round(logistic_regression_mod.score(x_test, y_test), 3)}")



pred_logistic_regression = logistic_regression_mod.predict(x_test)
cv_method = StratifiedKFold(n_splits=3, 

                            random_state=42

                            )
# Cross validate Logistic Regression model

scores_Logistic = cross_val_score(logistic_regression, x_train, y_train, cv =cv_method, n_jobs = 2, scoring = "accuracy")



print(f"Scores(Cross validate) for Logistic Regression model:\n{scores_Logistic}")

print(f"CrossValMeans: {round(scores_Logistic.mean(), 3)}")

print(f"CrossValStandard Deviation: {round(scores_Logistic.std(), 3)}")
params_LR = {"tol": [0.0001,0.0002,0.0003],

            "C": [0.01, 0.1, 1, 10, 100],

            "intercept_scaling": [1, 2, 3, 4]

              }
GridSearchCV_LR = GridSearchCV(estimator=linear_model.LogisticRegression(), 

                                param_grid=params_LR, 

                                cv=cv_method,

                                verbose=1, 

                                n_jobs=2,

                                scoring="accuracy", 

                                return_train_score=True

                                )
# Fit model with train data

GridSearchCV_LR.fit(x_train, y_train);
best_estimator_LR = GridSearchCV_LR.best_estimator_

print(f"Best estimator for LR model:\n{best_estimator_LR}")
best_params_LR = GridSearchCV_LR.best_params_

print(f"Best parameter values for LR model:\n{best_params_LR}")
print(f"Best score for LR model: {round(GridSearchCV_LR.best_score_, 3)}")
# Test with new parameter for LogisticRegression model

logistic_regression = linear_model.LogisticRegression(C=10, intercept_scaling=1, tol=0.0001, penalty="l2", solver="liblinear", random_state=42)

logistic_regression_mod = logistic_regression.fit(x_train, y_train)

pred_logistic_regression = logistic_regression_mod.predict(x_test)



mse_logistic_regression = mean_squared_error(y_test, pred_logistic_regression)

rmse_logistic_regression = np.sqrt(mean_squared_error(y_test, pred_logistic_regression))

score_logistic_regression_train = logistic_regression_mod.score(x_train, y_train)

score_logistic_regression_test = logistic_regression_mod.score(x_test, y_test)
print(f"Mean Square Error for Logistic Regression = {round(mse_logistic_regression, 3)}")

print(f"Root Mean Square Error for Logistic Regression = {round(rmse_logistic_regression, 3)}")

print(f"R^2(coefficient of determination) on training set = {round(score_logistic_regression_train, 3)}")

print(f"R^2(coefficient of determination) on testing set = {round(score_logistic_regression_test, 3)}")
print("Classification Report")

print(classification_report(y_test, pred_logistic_regression))
print("Confusion Matrix:")

print(confusion_matrix(y_test, pred_logistic_regression))
# Baseline model of K-Nearest Neighbors with default parameters:



knn = KNeighborsClassifier()

knn_mod = knn.fit(x_train, y_train)

print(f"Baseline K-Nearest Neighbors: {round(knn_mod.score(x_test, y_test), 3)}")



pred_knn = knn_mod.predict(x_test)
# Cross validate K-Nearest Neighbors model

scores_knn = cross_val_score(knn, x_train, y_train, cv = cv_method, n_jobs = 2, scoring = "accuracy")



print(f"Scores(Cross validate) for K-Nearest Neighbors model:\n{scores_knn}")

print(f"CrossValMeans: {round(scores_knn.mean(), 3)}")

print(f"CrossValStandard Deviation: {round(scores_knn.std(), 3)}")
params_knn = {"leaf_size": list(range(1,30)),

              "n_neighbors": list(range(1,21)),

              "p": [1,2]}
GridSearchCV_knn = GridSearchCV(estimator=KNeighborsClassifier(), 

                                param_grid=params_knn, 

                                cv=cv_method,

                                verbose=1, 

                                n_jobs=-1,

                                scoring="accuracy", 

                                return_train_score=True

                                )
# Fit model with train data

GridSearchCV_knn.fit(x_train, y_train);
best_estimator_knn = GridSearchCV_knn.best_estimator_

print(f"Best estimator for KNN model:\n{best_estimator_knn}")
best_params_knn = GridSearchCV_knn.best_params_

print(f"Best parameter values:\n{best_params_knn}")
best_score_knn = GridSearchCV_knn.best_score_

print(f"Best score for GNB model: {round(best_score_knn, 3)}")
# Test with new parameter for KNN model

knn = KNeighborsClassifier(leaf_size=1, n_neighbors=1 , p=1)

knn_mod = knn.fit(x_train, y_train)

pred_knn = knn_mod.predict(x_test)



mse_knn = mean_squared_error(y_test, pred_knn)

rmse_knn = np.sqrt(mean_squared_error(y_test, pred_knn))

score_knn_train = knn_mod.score(x_train, y_train)

score_knn_test = knn_mod.score(x_test, y_test)
print(f"Mean Square Error for K_Nearest Neighbor  = {round(mse_knn, 3)}")

print(f"Root Mean Square Error for K_Nearest Neighbor = {round(rmse_knn, 3)}")

print(f"R^2(coefficient of determination) on training set = {round(score_knn_train, 3)}")

print(f"R^2(coefficient of determination) on testing set = {round(score_knn_test, 3)}")
print("Classification Report")

print(classification_report(y_test, pred_knn))
print("Confusion Matrix:")

print(confusion_matrix(y_test, pred_knn))
gaussianNB = GaussianNB()

gaussianNB_mod = gaussianNB.fit(x_train, y_train)

print(f"Baseline Gaussin Navie Bayes: {round(gaussianNB_mod.score(x_test, y_test), 3)}")



pred_gaussianNB = gaussianNB_mod.predict(x_test)
# Cross validate Gaussian Naive Bayes model

scores_GNB = cross_val_score(gaussianNB, x_train, y_train, cv = cv_method, n_jobs = 2, scoring = "accuracy")



print(f"Scores(Cross validate) for Gaussian Naive Bayes model:\n{scores_GNB}")

print(f"CrossValMeans: {round(scores_GNB.mean(), 3)}")

print(f"CrossValStandard Deviation: {round(scores_GNB.std(), 3)}")
params_GNB = {"C": [0.1,0.25,0.5,1],

              "gamma": [0.1,0.5,0.8,1.0],

              "kernel": ["rbf","linear"]}
GridSearchCV_GNB = GridSearchCV(estimator=svm.SVC(), 

                                param_grid=params_GNB, 

                                cv=cv_method,

                                verbose=1, 

                                n_jobs=-1,

                                scoring="accuracy", 

                                return_train_score=True

                                )
# Fit model with train data

GridSearchCV_GNB.fit(x_train, y_train);
best_estimator_GNB = GridSearchCV_GNB.best_estimator_

print(f"Best estimator for DT model:\n{best_estimator_GNB}")
best_params_GNB = GridSearchCV_GNB.best_params_

print(f"Best parameter values:\n{best_params_GNB}")
best_score_GNB = GridSearchCV_GNB.best_score_

print(f"Best score for GNB model: {round(best_score_GNB, 3)}")
mse_gaussianNB = mean_squared_error(y_test, pred_gaussianNB)

rmse_gaussianNB = np.sqrt(mean_squared_error(y_test, pred_gaussianNB))

score_gaussianNB_train = gaussianNB_mod.score(x_train, y_train)

score_gaussianNB_test = gaussianNB_mod.score(x_test, y_test)
print(f"Mean Square Error for Gaussian Naive Bayes = {round(mse_gaussianNB, 3)}")

print(f"Root Mean Square Error for Gaussian Naive Bayes = {round(rmse_gaussianNB, 3)}")

print(f"R^2(coefficient of determination) on training set = {round(score_gaussianNB_train, 3)}")

print(f"R^2(coefficient of determination) on testing set = {round(score_gaussianNB_test, 3)}")
print("Classification Report")

print(classification_report(y_test, pred_gaussianNB))
print("Confusion Matrix:")

print(confusion_matrix(y_test, pred_gaussianNB))
svc = SVC()

svc_mod = svc.fit(x_train, y_train)

print(f"Baseline Support Vector Machine: {round(svc_mod.score(x_test, y_test), 3)}")



pred_svc = svc_mod.predict(x_test)
# Cross validate SVC model

scores_SVC = cross_val_score(svc, x_train, y_train, cv = cv_method, n_jobs = 2, scoring = "accuracy")



print(f"Scores(Cross validate) for SVC model:\n{scores_SVC}")

print(f"CrossValMeans: {round(scores_SVC.mean(), 3)}")

print(f"CrossValStandard Deviation: {round(scores_SVC.std(), 3)}")
params_SVC = {"C": [0.1, 1, 10, 100, 1000],  

              "gamma": [1, 0.1, 0.01, 0.001, 0.0001], 

              "kernel": ["rbf"]

              }
GridSearchCV_SVC = GridSearchCV(estimator=SVC(), 

                                param_grid=params_SVC, 

                                cv=cv_method,

                                verbose=1, 

                                n_jobs=-1,

                                refit = True,

                                scoring="accuracy", 

                                return_train_score=True

                                )
# Fit model with train data

GridSearchCV_SVC.fit(x_train, y_train);
best_estimator_SVC = GridSearchCV_SVC.best_estimator_

print(f"Best estimator for SVC model:\n{best_estimator_SVC}")
best_params_SVC = GridSearchCV_SVC.best_params_

print(f"Best parameter values:\n{best_params_SVC}")
best_score_SVC = GridSearchCV_SVC.best_score_

print(f"Best score for SVC model: {round(best_score_SVC, 3)}")
# Test with new parameter for SVC model

svc = SVC(C=100, gamma=0.1, kernel="rbf" , random_state=42)

svc_mod = svc.fit(x_train, y_train)

pred_svc = svc_mod.predict(x_test)



mse_svc = mean_squared_error(y_test, pred_svc)

rmse_svc = np.sqrt(mean_squared_error(y_test, pred_svc))

score_svc_train = svc_mod.score(x_train, y_train)

score_svc_test = svc_mod.score(x_test, y_test)
print(f"Mean Square Error for Linear Support Vector Machine = {round(mse_svc, 3)}")

print(f"Root Mean Square Error for Linear Support Vector Machine = {round(rmse_svc, 3)}")

print(f"R^2(coefficient of determination) on training set = {round(score_svc_train, 3)}")

print(f"R^2(coefficient of determination) on testing set = {round(score_svc_test, 3)}")
print("Classification Report")

print(classification_report(y_test, pred_svc))
print("Confusion Matrix:")

print(confusion_matrix(y_test, pred_svc))
decision_tree = DecisionTreeClassifier(random_state= 42)

decision_tree_mod = decision_tree.fit(x_train, y_train)

print(f"Baseline Decision Tree: {round(decision_tree_mod.score(x_test, y_test), 3)}")



pred_decision_tree = decision_tree_mod.predict(x_test)
# Cross validate Decision Tree model

scores_DT = cross_val_score(decision_tree, x_train, y_train, cv = cv_method, n_jobs = 2, scoring = "accuracy")



print(f"Scores(Cross validate) for Decision Tree model:\n{scores_DT}")

print(f"CrossValMeans: {round(scores_DT.mean(), 3)}")

print(f"CrossValStandard Deviation: {round(scores_DT.std(), 3)}")
params_DT = {"criterion": ["gini", "entropy"],

             "max_depth": [1, 2, 3, 4, 5, 6, 7, 8],

             "min_samples_split": [2, 3]}
GridSearchCV_DT = GridSearchCV(estimator=DecisionTreeClassifier(), 

                                param_grid=params_DT, 

                                cv=cv_method,

                                verbose=1, 

                                n_jobs=-1,

                                scoring="accuracy", 

                                return_train_score=True

                                )
# Fit model with train data

GridSearchCV_DT.fit(x_train, y_train);
best_estimator_DT = GridSearchCV_DT.best_estimator_

print(f"Best estimator for DT model:\n{best_estimator_DT}")
best_params_DT = GridSearchCV_DT.best_params_

print(f"Best parameter values:\n{best_params_DT}")
best_score_DT = GridSearchCV_DT.best_score_

print(f"Best score for DT model: {round(best_score_DT, 3)}")
decision_tree = DecisionTreeClassifier(criterion="gini", max_depth=8, min_impurity_split=2, min_samples_leaf=0.4, random_state=42)

decision_tree_mod = decision_tree.fit(x_train, y_train)

pred_decision_tree = decision_tree_mod.predict(x_test)



mse_decision_tree = mean_squared_error(y_test, pred_decision_tree)

rmse_decision_tree = np.sqrt(mean_squared_error(y_test, pred_decision_tree))

score_decision_tree_train = decision_tree_mod.score(x_train, y_train)

score_decision_tree_test = decision_tree_mod.score(x_test, y_test)
print(f"Mean Square Error for Decision Tree = {round(mse_decision_tree, 3)}")

print(f"Root Mean Square Error for Decision Tree = {round(rmse_decision_tree, 3)}")

print(f"R^2(coefficient of determination) on training set = {round(score_decision_tree_train, 3)}")

print(f"R^2(coefficient of determination) on testing set = {round(score_decision_tree_test, 3)}")
print("Classification Report")

print(classification_report(y_test, pred_decision_tree))
print("Confusion Matrix:")

print(confusion_matrix(y_test, pred_decision_tree))
random_forest = RandomForestClassifier()

random_forest_mod = random_forest.fit(x_train, y_train)

print(f"Baseline Random Forest: {round(random_forest_mod.score(x_test, y_test), 3)}")



pred_random_forest = random_forest_mod.predict(x_test)
# Cross validate Random forest model

scores_RF = cross_val_score(random_forest, x_train, y_train, cv = cv_method, n_jobs = 2, scoring = "accuracy")



print(f"Scores(Cross validate) for Random forest model:\n{scores_RF}")

print(f"CrossValMeans: {round(scores_RF.mean(), 3)}")

print(f"CrossValStandard Deviation: {round(scores_RF.std(), 3)}")
params_RF = {"min_samples_split": [2, 6, 20],

              "min_samples_leaf": [1, 4, 16],

              "n_estimators" :[100,200,300,400],

              "criterion": ["gini"]             

              }
GridSearchCV_RF = GridSearchCV(estimator=RandomForestClassifier(), 

                                param_grid=params_RF, 

                                cv=cv_method,

                                verbose=1, 

                                n_jobs=2,

                                scoring="accuracy", 

                                return_train_score=True

                                )
# Fit model with train data

GridSearchCV_RF.fit(x_train, y_train);
best_estimator_RF = GridSearchCV_RF.best_estimator_

print(f"Best estimator for RF model:\n{best_estimator_RF}")
best_params_RF = GridSearchCV_RF.best_params_

print(f"Best parameter values for RF model:\n{best_params_RF}")
best_score_RF = GridSearchCV_RF.best_score_

print(f"Best score for RF model: {round(best_score_RF, 3)}")
random_forest = RandomForestClassifier(criterion="gini", n_estimators=400, min_samples_leaf=1, min_samples_split=2, random_state=42)

random_forest_mod = random_forest.fit(x_train, y_train)

pred_random_forest = random_forest_mod.predict(x_test)



mse_random_forest = mean_squared_error(y_test, pred_random_forest)

rmse_random_forest = np.sqrt(mean_squared_error(y_test, pred_random_forest))

score_random_forest_train = random_forest_mod.score(x_train, y_train)

score_random_forest_test = random_forest_mod.score(x_test, y_test)
print(f"Mean Square Error for Random Forest = {round(mse_random_forest, 3)}")

print(f"Root Mean Square Error for Random Forest = {round(rmse_random_forest, 3)}")

print(f"R^2(coefficient of determination) on training set = {round(score_random_forest_train, 3)}")

print(f"R^2(coefficient of determination) on testing set = {round(score_random_forest_test, 3)}")
print("Classification Report")

print(classification_report(y_test, pred_random_forest))
print("Confusion Matrix:")

print(confusion_matrix(y_test, pred_random_forest))
plt.figure(figsize=(16,10))

viz = FeatureImportances(random_forest)

viz.fit(x_train, y_train)

viz.show()
# Plot learning curve

def plot_learning_curve(estimator, title, x, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

        

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="#80CBC4",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="#00897B",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
# Logistic Regression

plot_learning_curve(GridSearchCV_LR.best_estimator_,title = "Logistic Regressionr learning curve", x = x_train, y = y_train, cv = cv_method);
# KNN Classifier

plot_learning_curve(GridSearchCV_knn.best_estimator_,title = "KNN Classifier learning curve", x = x_train, y = y_train, cv = cv_method);
# Gaussian Naive Bayes

plot_learning_curve(GridSearchCV_GNB.best_estimator_,title = "Gaussian Naive Bayes learning curve", x = x_train, y = y_train, cv = cv_method);
# Support Vector Machine(SVM)

plot_learning_curve(GridSearchCV_SVC.best_estimator_,title = "Support Vector Machine(SVM) learning curve", x = x_train, y = y_train, cv = cv_method);
# Decision Tree

plot_learning_curve(GridSearchCV_DT.best_estimator_,title = "Decision Tree learning curve", x = x_train, y = y_train, cv = cv_method);
# Random Forest

plot_learning_curve(GridSearchCV_RF.best_estimator_,title = "Random Forest learning curve", x = x_train, y = y_train, cv = cv_method);
results = pd.DataFrame({

                        "Model": ["Logistic Regression",

                                    "KNN Classifier",

                                    "Gaussian Naive Bayes",

                                    "Support Vector Machine(SVM)",

                                    "Decision Tree",

                                    "Random Forest"],

                        "Score": [logistic_regression_mod.score(x_train, y_train),

                                    knn_mod.score(x_train, y_train),

                                    gaussianNB_mod.score(x_train, y_train),

                                    svc_mod.score(x_train, y_train),

                                    decision_tree_mod.score(x_train, y_train),

                                    random_forest_mod.score(x_train, y_train)]

                        })

result_df = results.sort_values(by="Score", ascending=False)

result_df = result_df.set_index("Score")

result_df.head(10)
vote = VotingClassifier([("Random Forest", random_forest_mod), ("KNN Classifier", knn_mod)])

vote_mod = vote.fit(x_train, y_train.ravel())

vote_pred = vote_mod.predict(x_test)



print(f"Root Mean Square Error test for ENSEMBLE METHODS: {round(np.sqrt(mean_squared_error(y_test, vote_pred)), 3)}")
test["Id"].value_counts()
Final_Submission_ForestCoverType = pd.DataFrame({

        "Id": test["Id"],

        "Cover_Type": vote_mod.predict(X_test)})



Final_Submission_ForestCoverType.to_csv("Final_Submission_ForestCoverType.csv", index=False)

Final_Submission_ForestCoverType.head(10)