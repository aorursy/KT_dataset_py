# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read Data

import numpy as np                     # Linear Algebra (calculate the mean and standard deviation)

import pandas as pd                    # manipulate data, data processing, load csv file I/O (e.g. pd.read_csv)



# Visualization

import seaborn as sns                  # Visualization using seaborn

import matplotlib.pyplot as plt        # Visualization using matplotlib

%matplotlib inline

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots



# style

import plotly.io as pio

pio.templates.default = "plotly_dark"

plt.style.use("fivethirtyeight")       # Set Graphs Background style using matplotlib

sns.set_style("darkgrid")              # Set Graphs Background style using seaborn



# ML model building; Pre Processing & Evaluation

from sklearn.model_selection import train_test_split                     # split  data into training and testing sets

from sklearn.linear_model import LinearRegression, Lasso, Ridge          # Linear Regression, Lasso and Ridge

from sklearn.linear_model import LogisticRegression                      # Logistic Regression

from sklearn.tree import DecisionTreeRegressor                           # Decision tree Regression

from sklearn.ensemble import RandomForestClassifier                      # this will make a Random Forest Classifier

from sklearn import svm                                                  # this will make a SVM classificaiton

from sklearn.svm import SVC                                              # import SVC from SVM

import xgboost

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, classification_report      # this creates a confusion matrix

from sklearn.metrics import roc_curve,auc                                # ROC

from sklearn.preprocessing import StandardScaler                         # Standard Scalar

from sklearn.model_selection import GridSearchCV                         # this will do cross validation



import warnings                        # Ignore Warnings

warnings.filterwarnings("ignore")
# Import first 5 rows

cover = pd.read_csv("/kaggle/input/forest-cover-type-prediction/train.csv")

test = pd.read_csv("/kaggle/input/forest-cover-type-prediction/test.csv")
display(cover.head())

display(test.head())
# checking dimension (num of rows and columns) of dataset

print("Training data shape (Rows, Columns):",cover.shape)

print("Training data shape (Rows, Columns):",test.shape)
cover['Cover_Type'].value_counts()
# check dataframe structure like columns and its counts, datatypes & Null Values

cover.info()
cover.dtypes.value_counts()
# Gives number of data points in each variable

cover.count()
# Listing Number of missing values by feature column wise

cover.isnull().sum()
# any() check null values by columns

cover.isnull().any()
plt.figure(figsize=(17,10))

sns.heatmap(cover.isnull(), yticklabels=False, cbar=False, cmap='viridis')
for column in cover.columns:

    print(column,cover[column].nunique())
numerical_features = cover.select_dtypes(exclude='object')

numerical_features
discrete_feature=[feature for feature in numerical_features if len(cover[feature].unique())<25]

print("Discrete Variables Count: {}".format(len(discrete_feature)))
continuous_features=[feature for feature in numerical_features if feature not in discrete_feature+['Cover_Type']]

print("Continuous feature Count {}".format(len(continuous_features)))
continuous_features
fig, ax = plt.subplots(3,4, figsize=(14,9))

sns.distplot(cover.Elevation, bins = 20, ax=ax[0,0]) 

sns.distplot(cover.Aspect, bins = 20, ax=ax[0,1]) 

sns.distplot(cover.Slope, bins = 20, ax=ax[0,2]) 

sns.distplot(cover.Horizontal_Distance_To_Hydrology, bins = 20, ax=ax[0,3])

sns.distplot(cover.Vertical_Distance_To_Hydrology, bins = 20, ax=ax[1,0]) 

sns.distplot(cover.Horizontal_Distance_To_Roadways, bins = 20, ax=ax[1,1]) 

sns.distplot(cover.Hillshade_9am, bins = 20, ax=ax[1,2]) 

sns.distplot(cover.Hillshade_Noon, bins = 20, ax=ax[1,3])

sns.distplot(cover.Hillshade_3pm, bins = 20, ax=ax[2,0])

sns.distplot(cover.Horizontal_Distance_To_Fire_Points, bins = 20, ax=ax[2,1])

plt.show()
plt.figure(figsize=(20,60), facecolor='white')

plotnumber =1

for feature in continuous_features:

    data=cover.copy()

    ax = plt.subplot(12,3,plotnumber)

    plt.scatter(cover[feature], cover['Cover_Type'])

    plt.xlabel(feature)

    plt.ylabel('Cover_Type')

    plt.title(feature)

    plotnumber+=1

plt.show()
# boxplot on numerical features to find outliers

plt.figure(figsize=(18,15), facecolor='white')

plotnumber =1

for numerical_feature in numerical_features:

    ax = plt.subplot(19,3,plotnumber)

    sns.boxplot(cover[numerical_feature])

    plt.xlabel(numerical_feature)

    plotnumber+=1

plt.show()
plt.figure(figsize = (14,12))

plt.title('Correlation of Numeric Features with Sale Price', y=1, size=16)

sns.heatmap(cover.corr(), square = True, vmax=0.8)
corr = cover.drop('Cover_Type', axis=1).corr()

plt.figure(figsize=(17, 14))



sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);
corrmat = cover.iloc[:,:10].corr()

f, ax = plt.subplots(figsize = (12,8))

sns.heatmap(corrmat, cmap='viridis', vmax=0.8, annot=True, square=True);
# descriptive statistics (numerical columns)

pd.set_option('display.max_columns', None)

cover.describe()
# Histogram for "Elevation"

plt.figure(figsize=(5,4))

sns.distplot(cover.Elevation,rug=True)
sns.distplot(cover.Aspect)

plt.grid()
# Histogram for "Elevation"

sns.boxplot(cover['Elevation'])
sns.boxplot(cover.Vertical_Distance_To_Hydrology)

plt.title('Vertical_Distance_To_Hydrology')
# Histogram for "All Features"

cover.hist(figsize=(16, 20), bins=50, xlabelsize=7, ylabelsize=7);
sns.violinplot(x=cover['Cover_Type'],y=cover['Elevation'])

plt.grid()
sns.violinplot(x=cover['Cover_Type'],y=cover['Aspect'])

plt.grid()
# vertical distance to the hydrology column

sns.violinplot(x=cover.Cover_Type, y=cover.Vertical_Distance_To_Hydrology)
# Line Plot between "Aspect" and "Cover_Type"

plt.figure(figsize=(7,6))

sns.lineplot(x=cover['Aspect'], y=cover['Cover_Type'])



plt.xlabel('Aspect', fontsize=15, fontweight='bold')

plt.ylabel('Cover_Type', fontsize=15, fontweight='bold')



plt.title('Aspect Vs Cover_Type', fontsize=18, fontweight='bold')



plt.xticks(fontsize=12)

plt.yticks(fontsize=12)



plt.show()
# Line Plot between "Slope" and "Cover_Type"

plt.figure(figsize=(7,6))

sns.lineplot(x=cover['Slope'], y=cover['Cover_Type'])



plt.xlabel('Slope', fontsize=15, fontweight='bold')

plt.ylabel('Cover_Type', fontsize=15, fontweight='bold')



plt.title('Slope Vs Cover_Type', fontsize=18, fontweight='bold')



plt.xticks(fontsize=12)

plt.yticks(fontsize=12)



plt.show()
fig = px.scatter(cover, x='Elevation', y= 'Horizontal_Distance_To_Roadways', color='Cover_Type', width=800, height=400)

fig.show()
# Scatter Plot between "GrLivArea" and "SalePrice"

plt.figure(figsize=(7,6))

sns.scatterplot(cover.Aspect, cover.Hillshade_3pm)



plt.xlabel('Aspect', fontsize=15, fontweight='bold')

plt.ylabel('Hillshade_3pm', fontsize=15, fontweight='bold')



plt.title('Aspect Vs Hillshade_3pm', fontsize=18, fontweight='bold')



plt.xticks(fontsize=12)

plt.yticks(fontsize=12)



plt.show()
# Scatter Plot between "Horizontal_Distance_To_Hydrology" and "Vertical_Distance_To_Hydrology" variable

plt.figure(figsize=(7,6))

sns.scatterplot(cover['Horizontal_Distance_To_Hydrology'], cover['Vertical_Distance_To_Hydrology'])



plt.xlabel('Horizontal_Distance_To_Hydrology', fontsize=15, fontweight='bold')

plt.ylabel('Vertical_Distance_To_Hydrology', fontsize=15, fontweight='bold')



plt.title('Horizontal_Distance_To_Hydrology Vs Vertical_Distance_To_Hydrology', fontsize=18, fontweight='bold')



plt.xticks(fontsize=12)

plt.yticks(fontsize=12)



plt.show()
# Scatter Plot between "Hillshade_Noon" and "Hillshade_3pm"

fig = px.scatter(cover,x='Hillshade_Noon',y= 'Hillshade_3pm',color='Cover_Type',width=800,height=400)

fig.show()
# Scatter Plot between "Aspect" and "Hillshade_9am"

fig = px.scatter(cover,x='Aspect',y= 'Hillshade_9am',color='Cover_Type',width=800,height=400)

fig.show()
# Scatter Plot between "Hillshade_9am" and "Hillshade_3pm"

fig = px.scatter(cover,x='Hillshade_9am',y= 'Hillshade_3pm',color='Cover_Type',width=800,height=400)

fig.show()
# Scatter Plot between "Slope" and "Hillshade_Noon"

fig = px.scatter(cover,x='Slope',y= 'Hillshade_Noon',color='Cover_Type',width=800,height=400)

fig.show()
# Count Plot for "Cover_Type"

plt.figure(figsize = (15, 9))

sns.countplot(x = 'Cover_Type', data = cover)

xt = plt.xticks(rotation=45)
# A violin plot is a hybrid of a box plot and a kernel density plot, which shows peaks in the data.

cols = cover.columns

size = len(cols) - 1 # We don't need the target attribute

# x-axis has target attributes to distinguish between classes

x = cols[size]

y = cols[0:size]



for i in range(0, size):

    sns.violinplot(data=cover, x=x, y=y[i])

    plt.show()
sns.set()

columns = cover.iloc[:,:10]

sns.pairplot(columns, kind ='scatter', diag_kind='kde')
# Checking the value count for different soil_types

for i in range(10, cover.shape[1]-1):

    j = cover.columns[i]

    print (cover[j].value_counts())
cover.iloc[:,:10].skew()
# Checking the skewness of "LotArea" attributes

sns.distplot(cover['Horizontal_Distance_To_Hydrology'])

Skew_Horizontal_Distance_To_Hydrology = cover['Horizontal_Distance_To_Hydrology'].skew()

plt.title("Skew:"+str(Skew_Horizontal_Distance_To_Hydrology))
# calculating the square for the column df['LotArea'] column

sns.distplot(np.sqrt(cover['Horizontal_Distance_To_Hydrology']))

Skew_Horizontal_Distance_To_Hydrology_sqrt = np.sqrt(cover['Horizontal_Distance_To_Hydrology']+1).skew()

plt.title("Skew:"+str(Skew_Horizontal_Distance_To_Hydrology_sqrt))
# Checking the skewness of "Vertical_Distance_To_Hydrology" attributes

sns.distplot(cover['Vertical_Distance_To_Hydrology'])

Skew_Vertical_Distance_To_Hydrology = cover['Vertical_Distance_To_Hydrology'].skew()

plt.title("Skew:"+str(Skew_Vertical_Distance_To_Hydrology))
# Checking the skewness of "Horizontal_Distance_To_Roadways" attributes

sns.distplot(cover['Horizontal_Distance_To_Roadways'])

Skew_Horizontal_Distance_To_Roadways = cover['Horizontal_Distance_To_Roadways'].skew()

plt.title("Skew:"+str(Skew_Horizontal_Distance_To_Roadways))
# calculating the square for the column df['Horizontal_Distance_To_Roadways'] column

sns.distplot(np.sqrt(cover['Horizontal_Distance_To_Roadways']))

Skew_Horizontal_Distance_To_Roadways_sqrt = np.sqrt(cover['Horizontal_Distance_To_Roadways']).skew()

plt.title("Skew:"+str(Skew_Horizontal_Distance_To_Roadways_sqrt))
# Checking the skewness of "Hillshade_9am" attributes

sns.distplot(cover['Hillshade_9am'])

Skew_Hillshade_9am = cover['Hillshade_9am'].skew()

plt.title("Skew:"+str(Skew_Hillshade_9am))
# calculating the square for the column df['Hillshade_9am'] column

sns.distplot(np.power(cover['Hillshade_9am'],5))

Skew_Hillshade_9am_power = np.power(cover['Hillshade_9am'],5).skew()

plt.title("Skew:"+str(Skew_Hillshade_9am_power))
# Checking the skewness of "Hillshade_Noon" attributes

sns.distplot(cover['Hillshade_Noon'])

Skew_Hillshade_Noon = cover['Hillshade_Noon'].skew()

plt.title("Skew:"+str(Skew_Hillshade_Noon))
# calculating the square for the column df['Hillshade_9am'] column

sns.distplot(np.power(cover['Hillshade_Noon'],5))

Skew_Hillshade_Noon_power = np.power(cover['Hillshade_Noon'],5).skew()

plt.title("Skew:"+str(Skew_Hillshade_Noon_power))
# Checking the skewness of "Horizontal_Distance_To_Fire_Points" attributes

sns.distplot(cover['Horizontal_Distance_To_Fire_Points'])

Skew_Horizontal_Distance_To_Fire_Points = cover['Horizontal_Distance_To_Fire_Points'].skew()

plt.title("Skew:"+str(Skew_Horizontal_Distance_To_Fire_Points))
# calculating the square for the column df['Horizontal_Distance_To_Fire_Points'] column

sns.distplot(np.cbrt(cover['Horizontal_Distance_To_Fire_Points']))

Skew_Horizontal_Distance_To_Fire_Points_cube = np.cbrt(cover['Horizontal_Distance_To_Fire_Points']).skew()

plt.title("Skew:"+str(Skew_Horizontal_Distance_To_Fire_Points_cube))
# Checking the skewness of "Slope" attributes

sns.distplot(cover['Slope'])

Skew_Slope = cover['Slope'].skew()

plt.title("Skew:"+str(Skew_Slope))
# calculating the square for the column df['Slope'] column

sns.distplot(np.sqrt(cover['Slope']))

Skew_Slope_sqrt = np.sqrt(cover['Slope']).skew()

plt.title("Skew:"+str(Skew_Slope_sqrt))
cover['dist_hydr'] = np.sqrt(cover['Vertical_Distance_To_Hydrology']**2 + cover['Horizontal_Distance_To_Hydrology']**2)

test['dist_hydr'] = np.sqrt(cover['Vertical_Distance_To_Hydrology']**2 + cover['Horizontal_Distance_To_Hydrology']**2)
sns.distplot(cover['dist_hydr'], color='green')
cover.head()
test.head()
# standardizing the columns except "soil type and wilderness_area" since they are binary  



cover_new = cover.iloc[:,:11]

cover_new['dist_hydr'] = cover['dist_hydr']

cover_new.info()
sc = StandardScaler()

sc.fit(cover_new)

cover_new = sc.transform(cover_new)
cover_new[:10,1]
cover.iloc[:,1:11] = cover_new[:,0:10]
cover['dist_hydr'] = cover_new[:,10]
# Correlation of "independant features" with "target" feature

# Drop least correlated features; since we have hign dimmensional data 

cover_corr = cover.corr()

cover_corr['Cover_Type'].abs().sort_values(ascending=False)
# Independant variable

X = cover.drop(columns='Cover_Type',axis=1)

# Dependant variable

y = cover['Cover_Type']
# split  data into training and testing sets of 70:30 ratio

# 20% of test size selected

# random_state is random seed

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=4)
# shape of X & Y test / train

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
clf_accuracy=[]
LogReg = LogisticRegression(max_iter=1000)

LogReg.fit(X_train, y_train)
y_pred_LogReg = LogReg.predict(X_test)

clf_accuracy.append(accuracy_score(y_test, y_pred_LogReg))

print(accuracy_score(y_test, y_pred_LogReg))
print("Train Score {:.2f} & Test Score {:.2f}".format(LogReg.score(X_train, y_train), LogReg.score(X_test, y_test)))
print("Model\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2")

print("""Logistic Regression \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(

            np.sqrt(mean_squared_error(y_test, y_pred_LogReg)),

            mean_squared_error(y_test, y_pred_LogReg),

            mean_absolute_error(y_test, y_pred_LogReg),

            r2_score(y_test, y_pred_LogReg)))
DTR = DecisionTreeRegressor()

DTR.fit(X_train, y_train)
y_pred_DTR = DTR.predict(X_test)
clf_accuracy.append(accuracy_score(y_test, y_pred_DTR))

print(accuracy_score(y_test, y_pred_DTR))
print("Train Score {:.2f} & Test Score {:.2f}".format(DTR.score(X_train, y_train), DTR.score(X_test, y_test)))
print("Model\t\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2")

print("""Decision Tree Regressor \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(

            np.sqrt(mean_squared_error(y_test, y_pred_DTR)),

            mean_squared_error(y_test, y_pred_DTR),

            mean_absolute_error(y_test, y_pred_DTR),

            r2_score(y_test, y_pred_DTR)))



plt.scatter(y_test, y_pred_DTR)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)



plt.xlabel("Predicted")

plt.ylabel("True")



plt.title("Decision Tree Regressor")



plt.show()
rf = RandomForestClassifier()

rf.fit(X_train,y_train)
pred_rf = rf.predict(X_test)
clf_accuracy.append(accuracy_score(y_test, pred_rf ))

print(accuracy_score(y_test, pred_rf ))
print("Train Score {:.2f} & Test Score {:.2f}".format(rf.score(X_train, y_train), rf.score(X_test, y_test)))
print("Model\t\t RMSE \t\t MSE \t\t MAE \t\t R2")

print("""Random Forest \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(

            np.sqrt(mean_squared_error(y_test, pred_rf )),

            mean_squared_error(y_test, pred_rf ),

            mean_absolute_error(y_test, pred_rf ),

            r2_score(y_test, pred_rf )))
KNN = KNeighborsClassifier()



l=[i for i in range(1,11)]

accuracy=[]



for i in l:

    KNN = KNeighborsClassifier(n_neighbors=i, weights='distance')

    KNN.fit(X_train, y_train)

    pred_knn = KNN.predict(X_test)

    accuracy.append(accuracy_score(y_test, pred_knn))



plt.plot(l,accuracy)

plt.title('knn_accuracy plot')

plt.xlabel('neighbors')

plt.ylabel('accuracy')

plt.grid()



print(max(accuracy))



clf_accuracy.append(max(accuracy))
print("Model\t\t RMSE \t\t MSE \t\t MAE \t\t R2")

print("""Random Forest \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(

            np.sqrt(mean_squared_error(y_test, pred_rf )),

            mean_squared_error(y_test, pred_rf ),

            mean_absolute_error(y_test, pred_rf ),

            r2_score(y_test, pred_rf )))
import xgboost

reg_xgb = xgboost.XGBClassifier(max_depth=7)

reg_xgb.fit(X_train,y_train)
# predicting X_test

y_pred_xgb = reg_xgb.predict(X_test)
print("Train Score {:.2f} & Test Score {:.2f}".format(reg_xgb.score(X_train,y_train),reg_xgb.score(X_test,y_test)))
clf_accuracy.append(accuracy_score(y_test, y_pred_xgb))

print(accuracy_score(y_test, y_pred_xgb))
print("Model\t\t RMSE \t\t MSE \t\t MAE \t\t R2")

print("""XGBClassifier \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(

            np.sqrt(mean_squared_error(y_test, pred_rf )),

            mean_squared_error(y_test, pred_rf ),

            mean_absolute_error(y_test, pred_rf ),

            r2_score(y_test, pred_rf )))
nb = GaussianNB()

nb.fit(X_train,y_train)
pred_nb = nb.predict(X_test)
clf_accuracy.append(accuracy_score(y_test, pred_nb))

print(accuracy_score(y_test, pred_nb))
print("Train Score {:.2f} & Test Score {:.2f}".format(nb.score(X_train,y_train),nb.score(X_test,y_test)))
print("Model\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2")

print("""Naive Bayes Classifier \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(

            np.sqrt(mean_squared_error(y_test, pred_nb )),

            mean_squared_error(y_test, pred_nb ),

            mean_absolute_error(y_test, pred_nb ),

            r2_score(y_test, pred_nb )))
# classification Report

print(classification_report(y_test, pred_rf))
# Confusion Matrix

cf_matrix = confusion_matrix(y_test, pred_rf)

print('Confusion Matrix \n',cf_matrix)
plt.figure(figsize=(7,6))

sns.heatmap(cf_matrix, cmap='coolwarm', annot=True, linewidth=1, fmt="d")

plt.show()
classifier_list=['Logistic Regression','Decision Tree','Random Forest','KNN','xgboost','nbayes']

clf_accuracy1 = [0.6488095238095238,0.781415343915344,0.8621031746031746,0.6458333333333334,0.8753306878306878,0.6504629629629629]
plt.figure(figsize=(7,6))

sns.barplot(x=clf_accuracy1, y=classifier_list)

plt.grid()

plt.xlabel('accuracy')

plt.ylabel('classifier')

plt.title('classifier vs accuracy plot')
models = [LogReg, DTR, rf, KNN, reg_xgb, nb]

names = ['Logistic Regression','Decision Tree','Random Forest','KNN','xgboost','nbayes']

rmses = []



for model in models:

    rmses.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))



x = np.arange(len(names)) 

width = 0.3



fig, ax = plt.subplots(figsize=(10,7))

rects = ax.bar(x, rmses, width)

ax.set_ylabel('RMSE')

ax.set_xlabel('Models')



ax.set_title('RMSE with Different Algorithms')



ax.set_xticks(x)

ax.set_xticklabels(names, rotation=45)



fig.tight_layout()
y_pred_test = reg_xgb.predict(test)
submission = pd.DataFrame({'Id': test['Id'], 'Cover_Type': y_pred_test})

submission.to_csv('Forest Covetype.csv', index=False)
from scipy import stats

from scipy.stats import f_oneway

from scipy.stats import ttest_ind
stats.ttest_1samp(cover['Elevation'],0)
street_table = pd.crosstab(cover['Elevation'], cover['Cover_Type'])

print(street_table)
street_table.values 
# Observed Values

Observed_Values = street_table.values 

print("Observed Values :-\n",Observed_Values)
val = stats.chi2_contingency(street_table)

val
Expected_Values = val[3]
no_of_rows = len(street_table.iloc[0:2,0])

no_of_columns = len(street_table.iloc[0,0:2])

ddof = (no_of_rows-1)*(no_of_columns-1)

print("Degree of Freedom:-",ddof)

alpha = 0.05
from scipy.stats import chi2

chi_square = sum([(o-e)**2./e for o,e in zip(Observed_Values, Expected_Values)])

chi_square_statistic = chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)
critical_value = chi2.ppf(q=1-alpha,df=ddof)

print('critical_value:',critical_value)
# p-value

p_value = 1-chi2.cdf(x=chi_square_statistic, df=ddof)

print('p-value:', p_value)

print('Significance level: ',alpha)

print('Degree of Freedom: ',ddof)

print('p-value:', p_value)
if chi_square_statistic>=critical_value:

    print("Reject H0,There is a relationship between 2 categorical variables")

else:

    print("Retain H0,There is no relationship between 2 categorical variables")

    

if p_value<=alpha:

    print("Reject H0,There is a relationship between 2 categorical variables")

else:

    print("Retain H0,There is no relationship between 2 categorical variables")
import statsmodels.api as sms

model = sms.OLS(y,X).fit()

model.summary()