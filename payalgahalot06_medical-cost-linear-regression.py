# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib.pyplot as plt  
import seaborn as sn
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix,r2_score
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the dataset 
data = pd.read_csv('../input/insurance1/Insurance.csv')
data.head()
data.info()
data.shape
## Dataset has 1338 records with 7 attributes
data.describe(include = 'all')
## We will first check the distribution of all the attributes in a dataset
## For continuous vars age,bmi wrt charges we will plot histogram using distplot from seaborn
sn.distplot(data.age, bins=20, kde=True, rug=True);
## Checking the distribution of age wrt charges using plot
data.plot(x='age', y='charges',style = 'o')  
plt.title('Age vs Charges')  
plt.xlabel('age')  
plt.ylabel('charges')  
plt.show()
## For Bmi
sn.distplot(data.bmi,bins = 20,kde = True, rug = False)
## Plotting bmi wrt charges
data.plot(x='bmi', y='charges',style = 'o')  
plt.title('BMI vs Charges')  
plt.xlabel('bmi')  
plt.ylabel('charges')  
plt.show()
## Plotting Categorical Vars wrt charges
## Note that the children attr in the data is not catgerical so it needs to be converted 1st
data.children.describe()
data.children = data.children.astype('category')
data.children.describe()
## Categorical vars are sex,children,smoker and region
## For children distribution we will use plot 
data['children'].value_counts().plot(kind='bar')
## Plotting Children wrt charges we will use seaborn barplot
sn.barplot(y = 'charges', x = 'children',data = data, edgecolor = 'w')
plt.show()
## For Sex
data.sex.value_counts()
data['sex'].value_counts().plot(kind='bar')
sn.barplot(y = 'charges', x = 'sex',data = data, edgecolor = 'w')
plt.show()
## For Smoker
data.smoker.value_counts()
data['smoker'].value_counts().plot(kind = 'bar')
sn.barplot(y = 'charges', x = 'smoker', data = data, edgecolor ='w')
## For Region
data.region.value_counts()
data['region'].value_counts().plot(kind = 'bar')
sn.barplot(y = 'charges', x = 'region', data = data, edgecolor ='w')
## Checking the corelation of the continous vars through seaborn heatmap
data.corr()
plt.figure(figsize=(12,10))
cor = data.corr()
sn.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
#Correlation with output variable
cor_target = abs(cor["charges"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features
## From the above HeatMap it can be seen that the vars age and bmi are not at all related to charges 
## Linear Regression 1st assumptions aim that there has to be a linear relation between dependent and independent vars
## Hence it can be infered that age and bmi are not significant vars in defining charges
## We can check the outliers in the data by plotting Box-Plot
sn.boxplot(data=data)
## Now there are two ways to detect and remove the outliers
## 1)InterQuartile Range
##   Data point that falls outside of 1.5 times of an Interquartile range above the 3rd quartile (Q3) and below the 1st quartile (Q1)
## 2)Z-Score
##     Data point that falls outside of 3 standard deviations. we can use a z score and if the z score falls outside of 2 standard deviation.
## IQR
Q1=data['bmi'].quantile(0.25)
Q3=data['bmi'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1 - (1.5*IQR)
Upper_Whisker = Q3 + (1.5*IQR)

print(Lower_Whisker, Upper_Whisker)
## Filtering the dataset containing non-outlier records

df1 = data[data['bmi'] < Upper_Whisker]
data.shape
df1.shape
## So the filtered outlier dataset has 1329 records using IQR
## Outlier Detection using Z-Score
from scipy import stats
z=np.abs(stats.zscore(data.bmi))
print(z)
threshold=3
print(np.where(z>3))
df2=data[(z< 3)]
print(df2)
df2.shape
## So the filtered outlier dataset has 1334 records using Z-score 
## We will take dataset after removal of outliers wrt Zscore first for analysis
## So our analysis Dataset is df2
df2.head()
df2.describe(include = 'all')
df2.info()
## For a Linear Regression Problem both Dependent and independent variables have to be numeric always 
## So using Label Encoder
le = LabelEncoder()
df2.sex    = le.fit_transform(df2.sex)
df2.smoker = le.fit_transform(df2.smoker)
df2.region = le.fit_transform(df2.region)

df2.sex    = df2.sex.astype('category')
df2.smoker = df2.smoker.astype('category')
df2.region = df2.region.astype('category')

df2.sex.describe()
df2.smoker.describe()
df2.region.describe()
## Now we need to do feature selection, We have already seen two features those are not important using heatmap
## But that is not most advisable method to select features as Filter method is less accurate. 
## So we will try using Recurrsive Feature Elimination method and we will do Modelling 
## So we need to split the data first
df2.head()
X = df2.iloc[:,0:6]
X.head()
Y = df2.iloc[:,6]
Y.head()
## Here there are 6 Features in the input
from sklearn.feature_selection import RFE
#no of features
nof_list=np.arange(1,6)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 1001)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
cols = list(X.columns)
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 5)             
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,Y)  
#Fitting the data to model
model.fit(X_rfe,Y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)
# So taking only these 5 features for Linear Regression removing sex attr
df2.columns
data_x = df2[['age', 'bmi', 'children','smoker','region']]
data_x.head()
data_y = df2.iloc[:,6]
data_y.head()
## Splitting dataset into Train-Test Data
data_x_train,data_x_test,data_y_train,data_y_test = train_test_split(data_x,data_y,test_size = 0.2, random_state = 101)
lr = LinearRegression()
lr.fit(data_x_train,data_y_train)
predval = lr.predict(data_x_test)
## Now compare the actual output values for data_x_test with the predicted values, execute the following script:
compare = pd.DataFrame({'Actual': data_y_test, 'Predicted': predval})
compare
## We can also visualize comparison result as a bar graph using the below script.

## As the number of records is huge, for representation purpose I’m taking just 25 records.
df1 = compare.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(data_y_test, predval))  
print('Mean Squared Error:', metrics.mean_squared_error(data_y_test, predval))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(data_y_test, predval)))
#To retrieve the intercept:
print(lr.intercept_)
#For retrieving the slope:
print(lr.coef_)
r2_score(data_y_test, predval)
le = LabelEncoder()
data.sex    = le.fit_transform(data.sex)
data.smoker = le.fit_transform(data.smoker)
data.region = le.fit_transform(data.region)

data.sex    = data.sex.astype('category')
data.smoker = data.smoker.astype('category')
data.region = data.region.astype('category')
data.sex.value_counts()
data.columns
## Remving attr sex as RFE initailly eliminated it
data_x1 = data[['age', 'bmi', 'children','smoker','region']]
data_x1.head()
data_y1 = data.iloc[:,6]
data_y1.head()
data_x1_train,data_x1_test,data_y1_train,data_y1_test = train_test_split(data_x1,data_y1,test_size = 0.2, random_state = 101)
lr1 = LinearRegression()
lr1.fit(data_x1_train,data_y1_train)
predval1 = lr1.predict(data_x1_test)
compare1 = pd.DataFrame({'Actual': data_y1_test, 'Predicted': predval1})
compare1
print('Mean Absolute Error:', metrics.mean_absolute_error(data_y1_test, predval1))  
print('Mean Squared Error:', metrics.mean_squared_error(data_y1_test, predval1))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(data_y1_test, predval1)))
#To retrieve the intercept:
print(lr.intercept_)
#For retrieving the slope:
print(lr.coef_)
r2_score(data_y1_test, predval1)
