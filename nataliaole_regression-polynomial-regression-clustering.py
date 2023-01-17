import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import os



import warnings
warnings.filterwarnings('ignore')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

sns.set(font_scale=1.3)
sklearn.__version__

raw_data = pd.read_csv('/kaggle/input/employee-satisfaction-index-dataset/Employee Satisfaction Index.csv')
raw_data.head()
raw_data.info()
raw_data.describe().T
nulls_summary = pd.DataFrame(raw_data.isnull().any(), columns=['Nulls'])   
nulls_summary['Num_of_nulls [qty]'] = pd.DataFrame(raw_data.isnull().sum())   
nulls_summary['Num_of_nulls [%]'] = round((raw_data.isnull().mean()*100),2)   
print(nulls_summary) 
raw_data['education'].value_counts()
raw_data['recruitment_type'].value_counts()
raw_data['location'].value_counts()
raw_data['Dept'].value_counts()
raw_data.set_index(['emp_id'])
def create_np_array_from_input_list(input_list,output_type):
    np_target = []
    
    entries = []
    entries_idx = []
    for entry in input_list:
        duplicate = 0
        for active_entry in entries:
            if entry == active_entry:
                duplicate = 1
        
        if duplicate == 0:
            entries.append(entry)
        
        no_entries = len(entries)
        
    for i in range(0,no_entries):
        entries_idx.append(i)
        
    for entry in input_list:
        for i in range(0,no_entries):
            if entry == entries[i]:
                np_target.append(entries_idx[i])
                
    if output_type == 'numpy':
        return(np_target)
    elif output_type == 'categories':
        return(entries)
    else:
        raise ValueError('output_type must be \'numpy\' or \'categories\'')
np_data = create_np_array_from_input_list(raw_data['education'],'numpy')
educ = create_np_array_from_input_list(raw_data['education'],'categories')
educ
data_copy = raw_data.copy()

for i in range(0,len(np_data)):
    data_copy.at[i,'education'] = np_data[i]


data_copy
for i in range(0,len(np_data)):
    data_copy.at[i,'location'] = np_data[i]
data_copy
data_copy.corr()
data_copy = data_copy.sort_values(by='job_level')
target1=data_copy[['salary']]
data1=data_copy[['job_level']]
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(data1, target1)
target1_pred = regressor.predict(data1)


regressor.score(data1, target1)


plt.figure(figsize=(8, 6))
plt.title('Linear regression')
plt.xlabel('job_level')
plt.ylabel('salary')
plt.scatter(data1, target1, label='job_level')
plt.plot(data1, target1_pred, color='red', label='model')
plt.legend()
plt.show()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data1, target1, test_size=0.25)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_train shape: {y_train.shape}')


plt.figure(figsize=(8, 6))
plt.title('Linear regression train vs test')
plt.xlabel('job_level')
plt.ylabel('salary')
plt.scatter(X_train, y_train, label='training set', color='gray', alpha=0.5)
plt.scatter(X_test, y_test, label='testing set', color='gold', alpha=0.5)
plt.legend()
plt.plot()
regressor = LinearRegression()
regressor.fit(X_train, y_train)


regressor.score(X_train, y_train)


regressor.score(X_test, y_test)
plt.figure(figsize=(8, 6))
plt.title('Linear regression train ')
plt.xlabel('job_level')
plt.ylabel('salary')
plt.scatter(X_train, y_train, label='train', color='gray', alpha=0.5)
plt.plot(X_train, regressor.intercept_ + regressor.coef_[0] * X_train, color='red')
plt.legend()
plt.plot()
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

data1_poly = poly.fit_transform(data1)
data1_poly
data1_poly.shape
regressor_poly = LinearRegression()
regressor_poly.fit(data1_poly, target1)
target1_pred_lin = regressor.predict(data1)
target1_pred_2 = regressor_poly.predict(data1_poly)
regressor_poly = LinearRegression()
regressor_poly.fit(data1_poly, target1)

target1_pred_2 = regressor_poly.predict(data1_poly)

plt.figure(figsize=(8, 6))
plt.title('Polynomial Regression')
plt.xlabel('job_level')
plt.ylabel('salary')
plt.scatter(data1, target1, label='job_level')
plt.plot(data1, target1_pred_lin, c='red', label='Linear regression')
plt.plot(data1, target1_pred_2, c='green', label='Polynomial Regression, st. 2')
plt.legend()
plt.show()


from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

target1_pred_lin = regressor.predict(data1)
results = pd.DataFrame(data={
    'name': ['Linear regression', 'Polynomial regression with degree=2'],
    'r2_score': [r2_score(target1, target1_pred_lin), r2_score(target1, target1_pred_2)],
    'mae': [mae(target1, target1_pred_lin), mae(target1, target1_pred_2)],
    'mse': [mse(target1, target1_pred_lin), mse(target1, target1_pred_2)]
        
    })
results
X= raw_data.iloc[:, [2,12]].values
#Visualise data points
plt.scatter(X[:,0],X[:,1],c='black')
plt.xlabel('age')
plt.ylabel('salary')
plt.show()

from sklearn.cluster import KMeans
kmeans5 = KMeans(n_clusters=5)
y_kmeans5 = kmeans5.fit_predict(X)
print(y_kmeans5)

kmeans5.cluster_centers_




Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(X)
    kmeans.fit(X)
    Error.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()


kmeansmodel = KMeans(n_clusters= 4)
y_kmeans= kmeansmodel.fit_predict(X)
kmeans.cluster_centers_
plt.scatter(X[:, 0], X[:, 1], s = 300, c = y_kmeans, cmap='rainbow')