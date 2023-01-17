import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline
data = pd.read_csv('Eureka_final_imputed1.csv')
data['converted_in_7days'].value_counts()
data['converted_in_7days'].head()
data.isna().sum()
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data['converted_in_7days'].describe()
# Get names of indexes for which column Age has value 30
indexNames = data[data['converted_in_7days'] == 2 ].index
# Delete these row indexes from dataFrame
data.drop(indexNames , inplace=True)
indexNames = data[data['converted_in_7days'] == 3 ].index
# Delete these row indexes from dataFrame
data.drop(indexNames , inplace=True)
data['converted_in_7days'].value_counts()
data.info()
x = data['converted_in_7days']
target = pd.DataFrame(data=x)
target.describe()
data.drop(['date','country','converted_in_7days','client_id' ], axis=1, inplace=True)
data.loc[(data['region'] != 'Maharashtra') &
(data['region'] != 'Uttar Pradesh') &
(data['region'] != 'Karnataka') &       
(data['region'] != 'Bihar') &     
(data['region'] != 'West Bengal'),'region'] = 'Other_Regions'
data.loc[(data['sourceMedium'] != 'google / cpc') &
(data['sourceMedium'] != 'google / organic') &
(data['sourceMedium'] != '(direct) / (none)') &
(data['sourceMedium'] != 'facebook / social'),'sourceMedium'] = 'Other_sourceMedium '
data['device'].hist()
data['region'].describe()
data['sourceMedium'].describe()
##creating dummies of the categorical variables
region_dummies = pd.get_dummies(data['region'])
sourceMedium_dummies = pd.get_dummies(data['sourceMedium'])
device_dummies = pd.get_dummies(data['device'])
print(device_dummies)
device_dummies
##dropping the orignal categorical variables
data.drop(['region','sourceMedium','device'], axis=1, inplace=True)
### joining the dummy variables
data = data.join(region_dummies)
data = data.join(sourceMedium_dummies)
data = data.join(device_dummies)
data.info()
##data.rename(columns={'Others':'other_regions'}, inplace=True)
##testing a code for grouping

#test = data['region'].apply(lambda x: x == 'Maharashtra')
#test.value_counts()
data.info()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, 
                                                    target, test_size=0.30, 
                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data)
scaled_data = scaler.transform(data)
from sklearn.decomposition import PCA
pca = PCA(n_components=43)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
scaled_data.shape
x_pca.shape
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1], c = data['contactus_top'])
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
pca.components_
df_comp = pd.DataFrame(pca.components_,columns=column_names)
df_comp
plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)
#The amount of variance that each PC explains
var= pca.explained_variance_ratio_

#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)


print(var1)
plt.plot(var1)