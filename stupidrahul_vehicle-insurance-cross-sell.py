import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from matplotlib import style
style.use('ggplot')
df_train = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
df_test = pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')
df_train.head()
df_train.describe(include = 'all')
df_train.info()
df_train.isnull().sum()
df_test.isnull().sum()
print('train-set shape:', df_train.shape)
print('test-set shape:',df_test.shape)
df_train['Response'].value_counts().plot.bar()
plt.show()
sns.distplot(df_train['Age'])
plt.show()
sns.distplot(df_train['Annual_Premium'])
plt.show()
sns.distplot(np.sqrt(df_train['Annual_Premium'].values))
plt.show()
plt.subplot(121)
df_train[df_train['Previously_Insured'] == 0]['Response'].value_counts().plot.bar()
plt.tight_layout()
plt.title('Previously not Insured')
plt.subplot(122)
df_train[df_train['Previously_Insured'] == 1]['Response'].value_counts().plot.bar()
plt.title('Previously Insured')
plt.tight_layout()
plt.show()
plt.subplot(121)
df_train[df_train['Vehicle_Damage'] == 'Yes']['Response'].value_counts().plot.bar()
plt.tight_layout()
plt.title('Vehicle Damage')
plt.subplot(122)
df_train[df_train['Vehicle_Damage'] == 'No']['Response'].value_counts().plot.bar()
plt.title('Vehicle not Damage')
plt.tight_layout()
plt.show()
sns.lmplot( x = 'Annual_Premium' , y = 'Age' , data = df_train)
plt.title('Annual Premium V/s Age')
plt.show()
sns.lmplot( x = 'Annual_Premium' , y = 'Region_Code' , data = df_train)
plt.title('Annual premium V/s Region Code')
plt.show()
sns.lmplot( x = 'Annual_Premium' , y = 'Policy_Sales_Channel' ,  data =  df_train)
plt.title('Annual premium V/s Policy sales channel')
plt.show()
sns.lmplot( x ='Annual_Premium' ,  y = 'Vintage' ,  data = df_train)
plt.title('Annual_Premium V/s Vintage ')
plt.show()
# let's put together the training dataset and the test dataset for that first we need to assign somthing in the
# response column of the test dataset so that we can filter test dataset once the feature engineering id done

df_test['Response'] = -1 
df = pd.concat([df_train , df_test])
print(df.shape)
df.head()
# Since 'Age' and 'Annual_Premium' is correlated so let's extract a feature using these two column
df['Age_Wise_Premium'] = df['Annual_Premium'] / df['Age']

# The person whose vehicle is Damage and not Insured, more tend to Insured their vehicle So lat's make a column
# using these two column

df['Not_Insured_Damage'] = df['Previously_Insured'].astype('str')+'_'+df['Vehicle_Damage'].astype('str')
df.head()
# Annual_Premium region wise mean
df['Region_Wise_Mean_Premium'] = df.groupby('Region_Code')['Annual_Premium'].transform('mean')
# Annual Premium mean according to policy sales channel
df['Channel_Wise_Mean_Premium'] = df.groupby('Policy_Sales_Channel')['Annual_Premium'].transform('mean')
# Annual Premium mean baseed on vehicle age
df['Age_Wise_Mean_Premium'] = df.groupby('Vehicle_Age')['Annual_Premium'].transform('mean')
df.head()
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm', axis = None).set_precision(2)
