import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
marathon_data = pd.read_csv('../input/MarathonData.csv', encoding='utf-8')
marathon_data.head(3)
marathon_data.describe()
marathon_data['CrossTraining'] = marathon_data['CrossTraining'].fillna('Nothing')
marathon_data['Wall21'] = marathon_data['Wall21'].replace(' -   ', 0)
marathon_data['Wall21'] = marathon_data['Wall21'].apply(pd.to_numeric, errors='ignore')
marathon_data.describe()
wall_max = np.max(marathon_data['Wall21'])
marathon_data['Wall21'] = marathon_data['Wall21'].apply(lambda x: wall_max - x)
sns.countplot(marathon_data['CATEGORY'])
sns.countplot(marathon_data['Category'])
plt.hist(marathon_data['MarathonTime'])
plt.plot(marathon_data['MarathonTime'])
sns.boxplot(x=marathon_data['sp4week'])
sns.boxplot(x=np.log(marathon_data['sp4week']))
marathon_data['sp4week'].skew()
marathon_data['km4week'].skew()
marathon_data['Wall21'].skew()
plt.plot(marathon_data['sp4week'])
plt.plot(np.log(marathon_data['sp4week']))
np.max(marathon_data['sp4week'])
marathon_data[marathon_data['sp4week'] == 11125]
marathon_data['sp4week'].describe()
marathon_data = marathon_data[marathon_data['sp4week'] != 11125]
marathon_data['sp4week'] = np.log(marathon_data['sp4week'])
df_mA = marathon_data[marathon_data['CATEGORY'] == 'A']

df_mB = marathon_data[marathon_data['CATEGORY'] == 'B']

df_mC = marathon_data[marathon_data['CATEGORY'] == 'C']

df_mD = marathon_data[marathon_data['CATEGORY'] == 'D']
sns.countplot(df_mA['Category'])
sns.countplot(df_mB['Category'])
sns.countplot(df_mC['Category'])
sns.countplot(df_mD['Category'])
pd.concat([df_mA['km4week'], df_mB['km4week'], df_mC['km4week'], df_mD['km4week']], axis=1).boxplot()
pd.concat([df_mA['sp4week'], df_mB['sp4week'], df_mC['sp4week'], df_mD['sp4week']], axis=1).boxplot()
m_df = marathon_data.drop(['CATEGORY', 'Category', 'CrossTraining', 'Marathon', 'Name', 'id'], axis=1)
m_df.head(2)
x_train= m_df.drop(['MarathonTime'],axis=1)

y_train= m_df['MarathonTime']
from sklearn.cross_validation import train_test_split
data_train, data_test, label_train, label_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
slr = LinearRegression()

slr.fit(data_train, label_train)
slr.score(data_test, label_test)
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
marathon_data = pd.read_csv('../input/MarathonData.csv', encoding='utf-8')
marathon_data.head(3)
marathon_data.describe()
marathon_data['CrossTraining'] = marathon_data['CrossTraining'].fillna('Nothing')
marathon_data['Wall21'] = marathon_data['Wall21'].replace(' -   ', 0)
marathon_data['Wall21'] = marathon_data['Wall21'].apply(pd.to_numeric, errors='ignore')
marathon_data.describe()
wall_max = np.max(marathon_data['Wall21'])
marathon_data['Wall21'] = marathon_data['Wall21'].apply(lambda x: wall_max - x)
sns.countplot(marathon_data['CATEGORY'])
sns.countplot(marathon_data['Category'])
plt.hist(marathon_data['MarathonTime'])
plt.plot(marathon_data['MarathonTime'])
sns.boxplot(x=marathon_data['sp4week'])
sns.boxplot(x=np.log(marathon_data['sp4week']))
marathon_data['sp4week'].skew()
marathon_data['km4week'].skew()
marathon_data['Wall21'].skew()
plt.plot(marathon_data['sp4week'])
plt.plot(np.log(marathon_data['sp4week']))
np.max(marathon_data['sp4week'])
marathon_data[marathon_data['sp4week'] == 11125]
marathon_data['sp4week'].describe()
marathon_data = marathon_data[marathon_data['sp4week'] != 11125]
marathon_data['sp4week'] = np.log(marathon_data['sp4week'])
df_mA = marathon_data[marathon_data['CATEGORY'] == 'A']

df_mB = marathon_data[marathon_data['CATEGORY'] == 'B']

df_mC = marathon_data[marathon_data['CATEGORY'] == 'C']

df_mD = marathon_data[marathon_data['CATEGORY'] == 'D']
sns.countplot(df_mA['Category'])
sns.countplot(df_mB['Category'])
sns.countplot(df_mC['Category'])
sns.countplot(df_mD['Category'])
pd.concat([df_mA['km4week'], df_mB['km4week'], df_mC['km4week'], df_mD['km4week']], axis=1).boxplot()
pd.concat([df_mA['sp4week'], df_mB['sp4week'], df_mC['sp4week'], df_mD['sp4week']], axis=1).boxplot()
m_df = marathon_data.drop(['CATEGORY', 'Category', 'CrossTraining', 'Marathon', 'Name', 'id'], axis=1)
m_df.head(2)
x_train= m_df.drop(['MarathonTime'],axis=1)

y_train= m_df['MarathonTime']
from sklearn.cross_validation import train_test_split
data_train, data_test, label_train, label_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
slr = LinearRegression()

slr.fit(data_train, label_train)
slr.score(data_test, label_test)