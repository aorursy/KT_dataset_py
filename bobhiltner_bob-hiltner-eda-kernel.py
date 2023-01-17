import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import gc
import os
# Comment this if the data visualisations doesn't work on your side
%matplotlib inline

plt.style.use('bmh')

FILE_DIR = "../input/hawaiiml-data"
print([f for f in os.listdir(path=FILE_DIR)])

for f in os.listdir(FILE_DIR):
    print('{0:<30}{1:0.2f}MB'.format(f, 1e-6*os.path.getsize(f'{FILE_DIR}/{f}')))
df = pd.read_csv(f'{FILE_DIR}/train.csv', encoding='ISO-8859-1')
df_test = pd.read_csv(f'{FILE_DIR}/test.csv', encoding='ISO-8859-1')
submission = pd.read_csv(f'{FILE_DIR}/sample_submission.csv', encoding='ISO-8859-1')
print(df.shape)
df.head()

df.info()
print("id count: " + str(df['id'].count()))
#any duplicated id values?
print("duplicated id count: " + str(df[df.duplicated(['id'], keep=False)]['id'].count()))

print("Of", df['id'].count(), "id values,", df[df.duplicated(['id'], keep=False)]['id'].count(), "were duplicates.")
print("Of", df['invoice_id'].count(), "invoice_id values,", df[df.duplicated(['invoice_id'], keep=False)]['invoice_id'].count(), "were duplicates.")
df[df.duplicated(['invoice_id'], keep=False)].head()

# Note: this is equivalent of SQL: select * from df_train 
# where invoice_id in 
#  (select invoice_id from df_train 
#   group by invoice_id having count(*) > 1)
df.loc[df['invoice_id']==6757].head()
#5 of ~163 rows for this invoice.
# Let's add count of distinct items (itemcount) for each invoice to the df for further inquiries.
df['itemcount'] = df['invoice_id'].map(df['invoice_id'].value_counts())
#df_test['itemcount'] = df_test['invoice_id'].map(df_test['invoice_id'].value_counts())
#NOTE: Remember to do this (and any) transform with test during feature engineering.
print('Of', df['invoice_id'].nunique(), "training invoice_id values,", df[df.duplicated(['invoice_id'], keep=False)]['invoice_id'].nunique(), "were for multiple items.")
print(df['quantity'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df['quantity'], color='g', bins=100, hist_kws={'alpha': 0.4});
df[(df['quantity'] >= 4000)]
#describe without big outlier.  Leave others for now. Look at log(qty) as well, as this might 
#reduce the outlier distortion.
#sns.distplot(df[(df['quantity'] <= 5000)'quantity'], color='g', bins=100, hist_kws={'alpha': 0.4});
df[(df['quantity'] < 4000)].describe()

plt.figure(figsize=(9, 8))
sns.distplot(np.log1p(df['quantity'])[ (df['quantity'] < 4000) ], color='g', bins=100, hist_kws={'alpha': 0.4});
plt.title("Distribution of log1p(quantity)-excluding outliers ", fontsize=15)
plt.show()

plt.figure(figsize=(9, 8))
sns.distplot(np.log1p(df['quantity']), color='g', bins=100, hist_kws={'alpha': 0.4});
plt.title("Distribution of log1p(quantity), including outliers", fontsize=15)
plt.show()
# Let's add log1p(quantity) to the df for further inquiries.
df['quantity_log1p'] = df['quantity'].apply(np.log1p)

#NOTE: In this case, no column for df_test because it does not contain 'quantity' field.

df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
df["wday"] = df['date'].dt.weekday
df["day"] = df['date'].dt.day
plt.figure(figsize=(12,8))
ax = sns.boxplot(x="wday", y="quantity_log1p", data=df)
plt.ylabel('quantity_log1p', fontsize=12)
plt.xlabel('Day of week', fontsize=12)
plt.title("Boxplot of log1p(quantity) by day of week", fontsize=15)
plt.show()
plt.figure(figsize=(12,8))
ax = sns.boxplot(x="day", y="quantity_log1p", data=df)
plt.ylabel('quantity_log1p', fontsize=12)
plt.xlabel('Day of month', fontsize=12)
plt.title("Boxplot of log1p(quantity) by day of month", fontsize=15)
plt.show()
df['hour'] = pd.to_datetime(df['time'], format='%H:%M').dt.hour
plt.figure(figsize=(12,8))
ax = sns.boxplot(x="hour", y="quantity_log1p", data=df)
plt.ylabel('quantity_log1p', fontsize=12)
plt.xlabel('Hour of Day', fontsize=12)
#plt.xticks(rotation='vertical')
plt.title("Boxplot of log1p(quantity) by Hour", fontsize=15)
plt.show()
