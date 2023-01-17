import pandas            as pd;
import numpy             as np;
import seaborn           as sns;
import matplotlib.pyplot as plt;

%matplotlib inline
df = pd.read_csv('../input/PURCHASE ORDER DATA EXTRACT 2012-2015_0.csv', parse_dates=["Creation Date"])
df.head()
df.info()
df.describe()
missing_df = df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')

ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()
df['Creation Day'] = df['Creation Date'].dt.day

cnt_srs = df['Creation Day'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='blue')
#plt.xticks(rotation='vertical')
plt.xlabel('Day of Creation', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()
df['Creation Month'] = df['Creation Date'].dt.month

cnt_srs = df['Creation Month'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='blue')
#plt.xticks(rotation='vertical')
plt.xlabel('Month of Creation', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()
df['Creation Year'] = df['Creation Date'].dt.year

cnt_srs = df['Creation Year'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='blue')
#plt.xticks(rotation='vertical')
plt.xlabel('Year of Creation', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()
# there are some erroneous values in the year like 2510 and 1014, so we'll clean that up

temp_df = pd.DataFrame()

temp_df['Purchase Date'] = pd.to_datetime([date[:-4]+'20'+date[-2:] for date in df['Purchase Date'].dropna().values])
temp_df['Purchase Day'] = temp_df['Purchase Date'].dt.day

cnt_srs = temp_df['Purchase Day'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='blue')
#plt.xticks(rotation='vertical')
plt.xlabel('Day of Purchase', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()
temp_df['Purchase Month'] = temp_df['Purchase Date'].dt.month

cnt_srs = temp_df['Purchase Month'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='blue')
#plt.xticks(rotation='vertical')
plt.xlabel('Month of Purchase', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()
temp_df['Purchase Year'] = temp_df['Purchase Date'].dt.year

cnt_srs = temp_df['Purchase Year'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='blue')
plt.xticks(rotation='vertical')
plt.xlabel('Year of Purchase', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()
cnt_srs = df['Acquisition Type'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='blue')
#plt.xticks(rotation='vertical')
plt.xlabel('Acquisition Type', fontsize=12)
plt.ylabel('Number of Purchases', fontsize=12)
plt.show()
cnt_srs = df['Sub-Acquisition Type'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='blue')
plt.xticks(rotation='vertical')
plt.xlabel('Acquisition Type', fontsize=12)
plt.ylabel('Number of Purchases', fontsize=12)
plt.show()
cnt_srs = df['Acquisition Method'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='blue')
plt.xticks(rotation='vertical')
plt.xlabel('Acquisition Method', fontsize=12)
plt.ylabel('Number of Purchases', fontsize=12)
plt.show()
cnt_srs = df['Sub-Acquisition Method'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='blue')
plt.xticks(rotation='vertical')
plt.xlabel('Sub-Acquisition Method', fontsize=12)
plt.ylabel('Number of Purchases', fontsize=12)
plt.show()
purchase_prices = [float(x[1:]) for x in df['Total Price'].dropna()]

print('Minimum Purchase Price = ' + str(min(purchase_prices)))
print('Maximum Purchase Price = ' + str(max(purchase_prices)))
print('Average Purchase Price = ' + str(np.mean(purchase_prices)))
# let's plot without these negative and extremely large positive numbers, which don't sound correct

plt.figure(figsize=(12,6))
sns.distplot([x for x in purchase_prices if x > 0 and x < 8e4], kde=False)
plt.xticks(rotation='vertical')
plt.xlabel('Supplier Name', fontsize=12)
plt.ylabel('Number of Purchases', fontsize=12)
plt.show()
