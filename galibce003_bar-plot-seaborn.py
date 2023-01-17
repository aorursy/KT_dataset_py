import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df = sns.load_dataset('tips')
df.head()
sns.barplot(x = 'day', y = 'tip', data = df)
plt.show()
sns.barplot(x = 'day', y = 'tip', hue = 'sex', data = df)
plt.show()
sns.barplot(x = 'day', y = 'tip', order = ['Sun', 'Thur', 'Fri', 'Sat'], data = df)
plt.show() 
sns.barplot(x = 'day', y = 'total_bill', palette = 'spring', data = df)
plt.show()
sns.barplot(x = 'day', y = 'total_bill', palette = 'winter_r', ci= 50, data = df)
plt.show()
sns.barplot(x = 'day', y = 'total_bill', palette = 'husl', capsize = 0.2, data = df)
plt.show()
sns.barplot(x = 'total_bill', y = 'day', palette = 'autumn', capsize = 0.1, data = df)
plt.show()