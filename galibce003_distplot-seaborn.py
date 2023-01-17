import matplotlib.pyplot as plt
import seaborn as sns
df = sns.load_dataset('tips')
df.head()
sns.distplot(df.total_bill)
plt.show()
sns.distplot(df.total_bill, kde = False)
plt.show()
sns.distplot(df.total_bill, kde = False, rug = True)
plt.show()
sns.distplot(df.total_bill, kde = False, rug = True, norm_hist = True)
plt.show()
from scipy import stats
sns.distplot(df.total_bill, kde = False, fit = stats.gamma)
plt.show()
sns.distplot(df.total_bill, vertical = True)
plt.show()