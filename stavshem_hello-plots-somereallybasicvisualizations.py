import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# suppress scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(suppress=True)
data_df = pd.read_csv('../input/credit_card_clients_split.csv')
data_df.sample(n=3)
data_df.describe()
data_df.avg_monthly_turnover.describe()
print('Data shape: %s' % (data_df.shape,))
education_title_to_numeric = {
    'US': 0, # unfinished general
    'SS': 1, # special general
    'S':  2, # general
    'UH': 3, # unfinished higher
    'H':  4, # higher
    'HH': 5, # several higher
    'A':  6  # PhD
}
data_df['education_numeric'] = data_df.education.apply(lambda e: education_title_to_numeric[e])
train_df = data_df[data_df.avg_monthly_turnover.notnull()]
test_df = data_df[data_df.avg_monthly_turnover.isnull()]
print('train size: %d, test size: %d' % (train_df.shape[0], test_df.shape[0]))
percentages = [0, 25, 50, 75, 85, 98, 100]
percentiles = np.percentile(train_df.avg_monthly_turnover.dropna(), percentages)
percentile_list = pd.DataFrame(
    {'percentages': percentages,
     'avg_monthly_turnover value': percentiles})
percentile_list
from pylab import rcParams
rcParams['figure.figsize'] = 8, 8
plt.style.use('fivethirtyeight')
plt.plot(percentages, percentiles)
plt.xlabel('Percentage')
plt.ylabel('Avg Monthly Turnover')
plt.title('Percentile to Avg Monthly Turnover');
outliers_threshold_percentage = .98
amt = train_df.avg_monthly_turnover
amt_qcut = train_df[train_df.avg_monthly_turnover < amt.quantile(outliers_threshold_percentage)]['avg_monthly_turnover']
plt.hist(amt_qcut, 100, facecolor='blue', alpha=0.5)
plt.xlabel('avg monthly turnover')
plt.title('%s%% avg monthly turnover histogram' % outliers_threshold_percentage);
import seaborn as sns
sns.countplot(data_df.education).set_title("Education");
partial_train_df = train_df[train_df.avg_monthly_turnover < amt.quantile(.95)]

(sns.FacetGrid(partial_train_df[(partial_train_df['education']\
                        .isin(partial_train_df['education']\
                              .value_counts()[:4].index.values))],
               hue='education', aspect=2)
  .map(sns.kdeplot, 'avg_monthly_turnover', shade=True)
 .add_legend()
)
plt.title('Distribuition of Avg Monthly Turnover by Education')
plt.show()
sns.kdeplot(train_df.age, label='Age train')
sns.kdeplot(test_df.age, label='Age test')

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages Across Train and Test');
meaningful_features =  ['income', 'age', 'education_numeric']
meaningful_columns = meaningful_features + ['avg_monthly_turnover']
correlations = train_df[meaningful_columns].corr()
correlations_to_target = correlations['avg_monthly_turnover'].sort_values()
sns.heatmap(correlations, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');

