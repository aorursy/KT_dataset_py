import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
%matplotlib inline
raw_data = pd.read_csv('../input/epi_r.csv')
raw_data.rating.describe()
# visualize distribution of reviews
raw_data.rating.hist(bins=20)
plt.title('Histogram of Recipe Ratings')
plt.show()
# Find columns with null values 
null_count = raw_data.isnull().sum()
null_count[null_count>0]
# creating a new dataframe without the missing values
# dropping title because object type - won't work in model

df = raw_data.drop(['rating', 'title', 'calories', 'protein', 'fat', 'sodium'], 1)
# turning this into a classification problem
# split reviews based on mean of rating
# finding mean of ratings
mean_rating = raw_data['rating'].mean()

#creating new column where ratings are 1 for good and 0 for bad
df['target'] = np.where(raw_data['rating']>=mean_rating, 1, 0)

#printing mean value
mean_rating
# checking size of each option
# many more good than bad reviews
df.target.value_counts()
#checking correlation of columns to the target
corr_matrix = df.corr()

#dropping highly correlated features
#code from: https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.75
to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
print('Columns to drop: ' , (len(to_drop)))

# Drop features 
df = df.drop(columns=to_drop)
print('train_features_df shape: ', df.shape)
# get positively correlated features
pos_corr = corr_matrix['target'].sort_values(ascending=False).head(20)

#converting to dataframe
pos_corr = pd.DataFrame(pos_corr)
pos_corr
# get list of column names
top_pos = [column for column in pos_corr.columns]

# create new column with sum across rows if in list
df['top_pos_tags'] = df[top_pos].sum(axis=1)
# get negatively correlated features
# interesting ...alcoholic recipes don't seem to get good reviews!
neg_corr = corr_matrix['target'].sort_values(ascending=True).head(20)

# convert to dataframe
neg_corr = pd.DataFrame(neg_corr)
neg_corr
# get list of column names
top_neg = [column for column in neg_corr.columns]

# create new column with sum across rows if in list
df['top_neg_tags'] = df[top_neg].sum(axis=1)
# create new column with sum of tags for each recipe
df['sum_tags'] = df.sum(axis=1)
df.head()
# it looks like our positive reviews have more tags than negative reviews
sns.boxplot(x=df.target, y=df.sum_tags);
# trying the model
from sklearn.svm import SVR
svr = SVR()
X = df[['top_pos_tags', 'top_neg_tags', 'sum_tags']]
y = df['target']
svr.fit(X,y)

y_ = svr.predict(X)

svr.score(X, y)
from sklearn.model_selection import cross_val_score
cross_val_score(svr, X, y, cv=5)