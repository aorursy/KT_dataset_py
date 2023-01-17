import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
# Now loading in data with Pandas
df = pd.read_csv('../input/Admission_Predict.csv')

print('dataframe shape: {}'.format(df.shape))
df.head(3)
# Now getting a better look at what each column represents
df.info()
# Dropping 'Serial No.'
df = df.drop(columns=['Serial No.'])
# Standardizing column names
df.columns = ['GRE_score', 'TOEFL_score', 'university_rating', 'statement_of_purpose', 'letter_of_recommendation', 'GPA', 'research', 'chance_of_admit']
df.head(3)
df['chance_of_admit'].hist(figsize=(10,4))
df['chance_of_admit'].plot(kind='density', subplots=True, figsize=(10, 4));
# Creating a correlation matrix:
corr_matrix = df.corr()
# Plotting heatmap
sns.heatmap(corr_matrix);
sns.jointplot(x='chance_of_admit', y='GPA', data=df, kind='scatter');
sns.lmplot('chance_of_admit', 'GPA', data=df, hue='research', fit_reg=False);
# taking care of our ML imports
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# Seperating our features and our target.
train_features = list(set(df.columns) - set(['chance_of_admit']))

train_X = df[train_features]
train_y = df['chance_of_admit']
# Create LinearRegression instance
linear_regression = LinearRegression()

# Begin training! This is also knows as 'fitting' the model to our data
linear_regression.fit(train_X, train_y)
pd.DataFrame(linear_regression.coef_, train_X.columns, columns=['Coefficient'])
# Loading in Test set
test_df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')

test_df.head(3)
test_df = test_df.drop(columns=['Serial No.'])
test_df.columns = ['GRE_score', 'TOEFL_score', 'university_rating', 'statement_of_purpose', 'letter_of_recommendation', 'GPA', 'research', 'chance_of_admit']
test_df.head(3)
test_X = test_df[train_features]
test_y = test_df['chance_of_admit']
y_pred = linear_regression.predict(test_X)
pd.DataFrame({'Prediction': y_pred, 'Actual': test_y}).head(10)
print("Mean Absolute Error: {}".format(metrics.mean_absolute_error(test_y, y_pred)))