import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab as plt
plt.style.use('ggplot')
import seaborn as sns
from sklearn.decomposition import PCA

# Input
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load the Data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Take a look at our first feature '48df886f9'
train['48df886f9'].hist(bins=50)
print('Feature 48df886f9')
print('The max value is {}'.format(train['48df886f9'].max()))
print('The min value is {}'.format(train['48df886f9'].min()))
print('The mean value is {}'.format(train['48df886f9'].mean()))
print('The standard deviation is'.format(train['48df886f9'].std()))
train_desc = train.describe().transpose()
# As you can see we have mean, stddev, min and max values
train_desc.head()
train_desc.sort_values('std', ascending=False).head()
train_desc['mean'].hist(bins=50, color='r', figsize=(10,5))
plt.title('Distribution of the Mean Value for Features')
train_desc['max'].hist(bins=50, color='b', figsize=(10,5))
plt.title('Distribution of the Max Value for Features')
train_desc['min'].hist(bins=50, color='g', figsize=(10,5))
plt.title('Distribution of the Min Value for Features')
train_desc['std'].hist(bins=50, color='k', figsize=(10,5))
plt.title('Distribution of the Standard Deviation of each Feature')
# Create a list of all the features
features = train.columns.tolist()[2:]
len(features) # That's a lot of features
# Lets just try the first 10 features
for feature in features[:10]:
    train.plot(x='target', y=feature, figsize=(10,1), kind='scatter', title=feature)
    plt.axis('off')
pca = PCA(n_components=5) # Selected 5 components just for fun
pca.fit(train[features].as_matrix())

print('The explained variance ratio is {}'.format(pca.explained_variance_ratio_))

pca_components = pd.DataFrame(pca.transform(train[features].as_matrix())) # Create a dataframe with PCA values
pca_components['target'] = train['target'] # Add the target back

# Pairplot of the PCA values vs Target
sns.pairplot(pca_components)