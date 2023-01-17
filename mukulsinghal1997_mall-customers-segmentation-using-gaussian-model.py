import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
df.drop(['CustomerID', 'Gender'], axis = 1, inplace = True)
df.isnull().sum()
from sklearn.preprocessing import StandardScaler
se = StandardScaler().fit_transform(df)
from sklearn.mixture import GaussianMixture as gmm
sns.scatterplot('Age','Annual Income (k$)', data = df)
sns.scatterplot('Age','Spending Score (1-100)', data = df)
sns.scatterplot('Annual Income (k$)','Spending Score (1-100)', data = df)
n_comp = np.arange(1,10)
cova_type = ['full', 'spherical', 'diag']
bic = []
aic = []
from itertools import product
gmm_params = list(product(n_comp, cova_type))
for i in gmm_params:
    GMM = gmm(n_components = i[0], covariance_type = i[1], random_state = 0).fit(se)
    bic.append(GMM.bic(se))
    aic.append(GMM.aic(se))
gmm_df = pd.DataFrame(gmm_params, columns = ['n_components', 'covariance'])
gmm_df.loc[:, 'bic'] = bic
gmm_df.loc[:, 'aic'] = aic
gmm_df.head(5)
sns.lineplot('n_components', 'bic', data = gmm_df, hue = 'covariance')
sns.lineplot('n_components', 'aic', data = gmm_df, hue = 'covariance')
GMM_Final = gmm(n_components = 5, covariance_type = 'diag', random_state = 0).fit(se)
labels = GMM_Final.predict(se)
labels
col = df.columns
GMM_data = pd.DataFrame(se, columns = col)
GMM_data.loc[:, 'label'] = labels
plt.figure(figsize=(10,8))
sns.scatterplot('Age', 'Annual Income (k$)', data = GMM_data, hue = 'label', legend = 'full', palette = 'Set1')
plt.figure(figsize=(10,8))
sns.scatterplot('Age', 'Spending Score (1-100)', data = GMM_data, hue = 'label', legend = 'full', palette = 'Set1')
plt.figure(figsize=(10,8))
sns.scatterplot('Annual Income (k$)', 'Spending Score (1-100)', data = GMM_data, hue = 'label', legend = 'full', palette = 'Set1')
GMM_data['label'].value_counts().to_frame()
sns.countplot(GMM_data['label'], data = GMM_data)