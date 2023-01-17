import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

from scipy import stats

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() # for plot styling
df = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')
df
# df.isnull().sum()

# df_dtype_nunique = pd.concat([df.dtypes, df.nunique()],axis=1)

# df_dtype_nunique.columns = ["dtype","unique"]   # Make sure you understand what we are checking here

# df_dtype_nunique
# df['Married'].value_counts()
# df.describe()
# X['TotalCharges'] = pd.to_numeric(X['TotalCharges'])

# X['TotalCharges'].to_numeric(df.my_var, errors='coerce').fillna(0).astype(int)

categorical_features = list(df.columns.values)



categorical_features.remove('SeniorCitizen')

categorical_features.remove('tenure')

categorical_features.remove('MonthlyCharges')

categorical_features.remove('Satisfied')

categorical_features.remove('custId')

categorical_features.remove('TotalCharges')

numerical_features = ['tenure', 'MonthlyCharges']



categorical_features
from sklearn.preprocessing import LabelEncoder

l = LabelEncoder()



for col in categorical_features:

    df[col]= l.fit_transform(df[col])

df.head()
print (df[pd.to_numeric(df.TotalCharges, errors='coerce').isnull()])

df = df.drop([544,1348,1553,2504,3083,4766])

# X = X.drop(1348)

# X.drop(1553)

# X.drop(2504)

# X.drop(3083)

# X.drop(4766)



df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])



df.dtypes
X = df[numerical_features+categorical_features]

y = df["Satisfied"]

X.head()
corr = df.corr()

corr_values=corr['Satisfied'].sort_values(ascending=False)

corr_values=abs(corr_values).sort_values(ascending=False)

print("Correlation of mentioned features wrt outcome in ascending order")

print(abs(corr_values).sort_values(ascending=False))
# # Compute the correlation matrix

# # corr = df.corr()



# # Generate a mask for the upper triangle

# mask = np.zeros_like(corr, dtype=np.bool)

# mask[np.triu_indices_from(mask)] = True



# # Set up the matplotlib figure

# f, ax = plt.subplots(figsize=(12, 9))



# # Generate a custom diverging colormap

# cmap = sns.diverging_palette(220, 10, as_cmap=True)



# # Draw the heatmap with the mask and correct aspect ratio

# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.5, center=0,

#             square=True, linewidths=.5, cbar_kws={"shrink": .5})



# plt.show()
from sklearn.preprocessing import scale

X = scale(X)

X
from sklearn.decomposition import PCA



covar_matrix = PCA(n_components = 15)
covar_matrix.fit(X)

new_X = covar_matrix.transform(X)

variance = covar_matrix.explained_variance_ratio_ #calculate variance ratios



var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)

var #cumulative sum of variance explained with [n] features
plt.ylabel('% Variance Explained')

plt.xlabel('# of Features')

plt.title('PCA Analysis')

plt.ylim(30,100.5)

plt.style.context('seaborn-whitegrid')





plt.plot(var)
new_X

PCA_components = pd.DataFrame(new_X)

PCA_components.head()
from sklearn.cluster import KMeans



ks = range(1, 10)

inertias = []

for k in ks:

    # Create a KMeans instance with k clusters: model

    model = KMeans(n_clusters=k)

    

    # Fit model to samples

    model.fit(PCA_components.iloc[:,:3])

    

    # Append the inertia to the list of inertias

    inertias.append(model.inertia_)

    

plt.plot(ks, inertias, '-o', color='black')

plt.xlabel('number of clusters, k')

plt.ylabel('inertia')

plt.xticks(ks)

plt.show()
kmeans = KMeans(n_clusters = 2)
X_clustered = kmeans.fit_predict(new_X)
Label_color_map = {0: 'r', 1: 'g'}

l_color = [Label_color_map[l] for l in X_clustered]



plt.figure(figsize = (10,10))

plt.scatter(new_X[:,0], new_X[:,2], c = l_color, alpha = 0.5)

plt.show()
for i in X_clustered:

    print(i)
test = pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')
test.head()
test.isnull().sum()

from sklearn.preprocessing import LabelEncoder

l = LabelEncoder()



for col in categorical_features:

    test[col]= l.fit_transform(test[col])

test.head()
X_test=test[numerical_features + categorical_features]

X_test.head()
X_test = scale(X_test)

X_test
y_pred = kmeans.fit_predict(X_test)
for i in y_pred:

    print(i)
var = pd.DataFrame({'custId': test['custId'], 'Satisfied': y_pred})

var.head()
var.to_csv("Submit2.csv", index = False)