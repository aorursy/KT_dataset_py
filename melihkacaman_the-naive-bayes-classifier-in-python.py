import pandas as pd

import numpy as np 

import seaborn as sns 

import matplotlib.pyplot as plt



import category_encoders as ce



from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split



import warnings



warnings.filterwarnings('ignore')



%matplotlib inline
data = pd.read_csv('../input/adult-dataset/adult.csv')
data.head() 
data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'never_married', 'marital_status', 'occupation', 'relationship',

             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']



data.tail() 
data.income = pd.get_dummies(data.income)[' >50K']
data.tail()
data.info() 
categorical_names = []

for feature in data.columns: 

    if data[feature].dtype == object: 

        categorical_names.append(feature)

categorical_names
data[categorical_names].head() 
data[categorical_names].isnull().any()
data[categorical_names].isna().any()
for feature in data[categorical_names].columns:

    print('FEATURE NAME:', feature)

    print(data[feature].value_counts())

    
for feature in data.columns:

    data[feature].replace(' ?', np.nan, inplace=True)
# check this, 



data[data.occupation == ' ?']
data.native_country.value_counts()
data[categorical_names].isnull().any() 
plt.figure(figsize=(10,6))

sns.heatmap(data[categorical_names].isnull(), yticklabels=False, cbar=False, cmap='viridis')

plt.show() 
numerical_features = [var for var in data.columns if data[var].dtype!='O']



data[numerical_features].head() 
data[numerical_features].isnull().any()
data[categorical_names].isnull().mean()
# The mode of a set of values is the value that appears most often. It can be multiple values.

data.workclass.mode()
data.workclass.value_counts()
na_colls = data.isnull().any().loc[data.isnull().any().values == True].index

na_colls
for i in na_colls:

    data[i].fillna(data[i].mode()[0], inplace=True)



data.isnull().any() 
plt.figure(figsize=(10,6))

sns.heatmap(data[categorical_names].isnull(), yticklabels=False, cbar=False, cmap='viridis')

plt.show() 
data[categorical_names].head() 
for i in categorical_names: 

    print(str.upper(i), data[i].value_counts().shape[0]) 
categorical_names_withoutone = categorical_names

categorical_names_withoutone.remove('native_country')

encoder = ce.OneHotEncoder(cols=categorical_names_withoutone)



data_encoded = encoder.fit_transform(data)



data_encoded.head()
print('native_country has got', data.native_country.value_counts().shape[0], 'features.')
mean_encoded_nativeCont = data_encoded.groupby(['native_country'])['income'].mean().to_dict() 

data_encoded.native_country = data_encoded.native_country.map(mean_encoded_nativeCont)
data_encoded.native_country
target = data_encoded.income 

features = data_encoded.drop('income', axis=1) 



X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33)



gnb = GaussianNB() 

gnb.fit(X_train, y_train)
prediction = gnb.predict(X_test)



prediction
correct = (y_test == prediction).sum() 

print('classified correctly', correct) 

wrong = X_test.shape[0] - correct 

print('classified incorretly', wrong)
print('The Accuracy is', correct / X_test.shape[0])
prediction_train = gnb.predict(X_train)

correct_train = (y_train == prediction_train).sum()

print('classified correctly in train set', correct_train) 

wrong_train = X_train.shape[0] - correct_train

print('classified incorrectly in train set', wrong_train)
print('The accuracy for train set is', correct_train / X_train.shape[0])
# chart styling info 



yaxis_label = '>50K'

xaxis_label = '<=50K'
log_probabilities = gnb.predict_proba(X_test)

prob0 = log_probabilities[:,0]

prob1 = log_probabilities[:,1]



summary_df = pd.DataFrame({yaxis_label: prob0, xaxis_label: prob1, 'labels':y_test})

summary_df
sns.lmplot(x=xaxis_label, y=yaxis_label, data=summary_df, height=6.5, fit_reg=False, legend=False,

          scatter_kws={'alpha': 0.5, 's': 25}, hue='labels', markers=['o', 'x'], palette='hls')







plt.show()