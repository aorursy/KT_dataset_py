import pandas as pd
import numpy as np
path_to_data = '/kaggle/input/heart-disease-uci/heart.csv'
data = pd.read_csv(path_to_data)
data.head()
features = data.columns
categorical_data = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
numerical_data = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
data['sex'].replace(1, 'Male', inplace=True)
data['sex'].replace(0, 'Female', inplace=True)

data['cp'].replace(0, 'Typical Angina', inplace=True)
data['cp'].replace(1, 'Atypical Angina', inplace=True)
data['cp'].replace(2, 'Non-anginal Pain', inplace=True)
data['cp'].replace(3, 'Asymptomatic', inplace=True)

data['fbs'].replace(1, 'fasting blood sugar > 120 mg/dl', inplace=True)
data['fbs'].replace(0, 'fasting blood sugar <= 120 mg/dl', inplace=True)

data['restecg'].replace(0, 'Normal', inplace=True)
data['restecg'].replace(1, 'having ST-T wave abnormality', inplace=True)
data['restecg'].replace(2, 'left ventricular hypertrophy', inplace=True)

data['exang'].replace(0, 'No', inplace=True)
data['exang'].replace(1, 'Yes', inplace=True)

data['slope'].replace(0, 'Upsloping', inplace=True)
data['slope'].replace(1, 'Flat', inplace=True)
data['slope'].replace(2, 'Downsloping', inplace=True)

data['target'].replace(0, '< 50% diameter narrowing', inplace=True)
data['target'].replace(1, '> 50% diameter narrowing', inplace=True)
data.head()
data.describe().T
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
for num, feature in enumerate(categorical_data):
    plt.figure(num)
    plot = sns.countplot(x=feature, data=data, palette='Blues')
    plot.set_xticklabels(plot.get_xticklabels(), rotation=20)
# numerical data
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
ax = np.array(axes).reshape(-1)

for num, feature in enumerate(numerical_data):
    sns.distplot(data[feature], ax=ax[num])
for num, feature in enumerate(numerical_data):
    plt.figure(num)
    plot = sns.boxplot(x='target', y =feature, data=data)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=20)
for num, feature in enumerate(numerical_data):
    plt.figure(num)
    ax = sns.barplot(x="target", y=feature, hue="sex", data=data)
    ax.set_xticklabels(plot.get_xticklabels(), rotation=20)
sns.pairplot(data[numerical_data])
