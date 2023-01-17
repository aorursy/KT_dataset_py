import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
b_data = pd.read_csv("../input/data-for-datavis/cancer_b.csv", index_col="Id")

m_data = pd.read_csv("../input/data-for-datavis/cancer_m.csv", index_col="Id")
sns.kdeplot(data=b_data['Radius (worst)'], shade=True, label="Benign")

sns.kdeplot(data=m_data['Radius (worst)'], shade=True, label="Malignant")
import pandas as pd

Breast_cancer_data = pd.read_csv("../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv")
Breast_cancer_data.isnull().sum()
Breast_cancer_data.isna().sum()
X = Breast_cancer_data.iloc[:, 1:6].values

Y = Breast_cancer_data.iloc[:, 5].values

#Encoding categorical data values

from sklearn.preprocessing import LabelEncoder

labelencoder_Y = LabelEncoder()

Y = labelencoder_Y.fit_transform(Y)
Y