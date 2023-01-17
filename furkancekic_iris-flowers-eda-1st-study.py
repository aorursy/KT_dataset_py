import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/iris/Iris.csv')
df.head()
df.describe()
df.info()
df.Species.head()
cat_df = df.select_dtypes(include = ['object'])
print(cat_df.Species.unique())
print(cat_df.Species.value_counts())
print('\n')
print(cat_df.Species.value_counts().count())
cat_df.Species.value_counts().plot.barh();
value = df.Species.value_counts()
plt.bar(value.index, value)
plt.xlabel('Species')
plt.ylabel('Frequency')
plt.title('Bar Plot of Species')
plt.show()
##SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm

all_features = pd.concat([df.SepalLengthCm,df.SepalWidthCm, df.PetalLengthCm, df.PetalWidthCm])
count = all_features.value_counts()
plt.scatter(count.index, count.values)
plt.xticks(count.index, count.index.values)
plt.show()
df.Species.unique()
setosa = df[df.Species == 'Iris-setosa']
versicolor = df[df.Species == 'Iris-versicolor']
virginica = df[df.Species == 'Iris-virginica']
plt.figure()
plt.scatter(setosa.SepalLengthCm, setosa.SepalWidthCm)
plt.scatter(versicolor.SepalLengthCm, versicolor.SepalWidthCm)
plt.scatter(virginica.SepalLengthCm, virginica.SepalWidthCm)
plt.show()
plt.scatter(setosa.SepalLengthCm, setosa.PetalLengthCm)
plt.scatter(versicolor.SepalLengthCm, versicolor.PetalLengthCm)
plt.scatter(virginica.SepalLengthCm, virginica.PetalLengthCm)
plt.show()
plt.scatter(setosa.SepalLengthCm, setosa.PetalWidthCm, label = 'Setosa')
plt.scatter(versicolor.SepalLengthCm, versicolor.PetalWidthCm, label = 'Versicolor')
plt.scatter(virginica.SepalLengthCm, virginica.PetalWidthCm, label = 'Virginica')
plt.legend()
plt.show()
df.groupby(['Species']).mean()
column = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for i in columns:
    plt.figure()
    df.boxplot(column = i);
def outlier_detection(df, features):
   
    outlier_indices = []
    
    
    for i in features:
        
        # 1st Quartile
        Q1 = np.percentile(df[i], 25)
        
        # 3rd Quartile
        Q3 = np.percentile(df[i], 75)
        
        # IQR
        IQR = Q3 - Q1
        
        # Outlier Step
        outlier_step = IQR * 1.5
        
        # Detect outlier and store their indeces
        outlier_list_col = df[(df[i] < Q1- outlier_step) | (df[i] > Q3 + outlier_step)].index
        
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(j for j, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
df.loc[outlier_detection(df, features)]
df.head()
df.columns.isnull().any()