import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import category_encoders as ce
filepath = '../input/80-cereals/cereal.csv'
cereal_data = pd.read_csv(filepath)
cereal_data.head()
cereal_data.isnull().sum()
object_cols = [col for col in cereal_data.columns if cereal_data[col].dtype == 'object']
object_nunique = list(map(lambda col: cereal_data[col].nunique(), object_cols))
list(zip(object_cols,object_nunique))
cereal_data.drop(['name'], axis = 1, inplace = True)
ce_one_hot = ce.OneHotEncoder(cols= ['mfr', 'type'])
OH_data = ce_one_hot.fit_transform(cereal_data)
OH_data
cereal_data
def remove_outliers(dataframe, features):
    for feature in features:
        Q1 = dataframe[feature].quantile(0.25)
        Q3 = dataframe[feature].quantile(0.75)
        IQR = Q3 - Q1
        # Filtering Values between Q1-1.5IQR and Q3+1.5IQR
        filtered = dataframe.query('(@Q1 - 1.5 * @IQR) <= '+ feature +' <= (@Q3 + 1.5 * @IQR)')
        dataframe = filtered
    return dataframe
num_cols = [col for col in cereal_data.columns if cereal_data[col].dtype != 'object']
num_cols
remove_outliers(OH_data, ['rating'])
import math
def plot_multiple_histograms(df, cols):
    num_plots = len(cols)
    num_cols = math.ceil(np.sqrt(num_plots))
    num_rows = math.ceil(num_plots/num_cols)
    
        
    fig, axs = plt.subplots(num_rows, num_cols,figsize=(20,10))
    fig.tight_layout(pad=3.0)
    
    
    for ind, col in enumerate(cols):
        i = math.floor(ind/num_cols)
        j = ind - i*num_cols
        try:    
            if num_rows == 1:
                if num_cols == 1:
                    sns.distplot(df[col], kde=True, ax=axs)
                else:
                    sns.distplot(df[col], kde=True, ax=axs[j])
            else:
                sns.distplot(df[col], kde=True, ax=axs[i, j])
        except:
            print('Expection is produced by column : ' + col)
plot_multiple_histograms(cereal_data,['calories','protein','fat','sodium','fiber','carbo','sugars','potass','shelf','rating'])
plt.figure(figsize=(20,20))
sns.heatmap(OH_data.corr(),annot=True)
sns.jointplot(x = 'protein', y = 'rating', data = OH_data)
sns.jointplot(x = 'fiber', y = 'rating', data = OH_data)
sns.jointplot(x = 'calories', y = 'rating', data = OH_data)
sns.jointplot(x = 'sugars', y = 'rating', data = OH_data)
y = OH_data.rating
OH_data.drop(['rating'], axis = 1, inplace=True)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(OH_data, y, random_state = 0)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_val)
score = mean_absolute_error(y_val, prediction)
print(score)