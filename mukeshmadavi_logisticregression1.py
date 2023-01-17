# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



# data visualization

import matplotlib.pyplot as plt

import seaborn as sns



# machine learning

from sklearn.preprocessing import StandardScaler



import sklearn.linear_model as skl_lm

from sklearn import preprocessing

from sklearn import neighbors

from sklearn.metrics import confusion_matrix, classification_report, precision_score

from sklearn.model_selection import train_test_split





import statsmodels.api as sm

import statsmodels.formula.api as smf





# initialize some package settings

sns.set(style="whitegrid", color_codes=True, font_scale=1.3)



%matplotlib inline


df = pd.read_csv('../input/brain-tumor/bt_dataset_t3.csv', index_col=0)

df.head()
df.info()
df.dtypes
plt.figure(figsize=(8, 4))

sns.countplot(df['Target'], palette='RdBu')



# count number of obvs in each class

benign, malignant = df['Target'].value_counts()

print('Number of cells labeled zero: ', benign)

print('Number of cells labeled one : ', malignant)

print('')

print('% of cells labeled zero', round(benign / len(df) * 100, 2), '%')

print('% of cells labeled one', round(malignant / len(df) * 100, 2), '%')
cols = ['Target',

        'Mean', 

        'Variance', 

        'Standard Deviation', 

        'Entropy', 

        'Skewness', 

        'Kurtosis', 

        'Contrast',

        'Energy', 

        'ASM', 

        'Homogeneity',

        'Dissimilarity',

        'Correlation',

        'Coarseness',

        'PSNR',

        'SSIM',

        'MSE',

        'DC']



sns.pairplot(data=df[cols], hue='Target', palette='RdBu')
corr = df.corr().round(2)



# Mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set figure size

f, ax = plt.subplots(figsize=(20, 20))



# Define custom colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap

sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)



plt.tight_layout()
X = df

y = df['Target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
cols = df.columns.drop(['Standard Deviation','Target'])

#cols = df.columns.drop('Target')

formula = 'Target ~ ' + ' + '.join(cols)

print(formula, '\n')
model = smf.glm(formula=formula, data=X_train, family=sm.families.Binomial())

logistic_fit = model.fit()



print(logistic_fit.summary())
predictions = logistic_fit.predict(X_test)

predictions[1:6]
predictions_nominal = [ "1" if x == 1.0 else "0" for x in predictions]

predictions_nominal[1:6]
print(classification_report(y_test.astype(int).astype(str), predictions_nominal, digits=3))



cfm = confusion_matrix(y_test.astype(int).astype(str), predictions_nominal)



#conf = confusion_matrix(pred["y"].values.astype(int).astype(str), pred["PredictedLabel"].values)

#conf = pd.DataFrame(conf)



true_negative = cfm[0][0]

false_positive = cfm[0][1]

false_negative = cfm[1][0]

true_positive = cfm[1][1]



print('Confusion Matrix: \n', cfm, '\n')



print('True Negative:', true_negative)

print('False Positive:', false_positive)

print('False Negative:', false_negative)

print('True Positive:', true_positive)

print('Correct Predictions', 

      round((true_negative + true_positive) / len(predictions_nominal) * 100, 1), '%')