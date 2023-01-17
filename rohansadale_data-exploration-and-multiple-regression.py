# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import matplotlib
from pandas.tools.plotting import scatter_matrix
%matplotlib inline
matplotlib.style.use('ggplot')

import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('../input/timesData.csv')
def parseFloat(x):
    try:
        x = float(x)
    except:
        x = 0
    return x

def parseStringCommas(x):
    try:
        x = int(x.replace(',',''))
    except:
        x = 0
    return x

def parseStringModulus(x):
    try:
        x = int(x.replace('%',''))
    except:
        x = 0
    return x

def convertToInt(x):
    try:
        x = int(x)
    except:
        x = 0
    return x

data['female'] = data['female_male_ratio'].str.split(':', expand=True)[0].apply(convertToInt)
data['male'] = data['female_male_ratio'].str.split(':', expand=True)[1].apply(convertToInt)
data['sex_ratio'] =  np.where(data['male'] == 0, 0, data['male']/data['female'])
    
columnstoFloat = ['world_rank', 'teaching', 'international', 'research', 'citations', 'income', 'total_score', 'student_staff_ratio']    

for column in columnstoFloat:
    data[column] = data[column].apply(parseFloat)
    
data['num_students'] = data['num_students'].apply(parseStringCommas)
data['year'] = data['year'].apply(lambda x : int(x))
data['international_students'] = data['international_students'].apply(parseStringModulus)
cleanData = data[(data.world_rank != 0) & (data.year == 2015)][['world_rank', 'teaching', 'international', 'research', 'citations', 'income', 'total_score', 'sex_ratio']]
# Plot for how research varies with world rank
cleanData.plot(y='world_rank', x = 'research', kind='scatter', xlim=(0,110))
cleanData.plot(y='world_rank', x = 'sex_ratio', kind='scatter')
scatter_matrix(cleanData, alpha=0.5, figsize=(15,15), diagonal='kde')
# Multiple Regression
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split


train, test = train_test_split(cleanData, test_size = 0.3)
#sample = cleanData.sample(frac=0.7)
#features = sample[['teaching', 'international', 'research', 'citations', 'income', 'male', 'female']]
features = train[['teaching','international', 'research', 'citations', 'sex_ratio']]
y = train['total_score']
lm = LinearRegression()
lm.fit(features, y)

print( '\n Coefficients')
print (lm.coef_)

print( '\n Intercept')
print (lm.intercept_)

print( '\n RSquared')
print (lm.score(features, y))
output = pd.DataFrame(lm.predict(test[['teaching','international', 'research', 'citations', 'sex_ratio']]), columns =['Prediction'])
output['Actual'] = test['total_score'].reset_index(drop=True)
print(output.head())