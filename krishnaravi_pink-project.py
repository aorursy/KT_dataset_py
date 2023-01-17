# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/happiness_2017.csv')
print('There are' , sum(df.isna().sum()), 'missing values in this dataset')
vis_columns = ['Happiness.Rank', 'Happiness.Score', 'Economy..GDP.per.Capita.', 'Family',
       'Health..Life.Expectancy.', 'Freedom', 'Generosity',
       'Trust..Government.Corruption.', 'Dystopia.Residual']
vis = df[vis_columns]
corr = vis.corr()
sns.heatmap(corr)
print('Heatmap of Feature Correlations')
plt.scatter(vis['Family'], vis['Happiness.Score'])
plt.xlabel('Level of Family Support')
plt.ylabel('Level of Happiness')
plt.title('Family vs Happiness Score')
plt.show()
plt.scatter(vis['Freedom'], vis['Happiness.Score'])
plt.xlabel('Level of Freedom')
plt.ylabel('Level of Happiness')
plt.title('Freedom vs Happiness Score')
plt.show()
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
def rmse_score(model, X, y):
    return np.sqrt(np.mean((y - model.predict(X)) ** 2))
## creating a train/test split
train, test = train_test_split(vis, test_size = 0.20)
x_cols = ['Economy..GDP.per.Capita.', 'Family',
       'Health..Life.Expectancy.', 'Freedom', 'Generosity',
       'Trust..Government.Corruption.', 'Dystopia.Residual']

x_train, y_train = train[x_cols], train['Happiness.Score']
x_test, y_test = test[x_cols], test['Happiness.Score']
## model training
linear_model = LinearRegression().fit(x_train, y_train)

print('Train RMSE:', rmse_score(linear_model, x_train, y_train))
print('Test RMSE:', rmse_score(linear_model, x_test, y_test))
def analyze():

    def translate(value, rightMin, rightMax, change = ''):
        if change == '':
            leftMin = 1
            leftMax = 10
        else:
            leftMin = change[0]
            leftMax = change[1]
        # Figure out how 'wide' each range is
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin

        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)
        if change != "":
            ret_val = rightMin + (valueScaled * rightSpan)
            if ret_val > 10:
                ret_val = 10
        else:
            ret_val = rightMin + (valueScaled * rightSpan)

        # Convert the 0-1 range into a value in the right range.
        return ret_val

    print('Answer these following questions on a scale from 1 to 10. Feel free to use decimals!')
    print('1 refers to a low application / lack of the attribute on your life. 10 refers to a high influence / abundance of the attribute on your life')


    econ = input('Economy/Financial Situation ')
    econ_trans = translate(float(econ), 0,1.87 )

    family = input('Family Support ')
    family_trans = translate(float(family),0,1.61)

    health = input('Health/Well-Being ')
    health_trans = translate(float(health), 0,0.95)

    freedom = input('Freedom ')
    freedom_trans = translate(float(freedom), 0,0.66)

    generosity = input('Generosity ')
    generosity_trans = translate(float(generosity),0,0.84)

    trust = input('Trust/Government Support ')
    trust_trans = translate(float(trust), 0,0.46)

    dystopia = input('Dystopia/Injustice ')
    dystopia_trans = translate(float(dystopia),0.38,3.12)

    pred_list = [[econ_trans, family_trans, health_trans, freedom_trans, generosity_trans, trust_trans,dystopia_trans]]
    prediction = linear_model.predict(pred_list)[0]
    print('Your Happiness Score on a scale of 1-10 is', translate(prediction, 1,10, [2.69,7.54]))
    country_score = min(list(df['Happiness.Score']), key=lambda x:abs(x-prediction))
    country = df[df['Happiness.Score'] == country_score].iloc[0].Country
    rank = df[df['Happiness.Score'] == country_score].iloc[0]['Happiness.Rank']

    print('Your Happiness Score is similar to those of people from', country)
    print('This country is ranked', rank, 'in the world for most happy nations.')
analyze()
