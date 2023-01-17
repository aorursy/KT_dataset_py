import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

%matplotlib inline
df = pd.read_csv('../input/data.csv')

df.head()
df.info()
df.drop(columns=['Unnamed: 0'], inplace=True)
df.sample(5)
df.describe()
sb.set_style('whitegrid')
bins = np.arange(df['Overall'].min(), df['Overall'].max()+1, 1)



plt.figure(figsize=[8,5])

plt.hist(df['Overall'], bins=bins)

plt.title('Overall Rating Distribution')

plt.xlabel('Mean Overall Rating')

plt.ylabel('Count')

plt.show()
plt.figure(figsize=[16,5])

plt.suptitle('Overall Rating Vs Age', fontsize=16)



plt.subplot(1,2,1)

bin_x = np.arange(df['Age'].min(), df['Age'].max()+1, 1)

bin_y = np.arange(df['Overall'].min(), df['Overall'].max()+2, 2)

plt.hist2d(x = df['Age'], y = df['Overall'], cmap="YlGnBu", bins=[bin_x, bin_y])

plt.colorbar()

plt.xlabel('Age (years)')

plt.ylabel('Overall Rating')



plt.subplot(1,2,2)

plt.scatter(x = df['Age'], y = df['Overall'], alpha=0.25, marker='.')

plt.xlabel('Age (years)')

plt.ylabel('Overall Rating')

plt.show()
plt.figure(figsize=[8,5])

sb.jointplot(x=df.Overall, y=df.Potential, kind='kde')

plt.show()
plt.figure(figsize=[8,5])

plt.scatter(x=df.Overall, y=df.Potential, c=df.Age, alpha=0.25, cmap='rainbow' )

plt.colorbar().set_label('Age')

plt.xlabel('Overall Rating')

plt.ylabel('Potential')

plt.show()
df_opa = df[['ID', 'Name', 'Age', 'Overall', 'Potential', 'Value', 'International Reputation', 'Height', 'Weight', 'Position', 'Wage']]

df_opa.head()
df_opa.head()
def currencystrtoint(amount):

    new_amount = []

    for s in amount:

        list(s)

        abbr = s[-1]

        if abbr is 'M':

            s = s[1:-1]

            s = float(''.join(s))

            s *= 1000000

        elif abbr is 'K':

            s = s[1:-1]

            s = float(''.join(s))

            s *= 1000

        else:

            s = 0

        new_amount.append(s)

    return new_amount
df_opa['Value'] = currencystrtoint(list(df_opa['Value']))
df_opa['Wage'] = currencystrtoint(list(df_opa['Wage']))
def lengthstrtoint(length):

    new_length = []

    for h in length:

        if type(h) is str:

            list(h)

            h = (float(h[0])*12) + float(h[2:])

        new_length.append(h)

    return new_length
df_opa['Height'] = lengthstrtoint(list(df_opa['Height']))
mean_height = df_opa['Height'].mean()
df_opa.Height.loc[df_opa['Height'].isnull()] = mean_height
def weightstrtoint(weight):

    new_weight = []

    for w in weight:

        if type(w) is str:

            w = float(w[0:-3])

        new_weight.append(w)

    return new_weight
df_opa['Weight'] = weightstrtoint(list(df_opa['Weight']))
mean_weight = df_opa['Weight'].mean()

df_opa.Weight.loc[df_opa['Weight'].isnull()] = mean_weight
mean_internationa_rep = df_opa['International Reputation'].mean()

df_opa['International Reputation'].loc[df['International Reputation'].isnull()] = mean_internationa_rep
df_opa.describe()
plt.figure(figsize=(20,15))

sb.pairplot(df_opa)
sb.lmplot(data=df_opa, x='Overall', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'} )
sb.lmplot(data=df_opa, x='Overall', y='Wage', order=2, scatter_kws={'alpha':0.3, 'color':'y'} )
sb.lmplot(data=df_opa, x='Overall', y='Wage', order=3, scatter_kws={'alpha':0.3, 'color':'y'} )
sb.lmplot(data=df_opa, x='Age', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'} )
sb.lmplot(data=df_opa, x='Age', y='Wage', order=2, scatter_kws={'alpha':0.3, 'color':'y'})
sb.lmplot(data=df_opa, x='Value', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'} )
sb.lmplot(data=df_opa, x='Height', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'} )
sb.lmplot(data=df_opa, x='Height', y='Wage', order=2, scatter_kws={'alpha':0.3, 'color':'y'} )
sb.lmplot(data=df_opa, x='Weight', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'} )
sb.lmplot(data=df_opa, x='Weight', y='Wage', order=2, scatter_kws={'alpha':0.3, 'color':'y'} )
sb.lmplot(data=df_opa, x='International Reputation', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'} )
df_opa = pd.get_dummies(df_opa, columns=['Position'], drop_first=True)
df_opa.info()
df_opa.describe()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn import metrics
X = df_opa.drop(['ID', 'Name', 'Wage'], axis=1)

y = df_opa['Wage']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.preprocessing import StandardScaler

stsc = StandardScaler()

Xtrain = stsc.fit_transform(Xtrain)

Xtest = stsc.fit_transform(Xtest)
def pred_wage(degree, Xtrain, Xtest, ytrain):

    if degree > 1:

        poly = PolynomialFeatures(degree = degree)

        Xtrain = poly.fit_transform(Xtrain)

        Xtest = poly.fit_transform(Xtest)

    lm = LinearRegression()

    lm.fit(Xtrain, ytrain)

    wages = lm.predict(Xtest)

    return wages
predicted_wages1 = pred_wage(1, Xtrain, Xtest, ytrain)
sb.regplot(ytest, predicted_wages1, scatter_kws={'alpha':0.3, 'color':'y'})

plt.xlabel('Actual Wage')

plt.ylabel('Predicted Wage')

plt.show()
print('Mean Absolute Error : ' + str(metrics.mean_absolute_error(ytest, predicted_wages1)))

print('Mean Squared Error : ' + str(metrics.mean_squared_error(ytest, predicted_wages1)))

print('Root Mean Squared Error : ' + str(np.sqrt(metrics.mean_squared_error(ytest, predicted_wages1))))
predicted_wages2 = pred_wage(2, Xtrain, Xtest, ytrain)
sb.regplot(ytest, predicted_wages2, scatter_kws={'alpha':0.3, 'color':'y'})

plt.xlabel('Actual Wage')

plt.ylabel('Predicted Wage')

plt.show()
print('Mean Absolute Error : ' + str(metrics.mean_absolute_error(ytest, predicted_wages2)))

print('Mean Squared Error : ' + str(metrics.mean_squared_error(ytest, predicted_wages2)))

print('Root Mean Squared Error : ' + str(np.sqrt(metrics.mean_squared_error(ytest, predicted_wages2))))
sb.distplot(ytest-predicted_wages1, bins=200, hist_kws={'color':'r'}, kde_kws={'color':'y'})

plt.xlim(-50000, 50000)