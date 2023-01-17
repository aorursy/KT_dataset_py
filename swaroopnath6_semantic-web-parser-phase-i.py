import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import optimize

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

import numpy as np
dataset = pd.read_csv('../input/semantic-web-parsing-dataset-part-i/semantic_web_parsing_dataset_pt_1.csv')
dataset.describe()
dataset['Label'].value_counts()
plt.figure(figsize = (30, 25))

g = sns.PairGrid(dataset, vars = ['Fraction Formatting Tags', 'Num Formatting Tags', 'Fraction Words', 'Num Words', 'Fraction p Children'], hue="Label")

g.map_diag(plt.hist)

g.map_offdiag(sns.scatterplot)

g.add_legend();
high_fpc_unnecessary = dataset.loc[(dataset['Label'] == 0) & (dataset['Fraction p Children'] > 0.08)]
high_fpc_unnecessary
low_fpc_necessary = dataset.loc[(dataset['Label'] == 1) & (dataset['Fraction p Children'] < 0.2)]
low_fpc_necessary
sns.scatterplot(x = 'Fraction p Children', y = 'Label', hue = 'Label', data = dataset)

plt.show()

sns.scatterplot(x = 'Num Words', y = 'Label', hue = 'Label', data = dataset)

plt.show()
dataset['Add'] = dataset.apply(lambda row: row['Fraction p Children'] + row['Num Words'], axis = 1)

dataset['Multiply'] = dataset.apply(lambda row: row['Fraction p Children'] * row['Num Words'], axis = 1)
plt.figure(figsize = (30, 25))

g = sns.PairGrid(dataset, vars = ['Fraction p Children', 'Add', 'Multiply'], hue = 'Label')

g.map_diag(plt.hist)

g.map_offdiag(sns.scatterplot)

g.fig.set_size_inches(15,15)

plt.show()

sns.scatterplot(x = 'Add', y = 'Label', hue = 'Label', data = dataset)

plt.show()

sns.scatterplot(x = 'Multiply', y = 'Label', hue = 'Label', data = dataset)

plt.show()
def sigmoid(X, theta_1, theta_2, a, b):

    X_1, X_2 = X

    return 1/(1 + np.exp(-a * (theta_1 * X_1 + theta_2 * X_2 - b)))
X_1 = dataset['Fraction p Children'].values

X_2 = dataset['Add'].values

y = dataset['Label'].values



params_optimal, _ = optimize.curve_fit(sigmoid, xdata = (X_1, X_2), ydata = y, p0 = (1, 1, 100, 0.5))



pseudo_X_1 = np.arange(0, 1, 0.1)

pseudo_X_2 = np.arange(0, 1, 0.1)

y_proba = sigmoid((pseudo_X_1, pseudo_X_2), *params_optimal)



dataset['Combination'] = dataset.apply(lambda row: params_optimal[0] * row['Fraction p Children'] 

                                       + params_optimal[1] * row['Add'], axis = 1)



sns.scatterplot(x = 'Combination', y = 'Label', hue = 'Label', data = dataset)

plt.vlines(0.2, ymin = 0, ymax = 1, linestyle = '--', color = 'c')

plt.legend()
X = dataset.loc[:, ['Fraction p Children', 'Add']]

y = dataset['Label'].values



sc_fp = StandardScaler()

sc_m = StandardScaler()

X.iloc[:, 0] = sc_fp.fit_transform(X.iloc[:, 0].values.reshape(-1, 1))

X.iloc[:, 1] = sc_m.fit_transform(X.iloc[:, 1].values.reshape(-1, 1))



model = LogisticRegression().fit(X, y)



theta_1, theta_2 = model.coef_[0]



dataset['Combination'] = theta_1 * sc_fp.transform(dataset['Fraction p Children'].values.reshape(-1, 1)) + theta_2 * sc_m.transform(dataset['Add'].values.reshape(-1, 1))



pseudo_X_1 = np.arange(0, 20, 0.05)

pseudo_X_2 = np.arange(0, 20, 0.05)

combination = theta_1 * pseudo_X_1 + theta_2 * pseudo_X_2



X_test = pd.DataFrame()

X_test['Fraction p Values'] = pseudo_X_1

X_test['Add'] = pseudo_X_2



y_proba = model.predict_proba(X_test.values)



sns.scatterplot(x = 'Combination', y = 'Label', data = dataset, hue = 'Label')

plt.plot(combination, y_proba, label = 'Fitted Data')

plt.hlines(y = 0.5, xmin = -1, xmax = 50, linestyle = '--', color = 'red')

plt.vlines(x = 9, ymin = 0, ymax = 1, linestyle = '--', color = 'cyan')

plt.legend()