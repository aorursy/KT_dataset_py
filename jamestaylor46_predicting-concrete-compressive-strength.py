import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df = pd.read_csv('/kaggle/input/yeh-concret-data/Concrete_Data_Yeh.csv')
df.shape
df.head()
df.info()
updated_col_names = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer',

                     'Coarse Aggregrate', 'Fine Aggregate', 'Age', 'Compressive Strength']



df.columns = updated_col_names
df.describe()
sns.distplot(df['Compressive Strength'])
sns.pairplot(df)

plt.show()
corr = df.corr()



plt.figure(figsize = (10,8))

sns.heatmap(corr, cmap = 'Blues', annot = True)

plt.title('Pearson Correlation Coefficients')

plt.show()
corr_sorted = corr.unstack().sort_values(kind='quicksort', ascending = False)
print(corr_sorted[corr_sorted!=1].head(10))

print(corr_sorted[corr_sorted!=1].tail(10))
fig, ax = plt.subplots(figsize=(10,8))

sns.scatterplot(y="Compressive Strength", x="Cement", hue="Water", size="Age", data=df, ax=ax, sizes=(50, 300),

                palette='RdYlGn', alpha=0.9)

ax.set_title("Compressive Strength vs Cement, Age, Water")

ax.legend()

plt.show()
fig, ax = plt.subplots(figsize=(10,8))

sns.scatterplot(y="Compressive Strength", x="Fine Aggregate", hue="Fly Ash", size="Superplasticizer", data=df, ax=ax, sizes=(50, 300),

                palette='RdYlBu', alpha=0.9)

ax.set_title("Compressive Strength vs Fine Aggregate, Fly Ash, Superplasticizer")

ax.legend(loc="upper left", bbox_to_anchor=(1,1)) # Moved outside the chart so it doesn't cover any data

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
# Splitting the features and target variable



cols = df.columns.drop('Compressive Strength')

X = df[cols]

y = df['Compressive Strength']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()



X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
lr = LinearRegression()

lr.fit(X_train, y_train)
ridge_alpha = RidgeCV(cv=10)

ridge_alpha.fit(X_train, y_train)

alpha = ridge_alpha.alpha_



ridge = Ridge(alpha=alpha, random_state=42)

ridge.fit(X_train, y_train)
lasso_alpha = LassoCV(cv=10)

lasso_alpha.fit(X_train, y_train)

alpha = lasso_alpha.alpha_



lasso = Lasso(alpha=alpha, random_state=42)

lasso.fit(X_train, y_train)
EN_alpha = ElasticNetCV(cv=10)

EN_alpha.fit(X_train, y_train)

alpha= EN_alpha.alpha_



EN = ElasticNet(alpha=alpha, random_state=42)

EN.fit(X_train, y_train)
def plot_coef (models):

    

    coefs = {}

    rows = []

    fig, ax = plt.subplots(figsize=(16,8))

    offset = 0

    width = 0.23

    x = np.arange(len(X.columns))

    

    # Creating a neat table to view

    for model in models:

        coefs[type(model).__name__] = model.coef_

        

    coefs_table = pd.DataFrame.from_dict(coefs, orient='index')

    coefs_table.columns = X.columns

    

    # Using the table to create a chart

    for i in range(len(models)):

        increment = width

        ax.bar(x - width + offset, coefs_table.iloc[i], width=width, label=type(models[i]).__name__)

        offset = offset + increment

        

    ax.set_ylabel('Coefficient')

    ax.set_xlabel('Features')

    ax.set_title('Feature Coefficients')

    ax.set_xticks(x)

    ax.set_xticklabels(X.columns)

    ax.legend()

    

    return coefs_table
models = [lr, ridge, lasso, EN]



coefs_table = plot_coef(models)
coefs_table
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def make_pred (models, X_test, y_test):

    

    results = {}

    

    for model in models:

        y_pred = model.predict(X_test)

        results[type(model).__name__] = [mean_squared_error(y_test, y_pred)**(1/2), 

                                         mean_absolute_error(y_test, y_pred), 

                                         r2_score(y_test, y_pred)]

        

        results = pd.DataFrame(results, index=['RMSE','MAE','R2'])

        

    return results
make_pred(models, X_test, y_test).T