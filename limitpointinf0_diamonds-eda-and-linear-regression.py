import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



sns.set(style="dark")



def handle_non_numeric(df):

    columns = df.columns.values

    for col in columns:

        text_digit_vals = {}

        def convert_to_int(val):

            return text_digit_vals[val]

        

        if df[col].dtype != np.int64 and df[col].dtype != np.float64:

            column_contents = df[col].values.tolist()

            unique_elements = set(column_contents)

            x = 0

            for unique in unique_elements:

                if unique not in text_digit_vals:

                    text_digit_vals[unique] = x

                    x += 1

            df[col] = list(map(convert_to_int,df[col]))

    return df



df = pd.read_csv('../input/diamonds.csv')

df = df.drop('Unnamed: 0',1)



print('imports done.')
corr = df.corr()

plt.figure(figsize=(20,10))

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True)

plt.title("Diamond Correlation Heatmap")



plt.show()
df = handle_non_numeric(df)

X = np.array(df.drop('price',1))

X = preprocessing.scale(X)

y = np.array(df['price'])



accuracy = []

x_range = []

for i in range(1000):

    x_range.append(i)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()

    model.fit(X_train,y_train)

    acc = model.score(X_test,y_test)

    accuracy.append(acc)



plt.figure(figsize=(20,10))

plt.title('Linear Regression Accuracy')

plt.plot(x_range, accuracy)

plt.xlabel('Iterations')

plt.ylabel('R^2')

plt.show()