import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing,cross_validation,neighbors

from sklearn.model_selection import cross_val_score, train_test_split

import matplotlib.pyplot as plt



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



df = pd.read_csv("../input/bird.csv")

df = df.drop('id',1)
cats = df['type'].unique()



for i in cats:

    plt.figure(i)

    plt.title('Type ' + str(i))

    _ = df[df['type'] == i].boxplot()
df_ratio = df.drop(['huml', 'humw', 'ulnal', 'ulnaw', 'feml', 'femw', 'tibl', 'tibw', 'tarl', 'tarw'],1)

df_ratio['hum_rat'] = df['huml']/df['humw']

df_ratio['ulna_rat'] = df['ulnal']/df['ulnaw']

df_ratio['fem_rat'] = df['feml']/df['femw']

df_ratio['tib_rat'] = df['tibl']/df['tibw']

df_ratio['tar_rat'] = df['tarl']/df['tarw']



for i in df['type'].unique():

    print(i)

    df_ratio_sub = df_ratio[df_ratio['type'] == i]

    for x in df_ratio_sub.columns[1:]:

        print(x)

        print(np.mean(df_ratio_sub[x]))

        





    
df = handle_non_numeric(df)

df = df.dropna()

X = np.array(df.drop('type',1))

X = preprocessing.scale(X)

y = np.array(df['type'])



accuracy = []

iterations = []

for i in range(1,10000):

    iterations.append(i)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=1)

    clf.fit(X_train,y_train)

    acc = clf.score(X_test,y_test)

    accuracy.append(acc)



plt.plot(iterations, accuracy)

plt.xlabel('Iteration')

plt.ylabel('Accuracy on Test')

plt.show()



print('average accuracy', np.mean(accuracy))

print('standard deviation', np.std(accuracy))