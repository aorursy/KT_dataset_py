import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv("../input/data.csv")

df.head()
df.index = df['id']

df.drop(["id", "Unnamed: 32"], axis = 1, inplace = True)

df.head()
df.head()
df.boxplot(by = 'diagnosis', figsize = (20, 20))

plt.show()
from sklearn.preprocessing import StandardScaler as ss

dft = df.drop(['diagnosis'], axis = 1, inplace = False)

scaler = ss().fit(dft)

scaled_df = pd.DataFrame(scaler.transform(dft))
scaled_df.index = df.index

scaled_df.columns = dft.columns

scaled_df['diagnosis'] = df['diagnosis']

scaled_df.head()
scaled_df.boxplot(by = "diagnosis", figsize = (20, 20))

plt.show()
diagnosis = df.diagnosis

df.drop(['diagnosis'], axis = 1, inplace = True)
from sklearn.cross_validation import train_test_split as tspl

df_train, df_test, diag_train, diag_test = tspl(df, diagnosis, test_size = 0.33)
from sklearn.neighbors import KNeighborsClassifier as KNN

knn = KNN(n_neighbors = 5)

knn.fit(df_train, diag_train)
knn.score(df_test, diag_test)
dft = df_train.copy()

scaler = ss().fit(dft)

scaled_dft = pd.DataFrame(scaler.transform(dft))

scaled_dft.index = dft.index

scaled_dft.columns = dft.columns

scaled_dft['diagnosis'] = diag_train

scaled_dft.head()
grouped_mean = scaled_dft.groupby(['diagnosis']).mean()
diff_col = np.array(grouped_mean.loc['B']) - np.array(grouped_mean.loc['M'])

diff_col = np.absolute(diff_col)

diff_col = list(diff_col)
to_drop = []

columns = df.columns

for i in range(len(columns)):

    if diff_col[i] < 1:

        to_drop.append(columns[i])

    

dftr = df_train.copy()

dftr.drop(to_drop, axis = 1, inplace = True)



dfts = df_test.copy()

dfts.drop(to_drop, axis = 1, inplace = True)
knn1 = KNN(n_neighbors = 5)

knn1.fit(dftr, diag_train)
knn1.score(dfts, diag_test)
for i in range(1, 20):

    knn = KNN(n_neighbors = i)

    knn.fit(dftr, diag_train)

    score = knn.score(dfts, diag_test)

    print("N = " + str(i) + " :: Score = " + str(score))
knn = KNN(n_neighbors= 5)

knn.fit(dftr, diag_train)
prediction_prob = knn.predict_proba(dfts)

predictions = knn.predict(dfts)
from sklearn.metrics import confusion_matrix as cfm

cfm(predictions, diag_test)
def predict_cutoff(pred_prob, cutoff = 0.5):

    prediction_var = []

    for element in pred_prob:

        if element[0] > cutoff:

            prediction_var.append("B")

        else:

            prediction_var.append("M")

    return np.array(prediction_var)
for i in range(1, 10):

    i = float(i) / 10

    print("Cutoff: " + str(i) + ";\n Confusion Matrix: ")

    print(cfm(predict_cutoff(prediction_prob, i), diag_test))

    print("\n\n")