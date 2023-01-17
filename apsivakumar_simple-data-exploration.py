import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



train_data = pd.read_csv("../input/train.csv")

train_label = train_data["y"]

# train_features = train_data.drop(["ID"], axis=1, inplace=False)



print(train_data.shape)
# Feature importance based on Variance of mean value

categorical_features = ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]

records = []

for feature in categorical_features:

    grouped_record = train_data.groupby(feature).agg({"y":'mean'}).reset_index()

    records.append((feature,np.var(grouped_record["y"])))

variance_of_mean = pd.DataFrame.from_records(records,columns=('Feature', 'Variance_of_Mean'))

variance_of_mean.sort_values(by="Variance_of_Mean", inplace=True)



plt.figure(figsize=(12,6))

plt.title("Feature importances")

plt.barh(range(len(variance_of_mean["Variance_of_Mean"])), variance_of_mean["Variance_of_Mean"], 0.5, color="#8ea7d1", align="center")

plt.yticks(range(len(variance_of_mean["Variance_of_Mean"])), variance_of_mean["Feature"], rotation='horizontal')

plt.show()