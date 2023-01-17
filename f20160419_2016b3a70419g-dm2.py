import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline
data_orig = pd.read_csv('../input/data-mining-assignment-2/train.csv',index_col=0)
df = data_orig.copy(deep=True)
object_cols = df.columns[(df.dtypes == object)]

binary_cols = [col for col in object_cols if not len(df[col].value_counts()) > 2]

object_cols = [col for col in object_cols if len(df[col].value_counts()) > 2]



df = pd.get_dummies(df,columns=object_cols,prefix=object_cols)

df = pd.get_dummies(df,columns=binary_cols,prefix=binary_cols,drop_first=True)
target = df.corr().abs().loc['Class']

target = target[target<0.1]

print(target.index)

for col in target.index:

    del df[col]
def correlation_filter(data, threshold):

    dataset = data.copy(deep=True)

    col_corr = set() # Set of all the names of deleted columns

    corr_matrix = dataset.corr()

    for i in range(len(corr_matrix.columns)):

        for j in range(i):

            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):

                colname = corr_matrix.columns[i] # getting the name of column

                col_corr.add(colname)

                if colname in dataset.columns:

                    del dataset[colname] # deleting the column from the dataset

    print(col_corr)

    return dataset,list(col_corr)
df,col_corr = correlation_filter(df,0.90)
from sklearn.preprocessing import StandardScaler,MinMaxScaler



scaler = StandardScaler()

scaled_df = scaler.fit_transform(df.loc[:,df.columns != 'Class'])

scaled_df = pd.DataFrame(scaled_df,columns = df.columns[df.columns != 'Class'])

scaled_df
test =pd.read_csv('../input/data-mining-assignment-2/test.csv',index_col = 0)

idx = test.index



test = pd.get_dummies(test,columns=object_cols,prefix=object_cols)

test = pd.get_dummies(test,columns=binary_cols,prefix=binary_cols,drop_first=True)



drop_cols = list(set(target.index).union(set(col_corr)))



for col in drop_cols:

    del test[col]

    

test = pd.DataFrame(scaler.transform(test),columns = test.columns)
from sklearn.ensemble import RandomForestClassifier

rf_best = RandomForestClassifier(n_estimators=100, max_depth = 13, min_samples_split=2,min_samples_leaf=1,random_state=201)

rf_best.fit(scaled_df,df['Class'])



output = pd.DataFrame(rf_best.predict(test),index=idx,columns=['Class'])

output.index.name = 'ID'

output
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(output)