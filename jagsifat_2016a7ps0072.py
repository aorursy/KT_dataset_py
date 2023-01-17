import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline



from collections import deque



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from math import sqrt

from sklearn.metrics import mean_squared_error,r2_score





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/bits-f464-l1/train.csv")
df_test = pd.read_csv("../input/bits-f464-l1/test.csv")
## Creating agents for different IDs

df_train_ID = df_train[["id"]]

df_train_ID_val = df_train_ID.values

df_train_ID_val = np.subtract(df_train_ID_val,1)

df_train_agent = np.remainder(df_train_ID_val,7)

df_train["Agent"] = df_train_agent





df_test_ID = df_test[['id']]

df_test_ID_val = df_test_ID.values

df_test_ID_val = np.subtract(df_test_ID_val,1)

df_test_agent  = np.remainder(df_test_ID_val,7)

df_test["Agent"] = df_test_agent
df_train = df_train.drop(columns=['id', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'])

df_test  = df_test.drop(columns=['id', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'])
deleted_col = []

col_7 = []



for col in df_train:

    count = len(df_train[col].unique())

    if count == 1:

      deleted_col.append(col)

    if count == 7:

      col_7.append(col)



print(deleted_col) 
df_mod1_train = df_train.copy()

df_mod1_train = df_mod1_train.drop(columns = deleted_col)

df_mod1_train.info()
df_mod1_test = df_test.copy()

df_mod1_test = df_mod1_test.drop(columns = deleted_col)

df_mod1_test.info()
df_train_agent1 = df_mod1_train[df_mod1_train['Agent']==1]

df_train_agent1
for i in df_train_agent1.columns:

  y = df_train_agent1[i].values

  x = df_train_agent1['time']

  plt.plot(x, y)

  plt.show()
## Making changes for PCA

x_time  = df_mod1_train[['time']].values

x_agent = df_mod1_train[['Agent']].values

y_label = df_mod1_train['label'].values



df_mod2_train = df_mod1_train.copy()

df_mod2_train = df_mod2_train.drop(columns=['Agent', 'time', 'label'])
df_mod2_test = df_mod1_test.copy()

df_mod2_test = df_mod2_test.drop(columns=['Agent', 'time'])
## PCA



from sklearn.preprocessing import StandardScaler 

scalar = StandardScaler() 

scalar.fit(df_mod2_train) 

scaled_data = scalar.transform(df_mod2_train) 



from sklearn.decomposition import PCA 

pca = PCA().fit(scaled_data)

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('#components')

plt.ylabel('variance')

plt.title('Pulsar Dataset Explained Variance')

plt.show()
scaled_data_test = scalar.transform(df_mod2_test)
## 40 is optimal (graph observation)

pca = PCA(n_components = 40) 

pca.fit(scaled_data) 

pca_X = pca.transform(scaled_data) 



pca_X_test = pca.transform(scaled_data_test)



df_mod3_train = pd.DataFrame(pca_X)

df_mod3_train
df_mod3_test = pd.DataFrame(pca_X_test)

df_mod3_test
df_mod3_train["label"] = df_mod1_train['label']

df_mod3_train["Agent"] = df_mod1_train["Agent"]



df_mod3_test['Agent'] = df_mod1_test['Agent']

df_mod3_test
##Window size 6 and 7 seem the most optimal



from sklearn.tree import DecisionTreeRegressor



pred_rmse = []

window_size = 6

final_data = []

labels_final = []

print("Agent Numbers:-")

for agent_number in range(7):

  print(agent_number)

  df_agent = df_mod3_train[df_mod3_train['Agent'] == agent_number]

  df_agent_label = df_agent[['label']].values

  df_agent = df_agent.drop(columns=['Agent', 'label'])

  li = deque()

  for i in range(window_size):

    row = df_agent.iloc[i].tolist()

    li.extend(row)

  new_data = []

  labels = []

  for i in range(window_size, df_agent.shape[0]):

    new_row = []

    values = df_agent.iloc[i].tolist()

    new_row.extend(li)

    new_row.extend(values)

    new_row.append(agent_number)

    labels.append(df_agent_label[i][0])

    for j in range(40):

      li.popleft();

    li.extend(values)

    new_data.append(new_row)

  labels_final.extend(labels)

  final_data.extend(new_data)



X = np.array(final_data)

Y = np.array(labels_final)



train_pct_index = int(0.9*len(X))



x_train, x_test = X[:train_pct_index], X[train_pct_index:]

y_train, y_test = Y[:train_pct_index], Y[train_pct_index:]



dtr = DecisionTreeRegressor()

dtr.fit(x_train, y_train)

y_pred1 = dtr.predict(x_test)



print("DT")

val = sqrt(mean_squared_error(y_test, y_pred1))

print(val)

pred_rmse.append(val)
final_data_test = []

labels_final_test = []

print("Agent Numbers:-")

for agent_number in range(7):

  print(agent_number)

  df_test_agent = df_mod3_test[df_mod3_test['Agent']==agent_number]

  df_train_agent = df_mod3_train[df_mod3_train['Agent']==agent_number]

  df_train_agent = df_train_agent.drop(columns=['Agent', 'label'])

  leng = df_train_agent.shape[0]

  df_test_agent = df_test_agent.drop(columns=['Agent'])

  li = deque()

  for i in range(leng - window_size, leng):

    row = df_train_agent.iloc[i].tolist()

    li.extend(row)

  new_data = []

  labels = []

  for i in range(0, df_test_agent.shape[0]):

    new_row = []

    values = df_test_agent.iloc[i].tolist()

    new_row.extend(li)

    new_row.extend(values)

    new_row.append(agent_number)

    for j in range(40):

      li.popleft();

    li.extend(values)

    new_data.append(new_row)

  final_data_test.extend(new_data)
X = np.array(final_data)

Y = np.array(labels_final)

print(X.shape)

print(Y.shape)



train_pct_index = int(0.9*len(X))



x_train, x_test = X[:train_pct_index], X[train_pct_index:]

y_train, y_test = Y[:train_pct_index], Y[train_pct_index:]
from sklearn.ensemble import RandomForestRegressor

dtr = DecisionTreeRegressor(max_leaf_nodes=100)

dtr.fit(x_train, y_train)

y_pred1 = dtr.predict(x_test)

print("DT")

print(sqrt(mean_squared_error(y_test, y_pred1)))
from sklearn.metrics import r2_score

y_train_pred1 = dtr.predict(x_train)

print(sqrt(mean_squared_error(y_train, y_train_pred1)))

print(r2_score(y_train, y_train_pred1))
from sklearn.metrics import r2_score

print("DT")

print(r2_score(y_test, y_pred1))
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train, y_train)

y_pred3 = lr.predict(x_test)

print(sqrt(mean_squared_error(y_test, y_pred3)))

print(r2_score(y_test, y_pred3))
y_train_pred3 = lr.predict(x_train)

print(sqrt(mean_squared_error(y_train, y_train_pred3)))

print(r2_score(y_train, y_train_pred3))
X = np.array(final_data)

Y = np.array(labels_final)

train_pct_index = int(0.8*len(X))

x_train, x_test = X[:train_pct_index], X[train_pct_index:]

y_train, y_test = Y[:train_pct_index], Y[train_pct_index:]
X = np.array(final_data)

Y = np.array(labels_final)

x_train = X

y_train = Y



from sklearn.ensemble import ExtraTreesRegressor

rf = ExtraTreesRegressor(n_estimators=100, verbose=1, n_jobs=-1, warm_start=True, max_leaf_nodes=10000)

rf.fit(x_train, y_train)



y_pred2 = rf.predict(x_test)

from sklearn.metrics import r2_score

print("RF")

print(sqrt(mean_squared_error(y_test, y_pred2)))

print(r2_score(y_test, y_pred2))

y_train_pred2 = rf.predict(x_train)

print(sqrt(mean_squared_error(y_train, y_train_pred2)))

print(r2_score(y_train, y_train_pred2))



final_data_test_reshuffled = []



for i in range(5756):

  for j in range(7):

    #print(j*5756 + i)

    final_data_test_reshuffled.append(final_data_test[j*5756 + i])



print(len(final_data_test_reshuffled))

print(len(final_data_test_reshuffled[0]))
x_test = final_data_test_reshuffled

y_pred_test = rf.predict(x_test)



result = pd.DataFrame({'label':y_pred_test})

result['id'] = ""

col = ['id', 'label']

result = result[col]



for index, row in result.iterrows():

  result.at[index, 'id'] = index + 1



result.to_csv("../output/bits-f464-l1/test.csv",index=False)
y_train_pred2 = rf.predict(x_train)

print(sqrt(mean_squared_error(y_train, y_train_pred2)))

print(r2_score(y_train, y_train_pred2))
print(rf.n_features_)

print(rf.n_outputs_)
for agent_number in range(7):

  df_agent = df3_train[df_mod3_train['Agent'] == agent_number]

  df_agent_label = df_agent[['label']]

  df_agent = df_agent.drop(columns=['Agent', 'label'])

  #defining deque

  li = deque()

  for i in range(window_size):

    row = df_agent.iloc[i].tolist()

    li.extend(row)

  #making new data format

  new_data = []

  for i in range(window_size, df_agent.shape[0]):

    new_row = []

    values = df_agent.iloc[i].tolist()

    new_row.extend(li)

    new_row.extend(values)

    for j in range(40):

      li.popleft();

    li.extend(values)

    new_data.append(new_row)



  #print(len(new_data[0]))

  

  new_df = pd.DataFrame.from_records(new_data)

  #remove first 5 entries from labels

  #print(df_agent_label)



  remove_initial = []

  for i in range(window_size):

    remove_initial.append(agent_number + i*7)

  

  print(remove_initial)



  df_agent_label = df_agent_label.drop(index=remove_initial)

  #print(df_agent_label)



  #new_df['label'] = df_agent_label.values

  new_df_values = new_df.to_numpy()

  

  X = new_df_values

  Y = df_agent_label.values

  print(X.shape)

  print(Y.shape)



  x_plot = []

  for i in range(window_size, 23024):

    x_plot.append(i)



  #plt.plot(x_plot, Y)

  #plt.show()



  train_pct_index = int(0.9*len(X))



  x_train, x_test = X[:train_pct_index], X[train_pct_index:]

  y_train, y_test = Y[:train_pct_index], Y[train_pct_index:]

  #train_test_split(X1, Y, test_size = 0.1)



  lr = LinearRegression()



  lr.fit(x_train, y_train)

  y_pred = lr.predict(x_test)

  print("Agent " + str(agent_number) + " RMSE")

  print(sqrt(mean_squared_error(y_test, y_pred)))
df_train_agent1 = df_mod3_train[df_mod3_train['Agent'] == 1]

print(df_train_agent1)