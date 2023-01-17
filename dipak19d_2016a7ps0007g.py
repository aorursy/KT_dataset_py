import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from collections import deque



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from math import sqrt

from sklearn.metrics import mean_squared_error

df = pd.read_csv("/content/drive/My Drive/ML Lab/lab1/train.csv")
df_test = pd.read_csv("/content/drive/My Drive/ML Lab/lab1/test.csv")
print(df.columns)
print(df.head())
#as we say that these 7 agents are doing different tasks, we will train different models for the seven of them

df2 = df[['id']]

df2_val = df2.values

df2_val = df2

df2_val = np.subtract(df2_val, 1)

agent1 = np.remainder(df2_val, 7).values



df["Agent"] = agent1
df2_test = df_test[['id']]

df2_val_test = df2_test.values

df2_val_test = df2_test

df2_val_test = np.subtract(df2_val_test, 1)

agent1_test = np.remainder(df2_val_test, 7).values



df_test["Agent"] = agent1_test
print(df_test)
print(df)
df = df.drop(columns=['id', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'])
df_test = df_test.drop(columns=['id', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'])
#Check null



print(df_test.isnull().values.any())

#No null values
to_be_removed = []

count_7 = []



for col in df:

    count = len(df[col].unique())

    if count == 1:

      print(col + ": " + str(count))

      to_be_removed.append(col)

    if count == 7:

      count_7.append(col)



print(to_be_removed) 
df1 = df.copy()

df1 = df1.drop(columns = to_be_removed)



print(df1)
df1_test = df_test.copy()

df1_test = df1_test.drop(columns = to_be_removed)



print(df1_test)
# corr = df1.corr()

# col_names = corr.columns.values

# print(col_names)

# corr_val = corr.values



# r, c = len(corr_val), len(corr_val[0])

# # print(r)

# # print(c)



# remove_attrib = []



# #remove attributes which are highly uncorrelated to resultant label

# print(col_names[90])

# for i in range(r):

#   #print(corr_val[i][90])

#   #for j in range(int(c/2) - 1):

#   j = 90

#   if corr_val[i][j] > -0.005 and corr_val[i][j] <0.005 and i != j:

#     # print(corr_val[i][j])

#     # print(str(i) + " " + str(j))

#     remove_attrib.append(col_names[i])



# #remove attributes highly correlated to each other

# for i in range(r):

#   for j in range(int(c/2) - 1):

#     if corr_val[i][j] < -0.995 or corr_val[i][j] > 0.995 and i != j:

#       # print(corr_val[i][j])

#       # print(str(i) + " " + str(j))

#       remove_attrib.append(col_names[i])





# remove_attrib = list(set(remove_attrib))

    

df2 = df1.copy()

# df2 = df2.drop(columns = remove_attrib)



# print(df2)
df2_test = df1_test.copy()
df_agent1 = df2[df2['Agent'] == 1]

print(df_agent1)
for item in df_agent1.columns:

  y = df_agent1[item].values

  x = df_agent1['time']

  plt.plot(x, y)

  plt.show()

  #df_agent1.plot(x='time', y=item, style='o')

  #plt.show()
#We will try to work with the data using a window of size 10 50 100 500 1000



#When going for PCA, I am removing time, label, Agent into separate df



labels_y = df2['label'].values

#print(labels_y)

agent_x = df2[['Agent']].values

time_x = df2[['time']].values

#print(agent_time_x)

df3 = df2.copy()

df3 = df3.drop(columns=['Agent', 'time', 'label'])
df3_test = df2_test.copy()

df3_test = df3_test.drop(columns=['Agent', 'time'])
print(df3_test)
print(df3.head())
#PCA analysis to figure out the best n_components parameter



# Importing standardscalar module  

from sklearn.preprocessing import StandardScaler 

from sklearn.preprocessing import MinMaxScaler

  

scalar = StandardScaler() 

  

# fitting 

scalar.fit(df3) 

scaled_data = scalar.transform(df3) 



# Importing PCA 

from sklearn.decomposition import PCA 



pca = PCA().fit(scaled_data)

#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Pulsar Dataset Explained Variance')

plt.show()
scaled_data_test = scalar.transform(df3_test)
#as we can see from the graph, 30 is the optimal number
#Principle Component Analysis PCA

pca = PCA(n_components = 40) 

pca.fit(scaled_data) 

x_pca = pca.transform(scaled_data) 
x_pca_test = pca.transform(scaled_data_test)
#convert x_pca to df so that we can easily add other rows



df4 = pd.DataFrame(x_pca)

# df3['Agent'] = df1['Agent']

# df3['time'] = df1['time']

print(df4)
df4_test = pd.DataFrame(x_pca_test)

print(df4_test)
df4["Agent"] = df1["Agent"]

df4["label"] = df1['label']
df4_test['Agent'] = df1_test['Agent']

print(df4_test)
print(df4)
#considering window values from 1 to 10

#7 window size is optimal



from sklearn.tree import DecisionTreeRegressor



#adding only for agent1

data_output = []





#



pred_rmse = []



window_size = 30

final_data = []

labels_final = []

print("QQ")

for agent_number in range(7):

  print(agent_number)

  df_agent = df4[df4['Agent'] == agent_number]

  df_agent_label = df_agent[['label']].values

  df_agent = df_agent.drop(columns=['Agent', 'label'])

  #defining deque

  li = deque()

  for i in range(window_size):

    row = df_agent.iloc[i].tolist()

    li.extend(row)

  #making new data format

  new_data = []

  labels = []

  for i in range(window_size, df_agent.shape[0]):

    new_row = []

    values = df_agent.iloc[i].tolist()

    new_row.extend(li)

    new_row.extend(values)

    #new_row.append(agent_number)

    labels.append(df_agent_label[i][0])

    for j in range(40):

      li.popleft();

    li.extend(values)

    new_data.append(new_row)

    #print(df_agent_label[i])

    #new_data.extend(df_agent_label[i])

  labels_final.append(labels)



  #print(len(new_data[0]))

  final_data.append(new_data)



# X = np.array(final_data)

# Y = np.array(labels_final)

# # print(X.shape)

# # print(Y.shape)



# train_pct_index = int(0.9*len(X))



# x_train, x_test = X[:train_pct_index], X[train_pct_index:]

# y_train, y_test = Y[:train_pct_index], Y[train_pct_index:]



# dtr = DecisionTreeRegressor()

# print("X)")

# dtr.fit(x_train, y_train)

# y_pred1 = dtr.predict(x_test)



# print("DT")

# val = sqrt(mean_squared_error(y_test, y_pred1))

# print(val)

# pred_rmse.append(val)

    



#
print(len(labels_final[0]))

print(len(final_data[0][0]))
print(len(final_data))

print(len(final_data[0]))
df_age1 = df4[df4['Agent'] == agent_number]

print(df_age1.shape[0])
final_data_test = []

labels_final_test = []

print("QQ")

for agent_number in range(7):

  print(agent_number)

  df_agent = df4_test[df4_test['Agent'] == agent_number]

  print(df_agent)

  df_age1 = df4[df4['Agent'] == agent_number]

  df_age1 = df_age1.drop(columns=['Agent', 'label'])

  leng = df_age1.shape[0]

  #print(leng-window_size)

  #print(leng)

  #df_agent_label = df_agent[['label']].values

  df_agent = df_agent.drop(columns=['Agent'])

  #defining deque

  li = deque()



  for i in range(leng - window_size, leng):

    row = df_age1.iloc[i].tolist()

    #print(row)

    li.extend(row)

  #making new data format

  #print("a")

  #print(len(li))

  new_data = []

  labels = []

  for i in range(0, df_agent.shape[0]):

    new_row = []

    values = df_agent.iloc[i].tolist()

    new_row.extend(li)

    new_row.extend(values)

    #new_row.append(agent_number)

    #labels.append(df_agent_label[i][0])

    for j in range(40):

      li.popleft();

    li.extend(values)

    new_data.append(new_row)

    #print(df_agent_label[i])

    #new_data.extend(df_agent_label[i])

  #labels_final.extend(labels)



  #print(len(new_data[0]))

  final_data_test.append(new_data)
print(len(final_data_test))

print(len(final_data_test[0]))
model_list = []



from sklearn.ensemble import RandomForestRegressor



for i in range(7):

  print(i)

  x_data = final_data[i]

  y_data = labels_final[i]



  X = np.asarray(x_data)

  Y = np.asarray(y_data)

  print(X.shape)

  print(Y.shape)



  # train_pct_index = int(0.8*len(X))



  # x_train, x_test = X[:train_pct_index], X[train_pct_index:]

  # y_train, y_test = Y[:train_pct_index], Y[train_pct_index:]

  

  x_train = X

  y_train = Y



  # print(len(x_data))

  # print(len(y_data))

  rf = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, warm_start=True)

  rf.fit(x_train, y_train)



  # y_pred2 = rf.predict(x_test)



  # from sklearn.metrics import r2_score

  # print("RF")

  # print(sqrt(mean_squared_error(y_test, y_pred2)))

  # print(r2_score(y_test, y_pred2))



  # y_train_pred2 = rf.predict(x_train)

  # print(sqrt(mean_squared_error(y_train, y_train_pred2)))

  # print(r2_score(y_train, y_train_pred2))

  model_list.append(rf)
predictions = []



for i in range(7):

  print(i)

  model = model_list[i]

  x_test = final_data_test[i]

  y_pred = model.predict(x_test)

  predictions.extend(y_pred)
print(len(predictions))
print(len(final_data_test))

print(len(final_data_test[0]))
print(len(final_data[0]))
X = np.array(final_data)

Y = np.array(labels_final)

print(X.shape)

print(Y.shape)



train_pct_index = int(0.8*len(X))



x_train, x_test = X[:train_pct_index], X[train_pct_index:]

y_train, y_test = Y[:train_pct_index], Y[train_pct_index:]

#train_test_split(X1, Y, test_size = 0.1)
#Linear regression



from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score



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

# print(X.shape)

# print(Y.shape)



train_pct_index = int(0.8*len(X))



x_train, x_test = X[:train_pct_index], X[train_pct_index:]

y_train, y_test = Y[:train_pct_index], Y[train_pct_index:]
X = np.array(final_data)

Y = np.array(labels_final)

x_train = X

y_train = Y
#Randomforest

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, warm_start=True, max_leaf_nodes=30000)

rf.fit(x_train, y_train)
y_pred2 = rf.predict(x_test)



from sklearn.metrics import r2_score

print("RF")

print(sqrt(mean_squared_error(y_test, y_pred2)))

print(r2_score(y_test, y_pred2))



y_train_pred2 = rf.predict(x_train)

print(sqrt(mean_squared_error(y_train, y_train_pred2)))

print(r2_score(y_train, y_train_pred2))

from sklearn.neighbors import KNeighborsRegressor



knr = KNeighborsRegressor()

knr.fit(x_train, y_train)
y_pred2 = knr.predict(x_test)



from sklearn.metrics import r2_score

print("RF")

print(sqrt(mean_squared_error(y_test, y_pred2)))

print(r2_score(y_test, y_pred2))



y_train_pred2 = knr.predict(x_train)

print(sqrt(mean_squared_error(y_train, y_train_pred2)))

print(r2_score(y_train, y_train_pred2))

#final_data_set need to be of a different format



#right now it is 0000000000 11111 2222 3333 4444



# we need 0 1 2 3 4 

#5756 entries per agent



final_data_test_reshuffled = []



for i in range(5756):

  for j in range(7):

    #print(j*5756 + i)

    final_data_test_reshuffled.append(final_data_test[j*5756 + i])



print(len(final_data_test_reshuffled))

print(len(final_data_test_reshuffled[0]))
y_final_reshuffled = []



for i in range(5756):

  for j in range(7):

    #print(j*5756 + i)

    y_final_reshuffled.append(predictions[j*5756 + i])



print(len(y_final_reshuffled))

# print(len(y_final_reshuffled[0]))
x_test = final_data_test_reshuffled

y_pred_test = rf.predict(x_test)

print(len(y_pred_test))
result = pd.DataFrame({'label':y_final_reshuffled})

result['id'] = ""

cols = ['id', 'label']

result = result[cols]
for index, row in result.iterrows():

  result.at[index, 'id'] = index + 1
print(result)
result.to_csv("/content/drive/My Drive/ML Lab/lab1/result.csv", index=False)
y_train_pred2 = rf.predict(x_train)

print(sqrt(mean_squared_error(y_train, y_train_pred2)))

print(r2_score(y_train, y_train_pred2))
print(rf.n_features_)

print(rf.n_outputs_)
data_output_x = []

for i in range(4, 9):

  data_output_x.append(i)



for i in range(7):

  plt.plot(data_output_x, data_output[i])

  plt.show()
for agent_number in range(7):

  df_agent = df3[df3['Agent'] == agent_number]

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
df_agent1 = df3[df3['Agent'] == 1]

print(df_agent1)
for item in df_agent1.columns:

  y = df_agent1[item].values

  x = df_agent1['time']

  plt.plot(x, y)

  plt.show()

  

#This shows that the pca's values are still dependent on time

#hence we can go forward with our analysis
#assuming its a queue



 

  

# Initializing a queue 

li = deque()



for i in range(5):

  row = df3.iloc[i].tolist()

  li.extend(row)



#in li the first 200 elements are the first 5 rows of df

#li first 40 are the x-5 the values

#li last 40 values are from previous row
df3.shape
new_data = []



for i in range(5, 161168):

  new_row = []

  values = df3.iloc[i].tolist()

  new_row.extend(li)

  new_row.extend(values)

  for j in range(40):

    li.popleft();

  li.extend(values)

  new_data.append(new_row)

transformed_df = pd.DataFrame.from_records(new_data)
df_agent = df1['Agent']

df_labels = df1['label']



df_agent = labels
#sliding_matrix will be a 5 rows 40 columns matrix

r, c = 5, 40;

sliding_matrix = [[0 for x in range(c)] for y in range(r)] 
#we need to initially fill this sliding matrix

for i in range(5):

  for j in range(40):

    sliding_matrix[i][j] = df3.at[i, j]
for i in range(5):

  print(sliding_matrix[i])
df3 = df3.drop(columns=['Agent', 'time'])
#copy the values

for i in range(5, 161168):

  for j in range(0, 40):

      for j in range(1, 6):

        #print(df3[index-j][i])

        df3[index][str(i) + "_" +str(j)] = df3[index-j][i]

  print(df3.iloc[[index]])
print(df3.iloc[[5]])
#fit lr model

y = df1['label']

x = df3.values



#there are a total of 161168 rows.

#we can assume a total of 20 cross validation steps for the model



from sklearn.model_selection import KFold



kf = KFold(n_splits=20)

kf.get_n_splits(x)



print(kf)
#figure out KNN' k elbow method

from sklearn.model_selection import train_test_split

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)



from sklearn.neighbors import KNeighborsRegressor



error_rate = []



for i in range(1, 10):

  knn = KNeighborsRegressor(n_neighbors=i)

  knn.fit(x_train, y_train)

  y_pred = knn.predict(x_test)

  error_rate.append(mean_squared_error(y_test, y_pred))
print(error_rate)
plt.plot(range(1,10),error_rate,color='blue', linestyle='dashed', marker='o',

 markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')



#Pretty clear from the graph that k = 3
knn = KNeighborsRegressor(n_neighbors=3)
from sklearn.linear_model import LinearRegression

from math import sqrt





lr = LinearRegression()



for train_index, test_index in kf.split(x):

  x_train, x_test = x[train_index], x[test_index]

  y_train, y_test = y[train_index], y[test_index]

  knn = KNeighborsRegressor(n_neighbors=3)

  knn.fit(x_train, y_train)

  y_pred = knn.predict(x_test)

  print("rms error:")

  print(sqrt(mean_squared_error(y_test, y_pred)))

  print("mean absolute error:")

  print(mean_absolute_error(y_test, y_pred))

  print("#")




