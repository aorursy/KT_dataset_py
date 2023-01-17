import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_excel('../input/application-inter/application_inter_final1.xlsx')

ks=df
ks=ks.replace(to_replace=['มากที่สุด','ปานกลาง','น้อย','ไม่มี'], value=[3.,2.,1.,0.], regex=True)
ks=ks.replace(to_replace=['กทม','ภาคกลาง','ภาคอีสาน','ภาคใต้','ภาคตะวันออก','ภาคเหนือ'],
              value=['Bkk', 'Central','NE','South','east','north'], regex=True)
#drop na data
ks=ks.dropna() 
ks = ks[ks.channal != 'na']

#encoder feature
from sklearn.preprocessing import LabelEncoder

cat_features = ['Status','location', 'source','channal']
encoder = LabelEncoder()

# Apply the label encoder to each column
encoded = ks[cat_features].apply(encoder.fit_transform)
encoded.head(10)

data = ks[['GPA','english','job', 'social', 'inter_study', 'exchange','famous','tuition_fee','ranking'
           ,'facility_classroom', 'self_influ', 'parrent_influ', 'brother_influ',
           'relativd_influ', 'senior_influ','income']].join(encoded)

cor=data.corr()
plt.subplots(figsize=(8,8));
cmap = sns.diverging_palette(220, 10, as_cmap=True)
mask = np.triu(np.ones_like(cor, dtype=np.bool))

sns.heatmap(cor,square=True,annot=False, mask=mask,cmap=cmap);
#split test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = data.drop(columns=['Status'])
y = data[['Status']]
scaler.fit(X)
X=scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)
#keras prediciton
import keras
from keras.layers import Dropout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error




model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(19,)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(Dropout(0.2)) 
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(Dropout(0.2)) 
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(Dropout(0.2)) 
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))






#output
model.add(keras.layers.Dense(1, activation= 'relu'))

model.compile (loss='mean_squared_error',
               optimizer = 'adam')
#train
history=model.fit(X_train, y_train, epochs=20000,validation_data=(X_test,y_test), 
                  batch_size=20,verbose=0 )
test=model.evaluate(X_test,y_test,verbose=0) 

print('error=', test*100, ' %')

#plot error
plt.plot(history.history['loss'], label='train')
plt.legend()
plt.show()
#evaluate results
predict=(model.predict(X_test, batch_size=1))
rmse = mean_squared_error(y_test, predict, squared=False)
print('RSME=', rmse)
plt.scatter(y_test, predict)
plt.xlabel('Actual')
plt.ylabel('Predict')
plt.show()

print('R_square=',R2)
from keras.models import load_model

model.save('admission.h5')  # creates a HDF5 file 'my_model.h5'
#decision tree
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import mean_squared_error


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
tree.plot_tree(clf.fit(X_train, y_train)) 
predict=(clf.predict(X_test))

rmse = mean_squared_error(y_test, predict, squared=False)
print("RMSE =", rmse)
import graphviz 

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("admission") 


from sklearn.ensemble import RandomForestClassifier
yt=y_train.to_numpy(dtype=int)
yt=np.ravel(yt)
clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(X_train, yt)
#tree.plot_tree(clf.fit(X_train, yt)) 
predict=(clf.predict(X_test))

rmse = mean_squared_error(y_test, predict, squared=False)
print("RMSE =", rmse)


from sklearn.neighbors import NearestCentroid
import numpy as np
clf = NearestCentroid()
clf.fit(X_train, yt)
predict=(clf.predict(X_test))
rmse = mean_squared_error(y_test, predict, squared=False)
print("RMSE =", rmse)

from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
clf = KMeans(n_clusters=2)
clf.fit(X_train)
centroids = clf.cluster_centers_
labels = clf.predict(X_train)
predict=(clf.predict(X_test))

rmse = mean_squared_error(y_test, predict, squared=False)
R2= r2_score(y_test, predict)
print("RMSE =", rmse)
print('R_square=',R2)
print(predict)
print(y_test)
import seaborn as sns
sns.set(style="ticks", palette="pastel")

fig, ax = plt.subplots()
fig.set_size_inches(8, 8)

ax=sns.boxplot(x="channal", y="GPA",
            hue="Status", palette=["m", "g"],
            data=ks,ax=ax)
sns.despine(offset=10, trim=True)
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(8, 8)

ax=sns.boxplot(x="income", y="GPA",
            hue="Status", palette=["m", "g"],
            data=ks,ax=ax)
sns.despine(offset=10, trim=True)
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(8, 8)

ax=sns.boxplot(x="channal", y="income",
            hue="Status", palette=["m", "g"],
            data=ks,ax=ax)
sns.despine(offset=10, trim=True)
plt.show()


fig, ax = plt.subplots()
fig.set_size_inches(8, 8)

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
ax=sns.boxplot(x="location", y="GPA",
            hue="Status", palette=["m", "g"],
            data=ks,ax=ax)
sns.despine(offset=10, trim=True)
plt.show();

fig, ax = plt.subplots()
fig.set_size_inches(8, 8)


ax=sns.boxplot(x="location", y="income",
            hue="Status", palette=["m", "g"],
            data=ks,ax=ax)
sns.despine(offset=10, trim=True)
plt.show();
#manage data
ks=ks.replace(to_replace=['Facebook, Web Site'], value=['Facebook'], regex=True)

f, ax = plt.subplots(figsize=(8, 5))
sns.countplot(y="source", data=ks,hue='Status',  palette="Set1");

fig, ax = plt.subplots()
fig.set_size_inches(15, 7)

plt.rcParams['font.family'] = ['sans-serif']
ax=sns.violinplot(x="source", y="income",
            hue="Status", palette=["m", "g"],
            data=ks,split=True,ax=ax)
sns.despine(offset=10, trim=True)
plt.show();

fig, ax = plt.subplots()
fig.set_size_inches(15, 7)


ax=sns.violinplot(x="source", y="GPA",
            hue="Status", palette=["m", "g"],
            data=ks,split=True,ax=ax)
sns.despine(offset=10, trim=True)
plt.show();

fig, ax = plt.subplots()
fig.set_size_inches(15, 7)


ax=sns.violinplot(x="source", y="exchange",
            hue="Status", palette=["m", "g"],
            data=ks,split=True,ax=ax)
sns.despine(offset=10, trim=True)
plt.show();
ks_plot=ks.drop(['income','GPA'],axis=1)

sns.set_style('ticks')
fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(8, 10)
sns.violinplot(data=ks_plot, orient="h",ax=ax)
sns.despine()
plt.show();

import plotly.express as px
df = ks_plot
fig = px.violin(df, y=('social'), x='location',color ='Status', box=True,
          hover_data=df.columns)
fig.show()

import plotly.express as px
df = ks_plot
fig = px.violin(df, y=('english'), x='location',color ='Status', box=True,
          hover_data=df.columns)
fig.show()

import plotly.express as px
df = ks_plot
fig = px.violin(df, y='famous', x='location',color ='Status', box=True,
          hover_data=df.columns)
fig.show()

import plotly.express as px
df = ks_plot
fig = px.violin(df, y='job', x='location',color ='Status', box=True,
          hover_data=df.columns)
fig.show()
import plotly.express as px
df = ks
fig = px.violin(df, y='GPA', x='source',color ='Status', box=True,
          hover_data=df.columns)
fig.show()
def box_plot_columns(df,categories_column,list_of_columns,legend_title,y_axis_title,**boxplotkwargs):
    columns = [categories_column] + list_of_columns
    newdf = df[columns].copy()
    data = newdf.melt(id_vars=[categories_column], var_name=legend_title, value_name=y_axis_title)
    return sns.violinplot(data=data, x=categories_column, y=y_axis_title, hue=legend_title, **boxplotkwargs)

    
fig, ax = plt.subplots(1,1)
fig.set_size_inches(15, 10)

ax = box_plot_columns(ks,"Status",
                      ["english","job",'social','ranking','famous','tuition_fee',
                       'ranking','conection','facility_lab','facility_classroom'],
                      "dataset","values",ax=ax)
ax.set_title("Admission")
plt.show()
def box_plot_columns(df,categories_column,list_of_columns,legend_title,y_axis_title,**boxplotkwargs):
    columns = [categories_column] + list_of_columns
    newdf = df[columns].copy()
    data = newdf.melt(id_vars=[categories_column], var_name=legend_title, value_name=y_axis_title)
    return sns.violinplot(data=data, x=categories_column, y=y_axis_title, hue=legend_title, **boxplotkwargs)

    
fig, ax = plt.subplots(1,1)
fig.set_size_inches(15, 10)

ax = box_plot_columns(ks,"Status",
                      ["self_influ","parrent_influ",'brother_influ','relativd_influ','senior_influ'],
                      "dataset","values",ax=ax)
ax.set_title("Admission")
plt.show()
ks['source']
X_train
