import pandas as pd

pd.set_option('display.max_rows',500)

import numpy as np

from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

import seaborn as sns
def change_to_number(column,data_frame):

    column_list = data_frame[column].unique().tolist()

    column_dict = dict()

    for index, attribute in enumerate(sorted(column_list)) :

        column_dict[attribute] = index

    data_frame[column].replace(column_dict,inplace=True)

    return data_frame
def give_mode_value(column,data_frame):

    

    mode_val = data_frame[column].mode()[0]

    data_frame[column].replace({'?' : mode_val},inplace=True)

    

    return change_to_number(column,data_frame)
def give_int_value(column,data_frame) :

    

    data_frame[column].replace({'?' : np.nan},inplace=True)

    data_frame[column].fillna(str(data_frame[column].astype('float64').mean()),inplace=True)

    return change_to_number(column,data_frame)
df = pd.read_csv('../input/dataset.csv')

df.head()
col_to_be_mode=['Account1','History','Motive','Credit1','InstallmentRate','Tenancy Period']

col_to_be_mean=['Monthly Period','InstallmentCredit','Yearly Period','Age']

col_to_change_number=['Account2', 'Employment Period', 'Gender&Type','Sponsors', 'Tenancy Period', 'Plotsize', 'Plan', 'Housing',

       '#Credits', 'Post', '#Authorities', 'Phone', 'Expatriate']

for abc in col_to_be_mean:

    df = give_int_value(abc,df)

for abc in col_to_be_mode:

    df = give_mode_value(abc,df)

for abc in col_to_change_number:

    df = change_to_number(abc,df)

    
list_of_cols = df.columns.tolist()

for col in list_of_cols :

    try :

        df[col] = df[col].astype('float64')

    except :

        pass
df.head()

plot_frame = df.drop(['id','Class'],1)
f, ax = plt.subplots(figsize=(10,8))

corr = plot_frame.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True)
%matplotlib notebook

fig = plt.figure()

ax = fig.add_subplot(111,projection='3d')

ax.scatter(df['Age'].astype('float64'),df['#Credits'].astype('float64'),df['Monthly Period'].astype('float64'))

ax.set_xlabel('#Credits')

ax.set_ylabel('Monthly Period')

ax.set_zlabel('Age')
%matplotlib notebook

fig = plt.figure()

ax = fig.add_subplot(111,projection='3d')

ax.scatter(df['InstallmentCredit'].astype('float64'),df['#Credits'].astype('float64'),df['Monthly Period'].astype('float64'))

ax.set_ylabel('#Credits')

ax.set_zlabel('Monthly Period')

ax.set_xlabel('InatCred')

X = df[['#Credits','Monthly Period','Age','#Authorities']]

X = pd.DataFrame(MinMaxScaler().fit_transform(X),columns=X.columns)

pca = PCA(n_components=2)

pca.fit(X)

T1 = pca.transform(X)
train_x = X[:175]

y_train = df['Class'][:175]

test_x = X[175:]
def cluster_to_class_label(cluster_list,alter_dict) :

    class_list = list()

    for e in cluster_list :

        class_list.append(alter_dict[e])

    return class_list
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,random_state=55,max_iter=1000).fit(train_x,y_train)

kmeans.cluster_centers_

train_preds = kmeans.predict(train_x)
plt.plot()

plt.title("Plotting Training clusters")

plt.scatter(T1[:, 0][:175], T1[:, 1][:175], c=train_preds)



centroids = kmeans.cluster_centers_

centroids = pca.transform(centroids)

plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3,\

        c = ['k', 'g', 'b'])
dict_list = [{0:0, 1:1, 2:2},{0:0, 1:2,2:1},{0:1,1:0,2:2},{0:1,1:2,2:0},{0:2,1:0,2:1},{0:2,1:1,2:0}]

best_dic = dict_list[0]

best_acc = 0

for dic in dict_list :

    train_modified_list = cluster_to_class_label(train_preds.tolist(),dic)

    train_acc = accuracy_score(train_modified_list,y_train.values.tolist())

    if train_acc > best_acc :

        best_acc = train_acc

        best_dic = dic

print('best training accuracy is %.6f'%best_acc)
test_preds = kmeans.predict(test_x)

test_values = test_preds.tolist()

plt.plot()

plt.title("Test clusters")

plt.scatter(T1[:, 0][175:], T1[:, 1][175:], c=test_preds)



centroids = kmeans.cluster_centers_

centroids = pca.transform(centroids)

plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3,\

        c = ['k', 'g', 'b'])



id_values= df['id'].values.tolist()[175:]

columns = ['id','class']

df_sub = pd.DataFrame({'id' : id_values,'class' : test_values})

df_sub.to_csv('submission.csv')
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(df)