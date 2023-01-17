#IMPORTING ALL THE NEEDED LIBRARIES



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler, StandardScaler



from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score
#IMPORTING THE DATASET AND CONVERTING ALL '?' TO NA VALUES



data_orig = pd.read_csv("../input/dataset.csv", sep=',',na_values = '?')

data = data_orig
data
#CHECKING NUMBER OF UNIQUE VALUES FOR EACH FEATURE



for column in data:

    print(column + " : " + str(data[column].nunique()))
#CREATING A DIFFRENT DF FOR CLASS COLUMN



Class_column = data[['id','Class']]

Class_column
#LOOKING FOR REDUNDANT VALUES IN THE COLUMNS, IDENTIFIED A FEW



for col in data.columns:

    if(data[col].nunique()<=50):

        print("Column Name: "+col + " Values: ",data[col].unique())

        print()
#FIXING THE REDUNDANCY BY MERGING SIMILAR VALUES USING REPLACE FUNCTION



data['Plotsize'].replace('me', 'ME',inplace =True)

data['Plotsize'].replace('M.E.', 'ME',inplace =True)

data['Plotsize'].replace('sm', 'SM',inplace =True)

data['Plotsize'].replace('la', 'LA',inplace =True)

data['Account2'].replace('Sacc4', 'sacc4',inplace =True)

data['Sponsors'].replace('g1', 'G1',inplace =True)
#COLUMN VALUES AFTER FIXING THE REDUNDANCIES



for col in data.columns:

    if(data[col].nunique()<=50):

        print("Column Name: "+col + " Values: ",data[col].unique())

        print()
#CORRELATION TEST TO SEE WHICH FURTHER COLUMNS CAN BE REMOVED



f, ax = plt.subplots(figsize=(10, 8))

corr = data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax, annot = True);


data = data.drop(['Class'],1)

data.info()
#REMOVING HIGHLY CORRELATED COLUMNS AND REMOVING ID COLUMN



data = data.drop(['id','Credit1','Yearly Period'], 1) 
#CHECKING NUMBER OF UNIQUE VALUES IN EACH COLUMN TO DECIDE WHICH COLUMN IS CATEGORICAL AND WHICH IS CONTINUOUS



for column in data:

    print(column + " : " + str(len(data[column].unique())))
#FILLING NA VALUES OF CATEGORICAL COLUMNS WITH MODE



a = ['Account1','History','Motive','InstallmentRate','Tenancy Period']

for i in a:

    data[i].fillna(data[i].mode()[0] ,inplace= True)
data.info()
#FILLING NA VALUES OF CONTINUOUS COLUMNS WITH MEDIAN



b = ['Monthly Period','Age','InstallmentCredit']

for i in b:

    data[i].fillna(data[i].median(),inplace= True)
data.info()
#DROPPING DUPLICATES IF ANY



data = data.drop_duplicates()

data.info()
#LABEL ENCODING



le = LabelEncoder()

datacopyoh = data.copy()

data_copy = data.copy()



for col in data.columns:

    if(data[col].dtype == np.object):

        le.fit(data[col])

        data[col] = le.transform(data[col])
#ONE HOT ENCODING



oh_cols = []

for col in datacopyoh.columns:

    if(datacopyoh[col].dtype == np.object):

        oh_cols.append(col)



datacopyoh = pd.get_dummies(datacopyoh, columns=oh_cols)
#SCALING THE LABEL-ENCODED FEATURES USING MINMAXSCALER



target = Class_column['Class']

sc = MinMaxScaler()

data_sc = pd.DataFrame(sc.fit_transform(data),columns = data.columns)

data_sc['Class'] = target
#SCALING THE ONE-HOT-ENCODED FEATURES USING MINMAXSCALER



target = Class_column['Class']

sc = MinMaxScaler()

datacopyoh_sc = pd.DataFrame(sc.fit_transform(datacopyoh),columns = datacopyoh.columns)

datacopyoh_sc['Class'] = target
#DEFINING A FUNCTION TO MEASURE ACCURACY OF THE MODEL



def measuring_accuracy(predicted,target):

    best_accuracy = 0 

    val = pd.DataFrame(predicted[:175],columns=['temp'])

    target = target.loc[:174]

    combinations = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]

    predicted = pd.DataFrame(predicted,columns=['temp'])

    output = predicted

    ite=0

    

    for i,comb in enumerate(combinations):

        pr_temp = val['temp'].apply(lambda x: comb[0] if x==0 else comb[1] if x==1 else comb[2])

        acc_temp = accuracy_score(pr_temp,target)

        if(acc_temp>best_accuracy):

            best_accuracy = acc_temp

            output = predicted['temp'].apply(lambda x: comb[0] if x==0 else comb[1] if x==1 else comb[2])

            ite=i

            

    return best_accuracy*100,output.as_matrix()
datacopyoh_sc.columns
#USING K-MEANS TO FORM CLUSTERS



target1 = datacopyoh_sc['Class']

df_main = data_copy.copy()



datacopyoh_sc = datacopyoh_sc.loc[:, datacopyoh_sc.columns != 'Class']



#USING ONLY THE IDENTIFIED FEATURES WHICH GIVE BEST RESULTS



datacopyoh_sc = datacopyoh_sc[['InstallmentRate', 'Age', 'Monthly Period', 'Account1_aa', 'Account1_ab',

       'Account1_ac', 'Account1_ad', 'History_c0', 'History_c1', 'History_c2',

       'History_c3', 'History_c4', 'Plotsize_SM', 'Plotsize_ME',

       'Plotsize_LA', 'Plotsize_XL', 'Housing_H1', 'Housing_H2',

       'Housing_H3']]



#USING PCA TO REDUCE THE FEATURES TO 2 COMPONENTS FOR VISUALIZING



pca = PCA(n_components=2)

pca.fit(datacopyoh_sc)

T1 = pca.transform(datacopyoh_sc)



#USING K_MEANS WITH 3 CLUSTERS



predictedvalues = KMeans(n_clusters = 3, random_state = 42).fit_predict(datacopyoh_sc)

acc,k = measuring_accuracy(predictedvalues,target1)



#PLOTTING THE RESULTS



plt.title(" Accuracy = " + str(acc))

plt.scatter(T1[:, 0], T1[:, 1], c=k)



plt.show()
#COPYING THE RESULTS TO A CSV FILE FOR SUBMISSION



if 'Class' in data_orig.columns:

    data_orig.drop(['Class'],axis=1,inplace=True)

data_orig['Class'] = k

data_orig[['id','Class']].loc[175:].to_csv("submission15.csv",index=False)
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



create_download_link(data_orig[['id','Class']].loc[175:])