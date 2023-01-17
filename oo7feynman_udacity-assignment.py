import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cross_validation import train_test_split as Tts
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#Visualization Libraries
import seaborn as sns
from matplotlib import pyplot as plt

import os
print(os.listdir("../input"))
customers=pd.read_csv(r'../input/customers.csv')
def SampleData(data):
    display(data.head())
    
SampleData(customers)

display(customers.describe())
import tqdm
for columns in tqdm.tqdm(customers.columns):
    print("Missing Data point check for {} column".format(columns), end=' >> ') 
    #pdb.set_trace()
    if (customers[columns].get_dtype_counts()).nunique() and customers[columns].isnull().sum()==0:
        print ("No Missing Value in {} column".format(columns))
    else:
        print("Missing Values were identified in {} column".format(column)) 
        print("Unique Values for {} column".format(columns))
        print(customers[columns].unique(),'\n')
        customers[columns]=np.where(customers[columns]=='',np.NaN,customers[columns])
        print("Processing Complete for {} column".format(columns))
RelevantColumns=[]
IrrelevantColumns=[]
for feature_dropped in tqdm.tqdm(customers.columns):
    print("Dropping Column - {}".format(feature_dropped))
    data=customers.copy()
    new_data = data.drop(feature_dropped,axis=1)
    labels=data[feature_dropped]
    X_train, X_test, y_train, y_test = Tts(new_data, labels, test_size=0.25, random_state=30)
    regressor = DecisionTreeRegressor(random_state=30)
    regressor.fit(X_train,y_train)
    score = regressor.score(X_test,y_test)
    print("R^2 Score for column {} is {}".format(feature_dropped,score))
    if score<0:
        print("Column {} is necesssary and we will lose relevant information without it".format(feature_dropped))
        RelevantColumns.append(feature_dropped)
    else:
        print("Column {} can be dropped".format(feature_dropped))
        IrrelevantColumns.append(feature_dropped)
    print("\n")
print("All Columns - ",list(customers.columns))
print("Relevant Columns - ",RelevantColumns)
print("Columns that can be dropped - ",IrrelevantColumns)
%matplotlib inline
customers.drop(['Region', 'Channel'], axis = 1, inplace = True)
pd.scatter_matrix(customers, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
#Correlation Matrix
f, ax = plt.subplots(figsize=(9, 6))
f.suptitle('Correlation matrix for categories')
sns.heatmap(customers.corr(), annot=True, linewidths=.5, ax=ax);
customers.corr().abs().unstack().sort_values(ascending=False)
customers.skew()
log_transformation=np.log(customers)
ss = StandardScaler()
ss.fit(log_transformation)
customers_log_transformation = pd.DataFrame(ss.transform(log_transformation),columns=log_transformation.columns)
pd.scatter_matrix(customers_log_transformation, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
pd.set_option('display.float_format', lambda x: '%.3f' % x)
customers_log_transformation.describe()
customers.describe()
def TukeyMethod(df,column):
    q1 = np.percentile(df[column],25)
    q3 = np.percentile(df[column],75)
    step = (q3-q1)*1.5
    iqr = (df[column] >= q1-step) & (df[column] <= q3+step)
    (print("Outlier Identified for columns - {}".format(column)))
    display(df.loc[~iqr].head())
    return df.loc[iqr]

for i,columns in enumerate(customers_log_transformation.columns):
    customers_log_transformation=TukeyMethod(customers_log_transformation,columns)
    
print("Outlier Removed by Using Tukey's Method")   
pca_base = PCA(n_components=len(customers_log_transformation.columns))
x_base = pca_base.fit_transform(customers_log_transformation)
Basedf = pd.DataFrame(data = x_base)
print(Basedf.head())
pca = PCA(n_components=2)
x_new = pca.fit_transform(customers_log_transformation)
ReducedDf = pd.DataFrame(data = x_new)
print(ReducedDf.head())
def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, customers_log_transformation.columns[i], color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

plt.figure(figsize=(8, 8), dpi=80)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("Principal Component {}".format(1))
plt.ylabel("Principal Component {}".format(2))
plt.grid()

#Call the function. Use only the 2 PCs.
myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]))
plt.show()
print("Explained variance - {} %".format((100*pca.explained_variance_ratio_)))
print("Explained variance for the two components {} %".format(sum(100*pca.explained_variance_ratio_)))
print("Eigen Vectors",pca.components_)
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(x_new).score(x_new) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
kmeans=KMeans(n_clusters=2)
kmeansoutput=kmeans.fit(x_new)
centers = np.array(kmeansoutput.cluster_centers_)
plt.figure('K-Means Cluster')
plt.scatter(x_new[:,0:1],x_new[:,1:2], c=pd.DataFrame(kmeansoutput.labels_))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.scatter(centers[:,0], centers[:,1], marker="x", color='r')
plt.title('K-Means Cluster')
plt.show()
log_centers = ss.inverse_transform(X=pca.inverse_transform(kmeansoutput.cluster_centers_))
true_centers = np.exp(log_centers)
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = customers_log_transformation.columns)
true_centers.index = segments
display(true_centers)
idx = np.random.randint(200, size=10)

for i, pred in enumerate(kmeansoutput.predict(x_new[idx,:])):
    print ("Sample point {}\n {} predicted to be in Cluster {}".format(str(i),str(np.exp(ss.inverse_transform(X=pca.inverse_transform(x_new[idx,:][i])))),str(pred)))
print("Code Run Completed")