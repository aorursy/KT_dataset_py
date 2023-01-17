# import libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sb



from matplotlib import rcParams

%matplotlib inline

rcParams['figure.figsize'] = 5, 4

sb.set_style('whitegrid')
# Import data

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# take a look at dataset

dataF = pd.read_csv('../input/HR_comma_sep.csv')

df=dataF

columns_names=df.columns.tolist()

print(columns_names)
df.head()


leftData=df['left']

df.info()
# first lets check what output the sales aand salary can have in the databse

Sal=df['sales'].unique()

Uni=df['salary'].unique()

print(Sal, Uni)
# conversion of string to dummy numbers

from sklearn.preprocessing import LabelEncoder # For change categorical variable into int

from sklearn.metrics import accuracy_score 



le=LabelEncoder()

df['salary']=le.fit_transform(df['salary'])

df['sales']=le.fit_transform(df['sales'])

df.corr()
# ploting heatmap for better visualization ( vmax changes the limit of the heatmap)

correlation = df.corr()

plt.figure(figsize=(15,15))

sb.heatmap(correlation, vmax=1, square=True,annot=True,linewidths=.5, cmap="coolwarm")



plt.title('Correlation between fearures')
# calclute the eigen values

# first normalize the dataset

from sklearn.preprocessing import StandardScaler

df_norm = StandardScaler().fit_transform(df)



# then we need to calculate the covarience matrix.

meanValue = np.mean(df_norm, axis=0)

covarMatrix = (df_norm - meanValue).T.dot((df_norm - meanValue)) / (df_norm.shape[0]-1)



eign_vals, eign_vecs = np.linalg.eig(covarMatrix)

print('Eigenvalues are as follow \n%s' %eign_vals)

print('Eigenvectors are as follow \n%s' %eign_vecs)



# now lets sort the eignevalues 

#sorted_values=sorted(eign_vals)

#print(sorted_values)



# Make a list of (eigenvalue, eigenvector) tuples

eig_pairs = [(np.abs(eign_vals[i]), eign_vecs[:,i]) for i in range(len(eign_vals))]



# Sort the (eigenvalue, eigenvector) tuples from high to low

eig_pairs.sort()

eig_pairs.reverse()



# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in descending order:')

for i in eig_pairs:

    print(i[0])
tot = sum(eign_vals)

var_exp = [(i / tot)*100 for i in sorted(eign_vals, reverse=True)]

cum_var_exp = np.cumsum(var_exp)

#print(var_exp)

#print(cum_var_exp)

with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(10, 10))



    plt.bar(range(10), var_exp, alpha=0.5, align='center',

            label='individual explained variance')

    plt.step(range(10), cum_var_exp, where='mid',

             label='cumulative explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
u,s,v = np.linalg.svd(df_norm.T)

print(u)
matrix_w = np.hstack((eig_pairs[0][1].reshape(10,1),

                      eig_pairs[1][1].reshape(10,1)))



print('Matrix W:\n', matrix_w)
Y = df_norm.dot(matrix_w)

from sklearn.model_selection import train_test_split

label = df.pop('left')

data_train, data_test, label_train, label_test = train_test_split(df, label, test_size = 0.2, random_state = 42)
#SVM ML



from sklearn.svm import SVC

svm = SVC()

svm.fit(data_train, label_train)

svm_score_train = svm.score(data_train, label_train)

print("Training score: ",svm_score_train)

svm_score_test = svm.score(data_test, label_test)

print("Testing score: ",svm_score_test)
# random Forest ML

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(data_train, label_train)



rf_score_train = rf.score(data_train, label_train)

print("Training score: ",rf_score_train)

rf_score_test = rf.score(data_test, label_test)

print("Testing score: ",rf_score_test)



clf = rf
# Decision Tree	ML

from sklearn import tree

dt = tree.DecisionTreeClassifier()

dt.fit(data_train, label_train)

dt_score_train = dt.score(data_train, label_train)

print("Training score: ",dt_score_train)

dt_score_test = dt.score(data_test, label_test)

print("Testing score: ",dt_score_test)
# Logistic Regression Ml

from sklearn.linear_model import LogisticRegression

logis = LogisticRegression()

logis.fit(data_train, label_train)

logis_score_train = logis.score(data_train, label_train)

print("Training score: ",logis_score_train)

logis_score_test = logis.score(data_test, label_test)

print("Testing score: ",logis_score_test)
# KNN Regression Ml

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(data_train, label_train)

knn_score_train = knn.score(data_train, label_train)

print("Training score: ",knn_score_train)

knn_score_test = knn.score(data_test, label_test)

print("Testing score: ",knn_score_test)
# Gaussian Naive Bays	

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(data_train,label_train)

GaussianNB_score_train = gnb.score(data_train, label_train)

print("Training score: ",GaussianNB_score_train)

GaussianNB_score_test = gnb.score(data_test, label_test)

print("Testing score: ",GaussianNB_score_train)
models = pd.DataFrame({

        'Model'          : ['SVM', 'random Forest', 'Decision Tree', 'Logistic Regression', 'KNN Regression','Gaussian Naive Bays'],

        'Testing_Score' : [svm_score_test, rf_score_test, dt_score_test, logis_score_test, knn_score_test, GaussianNB_score_test],

    })

models.sort_values(by='Testing_Score', ascending=False)
indices = np.argsort(rf.feature_importances_)[::-1]

# Print the feature ranking

print('variables ranking:')



for f in range(df.shape[1]):

    print('%d. feature %d %s (%f)' % (f+1 , indices[f], df.columns[indices[f]],

                                      rf.feature_importances_[indices[f]]))
# these are some known info about the past

print('# of people left = {}'.format(dataF[leftData==1].size))

print('# of people stayed = {}'.format(dataF[leftData==0].size))

#  information about furure, prediction

dataF = pd.read_csv('../input/HR_comma_sep.csv')

dfC = pd.get_dummies(dataF)

#dfC.info()

leave = dfC[dfC['left'] == 1]

leaveC = pd.get_dummies(leave)

            

df1 = leaveC

y = df1['left'].values

df1 = df1.drop(['left'],axis=1)

X = df1.values

pred = clf.predict_proba(X[:, :9])





# number of employees that definitely are leaving

sum(pred[:,1]==1)

# Who would likely will leave (probablity of greater than 0.5)

leave['leavingSoon'] = pred[:,1]



leave[leave['leavingSoon']>=0.5]