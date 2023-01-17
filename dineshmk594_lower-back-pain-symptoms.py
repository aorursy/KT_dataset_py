import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv('../input/Dataset_spine.csv')
data.info()
data.head(2).T
data=data.drop('Unnamed: 13',axis=1)
data=data.rename({'Class_att':'Dependent variable',
                   'Col1':'pelvic_incidence  (numeric)',  
                   'Col2':'pelvic_tilt  (numeric)',  
                   'Col3':'lumbar_lordosis_angle  (numeric)',  
                   'Col4':'sacral_slope  (numeric)',  
                   'Col5':'pelvic_radius  (numeric)',  
                   'Col6':'degree_spondylolisthesis  (numeric)',  
                   'Col7':'pelvic_slope(numeric)',  
                   'Col8':'Direct_tilt(numeric)',  
                   'Col9':'thoracic_slope(numeric)',  
                   'Col10':'cervical_tilt(numeric)', 
                   'Col11':'sacrum_angle(numeric)', 
                   'Col12':'scoliosis_slope(numeric)'},axis=1)
plt.title('Lower Backpain symptoms target class distribution')
plt.pie(data['Dependent variable'].value_counts(),
        labels=data['Dependent variable'].value_counts().index,
        autopct='%.2f',
       explode=[0,0.05],startangle=90)
plt.show()
data.hist(figsize=(15,10),grid=False)
plt.show()
data.describe().T
data.plot(kind='box',subplots=True,layout=(4,4),figsize=(15,8))
plt.show()
data=pd.get_dummies(data,drop_first=True)
plt.figure(figsize=(10,8))
sns.heatmap(data.corr()[data.corr().abs()>0.3],annot=True)
data.info()
sns.lmplot( x="pelvic_incidence", y="pelvic_radius", hue="Dependent variable_Normal", data=data)
sns.lmplot( x="pelvic_incidence", y="degree_spondylolisthesis", hue="Dependent variable_Normal", data=data)
sns.lmplot( x="pelvic_incidence", y="lumbar_lordosis_angle", hue="Dependent variable_Normal", data=data)
data.corr()['Dependent variable_Normal'].sort_values(ascending=False)[1:].plot.barh(title='Correlation b/w dependent & Independent Variable')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore
PCA_x=data.drop('Dependent variable_Normal',axis=1) # dropping target column
# Standardisation
sc = StandardScaler()
X_std =  sc.fit_transform(PCA_x)          
cov_matrix = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
print('Eigen Vectors \n%s', eig_vecs)
print('\n Eigen Values \n%s', eig_vals)
tot = sum(eig_vals)
var_exp = [( i /tot ) * 100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print("Cumulative Variance Explained", cum_var_exp)
pd.DataFrame(cum_var_exp).T.plot.bar(title='Cumulative variance at each variable')
PCA_x_std = sc.fit_transform(PCA_x)
bpc_reduced = PCA(n_components=7).fit_transform(PCA_x_std)
BP_reduced=pd.DataFrame(bpc_reduced)
BP_reduced.head()
sns.heatmap(BP_reduced.corr()[BP_reduced.corr().abs()>0.3],annot=True)
from sklearn.neighbors import KNeighborsClassifier
X =  BP_reduced
y =  data[["Dependent variable_Normal"]]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=1)
modelKNN=KNeighborsClassifier(n_neighbors=4)
modelKNN=modelKNN.fit(X_train,Y_train)
print('train score',modelKNN.score(X_train,Y_train))
print('test score',modelKNN.score(X_test,Y_test))
from sklearn.tree import DecisionTreeClassifier
data.corr()[data.corr().abs()>0.2]['Dependent variable_Normal'].sort_values(ascending=False)[1:7]
x_features=data.corr()[data.corr().abs()>0.2]['Dependent variable_Normal'].sort_values(ascending=False)[1:7].index
X =  data[x_features]
y =  data[["Dependent variable_Normal"]]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=1)
dt_model = DecisionTreeClassifier()
dt_model=dt_model.fit(X_train,Y_train)
dt_model.score(X_train,Y_train)
# Reguralisaction
dt_model = DecisionTreeClassifier(max_depth=2)
dt_model=dt_model.fit(X_train,Y_train)
print('Train accuracy',dt_model.score(X_train,Y_train))
print('Test accuracy',dt_model.score(X_test,Y_test))
from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression()
lr_model=lr_model.fit(X_train,Y_train)
print('Train Accuracy',lr_model.score(X_train,Y_train))
print('Test Accuracy',lr_model.score(X_test,Y_test))
lr_model=LogisticRegression(penalty='l1')
lr_model=lr_model.fit(X_train,Y_train)
print('Train Accuracy',lr_model.score(X_train,Y_train))
print('Test Accuracy',lr_model.score(X_test,Y_test))