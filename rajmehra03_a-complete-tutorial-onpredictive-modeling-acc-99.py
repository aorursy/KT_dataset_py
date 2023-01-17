# Ignore  the warnings

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



# data visualisation and manipulation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

import missingno as msno



#configure

# sets matplotlib to inline and displays graphs below the corressponding cell.

%matplotlib inline  

style.use('fivethirtyeight')

sns.set(style='whitegrid',color_codes=True)



#import the necessary modelling algos.

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import datasets

from sklearn.naive_bayes import GaussianNB



#model selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score

from sklearn.model_selection import GridSearchCV



#preprocess.

from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer,LabelEncoder,OneHotEncoder
train=pd.read_csv(r'../input/voice.csv')
train.head(10)
df=train.copy()
df.head(10)
df.shape
df.index   
df.columns # give a short description of each feature.
# check for null values.

df.isnull().any()   
msno.matrix(df)  # just to visualize. no missing value.
df.describe()
def calc_limits(feature):

    q1,q3=df[feature].quantile([0.25,0.75])

    iqr=q3-q1

    rang=1.5*iqr

    return(q1-rang,q3+rang)
def plot(feature):

    fig,axes=plt.subplots(1,2)

    sns.boxplot(data=df,x=feature,ax=axes[0])

    sns.distplot(a=df[feature],ax=axes[1],color='#ff4125')

    fig.set_size_inches(15,5)

    

    lower,upper = calc_limits(feature)

    l=[df[feature] for i in df[feature] if i>lower and i<upper] 

    print("Number of data points remaining if outliers removed : ",len(l))



plot('meanfreq')
plot('sd')
plot('median')
plot('Q25')
plot('IQR')
plot('skew')
plot('kurt')
plot('sp.ent')
plot('sfm')
plot('meanfun')
sns.countplot(data=df,x='label')
df['label'].value_counts()
temp = []

for i in df.label:

    if i == 'male':

        temp.append(1)

    else:

        temp.append(0)

df['label'] = temp
#corelation matrix.

cor_mat= df[:].corr()

mask = np.array(cor_mat)

mask[np.tril_indices_from(mask)] = False

fig=plt.gcf()

fig.set_size_inches(30,12)

sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)
df.drop('centroid',axis=1,inplace=True)
# drawing features against the target variable.



def plot_against_target(feature):

    sns.factorplot(data=df,y=feature,x='label',kind='box')

    fig=plt.gcf()

    fig.set_size_inches(7,7)
plot_against_target('meanfreq') # 0 for females and 1 for males.
plot_against_target('sd')
plot_against_target('median')
plot_against_target('Q25')
plot_against_target('IQR')
plot_against_target('sp.ent')
plot_against_target('sfm')
plot_against_target('meanfun')  
g = sns.PairGrid(df[['meanfreq','sd','median','Q25','IQR','sp.ent','sfm','meanfun','label']], hue = "label")

g = g.map(plt.scatter).add_legend()
# removal of any data point which is an outlier for any fetaure.

for col in df.columns:

    lower,upper=calc_limits(col)

    df = df[(df[col] >lower) & (df[col]<upper)]
df.shape
df.head(10)
temp_df=df.copy()



temp_df.drop(['skew','kurt','mindom','maxdom'],axis=1,inplace=True) # only one of maxdom and dfrange.

temp_df.head(10)

#df.head(10)
temp_df['meanfreq']=temp_df['meanfreq'].apply(lambda x:x*2)

temp_df['median']=temp_df['meanfreq']+temp_df['mode']

temp_df['median']=temp_df['median'].apply(lambda x:x/3)
temp_df.head(10) 
sns.boxplot(data=temp_df,y='median',x='label') # seeing the new 'median' against the 'label'.
temp_df['pear_skew']=temp_df['meanfreq']-temp_df['mode']

temp_df['pear_skew']=temp_df['pear_skew']/temp_df['sd']

temp_df.head(10)
sns.boxplot(data=temp_df,y='pear_skew',x='label') # plotting new 'skewness' against the 'label'.
scaler=StandardScaler()

scaled_df=scaler.fit_transform(temp_df.drop('label',axis=1))

X=scaled_df

Y=df['label'].as_matrix()
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=42)
clf_lr=LogisticRegression()

clf_lr.fit(x_train,y_train)

pred=clf_lr.predict(x_test)

print(accuracy_score(pred,y_test))
clf_knn=KNeighborsClassifier()

clf_knn.fit(x_train,y_train)

pred=clf_knn.predict(x_test)

print(accuracy_score(pred,y_test))
clf_svm=SVC()

clf_svm.fit(x_train,y_train)

pred=clf_svm.predict(x_test)

print(accuracy_score(pred,y_test))
clf_dt=DecisionTreeClassifier()

clf_dt.fit(x_train,y_train)

pred=clf_dt.predict(x_test)

print(accuracy_score(pred,y_test))
clf_rf=RandomForestClassifier()

clf_rf.fit(x_train,y_train)

pred=clf_rf.predict(x_test)

print(accuracy_score(pred,y_test))
clf_gb=GradientBoostingClassifier()

clf_gb.fit(x_train,y_train)

pred=clf_gb.predict(x_test)

print(accuracy_score(pred,y_test))
models=[LogisticRegression(),LinearSVC(),SVC(kernel='rbf'),KNeighborsClassifier(),RandomForestClassifier(),

        DecisionTreeClassifier(),GradientBoostingClassifier(),GaussianNB()]

model_names=['LogisticRegression','LinearSVM','rbfSVM','KNearestNeighbors','RandomForestClassifier','DecisionTree',

             'GradientBoostingClassifier','GaussianNB']



acc=[]

d={}



for model in range(len(models)):

    clf=models[model]

    clf.fit(x_train,y_train)

    pred=clf.predict(x_test)

    acc.append(accuracy_score(pred,y_test))

     

d={'Modelling Algo':model_names,'Accuracy':acc}
acc_frame=pd.DataFrame(d)

acc_frame
sns.barplot(y='Modelling Algo',x='Accuracy',data=acc_frame)
params_dict={'C':[0.001,0.01,0.1,1,10,100],'gamma':[0.001,0.01,0.1,1,10,100],'kernel':['linear','rbf']}

clf=GridSearchCV(estimator=SVC(),param_grid=params_dict,scoring='accuracy',cv=10)

clf.fit(x_train,y_train)
clf.best_score_
clf.best_params_
print(accuracy_score(clf.predict(x_test),y_test))
print(precision_score(clf.predict(x_test),y_test))