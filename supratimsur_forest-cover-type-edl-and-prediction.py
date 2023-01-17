import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode()
import plotly
%matplotlib inline
df_train = pd.read_csv('../input/forest-cover-type-prediction/train.csv',low_memory=False)
df_test = pd.read_csv('../input/forest-cover-type-prediction/test.csv', low_memory=False)
combine = [df_train, df_test]
df_train.head().T
print("training datasets contains {} nos of rows and {} nos of columns".format(df_train.shape[0],df_train.shape[1]))
print('*'*68)
print("test datasets contains {} nos of rows and {} nos of columns".format(df_test.shape[0],df_test.shape[1]))
set(df_train) - set(df_test)
print(df_train.info())
print('-'*70)
print(df_test.info())
df_train.isna().sum()
df_test.isna().sum()
# view data statistics
df_train.describe(include='all')
df_train.columns
correlation = df_train[['Elevation', 'Aspect', 'Slope',                  # did not consider the soiltype and wilderness area
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', # as it is a hotencoded columns.
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Cover_Type']].corr()
correlation
correlation.iloc[10,10]
correlation.columns[10]
correlation.index[10]
len(correlation.columns)
test = pd.DataFrame()
values=[]
cols=[]
indx =[]
plot = []

for i in range(0,11):
    for j in range (i+1,11): # avoid repition
        if (1> correlation.iloc[i,j] >= 0.5) or (correlation.iloc[i,j] <= -0.5):
            test = test.append(pd.DataFrame(correlation.iloc[i,j], columns=[correlation.columns[i]],index=[correlation.index[j]]))
            print (f"{correlation.columns[j]} and {correlation.index[i]} are highly correlated ({correlation.iloc[i,j]:.2f})")
            values.append(correlation.iloc[i,j])
            indx.append([correlation.index[i]])
            cols.append(correlation.columns[j])
            plot.append([correlation.iloc[i,j],correlation.index[i],correlation.columns[j]])
plt.figure(figsize=(12,12))
sns.heatmap(correlation,annot = True, cmap ='YlGnBu');
df_train.skew()
df_train.groupby('Cover_Type').size()
df_train = df_train.iloc[:,1:] #remove the first id column which is of no use now
sns.set(style='darkgrid')
for v,i,c, in plot:
    sns.pairplot(data=df_train, hue="Cover_Type", x_vars=i, y_vars =c, height=5)
    plt.show()
for i in range(len(df_train.columns)-1):
    plt.figure(figsize=(12,6))
    sns.violinplot(x= df_train.Cover_Type, y = df_train.columns[i], data = df_train, pallet = 'deep')
    plt.show()
    
    
    
#Elevation is has a separate distribution for most classes. Highly correlated with the target and hence an important attribute
#Aspect contains a couple of normal distribution for several classes
#Horizontal distance to road and hydrology have similar distribution
#Hillshade 9am and 12pm display left skew
#Hillshade 3pm is normal
#Lots of 0s in vertical distance to hydrology
#Wilderness_Area3 gives no class distinction. As values are not present, others gives some scope to distinguish
#Soil_Type, 1,5,8,9,12,14,18-22, 25-30 and 35-40 offer class distinction as values are not present for many classes
# rem = []
# for i in df_train.columns:
#     if df_train[i].std()==0:
#         rem.append(i)
# rem
rem = [i for i in df_train if df_train[i].std()==0]
rem # list of columns we need to remove as it contains zero or its standard deviations are zero
df_train.drop(rem,axis=1,inplace=True) # dropping the unnecessary column from the training data sets
df_train
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer,MinMaxScaler,StandardScaler
df_train.columns[:10] # point from where ctegorical data begins#
df_train.iloc[:,:10]
standard = StandardScaler()
minmax = MinMaxScaler()
normalizer = Normalizer()

X = df_train.drop('Cover_Type',axis=1)
y = df_train['Cover_Type']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
# apply standardscaler to only non categorical data

stn_x_train = standard.fit_transform(X_train.iloc[:,:10])
stn_x_valid = standard.fit_transform(X_test.iloc[:,:10])

# apply MinMaxScaler to only non categorical data

minmax_x_train = minmax.fit_transform(stn_x_train)
minmax_x_valid = minmax.transform(stn_x_valid)

# apply normalizer to only non categroical data
normal_x_train = normalizer.fit_transform(minmax_x_train)
normal_x_valid = normalizer.fit_transform(minmax_x_valid)


X_train_scaled = pd.DataFrame(data = normal_x_train,columns=df_train.columns[:10])
X_valid_scaled = pd.DataFrame(data = normal_x_valid,columns=df_train.columns[:10])
X_train_scaled
x_train_cat = X_train.iloc[:,10:]
x_valid_cat = X_test.iloc[:,10:] # dataframe of categorical colums already hotencoded
x_train_cat.reset_index(inplace=True,drop=True)
x_valid_cat.reset_index(inplace=True,drop=True)
x_train_cat
x_valid_cat
x_train = X_train_scaled.join(x_train_cat) # joined non categorical and categorical dataframe together after scaling
x_test = X_valid_scaled.join(x_valid_cat)  # joined non categorical and categorical dataframe together after scaling
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBRFClassifier,XGBClassifier


svc = LinearSVC(max_iter=10000,dual=False)
knc = KNeighborsClassifier()
clf = RandomForestClassifier()
xgbrf = XGBRFClassifier()
xgb = XGBClassifier()

def model_score(model):
    score=[]
    model_name = type(model).__name__
    a = model.fit(x_train,y_train)
    a.score(x_test,y_test)
    y_pred = a.predict(x_test)
    acc = accuracy_score(y_test,y_pred)  
    return model_name,acc     
models = [svc,knc,clf,xgbrf,xgb]
score =[]
for name in models:
    a,b = model_score(name)
    score.append([a,round(b*100,2)])
df = pd.DataFrame(data=score,columns=['name','accuracy_score'])
c = df[df.accuracy_score==df.accuracy_score.max()]
print (df)
print ('_'*50)
print (f'the best model is {c.name.values[0]} and its score is {df.accuracy_score.max()}')
# as we see the best  model is RandomForestClassifier lets tune its hyperparamter
clf.get_params()
grid = {

 'max_features': ['auto','sqrt', 'log2'],
 'min_samples_leaf': np.arange(1,10,2),
 'min_samples_split': np.arange(2,10,2),
 'n_estimators': np.arange(100,200,50),
}
from sklearn.model_selection import cross_val_score,GridSearchCV

cvs = cross_val_score(clf,X=x_train,y=y_train,n_jobs=-1,cv=10)
cvs.mean()
gridsearch = GridSearchCV(clf,param_grid = grid, n_jobs=-1, verbose=2, cv = 5)
grid_clf = gridsearch.fit(x_train,y_train)
y_pred = grid_clf.predict(x_test)
accuracy_score(y_pred,y_test)*100
grid_clf.best_params_
# fitting best parameter into the model

ideal_model = RandomForestClassifier(max_features= 'auto',
 min_samples_leaf= 1,
 min_samples_split= 2,
 n_estimators= 100,
 n_jobs= -1,
 random_state= 0,
 verbose= 2)
ideal_model.fit(x_train,y_train)
y_pred = ideal_model.predict(x_test)
accuracy_score(y_test,y_pred)
#classification report
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
#Multiclass ROC AUC score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)
multiclass_roc_auc_score(y_test,y_pred)
from sklearn.metrics import plot_roc_curve,confusion_matrix

confusion_matrix(y_test,y_pred)
# plotting confusion matrix
plt.figure(figsize = (12,6))
sns.heatmap(confusion_matrix(y_test,y_pred),annot = True, cmap ='YlGnBu')
#feature importance

importances = ideal_model.feature_importances_

df_tmp = pd.DataFrame(data = {'score': importances,'name':x_train.columns})

df_tmp.sort_values(by='score',ascending = False,inplace=True)

g = sns.barplot(x='name',y='score',data= df_tmp.head(10))
plt.xticks(rotation=90)


df_test.columns #checking the columns of the test dataframe

rem = [i for i in df_test if df_test[i].std()==0] # removing the zero deviation columns if present
rem
df_test.describe()
# we need to remove only id column as of now
df_tmp = df_test.drop('Id',axis = 1)
# lets predict with the dataframe after normalizer and minmax and standard scaling

# apply standardscaler to only non categorical data


stn_x_test = standard.fit_transform(df_tmp.iloc[:,:10])

# apply MinMaxScaler to only non categorical data


minmax_x_valid = minmax.transform(stn_x_test)

# apply normalizer to only non categroical data

normal_x_valid = normalizer.fit_transform(minmax_x_valid)



X_valid_scaled = pd.DataFrame(data = normal_x_valid,columns=df_train.columns[:10])


x_valid_cat = df_tmp.iloc[:,10:] # dataframe of categorical colums already hotencoded

x_valid_cat.reset_index(inplace=True,drop=True)

x_test = X_valid_scaled.join(x_valid_cat)  # joined non categorical and categorical dataframe together after scaling
x_test.columns
Cover_Type = clf.predict(x_test.drop(['Soil_Type7','Soil_Type15'],axis =1)) # predicting the values
submission = pd.DataFrame(data={'Id': df_test.Id,'Cover_Type': Cover_Type}) # creating dataframe with prediction and id
submission
submission.to_csv('submission.csv', index=False) # creating the csv for submission