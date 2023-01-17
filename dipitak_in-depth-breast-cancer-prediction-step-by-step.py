# import the library

# for data load and data manupulation, data analysis
import pandas as pd

# for data computation in matrix form or mullti-dimensional arrays
import numpy as np

#for figure
from sklearn import model_selection

#sklearn data preparing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#sklearn scaling and labelencoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# sklearn models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

#sklearn model optimization
from sklearn.model_selection import GridSearchCV

# sklearn metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

#plot
import matplotlib.pyplot as plt
import seaborn as sns

#save model
import pickle

# display image
from IPython.display import Image
df_raw = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
df_raw.head()
print('Dataset contains')
print("Number of rows:",df_raw.shape[0])
print("Number of columns:",df_raw.shape[1])
df_raw.columns
df = df_raw.drop('id',axis=1).drop('Unnamed: 32',axis=1)
df.head(2)
df.isnull().sum()
df.dtypes
df['diagnosis'] = df['diagnosis'].astype('category')
df.dtypes
print("X is the features and Y is the target data")
X= df.drop('diagnosis',axis=1)
Y= df['diagnosis']
# let do standardscaling then
model_scaler = StandardScaler()
X_norm =model_scaler.fit_transform(X)
X_norm = pd.DataFrame(X_norm,columns=X.columns)
X_norm.head(3)
#explore correlation
plt.rcParams['figure.figsize']=[38,16]
sns.set(font_scale=1.4)
corr = X.corr()
mask_upper_traingle = np.triu(np.ones_like(corr, dtype=np.bool))
sns.heatmap(corr,mask=mask_upper_traingle, cmap='coolwarm', annot=True, fmt='.1')
# sorted correlationmatrix
def sort_and_createCorrelationMatrix(dt):
    corr_matrix = dt.corr().abs()
    corr_sorted = corr_matrix.stack().sort_values(ascending=False)
    corr_sorted = pd.DataFrame(corr_sorted)
    corr_sorted.reset_index(inplace=True)

    corr_sorted.columns=['level_0', 'level_1', 'value']
    corr_sorted =corr_sorted.pivot(index ='level_0', columns ='level_1', values='value') 
    mask_upper_traingle = np.triu(np.ones_like(corr_sorted, dtype=np.bool))
    sns.heatmap(corr_sorted,mask=mask_upper_traingle, cmap='coolwarm', annot=True, fmt='.1')

plt.rcParams['figure.figsize']=(18,5)
fig,(ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10)=plt.subplots(1,10)
sns.boxplot(x='diagnosis',y='radius_mean',data=df,ax=ax1)
sns.boxplot(x='diagnosis',y='texture_mean',data=df,ax=ax2)
sns.boxplot(x='diagnosis',y='perimeter_mean',data=df,ax=ax3)
sns.boxplot(x='diagnosis',y='area_mean',data=df,ax=ax4)
sns.boxplot(x='diagnosis',y='smoothness_mean',data=df,ax=ax5)
sns.boxplot(x='diagnosis',y='compactness_mean',data=df,ax=ax6)
sns.boxplot(x='diagnosis',y='concavity_mean',data=df,ax=ax7)
sns.boxplot(x='diagnosis',y='concave points_mean',data=df,ax=ax8)
sns.boxplot(x='diagnosis',y='symmetry_mean',data=df,ax=ax9)
sns.boxplot(x='diagnosis',y='fractal_dimension_mean',data=df,ax=ax10)
fig.tight_layout()

fig,(ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10)=plt.subplots(1,10)
sns.boxplot(x='diagnosis',y='radius_se',data=df,ax=ax1)
sns.boxplot(x='diagnosis',y='texture_se',data=df,ax=ax2)
sns.boxplot(x='diagnosis',y='perimeter_se',data=df,ax=ax3)
sns.boxplot(x='diagnosis',y='area_se',data=df,ax=ax4)
sns.boxplot(x='diagnosis',y='smoothness_se',data=df,ax=ax5)
sns.boxplot(x='diagnosis',y='compactness_se',data=df,ax=ax6)
sns.boxplot(x='diagnosis',y='concavity_se',data=df,ax=ax7)
sns.boxplot(x='diagnosis',y='concave points_se',data=df,ax=ax8)
sns.boxplot(x='diagnosis',y='symmetry_se',data=df,ax=ax9)
sns.boxplot(x='diagnosis',y='fractal_dimension_se',data=df,ax=ax10)
fig.tight_layout()

plt.rcParams['figure.figsize']=(17,5)
fig,(ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10)=plt.subplots(1,10)
sns.boxplot(x='diagnosis',y='radius_worst',data=df,ax=ax1)
sns.boxplot(x='diagnosis',y='texture_worst',data=df,ax=ax2)
sns.boxplot(x='diagnosis',y='perimeter_worst',data=df,ax=ax3)
sns.boxplot(x='diagnosis',y='area_worst',data=df,ax=ax4)
sns.boxplot(x='diagnosis',y='smoothness_worst',data=df,ax=ax5)
sns.boxplot(x='diagnosis',y='compactness_worst',data=df,ax=ax6)
sns.boxplot(x='diagnosis',y='concavity_worst',data=df,ax=ax7)
sns.boxplot(x='diagnosis',y='concave points_worst',data=df,ax=ax8)
sns.boxplot(x='diagnosis',y='symmetry_worst',data=df,ax=ax9)
sns.boxplot(x='diagnosis',y='fractal_dimension_worst',data=df,ax=ax10)
fig.tight_layout()

# check the distribution of data
fig = sns.FacetGrid(df,col='diagnosis', hue='diagnosis')
fig.map(sns.distplot, 'radius_mean', hist=True, rug=True )

fig = sns.FacetGrid(df,col='diagnosis', hue='diagnosis')
fig.map(sns.distplot, 'perimeter_mean', hist=True, rug=True)

fig = sns.FacetGrid(df,col='diagnosis', hue='diagnosis')
fig.map(sns.distplot, 'smoothness_mean', hist=True, rug=True)

fig = sns.FacetGrid(df,col='diagnosis', hue='diagnosis')
fig.map(sns.distplot, 'concavity_mean', hist=True, rug=True)


fig = sns.FacetGrid(df,col='diagnosis', hue='diagnosis')
fig.map(sns.distplot, 'radius_se', hist=True, rug=True)

fig = sns.FacetGrid(df,col='diagnosis', hue='diagnosis')
fig.map(sns.distplot, 'perimeter_se', hist=True, rug=True)

fig = sns.FacetGrid(df,col='diagnosis', hue='diagnosis')
fig.map(sns.distplot, 'smoothness_se', hist=True, rug=True)

fig = sns.FacetGrid(df,col='diagnosis', hue='diagnosis')
fig.map(sns.distplot, 'concavity_se', hist=True, rug=True)


fig = sns.FacetGrid(df,col='diagnosis', hue='diagnosis')
fig.map(sns.distplot, 'radius_worst', hist=True, rug=True)

fig = sns.FacetGrid(df,col='diagnosis', hue='diagnosis')
fig.map(sns.distplot, 'perimeter_worst', hist=True, rug=True)

fig = sns.FacetGrid(df,col='diagnosis', hue='diagnosis')
fig.map(sns.distplot, 'smoothness_worst', hist=True, rug=True)

fig = sns.FacetGrid(df,col='diagnosis', hue='diagnosis')
fig.map(sns.distplot, 'concavity_worst', hist=True, rug=True)

fig = sns.FacetGrid(df,col='diagnosis', hue='diagnosis')
fig.map(sns.distplot, 'symmetry_worst', hist=True, rug=True)
Y.value_counts()
model_encoder = LabelEncoder()
Y_norm = model_encoder.fit_transform(Y)
print(model_encoder.classes_)
print(model_encoder.transform(model_encoder.classes_))
Y_norm = pd.DataFrame(Y_norm,columns=['diagnosis'])
Y_norm.head(5) 
Y.value_counts()/len(Y)
# SMOTE: Synthetic Minority Over-sampling Technique 
from imblearn.over_sampling import SMOTE

model_smote =SMOTE(random_state=13)
X_norm, Y_norm = model_smote.fit_resample(X_norm,Y_norm)
def FitModelByParameterTuning(X,Y,algo_name, algorithm,gridSearchParams,cv, result=None):
    np.random.seed(13)
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
    
    gridSearch = GridSearchCV(
        estimator=algorithm,
        param_grid=gridSearchParams,
        cv = cv,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1)
    gridSearch_result = gridSearch.fit(x_train,y_train)
    best_params = gridSearch_result.best_params_
    pred = gridSearch_result.predict(x_test)
    cm=confusion_matrix(y_test,pred)
    
    ### print the result
    print('Best Params:', best_params)
    print('Classification Report:\n', classification_report(y_test,pred))
    print('Accuracy Score:', str(accuracy_score(y_test,pred)))
    print('Confusion Matrix:\n',cm)
    if result is not None:
        result_data.append([str(gridSearch_result.estimator).split('(')[0],accuracy_score(y_test,pred)])
    
    return [gridSearch_result,str(accuracy_score(y_test,pred))]
# data frame to for result
result_data =[]
#result_df = pd.DataFrame(result_data,columns=['Model','Accuracy'])

# Logistic Regression
param = {
    'penalty' : ['l1', 'l2','elastic-net']
}
model_Logistic = FitModelByParameterTuning(X_norm, Y_norm, 'LogisticRegression', LogisticRegression(), param, cv=5, result=result_data)

param = {
    'C':[0.1,1,100,1000],
    'gamma':[0.0001,0.001, 0.005, 0.1, 1, 3, 5]
}
model_SVC = FitModelByParameterTuning(X_norm, Y_norm, 'SVC', SVC(), param, cv=5, result=result_data)
param={
    "n_estimators":[100,500,1000,2000]
}

model_RandomForest = FitModelByParameterTuning(X_norm, Y_norm, 'Random Forest', RandomForestClassifier(), param, cv=5, result=result_data)

param={
    'n_estimators':[100,500,1000,2000]
}

model_XGBClassifier = FitModelByParameterTuning(X_norm, Y_norm, 'XGBClassifier', XGBClassifier(), param, cv=5, result=result_data)
result_data
result_dt = pd.DataFrame(result_data,columns=['Model','Accuracy'])
plt.rcParams["figure.figsize"] = [10, 6]

fig, ax = plt.subplots()
ax.bar(x=result_dt['Model'],height=result_dt['Accuracy'],color=['r','g','b','y'])

#ax = result_dt.plot.bar(x='Model',y='Accuracy',color=['Red','Blue','Green','Yellow']) 
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Accuracy in different Model")
for p in ax.patches:
    ax.annotate(str('{0:.3f}'.format( p.get_height())), (p.get_x() * 1.001, p.get_height() * 1.001))
    
plt.tight_layout()    
