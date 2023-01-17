import numpy as np;

import pandas as pd;

import seaborn as sns;

import matplotlib.pyplot as plt;

%matplotlib inline



from scipy.stats import zscore



from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA



from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import KFold, cross_val_score, train_test_split



from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from scipy.stats import randint

from sklearn.model_selection import train_test_split
raw_data= pd.read_csv('../input/concrete-compressive-strength/concrete.csv');

raw_data.head()
raw_data.info();
# doesn't look like any NaN values

# let's confirm this by using the below technique.



raw_data.isnull().sum()
# finding columns which contains value as 0

col_with_zeros=[]

for col in raw_data.columns:

    zero_count = raw_data[raw_data[col]==0].count()[col];

    if (zero_count > 0):

        col_with_zeros.append([col, zero_count]);

        

pd.DataFrame(col_with_zeros,columns=['Columns Name','Count of Zeroes'])
features=raw_data.columns

features
#helper function - to provide index for 2 array.

def i_j_counter(rows,columns):

    i=0

    j=0

    indexes=[]

    while(1>0):

        indexes.append([i,j])

        if((j+1)%n_columns==0):

            j=0

            i=i+1

            if(i%n_rows == 0):

                break;

        else:

            j=j+1

    return indexes;
raw_data_desc=pd.DataFrame(raw_data.describe().transpose())

raw_data_desc['IQR']= raw_data_desc['75%'] - raw_data_desc['25%']



raw_data_desc
n_rows=3;

n_columns=3;

skew= pd.DataFrame(raw_data.skew(),columns=['value'])

fig, axes = plt.subplots(n_rows,n_columns, figsize=(16,10))

for col,index,skew in zip(features,i_j_counter(n_rows,n_columns),skew['value']):

    sns.distplot(raw_data[col], ax=axes[index[0],index[1]],label=f'skew {skew: .2f}', color='c')

    axes[index[0],index[1]].legend(loc='upper right')
# Normalizing the data

z_data= raw_data.apply(zscore);
#boxplot has better visulalization from the normalized data.

plt.figure(figsize=(18,7))

sns.boxplot(data=z_data[features[0:-1]]);
# cleaning outlier

def clean_outliers(data):

    for col in data.columns[0:-1]:

        Q1= data[col].quantile(0.25)

        Q3= data[col].quantile(0.75)

        IQR=Q3-Q1

        c1=Q1-(1.5*IQR)

        c2=Q3+(1.5*IQR)

        data.loc[data[col] < c1, col] = c1

        data.loc[data[col] > c2, col] = c2

    return data
z_data=clean_outliers(z_data);

raw_data=clean_outliers(raw_data);
#boxplot has better visulalization from the normalized data.

plt.figure(figsize=(18,7))

sns.boxplot(data=z_data[features[0:-1]]);
corr= raw_data.corr();

sns.set_context('notebook', font_scale=1.0, rc={"lines.linewidth":3.5});

plt.figure(figsize=(18,7));



#create a mask with all value as 1 of size of corr matrix

mask= np.zeros_like(corr);

#makring all elements one above the main diagonal as True

mask[np.triu_indices_from(mask,1)] =True

sns.heatmap(corr, mask=mask, fmt='0.2f', annot=True);
sns.pairplot(z_data, diag_kind='kde');
raw_data.head()
# we can derive different type of composite features from the given features. Currently we are going to use ratio method to do that.

# our feautres are ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg','fineagg', 'age']

comp_data= pd.DataFrame()

comp_data['strength'] = raw_data['strength']

for col1 in features[0:-1]:

    for col2 in features[0:-1]:

        if col1 != col2:

            col=f'{col1}/{col2}'

            new_col=raw_data[col1]/raw_data[col2]

            comp_data[col]=new_col

            

corr=pd.DataFrame(comp_data.corr())

clean_corr= pd.DataFrame(corr[abs(corr)> 0.50]['strength'])

clean_corr.dropna(inplace=True)

clean_corr
comp_raw_data= pd.DataFrame(raw_data).copy()

comp_raw_data['cement/water'] = raw_data['cement']/raw_data['water']

comp_raw_data['cement/coarseagg'] = raw_data['cement']/raw_data['coarseagg']

comp_raw_data['water/cement'] = raw_data['water']/raw_data['cement']

comp_raw_data['water/age'] = raw_data['water']/raw_data['age']

comp_raw_data['fineagg/age'] = raw_data['fineagg']/raw_data['age']

comp_raw_data['age/water'] = raw_data['age']/raw_data['water']

comp_raw_data.head()
def get_poly_pca_dataset(dataset,degree,explained_variance):

    poly= PolynomialFeatures(degree=degree, interaction_only=True)

    poly_data=poly.fit_transform(dataset.drop('strength',axis=1))

    poly_data= pd.DataFrame(poly_data, columns=['feat_'+str(x) for x in range(poly_data.shape[1])])

    poly_data=poly_data.join(dataset.drop('strength',axis=1)) # joinig back the oroginal columns of the dataset passed.

    print(f'-Dataset with polynomial degree: {degree} has shape: {poly_data.shape}\n')

    

    pca=PCA(n_components=explained_variance)

    pca.fit(poly_data)

    pca_data=pca.transform(poly_data)

    pca_data=pd.DataFrame(pca_data)

    print(f'-After PCA, as parameter to explaine variance by {explained_variance*100 : .2f}%.\n')

    print(f'-Percentage of the variance explained by each column is:\n {pca.explained_variance_ratio_*100}')

    

    return pca_data
# with degree 2

pca_data_2 =get_poly_pca_dataset(comp_raw_data,2,.99)

pca_data_2.head()
# with degree 3

pca_data_3 =get_poly_pca_dataset(comp_raw_data,3,.99)

pca_data_3.head()
# with degree 4

pca_data_4 =get_poly_pca_dataset(comp_raw_data,4,.99)

pca_data_4.head()
comp_z_data= comp_raw_data.apply(zscore)

clusters=range(2,20)

mean_distortions=[]

for val in clusters:

    kmeans =KMeans(n_clusters=val)

    kmeans.fit(comp_z_data)

    mean_distortions.append(sum(np.min(cdist(comp_z_data,kmeans.cluster_centers_),axis=1))/comp_z_data.shape[0])
plt.plot(clusters, mean_distortions,'bx-')

plt.xlabel('No. Of Clusters')

plt.ylabel('Distortion')

plt.title('Elbow Method')
labeled_data=pd.DataFrame(comp_raw_data).copy()

kmeans= KMeans(n_clusters=5)

kmeans.fit(comp_z_data)

labeled_data['labels']=kmeans.labels_
labeled_data.head()
labeled_data.boxplot(by='labels', layout=(6,3), figsize=(20,20));
labeled_data.groupby('labels').mean()
labeled_data.groupby('labels').count()
for val in features[0:-1]:

    with sns.axes_style("white"):

        plot = sns.lmplot(val,'strength',data=labeled_data,hue='labels');

    
pipelines=[]

pipelines.append(('LinearRegression',Pipeline([('Scaler', StandardScaler()), ('LinearRegression',LinearRegression())])))

pipelines.append(('DecisionTreeRegressor',Pipeline([('Scaler', StandardScaler()), ('DecisionTreeRegressor',DecisionTreeRegressor(random_state=1))])))

pipelines.append(('RandomForestRegressor',Pipeline([('Scaler', StandardScaler()), ('RandomForestRegressor',RandomForestRegressor(random_state=1))])))

pipelines.append(('SVR',Pipeline([('Scaler', StandardScaler()), ('SVR',SVR())])))

pipelines.append(('GradientBoostingRegressor',Pipeline([('Scaler', StandardScaler()), ('GradientBoostingRegressor',GradientBoostingRegressor(random_state=1))])))
def get_pipeline_with_degree(degree):

    pipelines_pca=[]

    pipelines_pca.append(('LinearRegression',Pipeline([('Scaler', StandardScaler()),('polynomialFeature',PolynomialFeatures(degree=degree,interaction_only=True)),('PCA',PCA(n_components=.99)), ('LinearRegression',LinearRegression())])))

    pipelines_pca.append(('DecisionTreeRegressor',Pipeline([('Scaler', StandardScaler()),('polynomialFeature',PolynomialFeatures(degree=degree,interaction_only=True)),('PCA',PCA(n_components=.99)), ('DecisionTreeRegressor',DecisionTreeRegressor(random_state=1))])))

    pipelines_pca.append(('RandomForestRegressor',Pipeline([('Scaler', StandardScaler()), ('polynomialFeature',PolynomialFeatures(degree=degree,interaction_only=True)),('PCA',PCA(n_components=.99)), ('RandomForestRegressor',RandomForestRegressor(random_state=1))])))

    pipelines_pca.append(('SVR',Pipeline([('Scaler', StandardScaler()),('polynomialFeature',PolynomialFeatures(degree=degree,interaction_only=True)),('PCA',PCA(n_components=.99)), ('SVR',SVR())])))

    pipelines_pca.append(('GradientBoostingRegressor',Pipeline([('Scaler', StandardScaler()),('polynomialFeature',PolynomialFeatures(degree=degree,interaction_only=True)),('PCA',PCA(n_components=.99)), ('GradientBoostingRegressor',GradientBoostingRegressor(random_state=1))])))

    

    return pipelines_pca;
def cv_results(X,y,pipelines):

    results = pd.DataFrame(columns=['Name','Mean Variance Explained', 'Standard Devivation'])

    for index,pipeline in enumerate(pipelines):

        name,pipeline = pipeline

        kfold = KFold(n_splits=10, shuffle=True, random_state=1)

        cv_results = cross_val_score(pipeline,X,y,cv=kfold, scoring='explained_variance')

        results.loc[index]= [name,cv_results.mean()*100, cv_results.std()*100]

    return results
X,X_test,y,y_test = train_test_split(raw_data.drop('strength',axis=1), raw_data['strength'], test_size=0.20, random_state=1)

X_train,X_val,y_train,y_val= train_test_split(X,y, test_size=0.20, random_state=1)



cv_results(X_train,y_train,pipelines)
X,X_test,y,y_test = train_test_split(comp_raw_data.drop('strength',axis=1), comp_raw_data['strength'], test_size=0.20, random_state=1)

X_train,X_val,y_train,y_val= train_test_split(X,y, test_size=0.20, random_state=1)



cv_results(X_train,y_train,pipelines)
X,X_test,y,y_test = train_test_split(pca_data_2, comp_raw_data['strength'], test_size=0.20, random_state=1)

X_train,X_val,y_train,y_val= train_test_split(X,y, test_size=0.20, random_state=1)



cv_results(X_train,y_train,pipelines)
X,X_test,y,y_test = train_test_split(pca_data_3, comp_raw_data['strength'], test_size=0.20, random_state=1)

X_train,X_val,y_train,y_val= train_test_split(X,y, test_size=0.20, random_state=1)



cv_results(X_train,y_train,pipelines)
X,X_test,y,y_test = train_test_split(pca_data_4, comp_raw_data['strength'], test_size=0.20, random_state=1)

X_train,X_val,y_train,y_val= train_test_split(X,y, test_size=0.20, random_state=1)



cv_results(X_train,y_train,pipelines)
X,X_test,y,y_test = train_test_split(comp_raw_data.drop('strength',axis=1), comp_raw_data['strength'], test_size=0.20, random_state=1)

X_train,X_val,y_train,y_val= train_test_split(X,y, test_size=0.20, random_state=1)
cv_results(X_train,y_train,get_pipeline_with_degree(2))
cv_results(X_train,y_train,get_pipeline_with_degree(3))
cv_results(X_train,y_train,get_pipeline_with_degree(4))
grb= GradientBoostingRegressor()

grb.get_params()
tunning_pipeline= Pipeline([('scaler',StandardScaler()),('gradientboostinregressor', GradientBoostingRegressor(random_state=1))])
params={'gradientboostinregressor__learning_rate': [0.1,0.2,0.3],

       'gradientboostinregressor__loss': ['ls', 'lad', 'huber', 'quantile'],

       'gradientboostinregressor__max_depth':range(1,3) ,

        'gradientboostinregressor__max_features': [None,2,3],

        'gradientboostinregressor__max_leaf_nodes': [None,2,3],

       }
X,X_test,y,y_test = train_test_split(comp_raw_data.drop('strength',axis=1), comp_raw_data['strength'], test_size=0.20, random_state=1)

X_train,X_val,y_train,y_val= train_test_split(X,y, test_size=0.20, random_state=1)



searchGrid= GridSearchCV(tunning_pipeline, param_grid=params, cv=10, scoring='explained_variance');

searchGrid.fit(X_train,y_train);

searchGrid.best_params_
print(f'Explained variance by the model on trainning data is {searchGrid.score(X_train,y_train)*100: .2f}%');

print(f'Explained variance by the model on validation data is {searchGrid.score(X_val,y_val)*100: .2f}%');
# Using the best params given by the Grid Search

final_pipeline = Pipeline([('scaler',StandardScaler()),

                           ('gradientboostinregressor', GradientBoostingRegressor(random_state=1, 

                                                                                learning_rate=0.3,

                                                                                loss='ls',

                                                                                max_depth=2,

                                                                                max_features=None,

                                                                                max_leaf_nodes=None))])

final_pipeline.fit(X_train,y_train);
print(f'Explained variance by the model on trainning data is {final_pipeline.score(X_train,y_train)*100: .2f}%');

print(f'Explained variance by the model on validation data is {final_pipeline.score(X_val,y_val)*100: .2f}%');
print(f'Explained variance by the model on test data is {final_pipeline.score(X_test,y_test)*100: .2f}%');
X=X_test

y=y_test

kfold=KFold(n_splits=10, shuffle=True, random_state=1)

cv_results=cross_val_score(final_pipeline,X,y, cv=kfold, scoring='explained_variance')
print(f'The cross validation result mean is: {cv_results.mean()*100: .2f}% and standard deviation: {cv_results.std()*100: .2f}%');


plt.plot(range(1,11),cv_results);

plt.xlabel('Split Count');

plt.ylabel('Explained variance');

plt.title('CV-KFold chart for Gradient Boosting Algo');

print(f'The algorithm can explain variance in range of {(cv_results.mean()-(2*cv_results.std()))*100: .2f}% to {(cv_results.mean()+(2*cv_results.std()))*100: .2f}% with 95% confidence.');