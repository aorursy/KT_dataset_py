import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import plotly as pl

import seaborn as sns 

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import sklearn



import sklearn.linear_model as linear_model

import xgboost as xgb

from sklearn.model_selection import KFold

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from IPython.display import HTML, display

train=pd.read_csv('train.csv')

columns=train.columns.values
train.info()
intcol=[]

flcol=[]

obcol=[]

for each in columns:

    if type(train[each][0])==np.int64:

        intcol.append(each)

    if type(train[each][0])==np.float64:

        flcol.append(each)

    if type(train[each][0])==str:

        obcol.append(each)
print(intcol,flcol,obcol)
print(intcol+flcol)
intcolco={}

flcolco={}

obcolco={}

for each in intcol:

    intcolco[each]=train[each].nunique()

for each in flcol:

    flcolco[each]=train[each].nunique()

for each in obcol:

    obcolco[each]=train[each].nunique()

#check for missing data and plot

missing=train.isnull().sum()

missing=missing[missing>0]

missing.sort_values(inplace=True)

missing.plot.bar()
#okay we need to predict the sales price so let's see the distribution of all the unit variables

#univariate analysis

fig=train[intcol+flcol].hist(figsize=(12,12))

plt.tight_layout()
#detailed sales distribution 

#let's plot johnson, norm and lognorm 

'''It is apparent that SalePrice doesn't follow normal distribution, so before performing regression it has to be transformed. 

While log transformation does pretty good job, best fit is unbounded Johnson distribution.'''

import scipy.stats as st

y=train['SalePrice']

plt.figure(1); plt.title('Johnson su')

sns.distplot(y,kde=False,fit=st.johnsonsu)

plt.figure(2); plt.title('Normal')

sns.distplot(y,kde=False,fit=st.norm)

plt.figure(3); plt.title('Log Normal')

sns.distplot(y,kde=False,fit=st.lognorm)
#let's check if all the quantitaive varibles follow normal distribution

#shapiro should be less then 0.01 for normality to be determined

test_normality =lambda x:stats.shapiro(x.fillna(0))[1]<0.01

normal =train[intcol+flcol]

normal =normal.apply(test_normality)

print(not normal.any())

#Also none of quantitative variables has normal distribution so these should be transformed as well.
#plot all the distribution using pd.melt and facetgrid

f=pd.melt(train,value_vars=intcol+flcol)

g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)

g = g.map(sns.distplot, "value")



'''Some independent variables look like good candidates for log transformation: TotalBsmtSF, KitchenAbvGr, LotFrontage,

LotArea and others. While ganining on regression transformation will smooth out some irregularities which could be 

important like large amount of houses with 0 2ndFlrSF. 

Such irregularities are good candidates for feature construction.'''
#categorical variables

print(obcol)

for c in obcol:

    train[c]=train[c].astype('category')

    if train[c].isnull().any():

        train[c]=train[c].cat.add_categories(['MISSING'])

        train[c]=train[c].fillna('MISSING')
def boxplot(x,y,**kwargs):

    sns.boxplot(x=x,y=y)

    x=plt.xticks(rotation=90)



f=pd.melt(train,id_vars=['SalePrice'],value_vars=obcol)

g=sns.FacetGrid(f,col="variable",col_wrap=2,sharex=False,sharey=False,size=5)

g=g.map(boxplot,"value","SalePrice")

plt.show()
'''Some categories seem to more diverse with respect to SalePrice than others. Neighborhood has big impact on house prices. 

Most expensive seems to be Partial SaleCondition. Having pool on property seems to improve price substantially. 

There are also differences in variabilities between category values.'''
#quick estimation of influence of categorical variable on sale price.For each variable SalePrices are 

#partitioned to distinct sets based on category values. Then check with ANOVA test if sets have similar distributions.

#If variable has minor impact then set means should be equal. 

#Decreasing pval is sign of increasing diversity in partitions.



def anova(frame):

    anv=pd.DataFrame()

    anv['feature']=obcol

    pvals=[]

    for c in obcol:

        samples=[]

        for cls in frame[c].unique():

            s=frame[frame[c]==cls]['SalePrice'].values

            samples.append(s)

        pval=stats.f_oneway(*samples)[1]

        pvals.append(pval)  

    anv['pval']=pvals

    

    return anv.sort_values('pval')



a=anova(train)

fig,ax =plt.subplots(1,figsize=(12,12))

a['disparity']=np.log(1./a['pval'].values)

sns.barplot(data=a,x='feature',y='disparity')

x=plt.xticks(rotation=90)
'''def encode(frame,feature):

    ordering =pd.DataFrame()

    ordering['val']=frame[feature].unique()

    ordering.index=ordering.val

    ordering['spmean']=frame[[feature,'SalePrice']].groupby(feature).mean()['SalePrice']

    ordering=ordering.sort_values('spmean')

    ordering['ordering']=range(1,ordering.shape[0]+1)

    ordering=ordering['ordering'].to_dict()

    print(ordering)

    frame

    for cat,o in ordering.items():

        frame.loc[frame[feature]==cat,feature+'_E']==o'''
'''Here houses are divided in two price groups: cheap (under 200000) and expensive. 

Then means of quantitative variables are compared. Expensive houses have pools, 

better overall qual and condition, open porch and increased importance of MasVnrArea.'''

features = intcol



standard = train[train['SalePrice'] < 200000]

pricey = train[train['SalePrice'] >= 200000]



diff = pd.DataFrame()

diff['feature'] = features

diff['difference'] = [(pricey[f].fillna(0.).mean() - standard[f].fillna(0.).mean())/(standard[f].fillna(0.).mean())

                      for f in features]



sns.barplot(data=diff, x='feature', y='difference')

x=plt.xticks(rotation=90)
#let's see which are these variables are correlated 



corrmat =train[intcol+flcol].corr()

fig,ax =plt.subplots(1,figsize=(12,12))

sns.heatmap(corrmat,vmax=0.8,square=True)
#columns with top 10 correlations

cols=corrmat.nlargest(10,'SalePrice').index
#now top 10 corrmatrices

fig,ax =plt.subplots(1,figsize=(12,12))

cm=np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},yticklabels=cols.values,xticklabels=cols.values)

plt.show()
#these are the cols which are highly correlated to salesprice

print(cols)
#distributions of these variabels 

fig,ax =plt.subplots(figsize=(10,10))

train[cols].hist(ax=ax)

plt.tight_layout()
#now let's do a pairplot 

#2D analysis

fig,ax=plt.subplots(figsize=(12,12))

sns.set()

pp=sns.pairplot(train[cols],size=3,height=1.8,aspect=1.2,diag_kind="kde",kind="reg")

plt.show()
#tsne 



features =intcol+flcol

model=TSNE(n_components=2,random_state=0,perplexity=50)

X=train[features].fillna(0.).values

tsne=model.fit_transform(X)
std = StandardScaler()

s = std.fit_transform(X)

pca = PCA(n_components=30)

pca.fit(s)

pc = pca.transform(s)

kmeans = KMeans(n_clusters=5)

kmeans.fit(pc)
fr = pd.DataFrame({'tsne1': tsne[:,0], 'tsne2': tsne[:, 1], 'cluster': kmeans.labels_})

sns.lmplot(data=fr, x='tsne1', y='tsne2', hue='cluster', fit_reg=False)

print(np.sum(pca.explained_variance_ratio_))
#train linear regression 

X=pc

y=train['SalePrice'].values
lasso=linear_model.LassoLarsCV(max_iter=10000)

lasso.fit(X,np.log(y))

ypred=np.exp(lasso.predict(X))
def error(actual, predicted):

    actual = np.log(actual)

    predicted = np.log(predicted)

    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))
error(y,ypred)
# now using advanced regression 



from scipy.stats import skew 

from scipy.stats import pearsonr

%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline
train=pd.read_csv('train.csv')

test =pd.read_csv('test.csv')
train.head()
all_data =pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))
'''Data preprocessing:¶

We're not going to do anything fancy here:



First I'll transform the skewed numeric features by taking log(feature + 1) - this will make the features more normal

Create Dummy variables for the categorical features

Replace the numeric missing values (NaN's) with the mean of their respective columns'''
fig,axes= plt.subplots(figsize=(12,12))

prices=pd.DataFrame({"price":train["SalePrice"],"log(price+1)":np.log1p(train["SalePrice"])})

prices.hist(ax=axes)

plt.show()
#log transofrm the target

train["SalePrice"]=np.log1p(train["SalePrice"])
#log transform skewed features



numeric_feats= all_data.dtypes[all_data.dtypes!="object"].index

skewed_feats= train[numeric_feats].apply(lambda x:skew(x.dropna()))

skewed_feats=skewed_feats[skewed_feats>0.75]



skewed_feats=skewed_feats.index

all_data[skewed_feats]=np.log1p(all_data[skewed_feats])
all_data=pd.get_dummies(all_data)
#filled NA's with mean of column

all_data =all_data.fillna(all_data.mean())
#creting matrices for sklearn



X_train =all_data[:train.shape[0]]

X_test =all_data[train.shape[0]:]

y=train.SalePrice
'''Models¶

Now we are going to use regularized linear regression models from the scikit learn module. 

I'm going to try both l_1(Lasso) and l_2(Ridge) regularization. I'll also define a function that returns the cross-validation

rmse error so we can evaluate our models and pick the best tuning par'''
from sklearn.linear_model import Ridge,RidgeCV,ElasticNet,LassoCV,LassoLarsCV

from sklearn.model_selection import cross_val_score
def rmse_cv(model):

    rmse =np.sqrt(-cross_val_score(model,X_train,y,scoring="neg_mean_squared_error",cv=5))

    return rmse
model_ridge=Ridge()
'''The main tuning parameter for the Ridge model is alpha - 

a regularization parameter that measures how flexible our model is. 

The higher the regularization the less prone our model will be to overfit. 

However it will also lose flexibility and might not capture all of the signal in the data.'''



alphas =[0.05,0.1,0.3,1,3,5,10,15,30,50,75]

cv_ridge =[rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]
cv_ridge=pd.Series(cv_ridge,index=alphas)

cv_ridge.plot(title="Validation- just do it")

plt.xlabel("alpha")

plt.ylabel("rmse")
'''ote the U-ish shaped curve above. When alpha is too large the regularization is too strong and the model cannot capture all the complexities in the data. If however we let the model be too flexible (alpha small) the model 

begins to overfit. A value of alpha = 10 is about right based on the plot above.'''
cv_ridge.min()
'''So for the Ridge regression we get a rmsle of about 0.127



Let' try out the Lasso model. 

We will do a slightly different approach here and use the built in Lasso CV to figure out the best alpha for us. 

For some reason the alphas in Lasso CV are really the inverse or the alphas in Ridge.'''



model_lasso =LassoCV(alphas=[1,0.1,0.001,0.0005]).fit(X_train,y)
rmse_cv(model_lasso).mean()
'''Nice! The lasso performs even better so we'll just use this one to predict on the test set. 

Another neat thing about the Lasso is that it does feature selection for you - 

setting coefficients of features it deems unimportant to zero.

Let's take a look at the coefficients:'''



coef =pd.Series(model_lasso.coef_,index=X_train.columns)
print("lasso picked  "+ str(sum(coef!=0)))
'''Good job Lasso. One thing to note here however is that the features selected are not necessarily the "correct" ones - 

especially since there are a lot of collinear features in this dataset. One idea to try here is run Lasso a few times 

on boostrapped samples and see how stable the feature selection is.'''
imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])
plt.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
'''The most important positive feature is GrLivArea - the above ground area by area square feet. 

This definitely sense. Then a few other location and quality features contributed positively.

Some of the negative features make less sense and would be worth looking into more -

it seems like they might come from unbalanced categorical variables.



Also note that unlike the feature importance you'd get from a random forest these are actual 

coefficients in your model - so you can say precisely why the predicted price is what it is.

The only issue here is that we log_transformed both the target and the numeric features so the 

actual magnitudes are a bit hard to interpret.'''
#Let's look at residuals also



preds =pd.DataFrame({"preds":model_lasso.predict(X_train),"true":y})

preds["residuals"]=preds["true"]-preds["preds"]
preds.plot(x="preds",y="residuals",kind="scatter")
from keras.layers import Dense

from keras.models import Sequential

from keras.regularizers import l1

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
import xgboost as xgb
dtrain=xgb.DMatrix(X_train,label=y)

dtest=xgb.DMatrix(X_test,label=y)
params ={"max_depth":2,"eta":0.1}

model=xgb.cv(params,dtrain,num_boost_round=500,early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(X_train, y)
xgb_preds = np.expm1(model_xgb.predict(X_test))

lasso_preds = np.expm1(model_lasso.predict(X_test))
predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})

predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
'''Many times it makes sense to take a weighted average of uncorrelated results - 

this usually imporoves the score although in this case it doesn't help that much.'''



preds = 0.7*lasso_preds + 0.3*xgb_preds
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("ridge_sol.csv", index = False)