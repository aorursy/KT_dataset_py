import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data1=pd.read_csv("/kaggle/input/janatahack-crosssell-prediction/train.csv")
data1.head()
data1.info()
data1.isnull().sum()
data1.isnull().mean()
data1.describe()
data1.describe(include='O')
data1.shape
data1['Response'].value_counts()
sns.countplot('Gender',data=data1,hue='Response')
plt.plot()
data1.groupby(['Gender'])['Response'].value_counts(normalize=True)*100
sns.distplot(data1['Age'])
plt.plot()
data1['Age'].describe()
def brac(x):
    if (x>=20) & (x<31):
        return '20-30'
    if(x>=31) & (x<41):
        return '31-40'
    if(x>=41) & (x<51):
        return '41-50'
    if(x>=51) & (x<61):
        return '51-60'
    if(x>=61) & (x<71):
        return '61-70'
    if(x>=71) & (x<81):
        return '71-80'
    if(x>=81) & (x<91):
        return '81-90'
        
data1['AgeBracket']=data1['Age'].apply(brac)
data1[['Age','AgeBracket']]
sns.countplot('AgeBracket',data=data1,hue='Response')
plt.plot()
t1=pd.DataFrame(data1.groupby(['AgeBracket'])['Response'].value_counts(normalize=True)*100)
t1
pd.DataFrame(data1.groupby(['Driving_License'])['Response'].value_counts(normalize=True))
pd.DataFrame(data1.groupby(['Previously_Insured'])['Response'].value_counts(normalize=True))
data1['Vehicle_Age'].unique()
pd.DataFrame(data1.groupby(['Vehicle_Age'])['Response'].value_counts(normalize=True))
data1['Vehicle_Damage'].unique()
pd.DataFrame(data1.groupby(['Vehicle_Damage'])['Response'].value_counts(normalize=True))
data1['Annual_Premium'].unique()
t3=pd.DataFrame(data1.groupby(['Annual_Premium'])['Response'].value_counts(normalize=True))
t3
data1['Annual_Premium'].describe()
data3=data1.copy()
sns.distplot(data1['Annual_Premium'])
sns.relplot('Response','Annual_Premium',data=data1)
data1['quantile_5']=pd.qcut(data1['Annual_Premium'], q=5)
pd.DataFrame(data1.groupby(['quantile_5'])['Response'].value_counts(normalize=True)*100)
pp=pd.DataFrame(data1.groupby(['Policy_Sales_Channel'])['Response'].value_counts(normalize=True)*100)
pp
def find_non_rare_labels(df, variable, tolerance):
    
    temp = df.groupby([variable])[variable].count() / len(df)
    
    non_rare = [x for x in temp.loc[temp>tolerance].index.values]
    
    return non_rare
def rare_encoding(new, variable, tolerance):
    frequent_cat = find_non_rare_labels(data3, variable, tolerance)

    # re-group rare labels
    data3[variable] = np.where(data3[variable].isin(
        frequent_cat), data3[variable], 'Rare')

    return data3
for variable in ['Policy_Sales_Channel']:
    
     data3= rare_encoding(data3, variable, 0.01)
for col in ['Policy_Sales_Channel']:

    temp_df = pd.Series(data3[col].value_counts() / len(data3) )

    # make plot with the above percentages
    fig = temp_df.sort_values(ascending=False).plot.bar()
    fig.set_xlabel(col)

    # add a line at 5 % to flag the threshold for rare categories
    fig.axhline(y=0.01, color='red')
    fig.set_ylabel('Percentage of houses')
    plt.show()
pd.DataFrame(data3.groupby(['Policy_Sales_Channel'])['Response'].value_counts(normalize=True)*100)
data1['Vintage'].plot(kind='hist')
data1['Vin_q']=pd.cut(data1['Vintage'], bins=10)
pd.DataFrame(data1.groupby(['Vin_q'])['Response'].value_counts(normalize=True)*100)
sns.distplot(data1['Vintage'])
sns.relplot('Response','Vintage',data=data1)
for col in ['Region_Code']:

    temp_df = pd.Series(data3[col].value_counts() / len(data3) )

    # make plot with the above percentages
    fig = temp_df.sort_values(ascending=False).plot.bar()
    fig.set_xlabel(col)

    # add a line at 5 % to flag the threshold for rare categories
    fig.axhline(y=0.05, color='red')
    fig.set_ylabel('Percentage of houses')
    plt.show()
def find_non_rare_labels(df, variable, tolerance):
    
    temp = df.groupby([variable])[variable].count() / len(df)
    
    non_rare = [x for x in temp.loc[temp>tolerance].index.values]
    
    return non_rare
find_non_rare_labels(data3, 'Region_Code', 0.02)

def rare_encoding(new, variable, tolerance):
    frequent_cat = find_non_rare_labels(data3, variable, tolerance)

    # re-group rare labels
    data3[variable] = np.where(data3[variable].isin(
        frequent_cat), data3[variable], 'Rare')

    return data3
for variable in ['Region_Code']:
    
     data3= rare_encoding(data3, variable, 0.01)
for col in ['Region_Code']:

    temp_df = pd.Series(data3[col].value_counts() / len(data3) )

    # make plot with the above percentages
    fig = temp_df.sort_values(ascending=False).plot.bar()
    fig.set_xlabel(col)

    # add a line at 5 % to flag the threshold for rare categories
    fig.axhline(y=0.01, color='red')
    fig.set_ylabel('Percentage of houses')
    plt.show()
pd.DataFrame(data3.groupby(['Region_Code'])['Response'].value_counts(normalize=True)*100)
data_contact=data3[(data3['Driving_License']==1) & 
      (data3['Previously_Insured']==0) &
     (data3['Vehicle_Age']=='> 2 Years') & (data3['Vehicle_Damage']=='Yes') & 
      ((data3['AgeBracket']=='31-40') | (data3['AgeBracket']=='41-50') | (data3['AgeBracket']=='51-60')) & 
                  ((data3['Region_Code']=='11.0') | (data3['Region_Code']=='18.0') |
                  (data3['Region_Code']=='20.0') | (data3['Region_Code']=='29.0') |
                  (data3['Region_Code']=='3.0') | (data3['Region_Code']=='25.0') |
                  (data3['Region_Code']=='39.0') | (data3['Region_Code']=='41.0')) & 
                  ((data3['Policy_Sales_Channel']=='154.0') | (data3['Policy_Sales_Channel']=='156.0') |
                  (data3['Policy_Sales_Channel']=='157.0'))]
data_contact.head()
data_contact.shape
cor1=data1.corr()
plt.figure(figsize = (16,10))
sns.heatmap(cor1,linewidths=1,annot=True)
plt.plot()
data2=data1.copy()
data2.drop(['AgeBracket','quantile_5','Vin_q'],axis=1,inplace=True)
data2.shape
data_test=pd.read_csv("/kaggle/input/janatahack-crosssell-prediction/test.csv")
data_test.shape
#data_test.drop(['Annual_Premium','Vintage'],axis=1,inplace=True)
new=pd.concat([data2,data_test],axis=0)
new['Vintage']=new['Vintage']/365
for col in ['Region_Code']:

    temp_df = pd.Series(new[col].value_counts() / len(new) )

    # make plot with the above percentages
    fig = temp_df.sort_values(ascending=False).plot.bar()
    fig.set_xlabel(col)

    # add a line at 5 % to flag the threshold for rare categories
    fig.axhline(y=0.05, color='red')
    fig.set_ylabel('Percentage of houses')
    plt.show()
def find_non_rare_labels(df, variable, tolerance):
    
    temp = df.groupby([variable])[variable].count() / len(df)
    
    non_rare = [x for x in temp.loc[temp>tolerance].index.values]
    
    return non_rare
find_non_rare_labels(new, 'Region_Code', 0.02)
[x for x in new['Region_Code'].unique(
) if x not in find_non_rare_labels(new, 'Region_Code', 0.02)]
new1=new.copy()
def rare_encoding(new, variable, tolerance):
    frequent_cat = find_non_rare_labels(new1, variable, tolerance)

    # re-group rare labels
    new1[variable] = np.where(new1[variable].isin(
        frequent_cat), new1[variable], 'Rare')

    return new1
for variable in ['Region_Code']:
    
     new1= rare_encoding(new1, variable, 0.01)
for col in ['Region_Code']:

    temp_df = pd.Series(new1[col].value_counts() / len(new1) )

    # make plot with the above percentages
    fig = temp_df.sort_values(ascending=False).plot.bar()
    fig.set_xlabel(col)

    # add a line at 5 % to flag the threshold for rare categories
    fig.axhline(y=0.01, color='red')
    fig.set_ylabel('Percentage of houses')
    plt.show()
g1=pd.DataFrame((new1.groupby(['Region_Code'])['Response'].value_counts(normalize=True)*100).sort_values(ascending=False))
for col in ['Policy_Sales_Channel']:

    temp_df = pd.Series(new[col].value_counts() / len(new) )

    # make plot with the above percentages
    fig = temp_df.sort_values(ascending=False).plot.bar()
    fig.set_xlabel(col)

    # add a line at 5 % to flag the threshold for rare categories
    fig.axhline(y=0.01, color='red')
    fig.set_ylabel('Percentage of houses')
    plt.show()
def find_non_rare_labels(df, variable, tolerance):
    
    temp = df.groupby([variable])[variable].count() / len(df)
    
    non_rare = [x for x in temp.loc[temp>tolerance].index.values]
    
    return non_rare
find_non_rare_labels(new1, 'Policy_Sales_Channel', 0.01)
def rare_encoding(new, variable, tolerance):
    frequent_cat = find_non_rare_labels(new1, variable, tolerance)

    # re-group rare labels
    new1[variable] = np.where(new1[variable].isin(
        frequent_cat), new1[variable], 'Rare')

    return new1
for variable in ['Policy_Sales_Channel']:
    
     new1= rare_encoding(new1, variable, 0.01)
for col in ['Policy_Sales_Channel']:

    temp_df = pd.Series(new1[col].value_counts() / len(new1) )

    # make plot with the above percentages
    fig = temp_df.sort_values(ascending=False).plot.bar()
    fig.set_xlabel(col)

    # add a line at 5 % to flag the threshold for rare categories
    fig.axhline(y=0.01, color='red')
    fig.set_ylabel('Percentage of houses')
    plt.show()
pc=pd.DataFrame((new1.groupby(['Policy_Sales_Channel'])['Response'].value_counts(normalize=True)*100).sort_values(ascending=False))
pc
new1.dtypes
#from feature_engine.categorical_encoders import CountFrequencyCategoricalEncoder
#dl=dict(new['Driving_License'].value_counts())
#new['Driving_License']=new['Driving_License'].replace(dl)
#a=dict(new['Region_Code'].value_counts())
#new['Region_Code']=new['Region_Code'].replace(a)
new1['Vehicle_Age']=new1['Vehicle_Age'].replace({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})
new1['Vehicle_Damage']=new1['Vehicle_Damage'].replace({'Yes':1,'No':0})
new1['Gender']=new1['Gender'].replace({'Male':1,'Female':0})
new1.head()
!pip install feature-engine
from feature_engine.categorical_encoders import OneHotCategoricalEncoder
#new.columns
new1.dtypes
new1['Region_Code'].unique()
#new1['Region_Code']=new1['Region_Code'].replace({'Rare':0})

#new1=new1.astype({'Region_Code':float})
new1.describe()
import scipy.stats as stats
# function to create histogram, Q-Q plot and
# boxplot. We learned this in section 3 of the course


def diagnostic_plots(df, variable):
    # function takes a dataframe (df) and
    # the variable of interest as arguments

    # define figure size
    plt.figure(figsize=(16, 4))

    # histogram
    plt.subplot(1, 3, 1)
    sns.distplot(df[variable], bins=30)
    plt.title('Histogram')

    # Q-Q plot
    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.ylabel('Variable quantiles')

    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')

    plt.show()
diagnostic_plots(new, 'Age')
diagnostic_plots(new, 'Annual_Premium')
diagnostic_plots(new, 'Vintage')
def find_skewed_boundaries(df, variable, distance):

    # Let's calculate the boundaries outside which sit the outliers
    # for skewed distributions

    # distance passed as an argument, gives us the option to
    # estimate 1.5 times or 3 times the IQR to calculate
    # the boundaries.

    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)

    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)

    return upper_boundary, lower_boundary
ap_upper_limit, ap_lower_limit = find_skewed_boundaries(new, 'Annual_Premium', 1.5)
ap_upper_limit, ap_lower_limit
# Now let's replace the outliers by the maximum and minimum limit

new['Annual_Premium']= np.where(new['Annual_Premium'] > ap_upper_limit, ap_upper_limit,
                       np.where(new['Annual_Premium'] < ap_lower_limit, ap_lower_limit, new['Annual_Premium']))
diagnostic_plots(new, 'Annual_Premium')
new['Annual_Premium'].describe()
ohe_enc = OneHotCategoricalEncoder(
    top_categories=None, # we can select which variables to encode
    drop_last=True) # to return k-1, false to return k


ohe_enc.fit(new1)
tmp = ohe_enc.transform(new1)

tmp.head()
#tmp.drop('Driving_License',axis=1,inplace=True)
tmp.drop('id',axis=1,inplace=True)
Train=tmp.iloc[0:381109]
Test=tmp.iloc[381109:]
X=Train.drop('Response',axis=1)
Y=Train['Response']
from imblearn.over_sampling import RandomOverSampler
os =  RandomOverSampler(0.70)
X_train_res, y_train_res = os.fit_sample(X, Y)
X_train_res.shape,y_train_res.shape
from collections import Counter
print('Original dataset shape {}'.format(Counter(Y)))
print('Resampled dataset shape {}'.format(Counter(y_train_res)))
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
ordered_rank_features=SelectKBest(score_func=chi2,k=5)
ordered_feature=ordered_rank_features.fit(X_train_res,y_train_res)

dfscores=pd.DataFrame(ordered_feature.scores_,columns=["Score"])
dfcolumns=pd.DataFrame(X.columns)


features_rank=pd.concat([dfcolumns,dfscores],axis=1)
features_rank.columns=['Features','Score']

X_train_res.shape
features_rank.nlargest(15,'Score')
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_train_res,
                                                    y_train_res,
                                                    test_size=0.3,
                                                    random_state=0)

X_train.shape, X_test.shape
Test.drop('Response',axis=1,inplace=True)
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_squared_log_error, roc_auc_score
test1=pd.read_csv("/kaggle/input/janatahack-crosssell-prediction/test.csv")
id1=test1['id']
from lightgbm import LGBMClassifier
lgbcl = LGBMClassifier(boosting_type='gbdt',n_estimators=500,depth=10,learning_rate=0.04,objective='binary',metric='auc',
                 colsample_bytree=0.5,reg_lambda=2,reg_alpha=2,random_state=294,n_jobs=-1)
lgbcl= lgbcl.fit(X_train, Y_train,eval_metric='auc',verbose=2)
y_lgb = lgbcl.predict(X_test)
probs_tr = lgbcl.predict_proba(X_train)[:, 1]
probs_te = lgbcl.predict_proba(X_test)[:, 1]
print(roc_auc_score(Y_train, probs_tr))
print(roc_auc_score(Y_test, probs_te))
lgbcl1=lgbcl.fit(X_train_res,y_train_res)
lgbcl1
lgb_pred= lgbcl1.predict_proba(Test)[:, 1]
lgb_pred
lgb=pd.concat([pd.DataFrame(id1),pd.DataFrame(lgb_pred)],axis=1)
lgb2=lgb.copy()
lgb2.rename(columns={0:'Response'},inplace=True)
lgb2.to_csv('lgb_onehot+label.csv',index=False)
te=lgbcl.predict(X_test)
tr=lgbcl.predict(X_train)
from sklearn.metrics import confusion_matrix
trc=confusion_matrix(Y_train,tr)

tec=confusion_matrix(Y_test,te)
recall_train=((trc[0][0])/(trc[0][0]+trc[1][0]))*100
recall_train
recall_test=((tec[0][0])/(tec[0][0]+tec[1][0]))*100
recall_test
precision_train=((trc[0][0])/(trc[0][0]+trc[0][1]))*100
precision_train
precision_test=((tec[0][0])/(tec[0][0]+trc[0][1]))*100
precision_test
from sklearn.ensemble import RandomForestClassifier
rf1=RandomForestClassifier()
cross_val_score(rf1,X_train_res,y_train_res,cv=5,scoring='accuracy')
rf1.fit(X_train,Y_train)
probs_tr_rf = rf1.predict_proba(X_train)[:, 1]
probs_te_rf = rf1.predict_proba(X_test)[:, 1]
print(roc_auc_score(Y_train, probs_tr_rf))
print(roc_auc_score(Y_test, probs_te_rf))
te=rf1.predict(X_test)
tr=rf1.predict(X_train)

from sklearn.metrics import confusion_matrix
trc=confusion_matrix(Y_train,tr)

tec=confusion_matrix(Y_test,te)
recall_train=((trc[0][0])/(trc[0][0]+trc[1][0]))*100
recall_train
recall_test=((tec[0][0])/(tec[0][0]+tec[1][0]))*100
recall_test
precision_train=((trc[0][0])/(trc[0][0]+trc[0][1]))*100
precision_train
precision_test=((tec[0][0])/(tec[0][0]+trc[0][1]))*100
precision_test
F1_train = 2 * (precision_train * recall_train) / (precision_train + recall_train)
F1_train
F1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
F1_test
rf_t=rf1.predict_proba(Test)
rf_t
rf123=pd.concat([pd.DataFrame(id1),pd.DataFrame(rf_t)],axis=1)
rf_done=rf123.drop(0,axis=1)
rf_done.rename(columns={1:'Response'},inplace=True)
rf_done.to_csv('AR_check.csv',index=False)