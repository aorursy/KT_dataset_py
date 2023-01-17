from IPython.display import HTML

from IPython.display import Image

Image(url= "https://images.mapsofindia.com/liveblog/2018/10/air-quality-index-of-the-biggest-cities-in-india-f1.jpg")
from IPython.core.display import HTML

HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

</script>

The raw code for this IPython notebook is by default hidden for easier reading.

To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.preprocessing import Imputer

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = (10, 7)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.feature_selection import RFE

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

import statsmodels.formula.api as sm

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from statsmodels.regression.linear_model import OLS

from statsmodels.tools import add_constant

from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn import metrics

from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings; warnings.simplefilter('ignore')
dataset=pd.read_csv('../input/data.csv',encoding="ISO-8859-1")

dataset.describe()
print(dataset.info())



#Now, we can immediatly see that there are quite a few nulls in various columns, 

#which need work and first need a closer inspection
dataset.head()
dataset.drop(['stn_code','agency','sampling_date','location_monitoring_station'],axis=1,inplace=True)

dataset.info()

dataset.head()
#Finding missing values in the data set 

total = dataset.isnull().sum()[dataset.isnull().sum() != 0].sort_values(ascending = False)

percent = pd.Series(round(total/len(dataset)*100,2))

pd.concat([total, percent], axis=1, keys=['total_missing', 'percent'])
def remove_outlier(df_in, col_name):

    q1 = df_in[col_name].quantile(0.25)

    q3 = df_in[col_name].quantile(0.75)

    iqr = q3-q1 #Interquartile range

    fence_low  = q1-1.5*iqr

    fence_high = q3+1.5*iqr

    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]

    #return df_out



remove_outlier(dataset,'so2')

remove_outlier(dataset,'no2')

remove_outlier(dataset,'rspm')

remove_outlier(dataset,'spm')



dataset.groupby('state')[['spm','pm2_5','rspm','so2','no2']].mean()
by_State=dataset.groupby('state')



def impute_mean(series):

    return series.fillna(series.mean())



dataset['rspm']=by_State['rspm'].transform(impute_mean)

dataset['so2']=by_State['so2'].transform(impute_mean)

dataset['no2']=by_State['no2'].transform(impute_mean)

dataset['spm']=by_State['spm'].transform(impute_mean)

dataset['pm2_5']=by_State['pm2_5'].transform(impute_mean)











#imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)

#imputer = imputer.fit(dataset.groupby('state').iloc[:, 3:8].values)

#dataset.iloc[:,3:8] = imputer.transform(dataset.iloc[:, 3:8].values)

#dataset.info()

#Missing values being filled in columns

for col in dataset.columns.values:

    if dataset[col].isnull().sum() == 0:

        continue

    if col == 'date':

        guess_values = dataset.groupby('state')['date'].apply(lambda x: x.mode().max())

    elif col=='type':

        guess_values = dataset.groupby('state')['type'].apply(lambda x: x.mode().max())

    else:

        guess_values = dataset.groupby('state')['location'].apply(lambda x: x.mode().max())

dataset.head()
def calculate_si(so2):

    si=0

    if (so2<=40):

     si= so2*(50/40)

    elif (so2>40 and so2<=80):

     si= 50+(so2-40)*(50/40)

    elif (so2>80 and so2<=380):

     si= 100+(so2-80)*(100/300)

    elif (so2>380 and so2<=800):

     si= 200+(so2-380)*(100/420)

    elif (so2>800 and so2<=1600):

     si= 300+(so2-800)*(100/800)

    elif (so2>1600):

     si= 400+(so2-1600)*(100/800)

    return si

dataset['si']=dataset['so2'].apply(calculate_si)

df= dataset[['so2','si']]

df.head()
#Function to calculate no2 individual pollutant index(ni)

def calculate_ni(no2):

    ni=0

    if(no2<=40):

     ni= no2*50/40

    elif(no2>40 and no2<=80):

     ni= 50+(no2-40)*(50/40)

    elif(no2>80 and no2<=180):

     ni= 100+(no2-80)*(100/100)

    elif(no2>180 and no2<=280):

     ni= 200+(no2-180)*(100/100)

    elif(no2>280 and no2<=400):

     ni= 300+(no2-280)*(100/120)

    else:

     ni= 400+(no2-400)*(100/120)

    return ni

dataset['ni']=dataset['no2'].apply(calculate_ni)

df= dataset[['no2','ni']]

df.head()
#Function to calculate rspm individual pollutant index(rpi)

def calculate_(rspm):

    rpi=0

    if(rpi<=30):

     rpi=rpi*50/30

    elif(rpi>30 and rpi<=60):

     rpi=50+(rpi-30)*50/30

    elif(rpi>60 and rpi<=90):

     rpi=100+(rpi-60)*100/30

    elif(rpi>90 and rpi<=120):

     rpi=200+(rpi-90)*100/30

    elif(rpi>120 and rpi<=250):

     rpi=300+(rpi-120)*(100/130)

    else:

     rpi=400+(rpi-250)*(100/130)

    return rpi

dataset['rpi']=dataset['rspm'].apply(calculate_si)

df= dataset[['rspm','rpi']]

df.head()

#df.tail()
#Function to calculate spm individual pollutant index(spi)

def calculate_spi(spm):

    spi=0

    if(spm<=50):

     spi=spm*50/50

    elif(spm>50 and spm<=100):

     spi=50+(spm-50)*(50/50)

    elif(spm>100 and spm<=250):

     spi= 100+(spm-100)*(100/150)

    elif(spm>250 and spm<=350):

     spi=200+(spm-250)*(100/100)

    elif(spm>350 and spm<=430):

     spi=300+(spm-350)*(100/80)

    else:

     spi=400+(spm-430)*(100/430)

    return spi

dataset['spi']=dataset['spm'].apply(calculate_spi)

df= dataset[['spm','spi']]

df.head()

#Function to calculate pm2_5 individual pollutant index(pmi)

def calculate_pmi(pm2_5):

    pmi=0

    if(pm2_5<=50):

     pmi=pm2_5*(50/50)

    elif(pm2_5>50 and pm2_5<=100):

     pmi=50+(pm2_5-50)*(50/50)

    elif(pm2_5>100 and pm2_5<=250):

     pmi= 100+(pm2_5-100)*(100/150)

    elif(pm2_5>250 and pm2_5<=350):

     pmi=200+(pm2_5-250)*(100/100)

    elif(pm2_5>350 and pm2_5<=450):

     pmi=300+(pm2_5-350)*(100/100)

    else:

     pmi=400+(pm2_5-430)*(100/80)

    return pmi

dataset['pmi']=dataset['pm2_5'].apply(calculate_pmi)

df= dataset[['pm2_5','pmi']]

#df.tail()
from IPython.display import Image

Image(url= "http://airquality.deq.idaho.gov/Information_AQI_files/image002.jpg")
#function to calculate the air quality index (AQI) of every data value

#its is calculated as per indian govt standards

def calculate_aqi(si,ni,spi,rpi):

    aqi=0

    if(si>ni and si>spi and si>rpi):

     aqi=si

    if(spi>si and spi>ni and spi>rpi):

     aqi=spi

    if(ni>si and ni>spi and ni>rpi):

     aqi=ni

    if(rpi>si and rpi>ni and rpi>spi):

     aqi=rpi

    return aqi

dataset['AQI']=dataset.apply(lambda x:calculate_aqi(x['si'],x['ni'],x['spi'],x['rpi']),axis=1)

df= dataset[['state','si','ni','rpi','spi','AQI']]

df.head()





df.head()
#Visualization of AQI across india 



dataset['date'] = pd.to_datetime(dataset['date'],format='%Y-%m-%d') # date parse

dataset['year'] = dataset['date'].dt.year # year

dataset['year'] = dataset['year'].fillna(0.0).astype(int)

dataset = dataset[(dataset['year']>0)]



df = dataset[['AQI','year','state']].groupby(["year"]).median().reset_index().sort_values(by='year',ascending=False)

f,ax=plt.subplots(figsize=(15,10))

sns.pointplot(x='year', y='AQI', data=df)
#Exploring air pollution state-wise

dataset.fillna(0.0,inplace=True)

states=dataset.groupby(['state','location'],as_index=False).mean()

state=states.groupby(['state'],as_index=False).mean()

state
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(40,40))

state_pollution = dataset[['state','so2']].groupby('state').mean()

state_pollution.reset_index(inplace = True)

state_pollution.sort_values('so2', ascending= False, inplace = True)

state_pollution.plot(kind = 'bar', figsize= (20,10), x = 'state', fontsize= 15, title = 'States & Pollutant Levels',ax=axes[0][0])



state_pollution_no2 = dataset[['state','no2']].groupby('state').mean()

state_pollution_no2.reset_index(inplace = True)

state_pollution_no2.sort_values('no2', ascending= False, inplace = True)

state_pollution_no2.plot(kind = 'bar', figsize= (20,10), x = 'state', fontsize= 15, title = 'States & Pollutant Levels',ax=axes[0][1])



state_pollution_rspm = dataset[['state','rspm']].groupby('state').mean()

state_pollution_rspm.reset_index(inplace = True)

state_pollution_rspm.sort_values('rspm', ascending= False, inplace = True)

state_pollution_rspm.plot(kind = 'bar', figsize= (20,40), x = 'state', fontsize= 15, title = 'States & Pollutant Levels',ax=axes[1][0])





state_pollution_spm = dataset[['state','spm']].groupby('state').mean()

state_pollution_spm.reset_index(inplace = True)

state_pollution_spm.sort_values('spm', ascending= False, inplace = True)

state_pollution_spm.plot(kind = 'bar', figsize= (20,40), x = 'state', fontsize= 15, title = 'States & Pollutant Levels',ax=axes[1][1])





state_pollution = dataset[['state','pm2_5']].groupby('state').mean()

state_pollution.reset_index(inplace = True)

state_pollution.sort_values('pm2_5', ascending= False, inplace = True)

state_pollution.plot(kind = 'bar', figsize= (20,10), x = 'state', fontsize= 15, title = 'States & Pollutant Levels')



#correlation

dataset.corr()

plt.figure(figsize=(16,12))

ax=plt.axes()

#sns.heatmap(data=fifa_dataset.iloc[:,2:].corr(),annot=True,fmt='.2f',cmap='coolwarm',ax=ax)

mask = np.zeros_like(dataset.iloc[:,:].corr())

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    ax = sns.heatmap(data=dataset.iloc[:,:].corr(), mask=mask, vmax=.3, annot=True,fmt='.2f', square=True, cmap='coolwarm')

    

ax.set_title('Heatmap showing correlated values for the Dataset')

plt.show()

#Heatmap Pivot with State as Row, Year as Col, AQI as Value

dataset['date'] = pd.to_datetime(dataset['date'],format='%Y-%m-%d') # date parse

dataset['year'] = dataset['date'].dt.year # year

dataset['year'] = dataset['year'].fillna(0.0).astype(int)

dataset = dataset[(dataset['year']>0)]

f, ax = plt.subplots(figsize=(40,40))

ax.set_title('{} by state and year'.format('AQI'))

sns.heatmap(dataset.pivot_table('AQI', index='state',

                columns=['year'],aggfunc='mean',margins=True),

                annot=True,cmap="BuPu", linewidths=.75, ax=ax,cbar_kws={'label': 'Annual Average'})
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(34,20))

plt.subplots_adjust(hspace=0.4)



z = pd.Series()

for col in dataset.columns.values[3:15]:

    if ((col!='AQI')&(col!='state')&(col!='location')&(col!='type')&(col!='date')&(col!='year')&(col!='state_label')&(col!='type_label')):

      

        colums=np.array(dataset[col])

        z[col]=colums

#p=z.loc[z.index]



for i in range(2):

    for j in range(5):

        

        #x=z.index.values[i*3+j]

        #sns.barplot(z.index[i*3+j],z.values[i*3+j])

        #x=z.index.values[i*3+j]

        

        y_label=z.index[i*5+j]

        x_label=z[i*5+j]

        

        sns.regplot(data=dataset, x=z.index[i*5+j], y='AQI',ax=axes[i,j])





fig.suptitle('Distribution of Correlated Factors', fontsize='25')

plt.show()
dataset['state_label'] = dataset['state'].astype('category')

cat_columns = dataset.select_dtypes(['category']).columns

dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)

dataset["type_label"]=dataset["type"].astype(str)

dataset["type_label"] = np.where(dataset["type"].str.contains('Residential, Rural and other Areas'), 1, 0)



dataset.head()
data_p = dataset.drop(['state', 'location', 'type','date','AQI','year'],axis=1)

corr = data_p.corr()

columns = np.full((corr.shape[0],), True, dtype=bool)

selected_column=data_p.columns[columns]                

data_p=data_p[selected_column]



x_=data_p

y=dataset['AQI']

#factors with p-value

selected_columns_1 = selected_column[0:10].values

selected_columns_2=selected_column[11:].values

selected_columns=np.concatenate((selected_columns_1,selected_columns_2),axis=0)

import statsmodels.formula.api as smf

def backwardelimination(x, Y, sl, columns):

    numVars = len(x[0])

    

    for i in range(0, numVars):

        regressor_OLS = smf.OLS(Y, x).fit()

        maxVar = max(regressor_OLS.pvalues).astype(float)

        

        if maxVar > sl:

          

            for j in range(0, numVars - i):

                if (regressor_OLS.pvalues[j].astype(float) == maxVar):

                    x = np.delete(x,j,1)

                    columns = np.delete(columns, j)

                    

    regressor_OLS.summary()

    return x, columns

SL = 0.05

data_modeled, selected_columns = backwardelimination(x_.iloc[:,:].values, y.values, SL,selected_columns )





result = pd.DataFrame()

result['AQI'] = dataset.iloc[:,10]

#result.head()



data_p_selected = pd.DataFrame(data= data_modeled[:,0:9], columns = selected_columns)

data_p_selected.head()
def linear_regression(X,y):

#SPLIT TEST AND TRAIN

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)



#One Hot Encoding

    X_train = pd.get_dummies(X_train)

    X_test = pd.get_dummies(X_test)



#Linear Regression

    LR = LinearRegression()

    LR.fit(X_train, y_train)

    predictions = LR.predict(X_test)



    print(X_test.shape,X_train.shape,y_test.shape,y_train.shape)

    print('r2_Square:%.2f '% r2_score(y_test, predictions))

    print('MSE:%.2f '% np.sqrt(mean_squared_error(y_test, predictions)))

    print("KfoldCrossVal mean score using Linear regression is %s" %cross_val_score(LR,X_train,y_train,cv=10).mean())

    

    regressor_OLS = smf.OLS(y_train, X_train).fit()

    

    plt.figure(figsize=(18,10))

    plt.scatter(predictions,y_test,alpha = 0.3)

    plt.xlabel('Predictions')

    plt.ylabel('AQI')

    plt.title("Linear Prediction ")

    plt.show()

#cross validation    

    Kfold = KFold(len(X), shuffle=True)

    #X_train = sc.fit_transform(X_train)

    #X_test = sc.transform(X_test)

    z=print(regressor_OLS.summary())

    return z



   

        

    

    

    
####function to calculate cross validation score only

def cross_val(X,y):

    #SPLIT TEST AND TRAIN

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)



#One Hot Encoding

    X_train = pd.get_dummies(X_train)

    X_test = pd.get_dummies(X_test)



#Linear Regression

    LR = LinearRegression()

    LR.fit(X_train, y_train)

    predictions = LR.predict(X_test)

    Kfold = KFold(len(X), shuffle=True)

    print("KfoldCrossVal mean score using Linear regression is %s" %cross_val_score(LR,X_train,y_train,cv=10).mean())

    
X_1=dataset[['si','ni','rpi','spi']]

y_1=dataset['AQI']

linear_regression(X_1,y_1)
#Linear Regression

X_2= dataset.drop([ 'AQI','state','location','type','rspm','si','ni','rpi','spi','pmi','date','year','state_label','type_label'], axis = 1)

y_2 = dataset['AQI']

linear_regression(X_2,y_2)

#regressor_OLS = smf.OLS(y_train, X_train).fit()

#print("Model Summary")

#regressor_OLS.summary()

#Linear Regression

X_3= dataset.drop(['AQI','state','location','type','date','pmi','pm2_5','year','state_label','type_label','so2','no2','si','ni'], axis = 1)

y_3 = dataset['AQI']

linear_regression(X_3,y_3)

#SPLIT TEST AND TRAIN

#X_train, X_test, y_train, y_test = train_test_split(dataset_LR_3, target, test_size=0.2)



#One Hot Encoding

#X_train = pd.get_dummies(X_train)

#X_test = pd.get_dummies(X_test)

#print(X_test.shape,X_train.shape,y_test.shape,y_train.shape)
def AQI_Range(x):

    if x<=50:

        return "Good"

    elif x>50 and x<=100:

        return "Moderate"

    elif x>100 and x<=200:

        return "Unhealthy for sensitive groups"

    elif x>200 and x<=300:

        return "Unhealthy"

    elif x>300 and x<=400:

        return "Very unhealthy"

    elif x>400:

        return "Hazardous"



dataset['AQI_Range'] = dataset['AQI'] .apply(AQI_Range)

dataset.head()

dataset['AQI_label'] = dataset['AQI_Range'].astype('category')

cat_columns = dataset.select_dtypes(['category']).columns

dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)



dataset.head()

def AQI_Range_Binary_Label(x):

    if x<=200:

        return 0

    elif x>200:

        return 1

    

dataset['AQI_Range_Binary'] = dataset['AQI'] .apply(AQI_Range_Binary_Label)

dataset.head()
dataset_box=dataset.drop(['state', 'location', 'date','state_label',

       'type_label','type','AQI_label','AQI_Range_Binary','year','AQI_Range'],axis=1)

f, ax = plt.subplots(figsize=(20, 15))



ax.set_facecolor('#FFFFFF')

plt.title("Box Plot AQI Dataset Scaled")

ax.set(xlim=(-10, 250))

ax = sns.boxplot(data = dataset_box, 

  orient = 'h', 

  palette = 'Set3')
predictor_names=dataset_box.columns.get_values()

predictor_names=predictor_names.tolist()

predictor_names.pop()

predictor_names

def rank_predictors(dat,l,f='AQI_Range_Binary'):

    rank={}

    max_vals=dat.max()

    median_vals=dat.groupby(f).median()  # We are using the median 

    for p in l:

        score=np.abs((median_vals[p][1]-median_vals[p][0])/max_vals[p])

        rank[p]=score

    return rank

cat_rank=rank_predictors(dataset,predictor_names) 

cat_rank
# Take the top predictors based on mean difference

cat_rank=sorted(cat_rank.items(), key=lambda x: x[1],reverse= True)



ranked_predictors=[]

for f in cat_rank:

    ranked_predictors.append(f[0])

ranked_predictors



data_log=dataset.drop(['state','type','date','location'],axis=1)

def logistic_regression(x,y):

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

    sc = StandardScaler()

    

    # Feature scaling

    x_train = sc.fit_transform(x_train)

    x_test = sc.fit_transform(x_test)

    

    #Fitting logistic regression to the training set

    classifier = LogisticRegression(random_state = 0)

    classifier.fit(x_train,y_train)

    

    

    # Logistic regression cross validation

    #Kfold = KFold(len(ranked_predictors), shuffle=False)

    #print("KfoldCrossVal mean score using Logistic regression is %s \n" %cross_val_score(classifier,x,y,cv=10).mean())

    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

    cvs=cross_val_score(classifier, x_train, y_train, cv=k_fold).mean()

    print("KfoldCrossVal mean score using Logistic regression is %s \n"%cvs)

    

    print("Logistic Analysis Report")

    y_pred = classifier.predict(x_test)

    print(classification_report(y_test,y_pred))

    print(y_pred)

    #Accuracy score

    print ("Accuracy Score:%.2f" % metrics.accuracy_score(y_test,classifier.predict(x_test)))

    #probabilty of dependent variable

    y_pred_proba = classifier.predict_proba(x_test)[::,1]

    print('Probabilty of dependent variable')

    print(y_pred_proba.mean())

    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

    plt.plot([0, 1], [0, 1],'r--')

    plt.legend(loc=4)

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.show()

   

    



def logit_summary(y,X):

    logit_model=sm.Logit(y,X)

    result=logit_model.fit()

    print("Model Summary")

    print(result.summary2())
data_log_1=dataset.drop(['state','type','date','location','AQI_label','AQI_Range','AQI','AQI_Range_Binary'],axis=1)

x=data_log_1.iloc[:,:]

y=data_log.iloc[:,16]



logistic_regression(x,y)

logit_summary(y,x)
#Logistic Regression Model 2

data_log_2=dataset.drop(['state','type','date','location','AQI_Range','type_label','AQI_label','AQI'],axis=1)

x=data_log_2.iloc[:,:]

y=dataset['type_label']

logistic_regression(x,y)

logit_summary(y,x)
x=dataset[predictor_names]

y=data_log.iloc[:,13]

logistic_regression(x,y)

logit_summary(y,x)
def variance_IF(X):

    vif=vif = pd.DataFrame()

    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif["features"] = X.columns

    return vif

##For first Linear Model

variance_IF(X_1)
##For second Linear Model

variance_IF(X_2)
##For third Linear Model

variance_IF(X_3)
#Linear Model 1:-

X_M1=X_1[['si','ni']]

print('Linear Model 1')

cross_val(X_M1,y_1)
#Linear Model 2:-

print('Linear Model 2')

cross_val(X_2,y_2)
#Linear Model 2:-

print('Linear Model 3')

cross_val(X_3,y_3)
#https://datascience.stackexchange.com/questions/937/does-scikit-learn-have-forward-selection-stepwise-regression-algorithm

import statsmodels.api as sm



X=X_2.astype(float)

y=dataset['AQI']





def stepwise_selection(X, y, 

                       initial_list=[], 

                       threshold_in=0.01, 

                       threshold_out = 0.05, 

                       verbose=True):

    

    included = list(initial_list)

    while True:

        changed=False

        # forward step

        excluded = list(set(X.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.argmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))



        # backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()

        # use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() # null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included



result = stepwise_selection(X, y)



print('resulting features:')

print(result)

dataset_LR_2=dataset[['so2', 'spm', 'no2', 'pm2_5']]

target=dataset['AQI']

linear_regression(dataset_LR_2,target)

#X_train, X_test, y_train, y_test = train_test_split(dataset_LR_2, target, test_size=0.2)

def evaluateModel (model):

    print("RSS = ", ((dataset.AQI - model.predict())**2).sum())

    print("R2 = ", model.rsquared)
modelAll = smf.ols('AQI ~ no2 + rspm + spm', dataset).fit()

print(modelAll.summary().tables[1])

evaluateModel(modelAll)
modelNo2_rspm = smf.ols('AQI ~ no2 + rspm ', dataset).fit()

print(modelNo2_rspm.summary().tables[1])

evaluateModel(modelNo2_rspm)
modelNo2_spm = smf.ols('AQI ~ no2 + spm ', dataset).fit()

print(modelNo2_spm.summary().tables[1])

evaluateModel(modelNo2_spm)
modelspm_rspm = smf.ols('AQI ~ spm + rspm ', dataset).fit()

print(modelspm_rspm.summary().tables[1])

evaluateModel(modelspm_rspm)

model_spm = smf.ols('AQI ~ spm', dataset).fit()

print(model_spm.summary().tables[1])

evaluateModel(model_spm)
model_no2 = smf.ols('AQI ~ no2', dataset).fit()

print(model_no2.summary().tables[1])

evaluateModel(model_no2)
model_rspm = smf.ols('AQI ~ rspm', dataset).fit()

print(model_rspm.summary().tables[1])

evaluateModel(model_rspm)
modelSynergy = smf.ols('AQI ~ spm + rspm + spm*rspm', dataset).fit()

print(modelSynergy.summary().tables[1])

evaluateModel(modelSynergy)
modelSynergy = smf.ols('AQI ~ no2 + spm + no2*spm', dataset).fit()

print(modelSynergy.summary().tables[1])

evaluateModel(modelSynergy)


#Regularization:- l2



from sklearn.linear_model import Ridge

#for i in range(0, 1): Matrix

X_train, X_test, y_train, y_test = train_test_split(X_1,y_1, test_size=0.2)

X_train = pd.get_dummies(X_train)

X_test = pd.get_dummies(X_test)



ridgeReg = Ridge(alpha=0.05, normalize=True)

ridgeReg.fit(X_train,y_train)

pred = ridgeReg.predict(X_test)



print(X_test.shape,X_train.shape,y_test.shape,y_train.shape)

print('r2_Square:%.2f '% r2_score(y_test, pred))

print('MSE:%.2f '% np.sqrt(mean_squared_error(y_test, pred)))



regressor_OLS = smf.OLS(y_train, X_train).fit()



plt.figure(figsize=(18,10))

plt.scatter(pred,y_test,alpha = 0.3)

plt.xlabel('Predictions')

plt.ylabel('AQI')

plt.title("Linear Prediction ")

plt.show()

#cross validation    

Kfold = KFold(len(X_1), shuffle=True)

    #X_train = sc.fit_transform(X_train)

    #X_test = sc.transform(X_test)

print("KfoldCrossVal mean score using Linear regression is %s" %cross_val_score(ridgeReg,X_train,y_train,cv=10).mean())





print(regressor_OLS.summary())











from sklearn.linear_model import Ridge



# Create an array of alphas and lists to store scores

alpha_space = np.logspace(-4, 0, 50)

ridge_scores = []

ridge_scores_std = []



# Create a ridge regressor: ridge

ridge = Ridge(normalize=True)



# Compute scores over range of alphas

for alpha in alpha_space:



    # Specify the alpha value to use: ridge.alpha

    ridge.alpha = alpha

    

    # Perform 10-fold CV: ridge_cv_scores

    ridge_cv_scores = cross_val_score(ridge, X_train, y_train, cv=10)

    

    # Append the mean of ridge_cv_scores to ridge_scores

    ridge_scores.append(np.mean(ridge_cv_scores))

    

    # Append the std of ridge_cv_scores to ridge_scores_std

    ridge_scores_std.append(np.std(ridge_cv_scores))



# Use this function to create a plot    

def display_plot(cv_scores, cv_scores_std):

    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)

    ax.plot(alpha_space, cv_scores)

    

    std_error = cv_scores_std / np.sqrt(10)



    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)

    ax.set_ylabel('CV Score +/- Std Error')

    ax.set_xlabel('Alpha')

    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')

    ax.set_xlim([alpha_space[0], alpha_space[-1]])

    ax.set_xscale('log')

    plt.show()



# Display the plot

display_plot(ridge_scores, ridge_scores_std)
"""X_train, X_test, y_train, y_test = train_test_split(X_3,dataset['AQI'], test_size=0.2)

X_train = pd.get_dummies(X_train)

X_test = pd.get_dummies(X_test)



lassoReg = Lasso(alpha=0.3, normalize=True)

lassoReg.fit(X_train,y_train)



print(X_test.shape,X_train.shape,y_test.shape,y_train.shape)

print('r2_Square:%.2f '% r2_score(y_test, pred))

print('MSE:%.2f '% np.sqrt(mean_squared_error(y_test, pred)))



regressor_OLS = smf.OLS(y_train, X_train).fit()



plt.figure(figsize=(18,10))

plt.scatter(pred,y_test,alpha = 0.3)

plt.xlabel('Predictions')

plt.ylabel('AQI')

plt.title("Linear Prediction ")

plt.show()

#cross validation    

Kfold = KFold(len(X), shuffle=True)

    #X_train = sc.fit_transform(X_train)

    #X_test = sc.transform(X_test)

print("KfoldCrossVal mean score using Linear regression is %s" %cross_val_score(lassoReg,X_train,y_train,cv=10).mean())





regressor_OLS.summary()"""