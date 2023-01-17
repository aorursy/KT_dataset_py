from IPython.display import HTML

from IPython.display import Image

Image(url= "http://www.techhuman.com/wp-content/uploads/2018/10/FIFA-19-Wallpaper-4-1080x675.jpg")
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
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#from pandas.plotting import scatter_matrix as sm



import os



import seaborn as sns



import matplotlib.pyplot as plt

from matplotlib import cm as cm



import scipy.stats  as stats



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.feature_selection import RFE

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report

from sklearn import metrics

from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



import statsmodels.formula.api as smf

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor



import warnings; warnings.simplefilter('ignore')



fifa_dataset = pd.read_csv('../input/FullData.csv',encoding="ISO-8859-1")
fifa_dataset.head()
fifa_dataset.drop(['Index','ID','Photo','Flag','Club Logo','Real Face','Jersey Number','Joined','Loaned From','Contract Valid Until','LS','ST','RS','LW','LF','CF','RF','RW','LAM','CAM','RAM','LM','LCM','CM','RCM','RM','LWB','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB'],axis=1,inplace=True)

#'Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling','Curve','FKAccuracy','LongPassing','BallControl','Acceleration','SprintSpeed','Agility','Reactions','Balance','ShotPower','Jumping','Stamina','Strength','LongShots','Aggression','Interceptions','Positioning','Vision','Penalties','Composure','Marking','StandingTackle','SlidingTackle','GKDiving','GKHandling','GKKicking','GKPositioning','GKReflexes'

fifa_dataset.head()
fifa_dataset.describe()
fifa_dataset.info()
print("Are there Null Values in the dataset? ")

fifa_dataset.isnull().values.any()
total = fifa_dataset.isnull().sum()[fifa_dataset.isnull().sum() != 0].sort_values(ascending = False)

percent = pd.Series(round(total/len(fifa_dataset)*100,2))

pd.concat([total, percent], axis=1, keys=['total_missing', 'percent'])
def value_to_float(x):

    if type(x) == float or type(x) == int:

        return x

    if 'K' in x:

        if len(x) > 1:

            return float(x.replace('K', '')) * 1000

        return 1000.0

    if 'M' in x:

        if len(x) > 1:

            return float(x.replace('M', '')) * 1000000

        return 1000000.0

    return 0.0

fifa_dataset['Release Clause'] = fifa_dataset['Release Clause'].str.replace('â\x82¬', '')  

fifa_dataset['Release Clause'] = fifa_dataset['Release Clause'] .apply(value_to_float)


fifa_dataset['Wage'] = fifa_dataset['Wage'].str.replace('â\x82¬', '')      

fifa_dataset['Wage'] = fifa_dataset['Wage'] .apply(value_to_float)
fifa_dataset['Value'] = fifa_dataset['Value'].str.replace('â\x82¬', '')

fifa_dataset['Value'] = fifa_dataset['Value'] .apply(value_to_float)
x=np.array(fifa_dataset['Release Clause'])

Release_Clause_no_nan= fifa_dataset['Release Clause'][~np.isnan(fifa_dataset['Release Clause'])]

y=np.array(Release_Clause_no_nan)



z=fifa_dataset[['Value','Release Clause','Wage']]

print (z)
#Filling Missing Values

for col in fifa_dataset.columns.values:

    if fifa_dataset[col].isnull().sum()==0:

        continue

    if col == 'Release Clause':

        guess_values = fifa_dataset['Release Clause'].fillna(value=fifa_dataset['Value'])

    else:

        guess_values = fifa_dataset['Club'].apply(lambda x: 'No Club')

        

    fifa_dataset[col].loc[(fifa_dataset[col].isnull())] = guess_values

    

   

     #if fifa_dataset['Release Clause'].isna().any():

    #fifa_dataset['Release Clause']. = fifa_dataset['Release Clause'].apply(lambda x: y+1)



    

 
fifa_dataset.isnull().values.any()
#correlation

fifa_dataset.corr()

plt.figure(figsize=(40,30))

ax=plt.axes()

#sns.heatmap(data=fifa_dataset.iloc[:,2:].corr(),annot=True,fmt='.2f',cmap='coolwarm',ax=ax)

mask = np.zeros_like(fifa_dataset.iloc[:,:].corr())

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    ax = sns.heatmap(data=fifa_dataset.iloc[:,:].corr(), mask=mask, vmax=.3, annot=True,fmt='.2f', square=True, cmap='coolwarm')

    

ax.set_title('Heatmap showing correlated values for the Dataset')

plt.show()

fifa_data_p = fifa_dataset.drop(['Name', 'Nationality', 'Club','Position','Work Rate', 'Preferred Foot' ,'Body Type', 'Height', 'Weight'],axis=1)



corr = fifa_data_p.corr()



columns = np.full((corr.shape[0],), True, dtype=bool)



for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.9:

            if columns[j]:

                columns[j] = False

selected_column=fifa_data_p.columns[columns]                

fifa_data_p=fifa_data_p[selected_column]
def general_position(x):

    if type(x) == str:

        

        if x=="LS" or x=="RS" or x=="ST":

            return "Striker"

        elif x=="LF" or x=="CF" or x=="RF":

            return "Forward"

        elif x=="LW" or x=="RW":

            return "Winger"

        elif x=="LAM" or x=="CAM" or x=="RAM":

            return "Attacking Midfielder"

        elif x=="LM" or x=="CM" or x=="RM" or x=="LCM" or x=="RCM":

            return "Central Midfielder"

        elif x=="LDM" or x=="RDM" or x=="CDM":

            return "Holding Midfielder"

        elif x=="LWB" or x=="RWB":

            return "WingBack"

        elif x=="LB" or x=="RB":

            return "Fullback"

        elif x=="RCB" or x=="LCB" or x=="CB":

            return "Defender"

        else:

            return "GoalKeeper"

            

        

fifa_dataset['GeneralPosition'] = fifa_dataset['Position'] .apply(general_position)





#at least one independent variable needs to be a multi-class categorical variable. and its conversion to numeric data

fifa_dataset['GP_Label'] = fifa_dataset['GeneralPosition'].astype('category')

cat_columns = fifa_dataset.select_dtypes(['category']).columns

fifa_dataset[cat_columns] = fifa_dataset[cat_columns].apply(lambda x: x.cat.codes)

fifa_dataset["PF_label"]=fifa_dataset["Preferred Foot"].astype(str)

fifa_dataset["PF_label"] = np.where(fifa_dataset["Preferred Foot"].str.contains('Left'), 1, 0)

fifa_dataset.head()
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(34,20))

plt.subplots_adjust(hspace=0.4)



fifa_new_3= fifa_dataset.drop(['Age', 'Special', 'Name', 'Club', 'Body Type', 'Position', 'Work Rate' ,'GeneralPosition', 'Height', 'Weight', 'Nationality','Preferred Foot', 'Weak Foot', 'Skill Moves', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes', 'PF_label','GP_Label'], axis = 1) 



z = pd.Series()

for col in fifa_new_3.columns.values[:]:

    if(col!='Overall'):

        colums=np.array(fifa_new_3[col])

        z[col]=colums

#p=z.loc[z.index]



for i in range(2):

    for j in range(3):

        

        #x=z.index.values[i*3+j]

        #sns.barplot(z.index[i*3+j],z.values[i*3+j])

        #x=z.index.values[i*3+j]

        

        y_label=z.index[i*3+j]

        x_label=z[i*3+j]

        

        sns.regplot(data=fifa_new_3, x=z.index[i*3+j], y='Overall',ax=axes[i,j])





fig.suptitle('Univariate Distribution of Positively Correlated Factors', fontsize='25')

plt.show()
x_=fifa_data_p.drop(['Overall'],axis=1)

x_.columns
#factors on the basis of p-value

selected_columns_1 = selected_column[0:1].values

selected_columns_2 = selected_column[2:].values

selected_columns = np.concatenate((selected_columns_1,selected_columns_2),axis=0)





def backwardelimination(x, Y, sl, columns):

    numVars = len(x[0])

    

    for i in range(0, numVars):

        regressor_OLS = smf.OLS(Y, x).fit()

        maxVar = max(regressor_OLS.pvalues).astype(float)

        

        if maxVar > sl:

            

            for j in range(0, numVars - i):

                if (regressor_OLS.pvalues[j].astype(float) == maxVar):

                    x = np.delete(x, j,1)

                    columns = np.delete(columns, j)

                    



    regressor_OLS.summary()

    return x, columns

SL = 0.05



Y=fifa_dataset['Overall'].values

data_modeled, selected_columns = backwardelimination(x_.values,fifa_data_p['Overall'].values, SL,selected_columns )

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



    regressor_OLS = smf.OLS(y_train, X_train).fit()

    

    plt.figure(figsize=(18,10))

    plt.scatter(predictions,y_test,alpha = 0.3)

    plt.xlabel('Predictions')

    plt.ylabel('Overall')

    plt.title("Linear Prediction ")

    plt.show()

#cross validation    

    Kfold = KFold(len(X), shuffle=True)

    #X_train = sc.fit_transform(X_train)

    #X_test = sc.transform(X_test)

    print("KfoldCrossVal mean score using Linear regression is %s" %cross_val_score(LR,X_train,y_train,cv=10).mean())

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
#Linear Regression 1



X_1= fifa_dataset.drop(['Overall','Preferred Foot','Name','Nationality','Club','Work Rate','GeneralPosition','Body Type','Weight','Height','Wage','Position','Value'], axis = 1)

y_1= fifa_dataset['Overall']

linear_regression(X_1,y_1)

#SPLIT TEST AND TRAIN

#X_train, X_test, y_train, y_test = train_test_split(fifa_new_LR_1, target, test_size=0.2)



#One Hot Encoding

#X_train = pd.get_dummies(X_train)

#X_test = pd.get_dummies(X_test)

#print(X_test.shape,X_train.shape,y_test.shape,y_train.shape)
y_2 = pd.DataFrame()

y_2['Overall'] = fifa_dataset.iloc[:,3]

y_2.head()

X_2 = pd.DataFrame(data= data_modeled, columns = selected_columns)

X_2.head()

linear_regression(X_2,y_2)
#Linear Regression with min p-value

X_3= fifa_dataset.drop(['Overall','Age', 'Special', 'Name', 'Club', 'Body Type', 'Position', 'Work Rate' , 'Height', 'Weight', 'Nationality','Preferred Foot', 'Skill Moves', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',  'SprintSpeed', 'Jumping',  'Aggression', 'Interceptions', 'Positioning', 'Vision',  'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause', 'GeneralPosition', 'GP_Label', 'PF_label'], axis = 1) 

#'Potential','Value','Wage','International Reputation','Reaction','Release Clause'

y_3 = fifa_dataset['Overall']

linear_regression(X_3,y_3)
#Linear Regression 4



X_4= fifa_dataset.drop(['Overall','Age', 'Name', 'Club', 'Body Type', 'Position', 'Work Rate' ,'GeneralPosition', 'Height', 'Weight', 'Nationality','Preferred Foot', 'Weak Foot', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',  'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Balance',  'Jumping', 'Stamina', 'Strength', 'Aggression', 'Interceptions', 'Positioning',  'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes'], axis = 1) 

#'Reactions','Vision','Composure','Crossing','Dribbling','ShotPower','LongShots','Penalties','Value','Release Clause','Wage','Potential'

#'Potential','Value','Wage','International Reputation','Reaction','Release Clause'

y_4 = fifa_dataset['Overall']

linear_regression(X_4,y_4)

#SPLIT TEST AND TRAIN

#X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(fifa_new_LR_3, target, test_size=0.2)



#One Hot Encoding

#X_train = pd.get_dummies(X_train)

#X_test = pd.get_dummies(X_test)

#print(X_test_3.shape,X_train_3.shape,y_test_3.shape,y_train_3.shape)
fifa_dataset_box=fifa_dataset.drop(['Name', 'Nationality', 'Club',  'Preferred Foot',  'Work Rate', 'Body Type', 'Position',

     'GeneralPosition', 'GP_Label', 'PF_label','Special','Value','International Reputation','Height','Weight',

       'Weak Foot', 'Skill Moves',

       'Wage', 'Release Clause'],axis=1)

f, ax = plt.subplots(figsize=(20, 30))



ax.set_facecolor('#FFFFFF')

plt.title("Box Plot AQI Dataset Scaled")

ax.set(xlim=(0, 100))

ax = sns.boxplot(data = fifa_dataset_box, 

  orient = 'h', 

  palette = 'Set3')
predictor_names=fifa_dataset_box.columns.get_values()

predictor_names=predictor_names.tolist()

predictor_names
def rank_predictors(dat,l,f='PF_label'):

    rank={}

    max_vals=dat.max()

    mean_vals=dat.groupby(f).mean()  # We are using the mean 

    for p in l:

        score=np.abs((mean_vals[p][1]-mean_vals[p][0])/max_vals[p])

        rank[p]=score

    return rank

cat_rank=rank_predictors(fifa_dataset,predictor_names) 

cat_rank
# Take the top predictors based on median difference

cat_rank=sorted(cat_rank.items(), key=lambda x: x[1],reverse= True)



ranked_predictors=[]

for f in cat_rank:

    ranked_predictors.append(f[0])

ranked_predictors



fifa_new_LG_3= fifa_dataset.drop(['Overall','Age', 'Name', 'Club', 'Body Type', 'Position', 'Work Rate' , 'Height', 'Weight', 'Nationality','Preferred Foot', 'Weak Foot', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',  'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Balance',  'Jumping', 'Stamina', 'Strength', 'Aggression', 'Interceptions', 'Positioning',  'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes'], axis = 1) 



def attacking_position(x):

    if type(x) == str:

        

        if x=="GK" or x=="RCB" or x=="LCB" or x=="CB" or x=="RCB" or x=="LCB" or x=="CB" or x=="LB" or x=="RB" or x=="LDM" or x=="RDM" or x=="CDM":

            return 0

        else:

            return 1     

        

fifa_dataset['AttackingPosition'] = fifa_dataset['Position'] .apply(attacking_position)

fifa_dataset.head()
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

    

    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

    cvs=cross_val_score(classifier, x_train, y_train, cv=k_fold).mean()

    print("KfoldCrossVal mean score using Logistic regression is %s \n"%cvs)

    

    

    

    print("Logistic Analysis Report")

    y_pred = classifier.predict(x_test)

    print(classification_report(y_test,y_pred))

    print(y_pred)

    print ("Accuracy Score:%.2f" % metrics.accuracy_score(y_test,classifier.predict(x_test)))

    

    

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

    

# Predicting the Test set results

#Precision – Accuracy of positive predictions.

#Precision = TP/(TP + FP)



#FN – False Negatives

#Recall (aka sensitivity or true positive rate): Fraction of positives That were correctly identified.

#Recall = TP/(TP+FN)



#F1 Score (aka F-Score or F-Measure) – A helpful metric for comparing two classifiers. F1 Score takes into account precision and the recall. It is created by finding the the harmonic mean of precision and recall.

#F1 = 2 x (precision x recall)/(precision + recall)

def logit_summary(y,X):

    logit_model=sm.Logit(y,X)

    result=logit_model.fit()

    print("Model Summary")

    print(result.summary2())
fifa_log = fifa_dataset.drop(['Club','Preferred Foot','GeneralPosition','Work Rate','Body Type','Position','Height', 'Weight'],axis=1)

fifa_log.head()
x=fifa_log.iloc[:,7:46]

#w=fifa_log.iloc[:,47]

#z.insert(loc=39, column='PF_label', value=w)



y=fifa_log.iloc[:,47]

logistic_regression(x,y)

logit_summary(y,x)
x=fifa_log.iloc[:,7:46]

#w=fifa_log.iloc[:,47]

#x.insert(loc=39, column='PF_label', value=w)

y=fifa_log.iloc[:,48]

logistic_regression(x,y)

logit_summary(y,x)
x=fifa_log[predictor_names]

y=fifa_log.iloc[:,47]

logistic_regression(x,y)

logit_summary(y,x)
## high variance inflation factor because they "explain" the same variance within this dataset. We would need to discard one of these variables before 

#moving on to model building or risk building a model with high multicolinearity.

def variance_IF(X):

    vif=vif = pd.DataFrame()

    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif["features"] = X.columns

    return vif

variance_IF(X_1)
variance_IF(X_2)
variance_IF(X_3)
variance_IF(X_4)
#Linear Model 1:-

##X_M1=X_1[[]]

print('Linear Model 1')

cross_val(X_1,y_1)
#Linear Model 2:-

print('Linear Model 2')

cross_val(X_2,y_2)
#Linear Model 3:-

print('Linear Model 3')

cross_val(X_3,y_3)
#Linear Model 4:-

##X_M1=X_1[[]]

print('Linear Model 1')

cross_val(X_4,y_4)
#https://datascience.stackexchange.com/questions/937/does-scikit-learn-have-forward-selection-stepwise-regression-algorithm



#X=fifa_dataset.drop(['date','state', 'location', 'type','AQI_Range'],axis=1)

X=X_1.astype(float)

y=fifa_dataset['Overall']





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



print('resulting features:',result)



StepwiseCrossValidate=fifa_dataset[['Reactions', 'Release Clause', 'International Reputation', 'Potential', 'Age', 'Stamina', 'GKDiving', 'Special', 'HeadingAccuracy', 'Balance', 'SprintSpeed', 'ShortPassing', 'Positioning', 'Composure', 'GKHandling', 'ShotPower', 'GP_Label', 'BallControl', 'Vision', 'GKPositioning', 'Skill Moves', 'LongShots']]

target=fifa_dataset['Overall']

linear_regression(StepwiseCrossValidate,target)
def evaluateModel (model):

    print("RSS = ", ((fifa_dataset.Overall - model.predict())**2).sum())

    print("R2 = ", model.rsquared)

    

fifa_dataset['InternationalReputation']=fifa_dataset['International Reputation']

modelAll = smf.ols('Overall ~ Age + InternationalReputation + Potential', fifa_dataset).fit()

print(modelAll.summary().tables[1])

evaluateModel (modelAll)
modelIntR = smf.ols('Overall ~ InternationalReputation', fifa_dataset).fit()

print(modelIntR.summary().tables[1])

evaluateModel (modelIntR)
modelAge = smf.ols('Overall ~ Age', fifa_dataset).fit()

print(modelAge.summary().tables[1])

evaluateModel (modelAge)
modelPot = smf.ols('Overall ~ Potential', fifa_dataset).fit()

print(modelPot.summary().tables[1])

evaluateModel (modelPot)
modelIntRAge = smf.ols('Overall ~ InternationalReputation + Age', fifa_dataset).fit()

print(modelIntRAge.summary().tables[1])

evaluateModel (modelIntRAge)
modelIntRPot = smf.ols('Overall ~ InternationalReputation + Potential', fifa_dataset).fit()

print(modelIntRPot.summary().tables[1])

evaluateModel (modelIntRPot)
modelAgePot = smf.ols('Overall ~ Age + Potential', fifa_dataset).fit()

print(modelAgePot.summary().tables[1])

evaluateModel (modelAgePot)
modelAgePot = smf.ols('Overall ~ Age + Potential + Age*Potential', fifa_dataset).fit()

print(modelAgePot.summary().tables[1])

evaluateModel (modelAgePot)
#Ridge



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





regressor_OLS.summary()



#Lasso

#X_train, X_test, y_train, y_test = train_test_split(X_3,y_3, test_size=0.2)

#X_train = pd.get_dummies(X_train)

#X_test = pd.get_dummies(X_test)



#lassoReg = Lasso(alpha=0.3, normalize=True)

#lassoReg.fit(X_train,y_train)



#print(X_test.shape,X_train.shape,y_test.shape,y_train.shape)

#print('r2_Square:%.2f '% r2_score(y_test, pred))

#print('MSE:%.2f '% np.sqrt(mean_squared_error(y_test, pred)))



#regressor_OLS = smf.OLS(y_train, X_train).fit()



#plt.figure(figsize=(18,10))

#plt.scatter(pred,y_test,alpha = 0.3)

#plt.xlabel('Predictions')

#plt.ylabel('AQI')

#plt.title("Linear Prediction ")

#plt.show()

#cross validation    

#Kfold = KFold(len(X_1), shuffle=True)

    #X_train = sc.fit_transform(X_train)

    #X_test = sc.transform(X_test)

#print("KfoldCrossVal mean score using Linear regression is %s" %cross_val_score(lassoReg,X_train,y_train,cv=10).mean())





#regressor_OLS.summary()