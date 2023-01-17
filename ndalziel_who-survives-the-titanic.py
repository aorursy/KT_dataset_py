# Load packages

import warnings

warnings.simplefilter(action='ignore')                             # suppress warnings

import numpy as np                                                 # linear algebra

import pandas as pd                                                # data analysis

import matplotlib.pyplot as plt                                    # visualization

import seaborn as sns                                              # visualization

from sklearn.svm import SVC                                        # Support Vector Machine classifier

from sklearn.naive_bayes import GaussianNB                         # Naive Bayes classifier

from sklearn.ensemble import RandomForestClassifier                # ensemble classifier

from sklearn.preprocessing import StandardScaler                   # scaler (for SVM model)

from sklearn.model_selection import GridSearchCV, cross_val_score  # parameter tuning

pd.set_option('display.float_format', lambda x: '%.0f' % x)        # format decimals

sns.set(font_scale=1.5) # increse font size for seaborn charts

%matplotlib inline

RANDOM_STATE = 42
def read_data(filename):

    df = pd.read_csv(filename,index_col="PassengerId")

    return df

train = read_data('../input/train.csv')

test = read_data('../input/test.csv')

train.head()
train['Sex'].groupby(train['Sex']).count()
train[train['Survived']==1]['Survived'].groupby(train['Sex']).count()
def PredictorPlot(df,var,freqvar,freqvalues,ticks=True, print_table=False,show_title=True):

    if  var in df.columns:

        df2 = df.loc[~df[freqvar].isnull()]

        n = len(freqvalues)

        Freq = []

        Pcent = []

        Total = df2[var].groupby(df2[var]).count()

    

        for i in range (0,n):

            Freq.append( df2[df2[freqvar]==freqvalues[i]][freqvar].groupby(df2[var]).count() )

            Pcent.append ( Freq[i].div(Total,fill_value=0))

            

        df3 = Pcent[0]

        for i in range (1,n):

            df3 = pd.concat([df3, Pcent[i]], axis=1)

            

        if print_table == True: print (df3)

    

        ax = df3.plot.bar(stacked=True,legend=False,figsize={16,5},colormap = 'RdYlGn',xticks=None)

        ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in ax.get_yticks()])

        if show_title == True: plt.title('Percentage of Survivors by Passenger Type (green=Survived)');

        w = 0.5

        if ticks==False:

            ax.xaxis.set_ticks([])

            w = 1

        for container in ax.containers:

            plt.setp(container, width=w)
PredictorPlot(df = train,var ='Sex',freqvar = 'Survived', freqvalues = [0,1])
def processSex(df,dropSex=True):

    df["Female"] = df["Sex"].apply(lambda sex: 0 if sex == "male" else 1)

    if dropSex == True: df = df.drop('Sex',axis=1)  

    return df



Xy_train_df = processSex(train)

X_test_df = processSex(test)  

Xy_train_df.head()
gnb_vars = ['Female']

X_train = Xy_train_df.drop('Survived',axis=1)[gnb_vars].values

X_test = X_test_df[gnb_vars].values

y = train['Survived'].ravel() 
gnb = GaussianNB()

gnb.fit(X_train, y)

gnb_scores = cross_val_score(gnb,X_train,y,cv=5,scoring='accuracy')

gnb_survivors = gnb.predict(X_test)
def print_classification_results(a,b,filename):

    print('Survivors: = {0:0.0f}'.format(a.sum()))

    print('Accuracy:  = {0:0.3f}'.format(b))

    test['Survived'] = a

    test['Survived'].to_csv(filename,header=True)

print_classification_results(gnb_survivors,gnb_scores.mean(),filename='Model 1 - GNB.csv')
PredictorPlot(df = Xy_train_df,var ='Pclass',freqvar = 'Survived', freqvalues = [0,1])
train.loc[(train['Fare'].isnull()) | (train['Fare']==0) ]
def fillFare(df):

    

    df.loc[((df['Fare'].isnull()) | (df['Fare']==0)) & (df['Pclass']==3),'Fare'] = df[

        'Fare'].loc[(df['Pclass']==3) & (df['Embarked']=='S')].median()

    df.loc[((df['Fare'].isnull()) | (df['Fare']==0)) & (df['Pclass']==2),'Fare'] = df[

        'Fare'].loc[(df['Pclass']==2) & (df['Embarked']=='S')].median()

    df.loc[((df['Fare'].isnull()) | (df['Fare']==0)) & (df['Pclass']==1),'Fare'] = df[

        'Fare'].loc[(df['Pclass']==1) & (df['Embarked']=='S')].median()

    

    return df



Xy_train_df = fillFare(Xy_train_df)
plt.figure(figsize=(16,5))

plt.hist(Xy_train_df['Fare'],bins=50);

plt.title('Distribution of Passenger Fares');
plt.figure(figsize=(16,5))

plt.hist(Xy_train_df['Fare'].apply(np.log),bins=50);

plt.title('Distribution of Log-transformed Passenger Fares');
def processFare(df):

    

    df = fillFare(df)

    df['LogFare'] = df['Fare'].apply(np.log).round().clip_upper(5)

    df['LogFare'] = df['LogFare'].astype(int)

    

    return df



Xy_train_df = processFare(Xy_train_df)
PredictorPlot(df = Xy_train_df,var ='LogFare',freqvar = 'Survived', freqvalues = [0,1])
plt.figure(figsize=(16,5))

sns.violinplot(x="Pclass", y="LogFare", data=Xy_train_df);

plt.title('Relationship between Fares and Passenger Class');
PredictorPlot(df = Xy_train_df.loc[Xy_train_df['Pclass']==1],var ='LogFare',

              freqvar = 'Survived', freqvalues = [0,1])
PredictorPlot(df = Xy_train_df,var ='Parch',freqvar = 'Survived', freqvalues = [0,1])

PredictorPlot(df = Xy_train_df,var ='SibSp',freqvar = 'Survived', freqvalues = [0,1])
train['FamilySize'] = train['Parch'] + train['SibSp'] + 1

plt.figure(figsize=(16,5))

plt.hist(train['FamilySize']);

plt.title('Distribution of Family Size');
def processParch_SibSp(df,LargeFamilySize=5,dropFamilySize=True):

    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1

    df['Alone'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)

    df['CapFamSize'] = df['FamilySize'].clip_upper(LargeFamilySize)

    if dropFamilySize==True: df = df.drop('FamilySize',axis=1) 

    return df



Xy_train_df = processParch_SibSp(Xy_train_df)

PredictorPlot(df = Xy_train_df,var ='CapFamSize',freqvar = 'Survived', freqvalues = [0,1])
train.loc[train['Embarked'].isnull()]
train['Embarked'].groupby(train['Embarked']).count()
def processEmbarked(df,dropEmbarked=True):

    df.loc[df['Embarked'].isnull(),'Embarked' ] ="S"

    df_embarked = pd.get_dummies(df['Embarked'])

    df = df.join(df_embarked)

    if dropEmbarked==True: df = df.drop('Embarked',axis=1) 

    return df



Xy_train_df = processEmbarked(Xy_train_df)
Xy_train_df.isnull().sum()[Xy_train_df.isnull().sum()>0]
def processCabin(df,dropCabin=True):

    if dropCabin==True: df = df.drop('Cabin',axis=1) 

    return df



Xy_train_df = processCabin(Xy_train_df)
train["TicketNum"] = train["Ticket"].str.extract('(\d{2,})', expand=True)

train["TicketNum"] = train["TicketNum"].apply(pd.to_numeric)

train.loc[train['TicketNum'].isnull(),'TicketNum'] = -1

PredictorPlot(df = train,var ='TicketNum',freqvar = 'Survived', freqvalues = [0,1],ticks=False)
PredictorPlot(df = train,var ='TicketNum',freqvar = 'Pclass', 

              freqvalues = [1,2,3],ticks=False,show_title=False)
def processTicket(df,dropTicket=True):

    if dropTicket==True: df = df.drop('Ticket',axis=1) 

    return df



Xy_train_df = processTicket(Xy_train_df)
Xy_train_df.head()
train['Title'] = train['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

pd.crosstab(train['Title'], train['Female'])
def processName(df, dropName = True,dropTitle = False ): 

    

    Title_Dictionary = {"Capt": "TitleX",

                        "Col":"TitleX",

                        "Don":"TitleX",

                        "Dona":"TitleX",

                        "Dr":"TitleX",

                        "Jonkheer": "TitleX",

                        "Lady":"TitleX",

                        "Major":"TitleX",

                        "Master" :"Master",

                        "Miss" :"Miss",

                        "Mlle":"Miss",

                        "Mr" :"Mr",

                        "Mrs":"Mrs",

                        "Mme":"Mrs",

                        "Ms":"Mrs",

                        "Rev":"TitleX",

                        "Sir" :"TitleX",

                        "the Countess":"TitleX"}



    df['Title'] = df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    df['Title'] = df.Title.map(Title_Dictionary) # use the Title_Dictionary to map the titles

    df_title = pd.get_dummies(df['Title'])

    df = pd.concat([df, df_title], axis=1)

    if dropName==True: df = df.drop('Name',axis=1)

    if dropTitle==True: df = df.drop('Title',axis=1)

            

    return df

    

Xy_train_df = processName(Xy_train_df)

pd.crosstab(Xy_train_df['Title'], Xy_train_df['Female'])
PredictorPlot(df = Xy_train_df,var = 'Title',freqvar = 'Survived',freqvalues = [0,1])
pd.pivot_table(Xy_train_df, values='Age', index=['Title'],columns=[], aggfunc=np.max)
pd.pivot_table(Xy_train_df, values='Age', index=['Title'],columns=['Parch'], aggfunc=np.median)
def processAge(df,dropAge = True):

    

    df['MaleCh'] = 0

    df['FemaleCh'] = 0

    

    df.loc[( (df['Female']==0) & (df['Age']<=12) ) | (df['Master']==1),'MaleCh'] = 1  

    

    df.loc[( ( (df['Female']==1) & (df['Age']<=12) ) | 

             ( (df['Female']==1) & (df['Age'].isnull()) & (df['Miss']==1)& (df['Parch']>0) )  ),

           'FemaleCh' ] = 1

    # Female logic - A female with the title Miss and Parents onboard is likely to be a child    

    

    if dropAge==True: df = df.drop('Age',axis=1)

    if 'Title' in df.columns: df = df.drop('Title',axis=1)

            

    return df



Xy_train_df = processAge(Xy_train_df)
Xy_train_df.drop('Survived',axis=1).head()
def process_data(filename):

    df = read_data(filename)

    df = processSex(df)

    df = processFare(df)

    df = processParch_SibSp(df)

    df = processEmbarked(df)

    df = processCabin(df)

    df = processTicket(df)

    df = processName(df)

    df = processAge(df)

    return df
Xy_train_df_new = process_data('../input/train.csv')

Xy_train_df_new.equals(Xy_train_df)
X_test_df = process_data('../input/test.csv')
fig, ax = plt.subplots(figsize=(12,12)) 

sns.heatmap(Xy_train_df.corr(), linewidths=0.1,cbar=True, annot=True, square=True, fmt='.1f')

plt.title('Correlation between Variables');
svm_params = {'kernel':'rbf','random_state' : RANDOM_STATE}

select_vars = (['Female', 'Pclass', 'CapFamSize', 'Alone', 'Master', 'LogFare'])

X_train_df = Xy_train_df.drop('Survived',axis=1)[select_vars]

X_train = X_train_df.values

X_test = X_test_df[select_vars].values 



svm = SVC(**svm_params)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

    

svm.fit(X_train_scaled, y)

svm_scores = cross_val_score(svm,X_train_scaled,y,cv=5,scoring='accuracy')

svm_survivors = svm.predict(X_test_scaled)

print_classification_results(svm_survivors,svm_scores.mean(),'Model 2 - SVC.csv')
def ParamChart(X,y,param,min,max,step,clf_params,UseOOB=False):

    error_rate = []

    

    for i in np.arange(min, max+1, step):

        

        new_param = {param:i}

        if UseOOB == True: 

            clf = RandomForestClassifier(**clf_params)

            clf.set_params(**new_param)

            clf.fit(X, y)

            

        if UseOOB==True:

            error_rate.append((i, 1 - clf.oob_score_))

        else:

            clf = SVC(**clf_params)

            clf.set_params(**new_param)

            scores = cross_val_score(clf,X,y,cv=5,scoring='accuracy')

            error_rate.append((i, 1 - scores.mean()))



        

    plt.figure(figsize=(16,5))

    xs, ys = zip(*error_rate)

    plt.plot(xs, ys)

    plt.xlim(min, max)

    plt.xlabel("Parameter")

    plt.ylabel("Error rate")

    plt.title('Error Rate by Parameter Value');

    plt.show()
ParamChart(X_train_scaled,y,param='C',min=0.1,max=10,step=0.1,clf_params=svm_params)
pd.set_option('display.float_format', lambda x: '%.3f' % x) 

def run_rf(X1,X2,params,filename='output.csv'):

    X_train = X1.values 

    X_test = X2.values 

    rf = RandomForestClassifier(**params)

    rf.fit(X_train, y)

    rf_survivors = rf.predict(X_test)



    print_classification_results(rf_survivors,rf.oob_score_,filename)

    feature_importance = pd.DataFrame(data=rf.feature_importances_,

                                      index=X_train_df.columns.values,columns=['FeatureScore'] )

    print (feature_importance.sort_values(ascending=False,by=['FeatureScore']))



rf_params = {'oob_score':True, 'warm_start': True,'random_state': RANDOM_STATE,'n_estimators':50}

select_vars = (['Female', 'Pclass', 'CapFamSize', 'Alone', 'Master', 'Fare'])

X_train_df = Xy_train_df.drop('Survived',axis=1)[select_vars]

run_rf(X_train_df,X_test_df[select_vars],rf_params,'Model 3 - RF1.csv')
def LearningCurve(X,clf_params):

    warnings.simplefilter(action='ignore')

    error_rate = []



    for i in range(10,len(train),50): 

        X_LC = X[:i].values

        y_LC = train[:i]['Survived']

        clf = RandomForestClassifier(**clf_params)

        clf.fit(X_LC, y_LC)

        oob_error = 1 - clf.oob_score_

        training_error = 1 - clf.score(X_LC,y_LC)

        error_rate.append((i, oob_error, training_error))



    plt.figure(figsize=(16,5))

    xs, ys, zs = zip(*error_rate)

    plt.plot(xs, ys)

    plt.plot(xs, zs)

    plt.xlim(0, len(train))

    plt.xlabel("Training Examples")

    plt.ylabel("Error rate")

    plt.title('Error Rate by Sammple Size (green=Training error, blue = Out-of-bag error)');

    plt.show()



rf_params = {'oob_score':True, 'warm_start': True,'random_state': RANDOM_STATE,'n_estimators':50}

LearningCurve (X_train_df,rf_params)
select_vars = (['Female', 'Pclass', 'CapFamSize', 'Alone', 'Master', 'LogFare'])

X_train_df = Xy_train_df.drop('Survived',axis=1)[select_vars]

run_rf(X_train_df,X_test_df[select_vars],rf_params,'Model 4 - RF2.csv')

LearningCurve (X_train_df,rf_params)
ParamChart(X_train_df,y,param='n_estimators',min=10,max=500,step=10,clf_params=rf_params,UseOOB=True)
rf_params = {'oob_score':True, 'warm_start': True,'random_state': RANDOM_STATE,'n_estimators':500}

rf = RandomForestClassifier(**rf_params)

ParamChart(X_train_df,y,param='min_samples_leaf',min=1,max=30,step=1,clf_params=rf_params,UseOOB=True)
rf_params = {'oob_score':True, 'warm_start': True,'random_state': RANDOM_STATE,'n_estimators':500,

            'min_samples_leaf':5}

ParamChart(X_train_df,y,param='max_depth',min=1,max=10,step=1,clf_params=rf_params,UseOOB=True)
rf_params = {'oob_score':True, 'warm_start': True,'random_state': RANDOM_STATE,'n_estimators':500,

            'min_samples_leaf':5,'max_depth':5}

select_vars = (['Female', 'Pclass', 'CapFamSize', 'Alone', 'Master', 'LogFare'])

X_train_df = Xy_train_df.drop('Survived',axis=1)[select_vars]

run_rf(X_train_df,X_test_df[select_vars],rf_params,'Model 5 - RF3.csv')

LearningCurve (X_train_df,rf_params)
# Define RandomForest parameters

rf_params = {'oob_score':True, 'warm_start': True,'random_state': RANDOM_STATE,'n_estimators':500,

            'min_samples_leaf':5,'max_depth':5}

# Select variables for RandomForest model

select_vars = (['Female', 'Pclass', 'CapFamSize', 'Alone', 'Master', 'LogFare'])

# Process data

X_train_df = process_data('../input/train.csv').drop('Survived',axis=1)[select_vars]

X_test_df = process_data('../input/test.csv')[select_vars]

# Run the model

run_rf(X_train_df,X_test_df,rf_params)