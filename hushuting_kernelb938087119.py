# -*- coding: utf-8 -*-

import numpy as np

import pandas as pd



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import PolynomialFeatures

import os



import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

import seaborn as sns



'''Read Data'''

app_train=pd.read_csv('../input/application_train.csv')

app_test=pd.read_csv('../input/application_test.csv')



'''EDA'''

#1.Target Distribution

#app_train['TARGET'].value_counts()

    #app_train['TARGET'].astype(int).plot.hist():数据分布的直方图

#plt.hist(app_train['TARGET'].astype(int))



#2.examine misssing values

    #实现的目的是确定有多少colunm是有缺失值的；每一列的缺失数量以及缺失率。

def missing_values_table(df):

    mis_val=df.isnull().sum()

    mis_val_percent=100*df.isnull().sum()/len(df)

    mis_val_table=pd.concat([mis_val,mis_val_percent],axis=1)

    #列的重命名

    mis_val_table_ren_columns=mis_val_table.rename(columns={0:'Missing Values',1:'% of Total Values'})

    mis_val_table_ren_columns=mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1]!=0].sort_values('% of Total Values',ascending=False).round(1)



    print('our selected dataframe has '+str(df.shape[1])+' columns.\n'

          'There are '+str(mis_val_table_ren_columns.shape[0])+' columns that have missing values.')



    return mis_val_table_ren_columns



#3.columns type

app_train.dtypes.value_counts()

app_train.select_dtypes('object').apply(pd.Series.nunique,axis=0)



#4.encoding categorical variables

    #对object中是类别变量的encoding

le=LabelEncoder()

    #函数对2类的类别变量进行label编码即0或者1

def object_cat_encoding(app_train,app_test,lecount):

    for col in app_train:

        if app_train[col].dtype=='object':

            if len(list(app_train[col].unique()))<=2:

                le.fit(app_train[col])

            

                app_train[col]=le.transform(app_train[col])

                app_test[col]=le.transform(app_test[col])

            

                lecount+=1

    

    print('%d columns were label encoded.' % lecount)

    return app_train,app_test

    #这两个操作是对上面的编码为0或者1变量进行ont-hot-encoding 

app_train,app_test=object_cat_encoding(app_train,app_test,0)

app_train=pd.get_dummies(app_train)

app_test=pd.get_dummies(app_test)  



#5.Aligning Training and Testing Data

def align_train_test_data(app_train,app_test):

    train_labels=app_train['TARGET']

    app_train,app_test=app_train.align(app_test,join='inner',axis=1)

    app_train['TARGET']=train_labels

    print('Training Features shape:',app_train.shape)

    print('Training Features shape:',app_test.shape)

    return app_train,app_test

app_train,app_test=align_train_test_data(app_train,app_test)



#6.Anomalies

def anom_dectect(app_train,app_test):

    #未出现异常点

    (app_train['DAYS_BIRTH']/-365).describe()

    #出现异常点

    app_train['DAYS_EMPLOYED'].describe()

    app_train['DAYS_EMPLOYED'].plot.hist(title='Days Emploment Histogram')

    plt.xlabel('Days Emploment')

    plt.ylabel('Frequency')

    

    anom=app_train[app_train['DAYS_EMPLOYED']==365243]

    non_anom=app_train[app_train['DAYS_EMPLOYED']!=365243]

    print('The anom default on %0.2f%% of loans' % (100*anom['TARGET'].mean()))

    print('the non_anom default on %0.2f%% of loans' % (100*non_anom['TARGET'].mean()))

    print('There are %d anom days of emploment' % len(anom))

#anomaly_dectect(app_train,app_test)



def delet_anom_hist_train(app_train):

    app_train['DAYS_EMPLOMED_ANOM']=app_train['DAYS_EMPLOYED']==365243

    app_train['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)

    

    '''print('There are %d anom in the train data out of %d entries' % (app_train['DAYS_EMPLOMED_ANOM'].sum(),len(app_train)))

    app_train['DAYS_EMPLOYED'].plot.hist(title='Days Empoment Hist')

    plt.xlabel('Days Emploment')'''

    return app_train

app_train=delet_anom_hist_train(app_train)

print('app_train的行列：',app_train.shape)



def delet_anom_hist_test(app_test):

    app_test['DAYS_EMPLOMED_ANOM']=app_test['DAYS_EMPLOYED']==365243

    app_test['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)

    '''print('There are %d anom in the test data out of %d entries' % (app_test['DAYS_EMPLOMED_ANOM'].sum(),len(app_test)))

    app_test['DAYS_EMPLOYED'].plot.hist(title='Days Empoment Hist')

    plt.xlabel('Days Emploment')'''

    return app_test

app_test=delet_anom_hist_test(app_test)

print('app_test的行列：',app_test.shape)



#7.correlatons

def correlation(app_train):

    correlations=app_train.corr()['TARGET'].sort_values()

    print('Most Positive Correlations:\n',correlations.tail(15))

    print('Most Negative Correlations:\n',correlations.head(15))

#correlation(app_train)

 

#7.1正相关的变量DAYS_BIRTH的EDA 

def daybirth_target_corr(app_train):

    app_train['DAYS_BIRTH']=abs(app_train['DAYS_BIRTH'])

    a=app_train['DAYS_BIRTH'].corr(app_train['TARGET'])

    print('The corr about abs of DAYS_BIRTH and TARGET is \n', a)

    

    #hist直方图分析年龄的分布

    '''plt.style.use('fivethirtyeight')

    plt.hist(abs(app_train['DAYS_BIRTH']/365),edgecolor='k',bins=25)

    plt.title('Age of client')

    plt.xlabel('Age(years)')

    plt.ylabel('count')'''

    

    #senborn kdeplot

    sns.kdeplot(app_train.loc[app_train['TARGET']==0,'DAYS_BIRTH']/365,label='target==0')

    sns.kdeplot(app_train.loc[app_train['TARGET']==1,'DAYS_BIRTH']/365,label='target==1')

    plt.xlabel('Age(years)')

    plt.ylabel('Density')

    plt.title('Distribution of Ages')

    

    #in another way to look at the relationship(DAYS_BIRTH,TARGET)

    age_data=app_train[['TARGET','DAYS_BIRTH']]

    age_data['YEARS_BIRTH']=age_data['DAYS_BIRTH']/365

    age_data['YEARS_BINNED']=pd.cut(age_data['YEARS_BIRTH'],bins=np.linspace(20,70,num=11))

    age_data.head(10)

    age_groups=age_data.groupby('YEARS_BINNED').mean()

    #print(age_groups)

    

    #fail to repay by age group

    plt.figure(figsize=(8,8))

    plt.bar(age_groups.index.astype(str),100*age_groups['TARGET'])

    plt.xlabel('Age Group(years)')

    plt.ylabel('Failure to Repay(%)')

    #通过年龄的分组和target的关系得出年龄越小还款失败率越高

#daybirth_target_corr(app_train)

#7.2负相关变量EXterior Sources(1/2/3)的EDA

def extsource_target_corr(app_train):

    ext_data=app_train[['TARGET','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']]

    ext_data_corrs=ext_data.corr()

    

    #heatpmap(热图)反应各个变量之间的corr

        #热图反应EXT_SOURCE与target负相关且EXT_SOURCE_1与DAYS_BIRTHU有很大相关

    plt.figure(figsize=(8,6))

    sns.heatmap(ext_data_corrs,cmap=plt.cm.RdYlBu_r,vmin= -0.25,annot=True,vmax=0.6)

    plt.title('Correlation Heatmap')

    

    #senborn kdeplot:EXT_SOURCE_1/2/3 and TARGET

        #分别画出三个变量与target的kdeplot

    plt.figure(figsize=(10,12))

    for i,source in enumerate(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']):

        plt.subplot(3,1,i+1)

        sns.kdeplot(app_train.loc[app_train['TARGET']==0,source],label='target==0')

        sns.kdeplot(app_train.loc[app_train['TARGET']==1,source],label='target==1')

        

        print('Distribution of %s by Target Value' % source)

        plt.xlabel('%s' % source)

        plt.ylabel('DEnsity')

    plt.tight_layout(h_pad=2.5)

    

    age_data=app_train[['TARGET','DAYS_BIRTH']]

    age_data['YEARS_BIRTH']=abs(age_data['DAYS_BIRTH'])/365

    age_data['YEARS_BINNED']=pd.cut(age_data['YEARS_BIRTH'],bins=np.linspace(20,70,num=11))

    plot_data = ext_data.drop(columns = ['DAYS_BIRTH']).copy()

    # Add in the age of the client in years

    plot_data['YEARS_BIRTH'] = age_data['YEARS_BIRTH']

    # Drop na values and limit to first 100000 rows

    plot_data = plot_data.dropna().loc[:100000, :]

    # Function to calculate correlation coefficient between two columns

    def corr_func(x, y, **kwargs):

        r = np.corrcoef(x, y)[0][1]

        ax = plt.gca()

        ax.annotate("r = {:.2f}".format(r),

                    xy=(.2, .8), xycoords=ax.transAxes,

                    size = 20)

    # Create the pairgrid object

    grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False,

                        hue = 'TARGET', 

                        vars = [x for x in list(plot_data.columns) if x != 'TARGET'])

    # Upper is a scatter plot

    grid.map_upper(plt.scatter, alpha = 0.2)

    # Diagonal is a histogram

    grid.map_diag(sns.kdeplot)

    # Bottom is density plot

    grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r)

    plt.suptitle('Ext Source and Age Features Pairs Plot', size = 32, y = 1.05)

#extsource_target_corr(app_train)

    

    

'''Feature Engineering:(feature construction)and(feature selection)'''

#1.feature construction:Polynomial features and Domain knowledge features

#1.1 Polynomial features

def feature_construction_polynominal(app_train,app_test):

    #app_train['DAYS_BIRTH']必须是绝对值。

    app_train['DAYS_BIRTH']=abs(app_train['DAYS_BIRTH'])

    poly_features=app_train[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH','TARGET']]

    poly_features_test=app_test[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']]

    imputer=Imputer(strategy='median')

    poly_target=poly_features['TARGET']

    poly_features=poly_features.drop(columns=['TARGET'])

    

    #impute missing value

    poly_features=imputer.fit_transform(poly_features)

    poly_features_test=imputer.transform(poly_features_test)

    

    #create the polynomial object with specified degree

    poly_transform=PolynomialFeatures(degree=3)

    

    #train the polynomial features

    poly_transform.fit(poly_features)

    

    poly_features=poly_transform.transform(poly_features)

    poly_features_test=poly_transform.transform(poly_features_test)

    print('Polynomial Features train shape: ',poly_features.shape,app_train.shape)

    print('Polynomial Features test shape: ',poly_features_test.shape,app_test.shape)

    #此时返回的poly_features和poly_features_test都是数组，转换成DataFRame进行操作

    

    #create a dataframe fo the poly_features

    poly_features=pd.DataFrame(poly_features,columns=poly_transform.get_feature_names(

            ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']))

    poly_features['TARGET']=poly_target

    poly_corrs=poly_features.corr()['TARGET'].sort_values()

    

    #create a dataframe of the poly_features_test

    poly_features_test=pd.DataFrame(poly_features_test,columns=poly_transform.get_feature_names(

            ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']))

    

    #merge polynomial features into training dataframe

    poly_features['SK_ID_CURR']=app_train['SK_ID_CURR']

    app_train_poly=app_train.merge(poly_features,on='SK_ID_CURR',how='left')

    #merge polynomial features into testing dataframe

    poly_features_test['SK_ID_CURR']=app_test['SK_ID_CURR']

    app_test_poly=app_test.merge(poly_features_test,on='SK_ID_CURR',how='left')

    

    app_train_poly,app_test_poly=app_train_poly.align(app_test_poly,join='inner',axis=1)

    print('Training data with polynomial features shape:',app_train_poly.shape)

    print('Testing data with polynomial features shape:',app_test_poly.shape) 

    return app_train_poly,app_test_poly

#app_train,app_test=feature_construction_polynominal(app_train,app_test)#此时是275个特征

#2.Domain knowledge features

def domain_knowledge_features(app_train,app_test):

    app_train_domain=app_train.copy()

    app_test_domain=app_test.copy()

    

    app_train_domain['CREDIT_INCOME_PERCENT']=app_train_domain['AMT_CREDIT']/app_train_domain['AMT_INCOME_TOTAL']

    app_train_domain['ANNUITY_INCOME_PERCENT']=app_train_domain['AMT_ANNUITY']/app_train_domain['AMT_INCOME_TOTAL']

    app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']

    app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']

    

    app_test_domain['CREDIT_INCOME_PERCENT']=app_test_domain['AMT_CREDIT']/app_test_domain['AMT_INCOME_TOTAL']

    app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']

    app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']

    app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']

    

    plt.figure(figsize = (12, 20))

    # iterate through the new features

    for i, feature in enumerate(['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']):

        

        # create a new subplot for each source

        plt.subplot(4, 1, i + 1)

        # plot repaid loans

        sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 0, feature], label = 'target == 0')

        # plot loans that were not repaid

        sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 1, feature], label = 'target == 1')

        

        # Label the plots

        plt.title('Distribution of %s by Target Value' % feature)

        plt.xlabel('%s' % feature); plt.ylabel('Density');   

    plt.tight_layout(h_pad = 2.5)

#domain_knowledge_features(app_train,app_test)



'''baseline'''

#1.logistic regression implementation

from sklearn.preprocessing import MinMaxScaler,Imputer

from sklearn.linear_model import LogisticRegression

def baseline_logistic(app_train,app_test):

    train_label=app_train['TARGET']

    if 'TARGET' in app_train:

        train=app_train.drop(columns=['TARGET'])

    else:

        train=app_train.copy()

    test=app_test.copy()

   

    imputer=Imputer(strategy='median')

    scaler=MinMaxScaler(feature_range=(0,1))

    

    imputer.fit(train)

    train=imputer.transform(train)

    test=imputer.transform(app_test)

    

    scaler.fit(train)

    train=scaler.transform(train)

    test=scaler.transform(test)

    print('Training data shape:',train.shape)

    print('Testing data shape:',test.shape)

    

    log_reg=LogisticRegression(C=0.0001, class_weight=None, dual=False,

          fit_intercept=True, intercept_scaling=1, max_iter=100,

          multi_class='ovr', n_jobs=None, penalty='l2', random_state=None,

          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

    log_reg.fit(train,train_label)

    log_reg_pred=log_reg.predict_proba(test)[:,1]

    submit=app_test[['SK_ID_CURR']]

    submit['TARGET']=log_reg_pred

    submit.to_csv('log_reg_output.csv',index=False)



baseline_logistic(app_train,app_test)

    

    

    

    

    

    

    





        





    

        

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    













































    









  
































































