import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import time

import sys

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

np.seterr(divide='ignore',invalid='ignore')

train0_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

sub0_data=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

sub_data=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')



country_list=train0_data.loc[:,'Country_Region'].unique()

#国家列表



y_Confirmed={}

y_Fatalities={}

models_c={}

models_f={}

mark_c={}

mark_f={}

for country in country_list :

    train0_data.loc[(train0_data['Country_Region']==country)& 

                    train0_data['Province_State'].isnull() , 'Province_State']= country 

for country in country_list :

    sub0_data.loc[(sub0_data['Country_Region']==country)& 

                    sub0_data['Province_State'].isnull() , 'Province_State']= country                     

#省份缺失值处理

    train0_data.loc[train0_data['ConfirmedCases']<0 |

            train0_data['ConfirmedCases'].isnull(),'ConfirmedCases']=0

            

    train0_data.loc[train0_data['Fatalities']<0 |

            train0_data['Fatalities'].isnull(),'Fatalities']=0        

#复数值处理



date_list=train0_data.loc[:,'Date'].unique()

x_date=[0]*len(date_list)

for i  in range(0,len(x_date)) :

    x_date[i]=i

x_date=np.array(x_date).reshape(-1,1)

x_train=np.array(x_date[0:-7]).reshape(-1,1)

x_test=np.array(x_date[-7:]).reshape(-1,1)

xx = np.linspace(0, 137, 274).reshape(-1,1)

x_d=np.array([136,137]).reshape(-1,1)

pre_list=[0]*43

for i in range (0,43):

    pre_list[i]=2*i+188

#自变量x

repeat=800

#重复次数设置



def addtodict2(thedict, key_a, key_b, val): 

    if key_a in thedict:

        thedict[key_a].update({key_b: val})

    else:

        thedict.update({key_a:{key_b: val}})

#二维字典处理函数

        



def loss(y,y_p,threshold,max_data,d):

    l=len(y)

    y1=y/max_data

    y_p1=y_p/max_data

    loss_sum=0.000

    loss_d=0

    if d[-1]<d[0]:

        loss_d=2

    else: 

        loss_d=0

    for i in range(0,l):

        if y1[i]-threshold <=0 :

            loss_sum+=0.1*(y1[i]-y_p1[i])*(y1[i]-y_p1[i])

        else:

            loss_sum+=y[i]*y_p[i]*(y1[i]-y_p1[i])*(y1[i]-y_p1[i])

            

    loss_sum=loss_sum/(2*l)+loss_d

    return loss_sum

#损失函数   

    

quadratic_featurizerx = PolynomialFeatures(degree=1)

quadratic_featurizery = PolynomialFeatures(degree=2)







#x_p=[0]*2

#x_p[0]=quadratic_featurizerx.transform(x_date_p.reshape(x_date_p.shape[0], 1))

#x_p[1]=quadratic_featurizery.transform(x_date_p.reshape(x_date_p.shape[0], 1))

   

for country in country_list:

    province_list=train0_data.loc[train0_data['Country_Region']==country,'Province_State'].unique()

    for province in province_list :

        confirmed_data=np.array(train0_data.loc[(train0_data['Country_Region']==country) &

                                       (train0_data['Province_State']==province),'ConfirmedCases'])

        fatalities_data=np.array(train0_data.loc[(train0_data['Country_Region']==country) &

                                       (train0_data['Province_State']==province),'Fatalities'])

        

        

        x_train_c,x_test_c, confirmed_train0, confirmed_test = train_test_split(x_date,

                                                                               confirmed_data,

                                                                               test_size = 0.25,

                                                                               random_state = 0)

        x_train_f,x_test_f, fatalities_train0, fatalities_test = train_test_split(x_date,

                                                                               fatalities_data,

                                                                               test_size = 0.25,

                                                                               random_state = 0)

        #划分训练，测试集

        quadratic_featurizer = PolynomialFeatures(degree=1)

        quadratic_featurizer1 = PolynomialFeatures(degree=2)

        #多项式拟合

        

        x_d0=quadratic_featurizer.fit_transform(x_d)

        x_d1=quadratic_featurizer1.fit_transform(x_d)

        X_train_quadratic = quadratic_featurizer.fit_transform(x_train_c)

        X_test_quadratic = quadratic_featurizer.transform(x_test_c)

        

        X_train_quadratic_f = quadratic_featurizer.fit_transform(x_train_f)

        X_test_quadratic_f = quadratic_featurizer.transform(x_test_f)

        

        

        

        X_train_quadratic1 = quadratic_featurizer1.fit_transform(x_train_c)

        X_test_quadratic1 = quadratic_featurizer1.transform(x_test_c)

        

        X_train_quadratic1_f = quadratic_featurizer1.fit_transform(x_train_f)

        X_test_quadratic1_f = quadratic_featurizer1.transform(x_test_f)

        

        

        

        

        

        

        

        regressor_quadratic = linear_model.LinearRegression()

        regressor_line=linear_model.LinearRegression()

        

        regressor_quadratic_f = linear_model.LinearRegression()

        regressor_line_f=linear_model.LinearRegression()

        

        

        

        xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

        xx_quadratic1 = quadratic_featurizer1.transform(xx.reshape(xx.shape[0], 1))

     #   plt.plot(x_date, confirmed_data, 'k.', label="数据点分布 c")

     #   plt.plot(x_date, confirmed_data, 'k.', label="数据点分布 c")

        depth_c=max(((max(confirmed_data)-min(confirmed_data))/50),1)

        depth_f=max(((max(fatalities_data)-min(fatalities_data))/50),1)

        loss_list_c=[0.000]*repeat

        loss_list_f=[0.000]*repeat

        loss_line_c=0.000

        loss_line_f=0.000

        for i in range(0,repeat):

            

            k_c=max(confirmed_data)+(i+1)*depth_c

            k_f=max(fatalities_data)+(i+1)*depth_c

            

            confirmed_train=-np.log(k_c/(confirmed_train0+0.2)-1)

            fatalities_train=-np.log(k_f/(fatalities_train0+0.2)-1)

    

            regressor_quadratic.fit(X_train_quadratic, confirmed_train)

            regressor_quadratic_f.fit(X_train_quadratic_f, fatalities_train)

          #  plt.plot(xx, 1+k_c/(1+np.exp(-regressor_quadratic.predict(xx_quadratic))), 'r-', label="多项式回归")

           # plt.plot(x_date, confirmed_data, 'k.', label="数据点分布")

           # plt.show()

            loss_list_c[i]=loss(np.array(confirmed_test),

                        np.array(k_c/(1+np.exp(-regressor_quadratic.predict(X_test_quadratic)))),

                        max(confirmed_data)/2+min(confirmed_data)/2,

                        max(confirmed_data),

                        regressor_quadratic.predict(x_d0))

            

            loss_list_f[i]=loss(np.array(fatalities_test),

                        np.array(k_c/(1+np.exp(-regressor_quadratic_f.predict(X_test_quadratic_f)))),

                        max(fatalities_data)/2+min(fatalities_data)/2,

                        max(fatalities_data),

                        regressor_quadratic_f.predict(x_d0))

            

            

        min_loss_num_c=loss_list_c.index(min(loss_list_c))

        k_c=max(confirmed_data)+(min_loss_num_c+1)*depth_c

        confirmed_train=-np.log(k_c/(confirmed_train0+0.2)-1)

        

        min_loss_num_f=loss_list_f.index(min(loss_list_f))

        k_f=max(fatalities_data)+(min_loss_num_f+1)*depth_f

        fatalities_train=-np.log(k_f/(fatalities_train0+0.2)-1)

        

        

        

        

        regressor_quadratic.fit(X_train_quadratic, confirmed_train)

        regressor_quadratic_f.fit(X_train_quadratic_f, fatalities_train)

        

        

        regressor_line.fit( X_train_quadratic1,confirmed_train0)

        regressor_line_f.fit( X_train_quadratic1,fatalities_train0)

        

        loss_line_c=loss(np.array(confirmed_test),

                        np.array(regressor_line.predict(X_test_quadratic1)),

                        max(confirmed_data)/2+min(confirmed_data)/2,

                        max(confirmed_data),

                        regressor_line.predict(x_d1))

        

        loss_line_f=loss(np.array(fatalities_test),

                        np.array(regressor_line_f.predict(X_test_quadratic1_f)),

                        max(fatalities_data)/2+min(fatalities_data)/2,

                        max(fatalities_data),

                        regressor_line_f.predict(x_d1))

        

        

        

        

 #       plt.plot(xx, 1+k_f/(1+np.exp(-regressor_quadratic_f.predict(xx_quadratic))), 'r-', label="多项式回归")

 #       plt.plot(xx, regressor_line_f.predict(xx_quadratic1), 'b-', label="逻辑多项式回归")

 #       plt.plot(x_date, fatalities_data, 'k.', label="数据点分布")

        plt.show()

        if (min(loss_list_c)<=loss_line_c):

            addtodict2(models_c,country,province,1+k_c/(1+np.exp(-regressor_quadratic.predict(xx_quadratic))))

            addtodict2(mark_c,country,province,1)

            

        else:

            addtodict2(models_c,country,province,regressor_line.predict(xx_quadratic1))

            addtodict2(mark_c,country,province,2)

        

        if (min(loss_list_f)<=loss_line_f):

            addtodict2(models_f,country,province,1+k_f/(1+np.exp(-regressor_quadratic_f.predict(xx_quadratic))))

            addtodict2(mark_f,country,province,1)

        else:

            addtodict2(models_f,country,province,regressor_line_f.predict(xx_quadratic1))

            addtodict2(mark_f,country,province,2)

        addtodict2(y_Confirmed,country,province,confirmed_data)

        addtodict2(y_Fatalities,country,province,fatalities_data)

        print(country,'-',province,"|| fit finished")

print("fitness are all finished!") 

       

for country in country_list:

    province_list=sub0_data.loc[sub0_data['Country_Region']==country,'Province_State'].unique()

    for province in province_list :     

        markc=mark_c[country][province]

        markf=mark_f[country][province]

        modelc=models_c[country][province]

        modelf=models_f[country][province]

        pre_c=[]

        pre_f=[]

       

        sub0_data.loc[(sub0_data['Country_Region']==country)& 

                           (sub0_data['Province_State']==province) ,

                           'ConfirmedCases']=np.trunc(modelc[pre_list])

        

        sub0_data.loc[(sub0_data['Country_Region']==country)& 

                    (sub0_data['Province_State']==province) , 

                    'Fatalities']=np.trunc(modelf[pre_list])

        print(country,'-',province,"|| predict finished")





print("predictions are all finished!")

sub_data.loc[:,'ConfirmedCases']=sub0_data.loc[:,'ConfirmedCases']

sub_data.loc[:,'Fatalities']=sub0_data.loc[:,'Fatalities']

sub_data.to_csv('/kaggle/working/submission.csv', index=False)        





    

    
