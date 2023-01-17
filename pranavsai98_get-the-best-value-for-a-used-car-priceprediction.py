import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_palette("tab10")

import plotly.graph_objs as go

import plotly.express as px



import category_encoders as ce



import re

import nltk

from nltk.corpus import stopwords

nltk.download('stopwords')





import warnings

warnings.filterwarnings('ignore')



from sklearn.metrics import mean_squared_error, mean_absolute_error
# reading .csv file

df=pd.read_csv('../input/input-data/CAR DETAILS FROM CAR DEKHO.csv')

print('Dataframe has {} rows and {} columns'.format(df.shape[0],df.shape[1]))

df.head()
df['brand']=df['name'].apply(lambda x: ' '.join(x.split(' ')[:2]) if 'Land' in x else x.split(' ')[0])

df['car']=df['name'].apply(lambda x: ' '.join(x.split(' ')[2:]) if 'Land' in x else ' '.join(x.split(' ')[1:]))

df.head()
#list of all brands and cars

brands=list(df['brand'].unique())

cars=list(df['car'].unique())
bharat_stage= [text for idx,text in enumerate(cars) if "BS" in text]

bharat_stage[:5]
def removing_BS(text):

    text= re.sub('BS-VI|BS-IV|BS-III|BS-II|BS VI|BS IV|BS III|BS II|BS I|BSVI|BSIV|BSIII|BSII|BS I|','',text)

    return text



Stopwords=set(stopwords.words('english'))

def removing_stopwords(text):

    return " ".join([word for word in str(text).split() if word not in Stopwords])



df['car']=df['car'].apply(lambda text: removing_BS(text).lower())

df['car']=df['car'].apply(lambda text:removing_stopwords(text))

df.head()
car_rename={'Maruti':{'alto':'Alto','dzire':'Swift Dzire','swift':'Swift','wagon':'Wagon R','ertiga':'Ertiga','sx4':'SX4',

                      'celerio':'Celerio','cross':'S-Cross','zen':'Zen','baleno':'Baleno','eeco':'Eeco','omni':'Omni',

                      'star':'A-Star','ritz':'Ritz','esteem':'Esteem','800':'800','vitara':'Brezza','ignis':'Ignis',

                      'gypsy':'Gypsy','presso':'S-Presso','ciaz':'Ciaz','estilo':'Estilo'},

           'Hyundai':{'elite':'Elite i20','grand':'Grand i10','active':'i20 active','eon':'Eon','santro':'Santro','sonata':'Sonata',

                      'i10':'i10','verna':'Verna','i20':'i20','creta':'Creta','xcent':'Xcent','accent':'Accent','getz':'Getz',

                      'elantra':'Elantra','venue':'Venue','tucson':'Tucson','santa':'Santa Fe'},

           'Datsun':{'redi':'Redigo','plus':'Go Plus','go':'Go'},

           'Honda':{'amaze':'Amaze','jazz':'Jazz','city':'City','br-v':'BR-V','wr-v':'WR-V','brio':'Brio','mobilio':'Mobilio',

                   'civic':'Civic','accord':'Accord','cr-v':'CR-V','brv':'BR-V'},

           'Tata':{'nano':'Nano','vista':'Indica Vista','indigo':'Indigo','indica':'Indica','safari':'Safari','nexon':'Nexon',

                  'manza':'Manza','zest':'Zest','sumo':'Sumo','bolt':'Bolt','hexa':'Hexa','tigor':'Tigor','tiago':'Tiago',

                  'xenon':'Nexon','harrier':'Harrier','aria':'Aria','altroz':'Altroz','venture':'Venture','spacio':'Sumo','winger':'Winger'},

           'Chevrolet':{'beat':'Beat','spark':'Spark','cruze':'Cruze','sail':'Sail','optra':'Optra','aveo':'Aveo','enjoy':'Enjoy',

                       'captiva':'Captiva','tavera':'Tavera'},

           'Toyota':{'innova':'Innova','fortuner':'Fortuner','etios':'Etios','corolla':'Corolla','camry':'Camry','yaris':'Yaris',

                    'qualis':'Qualis'},

           'Jaguar':{'xf 3.0 litre':'XF','xf 5.0 litre':'XF','xj 5.0':'XJ','xf 2.2 litre':'XF'},

           'Mercedes-Benz':{'c-class':'C-Class','e-class':'E-Class','m-class':'M-Class','b-class':'B-Class',

                            's-class':'S-Class','gls':'GLS','gl-class 350':'GL350','b class':'B-Class'},

           'Audi':{'a6':'A6','a4':'A4','q3':'Q3','q5':'Q5','a8':'A8','q3':'Q3','q7':'Q7','a5':'A5','rs7':'RS7'},

           'Skoda':{'laura':'Laura','rapid':'Rapid','superb':'Superb','octavia':'Octavia','fabia':'Fabia','yeti':'Yeti'},

           'BMW':{'x5':'X5','x1':'X1','7 series':'7 Series','5 series':'5 Series','3 series':'3 Series'},

           'Mahindra':{'xuv500':'XUV500','bolero':'Bolero','xylo':'Xylo','scorpio':'Scorpio','quanto':'Quanto','verito':'Verito',

                      'tuv 300':'TUV 300','kuv':'KUV','thar':'Thar','marazzo':'Marazzo','renault logan':'Verito','jeep':'Jeep',

                      'nuvosport':'NuvoSport','alturas':'Alturas','ingenio':'Imperio','xuv300':'XUV300','supro':'Supro'},

           'Ford':{'figo':'Figo','ecosport':'EcoSport','endeavour':'Endeavour','fiesta':'Fiesta','freestyle':'Freestyle','ikon':'Ikon',

                  'aspire':'Aspire','classic':'Fiesta','fusion':'Fusion'},

           'Nissan':{'micra':'Micra','sunny':'Sunny','terrano':'Terrano','evalia':'Evalia','trail':'X-Trail','kicks':'Kicks'},

           'Renault':{'kwid':'Kwid','duster':'Duster','scala':'Scala','lodgy':'Lodgy','captur':'Captur','fluence':'Fluence','pulse':'Pulse',

                     'triber':'Triber','koleos':'Koleos'},

           'Volkswagen':{'polo':'Polo','vento':'Vento','ameo':'Ameo','jetta':'Jetta','crosspolo':'CrossPolo','passat':'Passat'},

           'Volvo':{'v40':'V40','xc60':'XC60','xc 90':'XC90'},

           'Land Rover':{'discovery':'Discovery','evoque':'Range Rover Evoque','range rover 4.4 diesel lwb vogue':'range rover 4.4'}}
# function to rename car names

def rename_car(brnd,x):

    text=x

    temp_dict=car_rename[brnd]

    for key,val in temp_dict.items():

        if key in text:

            text=val

            return(text)

            break

        else:continue

df['Model']=0

col_num=df.columns.get_loc("Model")

for idx in range(0,df.shape[0]):

    brand=df.iloc[idx]['brand']

    car=df.iloc[idx]['car']

    df.iloc[idx,col_num]=rename_car(brand,car)

    

df.drop(['car','name'],axis=1,inplace=True)

df['vehicle_age']=2020-df['year']

col_order=[  'brand', 'Model','year','vehicle_age', 'km_driven', 'fuel','transmission', 'seller_type', 'owner','selling_price']

df=df[col_order]

df.head()
hatchback=['800','Wagon R','Alto','Celerio','Tigor','i10','Santro','Grand i10','i20','Swift','Indica','Eon',

          'Indica Vista','Getz','Elite i20','Brio','Micra','Kwid','Beat','Zen','Baleno','Nano','Figo','Spark',

          'Bolt','Fabia','Jazz','Tiago','A-Star','Polo','Ritz','Estilo','Pulse','Ignis','Freestyle','S-Presso',

          'Altroz','Redigo','Go Plus','Go','B-Class','V40']

sedan=['Verna','Indigo','Corolla','Ciaz','City','A6','Superb','3 Series','Elantra','Swift Dzire','Etios','Civic',

      'Rapid','A8','Jetta','A4','SX4','7 Series','Sonata','Cruze','Vento','Esteem','5 Series','Scala','Verito',

      'Optra','Manza','Accord','Ikon','Laura','Octavia','Accent','Sunny','A5','Camry','Passat','Fusion','Fluence',

      'RS7','Yaris','XF','XJ','C-Class','E-Class','S-Class','M-Class']

compact_sedan=['Amaze','Xcent','Sail','Ameo','Zest','Aspire','Aveo','Fiesta']

suv=['Q5','Q7','Scorpio','Jeep','XUV500','Bolero','Sumo','Yeti','Endeavour','Safari','Fortuner','BR-V','Tucson',

    'X5','Gypsy','Hexa','Captiva','Thar','Alturas','Aria','CR-V','Santa Fe','Koleos','Harrier','X-Trail','GLS','GL350',

    'Discovery','Range Rover Evoque','range rover 4.4','XC90','XC60']

muv=['Enjoy','Innova','Tavera','Xylo','Ertiga','Quanto','Mobilio','Marazzo','Lodgy','NuvoSport','Evalia','Winger','Qualis']

compact_suv=['Creta','Brezza','EcoSport','Terrano','Duster','X1','XUV300','TUV 300','WR-V','Q3','Kicks','Triber']

crossover=['Venue','S-Cross','i20 active','KUV','Nexon','Captur']

minivan=['Omni','Eeco','Supro','Venture']

pickup=['Imperio']



vehicle_class={'hatchback':hatchback,'compact_sedan':compact_sedan,'sedan':sedan,

               'suv':suv,'compact_suv':compact_suv,'muv':muv,'pickup':pickup,'crossover':crossover,'minivan':minivan}

df['Class']=0

col_num=df.columns.get_loc("Class")

for idx in range(0,df.shape[0]):

    car_name=df.iloc[idx]['Model']

    for key,val in vehicle_class.items():

        if car_name in val:

            df.iloc[idx,col_num]=key

df.head()
print((df['fuel'].value_counts()))

df=df[df['fuel']!='Electric'].reset_index()

df.drop('index',axis=1,inplace=True)
temp_df=df.copy()

temp_df['count']=1

temp_df['Fuel']='Fuel'

fig = px.sunburst(temp_df, path=[ 'Fuel','fuel', 'transmission'], values='count')

fig.show()
df0=df.loc[:,['Class']]

df0['count']=1

df0['Listed']='Vehicle Class'

fig = px.sunburst(df0, path=['Listed','Class'], values='count')

fig.show()
df0=df.loc[:,['brand','fuel','transmission']]

df0['count']=1

df0['Listed']='Cars Listed of Sale'

fig = px.sunburst(df0, path=['Listed','brand', 'fuel', 'transmission'], values='count',width=900, height=900,)

fig.show()
brands=df['brand'].value_counts().index



fig = plt.figure(figsize=(20,100))



for b,num in zip(brands, range(1,len(brands)+1)):

    if b=='Maruti':

        rot=30

    else:

        rot=0

    df0=df[df['brand']==b]

    listed_car_num=df0.shape[0]

    vc0=df0.Model.value_counts(normalize=True)

    ax = fig.add_subplot(len(brands),1,num)

    sns.barplot(x=vc0.index,y=vc0.values*100,ax=ax,palette='tab10')

    plt.xticks(rotation=rot,horizontalalignment='center',fontsize=15)

    plt.yticks(fontsize=15)

    plt.ylabel('% of listed cars',fontsize=15)

    ax.set_title(b+' \nNumber of cars listed for sale : '+str(listed_car_num),fontsize=17)



plt.tight_layout()

plt.show()
Luxury_brand=['Audi','BMW','Mercedes-Benz','Jaguar','Land Rover','Volvo']

luxury_df=df[df['brand'].isin(Luxury_brand)].reset_index()

luxury_df.drop('index',axis=1,inplace=True)

regular_df=df[~df['brand'].isin(Luxury_brand)].reset_index()

regular_df.drop('index',axis=1,inplace=True)

print(luxury_df.shape,regular_df.shape)

luxury_df.head()
plt.figure(figsize=(20,12))

plt.subplot(2,1,1)

class_order=list(regular_df.groupby(['Class'])['selling_price'].mean().reset_index().sort_values('selling_price')['Class'])



sns.boxplot(x='Class',y='selling_price',data=regular_df,hue='fuel',palette='tab10',order=class_order)

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.xticks(fontsize=11)

plt.xlabel('Class',fontsize=13)

plt.yticks(fontsize=10)

plt.ylabel('Selling Price',fontsize=13)

plt.title("selling price of different class vehicles(Regular) and fuels",fontsize=14)



plt.subplot(2,1,2)

class_order=list(luxury_df.groupby(['Class'])['selling_price'].mean().reset_index().sort_values('selling_price')['Class'])

sns.boxplot(x='Class',y='selling_price',data=luxury_df,hue='fuel',hue_order=['Petrol','Diesel'],palette='tab10',order=class_order)

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.xticks(fontsize=11)

plt.xlabel('Class',fontsize=13)

plt.yticks(fontsize=10)

plt.ylabel('Selling Price',fontsize=13)

plt.title("selling price of different class vehicles(Luxury) and fuels",fontsize=14)



plt.tight_layout()
plt.figure(figsize=(20,12))

plt.subplot(2,1,1)

class_order=list(regular_df.groupby(['Class'])['selling_price'].mean().reset_index().sort_values('selling_price')['Class'])



sns.boxplot(x='Class',y='selling_price',data=regular_df,hue='transmission',hue_order=['Automatic','Manual'],palette='tab10',order=class_order)

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.xticks(fontsize=11)

plt.xlabel('Class',fontsize=13)

plt.yticks(fontsize=10)

plt.ylabel('Selling Price',fontsize=13)

plt.title("selling price of different class vehicles(Regular) and transmission",fontsize=14)



plt.subplot(2,1,2)

class_order=list(luxury_df.groupby(['Class'])['selling_price'].mean().reset_index().sort_values('selling_price')['Class'])

sns.boxplot(x='Class',y='selling_price',data=luxury_df,hue='transmission',hue_order=['Automatic','Manual'],palette='tab10',order=class_order)

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.xticks(fontsize=11)

plt.xlabel('Class',fontsize=13)

plt.yticks(fontsize=10)

plt.ylabel('Selling Price',fontsize=13)

plt.title("selling price of different class vehicles(Luxury) and transmission",fontsize=14)



plt.tight_layout()
plt.figure(figsize=(25,12))

plt.subplot(2,1,1)

sns.boxplot(x='brand',y='km_driven',data=regular_df,palette='tab10')

plt.ylim(0,300000)

plt.xticks(fontsize=11)

plt.xlabel('Brand',fontsize=13)

plt.yticks(fontsize=11)

plt.ylabel('KM Driven',fontsize=13)

plt.title("range of km's driven in each regular brand ",fontsize=14)



plt.subplot(2,1,2)

sns.boxplot(x='brand',y='km_driven',data=luxury_df,palette='tab10')

plt.ylim(0,175000)

plt.xticks(fontsize=11)

plt.xlabel('Brand',fontsize=13)

plt.yticks(fontsize=11)

plt.ylabel('KM Driven',fontsize=13)

plt.title("range of km's driven in each luxury brand ",fontsize=14);
plt.figure(figsize=(25,12))



plt.subplot(2,1,1)

sns.boxplot(x='brand',y='selling_price',data=regular_df,palette='tab10')

plt.ylim(0,2200000)

plt.xticks(fontsize=11)

plt.xlabel('Brand',fontsize=13)

plt.yticks(fontsize=11)

plt.ylabel('selling_price',fontsize=13)

plt.title("range of selling_price for each regular brand ",fontsize=14)





plt.subplot(2,1,2)

sns.boxplot(x='brand',y='selling_price',data=luxury_df,palette='tab10')

plt.ylim(0,6000000)

plt.xticks(fontsize=11)

plt.xlabel('Brand',fontsize=13)

plt.yticks(fontsize=11)

plt.ylabel('selling_price',fontsize=13)

plt.title("range of selling_price for each luxury brand ",fontsize=14);
plt.figure(figsize=(25,12))

plt.subplot(2,1,1)



sns.boxplot(x='brand',y='selling_price',data=regular_df,hue='seller_type',hue_order=['Individual','Dealer','Trustmark Dealer'],palette='tab10')

#plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.xticks(fontsize=11)

plt.xlabel('Class',fontsize=13)

plt.yticks(fontsize=10)

plt.ylabel('Selling Price',fontsize=13)

plt.title("selling price of different class vehicles(Regular) and seller type",fontsize=14)



plt.subplot(2,1,2)

class_order=list(luxury_df.groupby(['seller_type'])['selling_price'].mean().reset_index().sort_values('selling_price')['seller_type'])

sns.boxplot(x='brand',y='selling_price',data=luxury_df,hue='seller_type',hue_order=['Individual','Dealer'],palette='tab10')

#plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.xticks(fontsize=11)

plt.xlabel('Class',fontsize=13)

plt.yticks(fontsize=10)

plt.ylabel('Selling Price',fontsize=13)

plt.title("selling price of different class vehicles(Luxury) and seller type",fontsize=14)



plt.tight_layout()
plt.figure(figsize=(25,7))

sns.lineplot(x='vehicle_age',y='selling_price',data=df)

plt.ticklabel_format(style='plain')
plt.figure(figsize=(25,25))

temp_df=df[((df['selling_price']<2000000)&(df['km_driven']<500000))]

fig = px.scatter_3d(temp_df, x='year', y='km_driven', z='selling_price',

              color='Class')

fig.show()
shape_before=df.shape

vc=df.brand.value_counts(normalize=True)

low_freq_brands=list(vc[vc.values<=0.02].index)



df=df[~df['brand'].isin(low_freq_brands)]

df=df.reset_index()

df.drop('index',axis=1,inplace=True)

df.head()

print(shape_before,df.shape)
df.drop(['brand','year'],axis=1,inplace=True)

df=df[['Model','vehicle_age','km_driven','fuel','transmission','Class','owner','seller_type','selling_price']]

df.head()
vc0=df.Model.value_counts()

low_freq=list(vc0[vc0.values<2].index)

shape_before=df.shape

df=df[~df['Model'].isin(low_freq)].reset_index()

df=df.drop('index',axis=1)

print(shape_before,df.shape)

df.head()
target='selling_price'

x=list(df.columns)

x.remove(target)
x_train, x_test, y_train, y_test = train_test_split(df[x],df[target],random_state=10,stratify=df['Model'],test_size=0.25)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)



x_train.head()
def outlier_flag(df_in, col_name):

    q1 = df_in[col_name].quantile(0.25)

    q3 = df_in[col_name].quantile(0.75)

    iqr = q3-q1 #Interquartile range

    fence_low  = q1-1.5*iqr

    fence_high = q3+1.5*iqr

    df_in.loc[((df_in[col_name] < fence_low) | (df_in[col_name] > fence_high)),'outlier_flag']='o_'+col_name

    return df_in
for data in [x_train,x_test]:

    for col in ['km_driven','vehicle_age']:

        data=outlier_flag(data,col)

    data['outlier_flag'].fillna(value='no_outlier',inplace=True)
te=ce.TargetEncoder(verbose=1,cols=['Model','Class'])
x_train=te.fit_transform(x_train,y_train,)

x_test=te.transform(x_test)

x_train.head()
x_train=pd.get_dummies(x_train)

x_test=pd.get_dummies(x_test)

print(x_train.shape,x_test.shape)
print(x_train.shape,x_test.shape)
#importing required packages for model building

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error,r2_score
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):

    

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor()
#creating parameter grid for random search

grid_forest_1={'criterion':['mse','mae'],

      'n_estimators':np.arange(5,100,10),

      'max_depth':np.arange(2,7,1),

      'min_samples_split':np.arange(0.01,0.1,0.01),

      'max_features':['log2','sqrt','auto'],    

      'min_weight_fraction_leaf':np.arange(0.001,0.25,0.05)

}



rf_random=RandomizedSearchCV(estimator=rf,param_distributions=grid_forest_1,n_iter=500,n_jobs=-1,cv=3,verbose=1,random_state=1)





rf_random.fit(x_train,y_train)

rf_random=rf_random.best_estimator_

np.sqrt(mean_squared_error(rf_random.predict(x_test),y_test))
plot_learning_curve(estimator=rf_random,title='RF_learning_curves',X=x_train,y=y_train,ylim=(0.5,1.05),cv=5)
rf_random
grid_forest_2={'criterion':['mae'],

      'n_estimators':np.arange(60,80,5),

      'max_depth':(6,7,8),

      'min_samples_split':np.arange(0.001,0.01,0.008),

      'max_features':['auto'],    

      'min_weight_fraction_leaf':np.arange(0.001,0.1,0.008)

}
rf=RandomForestRegressor()

grid_search_rf=GridSearchCV(estimator=rf,param_grid = grid_forest_2,cv=3,n_jobs=-1,verbose=1)

grid_search_rf.fit(x_train,y_train)
grid_search_rf=grid_search_rf.best_estimator_

grid_search_rf.fit(x_train,y_train)
plot_learning_curve(estimator=grid_search_rf,title='RF_learning_curves',X=x_train,y=y_train,ylim=(0.5,1.05),cv=5)
y_pred=grid_search_rf.predict(x_test)

print("\t\tError Table")

print('Mean Absolute Percentage Error  : ', mean_absolute_percentage_error(y_test, y_pred))

print('Root Mean Squared  Error        : ', np.sqrt(mean_squared_error(y_test, y_pred)))

print('R Squared Error                 : ', r2_score(y_test, y_pred))