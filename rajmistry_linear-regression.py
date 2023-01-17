import pandas as pd

pd.set_option('display.max_column',None)

pd.set_option('display.max_rows',None)

pd.set_option('display.max_seq_items',None)

pd.set_option('display.max_colwidth', 500)

pd.set_option('expand_frame_repr', True)

import numpy as np





import matplotlib.pyplot as plt

import seaborn as sns

sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)

from scipy import stats

import warnings

warnings.filterwarnings('ignore')



import statsmodels.api as sm  

from statsmodels.stats.outliers_influence import variance_inflation_factor
def drop_columns(dataframe, axis =1, percent=0.3):

    '''

    * drop_columns function will remove the rows and columns based on parameters provided.

    * dataframe : Name of the dataframe  

    * axis      : axis = 0 defines drop rows, axis =1(default) defines drop columns    

    * percent   : percent of data where column/rows values are null,default is 0.3(30%)

    '''

    df = dataframe.copy()

    ishape = df.shape

    if axis == 0:

        rownames = df.transpose().isnull().sum()

        rownames = list(rownames[rownames.values > percent*len(df)].index)

        df.drop(df.index[rownames],inplace=True) 

        print("\nNumber of Rows dropped\t: ",len(rownames))

    else:

        colnames = (df.isnull().sum()/len(df))

        colnames = list(colnames[colnames.values>=percent].index)

        df.drop(labels = colnames,axis =1,inplace=True)        

        print("Number of Columns dropped\t: ",len(colnames))

        

    print("\nOld dataset rows,columns",ishape,"\nNew dataset rows,columns",df.shape)



    return df

def column_univariate(df,col,type,nrow=None,ncol=None,hue =None):

    

    '''

    * Credit Univariate function will plot the graphs based on the parameters.

    * df      : dataframe name

    * col     : Column name

    * nrow    : no of rows in sub plot

    * ncol    : no of cols in sub plot

    * type : variable type : continuos or categorical

                Continuos(0)   : Distribution, Violin & Boxplot will be plotted.

                Categorical(1) : Countplot will be plotted.

                Categorical(2) : Subplot-Countplot will be plotted.

    * hue     : It's only applicable for categorical analysis.

    

    '''

    sns.set(style="darkgrid")

    

    if type == 0:

        

        fig, axes =plt.subplots(nrows =1,ncols=3,figsize=(14,6))

        axes [0].set_title(" Distribution Plot")

        sns.distplot(df[col],ax=axes[0])

        axes [1].set_title("Violin Plot")

        sns.violinplot(data =df, x=col,ax=axes[1], inner="quartile")

        axes [2].set_title(" Box Plot")

        sns.boxplot(data =df, x=col,ax=axes[2],orient='v')

        

        for ax in axes:

            ax.set_xlabel('Common x-label')

            ax.set_ylabel('Common y-label')

        

    if  type == 1:

        total_len = len(df[col])

        percentage_labels = round((df[col].value_counts()/total_len)*100,4)

    

        temp = pd.Series(data = hue)

        

        fig, ax=plt.subplots(nrows =1,ncols=1,figsize=(12,4))

        ax.set_title("Count Plot")

        width = len(df[col].unique()) + 6 + 4*len(temp.unique())

        fig.set_size_inches(width , 7)

        sns.countplot(data = df, x= col,

                           order=df[col].value_counts().index,hue = hue)  

        mystring = col.replace("_", " ").upper()

        plt.xlabel(mystring)

         

        

        if len(temp.unique()) > 0:

            for p in ax.patches:

                ax.annotate('{:1.1f}%'.format((p.get_height()*100)/float(len(df))),

                            (p.get_x()+0.05, p.get_height()+20))  

        else:

            for p in ax.patches:

                height = p.get_height()

                ax.text(p.get_x() + p.get_width()/2.,

                height + 2,'{:.2f}%'.format(100*(height/total_len)),

                        fontsize=14, ha='center', va='bottom')

        del temp

        

    elif type == 2:

        fig, ax = plt.subplots(nrow, ncol, figsize=(14, 10))

        for variable, subplot in zip(col, ax.flatten()):

            total = float(len(df[variable]))

            ax=sns.countplot(data = df, x= variable,ax=subplot,

                           order=df[variable].value_counts().index) 

            for p in ax.patches:    

                height = p.get_height()

                ax.text(p.get_x()+p.get_width()/2., height + 3, '{:1.2f}%'.format((height/total)*100),

                        ha="center") 

    else:

        exit

    

    plt.show()

def drop_rows(df,col,percent):

    '''

    * drop_rows function drop null values  based on the parameters.

    * df      : dataframe name

    * col     : Column name

    * percent : percentage to be dropped.

    '''

    drop_record = (df[col].value_counts()*100)/len(df)

    drop_record = drop_record [(drop_record  < percent) | (drop_record.index == 'other')]



    df.drop(labels = df[df[col].isin(drop_record.index)].index, inplace=True)

    print("left with",df.shape ,"rows & columns.")

    print(df[col].unique())

def remove_outlier(df_in, col_name):

    '''

    * remove_outlier function to remove outliers from col

    * df_in      : dataframe name

    * col_name     : Column name

    '''

    q1 = df_in[col_name].quantile(0.25)

    q3 = df_in[col_name].quantile(0.75)

    iqr = q3-q1 #Interquartile range 

    fence_low  = q1-1.5*iqr 

    fence_high = q3+1.5*iqr 

    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]

    return df_out    



def count_outlier(df_in, col_name):

    '''

    * count_outlier function count and report outlier in percentage based on parameters.

    * df_in      : dataframe name

    * col_name     : Column name

    '''

    if df_in[col_name].nunique() > 2:

        orglength = len(df_in[col_name])

        q1 = df_in[col_name].quantile(0.25)

        q3 = df_in[col_name].quantile(0.75)

        iqr = q3-q1 #Interquartile range 

        fence_low  = q1-1.5*iqr 

        fence_high = q3+1.5*iqr 

        df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]

        newlength = len(df_out[col_name])

        return round(100 - (newlength*100/orglength),2)  

    else:

        return 0



def column_scatter(df,xcol,ycol,nrows,ncols,figno):

    '''

    * column_scatter will plot scatter graph based on the parameters.

    * df   : dataframe

    * xcol : xcol/x-axis variable

    * ycol : ycol/y-axis variable

    * nrows: no. of rows for sub-plots

    * ncols: no. of cols for sub-plots

    

    '''

    plt.subplot(nrows,ncols,figno)

    plt.scatter(df[xcol],df[ycol])

    plt.title(xcol+' vs ' + ycol)

    plt.ylabel(ycol)

    plt.xlabel(xcol)

    plt.tight_layout()  

    

def column_percent(df_in,col):

    '''

    * column percent will calculate percentage based on the parameters.

    * df_in        : dataframe name

    * col_name     : Column name

    '''

    print(100*df_in.groupby(col)['sk_id_curr'].count()/len(df_in[col]))

def get_column_dummies(col,df_in):

    '''

    * get_column_dummies will get/map column dummies values based on the parameters.

    * col   : column name

    * df_in  : dataframe

    '''

    temp = pd.get_dummies(df_in[col], drop_first = True)

    df_out = pd.concat([df_in, temp], axis = 1)

    df_out.drop([col], axis = 1, inplace = True)

    return df_out

def build_state_model(X,y):

    '''

    * build_state_model will build stat-model based on the parameters.

    * X   : constant

    * y   : target/dependent variable 

    '''

    X = sm.add_constant(X) #Adding the constant

    lm = sm.OLS(y,X).fit() # fitting the model

    print(lm.summary()) # model summary

    return X

def get_vifs(X):

    '''

    * get_vifs will calculate variance inflation factor based on formula for calculating `VIF` is:

    * ### $ VIF_i = \frac{1}{1 - {R_i}^2} $

    * X: dataframe

    '''

    vif = pd.DataFrame()

    vif['Features'] = X.columns

    vif['Variance Inflation Factor(VIF)'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif['Variance Inflation Factor(VIF)'] = round(vif['Variance Inflation Factor(VIF)'], 2)

    vif = vif.sort_values(by = "Variance Inflation Factor(VIF)", ascending = False)

    return(vif)
#Columns description view of data source file : "Data Dictionary - carprices.xlsx"

data_dic = pd.read_excel("../input/Data Dictionary - carprices.xlsx")

data_dic.rename({"Unnamed: 7":"column"}, axis="columns", inplace=True)

data_dic.rename({"Unnamed: 11":"Description"}, axis="columns", inplace=True)

data_dic.drop(data_dic.columns[data_dic.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

data_dic.drop(data_dic.index[0:2],inplace = True)

data_dic=data_dic[:-2].reset_index(drop=True)

data_dic
df=pd.read_csv("../input/CarPrice_Assignment.csv")
df.head(3)
df.shape
df.info()
df.describe()
df.dtypes.value_counts()
df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
plt.figure(figsize = (20,10))  

sns.heatmap(df.corr(method='pearson'), annot = True, cmap="RdBu")

#Splitting company name from CarName column

companyname = df.CarName.apply(lambda value : value.split()[0])

df.insert(1,"companyName",companyname)

df.drop(['CarName'],axis=1,inplace=True)

df.head()
#checking unique values for car producer/company.

df.companyName.unique()
df['companyName'].replace('maxda','mazda',inplace=True)

df['companyName'].replace('Nissan','nissan',inplace=True)

df['companyName'].replace('porcshce','porsche',inplace=True)

df['companyName'].replace('toyouta','toyota',inplace=True)

df['companyName'].replace(['vokswagen','vw'],'volkswagen',inplace=True)
df.isnull().sum()*100/df.shape[0]
df.loc[df.duplicated()]
df.columns
cleancolumn = []

for i in range(len(df.columns)):

    cleancolumn.append(df.columns[i].replace('-', '').lower())

df.columns = cleancolumn
numerical = df.select_dtypes(include=[np.number]).columns.tolist()

numerical
categorical = df.select_dtypes(include=[np.object]).columns.tolist()

categorical
for col in numerical:

    if col !="car_id" :

        print(col,"=",count_outlier(df,col))
cols=['price','symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight',

 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',

 'horsepower',  'peakrpm', 'citympg', 'highwaympg']

dfo=df[cols]

plt.figure(figsize=(12,6))

plt.title("Contneious variable values distribution")

sns.boxplot(x="variable", y="value", data=pd.melt(dfo))

plt.xticks(rotation=90)

plt.yscale('log')

plt.show()
print("Plot 1:")

column_univariate(df,"price",0) 

summary=df.price.describe(percentiles = [0.25,0.50,0.75,0.85,0.90,1])



print("Plot 2:")

plt.rcParams['figure.figsize'] = (14,6)

prices = pd.DataFrame({"price":df["price"], "log(price + 1)":np.log1p(df["price"])})

prices.hist(bins = 50)

print(summary)
print("Plot 3:")

cols=['carlength', 'carwidth',  'carheight', 'curbweight' ]

plt.figure(figsize=(10,10))



nrows,ncols=2,2

count=nrows*ncols

for col in cols:

    column_scatter(df,col,'price',nrows,ncols,count)

    count-=1    
print("Plot 4:")

cols=['enginesize', 'boreratio', 'stroke', 

      'compressionratio','horsepower','peakrpm',

      'wheelbase', 'citympg','highwaympg'

       ]

plt.figure(figsize=(20,15))

nrows,ncols=3,3

count=nrows*ncols

for col in cols:

    column_scatter(df,col,'price',nrows,ncols,count)

    count-=1



    
print("Plot 4.1:")

plt.figure(figsize=(16,4))

column_scatter(df,'cylindernumber','price',1,4,1)

column_scatter(df,'horsepower','peakrpm',1,4,2)

column_scatter(df,'cylindernumber','enginesize',1,4,3)

 
print("Plot 5:")

column_univariate(df,"companyname",1) 
print("Plot 6:")

column_univariate(df,"fueltype",1) 

print("Plot 7:")

column_univariate(df,"carbody",1) 
print("Plot 8:")

column_univariate(df,'aspiration',1)
print("Plot 9:")

col=['doornumber','drivewheel', 'enginelocation', 

     'enginetype', 'cylindernumber', 'fuelsystem']

column_univariate(df,col,2,2,3)

print("Plot 10:")

result = df[['fueltype','price']].groupby("fueltype",

                                        as_index = False).mean().rename(columns={'price':'Fuel Type Average Price'})

Plt = result.plot(x = 'fueltype', kind='bar',legend = False, sort_columns = True,figsize=(8,6))

Plt.set_xlabel("Fuel Type")

Plt.set_ylabel("Average Price")

plt.xticks(rotation = 0)

plt.title("Fuel Type vs Average Price")

plt.show()

result

print("Plot 11:")

plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

plt.title('Symboling Distribution')

sns.distplot(df.symboling)



plt.subplot(1,2,2)

plt.title('Symboling vs Price')

sns.boxplot(x=df.symboling, y=df.price )

plt.show()
print("Plot 12:")

result = df[['aspiration','price']].groupby("aspiration",

                                            as_index = False).mean().rename(columns={'price':'Aspiration Average Price'})

Plt = result.plot(x = 'aspiration', kind='bar',legend = False, sort_columns = True,figsize = (9,6))

Plt.set_xlabel("Aspiration")

Plt.set_ylabel("Average Price")

plt.title("Aspiration vs Average Price ")

plt.show()

result
print("Plot 13:")

result = df[['companyname','price']].groupby("companyname",

as_index = False).mean().rename(columns={'price':'company_car_avg_price'}).sort_values(by='company_car_avg_price',

                                     ascending=False)

Plt = result.plot(x = 'companyname', kind='bar',legend = False, sort_columns = True, figsize = (15,5))

Plt.set_xlabel("Company Name")

Plt.set_ylabel("Average Price")

plt.title("Company vs Average Price")

plt.show()

result.nlargest(5,'company_car_avg_price') 
print("Plot 14:")

result = df[['doornumber','price']].groupby("doornumber", 

                                            as_index = False).mean().rename(columns={'price':'door_avg_price'})

print(result)

Plt = result.plot(x = 'doornumber', kind='bar',legend = False, sort_columns = True,figsize = (8,6))

Plt.set_xlabel("No of Doors")

Plt.set_ylabel("Average Price")

plt.title("No of Doors Vs Average Price")

plt.show()

print("Plot 15:")

result = df[['carbody','price']].groupby("carbody", 

    as_index = False).mean().rename(columns={'price':'carbody_avg_price'}).sort_values(by='carbody_avg_price',

                                                                                       ascending=False) 

 

Plt = result.plot(x = 'carbody', kind='bar',legend = False, sort_columns = True,figsize = (9,6))

Plt.set_xlabel("Car Body")

Plt.set_ylabel("Average Price")

plt.title("Car Body Type vs Average Price")

plt.show() 

result



print("Plot 16:")

result = df[['drivewheel','price']].groupby("drivewheel", 

            as_index = False).mean().rename(

            columns={'price':'drivewheel_avg_price'}).sort_values(by='drivewheel_avg_price',ascending=False)

Plt = result.plot(x = 'drivewheel', 

                                    kind='bar', sort_columns = True,legend = False,figsize = (9,6))

Plt.set_xlabel("Drive Wheel Type")

Plt.set_ylabel("Average Price")

plt.title("Drive Wheel Type vs Average Price")

plt.show()

result

print("Plot 17:")

result = df[['enginelocation','price']].groupby("enginelocation", 

            as_index = False).mean().rename(

            columns={'price':'enginelocation_avg_price'}).sort_values(by='enginelocation_avg_price',ascending=False)

Plt = result.plot(x = 'enginelocation', 

                                    kind='bar', sort_columns = True,legend = False,figsize = (9,6))

Plt.set_xlabel("Engine Location Type")

Plt.set_ylabel("Average Price")

plt.title("Engine Location vs Average Price")

plt.show()

result
df['fuel-efficiency'] = (0.55 * df['citympg']) + (0.45 * df['highwaympg'])
print("Plot 18:")

df['price'] = df['price'].astype('int')

temp = df.copy()

table = temp.groupby(['companyname'])['price'].mean()

temp = temp.merge(table.reset_index(), how='left',on='companyname')

bins = [0,10000,20000,40000]

df_bin=['LowTier','MediumTier','HighTier']

df['company-segment'] = pd.cut(temp['price_y'],bins,right=False,labels=df_bin)

df.head()
print("Plot 19:")

plt.figure(figsize=(8,8))

plt.title('Fuel Efficiency vs Price')

sns.scatterplot(x='fuel-efficiency',y='price',data=df,hue='company-segment')

plt.xlabel('Fuel Efficiency')

plt.ylabel('Price')

plt.show()

plt.tight_layout() 

print("Plot 20:")

result = pd.DataFrame(df.groupby(['fuelsystem','drivewheel','company-segment'])['price'].mean().unstack(fill_value=0))

ax=result.plot(kind='barh', figsize=(10, 10), zorder=1, width=0.8,stacked=True)

plt.title('Company Segment vs Average Price')

plt.xlabel('Average Price ')

plt.ylabel(' Fuel system,Drive Wheel and Body Type')

plt.show()
print("Plot 21:")

plt.figure(figsize=(10,8))

plt1 = sns.scatterplot(x = 'horsepower', y = 'price', hue = 'company-segment', data = df)

plt1.set_xlabel('Horsepower')

plt1.set_ylabel('Price of Car ')

plt.title("Horse Power vs Company Segment")

plt.show()
dfs = df[['price', 

          'company-segment','enginetype','fueltype', 'carbody', 'aspiration',

          'cylindernumber','drivewheel','curbweight','carlength','carwidth',

          'enginesize', 'boreratio','horsepower', 'wheelbase','fuel-efficiency'

         ]]

dfs.head()

print("Plot 22:")

plt.figure(figsize=(15, 15))

sns.pairplot(dfs, hue="company-segment",palette='dark') 

plt.show()
categorical = [ 'fueltype','aspiration',

                'carbody', 'drivewheel',

                'enginetype','cylindernumber',

                'company-segment'] 



for col in categorical:

    dfs=get_column_dummies(col,dfs)

dfs.head()
dfs.shape
from sklearn.model_selection import train_test_split



np.random.seed(0) 

dfs_train, dfs_test = train_test_split(dfs, train_size = 0.7, test_size = 0.3, random_state = 100)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

numeric_cols = ['wheelbase', 'curbweight', 'enginesize',

                'boreratio', 'horsepower','fuel-efficiency',

                'carlength','carwidth','price']

dfs_train[numeric_cols] = scaler.fit_transform(dfs_train[numeric_cols])
dfs_train.head()
dfs_train.describe()
print("Plot 23:")

plt.figure(figsize = (18, 12))

result=dfs_train.corr(method='pearson')

sns.heatmap(result, annot = True, cmap="RdBu")

plt.show()

 
y_train = dfs_train.pop('price')

X_train = dfs_train
X_train_check = build_state_model(X_train,y_train)

get_vifs(X_train_check)
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

 
lm = LinearRegression()

lm.fit(X_train,y_train)

rfe = RFE(lm, 10)

rfe = rfe.fit(X_train, y_train)


list(zip(X_train.columns,rfe.support_,rfe.ranking_))
X_train.columns[rfe.support_]
X_train_rfe = X_train[X_train.columns[rfe.support_]]

X_train_rfe.head()
X_train_new = build_state_model(X_train_rfe,y_train)
get_vifs(X_train_new)
X_train_new = X_train_rfe.drop(["twelve"], axis = 1)
X_train_new = build_state_model(X_train_new,y_train)
get_vifs(X_train_new)
X_train_new = X_train_new.drop(["fuel-efficiency"], axis = 1)
X_train_new = build_state_model(X_train_new,y_train)
get_vifs(X_train_new)
X_train_new = X_train_new.drop(["curbweight"], axis = 1)
X_train_new = build_state_model(X_train_new,y_train)
get_vifs(X_train_new)
X_train_new = X_train_new.drop(["sedan"], axis = 1)
X_train_new = build_state_model(X_train_new,y_train)
get_vifs(X_train_new)
X_train_new = X_train_new.drop(["wagon"], axis = 1)
X_train_new = build_state_model(X_train_new,y_train)
get_vifs(X_train_new)
X_train_new = X_train_new.drop(["dohcv"], axis = 1)

X_train_new = build_state_model(X_train_new,y_train)

 
get_vifs(X_train_new)
lm = sm.OLS(y_train,X_train_new).fit()

y_train_price = lm.predict(X_train_new)
# Plot the histogram of the error terms

print("Plot 24:")

fig = plt.figure(figsize=(8,6))

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)#  

plt.xlabel('Errors', fontsize = 18)





              
#Scaling the test set

num_vars = ['wheelbase', 'curbweight','enginesize',

            'boreratio', 'horsepower','fuel-efficiency',

            'carlength',  'carwidth',

            'price']



dfs_test[num_vars] = scaler.transform(dfs_test[num_vars])

y_test = dfs_test.pop('price')

X_test = dfs_test
# Now let's use our model to make predictions.

X_train_new = X_train_new.drop('const',axis=1)

# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[X_train_new.columns]



# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
y_pred = lm.predict(X_test_new)
from sklearn.metrics import mean_squared_error,r2_score 



mse = mean_squared_error(y_test, y_pred)

r_squared = r2_score(y_test, y_pred)



print('Mean_Squared_Error :' ,mse)

print('r_square_value :',r_squared)
print("Plot 25:")

# Plotting y_test and y_pred to understand the spread.

fig = plt.figure(figsize=(8,8))

sns.scatterplot(y_test,y_pred)



fig.suptitle('y_test vs y_pred', fontsize=20)       

plt.xlabel('y_test', fontsize=18)                    

plt.ylabel('y_pred', fontsize=16)

print(lm.summary())
factors = [('Popular Car Features', ['Fuel Type','Car Body Style','Aspiration',

                                'Number of Doors in Car','Drive Wheel','Engine Location',

                                'Engine Type','Number of Cylinder in Car','Fuel System',

                                'Symboling/Risk Factor']),

         ('Value', ['Gas',

                    'Sedan',

                    'Standard',

                    'Four',

                    'Forward','Front' ,'Ohc','Four','mpfi','moderate (0,1)'])

         ]

dfi = pd.DataFrame.from_items(factors)

dfi