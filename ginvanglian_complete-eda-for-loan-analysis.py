import numpy as np

print('numpy version\t:',np.__version__)

import pandas as pd

print('pandas version\t:',pd.__version__)

import matplotlib.pyplot as plt

%matplotlib inline

from scipy import stats



# Regular expressions

import re



# seaborn : advanced visualization

import seaborn as sns

print('seaborn version\t:',sns.__version__)



pd.options.mode.chained_assignment = None #set it to None to remove SettingWithCopyWarning

pd.options.display.float_format = '{:.4f}'.format #set it to convert scientific noations such as 4.225108e+11 to 422510842796.00

pd.set_option('display.max_columns', 100) # to display all the columns



np.set_printoptions(suppress=True,formatter={'float_kind':'{:f}'.format})



import os



import warnings

warnings.filterwarnings('ignore') # if there are any warning due to version mismatch, it will be ignored
print(os.listdir("../input"))
loan = pd.read_csv('../input/loan.csv',dtype='object')

print(loan.shape)
loan.head(2)
NA_col = loan.isnull().sum()

NA_col = NA_col[NA_col.values >(0.3*len(loan))]

plt.figure(figsize=(20,4))

NA_col.plot(kind='bar')

plt.title('List of Columns & NA counts where NA values are more than 30%')

plt.show()
NA_col
def removeNulls(dataframe, axis =1, percent=0.3):

    '''

    * removeNull function will remove the rows and columns based on parameters provided.

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
loan = removeNulls(loan, axis =1,percent = 0.3)
loan = removeNulls(loan, axis =0,percent = 0.3)
unique = loan.nunique()

unique = unique[unique.values == 1]
loan.drop(labels = list(unique.index), axis =1, inplace=True)

print("So now we are left with",loan.shape ,"rows & columns.")
print(loan.emp_length.unique())

loan.emp_length.fillna('0',inplace=True)

loan.emp_length.replace(['n/a'],'Self-Employed',inplace=True)

print(loan.emp_length.unique())
#not_required_columns = ["id","member_id","url","zip_code"]

#loan.drop(labels = not_required_columns, axis =1, inplace=True)

#print("So now we are left with",loan.shape ,"rows & columns.")
numeric_columns = ['loan_amnt','funded_amnt','funded_amnt_inv','installment','int_rate','annual_inc','dti']



loan[numeric_columns] = loan[numeric_columns].apply(pd.to_numeric)
loan.tail(3)
(loan.purpose.value_counts()*100)/len(loan)
del_loan_purpose = (loan.purpose.value_counts()*100)/len(loan)

del_loan_purpose = del_loan_purpose[(del_loan_purpose < 0.75) | (del_loan_purpose.index == 'other')]



loan.drop(labels = loan[loan.purpose.isin(del_loan_purpose.index)].index, inplace=True)

print("So now we are left with",loan.shape ,"rows & columns.")



print(loan.purpose.unique())
(loan.loan_status.value_counts()*100)/len(loan)
del_loan_status = (loan.loan_status.value_counts()*100)/len(loan)

del_loan_status = del_loan_status[(del_loan_status < 1.5)]



loan.drop(labels = loan[loan.loan_status.isin(del_loan_status.index)].index, inplace=True)

print("So now we are left with",loan.shape ,"rows & columns.")



print(loan.loan_status.unique())
loan['loan_income_ratio']= loan['loan_amnt']/loan['annual_inc']
loan['issue_month'],loan['issue_year'] = loan['issue_d'].str.split('-', 1).str

loan[['issue_d','issue_month','issue_year']].head()
months_order = ["Jan", "Feb", "Mar", "Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

loan['issue_month'] = pd.Categorical(loan['issue_month'],categories=months_order, ordered=True)
bins = [0, 5000, 10000, 15000, 20000, 25000,40000]

slot = ['0-5000', '5000-10000', '10000-15000', '15000-20000', '20000-25000','25000 and above']

loan['loan_amnt_range'] = pd.cut(loan['loan_amnt'], bins, labels=slot)
bins = [0, 25000, 50000, 75000, 100000,1000000]

slot = ['0-25000', '25000-50000', '50000-75000', '75000-100000', '100000 and above']

loan['annual_inc_range'] = pd.cut(loan['annual_inc'], bins, labels=slot)
bins = [0, 7.5, 10, 12.5, 15,20]

slot = ['0-7.5', '7.5-10', '10-12.5', '12.5-15', '15 and above']

loan['int_rate_range'] = pd.cut(loan['int_rate'], bins, labels=slot)
def univariate(df,col,vartype,hue =None):

    

    '''

    Univariate function will plot the graphs based on the parameters.

    df      : dataframe name

    col     : Column name

    vartype : variable type : continuos or categorical

                Continuos(0)   : Distribution, Violin & Boxplot will be plotted.

                Categorical(1) : Countplot will be plotted.

    hue     : It's only applicable for categorical analysis.

    

    '''

    sns.set(style="darkgrid")

    

    if vartype == 0:

        fig, ax=plt.subplots(nrows =1,ncols=3,figsize=(20,8))

        ax[0].set_title("Distribution Plot")

        sns.distplot(df[col],ax=ax[0])

        ax[1].set_title("Violin Plot")

        sns.violinplot(data =df, x=col,ax=ax[1], inner="quartile")

        ax[2].set_title("Box Plot")

        sns.boxplot(data =df, x=col,ax=ax[2],orient='v')

    

    if vartype == 1:

        temp = pd.Series(data = hue)

        fig, ax = plt.subplots()

        width = len(df[col].unique()) + 6 + 4*len(temp.unique())

        fig.set_size_inches(width , 7)

        ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue) 

        if len(temp.unique()) > 0:

            for p in ax.patches:

                ax.annotate('{:1.1f}%'.format((p.get_height()*100)/float(len(loan))), (p.get_x()+0.05, p.get_height()+20))  

        else:

            for p in ax.patches:

                ax.annotate(p.get_height(), (p.get_x()+0.32, p.get_height()+20)) 

        del temp

    else:

        exit

        

    plt.show()
univariate(df=loan,col='loan_amnt',vartype=0)
univariate(df=loan,col='int_rate',vartype=0)
loan["annual_inc"].describe()
q = loan["annual_inc"].quantile(0.995)

loan = loan[loan["annual_inc"] < q]

loan["annual_inc"].describe()
univariate(df=loan,col='annual_inc',vartype=0)
univariate(df=loan,col='loan_status',vartype=1)
univariate(df=loan,col='purpose',vartype=1,hue='loan_status')
loan.home_ownership.unique()
# Remove rows where home_ownership'=='OTHER', 'NONE', 'ANY'

rem = ['OTHER', 'NONE', 'ANY']

loan.drop(loan[loan['home_ownership'].isin(rem)].index,inplace=True)

loan.home_ownership.unique()
univariate(df=loan,col='home_ownership',vartype=1,hue='loan_status')
year_wise =loan.groupby(by= [loan.issue_year])[['loan_status']].count()

year_wise.rename(columns={"loan_status": "count"},inplace=True)

ax =year_wise.plot(figsize=(20,8))

year_wise.plot(kind='bar',figsize=(20,8),ax = ax)

plt.show()
univariate(df=loan,col='term',vartype=1,hue='loan_status')
loan.head(3)
plt.figure(figsize=(16,12))

sns.boxplot(data =loan, x='purpose', y='loan_amnt', hue ='loan_status')

plt.title('Purpose of Loan vs Loan Amount')

plt.show()
loan_correlation = loan.corr()

loan_correlation
f, ax = plt.subplots(figsize=(14, 9))

sns.heatmap(loan_correlation, 

            xticklabels=loan_correlation.columns.values,

            yticklabels=loan_correlation.columns.values,annot= True)

plt.show()
loanstatus=loan.pivot_table(index=['loan_status','purpose','emp_length'],values='loan_amnt',aggfunc=('count')).reset_index()

loanstatus=loan.loc[loan['loan_status']=='Charged Off']
ax = plt.figure(figsize=(30, 18))

ax = sns.boxplot(x='emp_length',y='loan_amnt',hue='purpose',data=loanstatus)

ax.set_title('Employment Length vs Loan Amount for different pupose of Loan',fontsize=22,weight="bold")

ax.set_xlabel('Employment Length',fontsize=16)

ax.set_ylabel('Loan Amount',color = 'b',fontsize=16)

plt.show()
def crosstab(df,col):

    '''

    df : Dataframe

    col: Column Name

    '''

    crosstab = pd.crosstab(df[col], df['loan_status'],margins=True)

    crosstab['Probability_Charged Off'] = round((crosstab['Charged Off']/crosstab['All']),3)

    crosstab = crosstab[0:-1]

    return crosstab
# Probability of charge off

def bivariate_prob(df,col,stacked= True):

    '''

    df      : Dataframe

    col     : Column Name

    stacked : True(default) for Stacked Bar

    '''

    # get dataframe from crosstab function

    plotCrosstab = crosstab(df,col)

    

    linePlot = plotCrosstab[['Probability_Charged Off']]      

    barPlot =  plotCrosstab.iloc[:,0:2]

    ax = linePlot.plot(figsize=(20,8), marker='o',color = 'b')

    ax2 = barPlot.plot(kind='bar',ax = ax,rot=1,secondary_y=True,stacked=stacked)

    ax.set_title(df[col].name.title()+' vs Probability Charge Off',fontsize=20,weight="bold")

    ax.set_xlabel(df[col].name.title(),fontsize=14)

    ax.set_ylabel('Probability of Charged off',color = 'b',fontsize=14)

    ax2.set_ylabel('Number of Applicants',color = 'g',fontsize=14)

    plt.show()
filter_states = loan.addr_state.value_counts()

filter_states = filter_states[(filter_states < 10)]



loan_filter_states = loan.drop(labels = loan[loan.addr_state.isin(filter_states.index)].index)
states = crosstab(loan_filter_states,'addr_state')

display(states.tail(20))



bivariate_prob(df =loan_filter_states,col ='addr_state')
purpose = crosstab(loan,'purpose')

display(purpose)



bivariate_prob(df =loan,col ='purpose',stacked=False)
grade = crosstab(loan,'grade')

display(grade)



bivariate_prob(df =loan,col ='grade',stacked=False)

bivariate_prob(df =loan,col ='sub_grade')
annual_inc_range = crosstab(loan,'annual_inc_range')

display(annual_inc_range)



bivariate_prob(df =loan,col ='annual_inc_range')
int_rate_range = crosstab(loan,'int_rate_range')

display(int_rate_range)



bivariate_prob(df =loan,col ='int_rate_range')
emp_length = crosstab(loan,'emp_length')

display(emp_length)



bivariate_prob(df =loan,col ='emp_length')