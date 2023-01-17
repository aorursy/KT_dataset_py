# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import rc
from matplotlib.ticker import StrMethodFormatter
%matplotlib inline
from scipy import stats

# Regular expressions
import re

# seaborn : advanced visualization
import seaborn as sns
print('seaborn version\t:',sns.__version__)

df = pd.read_csv('../input/tvs-loan-default/TVS.csv')
df.head()
train_data_bkp = df.rename(columns={'V1': 'Customer ID',
'V2': 'Customer has bounced in first EMI', 
'V3': 'No of times bounced 12 months',
'V4': 'Maximum MOB',
'V5': 'No of times bounced while repaying the loan',
'V6': 'EMI',
'V7': 'Loan Amount',
'V8': 'Tenure',
'V9': 'Dealer codes from where customer has purchased the Two wheeler',
'V10': 'Product code of Two wheeler', 
'V11': 'No of advance EMI paid',
'V12': 'Rate of interest',
'V13': 'Gender',
'V14': 'Employment type',
'V15': 'Resident type of customer',
'V16': 'Date of birth',
'V17': 'Customer age when loanwas taken',
'V18': 'No of loans',
'V19': 'No of secured loans',
'V20': 'No of unsecured loans',
'V21': 'Max amount sanctioned in the Live loans',
'V22': 'No of new loans in last 3 months',
'V23': 'Total sanctioned amount in the secured Loans which are Live',
'V24': 'Total sanctioned amount in the unsecured Loans which are Live',
'V25': 'Maximum amount sanctioned for any Two wheeler loan',
'V26': 'Time since last Personal loan taken (in months)',
'V27': 'Time since first consumer durables loan taken (in months)',
'V28': 'No of times 30 days past due in last 6 months',
'V29': 'No of times 60 days past due in last 6 months',
'V30': 'No of times 90 days past due in last 3 months',
'V31': 'Tier',
'V32': 'Target variable'})

train_data_bkp.head()
import pandas as pd
from io import StringIO
from datetime import datetime

train_data_bkp["Date of birth"] = train_data_bkp["Date of birth"].fillna(datetime.today().strftime('%Y-%m-%d'))

train_data_bkp["Date of birth"] = pd.to_datetime(train_data_bkp["Date of birth"])

from dateutil.relativedelta import relativedelta

def f(end):
    r = relativedelta(pd.to_datetime('now'), end) 
    return '{}'.format(r.years)

train_data_bkp['Age'] = train_data_bkp["Date of birth"].apply(f).astype(int)
train_data_bkp['Age'] = train_data_bkp['Age'].replace(0, np.nan)
train_data_bkp = train_data_bkp.drop(['Customer ID','Date of birth','Time since last Personal loan taken (in months)','Time since first consumer durables loan taken (in months)','Total sanctioned amount in the secured Loans which are Live','Total sanctioned amount in the unsecured Loans which are Live','Max amount sanctioned in the Live loans','No of new loans in last 3 months'],axis=1)
train_data_bkp['Dealer codes from where customer has purchased the Two wheeler'] = train_data_bkp['Dealer codes from where customer has purchased the Two wheeler'].astype(str)
train_data = df.rename(columns={'V1': 'Customer ID',
'V2': 'Customer has bounced in first EMI', 
'V3': 'No of times bounced 12 months',
'V4': 'Maximum MOB',
'V5': 'No of times bounced while repaying the loan',
'V6': 'EMI',
'V7': 'Loan Amount',
'V8': 'Tenure',
'V9': 'Dealer codes from where customer has purchased the Two wheeler',
'V10': 'Product code of Two wheeler', 
'V11': 'No of advance EMI paid',
'V12': 'Rate of interest',
'V13': 'Gender',
'V14': 'Employment type',
'V15': 'Resident type of customer',
'V16': 'Date of birth',
'V17': 'Customer age when loanwas taken',
'V18': 'No of loans',
'V19': 'No of secured loans',
'V20': 'No of unsecured loans',
'V21': 'Max amount sanctioned in the Live loans',
'V22': 'No of new loans in last 3 months',
'V23': 'Total sanctioned amount in the secured Loans which are Live',
'V24': 'Total sanctioned amount in the unsecured Loans which are Live',
'V25': 'Maximum amount sanctioned for any Two wheeler loan',
'V26': 'Time since last Personal loan taken (in months)',
'V27': 'Time since first consumer durables loan taken (in months)',
'V28': 'No of times 30 days past due in last 6 months',
'V29': 'No of times 60 days past due in last 6 months',
'V30': 'No of times 90 days past due in last 3 months',
'V31': 'Tier',
'V32': 'Target variable'})

train_data.head()
target_nm = 'Target variable'
c = train_data[target_nm].value_counts(dropna=False)
p = train_data[target_nm].value_counts(dropna=False, normalize=True)
pd.concat([c,p], axis=1, keys=['counts', '%']).to_excel("Target_Variable_Distribution.xlsx", header=True)
print(pd.concat([c,p], axis=1, keys=['counts', '%']))
train_data[target_nm].value_counts().plot.bar()
sns.boxplot('Tier','Rate of interest',data=train_data)
plt.figure(figsize=(10,5))
sns.distplot(train_data['Rate of interest'])
plt.show()
sns.violinplot(target_nm,'Rate of interest',data=train_data,bw='scott')
train_data.hist(figsize=(15,20))
import pandas as pd
from io import StringIO
from datetime import datetime

train_data["Date of birth"] = train_data["Date of birth"].fillna(datetime.today().strftime('%Y-%m-%d'))

train_data["Date of birth"] = pd.to_datetime(train_data["Date of birth"])

from dateutil.relativedelta import relativedelta

def f(end):
    r = relativedelta(pd.to_datetime('now'), end) 
    return '{}'.format(r.years)

train_data['Age'] = train_data["Date of birth"].apply(f).astype(int)
train_data['Age'] = train_data['Age'].replace(0, np.nan)
train_data= train_data.drop(['Customer ID','Date of birth','Time since last Personal loan taken (in months)','Time since first consumer durables loan taken (in months)','Total sanctioned amount in the secured Loans which are Live','Total sanctioned amount in the unsecured Loans which are Live','Max amount sanctioned in the Live loans','No of new loans in last 3 months'],axis=1)
train_data['Dealer codes from where customer has purchased the Two wheeler'] = train_data['Dealer codes from where customer has purchased the Two wheeler'].astype(str)
train_data.describe(include='all').to_csv("Descriptive_Stats_Continuous.csv")
sns.heatmap(train_data.corr())
corr_matrix = train_data.corr()
print(corr_matrix['Target variable'].sort_values(ascending=False))
import numpy as np
df_cont = train_data.select_dtypes(include=[np.number])
df_cat = train_data.select_dtypes(exclude=[np.number]) 
df_cat['Target variable'] = train_data['Target variable']
print(df_cont.info())
print(df_cat.info())
################# Categorical field Weight of Evidence and IV Calculation ####################
'''This User Defined Function performs Binning of Categorical Variables and generate Weight of Evidence and Information Value of the same'''

'''
Input  :Dataframe having only Categorical Features and Target
        Target column name
        Output Excel filename
        
Output :Excel File With Binning information along with WOE and IV

Returns:Modified DataFrame with New Categorical columns with suffix "WOE" 
        having corresponding WOE Value
'''

import xlsxwriter
import os
import math
import pandas as pd
import numpy as np

def calc_iv(var_name):
    '''Calculates the Wieght of Evidence (WOE) and Information Value(IV) for Catgeorical fields'''
    global categorical_target_nx
    global data_fnl
    global IV_lst
    lst=[]
    
    categorical_target_nx.Event=categorical_target_nx.Event.fillna(0)
    categorical_target_nx['Non-Event']=categorical_target_nx['Non-Event'].fillna(0)
    row_cnt_without_total=len(categorical_target_nx)-int(1)
    
    for i in range(len(categorical_target_nx)):
        data_bin = categorical_target_nx.Levels[i]
        data_All = int(categorical_target_nx.Total[i])
        data_Target_1 = int(categorical_target_nx.Event[i])
        data_Target_0 = int(categorical_target_nx['Non-Event'][i])
        data_Target_1_Rate = categorical_target_nx.Event[i] / categorical_target_nx.Total[i]
        data_Target_0_Rate = categorical_target_nx['Non-Event'][i] / categorical_target_nx.Total[i]
        data_Distribution_Target_1 = int(categorical_target_nx['Event'][i])/ categorical_target_nx['Event'].head(row_cnt_without_total).sum().sum()
        data_Distribution_Target_0 = int(categorical_target_nx['Non-Event'][i])/categorical_target_nx['Non-Event'].head(row_cnt_without_total).sum()
        #'WOE' value by ln(Distribution Good/Distribution Bad)
        data_WoE = np.log(data_Distribution_Target_1 / data_Distribution_Target_0)
        
        data_vol_p   = categorical_target_nx["Volume(%)"][i]
        data_event_p = categorical_target_nx["Event(%)"][i]
        
        if (data_WoE == np.inf) or (data_WoE == -np.inf):
            data_WoE = 0
            
        data_IV = data_WoE * (data_Distribution_Target_1 - data_Distribution_Target_0)
        data=[data_bin,data_Target_1,data_Target_0,data_All,data_vol_p,data_event_p,
              data_Target_1_Rate,data_Target_0_Rate,data_Distribution_Target_1,data_Distribution_Target_0,
              data_WoE,data_IV]
        lst.append(data)
    
    data_fnl = pd.DataFrame(lst,columns=['Levels', 'Event', 'Non-Event', 'Total','Volume(%)','Event(%)',                                         'Event_Rate','Non_Event_Rate','Distribution_Event','Distribution_Non_Event',
                        'WOE','IV'])
    iv_val=[var_name,data_fnl['IV'].head(row_cnt_without_total).sum()]
    IV_lst.append(iv_val)

    
def cat_bin_trend(df_cat_fnl,df_categorical_column,i,target_col,filename):
    '''This User defined function creates the bins / groups on the 'Levels' of the Categorical Columns
    1. Event -> Target = 1
    2. Non-Event -> Target = 0 
    3. ALong with the Levels of the categorical Columns, A summary record is also created with header "Total" '''

    global categorical_target_nx

    categorical_target_nx = pd.DataFrame([df_cat_fnl[(df_cat_fnl[target_col] == 1)].groupby('Levels')[df_categorical_column[i]].count(),
                                         df_cat_fnl[(df_cat_fnl[target_col] == 0)].groupby('Levels')[df_categorical_column[i]].count(),
                                         df_cat_fnl.groupby('Levels')[df_categorical_column[i]].count()]).T

    categorical_target_nx.columns = ["Event","Non-Event","Total"]    
    categorical_target_nx['Event'] = categorical_target_nx['Event'].fillna(0)
    categorical_target_nx['Non-Event'] = categorical_target_nx['Non-Event'].fillna(0)
    categorical_target_nx['Total'] = categorical_target_nx['Total'].fillna(0)

    categorical_target_nx=categorical_target_nx.reset_index()
    categorical_target_nx = categorical_target_nx.rename(columns={categorical_target_nx.columns[0]: "Levels"})    
        
    list_vol_pct=[]
    list_event_pct=[]

    for j in range(len(categorical_target_nx.Event)):

        list_vol_pct.append(categorical_target_nx['Total'][j]/categorical_target_nx['Total'].sum())
        list_event_pct.append(categorical_target_nx['Event'][j]/categorical_target_nx['Total'][j])
    
    categorical_target_nx = pd.concat([categorical_target_nx,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)
    
    
    categorical_target_nx = categorical_target_nx[["Levels","Event","Non-Event","Total",0,1]]        
    categorical_target_nx = categorical_target_nx.rename(columns={categorical_target_nx.columns[len(categorical_target_nx.keys())-2]: "Volume(%)"})
    categorical_target_nx = categorical_target_nx.rename(columns={categorical_target_nx.columns[len(categorical_target_nx.keys())-1]: "Event(%)"})
    categorical_target_nx = categorical_target_nx.sort_values(by=['Total'], ascending=False)
    
    categorical_target_nx = categorical_target_nx.append(
        {"Levels":"Total",
         "Event":categorical_target_nx['Event'].sum(),
         "Non-Event":categorical_target_nx['Non-Event'].sum(),
         "Total":categorical_target_nx['Total'].sum(),
         "Volume(%)":categorical_target_nx['Volume(%)'].sum(),
         "Event(%)":categorical_target_nx['Event'].sum()/categorical_target_nx['Total'].sum()
        },ignore_index=True)
                 

def cat_bin(df,df_categorical,target_col,filename):
    ''' This User Defined Function performs the following :
    1. Replace the NAN value with "Missing" value 
    2. Binning is done on the unique Labels of the Categorical columns
    3. For Missing Values, it has been treated as seperate Label - "Missing"
    4. Calculates Weight of Evidence (WOE) of each bin/label of Categorical Variables
    5. Calculates Information Vale (IV) for each Categorical Variables  '''

    global categorical_target_nx,data_fnl,IV_lst
    
    df_categorical_column = list(df_categorical.columns)    
    
    '''Initialization of list and excel workbook'''
    IV_lst=[]
    writer1 = pd.ExcelWriter(filename,engine='xlsxwriter')
    workbook=writer1.book
    worksheet=workbook.add_worksheet('WOE')
    writer1.sheets['WOE'] = worksheet
    n = 0
    m = -1
    
    for i in range(len(df_categorical_column)):

        if (df_categorical_column[i] != target_col):       
            '''Repplacing the NAN Value with "Missing" Value for treating the Missing Value as seperate bin/group'''
            nparray_cat=df_categorical[df_categorical_column[i]].fillna('Missing').unique()
            nparray_sort=np.sort(nparray_cat)        
            df_cat = pd.concat([pd.Series(nparray_sort),pd.Series(nparray_sort)],axis=1, keys=[df_categorical_column[i],'Levels'])       
            df_tst = df.loc[:, [df_categorical_column[i],target_col]].sort_values(by=[df_categorical_column[i]]).fillna('Missing')            
            df_cat_fnl = pd.merge(df_tst, df_cat, how='left', on=[df_categorical_column[i]]) 
            
            ''' Creates Groups for each of the unique values of categorical variables '''
            cat_bin_trend(df_cat_fnl,df_categorical_column,i,target_col,filename)                                   

            ''' Calculates WOE and IV '''
            calc_iv(df_categorical_column[i])
     
            ''' Writing the WOE in seperate worksheet "WOE" of Final Excel '''
            worksheet.write_string(n, 0, df_categorical_column[i])
            data_fnl.to_excel(writer1,sheet_name='WOE',startrow=n+1 , startcol=0,index = False)
            n += len(categorical_target_nx.index) + 4
    
    ''' Writing the IV in seperate worksheet "IV" of Final Excel '''
    data_IV = pd.DataFrame(IV_lst,columns=['Variable','IV_value'])
    data_IV = data_IV.sort_values(by=['Variable','IV_value'],ascending=[True,False])
    data_IV.to_excel(writer1,sheet_name='IV',startrow=m+1 , startcol=0,index = False)       
    writer1.save()

        
def automate_woe_population(df,df_categorical_list,filename):
    '''This User Defined Function creates a new field with suffix "_WOE" and gets populated with Weight of Evidence
    as obtained for each 'Levels' of the Categorical Variables'''
    import pandas as pd
    woe_df=pd.read_excel(filename,sheet_name='WOE',header=None)
    woe_col_nbr = ''    

    for cat_i in df_categorical_list:
        match_fnd=""
        new_col=str(cat_i)+ "_WOE"

        for j in range(len(woe_df)):        
            if str(cat_i) == str(woe_df.iloc[j][0]) and match_fnd =="":            
                match_fnd='y'

            if str(woe_df.iloc[j][0]) == "Total":
                match_fnd = ""

            ''' Get the Column Number of WOE '''
            if (str(woe_df.iloc[j][0]) == "Levels") and woe_col_nbr == '' :
                for k in range(len(woe_df.columns)):
                    if str(woe_df.iloc[j][k]) == "WOE":
                        woe_col_nbr = int(k)                    
                                        
            if match_fnd == 'y':            
                if (str(woe_df.iloc[j][0]) != "Levels") :
                    
                    if (str(cat_i) == str(woe_df.iloc[j][0])):
                        woe_ln=str("df['") + str(cat_i) + str("_WOE']=0")
                        df[new_col]=0
                    else:

                        if  str(woe_df.iloc[j][0]) == "Missing":
                            woe_ln = str("df.loc[df['") + str(cat_i) + str("'].isna()")  + str(",'") + str(cat_i) + str("_WOE']=") + str(woe_df.iloc[j][woe_col_nbr])

                            df.loc[df[cat_i].isna(),new_col]=woe_df.iloc[j][woe_col_nbr]

                        else:
                            woe_ln = str("df.loc[df['") + str(cat_i) + str("']==") + str('"') + str(woe_df.iloc[j][0]) + str('"') + str(",'") + str(cat_i) + str("_WOE']=") + str(woe_df.iloc[j][woe_col_nbr])
                            print(woe_df.iloc[j][woe_col_nbr])
                            df.loc[df[cat_i]==str(woe_df.iloc[j][0]),new_col]=woe_df.iloc[j][woe_col_nbr]



                    print(woe_ln)

    return df    
cat_bin(train_data,df_cat,target_col=target_nm,filename="Categorical_WOE.xlsx")
train_data=automate_woe_population(train_data,df_cat,filename="Categorical_WOE.xlsx")
df_cat_list = df_cat.columns.to_list()
df_cat_list.remove(target_nm)
print(df_cat_list)
train_data=automate_woe_population(train_data,df_cat_list[:1],filename='../input/coarse-classing/Categorical_WOE_coarse_classing.xlsx')
train_data.columns
#################### Categorical field Descriptive statistics with PLOT  ####################
'''This User Defined Function performs Binning of Categorical Variables and generate Rank and Plot of the same'''

'''
Input  :Dataframe having only Categorical Features and Target
        Number of Bins to be created
        Target column name
        Output Excel filename
        
Output :Excel File With Binning information along with Line and Scatter Plot

Returns: None
'''

import xlsxwriter
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import numpy as np
import os
import math

''' Creating two new directory in the working directory to store the Plots of variables ''' 
work_dir=os.getcwd()
print(work_dir)
new_dir=work_dir+"/line_plot/"
check_dir_present=os.path.isdir(new_dir)
if check_dir_present == False:
    os.mkdir(new_dir)
    print("New Directory created : " + str(new_dir))
else:
    print("Existing Directory used : " + str(new_dir))
    
new_dir=work_dir+"/scatter_plot/"
check_dir_present=os.path.isdir(new_dir)
if check_dir_present == False:
    os.mkdir(new_dir)
    print("New Directory created : " + str(new_dir))
else:
    print("Existing Directory used : " + str(new_dir))

    
def plot_stat(df_categorical,title_name,target_col):
    '''This User Defined Functions performs following
    1. Line Plot showing Volume% and Event% against 'Levels' of the Categorical Variables
    2. Scatter Plot showing Target against 'Levels' of the Categorical Variables'''
    
    global df_plot,file_i,img_name,work_dir,scatter_img_name
    
    ''' Line Plot '''
    width=0.35
    df_plot['Volume(%)']=df_plot['Volume(%)']*100
    df_plot['Event(%)']=df_plot['Event(%)']*100

    df_plot.plot(x='Levels',y='Volume(%)',kind='bar',width=width,label=('Volume(%)'),color='b')
    y_pos=range(len(df_plot['Levels']))
    plt.ylabel('Volume(%)',color='b')
    plt.ylim((0,100))

    df_plot['Event(%)'].plot(secondary_y=True,label=('Event(%)'),color='r',rot=90)
    plt.ylabel('Event(%)',color='r')
    plt.ylim((0,100))
    plt.legend(bbox_to_anchor=(.965,.88),loc='upper right',ncol=2,borderaxespad=0)
    
    axis_1=plt.gca()

    for i,j in zip(df_plot['Volume(%)'].index,df_plot['Volume(%)']):
        i=round(i,2)
        j=round(j,2)
        axis_1.annotate('%s' %j,xy=(i,j),color='b')
        
        
    for i,j in zip(df_plot['Event(%)'].index,df_plot['Event(%)']):
        i=round(i,2)
        j=round(j,2)
        axis_1.annotate('%s' %j,xy=(i,j),color='r')
        

    plt.xlim([-width,len(df_plot['Volume(%)'])-width])
    plt.title(title_name)
    plt.xlabel('Levels')
    plt.grid()
    img_name=str(work_dir)+str("/line_plot/")+str(title_name)+ str(".png")
    plt.savefig(img_name,dpi=300,bbox_inches='tight')
    plt.clf()
    
    ''' Scatter Plot '''
    fig=plt.figure(figsize=(6.4,4.8))
    df_categorical[title_name]=df_categorical[title_name].fillna("Missing")
    df_cat_mod =  df_categorical.loc[df_categorical[title_name].isin(list(df_plot['Levels']))]
    plt.scatter(df_cat_mod[title_name],df_cat_mod[target_col],c='DarkBlue')                       
    plt.ylabel(target_col,color='b')
    plt.xlabel(title_name,color='b')
    plt.xticks(rotation=90)
    plt.title(title_name)
    plt.grid()
    scatter_img_name=str(work_dir)+str("/scatter_plot/")+str(title_name)+ str(".png")
    plt.show()
    fig.savefig(scatter_img_name,dpi=300,bbox_inches='tight')
    plt.clf()

def add_table_plot(df_categorical,in_file,sheet_nm,target_col,n_levels,out_file):
    ''' This User Defined Function adds Line Plot and Scatter Plot in an excel 
    just on the beside of the binning data of the categorical Fields.
    This gives better readability in analysing the data.
    '''
    
    global df_plot,img_name,work_dir,scatter_img_name
    
    work_dir=os.getcwd()
    df_base=pd.read_excel(in_file,header=None,sheet_name=sheet_nm)

    df_base.columns=['Levels', 'Event', 'Non-Event', 'Total','Volume(%)','Event(%)',
                     'Event_Rate','Non_Event_Rate','Distribution_Event','Distribution_Non_Event',
                     'WOE','IV']   

    df_base=df_base[['Levels', 'Event', 'Non-Event', 'Total','Volume(%)','Event(%)']]
    
    df_base=df_base.fillna('')

    wb=xlsxwriter.Workbook(out_file)
    ws=wb.add_worksheet('Rank_Plot')
    wrt_pos_i=0
    img_pos_i=0
    
    for i in range(len(df_base)):
        if wrt_pos_i == 0:
            wrt_pos_i = i
        else:
            wrt_pos_i = wrt_pos_i + 1        
        
        for j in range(len(df_base.columns)):
            col_pos = chr(ord('A') + j)
            wrt_pos=str(col_pos)+str(wrt_pos_i)
            ws.write(wrt_pos,df_base.iloc[i,j])
       
        if df_base.iloc[:,0][i] == "Levels":
            img_pos_i = wrt_pos_i
            pos_min_loc=i+int(1)
            if pos_min_loc==1:
                title_name=df_base.columns[0]
            else:
                title_loc=int(pos_min_loc)-int(2)
                title_name=df_base.iloc[:,0][title_loc]
        
        if df_base.iloc[:,0][i] == "Total":

            pos_max_loc=i

            if n_levels > 0 :  
                df_plot=df_base[pos_min_loc:pos_max_loc].head(n_levels)
            else:
                df_plot=df_base[pos_min_loc:pos_max_loc]

            df_plot.columns=['Levels','Event','Non-Event','TOTAL','Volume(%)','Event(%)']
            df_plot=df_plot.reset_index()

            ''' Calls plot_stat() to create line plot and scatter plot '''
            plot_stat(df_categorical,title_name,target_col)
            img_pos=str('H') + str(int(img_pos_i))   
            img2=mimg.imread(img_name)
            imgplot2=plt.imshow(img2)

            ''' Inset line plot in the excel '''
            ws.insert_image(img_pos,img_name,{'x_scale' : 0.6, 'y_scale' : 0.6})

            ''' Insert Scatter plot in the excel '''
            scatter_img_pos=str('N') + str(int(img_pos_i))   
            ws.insert_image(scatter_img_pos,scatter_img_name,{'x_scale' : 0.6, 'y_scale' : 0.6})
            
            ''' Provides spacing for image - 14 vertical cells requried for the image '''
            if len(df_plot) < 14 :
                wrt_pos_i = wrt_pos_i + 14 - len(df_plot)
    wb.close()
    
def cat_bin_trend(df_cat_fnl,df_categorical_column,i,target_col,filename):
    '''This User defined function creates the bins / groups on the 'Levels' of the Categorical Columns
    1. Event -> Target = 1
    2. Non-Event -> Target = 0 
    3. ALong with the Levels of the categorical Columns, A summary record is also created with header "Total" '''

    global categorical_target_nx

    categorical_target_nx = pd.DataFrame([df_cat_fnl[(df_cat_fnl[target_col] == 1)].groupby('Levels')[df_categorical_column[i]].count(),
                                         df_cat_fnl[(df_cat_fnl[target_col] == 0)].groupby('Levels')[df_categorical_column[i]].count(),
                                         df_cat_fnl.groupby('Levels')[df_categorical_column[i]].count()]).T

    categorical_target_nx.columns = ["Event","Non-Event","Total"]    
    categorical_target_nx['Event'] = categorical_target_nx['Event'].fillna(0)
    categorical_target_nx['Non-Event'] = categorical_target_nx['Non-Event'].fillna(0)
    categorical_target_nx['Total'] = categorical_target_nx['Total'].fillna(0)

    categorical_target_nx=categorical_target_nx.reset_index()
    categorical_target_nx = categorical_target_nx.rename(columns={categorical_target_nx.columns[0]: "Levels"})    
        
    list_vol_pct=[]
    list_event_pct=[]

    for j in range(len(categorical_target_nx.Event)):

        list_vol_pct.append(categorical_target_nx['Total'][j]/categorical_target_nx['Total'].sum())
        list_event_pct.append(categorical_target_nx['Event'][j]/categorical_target_nx['Total'][j])
    
    categorical_target_nx = pd.concat([categorical_target_nx,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)
    
    
    categorical_target_nx = categorical_target_nx[["Levels","Event","Non-Event","Total",0,1]]        
    categorical_target_nx = categorical_target_nx.rename(columns={categorical_target_nx.columns[len(categorical_target_nx.keys())-2]: "Volume(%)"})
    categorical_target_nx = categorical_target_nx.rename(columns={categorical_target_nx.columns[len(categorical_target_nx.keys())-1]: "Event(%)"})
    categorical_target_nx = categorical_target_nx.sort_values(by=['Total'], ascending=False)
    
    categorical_target_nx = categorical_target_nx.append(
        {"Levels":"Total",
         "Event":categorical_target_nx['Event'].sum(),
         "Non-Event":categorical_target_nx['Non-Event'].sum(),
         "Total":categorical_target_nx['Total'].sum(),
         "Volume(%)":categorical_target_nx['Volume(%)'].sum(),
         "Event(%)":categorical_target_nx['Event'].sum()/categorical_target_nx['Total'].sum()
        },ignore_index=True)
    print(categorical_target_nx)
                 

def cat_bin(df,df_categorical,target_col,filename):
    ''' This User Defined Function performs the following :
    1. Replace the NAN value with "Missing" value 
    2. Binning is done on the unique Labels of the Categorical columns
    3. For Missing Values, it has been treated as seperate Label - "Missing"
    4. Calculates Weight of Evidence (WOE) of each bin/label of Categorical Variables
    5. Calculates Information Vale (IV) for each Categorical Variables  '''

    global categorical_target_nx,data_fnl,IV_lst
    
    df_categorical_column = list(df_categorical.columns)    
    
    '''Initialization of list and excel workbook'''
    IV_lst=[]
    writer1 = pd.ExcelWriter(filename,engine='xlsxwriter')
    workbook=writer1.book
    worksheet=workbook.add_worksheet('WOE')
    writer1.sheets['WOE'] = worksheet
    n = 0
    m = -1
    
    for i in range(len(df_categorical_column)):

        if (df_categorical_column[i] != target_col):       
            '''Repplacing the NAN Value with "Missing" Value for treating the Missing Value as seperate bin/group'''
            nparray_cat=df_categorical[df_categorical_column[i]].fillna('Missing').unique()
            nparray_sort=np.sort(nparray_cat)        
            df_cat = pd.concat([pd.Series(nparray_sort),pd.Series(nparray_sort)],axis=1, keys=[df_categorical_column[i],'Levels'])       
            df_tst = df.loc[:, [df_categorical_column[i],target_col]].sort_values(by=[df_categorical_column[i]]).fillna('Missing')            
            df_cat_fnl = pd.merge(df_tst, df_cat, how='left', on=[df_categorical_column[i]]) 
            
            ''' Creates Groups for each of the unique values of categorical variables '''
            cat_bin_trend(df_cat_fnl,df_categorical_column,i,target_col,filename)                                   

            ''' Calculates WOE and IV '''
            calc_iv(df_categorical_column[i])
     
            ''' Writing the WOE in seperate worksheet "WOE" of Final Excel '''
            worksheet.write_string(n, 0, df_categorical_column[i])
            data_fnl.to_excel(writer1,sheet_name='WOE',startrow=n+1 , startcol=0,index = False)
            n += len(categorical_target_nx.index) + 4
    
    ''' Writing the IV in seperate worksheet "IV" of Final Excel '''
    data_IV = pd.DataFrame(IV_lst,columns=['Variable','IV_value'])
    data_IV = data_IV.sort_values(by=['Variable','IV_value'],ascending=[True,False])
    data_IV.to_excel(writer1,sheet_name='IV',startrow=m+1 , startcol=0,index = False)       
    writer1.save()    
cat_bin(train_data,df_cat,target_col=target_nm,filename="Categorical_Base.xlsx")       
add_table_plot(df_cat,in_file= "Categorical_Base.xlsx",sheet_nm='WOE',target_col=target_nm,n_levels=5,out_file="Categorical_Rank_Plot.xlsx")
################ Continuous Field  : Bin Based WOE and IV calculation ##########
'''This User Defined Function performs Binning of Continuous Variables and generate Weight of Evidence and Information Value of the same'''

'''
Input  :Dataframe having only Continuous Features
        Number of Bins to be created
        Target column name
        Output Excel filename
        
Output :Excel File With Binning information along with WOE and IV

Returns: None
'''

import xlsxwriter
import pandas as pd
import numpy as np
import os
import math



def create_volume_group(df_continuous,curr_var,target_col,n_bin):
    '''This User defined function creates bins on the basis of parameter n_bin (number of bins) provided
    This algorithm creates almost eqi_volume groups with unique values in groups : 
    1. It calculates the Average Bin Volume by dividing the total volume of data by number of bins
    2. It sorts the data based on the value of the continous variables
    3. It directly moves to index (I1) having value Average Bin Volume (ABV) 
    4. It checks the value of continous variable at the index position decided in previosu step
    5. It finds the index(I2) of last position of the value identified at previous step 4
    6. It concludes the data of the First Bin within the range (0 to I2)
    7. The Index I1 is again calculated as I1 = I2 + ABV and step 4-6 is repeated
    8. This Process is continued till the desired number of bins are created
    9. Seperate Bin is created if the continuous variable is having missing value
    
    Note : qcut() does provide equi-volume groups but does not provide unique values in the groups.
    hence,qcut() is not used in binning the data.
    '''
    
    global df_fnl
    df_continuous_column=[]
    df_continuous_column.append(curr_var)
    ttl_vol=len(df_continuous)
    avg_bin_vol = round(ttl_vol/n_bin)
    lst=[]
    df_fnl=pd.DataFrame(lst)
    df_mod=pd.DataFrame(lst)
    df_mod_null=pd.DataFrame(lst)
        
    for i in range(len(df_continuous_column)):
        
        if (df_continuous_column[i] != target_col):
            curr_var=df_continuous_column[i]

            df_mod1=pd.DataFrame(df_continuous[[curr_var,target_col]])

            #### Sort the Data ####
            df_mod1=df_mod1.sort_values(by=[curr_var],ascending=True)
            df_mod1=df_mod1.reset_index()
        
            df_mod_null=df_mod1[pd.isnull(df_mod1[curr_var])]
            df_mod1=df_mod1.dropna(subset=[curr_var]) 
        
            seq=list(range(1,len(df_mod1)+int(1)))
            df_seq=pd.DataFrame(seq,columns=['sequence'])

            df_mod1=pd.concat([df_mod1,df_seq],axis=1)
        
            '''Creating the Missing BIN'''
            if len(df_mod_null) > int(0):
                ttl_vol=len(df_mod1)
                avg_bin_vol = round(ttl_vol/n_bin)
                group_num='missing'

        
            pos_indx_max = 0
            val_list=df_mod1[curr_var].unique()
            
            '''Checks if the Unique Values of the continuos variable is 2,then simply create 2 bins'''       
            if len(val_list) == 2:
                for i in range(len(val_list)):
                    val_of_indx=val_list[i]
                    df_mod3=df_mod1[df_mod1[curr_var] == val_of_indx]
                    group_num=i+int(1)
                    df_mod3['Decile']=group_num
                    df_fnl=pd.concat([df_fnl,df_mod3])

            '''Checks if the Unique Values of the continuos variable is more than 2'''
            if len(val_list) != 2:        
                for bin_num in range(n_bin):
                    if  pos_indx_max < ttl_vol-1 :
                        '''For the First Group, index is assigned to length of average bin volume'''
                        if bin_num == 0:
                            indx = (bin_num+int(1)) * avg_bin_vol
                        else:
                            '''Next Groups:index = length of average bin volume 
                            plus the index of last group'''
                            indx = int(pos_indx_max) + int(avg_bin_vol)
                
                        if indx > ttl_vol:
                            '''Setting the Index for last Group'''
                            indx = ttl_vol-int(1)

                        val_of_indx = df_mod1[curr_var].iat[indx]
                                        
                        if math.isnan(val_of_indx) == True:
                            pos_indx_min=pos_indx_max+int(1)
                            pos_indx_max=ttl_vol-int(1)
                        else:
                            df_mod3=df_mod1[df_mod1[curr_var] == val_of_indx]
                            pos_indx_min = pos_indx_max
                            pos_indx_max = df_mod3['sequence'].iat[-1]
                       
                        df_mod3=df_mod1[pos_indx_min:pos_indx_max]
                        group_num=bin_num+int(1)
                        df_mod3['Decile']=group_num
                    
                        df_fnl=pd.concat([df_fnl,df_mod3])
                       
            df_fnl=pd.concat([df_fnl,df_mod_null])
            
def cont_bin_Miss(df_dcl_fnl,df_continuous_column,i,target_col,filename):
    '''This User Defined Function creates BINS for continous variables having some Missing values'''
    global continuous_target_nx

    '''The Missing Values have been grouped together as a seperate bin - "Missing" '''
    df_dcl_fnl[df_continuous_column[i]] = df_dcl_fnl[df_continuous_column[i]].astype(object)
    df_dcl_fnl[df_continuous_column[i]] = df_dcl_fnl[df_continuous_column[i]].fillna('Missing')
    df_dcl_fnl['Decile'] = df_dcl_fnl['Decile'].fillna('Missing')
    

    continuous_target_nx = pd.DataFrame([df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].min(),
                                                 df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].max(),
                                                 df_dcl_fnl[(df_dcl_fnl[target_col] == 1)].groupby('Decile')[df_continuous_column[i]].count(),
                                                 df_dcl_fnl[(df_dcl_fnl[target_col] == 0)].groupby('Decile')[df_continuous_column[i]].count(),
                                                 df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].count()]).T
    continuous_target_nx.columns = ["MIN","MAX","Event","Non-Event","TOTAL"]

    continuous_target_nx.Event=continuous_target_nx.Event.fillna(0)
    continuous_target_nx['Non-Event']=continuous_target_nx['Non-Event'].fillna(0)
    
    continuous_target_nx=continuous_target_nx.reset_index()
    list1=[]
    list_vol_pct=[]
    list_event_pct=[]
    for i in range(len(continuous_target_nx.MIN)):
        list1.append(str(continuous_target_nx['MIN'][i])+'-'+str(continuous_target_nx['MAX'][i]))
        list_vol_pct.append(continuous_target_nx['TOTAL'][i]/continuous_target_nx['TOTAL'].sum())
        list_event_pct.append(continuous_target_nx['Event'][i]/continuous_target_nx['TOTAL'][i])
    continuous_target_nx = pd.concat([pd.Series(list1),continuous_target_nx,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)
    continuous_target_nx=continuous_target_nx.reindex(columns=["Decile","MIN","MAX",0,"Event","Non-Event","TOTAL",1,2])
    continuous_target_nx = continuous_target_nx[["Decile","MIN","MAX",0,"Event","Non-Event","TOTAL",1,2]]

    #print(continuous_target_nx.head(10))
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-6]: "Range"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-2]: "Volume(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-1]: "Event(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-9]: "BINS"})       
    continuous_target_nx = continuous_target_nx.append({"BINS":"Total",
                                 "MIN":" ",
                                 "MAX":" ",
                                 "Range":" ",
                                 "Event":continuous_target_nx['Event'].sum(),
                                 "Non-Event":continuous_target_nx['Non-Event'].sum(),
                                 "TOTAL":continuous_target_nx['TOTAL'].sum(),
                                 "Volume(%)":continuous_target_nx['Volume(%)'].sum(),
                                 "Event(%)":continuous_target_nx['Event'].sum()/continuous_target_nx['TOTAL'].sum()
                                                       },ignore_index=True)
    
def cont_bin_NO_Miss(df_dcl_fnl,df_continuous_column,i,target_col,filename):
    '''This User defined function creates BINS for continuous variables having no missing values'''
    global continuous_target_nx

    continuous_target_nx = pd.DataFrame([df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].min(),
                                         df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].max(),
                                         df_dcl_fnl[(df_dcl_fnl[target_col] == 1)].groupby('Decile')[df_continuous_column[i]].count(),
                                         df_dcl_fnl[(df_dcl_fnl[target_col] == 0)].groupby('Decile')[df_continuous_column[i]].count(),
                                         df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].count()]).T
    continuous_target_nx.columns = ["MIN","MAX","Event","Non-Event","TOTAL"]

    continuous_target_nx=continuous_target_nx.reset_index()
    list1=[]
    list_vol_pct=[]
    list_event_pct=[]
    for i in range(len(continuous_target_nx.MIN)):
        list1.append(str(continuous_target_nx['MIN'][i])+'-'+str(continuous_target_nx['MAX'][i]))
        list_vol_pct.append(continuous_target_nx['TOTAL'][i]/continuous_target_nx['TOTAL'].sum())
        list_event_pct.append(continuous_target_nx['Event'][i]/continuous_target_nx['TOTAL'][i])
    continuous_target_nx = pd.concat([pd.Series(list1),continuous_target_nx,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)
    continuous_target_nx = continuous_target_nx[["Decile","MIN","MAX",0,"Event","Non-Event","TOTAL",1,2]]
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-6]: "Range"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-2]: "Volume(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-1]: "Event(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-9]: "BINS"})       
    continuous_target_nx = continuous_target_nx.append({"BINS":"Total",
                                 "MIN":" ",
                                 "MAX":" ",
                                 "Range":" ",
                                 "Event":continuous_target_nx['Event'].sum(),
                                 "Non-Event":continuous_target_nx['Non-Event'].sum(),
                                 "TOTAL":continuous_target_nx['TOTAL'].sum(),
                                 "Volume(%)":continuous_target_nx['Volume(%)'].sum(),
                                 "Event(%)":continuous_target_nx['Event'].sum()/continuous_target_nx['TOTAL'].sum()
                                                       },ignore_index=True)
    

def calc_iv(var_name):
    '''Calculates the Wieght of Evidence (WOE) and Information Value(IV) for Continuous fields'''
    global continuous_target_nx
    global data_fnl
    global IV_lst
    lst=[]
    
    continuous_target_nx.Event=continuous_target_nx.Event.fillna(0)
    continuous_target_nx['Non-Event']=continuous_target_nx['Non-Event'].fillna(0)
    row_cnt_without_total=len(continuous_target_nx)-int(1)
    
    for i in range(len(continuous_target_nx)):
        data_bin = continuous_target_nx.BINS[i]
        data_min = continuous_target_nx.MIN[i]
        data_max = continuous_target_nx.MAX[i]
        data_Value = continuous_target_nx.Range[i]
        data_Total = int(continuous_target_nx.TOTAL[i])
        data_Target_1 = int(continuous_target_nx.Event[i])
        data_Target_0 = int(continuous_target_nx['Non-Event'][i])
        data_Target_1_Rate = continuous_target_nx.Event[i] / continuous_target_nx.TOTAL[i]
        data_Target_0_Rate = continuous_target_nx['Non-Event'][i] / continuous_target_nx.TOTAL[i]
        data_Distribution_Target_1 = int(continuous_target_nx['Event'][i])/ continuous_target_nx['Event'].head(row_cnt_without_total).sum().sum()
        data_Distribution_Target_0 = int(continuous_target_nx['Non-Event'][i])/continuous_target_nx['Non-Event'].head(row_cnt_without_total).sum()
        
        data_vol_p = continuous_target_nx["Volume(%)"][i]
        data_event_p = continuous_target_nx["Event(%)"][i]
        
        #'WOE' value by ln(Distribution Good/Distribution Bad)
        data_WoE = np.log(data_Distribution_Target_1 / data_Distribution_Target_0)
        
        if (data_WoE == np.inf) or (data_WoE == -np.inf):
            data_WoE = 0
            
        data_IV = data_WoE * (data_Distribution_Target_1 - data_Distribution_Target_0)
        data=[data_bin,data_min,data_max,data_Value,data_Target_1,data_Target_0,data_Total,data_vol_p,data_event_p,
              data_Target_1_Rate,data_Target_0_Rate,data_Distribution_Target_1,data_Distribution_Target_0,
              data_WoE,data_IV]
        lst.append(data)
    
    
    data_fnl = pd.DataFrame(lst,columns=
                            ['Bins', 'Min' , 'Max' , 'Range', 'Event', 'Non-Event','Total','Volume(%)','Event(%)',
                             'Event_Rate','Non_Event_Rate','Distribution_Event','Distribution_Non_Event',
                             'WOE','IV'])
    iv_val=[var_name,data_fnl['IV'].head(row_cnt_without_total).sum()]
    IV_lst.append(iv_val)
    
def cont_bin(df_continuous,n_bin,target_col,filename):
    ''' This User Defined Function performs the following 
    1. Intiialize Excel Workbook
    2. Calls cont_bin_Miss() or cont_bin_NO_Miss() to perform binning based on the n_bin provided 
    3. Write to excel with the binning information'''
    global continuous_target_nx
    global data_fnl
    global IV_lst
    global df_fnl
    
    df_continuous_column = list(df_continuous.columns)
    
    '''Initialize Excel Work Book for writing'''
    IV_lst=[]
    writer1 = pd.ExcelWriter(filename,engine='xlsxwriter')
    workbook=writer1.book
    worksheet=workbook.add_worksheet('WOE')
    writer1.sheets['WOE'] = worksheet
    n = 1
    m = -1
    
    for i in range(len(df_continuous_column)):
        if (df_continuous_column[i] != target_col):  
            '''calls create_volume_group() to create bins - equal volume of bins with unique values '''
            '''n_bin - > it indicates the number of bins to be created'''
            
            create_volume_group(df_continuous,df_continuous_column[i],target_col,n_bin)
            
            df_fnl=df_fnl[[df_continuous_column[i],target_col,'Decile']]            

            ''' Checking the Continous Variable is having Missing Values '''
            if df_fnl[df_continuous_column[i]].isnull().sum() > 0:
                '''calls cont_bin_Miss() for the conitnous variable having missing value'''                                
                cont_bin_Miss(df_fnl,df_continuous_column,i,target_col,filename)                
            else:                
                '''calls cont_bin_NO_Miss() for the conitnous variable having NO missing value'''             
                cont_bin_NO_Miss(df_fnl,df_continuous_column,i,target_col,filename)
            
            calc_iv(df_continuous_column[i])

            '''Write to excel with the binning information'''
            worksheet.write_string(n, 0, df_continuous_column[i])
            data_fnl.to_excel(writer1,sheet_name='WOE',startrow=n+1 , startcol=0,index = False)
            n += len(continuous_target_nx.index) + 4
    
    data_IV=pd.DataFrame(IV_lst,columns=['Variable','IV_value'])
    data_IV=data_IV.sort_values(by=['Variable','IV_value'],ascending=[True,False])
    print(data_IV)
    data_IV.to_excel(writer1,sheet_name='IV',startrow=m+1 , startcol=0,index = False)       
    writer1.save() 
       
    
cont_bin(df_cont,n_bin=5,target_col=target_nm,filename="Continuous_WOE.xlsx")
################ Continuous Field  : Bin Based WOE and IV calculation ##########
'''This User Defined Function performs Binning of Continuous Variables and generate Weight of Evidence and Information Value of the same'''

'''
Input  :Dataframe having only Continuous Features
        Number of Bins to be created
        Target column name
        Output Excel filename
        
Output :Excel File With Binning information along with WOE and IV

Returns: None
'''

import xlsxwriter
import pandas as pd
import numpy as np
import os
import math



def create_volume_group(df_continuous,curr_var,target_col,n_bin):
    '''This User defined function creates bins on the basis of parameter n_bin (number of bins) provided
    This algorithm creates almost eqi_volume groups with unique values in groups : 
    1. It calculates the Average Bin Volume by dividing the total volume of data by number of bins
    2. It sorts the data based on the value of the continous variables
    3. It directly moves to index (I1) having value Average Bin Volume (ABV) 
    4. It checks the value of continous variable at the index position decided in previosu step
    5. It finds the index(I2) of last position of the value identified at previous step 4
    6. It concludes the data of the First Bin within the range (0 to I2)
    7. The Index I1 is again calculated as I1 = I2 + ABV and step 4-6 is repeated
    8. This Process is continued till the desired number of bins are created
    9. Seperate Bin is created if the continuous variable is having missing value
    
    Note : qcut() does provide equi-volume groups but does not provide unique values in the groups.
    hence,qcut() is not used in binning the data.
    '''
    
    global df_fnl
    df_continuous_column=[]
    df_continuous_column.append(curr_var)
    ttl_vol=len(df_continuous)
    avg_bin_vol = round(ttl_vol/n_bin)
    lst=[]
    df_fnl=pd.DataFrame(lst)
    df_mod=pd.DataFrame(lst)
    df_mod_null=pd.DataFrame(lst)

    for i in range(len(df_continuous_column)):
        if (df_continuous_column[i] != target_col):
            curr_var=df_continuous_column[i]

            df_mod1=pd.DataFrame(df_continuous[[curr_var,target_col]])

            #### Sort the Data ####
            df_mod1=df_mod1.sort_values(by=[curr_var],ascending=True)
            df_mod1=df_mod1.reset_index()
        
            df_mod_null=df_mod1[pd.isnull(df_mod1[curr_var])]
            df_mod1=df_mod1.dropna(subset=[curr_var]) 
        
            seq=list(range(1,len(df_mod1)+int(1)))
            df_seq=pd.DataFrame(seq,columns=['sequence'])

            df_mod1=pd.concat([df_mod1,df_seq],axis=1)
        
            '''Creating the Missing BIN'''
            if len(df_mod_null) > int(0):
                ttl_vol=len(df_mod1)
                avg_bin_vol = round(ttl_vol/n_bin)
                group_num='missing'

        
            pos_indx_max = 0
            val_list=df_mod1[curr_var].unique()
            
            '''Checks if the Unique Values of the continuos variable is 2,then simply create 2 bins'''       
            if len(val_list) == 2:
                for i in range(len(val_list)):
                    val_of_indx=val_list[i]
                    df_mod3=df_mod1[df_mod1[curr_var] == val_of_indx]
                    group_num=i+int(1)
                    df_mod3['Decile']=group_num
                    df_fnl=pd.concat([df_fnl,df_mod3])

            '''Checks if the Unique Values of the continuos variable is more than 2'''
            if len(val_list) != 2:        
                for bin_num in range(n_bin):
                    if  pos_indx_max < ttl_vol-1 :
                        '''For the First Group, index is assigned to length of average bin volume'''
                        if bin_num == 0:
                            indx = (bin_num+int(1)) * avg_bin_vol
                        else:
                            '''Next Groups:index = length of average bin volume 
                            plus the index of last group'''
                            indx = int(pos_indx_max) + int(avg_bin_vol)
                
                        if indx > ttl_vol:
                            '''Setting the Index for last Group'''
                            indx = ttl_vol-int(1)

                        val_of_indx = df_mod1[curr_var].iat[indx]
                                        
                        if math.isnan(val_of_indx) == True:
                            pos_indx_min=pos_indx_max+int(1)
                            pos_indx_max=ttl_vol-int(1)
                        else:
                            df_mod3=df_mod1[df_mod1[curr_var] == val_of_indx]
                            pos_indx_min = pos_indx_max
                            pos_indx_max = df_mod3['sequence'].iat[-1]
                       
                        df_mod3=df_mod1[pos_indx_min:pos_indx_max]
                        group_num=bin_num+int(1)
                        df_mod3['Decile']=group_num
                    
                        df_fnl=pd.concat([df_fnl,df_mod3])
            
            df_fnl=pd.concat([df_fnl,df_mod_null])
            
def cont_bin_Miss(df_dcl_fnl,df_continuous_column,i,target_col,filename):
    '''This User Defined Function creates BINS for continous variables having some Missing values'''
    global continuous_target_nx

    '''The Missing Values have been grouped together as a seperate bin - "Missing" '''
    df_dcl_fnl[df_continuous_column[i]] = df_dcl_fnl[df_continuous_column[i]].astype(object)
    df_dcl_fnl[df_continuous_column[i]] = df_dcl_fnl[df_continuous_column[i]].fillna('Missing')
    df_dcl_fnl['Decile'] = df_dcl_fnl['Decile'].fillna('Missing')
    

    continuous_target_nx = pd.DataFrame([df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].min(),
                                                 df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].max(),
                                                 df_dcl_fnl[(df_dcl_fnl[target_col] == 1)].groupby('Decile')[df_continuous_column[i]].count(),
                                                 df_dcl_fnl[(df_dcl_fnl[target_col] == 0)].groupby('Decile')[df_continuous_column[i]].count(),
                                                 df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].count()]).T
    continuous_target_nx.columns = ["MIN","MAX","Event","Non-Event","TOTAL"]

    continuous_target_nx.Event=continuous_target_nx.Event.fillna(0)
    continuous_target_nx['Non-Event']=continuous_target_nx['Non-Event'].fillna(0)
    
    continuous_target_nx=continuous_target_nx.reset_index()
    list1=[]
    list_vol_pct=[]
    list_event_pct=[]
    for i in range(len(continuous_target_nx.MIN)):
        list1.append(str(continuous_target_nx['MIN'][i])+'-'+str(continuous_target_nx['MAX'][i]))
        list_vol_pct.append(continuous_target_nx['TOTAL'][i]/continuous_target_nx['TOTAL'].sum())
        list_event_pct.append(continuous_target_nx['Event'][i]/continuous_target_nx['TOTAL'][i])
    continuous_target_nx = pd.concat([pd.Series(list1),continuous_target_nx,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)
    continuous_target_nx=continuous_target_nx.reindex(columns=["Decile","MIN","MAX",0,"Event","Non-Event","TOTAL",1,2])
    continuous_target_nx = continuous_target_nx[["Decile","MIN","MAX",0,"Event","Non-Event","TOTAL",1,2]]

    #print(continuous_target_nx.head(10))
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-6]: "Range"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-2]: "Volume(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-1]: "Event(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-9]: "BINS"})       
    continuous_target_nx = continuous_target_nx.append({"BINS":"Total",
                                 "MIN":" ",
                                 "MAX":" ",
                                 "Range":" ",
                                 "Event":continuous_target_nx['Event'].sum(),
                                 "Non-Event":continuous_target_nx['Non-Event'].sum(),
                                 "TOTAL":continuous_target_nx['TOTAL'].sum(),
                                 "Volume(%)":continuous_target_nx['Volume(%)'].sum(),
                                 "Event(%)":continuous_target_nx['Event'].sum()/continuous_target_nx['TOTAL'].sum()
                                                       },ignore_index=True)
    
def cont_bin_NO_Miss(df_dcl_fnl,df_continuous_column,i,target_col,filename):
    '''This User defined function creates BINS for continuous variables having no missing values'''
    global continuous_target_nx

    continuous_target_nx = pd.DataFrame([df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].min(),
                                         df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].max(),
                                         df_dcl_fnl[(df_dcl_fnl[target_col] == 1)].groupby('Decile')[df_continuous_column[i]].count(),
                                         df_dcl_fnl[(df_dcl_fnl[target_col] == 0)].groupby('Decile')[df_continuous_column[i]].count(),
                                         df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].count()]).T
    continuous_target_nx.columns = ["MIN","MAX","Event","Non-Event","TOTAL"]

    continuous_target_nx=continuous_target_nx.reset_index()
    list1=[]
    list_vol_pct=[]
    list_event_pct=[]
    for i in range(len(continuous_target_nx.MIN)):
        list1.append(str(continuous_target_nx['MIN'][i])+'-'+str(continuous_target_nx['MAX'][i]))
        list_vol_pct.append(continuous_target_nx['TOTAL'][i]/continuous_target_nx['TOTAL'].sum())
        list_event_pct.append(continuous_target_nx['Event'][i]/continuous_target_nx['TOTAL'][i])
    continuous_target_nx = pd.concat([pd.Series(list1),continuous_target_nx,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)
    continuous_target_nx = continuous_target_nx[["Decile","MIN","MAX",0,"Event","Non-Event","TOTAL",1,2]]
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-6]: "Range"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-2]: "Volume(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-1]: "Event(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-9]: "BINS"})       
    continuous_target_nx = continuous_target_nx.append({"BINS":"Total",
                                 "MIN":" ",
                                 "MAX":" ",
                                 "Range":" ",
                                 "Event":continuous_target_nx['Event'].sum(),
                                 "Non-Event":continuous_target_nx['Non-Event'].sum(),
                                 "TOTAL":continuous_target_nx['TOTAL'].sum(),
                                 "Volume(%)":continuous_target_nx['Volume(%)'].sum(),
                                 "Event(%)":continuous_target_nx['Event'].sum()/continuous_target_nx['TOTAL'].sum()
                                                       },ignore_index=True)
    

def calc_iv(var_name):
    '''Calculates the Wieght of Evidence (WOE) and Information Value(IV) for Continuous fields'''
    global continuous_target_nx
    global data_fnl
    global IV_lst
    lst=[]
    
    continuous_target_nx.Event=continuous_target_nx.Event.fillna(0)
    continuous_target_nx['Non-Event']=continuous_target_nx['Non-Event'].fillna(0)
    row_cnt_without_total=len(continuous_target_nx)-int(1)
    
    for i in range(len(continuous_target_nx)):
        data_bin = continuous_target_nx.BINS[i]
        data_min = continuous_target_nx.MIN[i]
        data_max = continuous_target_nx.MAX[i]
        data_Value = continuous_target_nx.Range[i]
        data_Total = int(continuous_target_nx.TOTAL[i])
        data_Target_1 = int(continuous_target_nx.Event[i])
        data_Target_0 = int(continuous_target_nx['Non-Event'][i])
        data_Target_1_Rate = continuous_target_nx.Event[i] / continuous_target_nx.TOTAL[i]
        data_Target_0_Rate = continuous_target_nx['Non-Event'][i] / continuous_target_nx.TOTAL[i]
        data_Distribution_Target_1 = int(continuous_target_nx['Event'][i])/ continuous_target_nx['Event'].head(row_cnt_without_total).sum().sum()
        data_Distribution_Target_0 = int(continuous_target_nx['Non-Event'][i])/continuous_target_nx['Non-Event'].head(row_cnt_without_total).sum()
        
        data_vol_p = continuous_target_nx["Volume(%)"][i]
        data_event_p = continuous_target_nx["Event(%)"][i]
        
        #'WOE' value by ln(Distribution Good/Distribution Bad)
        data_WoE = np.log(data_Distribution_Target_1 / data_Distribution_Target_0)
        
        if (data_WoE == np.inf) or (data_WoE == -np.inf):
            data_WoE = 0
            
        data_IV = data_WoE * (data_Distribution_Target_1 - data_Distribution_Target_0)
        data=[data_bin,data_min,data_max,data_Value,data_Target_1,data_Target_0,data_Total,data_vol_p,data_event_p,
              data_Target_1_Rate,data_Target_0_Rate,data_Distribution_Target_1,data_Distribution_Target_0,
              data_WoE,data_IV]
        lst.append(data)
    
    
    data_fnl = pd.DataFrame(lst,columns=
                            ['Bins', 'Min' , 'Max' , 'Range', 'Event', 'Non-Event','Total','Volume(%)','Event(%)',
                             'Event_Rate','Non_Event_Rate','Distribution_Event','Distribution_Non_Event',
                             'WOE','IV'])
    iv_val=[var_name,data_fnl['IV'].head(row_cnt_without_total).sum()]
    IV_lst.append(iv_val)
    
def cont_bin(df_continuous,n_bin,target_col,filename):
    ''' This User Defined Function performs the following 
    1. Intiialize Excel Workbook
    2. Calls cont_bin_Miss() or cont_bin_NO_Miss() to perform binning based on the n_bin provided 
    3. Write to excel with the binning information'''
    global continuous_target_nx
    global data_fnl
    global IV_lst
    global df_fnl
    
    df_continuous_column = list(df_continuous.columns)
    
    '''Initialize Excel Work Book for writing'''
    IV_lst=[]
    writer1 = pd.ExcelWriter(filename,engine='xlsxwriter')
    workbook=writer1.book
    worksheet=workbook.add_worksheet('Continous')
    writer1.sheets['Continous'] = worksheet
    n = 1
    m = -1
    
    for i in range(len(df_continuous_column)):
        if (df_continuous_column[i] != target_col):  
            '''calls create_volume_group() to create bins - equal volume of bins with unique values '''
            '''n_bin - > it indicates the number of bins to be created'''
            create_volume_group(df_continuous,df_continuous_column[i],target_col,n_bin)
            df_fnl=df_fnl[[df_continuous_column[i],target_col,'Decile']]

            ''' Checking the Continous Variable is having Missing Values '''
            if df_fnl[df_continuous_column[i]].isnull().sum() > 0:
                '''calls cont_bin_Miss() for the conitnous variable having missing value'''                
                cont_bin_Miss(df_fnl,df_continuous_column,i,target_col,filename)                
            else:
                '''calls cont_bin_NO_Miss() for the conitnous variable having NO missing value'''             
                cont_bin_NO_Miss(df_fnl,df_continuous_column,i,target_col,filename)
            
            calc_iv(df_continuous_column[i])

            '''Write to excel with the binning information'''
            worksheet.write_string(n, 0, df_continuous_column[i])
            data_fnl.to_excel(writer1,sheet_name='Continous',startrow=n+1 , startcol=0,index = False)
            n += len(continuous_target_nx.index) + 4
    
    data_IV=pd.DataFrame(IV_lst,columns=['Variable','IV_value'])
    data_IV=data_IV.sort_values(by=['Variable','IV_value'],ascending=[True,False])
    print(data_IV)
    data_IV.to_excel(writer1,sheet_name='IV',startrow=m+1 , startcol=0,index = False)       
    writer1.save()      

    '''This User Defined Function performs Rank and Plot of Continuous Variables'''

'''
Input  :Excel File With Binning information
        
Output :Excel File With Binning information along with Line and Scatter Plot

Returns: None
'''

import xlsxwriter
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import numpy as np
import os
import math
import pandas as pd
import numpy as np

''' Creating two new directory in the working directory to store the Plots of variables ''' 
work_dir=os.getcwd()
print(work_dir)
new_dir=work_dir+"/line_plot/"
check_dir_present=os.path.isdir(new_dir)
if check_dir_present == False:
    os.mkdir(new_dir)
    print("New Directory created : " + str(new_dir))
else:
    print("Existing Directory used : " + str(new_dir))
    
new_dir=work_dir+"/scatter_plot/"
check_dir_present=os.path.isdir(new_dir)
if check_dir_present == False:
    os.mkdir(new_dir)
    print("New Directory created : " + str(new_dir))
else:
    print("Existing Directory used : " + str(new_dir))

def plot_stat(df_continuous,title_name,target_col):
    '''This User Defined Functions performs following
    1. Line Plot showing Volume% and Event% against 'BINS' of the Continuous Variables
    2. Scatter Plot showing Target against 'BINS' of the Continuous Variables'''
    
    global df_plot,file_i,img_name,work_dir,scatter_img_name
    
    ''' Line Plot '''
    
    width=0.35
    df_plot['Volume(%)']=df_plot['Volume(%)']*100
    df_plot['Event(%)']=df_plot['Event(%)']*100

    df_plot.plot(x='Bins',y='Volume(%)',kind='bar',width=width,label=('Volume(%)'),color='b')
    plt.ylabel('Volume(%)',color='b')
    plt.ylim((0,100))
    
    df_plot['Event(%)'].plot(secondary_y=True,label=('Event(%)'),color='r')
    plt.ylabel('Event(%)',color='r')
    plt.ylim((0,100))
    plt.legend(bbox_to_anchor=(.965,.88),loc='upper right',ncol=2,borderaxespad=0)
    
    
    axis_1=plt.gca()

    for i,j in zip(df_plot['Volume(%)'].index,df_plot['Volume(%)']):
        i=round(i,1)
        j=round(j,1)
        axis_1.annotate('%s' %j,xy=(i,j),color='b')
        
    for i,j in zip(df_plot['Event(%)'].index,df_plot['Event(%)']):
        i=round(i,1)
        j=round(j,1)
        axis_1.annotate('%s' %j,xy=(i,j),color='r')

    plt.xlim([-width,len(df_plot['Volume(%)'])-width])
    plt.title(title_name)
    plt.xlabel("BINS")
    plt.grid()
    img_name=str(work_dir)+str("/line_plot/")+str(title_name)+ str(".png")
    plt.savefig(img_name,dpi=300,bbox_inches='tight')
    plt.clf()
    
    ''' Scatter Plot '''
    fig=plt.figure(figsize=(6.4,4.8))
    plt.scatter(df_continuous[title_name],df_continuous[target_col],c='DarkBlue')
    plt.ylabel(target_col,color='b')
    plt.xlabel(title_name,color='b')
    plt.title(title_name)
    plt.grid()
    scatter_img_name=str(work_dir)+str("/scatter_plot/")+str(title_name)+ str(".png")
    #plt.show()
    fig.savefig(scatter_img_name,dpi=300,bbox_inches='tight')
    plt.clf()

def add_table_plot(df_continuous,in_file,sheet_nm,out_file,target_col):
    ''' This User Defined Function adds Line Plot and Scatter Plot in an excel 
    just on the beside of the binning data of the categorical Fields.
    This gives better readability in analysing the data.
    '''
    
    global df_plot,img_name,work_dir,scatter_img_name
    
    work_dir=os.getcwd()
    df_cont=pd.read_excel(in_file,sheet_name=sheet_nm,header=None)
    
    df_cont.columns=['Bins', 'Min' , 'Max' , 'Range', 'Event', 'Non-Event','Total','Volume(%)','Event(%)',
                             'Event_Rate','Non_Event_Rate','Distribution_Event','Distribution_Non_Event',
                             'WOE','IV' ]   

    df_cont=df_cont[['Bins','Min','Max','Range','Event','Non-Event','Total','Volume(%)','Event(%)']]
    
    df_cont=df_cont.fillna('')
    wb=xlsxwriter.Workbook(out_file)
    ws=wb.add_worksheet('Rank_Plot')
    wrt_pos_i=0
    img_pos_i=0    
    
    for i in range(len(df_cont)):
        if wrt_pos_i == 0:
            wrt_pos_i = i
        else:
            wrt_pos_i = wrt_pos_i + 1        
        
        for j in range(len(df_cont.columns)):
            col_pos = chr(ord('A') + j)
            wrt_pos=str(col_pos)+str(wrt_pos_i)
            ws.write(wrt_pos,df_cont.iloc[i,j])

        if df_cont.iloc[:,0][i] == "Bins":
            img_pos_i = wrt_pos_i
            pos_min_loc=i+int(1)
            if pos_min_loc==1:
                title_name=df_cont.columns[0]
                
            else:
                title_loc=int(pos_min_loc)-int(2)
                title_name=df_cont.iloc[:,0][title_loc]                
        
        if df_cont.iloc[:,0][i] == "Total":
            pos_max_loc=i
            df_plot=df_cont[pos_min_loc:pos_max_loc]
            df_plot.columns=['Bins','Min','Max','Range','Event','Non-Event','Total','Volume(%)','Event(%)']
            df_plot=df_plot.reset_index()
            
            ''' Calls plot_stat() to create line plot and scatter plot '''
            
            plot_stat(df_continuous,title_name,target_col)
            img_pos=str('K')+ str(int(img_pos_i)) 
            img2=mimg.imread(img_name)
            imgplot2=plt.imshow(img2)

            ''' Insert line plot in the excel '''
            ws.insert_image(img_pos,img_name,{'x_scale' : 0.6, 'y_scale' : 0.6})
            
            ''' Insert scatter plot in the excel '''
            scatter_img_pos=str('Q')+ str(int(img_pos_i))
            ws.insert_image(scatter_img_pos,scatter_img_name,{'x_scale' : 0.6, 'y_scale' : 0.6})
            
            ''' Provides spacing for image - 14 vertical cells requried for the image '''
            if len(df_plot) < 14 :
                wrt_pos_i = wrt_pos_i + 14 - len(df_plot)
           
            
    wb.close()
    
cont_bin(df_cont,n_bin=5,target_col=target_nm,filename="Continuous_Base.xlsx") 
add_table_plot(df_cont,in_file="Continuous_Base.xlsx",sheet_nm='Continous',target_col=target_nm,out_file="Continuous_Rank_Plot.xlsx")
def automate_woe_population(df,df_continuous_list,filename):
    '''This User Defined Function creates a new field with suffix "_WOE" and gets populated with Weight of Evidence
    as obtained for each 'Levels' of the Categorical Variables'''
    import pandas as pd
    woe_df=pd.read_excel(filename,header=0,sheet_name='WOE')
    woe_col_nbr = ''    

    for cat_i in df_continuous_list:
        match_fnd=""
        new_col=str(cat_i)+ "_WOE"
        print(new_col)        

        for j in range(len(woe_df)):        
            if str(cat_i) == str(woe_df.iloc[j][0]) and match_fnd =="":            
                match_fnd='y'

            if str(woe_df.iloc[j][0]) == "Total":
                match_fnd = ""

            ''' Get the Column Number of WOE '''
            if (str(woe_df.iloc[j][0]) == "Bins") and woe_col_nbr == '' :
                for k in range(len(woe_df.columns)):
                    if str(woe_df.iloc[j][k]) == "WOE":
                        woe_col_nbr = int(k)                    
                                        
            if match_fnd == 'y':            
                if (str(woe_df.iloc[j][0]) != "Levels") :
                    
                    if (str(cat_i) == str(woe_df.iloc[j][0])):
                        woe_ln=str("df['") + str(cat_i) + str("_WOE']=0")
                        df[new_col]=0
                    else:

                        if  str(woe_df.iloc[j][0]) == "Missing":
                            woe_ln = str("df.loc[df['") + str(cat_i) + str("'].isna()")  + str(",'") + str(cat_i) + str("_WOE']=") + str(woe_df.iloc[j][woe_col_nbr])

                            df.loc[df[cat_i].isna(),new_col]=woe_df.iloc[j][woe_col_nbr]
                            df[new_col] = df[new_col].astype(float)                        
                        
                        elif str(woe_df.iloc[j][0]) == "Bins":
                            continue
                            
                        else:
                            woe_ln = str("df.loc[df['") + str(cat_i) + str("']>=") + str(woe_df.iloc[j][1]) + str(' & ') + str("df.loc[df['") + str(cat_i) + str("']<=") + str(woe_df.iloc[j][2]) + str(",'") + str(cat_i) + str("_WOE']=") + str(woe_df.iloc[j][woe_col_nbr])                           
                            df.loc[(df[cat_i] >= float(woe_df.iloc[j][1])) & (df[cat_i] <= float(woe_df.iloc[j][2])),new_col]=woe_df.iloc[j][woe_col_nbr]
                            df[new_col] = df[new_col].astype(float)



                    print(woe_ln)

    return df
#train_data=automate_woe_population(train_data,df_cont,filename="Continuous_Base.xlsx")
df_cont_list = df_cont.columns.to_list()
df_cont_list.remove(target_nm)
print(df_cont_list)
train_data=automate_woe_population(train_data,df_cont_list,filename='../input/coarse-classing/Continuous_WOE_coarse_classing.xlsx')
train_data.columns
slct_cont_cat_woe = ['Dealer codes from where customer has purchased the Two wheeler_WOE','Product code of Two wheeler_WOE','Gender_WOE','Employment type_WOE','Resident type of customer_WOE','Tier_WOE','Customer has bounced in first EMI_WOE','No of times bounced 12 months_WOE','Maximum MOB_WOE','No of times bounced while repaying the loan_WOE','EMI_WOE','Loan Amount_WOE','Tenure_WOE','No of advance EMI paid_WOE','Rate of interest_WOE','Customer age when loanwas taken_WOE','No of loans_WOE','No of secured loans_WOE','No of unsecured loans_WOE','Maximum amount sanctioned for any Two wheeler loan_WOE','No of times 30 days past due in last 6 months_WOE','No of times 60 days past due in last 6 months_WOE','No of times 90 days past due in last 3 months_WOE','Age_WOE','Target variable']
train_data_dummy = train_data[slct_cont_cat_woe]
train_data_dummy['Dealer codes from where customer has purchased the Two wheeler_WOE'].unique()
Dealer_code_dummies = pd.get_dummies(train_data_dummy['Dealer codes from where customer has purchased the Two wheeler_WOE'], drop_first=True, prefix='Dealer_code')
Product_code_dummies = pd.get_dummies(train_data_dummy['Product code of Two wheeler_WOE'], drop_first=True, prefix='Product_code')
Gender_dummies = pd.get_dummies(train_data_dummy['Gender_WOE'], drop_first=True, prefix='Gender')
Employment_type_dummies = pd.get_dummies(train_data_dummy['Employment type_WOE'], drop_first=True, prefix='Employment_type')
Res_type_cust_dummies = pd.get_dummies(train_data_dummy['Resident type of customer_WOE'], drop_first=True, prefix='Res_type_cust')
Tier_dummies = pd.get_dummies(train_data_dummy['Tier_WOE'], drop_first=True, prefix='Tier')
Cust_bounced_first_EMI_dummies = pd.get_dummies(train_data_dummy['Customer has bounced in first EMI_WOE'], drop_first=True, prefix='Cust_bounced_first')
N_times_bounced_12m_dummies = pd.get_dummies(train_data_dummy['No of times bounced 12 months_WOE'], drop_first=True, prefix='N_times_bounced_12m')
Maximum_MOB_dummies = pd.get_dummies(train_data_dummy['Maximum MOB_WOE'], drop_first=True, prefix='Maximum_MOB')
N_time_bounced_dummies = pd.get_dummies(train_data_dummy['No of times bounced while repaying the loan_WOE'], drop_first=True, prefix='N_time_bounced')
EMI_dummies = pd.get_dummies(train_data_dummy['EMI_WOE'], drop_first=True, prefix='EMI')
Loan_Amount_dummies = pd.get_dummies(train_data_dummy['Loan Amount_WOE'], drop_first=True, prefix='Loan Amount')
Tenure_dummies = pd.get_dummies(train_data_dummy['Tenure_WOE'], drop_first=True, prefix='Tenure')
No_of_adv_EMI_dummies = pd.get_dummies(train_data_dummy['No of advance EMI paid_WOE'], drop_first=True, prefix='No_of_adv_EMI')
Rate_of_interest_dummies = pd.get_dummies(train_data_dummy['Rate of interest_WOE'], drop_first=True, prefix='Rate_of_interest')
Customer_age_ln_taken_dummies = pd.get_dummies(train_data_dummy['Customer age when loanwas taken_WOE'], drop_first=True, prefix='Customer_age_ln_taken')
No_of_loans_dummies = pd.get_dummies(train_data_dummy['No of loans_WOE'], drop_first=True, prefix='No_of_loans')
No_of_secured_loans_dummies = pd.get_dummies(train_data_dummy['No of secured loans_WOE'], drop_first=True, prefix='No of secured loans')
No_of_unsecured_loans_dummies = pd.get_dummies(train_data_dummy['No of unsecured loans_WOE'], drop_first=True, prefix='No_of_unsecured_loans')
Min_amt_sanc_dummies = pd.get_dummies(train_data_dummy['Maximum amount sanctioned for any Two wheeler loan_WOE'], drop_first=True, prefix='Min_amt_sanc')
N_time_30dpd_last6m_dummies = pd.get_dummies(train_data_dummy['No of times 30 days past due in last 6 months_WOE'], drop_first=True, prefix='N_time_30dpd_last6m')
N_time_60dpd_last6m_dummies = pd.get_dummies(train_data_dummy['No of times 60 days past due in last 6 months_WOE'], drop_first=True, prefix='N_time_60dpd_last6m')
N_time_90dpd_last3m_dummies = pd.get_dummies(train_data_dummy['No of times 90 days past due in last 3 months_WOE'], drop_first=True, prefix='N_time_90dpd_last3m')
Age_dummies = pd.get_dummies(train_data_dummy['Age_WOE'], drop_first=True, prefix='Age')

train_data_dummy_final = pd.concat([Dealer_code_dummies ,Product_code_dummies ,Gender_dummies ,Employment_type_dummies ,Res_type_cust_dummies ,Tier_dummies ,Cust_bounced_first_EMI_dummies ,N_times_bounced_12m_dummies ,Maximum_MOB_dummies ,N_time_bounced_dummies ,EMI_dummies ,Loan_Amount_dummies ,Tenure_dummies ,No_of_adv_EMI_dummies ,Rate_of_interest_dummies ,Customer_age_ln_taken_dummies ,No_of_loans_dummies ,No_of_secured_loans_dummies ,No_of_unsecured_loans_dummies ,Min_amt_sanc_dummies ,N_time_30dpd_last6m_dummies ,N_time_60dpd_last6m_dummies ,N_time_90dpd_last3m_dummies ,Age_dummies ,
        train_data_dummy[['Target variable']]], axis=1)
train_data_dummy_final.columns
train_data = pd.concat([train_data,train_data_dummy_final], axis=1)
print(train_data.shape)
print(train_data.columns)
plt.figure(figsize = (20,10))
sns.set(font_scale = 1.25)
sns.heatmap(train_data_dummy_final[train_data_dummy_final.columns[1:]].corr(),annot = True, fmt = ".1f", 
           cmap = (sns.cubehelix_palette(20, start = 0.5, rot = -0.75)))
plt.show()
plt.savefig("Correlation_features_Heatmap.png",dpi=300,bbox_inches='tight')
sns.set(style="darkgrid")

fig = plt.figure(figsize=(13,6))
plt.subplot(121)

train_data["Gender"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["gold","b"],startangle = 60,
                                                                        wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.05,0],shadow =True)
plt.title("distribution of client owning a car")
plt.show()
sns.set(style="darkgrid")

fig, ax=plt.subplots(nrows =1,ncols=3,figsize=(25,8))
ax[0].set_title("Loan Amount (Distribution Plot)")
sns.distplot(train_data['Loan Amount'],ax=ax[0])
ax[1].set_title("Loan Amount (Violin Plot)")
sns.violinplot(data =train_data, x='Loan Amount',ax=ax[1], inner="quartile")
ax[2].set_title("Loan Amount (Box Plot)")
sns.boxplot(data =train_data, x='Loan Amount',ax=ax[2],orient='v')
plt.show()
sns.set(style="darkgrid")

fig, ax=plt.subplots(nrows =1,ncols=3,figsize=(25,8))
ax[0].set_title("Loan Amount (Distribution Plot)")
sns.distplot(train_data['Rate of interest'],ax=ax[0])
ax[1].set_title("Loan Amount (Violin Plot)")
sns.violinplot(data =train_data, x='Rate of interest',ax=ax[1], inner="quartile")
ax[2].set_title("Loan Amount (Box Plot)")
sns.boxplot(data =train_data, x='Rate of interest',ax=ax[2],orient='v')
plt.show()
plt.figure(figsize=(18,6))

#bar plot
train_data['Customer age when loanwas taken'].value_counts().plot(kind='bar',color='b',alpha=0.7, edgecolor='black')
plt.xlabel("Age", labelpad=14)
plt.ylabel("Count of People", labelpad=14)
plt.title(" Age of Customer when the loan was approved")
plt.legend(loc="best",prop={"size":12})
plt.show()
plt.figure(figsize=(18,6))

#histogarm
sns.distplot(train_data["Customer age when loanwas taken"],color="b")
plt.xlabel('Age')
plt.show()
plt.figure(figsize=(18,6))

#bar plot
train_data['Age'].value_counts().plot(kind='bar',color='b',alpha=0.7, edgecolor='black')
plt.xlabel("Age", labelpad=14)
plt.ylabel("Count of People", labelpad=14)
plt.title("Customer Age as of today")
plt.legend(loc="best",prop={"size":12})
plt.show()
plt.figure(figsize=(18,6))

#histogarm
sns.distplot(train_data["Age"],color="b")
plt.xlabel('Age')
plt.show()
plt.figure(figsize=(8,8))
train_data["Product code of Two wheeler"].value_counts().plot.pie(autopct = "%1.1f%%",fontsize=8,
                                                             colors = sns.color_palette("prism",5),
                                              wedgeprops={"linewidth":2,"edgecolor":"white"},
                                                               shadow =True)
plt.title("Two Wheeler Type")
#MC : Motorcycle , MO : Moped, SC : Scooter
plt.show()
plt.figure(figsize=(24,8))
sns.countplot(x="No of times bounced while repaying the loan", hue="Employment type", data=train_data, edgecolor='k')
plt.xlim(0, 6)
plt.ylim(0, 16000)
plt.legend(loc='upper right')
plt.show()
train_data.columns
df_missing=0
def missing_values_table(df_phase3):
    mis_val = df_phase3.isnull().sum()
    mis_val_percent = 100*df_phase3.isnull().sum()/len(df_phase3)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis = 'columns')
    mis_val_table_ren_columns = mis_val_table.rename(columns ={0: 'Count of Missing Values', 1: '% of Total Values'} )

    #Sort the table by percent of missing values descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0 ].sort_values(by='% of Total Values', ascending = False).round(1)
    mis_val_table_ren_columns = mis_val_table_ren_columns.reset_index()
    #print summary
    #print("The dataset has " + str(df_phase3.shape[0]) + " rows and " + str(df_phase3.shape[1]) + " columns." )
    #print(str(mis_val_table_ren_columns.shape[0]) )

    return mis_val_table_ren_columns;
    
df_missing=missing_values_table(train_data)
print(df_missing)
df_dummy_missing=missing_values_table(train_data_dummy_final)
print(df_dummy_missing)
pd.DataFrame(missing_values_table(train_data_bkp)).to_excel("Missing_statistics.xlsx")
train_data_bkp['No of times bounced while repaying the loan'] = train_data_bkp['No of times bounced while repaying the loan'].replace(np.nan,1)
train_data_bkp['Customer age when loanwas taken'] = train_data_bkp['Customer age when loanwas taken'].replace(np.nan,32)
train_data_bkp['Age'] = train_data_bkp['Age'].replace(np.nan,36)
train_data_bkp['Maximum MOB'] = train_data_bkp['Maximum MOB'].replace(np.nan,14)
train_data_bkp['Rate of interest'] = train_data_bkp['Rate of interest'].replace(np.nan,13.01)
train_data_bkp['No of advance EMI paid'] = train_data_bkp['No of advance EMI paid'].replace(np.nan, 2)
train_data_bkp['Tenure'] = train_data_bkp['Tenure'].replace(np.nan,25)
train_data_bkp['Loan Amount'] = train_data_bkp['Loan Amount'].replace(np.nan, 25012)
train_data_bkp['EMI'] = train_data_bkp['EMI'].replace(np.nan, 2158)
train_data_bkp['Maximum amount sanctioned for any Two wheeler loan'] = train_data_bkp['Maximum amount sanctioned for any Two wheeler loan'].replace(np.nan, 33204)

train_data['No of times bounced while repaying the loan_MSB'] = train_data_bkp['No of times bounced while repaying the loan'] 
train_data['Customer age when loanwas taken_MSB'] = train_data_bkp['Customer age when loanwas taken'] 
train_data['Age_MSB'] = train_data_bkp['Age'] 
train_data['Maximum MOB_MSB'] = train_data_bkp['Maximum MOB'] 
train_data['Rate of interest_MSB'] = train_data_bkp['Rate of interest'] 
train_data['No of advance EMI paid_MSB'] = train_data_bkp['No of advance EMI paid'] 
train_data['Tenure_MSB'] = train_data_bkp['Tenure'] 
train_data['Loan Amount_MSB'] = train_data_bkp['Loan Amount'] 
train_data['EMI_MSB'] = train_data_bkp['EMI'] 
train_data['Maximum amount sanctioned for any Two wheeler loan_MSB'] = train_data_bkp['Maximum amount sanctioned for any Two wheeler loan'] 
#Replace Null Values (np.nan) with mean
train_data['EMI'] = train_data['EMI'].replace(np.nan, train_data['EMI'].mean())
train_data['Loan Amount'] = train_data['Loan Amount'].replace(np.nan, train_data['Loan Amount'].mean())
train_data['Maximum amount sanctioned for any Two wheeler loan'] = train_data['Maximum amount sanctioned for any Two wheeler loan'].replace(np.nan, train_data['Maximum amount sanctioned for any Two wheeler loan'].mean())

#Checking for null values in Age column
print(train_data['EMI'].isnull().sum())
print(train_data['Loan Amount'].isnull().sum())
print(train_data['Maximum amount sanctioned for any Two wheeler loan'].isnull().sum())
#In the same way we can impute using median and mode
train_data['Maximum MOB'] = train_data['Maximum MOB'].replace(np.nan, train_data['Maximum MOB'].median())
train_data['No of times bounced while repaying the loan'] = train_data['No of times bounced while repaying the loan'].replace(np.nan, train_data['No of times bounced while repaying the loan'].median())
train_data['Tenure'] = train_data['Tenure'].replace(np.nan, train_data['Tenure'].median())
train_data['No of advance EMI paid'] = train_data['No of advance EMI paid'].replace(np.nan, train_data['No of advance EMI paid'].median())
train_data['Rate of interest'] = train_data['Rate of interest'].replace(np.nan, train_data['Rate of interest'].median())
train_data['Customer age when loanwas taken'] = train_data['Customer age when loanwas taken'].replace(np.nan, train_data['Customer age when loanwas taken'].median())
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mode())
#Checking for null values in Age column
print(train_data['Maximum MOB'].isnull().sum())
print(train_data['No of times bounced while repaying the loan'].isnull().sum())
print(train_data['Tenure'].isnull().sum())
print(train_data['No of advance EMI paid'].isnull().sum())
print(train_data['Rate of interest'].isnull().sum())
print(train_data['Customer age when loanwas taken'].isnull().sum())
slct_cat_woe = ['Customer has bounced in first EMI','No of times bounced 12 months','Maximum MOB','No of times bounced while repaying the loan','EMI','Loan Amount','Tenure','No of advance EMI paid','Rate of interest','Customer age when loanwas taken','No of loans','No of secured loans','No of unsecured loans','Maximum amount sanctioned for any Two wheeler loan','No of times 30 days past due in last 6 months','No of times 60 days past due in last 6 months','No of times 90 days past due in last 3 months','Age','Dealer codes from where customer has purchased the Two wheeler_WOE','Product code of Two wheeler_WOE','Gender_WOE','Employment type_WOE','Resident type of customer_WOE','Tier_WOE']
slct_cat_cont_woe = ['Dealer codes from where customer has purchased the Two wheeler_WOE','Product code of Two wheeler_WOE','Gender_WOE','Employment type_WOE','Resident type of customer_WOE','Tier_WOE','Customer has bounced in first EMI_WOE','No of times bounced 12 months_WOE','Maximum MOB_WOE','No of times bounced while repaying the loan_WOE','EMI_WOE','Loan Amount_WOE','Tenure_WOE','No of advance EMI paid_WOE','Rate of interest_WOE','Customer age when loanwas taken_WOE','No of loans_WOE','No of secured loans_WOE','No of unsecured loans_WOE','Maximum amount sanctioned for any Two wheeler loan_WOE','No of times 30 days past due in last 6 months_WOE','No of times 60 days past due in last 6 months_WOE','No of times 90 days past due in last 3 months_WOE','Age_WOE']
slct_cont_msb_cat_woe = ['Age_MSB','Customer age when loanwas taken_MSB','Customer has bounced in first EMI','Dealer codes from where customer has purchased the Two wheeler_WOE','EMI_MSB','Employment type_WOE','Gender_WOE','Loan Amount_MSB','Maximum amount sanctioned for any Two wheeler loan_MSB','Maximum MOB_MSB','No of loans','No of times 60 days past due in last 6 months','No of times 90 days past due in last 3 months','No of times bounced 12 months','No of times bounced while repaying the loan_MSB','No of unsecured loans','No of secured loans','No of times 30 days past due in last 6 months','No of advance EMI paid_MSB','Product code of Two wheeler_WOE','Rate of interest_MSB','Resident type of customer_WOE','Tenure_MSB','Tier_WOE']
slct_woe_dummy = ['Dealer_code_0.0','Dealer_code_0.9559735155414971','Dealer_code_1.5370520479052074','Dealer_code_2.688962596900705','Product_code_-0.1421213491381836','Product_code_-0.05826982136746174','Product_code_0.0','Product_code_0.2804872039936926','Gender_0.03696311356576965','Employment_type_-0.7765767471804874','Employment_type_-0.5012532942327598','Employment_type_-0.1998256728216269','Employment_type_0.08504806076174212','Res_type_cust_0.02044958160637848','Res_type_cust_0.162970846857935','Tier_-0.1321380510081557','Tier_-0.04634936386258279','Tier_0.5574790752604555','Cust_bounced_first_0.4048291007267008','N_times_bounced_12m_0.3834542680840288','N_times_bounced_12m_0.9716050326130725','Maximum_MOB_-0.007371935931145213','Maximum_MOB_0.006369669776595649','Maximum_MOB_0.02113918192071991','Maximum_MOB_0.1901809851536456','N_time_bounced_0.03673379522856265','N_time_bounced_0.6774751994813883','EMI_-0.04318109198821392','EMI_-0.02880476453071618','EMI_-0.0007890225289003329','EMI_0.09630845948690948','Loan Amount_-0.1395277728465895','Loan Amount_0.06770667473690144','Loan Amount_0.07420114820122564','Loan Amount_0.1275794007605065','Tenure_-0.04662229017942052','Tenure_0.135602511064546','No_of_adv_EMI_-0.009504717159535709','No_of_adv_EMI_0.09060713127320336','Rate_of_interest_-0.07496472192459401','Rate_of_interest_-0.05066882693021101','Rate_of_interest_0.08283613016745002','Rate_of_interest_0.08362052533588261','Customer_age_ln_taken_-0.2321446559895609','Customer_age_ln_taken_-0.1199394823627936','Customer_age_ln_taken_0.150577032084122','Customer_age_ln_taken_0.3537724451758127','No_of_loans_-0.06929027459199408','No_of_loans_0.01329016868114212','No_of_loans_0.06903902702373141','No of secured loans_0.02712016444232146','No of secured loans_0.03335534453738747','No_of_unsecured_loans_-0.001442499592240792','No_of_unsecured_loans_0.1528126652237951','Min_amt_sanc_-0.1132248423693573','Min_amt_sanc_0.04771310010720148','Min_amt_sanc_0.08600392248638013','Min_amt_sanc_0.1195470256915129','N_time_30dpd_last6m_1.288083810799486','N_time_60dpd_last6m_1.328746050835782','N_time_90dpd_last3m_1.322251040827673','Age_-0.2299749873451578','Age_-0.1059038923019072','Age_0.1542023151831859','Age_0.3474719468108172']

print(len(slct_cat_woe))
print(len(slct_cat_cont_woe))
print(len(slct_cont_msb_cat_woe))
print(len(slct_woe_dummy))

y = train_data['Target variable']
y = y.loc[:,~y.columns.duplicated()]
X = train_data[slct_woe_dummy]

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["features"] = X.columns
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif.to_csv("VIF_dummy_variable.csv")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 100)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
max_depth = [10]
max_feature = ['auto']
criterion=["entropy", "gini"]

param = {'max_depth':max_depth, 
         'max_features':max_feature, 
         'criterion': criterion}
grid = GridSearchCV(DecisionTreeClassifier(), 
                                param_grid = param, 
                                verbose=False, 
                                cv=StratifiedKFold(n_splits=20, random_state=15, shuffle=True),
                                n_jobs = -1)
grid.fit(X_train,y_train)
print( grid.best_params_)
print (grid.best_score_)
print (grid.best_estimator_)
dectree_grid = rf_grid = grid.best_estimator_
dectree_grid.score(X_train,y_train)
## feature importance
feature_importances = pd.DataFrame(dectree_grid.feature_importances_,
                                   index = X.columns,
                                    columns=['importance'])
pd.DataFrame(feature_importances.sort_values(by='importance', ascending=False)).to_excel("Decision_tree_feature_importance.xlsx")
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=10,min_samples_split=50,min_samples_leaf=10,max_features=10,max_leaf_nodes=30,random_state=100)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score
# printing confision matrix
print('Confusion Matrix:',pd.DataFrame(confusion_matrix(y_train,y_pred),\
            columns=["Predicted Not-Delinquent", "Predicted Delinquent"],\
            index=["Not-Delinquent","Delinquent"] ))
print('Accuracy Score:', accuracy_score(y_train, y_pred))
print('Recall Score:', recall_score(y_train, y_pred))
print('Precission Score:', precision_score(y_train, y_pred))
print('Classofication Report:',classification_report(y_train, y_pred))
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
n_estimators = [200];
max_depth = [10];
criterions = ['gini', 'entropy'];
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

parameters = {'n_estimators':n_estimators,
              'max_depth':max_depth,
              'criterion': criterions
              
        }
grid = GridSearchCV(estimator=RandomForestClassifier(max_features='auto'),
                                 param_grid=parameters,
                                 cv=cv,
                                 n_jobs = -1)
grid.fit(X_train,y_train)
print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)
rf_grid = grid.best_estimator_
rf_grid.score(X_train,y_train)
## feature importance
feature_importances = pd.DataFrame(rf_grid.feature_importances_,
                                   index = X.columns,
                                    columns=['importance'])
pd.DataFrame(feature_importances.sort_values(by='importance', ascending=False)).to_excel("Random_forest_feature_importance.xlsx")
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=200,max_depth=10,min_samples_split=50,min_samples_leaf=10,max_features=10,max_leaf_nodes=30, bootstrap=True
                             ,n_jobs=-1,random_state=100)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score
# printing confusion matrix
print('Confusion Matrix:',pd.DataFrame(confusion_matrix(y_train,y_pred),\
            columns=["Predicted Not-Delinquent", "Predicted Delinquent"],\
            index=["Not-Delinquent","Delinquent"] ))
print('Accuracy Score:', accuracy_score(y_train, y_pred))
print('Recall Score:', recall_score(y_train, y_pred))
print('Precission Score:', precision_score(y_train, y_pred))
print('Classofication Report:',classification_report(y_train, y_pred))
y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score
# printing confision matrix
print('Confusion Matrix:',pd.DataFrame(confusion_matrix(y_test,y_pred),
            columns=["Predicted Not-Delinquent", "Predicted Delinquent"],
            index=["Not-Delinquent","Delinquent"] ))
print('Accuracy Score:', accuracy_score(y_test, y_pred))
print('Recall Score:', recall_score(y_test, y_pred))
print('Precission Score:', precision_score(y_test, y_pred))
print('Classofication Report:',classification_report(y_test, y_pred))

from sklearn import metrics

y_pred = clf.predict(X_train)
y_train_score = clf.predict_proba(X_train)
random_forest_model_accuracy = metrics.accuracy_score(y_train,y_pred)
                
print("====== Classification Metrics - Development ======")
print(" Accuracy : "  + str(metrics.accuracy_score(y_train,y_pred)))
print(" Recall : "  + str(metrics.recall_score(y_train,y_pred)))
print(" Precision : "  + str(metrics.precision_score(y_train,y_pred)))
print(" F1_Score : "  + str(metrics.f1_score(y_train,y_pred)))
print(" Confusion_metrics : "  + str(metrics.confusion_matrix(y_train,y_pred)))
print(" ")

y_train_score_df = pd.DataFrame(y_train_score, index=range(y_train_score.shape[0]),columns=range(y_train_score.shape[1]))
y_train_score_df['Actual'] = pd.Series(y_train.values[:,0], index=y_train_score_df.index)
y_train_score_df['Predicted'] = pd.Series(y_pred, index=y_train_score_df.index)
y_train_score_df['Decile'] = pd.qcut(y_train_score_df[1],10,duplicates='drop')

lift_tbl = pd.DataFrame([y_train_score_df.groupby('Decile')[1].min(),
                                                 y_train_score_df.groupby('Decile')[1].max(),
                                                 y_train_score_df[(y_train_score_df['Actual'] == 1)].groupby('Decile')[1].count(),
                                                 y_train_score_df[(y_train_score_df['Actual'] == 0)].groupby('Decile')[1].count(),
                                                 y_train_score_df.groupby('Decile')[1].count()]).T
lift_tbl.columns = ["MIN","MAX","Event","Non-Event","TOTAL"]
lift_tbl = lift_tbl.sort_values("MIN", ascending=False)
lift_tbl = lift_tbl.reset_index()

list_vol_pct=[]
list_event_pct=[]

for i in range(len(lift_tbl.Event)):
    list_vol_pct.append(lift_tbl['TOTAL'][i]/lift_tbl['TOTAL'].sum())
    list_event_pct.append(lift_tbl['Event'][i]/lift_tbl['TOTAL'][i])

lift_tbl = pd.concat([lift_tbl,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)


lift_tbl = lift_tbl[["Decile","MIN","MAX","Event","Non-Event","TOTAL",0,1]]        
lift_tbl = lift_tbl.rename(columns={lift_tbl.columns[len(lift_tbl.keys())-2]: "Volume(%)"})
lift_tbl = lift_tbl.rename(columns={lift_tbl.columns[len(lift_tbl.keys())-1]: "Event(%)"})

lift_tbl["Cumm_Event"] = lift_tbl["Event"].cumsum()
lift_tbl["Cumm_Event_Pct"] = lift_tbl["Cumm_Event"] / lift_tbl["Event"].sum()
#lift_tbl
lift_tbl.to_excel("TVS_Loan_data_Lift_Chart_optimized_rf_20August2020.xlsx", index = None, header=True)
lift_tbl
# logistic regression ------
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', C=0.9, solver='saga', n_jobs=-1)
lr.fit(X_train, y_train)
print(lr.coef_)
print(lr.intercept_)

# predicted proability
train_pred = lr.predict_proba(X_train)[:,1]
test_pred = lr.predict_proba(X_test)[:,1]
!pip install git+git://github.com/shichenxie/scorecardpy.git
import scorecardpy as sc
train, test = sc.split_df(train_data_bkp,target_nm).values()
bins = sc.woebin(train_data_bkp, y= target_nm)
sc.woebin_plot(bins)

#Writing BINS statitics on csv file
tst_df = pd.DataFrame()
for x in bins:
    tmp_df = pd.DataFrame()
    for y in bins[x]:
        tmp_df = pd.concat([tmp_df,pd.DataFrame(bins[x][y])], axis=1)
    #print(tmp_df)            
    tst_df = tst_df.append(tmp_df)
    tst_df.reindex(tst_df.index.tolist()).reset_index()
    
tst_df.to_csv("WOE_ALL_by_package_scoremodel.csv", header=1)
train_woe = sc.woebin_ply(train, bins)
test_woe = sc.woebin_ply(test, bins)
y_train = train_woe.loc[:,target_nm]
X_train = train_woe.loc[:,train_woe.columns != target_nm]
y_test = test_woe.loc[:,target_nm]
X_test = test_woe.loc[:,train_woe.columns != target_nm]
# logistic regression ------
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', C=0.9, solver='saga', n_jobs=-1)
lr.fit(X_train, y_train)
print(lr.coef_)
print(lr.intercept_)

# predicted proability of LR
train_pred = lr.predict_proba(X_train)[:,1]
test_pred = lr.predict_proba(X_test)[:,1]
# performance ks & roc ------
train_perf = sc.perf_eva(y_train, train_pred, title = "train")
test_perf = sc.perf_eva(y_test, test_pred, title = "test")
print('Measures of Model Discrimination for Train data: ',train_perf)
print('Measures of Model Discrimination for Test data: ',test_perf)
card = sc.scorecard(bins, lr, X_train.columns)

# credit score
train_score = sc.scorecard_ply(train, card, print_step=0)
test_score = sc.scorecard_ply(test, card, print_step=0)
import matplotlib.pyplot as plt
import matplotlib.image as mimg

# credit score
train_score = sc.scorecard_ply(train, card, print_step=0)
test_score = sc.scorecard_ply(test, card, print_step=0)

# psi
psi = sc.perf_psi(
  score = {'train':train_score, 'test':test_score},
  label = {'train':y_train, 'test':y_test}
)
print(psi)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
log =LogisticRegression()
log.fit(X_train,y_train)

#model on train using all the independent values in df
y_pred = log.predict(X_train)
log_score= accuracy_score(y_train,y_pred)
print('Accuracy score on train set using Logistic Regression :',log_score)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
# printing confision matrix
print('Confusion Matrix:',pd.DataFrame(confusion_matrix(y_train,y_pred),
            columns=["Predicted Not-Delinquent", "Predicted Delinquent"],
            index=["Not-Delinquent","Delinquent"] ))
print('Recall Score:', recall_score(y_train, y_pred))
print('Precission Score:', precision_score(y_train, y_pred))
print('F1_sccore on train set :',f1_score(y_train, y_pred))
print('Classofication Report:',classification_report(y_train, y_pred))
## call on the model object
logreg = LogisticRegression(solver='liblinear',
                            penalty= 'l1',random_state = 100                           
                            )
## fit the model with "train_x" and "train_y"
logreg.fit(X_train,y_train)
## Once the model is trained we want to find out how well the model is performing, so we test the model. 
## we use "X_test" portion of the data(this data was not used to fit the model) to predict model outcome. 
y_pred = logreg.predict(X_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
# printing confision matrix
print('Confusion Matrix:',pd.DataFrame(confusion_matrix(y_train,y_pred),
            columns=["Predicted Not-Delinquent", "Predicted Delinquent"],
            index=["Not-Delinquent","Delinquent"] ))
print('Accuracy Score:', accuracy_score(y_train, y_pred))
print('Recall Score:', recall_score(y_train, y_pred))
print('Precission Score:', precision_score(y_train, y_pred))
print('F1_sccore on train set :',f1_score(y_train, y_pred))
print('Classofication Report:',classification_report(y_train, y_pred))
y_pred = logreg.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score
# printing confision matrix
print('Confusion Matrix:',pd.DataFrame(confusion_matrix(y_test,y_pred),
            columns=["Predicted Not-Delinquent", "Predicted Delinquent"],
            index=["Not-Delinquent","Delinquent"] ))
print('Accuracy Score:', accuracy_score(y_test, y_pred))
print('Recall Score:', recall_score(y_test, y_pred))
print('Precission Score:', precision_score(y_test, y_pred))
print('Classofication Report:',classification_report(y_test, y_pred))
from sklearn.metrics import roc_curve, auc
#plt.style.use('seaborn-pastel')
y_score = logreg.decision_function(X_test)

FPR, TPR, _ = roc_curve(y_test, y_score)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

plt.figure(figsize =[11,9])
plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC for Delinquent Customer', fontsize= 18)
plt.show()
from sklearn.metrics import precision_recall_curve

y_score = logreg.decision_function(X_test)

precision, recall, _ = precision_recall_curve(y_test, y_score)
PR_AUC = auc(recall, precision)

plt.figure(figsize=[11,9])
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision Recall Curve for Titanic survivors', fontsize=18)
plt.legend(loc="lower right")
plt.show()
from sklearn.model_selection import GridSearchCV, StratifiedKFold
## C_vals is the alpla value of lasso and ridge regression(as alpha increases the model complexity decreases,)
## remember effective alpha scores are 0<alpha<infinity 
C_vals = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,16.5,17,17.5,18]
## Choosing penalties(Lasso(l1) or Ridge(l2))
penalties = ['l1','l2']
## Choose a cross validation strategy. 
cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25)

## setting param for param_grid in GridSearchCV. 
param = {'penalty': penalties, 'C': C_vals}

logreg = LogisticRegression(solver='liblinear')
## Calling on GridSearchCV object. 
grid = GridSearchCV(estimator=LogisticRegression(), 
                           param_grid = param,
                           scoring = 'accuracy',
                            n_jobs =-1,
                           cv = cv
                          )
## Fitting the model
grid.fit(X_train,y_train)
## Getting the best of everything. 
print (grid.best_score_)
print (grid.best_params_)
print(grid.best_estimator_)
### Using the best parameters from the grid-search.
logreg_grid = grid.best_estimator_
logreg_grid.score(X_test,y_test)
from sklearn.linear_model import Ridge,RidgeCV,ElasticNet,Lasso,LassoCV,LassoLarsCV,LassoLarsIC
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import pandas as pd
# LassoLinear
clf=Lasso(alpha=0.00001,normalize=True)
clf.fit(X_train,y_train)
clf.score(X_train,y_train)
#clf.coef_
coef=pd.Series(clf.coef_,index=X_train.columns)
print("Lasso linear")
print(coef)
import numpy as np

#Number of tress in Random Forest
n_estimators = [int(x) for x in np.linspace(start = 200,stop = 1000, num = 5)]

#Number of features to consider while splitting
max_features = ['auto', 'sqrt']

#Maximum number of levels in the tree
max_depth = [int(x) for x in np.linspace(start = 7, stop = 10, num = 4)]

#Minimum # of samples required to split the node
min_samples_split = [10,15]

#Minimum # of samples required at each leaf node
min_samples_leaf = [3,5]

#Method of selecting samples from training each tree
bootstrap = [True, False]

from sklearn.metrics import accuracy_score,make_scorer,precision_score,recall_score,roc_auc_score,f1_score
scoring={'AUC' : make_scorer(roc_auc_score) , 
         'Accuracy':make_scorer(accuracy_score), 
         'Recall':make_scorer(recall_score), 
         'Precision':make_scorer(precision_score),
         'F1 Score':make_scorer(f1_score)}

#Create the random grid:
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf ,
               'bootstrap': bootstrap                
              }
from sklearn.ensemble import RandomForestClassifier
RFcl = RandomForestClassifier(random_state = 0, n_jobs = -1) 

from sklearn.model_selection import RandomizedSearchCV
CV_rfc = RandomizedSearchCV(estimator=RFcl, param_distributions =random_grid, n_jobs = -1, cv= 10,scoring=scoring,refit='Recall',return_train_score=True,n_iter=10)
CV_rfc.fit(X_train, y_train)
print(CV_rfc.cv_results_)
def format_grid_search_result(res):
    global df_gs_result
    gs_results=res
    
    gs_model=gs_results['params']
    
    # Grid Search : AUC Metrics
    gs_mean_test_AUC=pd.Series(gs_results['mean_test_AUC'])
    gs_std_test_AUC=pd.Series(gs_results['std_test_AUC'])
    gs_rank_test_AUC=pd.Series(gs_results['rank_test_AUC'])
    
    # Grid Search : Accuracy Metrics
    gs_mean_test_Accuracy=pd.Series(gs_results['mean_test_Accuracy'])
    gs_std_test_Accuracy=pd.Series(gs_results['std_test_Accuracy'])
    gs_rank_test_Accuracy=pd.Series(gs_results['rank_test_Accuracy'])
    
    # Grid Search : Recall Metrics
    gs_mean_test_Recall=pd.Series(gs_results['mean_test_Recall'])
    gs_std_test_Recall=pd.Series(gs_results['std_test_Recall'])
    gs_rank_test_Recall=pd.Series(gs_results['rank_test_Recall'])

    # Grid Search : Precision Metrics
    gs_mean_test_Precision=pd.Series(gs_results['mean_test_Precision'])
    gs_std_test_Precision=pd.Series(gs_results['std_test_Precision'])
    gs_rank_test_Precision=pd.Series(gs_results['rank_test_Precision'])
    
    # Grid Search : F1-Score Metrics
    gs_mean_test_F1_Score=pd.Series(gs_results['mean_test_F1 Score'])
    gs_std_test_F1_Score=pd.Series(gs_results['std_test_F1 Score'])
    gs_rank_test_F1_Score=pd.Series(gs_results['rank_test_F1 Score'])   

    
    gs_model_split=str(gs_model).replace("[{","").replace("}]","").split('}, {')
    df_gs_result=pd.DataFrame(gs_model_split,index=None,columns=['Model_attributes'])
    df_gs_result=pd.concat([df_gs_result,gs_mean_test_AUC,gs_std_test_AUC,gs_rank_test_AUC,gs_mean_test_Accuracy,gs_std_test_Accuracy,gs_rank_test_Accuracy,gs_mean_test_Recall,gs_std_test_Recall,gs_rank_test_Recall,gs_mean_test_Precision,gs_std_test_Precision,gs_rank_test_Precision,gs_mean_test_F1_Score,gs_std_test_F1_Score,gs_rank_test_F1_Score],axis=1)
    
    df_gs_result.columns=['Model_attributes','mean_test_AUC','std_test_AUC','rank_test_AUC','mean_test_Accuracy','std_test_Accuracy','rank_test_Accuracy','mean_test_Recall','std_test_Recall','rank_test_Recall','mean_test_Precision','std_test_Precision','rank_test_Precision','mean_test_F1_Score','std_test_F1_Score','rank_test_F1_Score']  
import numpy as np

#Number of tress in Random Forest
n_estimators = [int(x) for x in np.linspace(start = 200,stop = 1000, num = 5)]

#Number of features to consider while splitting
max_features = ['auto', 'sqrt']

#Maximum number of levels in the tree
max_depth = [int(x) for x in np.linspace(start = 7, stop = 10, num = 4)]

#Minimum # of samples required to split the node
min_samples_split = [10,15]

#Minimum # of samples required at each leaf node
min_samples_leaf = [3,5]

#Method of selecting samples from training each tree
bootstrap = [True, False]

from sklearn.metrics import accuracy_score,make_scorer,precision_score,recall_score,roc_auc_score,f1_score
scoring={'AUC' : make_scorer(roc_auc_score) , 
         'Accuracy':make_scorer(accuracy_score), 
         'Recall':make_scorer(recall_score), 
         'Precision':make_scorer(precision_score),
         'F1 Score':make_scorer(f1_score)}

#Create the grid:
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf ,
               'bootstrap': bootstrap                
              }
   
from sklearn.ensemble import RandomForestClassifier
RFcl = RandomForestClassifier(random_state = 0, n_jobs = -1) 

from sklearn.model_selection import cross_val_score, GridSearchCV
GS_rfc = GridSearchCV(estimator=RFcl, param_grid=random_grid, cv= 10, n_jobs = -1,scoring=scoring,refit='Recall',return_train_score=True)
GS_rfc.fit(X_train, y_train)
print(GS_rfc.best_score_)
#print(GS_rfc.cv_results_)
    
format_grid_search_result(GS_rfc.cv_results_)
df_gs_result.to_excel('Random_forest_Grid_Search_19August.xlsx')
X_train.columns
from xgboost import XGBClassifier
from sklearn import metrics

xgb_model=XGBClassifier(learning_rate=0.1,            
                        n_estimators=200,
                        max_depth=10,
                        objective='binary:logistic',
                        nthread=4,scale_pos_weight=9.8,seed=100).fit(X_train,y_train)

y_pred    = xgb_model.predict(X_train)  
accuracy  = str(metrics.accuracy_score(y_train,y_pred))
recall    = str(metrics.recall_score(y_train,y_pred))
precision = str(metrics.precision_score(y_train,y_pred))
f1_score  = str(metrics.f1_score(y_train,y_pred))
conf_mat  = str(metrics.confusion_matrix(y_train,y_pred))

print("====== XGBOOST Development Metrics ======")
print(" Accuracy : "  + str(accuracy))
print(" Recall : "  + str(recall))
print(" Precision : "  + str(precision))
print(" F1_Score : "  + str(f1_score))
print(" Confusion_metrics : "  + str(conf_mat))
print(" ")

y_pred = xgb_model.predict(X_test)
y_test_score = xgb_model.predict_proba(X_test)

print("####### XGBOOST Validation Metrics ########")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("f1 score:", metrics.f1_score(y_test, y_pred))
print("Confusion Matrix:", metrics.confusion_matrix(y_test, y_pred))


''' Variance Inflation Factor (VIF) Calculation '''
from statsmodels.stats.outliers_influence import variance_inflation_factor
X=X_train
vif = pd.DataFrame()
vif["features"] = X.columns
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)
from sklearn import metrics

y_pred = xgb_model.predict(X_train)
y_train_score = xgb_model.predict_proba(X_train)
xgboost_model_accuracy = metrics.accuracy_score(y_train,y_pred)
                
print("====== Classification Metrics - Development ======")
print(" Accuracy : "  + str(metrics.accuracy_score(y_train,y_pred)))
print(" Recall : "  + str(metrics.recall_score(y_train,y_pred)))
print(" Precision : "  + str(metrics.precision_score(y_train,y_pred)))
print(" F1_Score : "  + str(metrics.f1_score(y_train,y_pred)))
print(" Confusion_metrics : "  + str(metrics.confusion_matrix(y_train,y_pred)))
print(" ")

y_train_score_df = pd.DataFrame(y_train_score, index=range(y_train_score.shape[0]),columns=range(y_train_score.shape[1]))
y_train_score_df['Actual'] = pd.Series(y_train.values, index=y_train_score_df.index)
y_train_score_df['Predicted'] = pd.Series(y_pred, index=y_train_score_df.index)
y_train_score_df['Decile'] = pd.qcut(y_train_score_df[1],10,duplicates='drop')

lift_tbl = pd.DataFrame([y_train_score_df.groupby('Decile')[1].min(),
                                                 y_train_score_df.groupby('Decile')[1].max(),
                                                 y_train_score_df[(y_train_score_df['Actual'] == 1)].groupby('Decile')[1].count(),
                                                 y_train_score_df[(y_train_score_df['Actual'] == 0)].groupby('Decile')[1].count(),
                                                 y_train_score_df.groupby('Decile')[1].count()]).T
lift_tbl.columns = ["MIN","MAX","Event","Non-Event","TOTAL"]
lift_tbl = lift_tbl.sort_values("MIN", ascending=False)
lift_tbl = lift_tbl.reset_index()

list_vol_pct=[]
list_event_pct=[]

for i in range(len(lift_tbl.Event)):
    list_vol_pct.append(lift_tbl['TOTAL'][i]/lift_tbl['TOTAL'].sum())
    list_event_pct.append(lift_tbl['Event'][i]/lift_tbl['TOTAL'][i])

lift_tbl = pd.concat([lift_tbl,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)


lift_tbl = lift_tbl[["Decile","MIN","MAX","Event","Non-Event","TOTAL",0,1]]        
lift_tbl = lift_tbl.rename(columns={lift_tbl.columns[len(lift_tbl.keys())-2]: "Volume(%)"})
lift_tbl = lift_tbl.rename(columns={lift_tbl.columns[len(lift_tbl.keys())-1]: "Event(%)"})

lift_tbl["Cumm_Event"] = lift_tbl["Event"].cumsum()
lift_tbl["Cumm_Event_Pct"] = lift_tbl["Cumm_Event"] / lift_tbl["Event"].sum()
#lift_tbl
lift_tbl.to_excel("TVS_Loan_data_Validation_Lift_Chart_optimized_xgboost_20August2020.xlsx", index = None, header=True)
lift_tbl
# standard neural network on an imbalanced classification dataset
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


 
# define the neural network model
def define_model(n_input):
    # define model
    model = Sequential()
    # define LSTM model
    model.add(LSTM(5, input_shape=(2,1)))
    # define first hidden layer and visible layer
    model.add(Dense(50, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
    # define output layer
    model.add(Dense(1, activation='sigmoid'))
    # define loss and optimizer
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model
 
# define the model
n_input = X_train.shape[1]
model = define_model(n_input)
# fit model
model.fit(X_train, y_train, batch_size=10, epochs=100, verbose=0)
# make predictions on the test dataset
yhat = model.predict(X_test)
# evaluate the ROC AUC of the predictions
score = roc_auc_score(y_test, yhat)
print('ROC AUC: %.3f' % score)
from sklearn import metrics
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

n_input = X_train.shape[1]
model = define_model(n_input)
# fit model
model.fit(X_train, y_train, epochs=100, verbose=0)

# predict probabilities for train set
yhat_probs = model.predict(X_train, verbose=0)
# predict crisp classes for train set
yhat_classes = model.predict_classes(X_train, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
 
print("####### LSTM Developement Evaluation Metrics ########")
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_train, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_train, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_train, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_train, yhat_classes)
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(y_train, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(y_train, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(y_train, yhat_classes)
print(matrix)

# predict probabilities for test set
yhat_probs = model.predict(X_test, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(X_test, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
 
print("####### LSTM Validation Evaluation Metrics ########")
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(y_test, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(y_test, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(y_test, yhat_classes)
print(matrix)
from sklearn import metrics

yhat_classes = model.predict_classes(X_train, verbose=0)
# predict probabilities for train set
y_train_score = model.predict(X_train, verbose=0)

# reduce to 1d array
yhat_probs = y_train_score[:, 0]
y_pred = yhat_classes[:, 0]

y_train_score_df = pd.DataFrame(y_train_score, index=range(y_train_score.shape[0]),columns=range(y_train_score.shape[1]))
y_train_score_df['Actual'] = pd.Series(y_train.values, index=y_train_score_df.index)
y_train_score_df['Predicted'] = pd.Series(y_pred, index=y_train_score_df.index)
y_train_score_df['Decile'] = pd.qcut(y_train_score_df[[0]].values[:,0],10,duplicates='drop')

lift_tbl = pd.DataFrame([y_train_score_df.groupby('Decile')[0].min(),
                                                 y_train_score_df.groupby('Decile')[0].max(),
                                                 y_train_score_df[(y_train_score_df['Actual'] == 1)].groupby('Decile')[0].count(),
                                                 y_train_score_df[(y_train_score_df['Actual'] == 0)].groupby('Decile')[0].count(),
                                                 y_train_score_df.groupby('Decile')[0].count()]).T
lift_tbl.columns = ["MIN","MAX","Event","Non-Event","TOTAL"]
lift_tbl = lift_tbl.sort_values("MIN", ascending=False)
lift_tbl = lift_tbl.reset_index()

list_vol_pct=[]
list_event_pct=[]

for i in range(len(lift_tbl.Event)):
    list_vol_pct.append(lift_tbl['TOTAL'][i]/lift_tbl['TOTAL'].sum())
    list_event_pct.append(lift_tbl['Event'][i]/lift_tbl['TOTAL'][i])

lift_tbl = pd.concat([lift_tbl,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)


lift_tbl = lift_tbl[["Decile","MIN","MAX","Event","Non-Event","TOTAL",0,1]]        
lift_tbl = lift_tbl.rename(columns={lift_tbl.columns[len(lift_tbl.keys())-2]: "Volume(%)"})
lift_tbl = lift_tbl.rename(columns={lift_tbl.columns[len(lift_tbl.keys())-1]: "Event(%)"})

lift_tbl["Cumm_Event"] = lift_tbl["Event"].cumsum()
lift_tbl["Cumm_Event_Pct"] = lift_tbl["Cumm_Event"] / lift_tbl["Event"].sum()
#lift_tbl
lift_tbl.to_excel("TVS_Loan_data_Validation_Lift_Chart_optimized_lstm_24August2020.xlsx", index = None, header=True)
lift_tbl
# decision tree  on imbalanced dataset with SMOTE oversampling and random undersampling
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

#model = define_model(n_input)
model = DecisionTreeClassifier(max_depth=10,min_samples_split=50,min_samples_leaf=10,max_features=10,max_leaf_nodes=30,random_state=100)
over = SMOTE()
#under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('over', over), ('model', model)]
pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
print('Mean ROC AUC: %.3f' % mean(scores))