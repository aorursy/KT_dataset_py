import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display_html 
df = pd.read_csv('/kaggle/input/google-analytics-api-ecommerce-data/GA_API_Ecommerce_Data_RAW.csv',index_col=0)
# looking for data formats and null values 

df.info()
df.head()
## Creating new columns using on 'ga:dateHour'

df['ga:dateHour'] = pd.to_datetime(df['ga:dateHour'],format='%Y%m%d%H',errors='coerce')
df['ga:day'] = [d.day for d in df['ga:dateHour']]
df['ga:hour'] = [d.hour for d in df['ga:dateHour']]
## Changing Sessiondurationbucket to 'MinutesessionDuration'

df['minutesessionDuration'] = df['ga:sessionDurationBucket']//60
df['ga:sourceMedium'].value_counts()
# Source Medium split 
df['source'] = df['ga:sourceMedium'].str.split(' / ').str[0]
df['source'].value_counts()
# medium lists
organic=['google / organic',
 'ecosia.org / organic',
 'yahoo / organic',
 'bing / organic',
 'duckduckgo / organic',
 'google.com / referral',
 'google.com.br / referral',
 'accounts.google.com.br / referral']
 
paidsearch=['google / cpc',
     'cpc / GoogleAds',
     'ads.google.com / referral']
 
social=['Instagram / Bio',
     'instagram.com / referral',
     'm.facebook.com / referral',
     'l.instagram.com / referral',
     'IGShopping / Social',
     'facebook.com / referral',
     'youtube.com / referral',
     'l.facebook.com / referral',
     'm.youtube.com / referral',
     'influenciador / youtube',
     'lm.facebook.com / referral',
     'faceads / linkpatrocinado1'
     'faceads / stories',
     'faceads / lp',
     'pinterest.com / referral',
     'mobile.facebook.com / referral',
     'web.facebook.com / referral',
     'faceads / stories',
     'faceads / linkpatrocinado1']
 
direct=['(direct) / (none)']     

others=['t.co / referral',
    'blog / postblog',
    'outlook.live.com / referral',      
    'br.search.yahoo.com / referral',
    'blog / post',
    'adwords.corp.google.com / referral',
    'accounts.google.com / referral',     
    'mail.google.com / referral',  
    'ebit.com.br / referral',
    'googleweblight.com / referral',
    'qpl-search.com / referral',
    'qo-br.com / referral',
    'g.results.supply / referral',
    'mail1.uol.com.br / referral',
    'baidu.com / referral',
    'br-nav.com / referral',     
    'org-search.com / referral',
    'sts-sec.lhoist.com / referral',
    'bmail1.uol.com.br / referral'
        ]


# loc the mediums

df.loc[df['ga:sourceMedium'].isin(organic),'medium']='Organic'
df.loc[df['ga:sourceMedium'].isin(paidsearch),'medium']='Paid Search'
df.loc[df['ga:sourceMedium'].isin(direct),'medium']='Direct'
df.loc[df['ga:sourceMedium'].isin(social),'medium']='Social'
df.loc[df['ga:sourceMedium'].isin(others),'medium']='Others'
df['medium'].value_counts()
# sources lists

# Social
instagram=['Instagram',
        'l.instagram.com',
       'instagram.com',
       'IGShopping']

facebook=['m.facebook.com',
          'facebook.com',
          'l.facebook.com',
          'web.facebook.com',
          'lm.facebook.com',
          'faceads',
          'mobile.facebook.com']

youtube=['youtube.com',
         'm.youtube.com',
         'influenciador']

pinterest=['pinterest.com']

# Paidsearch
googleAds = ['google',
'ads.google.com',
'cpc' ]

#organic
google=['google.com',
        'accounts.google.com.br',
        'google.com.br']
# loc sources

df.loc[(df['medium']=='Social')&(df['source'].isin(instagram)),'source']='instagram'
df.loc[(df['medium']=='Social')&(df['source'].isin(facebook)),'source']='facebook'
df.loc[(df['medium']=='Social')&(df['source'].isin(youtube)),'source']='youtube'
df.loc[(df['medium']=='Social')&(df['source'].isin(pinterest)),'source']='pinterest'
df.loc[(df['medium']=='Paid Search')&(df['source'].isin(googleAds)),'source']='google ads'
df.loc[(df['medium']=='Organic ')&(df['source'].isin(google)),'source']='google'

df['source'].value_counts()
df.columns
cols=['ga:transactions',
 'source',
 'medium',
 'ga:transactionRevenue',
 'ga:itemQuantity',
 'minutesessionDuration',
 'ga:pageDepth',
 'ga:hits',
 'ga:daysSinceLastSession',
 'ga:operatingSystem',
 'ga:region',
 'ga:userType',
 'ga:day',
 'ga:hour',
 ]
df = df[cols]
df.tail()
# Getting descriptive infos

df.describe()
# Percentage of Bounce Rate

pd.DataFrame(df['minutesessionDuration'].value_counts(normalize=True).head())
# Numeric columns 
columns_n = ['minutesessionDuration','ga:pageDepth','ga:hits']
df_n = df[columns_n]

# Plotting the distribution

l = df_n.columns.values
ncols = int(len(l))
nrows = 1

fig,ax2d = plt.subplots(nrows,ncols)
fig.set_size_inches(15,6)
fig.subplots_adjust(wspace=3)

ax = np.ravel(ax2d)

for count,i in enumerate(df_n):
    sns.boxplot(y = df_n[i],ax = ax[count],color = 'mediumpurple')
def Iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return IQR,Q3
def Sales_Central_Tendency(df):
    
    #DF total sales
    
    sales = df[df['ga:transactionRevenue']!=0]
    
    #Creating the table
    
    total_sales = {' ': ['ga:itemQuantity',
                      'ga:transactions', 
                      'ga:transactionRevenue'
                     ],
                'Mean': [sales['ga:itemQuantity'].mean(), 
                         sales['ga:transactions'].mean(), 
                         '%.2f' %sales['ga:transactionRevenue'].mean()
                         ],
              'Median': [sales['ga:itemQuantity'].median(), 
                          sales['ga:transactions'].median(),       
                          sales['ga:transactionRevenue'].median()
                         ],
                'Mode': [sales['ga:itemQuantity'].mode()[0], 
                         sales['ga:transactions'].mode()[0],       
                         sales['ga:transactionRevenue'].mode()[0]
                         ],
               'Total': [sales['ga:itemQuantity'].sum(), 
                          sales['ga:transactions'].sum(),        
                        '%.2f' % sales['ga:transactionRevenue'].sum()
                         ]
                    }
     
    total_sales = pd.DataFrame(total_sales, columns = [' ','Mean','Median','Mode','Total']).set_index(' ')
    
    
    return total_sales
# Sales central tendency with outliers

Sales_Central_Tendency(df)
def Conversion_Rate(df):
    print(len(df),'Sessions in this period')
    print('Non-transactions', round(df['ga:transactions'].value_counts()[0]/len(df) * 100,2), '% ')
    print('Transactions', round(df['ga:transactions'].value_counts()[1]/len(df) * 100,2), '% ')
Conversion_Rate(df)
def Feature_Reduction(df):
        
    drop_elements=['ga:itemQuantity','ga:transactionRevenue'] 
    df = df.drop(drop_elements, axis = 1)
    
        # Top 10 Regions
    
    all_states = pd.DataFrame(df['ga:region'].value_counts())
    states = all_states.head(10)
        ##Other states
    all_others_states = all_states[~all_states['ga:region'].isin(list(states['ga:region']))].reset_index()
    others_states = pd.DataFrame(all_others_states['index'])
        ##Changing other regions per '(Others)'
    df.loc[df['ga:region'].isin(others_states['index']),'ga:region']='(Others)'


        #Top 5 operatingSystem 
        
    all_operatingSystem = pd.DataFrame(df['ga:operatingSystem'].value_counts())
    operatingSystem = all_operatingSystem.head()
        ##Other operatingSystem 
    all_others_operatingSystem = all_operatingSystem[~all_operatingSystem['ga:operatingSystem'].isin(list(operatingSystem['ga:operatingSystem']))]
    others_operatingSystem = list(all_others_operatingSystem['ga:operatingSystem'])
        ##Changing other operatingSystem per '(Others)'
    df.loc[df['ga:operatingSystem'].isin(others_operatingSystem),'ga:operatingSystem']='(Others)'
    
    
    df=df.reset_index(drop=True)
    
    return df
def Buyer_Profile(df):
    
    # Transaction = 1
    sales = df[df['ga:transactions']!=0]
    
    sales = Feature_Reduction(sales)
    
    # Checking if there is more than 1 channel Groupping in the DF
    
    if len(sales['medium'].value_counts()) > 1:
        sales = sales[['ga:transactions','ga:operatingSystem','ga:region','source','medium']]
    elif len(sales['source'].value_counts()) == 1:
        sales = sales[['ga:transactions','ga:operatingSystem','ga:region']]
    else:
        sales = sales[['ga:transactions','ga:operatingSystem','ga:region','source']]
    
    # Prepareing the df to plot the categorical variables of the buyers
    
    catdf = sales.columns[1::]
    
    nrows = 1
    ncols = len(catdf)

    fig,ax2d = plt.subplots(nrows,ncols)
    fig.set_size_inches(30,5)
    fig.subplots_adjust(wspace=0.2)

    ax=np.ravel(ax2d)

    for count,i in enumerate(catdf):
            plot1=sns.barplot(x=i,
                y='ga:transactions',
                hue='ga:transactions',
                color='#4B0082',
                data=sales,
                ax=ax[count],
                estimator=lambda x: len(x) / len(df) * 100)
    
            plot1.set_xticklabels(plot1.get_xticklabels(),rotation=90)
            plot1.set(ylabel = "Transaction Frequencie")
            plot1.legend().remove()
        
Buyer_Profile(df)
def Behavior_Dist(df):  
    
    #spliting the data
    sales = df[df['ga:transactions']!=0]
    non_sales = df[df['ga:transactions']==0]
    
    behavior=['minutesessionDuration','ga:pageDepth','ga:hits']
    
    sales = sales[behavior]
    non_sales = non_sales[behavior]
    
    
    #Calculing the Interquantilerange
        
    for count,i in enumerate(behavior):
        Q1 = sales[behavior[count]].quantile(0.25)
        Q3 = sales[behavior[count]].quantile(0.75)
        IQR = Q3 - Q1
        #Apling the 1.5 rule for higer out liers
        sales = sales[~(sales[behavior[count]] > (Q3 + 1.5 * IQR))]
    
    sales = sales[sales['minutesessionDuration']>0]   
    
    for count,i in enumerate(behavior):
        Q1 = non_sales[behavior[count]].quantile(0.25)
        Q3 = non_sales[behavior[count]].quantile(0.75)
        IQR = Q3 - Q1
        #Apling the 1.5 rule for higer out liers
        non_sales = non_sales[~(non_sales[behavior[count]] > (Q3 + 1.5 * IQR))]
                
    non_sales = non_sales[non_sales['minutesessionDuration']>0]   
    
    
    non_sales = non_sales.add_prefix("non_sale ")
    sales = sales.add_prefix("sale ")

    # For loop to plot the distribution both dfs
    
    l = sales.columns.values

    nrows = 2
    ncols = int(len(l))

    fig,ax2d = plt.subplots(nrows,ncols)
    fig.set_size_inches(20,10)
    fig.subplots_adjust(wspace=0.4)

    ax=np.ravel(ax2d)
    
    #sale_dist_=pd.DataFrame()
    lst=[]
    lst2=[]
    columns = [' ','Mean','Median','Mode','IQR','12.5th middle','87.5th middle']
    
    for count,i in enumerate(sales):
        sns.distplot(sales[i], kde=True, ax=ax[count], color='#DAB5F8')
        sns.distplot(non_sales["non_"+i], kde=True, ax=ax[count+ncols], color='#F5AE33')
    
        lst.append([i,sales[i].mean(),sales[i].median(),sales[i].mode()[0],Iqr(sales[i])[0],
                            sales[i].quantile(0.5 - 0.75/2),sales[i].quantile(0.5 + 0.75/2)])
        
        lst2.append(["non_"+i,non_sales["non_"+i].mean(),non_sales["non_"+i].median(),non_sales["non_"+i].mode()[0],
                           Iqr(non_sales["non_"+i])[0],non_sales["non_"+i].quantile(0.5 - 0.75/2),non_sales["non_"+i].quantile(0.5 + 0.75/2)])
                                
        sale_dist = pd.DataFrame(lst,columns=columns).set_index(' ')
        non_sale_dist = pd.DataFrame(lst2,columns=columns).set_index(' ')
        
    return sale_dist,non_sale_dist             

full_sale_dist, full_non_sale_dist = Behavior_Dist(df)
def two_tables(df1,df2):
    df1_styler = df1.style.set_table_attributes("style='display:inline'").set_caption('Sale')
    df2_styler = df2.style.set_table_attributes("style='display:inline'").set_caption('Non Sale')
    
    display_html(df1_styler._repr_html_()+df2_styler._repr_html_(), raw=True)
two_tables(full_sale_dist,full_non_sale_dist)
def Transaction_Hour_Plot(df):
    # Df with the hour of transaction
    sales = df[df['ga:transactions']!=0]
    df2=df[df['ga:transactions']!=0]
    print(df2['ga:transactions'].corr(df['ga:hour']))
    transaction_hour = sales[['ga:hour','ga:transactions']]
    transaction_hour = transaction_hour.groupby(['ga:hour'],as_index=False).sum().sort_values('ga:hour',
                                                                                  ascending=True).set_index('ga:hour')
    # Plot the heatmap
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(30,10)) 
    sns.heatmap(transaction_hour.T,cmap = "Purples", ax=ax1, linewidths=.2 )
    
    # The dependent variable is denoted "Y" and the independent variables are denoted by "X"
    sns.regplot(y = transaction_hour.reset_index().columns[1],
                x = transaction_hour.reset_index().columns[0],
                data = transaction_hour.reset_index(), ax=ax2, color='#4B0082')
    
Transaction_Hour_Plot(df)
def Transactions_Day_Plot(df):

    # df with days of transaction
    sales = df[df['ga:transactions']!=0]
  
    transaction_day = sales[['ga:day','ga:transactions']]
    transaction_day = transaction_day.groupby(['ga:day'],as_index=False).sum().sort_values('ga:day',
                                                                                   ascending=True).set_index('ga:day')
    
    #plot
    fig, (ax1 ,ax2) = plt.subplots(1, 2, figsize=(20,7))         

    sns.heatmap(transaction_day,cmap = "Purples", ax=ax1, linewidths=.2 )
    
    sns.regplot(y = transaction_day.reset_index().columns[1],
                x = transaction_day.reset_index().columns[0],
                data = transaction_day.reset_index(), ax=ax2, color='#4B0082')
Transactions_Day_Plot(df)
def Transaction_Corr(df):
    
    drop_elements = ['ga:day','ga:hour']
    
    behavior=['minutesessionDuration','ga:pageDepth','ga:hits']
   

    #Calculing the Interquartile Range
        
    for count,i in enumerate(behavior):
        Q1 = df[behavior[count]].quantile(0.25)
        Q3 = df[behavior[count]].quantile(0.75)
        IQR = Q3 - Q1
        #Apling the 1.5 rule for higher outliers
        df_without = df[~(df[behavior[count]] > (Q3 + 1.5 * IQR))]
    
    df = df_without[df_without['minutesessionDuration']>0]   
    
    ## drop sources columns
    for columns in df:
        if columns[:6]=='source':
            drop_elements.append(columns)
        else:
            pass
    
    df = df.drop(drop_elements, axis = 1)
    
    
    corr_df = pd.DataFrame(df.corr()['ga:transactions']).sort_values(['ga:transactions'],ascending=False)[1:]
    plt.figure(figsize=(10,10))
    plt.title('Feature Correlation with ga:transactions', fontdict={'fontweight':'bold'})
    sns.barplot(x ='ga:transactions',y = corr_df.index,data = corr_df ,color='#DAB5F8')

# Preparation

df_p = df.copy()

df_p = Feature_Reduction(df_p)

df_p = pd.get_dummies(df_p)
Transaction_Corr(df_p)
df_organic = df[df['medium']=='Organic']
df_paidsearch = df[df['medium']=='Paid Search']
df_direct = df[df['medium']=='Direct']
df_social = df[df['medium']=='Social']
df_others = df[df['medium']=='Others']
Sales_Central_Tendency(df_organic) 
Conversion_Rate(df_organic)
Buyer_Profile(df_organic)
o_sale_dist, o_non_sale_dist = Behavior_Dist(df_organic)
two_tables(o_sale_dist,o_non_sale_dist)
Transaction_Hour_Plot(df_organic)
Transactions_Day_Plot(df_organic)
df_organic_p = df_organic.copy()

df_organic_p = Feature_Reduction(df_organic_p)

df_organic_p = pd.get_dummies(df_organic_p)
Transaction_Corr(df_organic_p)
Sales_Central_Tendency(df_paidsearch)
Conversion_Rate(df_paidsearch)
Buyer_Profile(df_paidsearch)
ps_sale_dist,ps_non_sale_dist=Behavior_Dist(df_paidsearch)
two_tables(ps_sale_dist, ps_non_sale_dist)
Transaction_Hour_Plot(df_paidsearch)
Transactions_Day_Plot(df_paidsearch)
df_paidsearch_p = df_paidsearch.copy()

df_paidsearch_p = Feature_Reduction(df_paidsearch_p)

df_paidsearch_p = pd.get_dummies(df_paidsearch_p)

Transaction_Corr(df_paidsearch_p)
Sales_Central_Tendency(df_direct)
Conversion_Rate(df_direct)
Buyer_Profile(df_direct)
d_sale_dist,d_non_sale_dist = Behavior_Dist(df_direct)
two_tables(d_sale_dist, d_non_sale_dist)
Transaction_Hour_Plot(df_direct)
Transactions_Day_Plot(df_direct)
df_direct_p = df_direct.copy()

df_direct_p = Feature_Reduction(df_direct_p)

df_direct_p = pd.get_dummies(df_direct_p)

Transaction_Corr(df_direct_p)
Sales_Central_Tendency(df_social)
Conversion_Rate(df_social)
Buyer_Profile(df_social)
s_sale_dist,s_non_sale_dist = Behavior_Dist(df_social)
two_tables(s_sale_dist, s_non_sale_dist)
Transaction_Hour_Plot(df_social)
Transactions_Day_Plot(df_social)
df_social_p = df_social.copy()

df_social_p = Feature_Reduction(df_social_p)

df_social_p = pd.get_dummies(df_social_p )

Transaction_Corr(df_social_p )
Sales_Central_Tendency(df_others)
Buyer_Profile(df_others)
df[df['minutesessionDuration']==0]['ga:operatingSystem'].value_counts(normalize=True).head()
df[df['minutesessionDuration']==0]['ga:userType'].value_counts(normalize=True).head()
df[(df['minutesessionDuration']==0)&
   (df['ga:userType']=='New Visitor')]['ga:operatingSystem'].value_counts(normalize=True).head()
df[(df['minutesessionDuration']==0)&
   (df['ga:userType']=='New Visitor')&
   (df['ga:operatingSystem']=='Android')]['ga:region'].value_counts(normalize=True).head()
