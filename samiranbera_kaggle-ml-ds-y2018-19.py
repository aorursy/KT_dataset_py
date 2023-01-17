import pandas as pd

import pandas_profiling



import numpy as np

import plotly.express as px

import plotly.offline as py

py.init_notebook_mode(connected=False)

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import requests



import warnings

warnings.filterwarnings('ignore')



df19 = pd.read_csv('/kaggle/input/kaggle-ml-ds-survey-2017-19/Y19.csv')

df18 = pd.read_csv('/kaggle/input/kaggle-ml-ds-survey-2017-19/Y18.csv')

df17 = pd.read_csv('/kaggle/input/kaggle-ml-ds-survey-2017-19/Y17.csv',encoding='ISO-8859-1')



# Create Copy of Original DataFrame

data17=df17

data18=df18

data19=df19



print("Shape=",df17.shape,df18.shape,df19.shape)
def get_pie_values(df):        

    unique_cols = df.unique()

    n_col       = unique_cols.size    

    

    count=[]

    names=[]    

    for i in unique_cols:

        names.append(i)

        count.append(df[df==i].count().max())        

    return names,count

    

def draw_pie(labels,values,text):

    fig=px.pie(labels=labels,values=values,names=labels,hole=0.45)

    fig.update_traces(textinfo='percent+label')

    fig.update_layout(annotations=[dict(text=text,showarrow=False)])

    print(labels,values,text)

    fig.show()



def pie_3_year(a_label,a_value,a_text,b_label,b_value,b_text,c_label,c_value,c_text):

    specs = [[{'type':'domain'},{'type':'domain'},{'type':'domain'}]]

    fig=make_subplots(rows=1,cols=3,specs=specs)

    fig.add_trace(go.Pie(labels=a_label,values=a_value,hole=0.45,title=a_text),1,1)

    fig.add_trace(go.Pie(labels=b_label,values=b_value,hole=0.45,title=b_text),1,2)

    fig.add_trace(go.Pie(labels=c_label,values=c_value,hole=0.45,title=c_text),1,3)

    fig.show()

    

def get_sorted(labels,values):

    df=pd.DataFrame([])

    df['labels']=labels

    df['values']=values   

    df.sort_values(by=['values'],inplace=True,ascending=False)    

    return df['labels'],df['values'] 



def currency_detail():

    url = 'http://data.fixer.io/api/latest?access_key=1bab5f24a96b33ee3e2ec8827b5b59b0'

    data = requests.get(url).json()["rates"]

    return data

    

def histogram_overlayed(a_label,a_text,b_label,b_text,c_label,c_text,main_title,x_axis_title):    

    fig = go.Figure()

    fig.add_trace(go.Histogram(x=a_label,name=a_text))

    fig.add_trace(go.Histogram(x=b_label,name=b_text))

    fig.add_trace(go.Histogram(x=c_label,name=c_text))

    fig.update_layout(barmode='overlay')

    fig.update_traces(opacity=0.5)

    fig.update_layout(title_text=main_title,xaxis_title_text=x_axis_title,yaxis_title_text='Count')

    fig.show()

    

def barh_3_year(a_label,a_value,b_label,b_value,c_label,c_value):

    fig = make_subplots(rows=1, cols=3,specs=[[{"type": "xy"}, {"type": "xy"},{"type": "xy"}]])

    fig.add_trace(go.Bar(x=a_value,y=a_label,orientation='h',name='2017'),row=1, col=1)

    fig.add_trace(go.Bar(x=b_value,y=b_label,orientation='h',name='2018'),row=1, col=2)

    fig.add_trace(go.Bar(x=c_value,y=c_label,orientation='h',name='2019'),row=1, col=3)    

    fig.show()    



def count_country_by_jobrole(df,filter_country):  

    df.columns = ['Country','JobRole','Salary']

    if filter_country==1:

        df = filter_in_country(df)         

    ds = df[df.JobRole=='Data Scientist'] 

    da = df[df.JobRole=='Data Analyst']   

    ba = df[df.JobRole=='Business Analyst']       

    ds = count_by_country(ds)  

    da = count_by_country(da)

    ba = count_by_country(ba)    

    ds['JobRole'] = 'Data Scientist'

    da['JobRole'] = 'Data Analyst'

    ba['JobRole'] = 'Business Analyst'    

    ds = ds.append(da)

    ds = ds.append(ba)

    return ds



def count_by_country(df):    

    country_list = df.Country.unique()       

    count =[]        

    salary_mean=[]

    salary_sum =[]    

    for i in country_list:

        count.append(df.Country[df.Country==i].size)        

        salary_mean.append(df.Salary[df.Country==i].mean())

        salary_sum.append(df.Salary[df.Country==i].sum())

    temp1 = pd.DataFrame()

    temp1['Country'] = country_list    

    temp1['Y_Value'] = count

    temp1['Y_Type']  = 'Count'  

    

    temp2 = pd.DataFrame()

    temp2['Country'] = country_list    

    temp2['Y_Value'] = salary_mean

    temp2['Y_Type']  = 'Salary_Mean' 

    

    temp3 = pd.DataFrame()

    temp3['Country'] = country_list    

    temp3['Y_Value'] = salary_sum 

    temp3['Y_Type']  = 'Salary_Sum' 

    

    temp1=temp1.append(temp3)    

    temp1=temp1.append(temp2)    

    return temp1



def count_country_by_gender(df,filter_country):      

    df.columns = ['Country','Gender','Salary']

    if filter_country==1:

        df = filter_in_country(df)       

    male     = df[df.Gender=='Male']  # Filter by Gender

    female   = df[df.Gender=='Female']       

    male   = count_by_country(male)  # Sub-Filter by Country - Count + Salary

    female = count_by_country(female)        

    male['Gender']  ='male'

    female['Gender']='female'    

    return male.append(female)

    

def filter_in_country(df):         

    index = ((df.Country=="United States of America") | (df.Country=="India")      | (df.Country=='Brazil')   | 

             (df.Country=='Canada')                   | (df.Country=='Australia')  | (df.Country=='Taiwan')   | 

             (df.Country=='Turkey')                   | (df.Country=='Indonesia')  | (df.Country=='Columbia') | 

             (df.Country=='Hungary')                  | (df.Country=='Chile')      | (df.Country=='Germany')  | 

             (df.Country=='France')                   | (df.Country=='Netherland') | (df.Country=='Ireland'))

    return df[index]



def count_country_by_education(df,filter_country):  

    df.columns = ['Country','education','Salary']

    if filter_country==1:

        df = filter_in_country(df)        

    bachelors = df[df.education=='Bachelors']  # Filter by Gender

    masters   = df[df.education=='Masters']   

    doctors   = df[df.education=='Doctoral degree']   

    

    bachelors = count_by_country(bachelors)  # Sub-Filter by Country - Count + Salary

    masters   = count_by_country(masters)

    doctors   = count_by_country(doctors)

    

    bachelors['education'] ='Bachelors'

    masters['education']   ='Masters'    

    doctors['education']   ='Doctoral degree'            

    

    bachelors = bachelors.append(masters)

    bachelors = bachelors.append(doctors)

    return bachelors
data17[(df17.GenderSelect!='Male') & (df17.GenderSelect!='Female')]='Others'

data18[(df18.Q1!='Male') & (df18.Q1!='Female')]='Others'

data19[(df19.Q2!='Male') & (df19.Q2!='Female')]='Others'



a_label,a_value = get_pie_values(data17.iloc[:,0])

b_label,b_value = get_pie_values(data18.iloc[:,1])

c_label,c_value = get_pie_values(data19.iloc[:,2])



pie_3_year(a_label,a_value,'2017 Gender',b_label,b_value,'2018 Gender',c_label,c_value,'2019 Gender')
data17[data17.Age=='Others']=0

index  = data17.Age.isnull()

data17 = data17[~index]

data17.Age[data17.Age<18]                      =0

data17.Age[(data17.Age>=18) & (data17.Age<=21)]=1

data17.Age[(data17.Age>=22) & (data17.Age<=24)]=2

data17.Age[(data17.Age>=25) & (data17.Age<=29)]=3

data17.Age[(data17.Age>=30) & (data17.Age<=34)]=4

data17.Age[(data17.Age>=35) & (data17.Age<=39)]=5

data17.Age[(data17.Age>=40) & (data17.Age<=44)]=6

data17.Age[(data17.Age>=45) & (data17.Age<=49)]=7

data17.Age[(data17.Age>=50) & (data17.Age<=54)]=8

data17.Age[(data17.Age>=55) & (data17.Age<=59)]=9

data17.Age[(data17.Age>=60) & (data17.Age<=69)]=10

data17.Age[ data17.Age>=70]                    =11





data18.Q2[data18.Q2=='What is your age (# years)?']=np.nan

data18.Q2[data18.Q2=='Others']=np.nan

index  = data18.Q2.isnull()

data18 = data18[~index]

data18.Q2[data18.Q2=='18-21']=1

data18.Q2[data18.Q2=='22-24']=2

data18.Q2[data18.Q2=='25-29']=3

data18.Q2[data18.Q2=='30-34']=4

data18.Q2[data18.Q2=='35-39']=5

data18.Q2[data18.Q2=='40-44']=6

data18.Q2[data18.Q2=='45-49']=7

data18.Q2[data18.Q2=='50-54']=8

data18.Q2[data18.Q2=='55-59']=9

data18.Q2[data18.Q2=='60-69']=10

data18.Q2[data18.Q2=='70-79']=11

data18.Q2[data18.Q2=='80+']=11





data19.Q1[data19.Q1=='What is your age (# years)?']=np.nan

data19.Q1[data19.Q1=='Others']=np.nan

index  = data19.Q1.isnull()

data19 = data19[~index]

data19.Q1[data19.Q1=='18-21']=1

data19.Q1[data19.Q1=='22-24']=2

data19.Q1[data19.Q1=='25-29']=3

data19.Q1[data19.Q1=='30-34']=4

data19.Q1[data19.Q1=='35-39']=5

data19.Q1[data19.Q1=='40-44']=6

data19.Q1[data19.Q1=='45-49']=7

data19.Q1[data19.Q1=='50-54']=8

data19.Q1[data19.Q1=='55-59']=9

data19.Q1[data19.Q1=='60-69']=10

data19.Q1[data19.Q1=='70+']=11



a_label = np.array(data17.Age)

b_label = np.array(data18.Q2)

c_label = np.array(data19.Q1)



histogram_overlayed(a_label,'2017',

                    b_label,'2018',

                    c_label,'2019',

                    'Age Distribution 2017-2019',

                    'Age')
data17.Country[df17.Country=='United States']='United States of America'

index  = data17.Country=='0'

data17 = data17[~index]



labels,values=get_pie_values(data17.Country)        

a_label,a_value=get_sorted(labels[0:7],values[0:7])



labels,values=get_pie_values(data18.Q3)        

b_label,b_value=get_sorted(labels[0:7],values[0:7])



labels,values=get_pie_values(data19.Q3)        

c_label,c_value=get_sorted(labels[0:7],values[0:7])



pie_3_year(a_label,a_value,'2017 Country',b_label,b_value,'2018 Country',c_label,c_value,'2019 Country')
data17.FormalEducation[data17.FormalEducation=='Bachelor\'s degree']='Bachelors'

data18.Q4[data18.Q4=='Bachelor’s degree']='Bachelors'

data19.Q4[data19.Q4=='Bachelor’s degree']='Bachelors'



data17.FormalEducation[data17.FormalEducation=='Master\'s degree']='Masters'

data18.Q4[data18.Q4=='Master’s degree']='Masters'

data19.Q4[data19.Q4=='Master’s degree']='Masters'



# Doctoral degree



data17[(data17.FormalEducation!='Bachelors') & (data17.FormalEducation!='Masters') & (data17.FormalEducation!='Doctoral degree')]='Others'

data18[(data18.Q4!='Bachelors') & (data18.Q4!='Masters')  & (data18.Q4!='Doctoral degree')]='Others'

data19[(data19.Q4!='Bachelors') & (data19.Q4!='Masters')  & (data19.Q4!='Doctoral degree')]='Others'



labels,values=get_pie_values(data17.FormalEducation)

a_label,a_value=get_sorted(labels,values)



labels,values=get_pie_values(data18.Q4)        

b_label,b_value=get_sorted(labels,values)



labels,values=get_pie_values(data19.Q4)        

c_label,c_value=get_sorted(labels[0:7],values[0:7])



pie_3_year(a_label,a_value,'2017 Education',b_label,b_value,'2018 Education',c_label,c_value,'2019 Education')
data17[data17.CurrentJobTitleSelect=='DBA/Database Engineer']='DB Engg'

data17[data17.CurrentJobTitleSelect=='Operations Research Practitioner']='OR Practitioner'

data17[data17.CurrentJobTitleSelect=='Software Developer/Software Engineer']='SE/SD'

data17[data17.CurrentJobTitleSelect=='Machine Learning Engineer']='ML Engg'

data17[data17.CurrentJobTitleSelect=='Others']='Other'

data17[(data17.CurrentJobTitleSelect!='Data Scientist')       & 

       (data17.CurrentJobTitleSelect!='SE/SD')                &

       (data17.CurrentJobTitleSelect!='Data Analyst')         &

       (data17.CurrentJobTitleSelect!='Scientist/Researcher') &

       (data17.CurrentJobTitleSelect!='Business Analyst')     &          

       (data17.CurrentJobTitleSelect!='Engineer')             &         

       (data17.CurrentJobTitleSelect!='Researcher')           &  

       (data17.CurrentJobTitleSelect!='ML Engg')              &

       (data17.CurrentJobTitleSelect!='Computer Scientist')   &     

       (data17.CurrentJobTitleSelect!='Statistician')          

      ]='Other'

a_labels,a_values=get_pie_values(data17.CurrentJobTitleSelect)

a_labels,a_values=get_sorted(a_labels,a_values)

df_a=pd.DataFrame()

df_a['label']=a_labels

df_a['value']=a_values





data18[data18.Q6=='DBA/Database Engineer']='DB Engg'

data18[data18.Q6=='Others']='Other'

data18[(data18.Q6!='Student')            &

       (data18.Q6!='Data Scientist')     &

       (data18.Q6!='Software Engineer')  &

       (data18.Q6!='Data Analyst')       &

       (data18.Q6!='Research Scientist') &       

       (data18.Q6!='Not employed')       &

       (data18.Q6!='C onsultant')         &

       (data18.Q6!='Business Analyst')   &

       (data18.Q6!='Data Engineer')      &

       (data18.Q6!='Research Assistant')     

      ]='Other'

b_labels,b_values=get_pie_values(data18.Q6)

b_labels,b_values=get_sorted(b_labels,b_values)

df_b=pd.DataFrame()

df_b['label']=b_labels

df_b['value']=b_values







data19[data19.Q5=='DBA/Database Engineer']='DB Engg'

data19[data19.Q5=='Others']='Other'

data19[(data19.Q5!='Data Scientist')          &

       (data19.Q5!='Student')                 &

       (data19.Q5!='Software Engineer')       &

       (data19.Q5!='Data Analyst')            &

       (data19.Q5!='Research Scientist')      &       

       (data19.Q5!='Business Analyst')        &

       (data19.Q5!='Product/Project Manager') &

       (data19.Q5!='Data Engineer')           &

       (data19.Q5!='Statistician')            &

       (data19.Q5!='DB Engg')                 

      ]='Other'

c_labels,c_values=get_pie_values(data19.Q5)

c_labels,c_values=get_sorted(c_labels,c_values)

df_c=pd.DataFrame()

df_c['label']=c_labels

df_c['value']=c_values



barh_3_year(a_labels,a_values,b_labels,b_values,c_labels,c_values)   
data17.CompensationAmount[data17.CompensationAmount=='-']=np.nan

data17.CompensationAmount[data17.CompensationAmount=='Others']=np.nan

data17.CompensationAmount[data17.CompensationAmount=='Other']=np.nan

data17.CompensationAmount[data17.CompensationAmount=='SE/SD']=np.nan

data17.CompensationAmount[data17.CompensationAmount=='ML Engg']=np.nan

index = data17.CompensationAmount.isnull()

data17=data17[~index]

index = data17.CompensationCurrency.isnull()

data17=data17[~index]

data17.CompensationAmount[data17.CompensationAmount=='1.00E+11']=100000000000

data17.CompensationAmount=pd.Series(data17.CompensationAmount).str.replace(',', '', regex=True)

data17['CompensationAmount']=data17['CompensationAmount'].astype(float)

index=data17.CompensationCurrency=='SPL'

data17=data17[~index]



#data = currency_detail()

data = {'AED': 4.082165, 'AFN': 87.209783, 'ALL': 121.91121, 'AMD': 532.763209, 'ANG': 1.870829, 'AOA': 535.960801, 'ARS': 66.328102, 'AUD': 1.606111, 'AWG': 2.000571, 'AZN': 1.888563, 'BAM': 1.960997, 'BBD': 2.241807, 'BDT': 94.25855, 'BGN': 1.961793, 'BHD': 0.419205, 'BIF': 2085.935974, 'BMD': 1.111428, 'BND': 1.504765, 'BOB': 7.677729, 'BRL': 4.539574, 'BSD': 1.110325, 'BTC': 0.000154, 'BTN': 79.168486, 'BWP': 11.849721, 'BYN': 2.324023, 'BYR': 21783.995898, 'BZD': 2.237997, 'CAD': 1.462634, 'CDF': 1872.756578, 'CHF': 1.09032, 'CLF': 0.030274, 'CLP': 835.348006, 'CNY': 7.768107, 'COP': 3667.535767, 'CRC': 628.636318, 'CUC': 1.111428, 'CUP': 29.452852, 'CVE': 110.556291, 'CZK': 25.596087, 'DJF': 197.522861, 'DKK': 7.487362, 'DOP': 58.756811, 'DZD': 133.043544, 'EGP': 17.788849, 'ERN': 16.671141, 'ETB': 35.329586, 'EUR': 1, 'FJD': 2.40163, 'FKP': 0.903458, 'GBP': 0.858017, 'GEL': 3.173115, 'GGP': 0.857773, 'GHS': 6.317545, 'GIP': 0.903458, 'GMD': 57.01604, 'GNF': 10593.310433, 'GTQ': 8.560355, 'GYD': 231.199161, 'HKD': 8.657527, 'HNL': 27.345048, 'HRK': 7.466372, 'HTG': 105.915066, 'HUF': 332.9065, 'IDR': 15466.788242, 'ILS': 3.861767, 'IMP': 0.857773, 'INR': 79.174268, 'IQD': 1325.482352, 'IRR': 46796.690987, 'ISK': 135.883263, 'JEP': 0.857773, 'JMD': 148.047071, 'JOD': 0.788037, 'JPY': 121.556562, 'KES': 112.05452, 'KGS': 77.574586, 'KHR': 4502.262484, 'KMF': 493.613097, 'KPW': 1000.373778, 'KRW': 1290.113117, 'KWD': 0.337152, 'KYD': 0.925238, 'KZT': 423.054659, 'LAK': 9854.965285, 'LBP': 1678.723075, 'LKR': 201.407569, 'LRD': 208.531748, 'LSL': 15.749025, 'LTL': 3.281759, 'LVL': 0.672292, 'LYD': 1.558306, 'MAD': 10.710018, 'MDL': 19.20811, 'MGA': 4077.563474, 'MKD': 61.777771, 'MMK': 1657.1159, 'MNT': 3043.273718, 'MOP': 8.905865, 'MRO': 396.779741, 'MUR': 40.745205, 'MVR': 17.17119, 'MWK': 817.905019, 'MXN': 21.093684, 'MYR': 4.597423, 'MZN': 68.80814, 'NAD': 15.749214, 'NGN': 402.917973, 'NIO': 37.455687, 'NOK': 9.908512, 'NPR': 126.669746, 'NZD': 1.675101, 'OMR': 0.428008, 'PAB': 1.110325, 'PEN': 3.685611, 'PGK': 3.832598, 'PHP': 56.572224, 'PKR': 172.001386, 'PLN': 4.371748, 'PYG': 7154.249808, 'QAR': 4.046433, 'RON': 4.791476, 'RSD': 117.755705, 'RUB': 68.738174, 'RWF': 1052.503098, 'SAR': 4.172381, 'SBD': 9.229192, 'SCR': 15.231131, 'SDG': 50.143147, 'SEK': 10.459419, 'SGD': 1.506203, 'SHP': 1.468086, 'SLL': 10864.212151, 'SOS': 645.739756, 'SRD': 8.289053, 'STD': 23963.273517, 'SVC': 9.715598, 'SYP': 572.385958, 'SZL': 15.747892, 'THB': 32.8536, 'TJS': 10.762958, 'TMT': 3.901114, 'TND': 3.157012, 'TOP': 2.550397, 'TRY': 6.596366, 'TTD': 7.501966, 'TWD': 33.270552, 'TZS': 2553.944716, 'UAH': 25.814716, 'UGX': 4069.22143, 'USD': 1.111428, 'UYU': 41.652745, 'UZS': 10575.564254, 'VEF': 11.100393, 'VND': 25754.573722, 'VUV': 128.964041, 'WST': 2.948506, 'XAF': 657.689868, 'XAG': 0.062556, 'XAU': 0.000741, 'XCD': 3.003691, 'XDR': 0.807298, 'XOF': 657.689867, 'XPF': 119.575054, 'YER': 278.218311, 'ZAR': 15.757607, 'ZMK': 10004.192019, 'ZMW': 14.594854, 'ZWL': 357.879934}

v_currency = data17.CompensationCurrency.unique()

for i in v_currency:

    index = data17.CompensationCurrency==i

    

    if i!='EUR':    

        data17.CompensationAmount[index]=data17.CompensationAmount[index]/data[i]   

    data17.CompensationAmount[index]=data17.CompensationAmount[index]*1.111606

 

 

    

data18.Q9[data18.Q9=='I do not wish to disclose my approximate yearly compensation']=np.nan

data18.Q9[data18.Q9=='What is your current yearly compensation (approximate $USD)?']=np.nan

data18.Q9[data18.Q9=='Other']=np.nan

index = data18.Q9.isnull()

data18=data18[~index]

data18.Q9[data18.Q9=='0-10,000']=5000

data18.Q9[data18.Q9=='20-30,000']=25000

data18.Q9[data18.Q9=='30-40,000']=35000

data18.Q9[data18.Q9=='10-20,000']=15000

data18.Q9[data18.Q9=='50-60,000']=55000

data18.Q9[data18.Q9=='70-80,000']=75000

data18.Q9[data18.Q9=='500,000+']=500000

data18.Q9[data18.Q9=='80-90,000']= 85000

data18.Q9[data18.Q9=='60-70,000']= 65000

data18.Q9[data18.Q9=='90-100,000']=95000

data18.Q9[data18.Q9=='40-50,000']= 45000

data18.Q9[data18.Q9=='100-125,000']= 112500

data18.Q9[data18.Q9=='125-150,000']=137500

data18.Q9[data18.Q9=='150-200,000']= 175000

data18.Q9[data18.Q9=='200-250,000']= 225000

data18.Q9[data18.Q9=='250-300,000']= 275000

data18.Q9[data18.Q9=='300-400,000']= 350000

data18.Q9[data18.Q9=='400-500,000']= 450000







data19.Q10[data19.Q10=='DB Engg']=np.nan

data19.Q10[data19.Q10=='Other']=np.nan

data19.Q10[data19.Q10=='What is your current yearly compensation (approximate $USD)?']=np.nan

index = data19.Q10.isnull()

data19=data19[~index]

data19.Q10[data19.Q10=='30,000-39,999']=35000 

data19.Q10[data19.Q10=='5,000-7,499']  =6250

data19.Q10[data19.Q10=='60,000-69,999']=65000

data19.Q10[data19.Q10=='10,000-14,999']=12500

data19.Q10[data19.Q10=='80,000-89,999']=85000

data19.Q10[data19.Q10=='70,000-79,999']=75000

data19.Q10[data19.Q10=='125,000-149,999']=137500

data19.Q10[data19.Q10=='$0-999']=500

data19.Q10[data19.Q10=='40,000-49,999']=45000

data19.Q10[data19.Q10=='20,000-24,999']=22500

data19.Q10[data19.Q10=='15,000-19,999']=17500

data19.Q10[data19.Q10=='100,000-124,999']=112500

data19.Q10[data19.Q10=='90,000-99,999']=95000

data19.Q10[data19.Q10=='7,500-9,999']=8750

data19.Q10[data19.Q10=='150,000-199,999']=175000

data19.Q10[data19.Q10=='25,000-29,999']=27500 

data19.Q10[data19.Q10=='3,000-3,999']=3500

data19.Q10[data19.Q10=='2,000-2,999']=2500

data19.Q10[data19.Q10=='200,000-249,999']=225000

data19.Q10[data19.Q10=='50,000-59,999']=55000

data19.Q10[data19.Q10=='1,000-1,999']=1500

data19.Q10[data19.Q10=='4,000-4,999']=4500

data19.Q10[data19.Q10=='300,000-500,000']=400000

data19.Q10[data19.Q10=='250,000-299,999']=275000

data19.Q10[data19.Q10=='> $500,000']=500000        







data_17 = data17[data17.CompensationAmount<=500000]

data_18 = data18[data18.Q9<=500000]

data_19 = data19[data19.Q10<=500000]

a_label = np.array(data_17.CompensationAmount)    

b_label = np.array(data_18.Q9)

c_label = np.array(data_19.Q10)

    

# No proper response/ignored !!



histogram_overlayed(a_label,'2017',

                    b_label,'2018',

                    c_label,'2019',

                    'Annual Compensation from 2017-2019',

                    'Annual Compensation (in USD)')
data_17_temp = data17[['Country','GenderSelect','CompensationAmount']]

data_18_temp = data18[['Q3','Q1','Q9']]

data_19_temp = data19[['Q3','Q2','Q10']]    



temp=count_country_by_gender(data_17_temp,filter_country=1)

temp['Year']=2017

result = temp



temp=count_country_by_gender(data_18_temp,filter_country=1)

temp['Year']=2018

result = result.append(temp)



temp=count_country_by_gender(data_19_temp,filter_country=1)

temp['Year']=2019

result = result.append(temp)



fig = px.bar(result, x="Country", y="Y_Value", color='Gender', facet_row="Year",facet_col="Y_Type",height=700,barmode='group')

fig.update_yaxes(matches=None)

fig.show()
data_17_temp = data17[['Country','FormalEducation','CompensationAmount']]

data_18_temp = data18[['Q3','Q4','Q9']]

data_19_temp = data19[['Q3','Q4','Q10']]    



temp=count_country_by_education(data_17_temp,filter_country=1)

temp['Year']=2017

result = temp



temp=count_country_by_education(data_18_temp,filter_country=1)

temp['Year']=2018

result = result.append(temp)



temp=count_country_by_education(data_19_temp,filter_country=1)

temp['Year']=2019

result = result.append(temp)



fig = px.bar(result, x="Country", y="Y_Value", color='education', facet_row="Year",facet_col="Y_Type",height=700,barmode='group')

fig.update_yaxes(matches=None)

fig.show()
data_17_temp = data17[['Country','CurrentJobTitleSelect','CompensationAmount']]

data_18_temp = data18[['Q3','Q6','Q9']]

data_19_temp = data19[['Q3','Q5','Q10']]    



temp=count_country_by_jobrole(data_17_temp,filter_country=1)

temp['Year']=2017

result = temp



temp=count_country_by_jobrole(data_18_temp,filter_country=1)

temp['Year']=2018

result = result.append(temp)



temp=count_country_by_jobrole(data_19_temp,filter_country=1)

temp['Year']=2019

result = result.append(temp)



fig = px.bar(result, x="Country", y="Y_Value", color='JobRole', facet_row="Year",facet_col="Y_Type",height=700,barmode='group')

fig.update_yaxes(matches=None)

fig.show()