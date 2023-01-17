# Have used plotty so please make sure to install below commands 

#!pip install plotly

#!pip install cufflinks
import pandas as pd

import numpy as np



import cufflinks as cf

import plotly as py

import plotly.graph_objs as go

import ipywidgets as widgets

from scipy import special 

import matplotlib.pyplot as plt

import seaborn as sns

import math



sns.set(style="whitegrid")



pd.set_option('display.max_columns', 100)



py.offline.init_notebook_mode(connected=True) # plotting in offilne mode 

cf.set_config_file(offline=False, world_readable=True, theme='ggplot')



pd.set_option('display.max_colwidth', -1) # make sure data and columns are displayed correctly withput purge

pd.options.display.float_format = '{:20,.2f}'.format # display float value with correct precision 



encoding_latin="latin"

loan_df = pd.read_csv("../input/loan.csv",low_memory = False,encoding = encoding_latin)
def plot_bar_chat(plotting_frame,x_col_name,y_col_name) :

            

        x_axis_title = x_col_name.title()

        y_axis_title = y_col_name.title()

        

        graph_title = "Bar Chart [" + x_axis_title.title() + " Vs " + y_axis_title.title() + "]"

        

        layout = go.Layout(

             title = graph_title,

             yaxis=dict(

                title=y_axis_title

             ),

             xaxis=dict(

                 title=x_axis_title

             )

        )



        data_to_be_plotted = [

            go.Bar(

                x=plotting_frame[x_col_name], 

                y=plotting_frame[y_col_name]

            )

        ]





        figure = go.Figure(data=data_to_be_plotted,layout=layout)

        py.offline.iplot(figure)

        

        

def plot_pie_chat(plotting_frame,x_col_name,y_col_name) : 

        

        labels = plotting_frame[x_col_name].tolist()

        values = plotting_frame[y_col_name].tolist()



        trace = go.Pie(labels=labels, values=values)



        py.offline.iplot([trace])



        

def plot_box_chart(dataframe) :

    data = []

    for index, column_name in enumerate(dataframe) :

        data.append(

        go.Box(

            y=dataframe.iloc[:, index],

            name=column_name

         ))   

        

    layout = go.Layout(

    yaxis=dict(

        title="Frequency",

        zeroline=False

    ),

       boxmode='group'

    )

    

    fig = go.Figure(data=data, layout=layout)    

    py.offline.iplot(fig) 

    

def plot_group_bar_chart(plot,col,hue) : 

    hue_col = pd.Series(data = hue)

    fig, ax = plt.subplots()

    width = len(plot[col].unique()) + 6 + 5*len(hue_col.unique())

    fig.set_size_inches(width , 10)

    ax = sns.countplot(data = loan_plot, x= col, order=plot[col].value_counts().index,hue = hue,palette="Set2") 

    

    for p in ax.patches:

                # Some segment wise value we are getting as Nan as respective value not present to tackle the Nan using temp_height

                temp_height = p.get_height()

                

                if math.isnan(temp_height):

                    temp_height = 0.01

                    

                

                ax.annotate('{:1.1f}%'.format((temp_height*100)/float(len(loan_plot))), (p.get_x()+0.05, temp_height+20)) 

    

    plt.show()



def validate_year(date) :

    temp = date.split('-')[1]

    lenght = len(temp)

    if lenght == 2 :

        temp = "20"+temp

    else :

        temp = "200"+temp

        

    return temp    
# Method to get Meta-Data about any dataframe passed 

def get_meta_data(dataframe) :

    metadata_matrix = pd.DataFrame({

                    'Datatype' : dataframe.dtypes, # data types of columns

                    'Total_Element': dataframe.count(), # total elements in columns

                    'Null_Count': dataframe.isnull().sum(), # total null values in columns

                    'Null_Percentage': dataframe.isnull().sum()/len(dataframe) * 100 # percentage of null values

                       })

    return metadata_matrix
loan_metadata_matrix = get_meta_data(loan_df)

loan_metadata_matrix.head(2)
loan_metadata_matrix_group = loan_metadata_matrix.groupby("Null_Percentage").count().reset_index()

loan_metadata_matrix_group.sort_values(["Null_Percentage"], axis=0,ascending=False, inplace=True)
plot_pie_chat(loan_metadata_matrix_group,"Null_Percentage","Null_Count")
missing_data_100_precent = loan_metadata_matrix[loan_metadata_matrix["Null_Percentage"] == 100.0]

drop_missing_value_col = missing_data_100_precent.index.tolist()

print("Null Columns before deleting  : " + str(loan_df.shape[1]))

loan_df.drop(drop_missing_value_col,inplace=True,axis=1)

print("Null Columns after deleting : " + str(loan_df.shape[1]))
loan_metadata_matrix = get_meta_data(loan_df)

loan_metadata_matrix_group = loan_metadata_matrix.groupby("Null_Percentage").count().reset_index()

plot_pie_chat(loan_metadata_matrix_group,"Null_Percentage","Null_Count")
missing_data_greater_75_precent = loan_metadata_matrix[(loan_metadata_matrix["Null_Percentage"] > 75.0)]

drop_missing_value_col_75 = missing_data_greater_75_precent.index.tolist()

loan_df.drop(drop_missing_value_col_75,inplace=True,axis=1)
print("Shape after deleting 75% columns ",loan_df.shape ,"rows & columns.")
unique_value = loan_df.nunique()

col_with_only_one_value = unique_value[unique_value.values == 1]

col_to_drop = col_with_only_one_value.index.tolist()

loan_df.drop(col_to_drop, axis =1, inplace=True)

print("Shape after deleting unique value columns ",loan_df.shape ,"rows & columns.")
col_to_drop = ["url","desc","zip_code","id","member_id"]

loan_df.drop(col_to_drop,inplace=True,axis=1)

loan_df.head(5)
loan_df.shape
loan_data_type = get_meta_data(loan_df)

loan_data_type_float = loan_data_type[loan_data_type["Datatype"] == "float64"]

loan_data_type_int = loan_data_type[loan_data_type["Datatype"] == "int64"]
loan_data_type_float_group = loan_data_type_float.groupby("Null_Percentage").count().reset_index()

plot_pie_chat(loan_data_type_float_group,"Null_Percentage","Null_Count")

loan_data_type_float
columns_to_convert_numeric = loan_data_type_float[loan_data_type_float["Null_Count"] == 0].index.tolist()
loan_data_type_int_group = loan_data_type_int.groupby("Null_Percentage").count().reset_index()

plot_pie_chat(loan_data_type_int_group,"Null_Percentage","Null_Count")

loan_data_type_int
# adding more columns which needs to be converted to float

columns_to_convert_numeric.extend(["loan_amnt","funded_amnt"])

columns_to_convert_numeric
loan_df[columns_to_convert_numeric] = loan_df[columns_to_convert_numeric].astype(float)



# Converting Interest Rate Column into Floating , which will be used during 

#further analysis , converting str field to float 

loan_df["int_rate"] = loan_df["int_rate"].apply(lambda x: x.rstrip("%")).astype(float)





loan_df.head(2)
columns_need_cleaning = get_meta_data(loan_df)

columns_need_cleaning = columns_need_cleaning[columns_need_cleaning["Null_Count"] > 0]

columns_need_cleaning.head(20)
# not required at the moment for analysis and has 64% null values which cannot be substitued 

to_drop = ["mths_since_last_delinq","emp_title"]

loan_df.drop(to_drop , inplace=True,axis=1)
loan_df_temp = loan_df.filter(columns_need_cleaning.index.tolist())

loan_df_temp.head()
print(loan_df["emp_length"].unique())

loan_df["emp_length"].fillna('Self-Employed',inplace=True)

print(loan_df.emp_length.unique())
# Adding frequnecy columns so as to use it while plotting graphs

loan_df["frequency"] = loan_df["loan_amnt"] - loan_df["loan_amnt"]

len(loan_df["frequency"].astype(int))
loan_df['issue_month']  = loan_df['issue_d'].apply(lambda date:date.split('-')[0])

loan_df['issue_year'] = loan_df['issue_d'].apply(validate_year)

loan_df[['issue_d','issue_month','issue_year']].head()
loan_df['loan_amt_inc_ratio']=loan_df['loan_amnt']/loan_df['annual_inc']

loan_df[['loan_amt_inc_ratio','loan_amnt','annual_inc']].head()
loan_df["loan_amnt"].describe()
plot_box_chart(pd.DataFrame(loan_df["loan_amnt"])) # hover cursor above graph
plot_box_chart(pd.DataFrame(loan_df["int_rate"])) # hover cursor above graph
plot_box_chart(pd.DataFrame(loan_df["annual_inc"])) # hover cursor above graph
loan_df["annual_inc"].describe()
outlier_value = loan_df["annual_inc"].quantile(0.995)

loan_df = loan_df[loan_df["annual_inc"] < outlier_value]

loan_df["annual_inc"].describe()
plot_box_chart(pd.DataFrame(loan_df["annual_inc"]))  # hover cursor above graph
loan_df.head()
plot = loan_df.groupby("emp_length").frequency.count().reset_index()

plot_bar_chat(plot,"emp_length","frequency")
plot_home_ownership = loan_df.groupby("home_ownership").frequency.count().reset_index()

plot_bar_chat(plot_home_ownership,"home_ownership","frequency")
loan_df = loan_df[loan_df["home_ownership"].isin(["OWN","RENT","MORTGAGE"]) ]

plot_home_ownership_1 = loan_df.groupby("home_ownership").frequency.count().reset_index()

plot_bar_chat(plot_home_ownership_1,"home_ownership","frequency")
plot_verification_status = loan_df.groupby("verification_status").frequency.count().reset_index()

plot_bar_chat(plot_verification_status,"verification_status","frequency")
loan_df["verification_status"] = loan_df["verification_status"].replace("Source Verified","Verified")

plot_verification_status = loan_df.groupby("verification_status").frequency.count().reset_index()

plot_bar_chat(plot_verification_status,"verification_status","frequency")
plot_loan_status = loan_df.groupby("loan_status").frequency.count().reset_index()

plot_pie_chat(plot_loan_status,"loan_status","frequency")
plot_purpose = loan_df.groupby("purpose").frequency.count().reset_index()

plot_bar_chat(plot_purpose,"purpose","frequency")
plot_issue_year = loan_df.groupby("issue_year").frequency.count().reset_index()

plot_bar_chat(plot_issue_year,"issue_year","frequency")
loan_plot = loan_df[["purpose","loan_status"]]

plot_group_bar_chart(loan_plot,"purpose","loan_status")
loan_plot = loan_df[["home_ownership","loan_status"]]

plot_group_bar_chart(loan_plot,"home_ownership","loan_status")
loan_plot = loan_df[["term","loan_status"]]

plot_group_bar_chart(loan_plot,"term","loan_status")
loan_df.columns
# picking relevent (contineous variable) fields to check if they are corelated , 

col_for_corr = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv','int_rate',

                'installment','annual_inc', 'dti', 'out_prncp', 'out_prncp_inv',

                'loan_amt_inc_ratio']

loan_corr = loan_df.filter(col_for_corr)
corr = loan_corr.corr(method ='pearson')

corr
plt.subplots(figsize=(18, 12))

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,annot= True)

plt.show()
plt.figure(figsize=(20,16))

sns.boxplot(data =loan_df, x='loan_amnt', y='purpose', hue ='loan_status')



plt.show()
loan_plot = loan_df[["emp_length","loan_status"]]

plot_group_bar_chart(loan_plot,"emp_length","loan_status")


# Employment Length vs Loan Amount for different pupose of Loan



loanstatus=loan_df.pivot_table(index=['loan_status','purpose','emp_length'],values='loan_amnt',aggfunc=('count')).reset_index()

loanstatus=loan_df.loc[loan_df['loan_status']=='Charged Off']



ax = plt.figure(figsize=(35, 18))

ax = sns.boxplot(x='emp_length',y='loan_amnt',hue='purpose',data=loanstatus)

ax.set_title('Employment Length vs Loan Amount for different pupose of Loan',fontsize=22,weight="bold")

ax.set_xlabel('Employment Length',fontsize=16)

ax.set_ylabel('Loan Amount',color = 'b',fontsize=16)

plt.show()
loan_df.head()
def calculate_defaulter_precentage(dataframe,column) :

    def_tab = pd.crosstab(dataframe[column], dataframe['loan_status'],margins=True)

    def_tab['All'] = def_tab['Charged Off'] + def_tab['Current'] + def_tab['Fully Paid']

    def_tab['Loan Default Probability'] = round((def_tab['Charged Off']/def_tab['All']),3)

    def_tab = def_tab[0:-1] # removing last row with sum totol 

    return def_tab
def plot_bar_line_chart(dataframe,column,stacked=False):

    

    plot = calculate_defaulter_precentage(dataframe,column)

    

    display(plot)

    

    #initializing line plot

    linePlot = plot[['Loan Default Probability']] 

    line = linePlot.plot(figsize=(20,8), marker='o',color = 'r',lw=2)

    line.set_title(dataframe[column].name.title()+' vs Loan Default Probability',fontsize=20,weight="bold")

    line.set_xlabel(dataframe[column].name.title(),fontsize=14)

    line.set_ylabel('Loan Default Probability',color = 'r',fontsize=20)

    

    #initializing bar plot

    barPlot =  plot.iloc[:,0:3] 

    bar = barPlot.plot(kind='bar',ax = line,rot=1,secondary_y=True,stacked=stacked)

    bar.set_ylabel('Number of Applicants',color = 'r',fontsize=20)

    

    plt.show()
plot_bar_line_chart(loan_df,"emp_length")
plot_bar_line_chart(loan_df,"grade")
plot_bar_line_chart(loan_df,"sub_grade",True)
plot_bar_line_chart(loan_df,"purpose")
# Removing  Address state those having vary small value, which will impact the Probability Analysis



filter_states = loan_df.addr_state.value_counts()

filter_states = filter_states[(filter_states < 10)]



loan_filter_states = loan_df.drop(labels = loan_df[loan_df.addr_state.isin(filter_states.index)].index)



plot_bar_line_chart(loan_filter_states,"addr_state",True)
# Create Bins for range of Loan Amount



bins = [0, 5000, 10000, 15000, 20000, 25000,40000]

slot = ['0-5000', '5000-10000', '10000-15000', '15000-20000', '20000-25000','25000 and above']

loan_df['loan_amnt_range'] = pd.cut(loan_df['loan_amnt'], bins, labels=slot)



# Create Bins for range of Annual Income



bins = [0, 25000, 50000, 75000, 100000,1000000]

slot = ['0-25000', '25000-50000', '50000-75000', '75000-100000', '100000 and above']

loan_df['annual_inc_range'] = pd.cut(loan_df['annual_inc'], bins, labels=slot)



# Create Bins for range of Interest rates



bins = [0, 7.5, 10, 12.5, 15,20]

slot = ['0-7.5', '7.5-10', '10-12.5', '12.5-15', '15 and above']

loan_df['int_rate_range'] = pd.cut(loan_df['int_rate'], bins, labels=slot)



loan_df.head()
plot_bar_line_chart(loan_df,"annual_inc_range",stacked=True)
plot_bar_line_chart(loan_df,"int_rate_range",stacked=True)
plot_bar_line_chart(loan_df,"loan_amnt_range",stacked=True)