#importing required packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings("ignore")
%matplotlib inline
sns.set(style="whitegrid")
import missingno as msno
#Interactive
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from IPython.display import display, HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
 $('div.cell.code_cell.rendered.selected div.input').hide();
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" class="btn btn-primary" value="Click here to toggle on/off the raw code."></form>''')
#Reading data
storedata = pd.read_csv("../input/googleplaystore.csv")

print("\n First 7 rows in our dataset")
display(storedata.head(7))

print("\n Number of rows in our dataset = " + str(storedata.shape[0]))

#Last Updated to (Month, Year) to number
storedata['Last Updated'] = pd.to_datetime(storedata['Last Updated'],format='%B %d, %Y',errors='coerce').astype('str')

def split_mul(data):
    try:
        data=list(map(int,data.split('-')))
        return data[0]+(data[1]*12)+data[2]
    except:
        return "Nan"
storedata['Last Updated'] = [split_mul(x) for x in storedata['Last Updated']]

#Improve 'Android Ver' and 'Installs' representation
storedata["Android Ver"] = storedata["Android Ver"].str.split(n=1, expand=True)

def deal_with_abnormal_strings(data):
    data[data.str.isnumeric()==False]=-1
    data=data.astype(np.float32)
    return data

storedata.Installs = [x.strip().replace('+', '').replace(',','') for x in storedata.Installs]
storedata.Installs = deal_with_abnormal_strings(storedata.Installs)

storedata.Size = [x.strip().replace('M', '').replace(',','') for x in storedata.Size]

def convert_float(val):
    try:
        return float(val)
    except ValueError:
        try:
            val=val.split('.')
            return float(val[0]+'.'+val[1])
        except:
            return np.nan


storedata.head(7)
#Number of categories of apps in the store.....
def plot_number_category():
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 7)
    fig.autofmt_xdate()
    countplot=sns.categorical.countplot(storedata.Category,ax=ax)
    plt.show(countplot)

plot_number_category()

# Tabular representation
top_cat=storedata.groupby('Category').size().reset_index(name='Count').nlargest(6,'Count')
display(top_cat)



cat=top_cat.Category.tolist()
data_top6=storedata.groupby('Category')['Installs'].agg('sum').loc[cat].reset_index(name='Number_Installations')
data=storedata.groupby('Category')['Installs'].agg('sum').reset_index(name='Number_Installations')

#Comparing top 5 category on the basis of 'Installs'
def compare_6(data):
    fig = plt.figure(figsize=(12,4))
    title=plt.title('Comparing top 5 category on the basis of Installs')
    bar=sns.barplot(y=data['Category'],x=data['Number_Installations'])
    plt.show(bar)

#Comparing all categoryies on the basis of 'Installs'
def compare_all(data):
    fig = plt.figure(figsize=(12,7))
    title=plt.title('Comparing all categories on the basis of Installs')
    bar=sns.barplot(y=data['Category'],x=data['Number_Installations'])
    plt.show(bar)
    
compare_6(data_top6)


compare_all(data)
print('\nTabular Rep.Of Top 5 Number of Installation by Category')
display(data.nlargest(6,'Number_Installations'))
#features to use for correlation
corr_cat=['Rating','Reviews','Size','Installs','Current Ver','Android Ver','Last Updated']
for i in corr_cat:
    storedata[i]=storedata[i].apply(lambda x: convert_float(x)) #To get it compatible to check correlation

correlation = storedata[corr_cat].corr()

print("\n Correlation of Installs with other selected features ")
display(correlation['Installs'].sort_values(ascending=False))


#Correlation Heatmap 
f , ax = plt.subplots(figsize = (14,12))
title=plt.title('Correlation of Numeric Features with Installs',y=1,size=16)
heatmap=sns.heatmap(correlation,square = True,  vmax=0.8)
plt.show(heatmap)
install_sum_content=storedata.groupby('Content Rating')['Installs'].agg('sum').reset_index(name='Number_Installations')
app_sum_content=data=storedata.groupby('Content Rating')['Installs'].size().reset_index(name='Number_Apps')

def content_bar_sum(data):
    fig=plt.figure(figsize=(12,6))
    
    title=plt.title('Comparision of content ratings (Number of Installations)')
    content_bar = sns.barplot(x=data['Content Rating'],y=data['Number_Installations'])
    plt.show(content_bar)
    
def content_bar_count(data):
    fig=plt.figure(figsize=(12,6))
    
    title=plt.title('Comparision of content ratings (Number of Apps in Market)')
    content_bar = sns.barplot(x=data['Content Rating'],y=data['Number_Apps'])
    plt.show(content_bar)
    
content_bar_sum(install_sum_content)
content_bar_count(app_sum_content)

#Temporary dataframe with improved comparision metric for content rating
content=pd.DataFrame()
content['Content Rating'] = app_sum_content['Content Rating']
content['No_Installations/Total_Apps']=install_sum_content['Number_Installations']/app_sum_content['Number_Apps']
#Visualize content
figure=plt.figure(figsize=(12,7))
title=plt.title('Content Rating Comparision')
bar=sns.barplot(x=content['Content Rating'],y=content['No_Installations/Total_Apps'])
plt.show(bar)
install_sum_type=storedata.groupby('Type')['Installs'].agg('sum').reset_index(name='Number_Installations')

def type_bar_sum(data):
    fig=plt.figure(figsize=(12,6))
    
    title=plt.title('Comparision of  types (Number of Installations)')
    content_bar = sns.barplot(x=data['Type'],y=data['Number_Installations'])
    plt.show(content_bar)
type_bar_sum(install_sum_type)
storedata['Name_check']=['>2 words' if len(x.split())>2 else '<=2words' for x in storedata['App'] ]

data_install= storedata.groupby('Name_check')['Installs'].agg('sum').reset_index(name='Number_Installations')
data_apps= storedata.groupby('Name_check').size().reset_index(name='Number_Apps')


fig,axes = plt.subplots(figsize=(15,3),ncols=2, nrows=1)

title=axes[0].set_title("No. of Installations", y = 1.1)
title=axes[1].set_title("No of Apps", y = 1.1)

plot1=sns.barplot( x=data_install['Name_check'],y=data_install['Number_Installations'] , ax=axes[0])

plot2=sns.barplot( x=data_apps['Name_check'],y=data_apps['Number_Apps'] , ax=axes[1])

plt.show(fig)

# No. of installation / No. of apps

figure=plt.figure(figsize=(12,5))
title=plt.title("Installations/Total Apps", y = 1.0)
plot3=sns.barplot( x=data_apps['Name_check'],y=data_install['Number_Installations']/data_apps['Number_Apps'] ,palette=sns.color_palette(palette="Set1",n_colors=2,desat=.8))
plt.show(figure)
#Number of null values in each feature
storedata.isnull().sum()

#Visualising missing data
missing = msno.matrix(storedata.sample(250))
mheatmap=msno.heatmap(storedata)