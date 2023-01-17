# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import collections
from functools import reduce 
import operator
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
# Import Normalizer
from sklearn.preprocessing import Normalizer
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import StandardScaler
pd.set_option('max_colwidth',40)
pd.set_option('display.max_colwidth', -1)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
response = pd.read_csv('../input/freeFormResponses.csv', nrows=10000) #
response.style.set_properties(subset=['text'], **{'width': '600px'})
resplist= list(response.columns.values)
#print(resplist) # 35 column values
response.head(1)
tempA = pd.read_csv('../input/multipleChoiceResponses.csv', usecols=range(1,10))
demograph=tempA  
demograph.Q7.fillna('Other',inplace=True)
S1 = demograph[demograph['Q1'] == 'Male'].Q7
S2 = demograph[demograph['Q1'] == 'Female'].Q7
Company_Female = pd.DataFrame({'Female':S2})
Company_Male = pd.DataFrame({'Male': S1})
Company_Female.fillna('Other', inplace=True)
Company_Male.fillna('Other', inplace=True)
len_f = len(Company_Female)
len_m = len(Company_Male)
result_f=[]
result_m = []
for i in range(len_f):
    result_f.append(Company_Female.Female.iloc[i])
c1 = collections.Counter(result_f)
catagory = c1.keys()
for i in range(len_m):
    result_f.append(Company_Male.Male.iloc[i])
c2 = collections.Counter(result_f)
list_c = []
for i, value in enumerate(catagory):
    list_c.append({'Female':c1[value]/(len_f+len_m), 'Male':c2[value]/(len_f+len_m), 'Category of Industries': value})
list_company = pd.DataFrame(list_c)
list_company.head()
sns.set(rc={"figure.figsize": (14, 6)})
sns.set_style("white")
list_company.plot(x="Category of Industries", y=["Female", "Male"], kind="bar")
plt.title('Female/Male ratio in different industries based on total number of candidates', fontsize = 16)
plt.ylabel('Percentage ratio ', fontsize=14)
plt.xlabel('Category of Industries ', fontsize=14)
tempA = pd.read_csv('../input/multipleChoiceResponses.csv', usecols=range(1,12))
demograph=tempA  
tempA.head()
demograph.Q8.fillna('Other',inplace=True)
S1 = demograph[demograph['Q1'] == 'Male'].Q8
S2 = demograph[demograph['Q1'] == 'Female'].Q8
Company_Female = pd.DataFrame({'Female':S2})
Company_Male = pd.DataFrame({'Male': S1})
Company_Female.fillna('Other', inplace=True)
Company_Male.fillna('Other', inplace=True)
len_f = len(Company_Female)
len_m = len(Company_Male)
result_f=[]
result_m = []
for i in range(len_f):
    result_f.append(Company_Female.Female.iloc[i])
c1 = collections.Counter(result_f)
catagory = c1.keys()
for i in range(len_m):
    result_f.append(Company_Male.Male.iloc[i])
c2 = collections.Counter(result_f)
list_c = []
for i, value in enumerate(catagory):
    list_c.append({'Female':c1[value]/(len_f+len_m), 'Male':c2[value]/(len_f+len_m), 'Category of Industries': value})
list_company = pd.DataFrame(list_c)
list_company.head()

sns.set(rc={"figure.figsize": (14, 6)})
sns.set_style("white")
list_company.plot(x="Category of Industries", y=["Female", "Male"], kind="bar")
plt.title('Female/Male ratio based on experience', fontsize = 16)
plt.ylabel('Percentage ratio ', fontsize=14)
plt.xlabel('Experience Level ', fontsize=14)
all = pd.read_csv('../input/multipleChoiceResponses.csv', nrows=100)
alllist= list(all.columns.values)
#print(alllist)

def column_arrange(part_str, mat, i, rnum):
    survlist=[]
    for j in range(1,rnum):
        label =  part_str +str(j)
        if (mat[label].iloc[i] == 0):
            continue
        else:
            survlist.append(mat[label].iloc[i])
    return survlist
multchoice1 = pd.read_csv('../input/multipleChoiceResponses.csv', usecols=range(1,22))
rowindex = multchoice1[ ((multchoice1['Q6']=='Data Scientist') | (multchoice1['Q6']=='Data Analyst') |\
          (multchoice1['Q6']=='Data Engineer') |  (multchoice1['Q6']=='Software Engineer') |\
          (multchoice1['Q6']=='Research Scientist')|(multchoice1['Q6']=='Research Assistant') ) & (multchoice1['Q7']!='I am a student') ].index
analystmat1=multchoice1.loc[(rowindex)]        
analystmat1.fillna(0, inplace=True)
############################################################################
multchoice2 = pd.read_csv('../input/multipleChoiceResponses.csv', usecols=range(22,45))
analystmat2=multchoice2.loc[(rowindex)] 
analystmat2.fillna(0, inplace=True)
##########################################################################
multchoice3 = pd.read_csv('../input/multipleChoiceResponses.csv', usecols=range(45,65))
analystmat3=multchoice3.loc[(rowindex)]  
analystmat3.fillna(0, inplace=True)
#########################################################################
multchoice4 = pd.read_csv('../input/multipleChoiceResponses.csv', usecols=range(65,84))
analystmat4=multchoice4.loc[(rowindex)]           
analystmat4.fillna(0, inplace=True)
########################################################################
multchoice5 = pd.read_csv('../input/multipleChoiceResponses.csv', usecols=range(84,108))
analystmat5=multchoice5.loc[(rowindex)]           
analystmat5.fillna(0, inplace=True)
#######################################################################
multchoice6 = pd.read_csv('../input/multipleChoiceResponses.csv', usecols=range(108, 128))
analystmat6=multchoice6.loc[(rowindex)]           
analystmat6.fillna(0, inplace=True)
######################################################################
multchoice7 = pd.read_csv('../input/multipleChoiceResponses.csv', usecols=range(128,151))
analystmat7=multchoice7.loc[(rowindex)]  
analystmat7.fillna(0, inplace=True)
#####################################################################
multchoice8 = pd.read_csv('../input/multipleChoiceResponses.csv', usecols=range(151,195))
analystmat8=multchoice8.loc[(rowindex)]           
analystmat8.fillna(0, inplace=True)
####################################################################
multchoice9 = pd.read_csv('../input/multipleChoiceResponses.csv', usecols=range(195, 224))
analystmat9=multchoice9.loc[(rowindex)]           
analystmat9.fillna(0, inplace=True)
#####################################################################
multchoice10 = pd.read_csv('../input/multipleChoiceResponses.csv', usecols=range(224,250))
analystmat10=multchoice10.loc[(rowindex)]           
analystmat10.fillna(0, inplace=True)
####################################################################
multchoice11 = pd.read_csv('../input/multipleChoiceResponses.csv', usecols=range(250,277))
analystmat11=multchoice11.loc[(rowindex)]           
analystmat11.fillna(0, inplace=True)
###################################################################
multchoice12 = pd.read_csv('../input/multipleChoiceResponses.csv', usecols=range(277,305))
analystmat12=multchoice12.loc[(rowindex)]           
analystmat12.fillna(0, inplace=True)
##################################################################
multchoice13 = pd.read_csv('../input/multipleChoiceResponses.csv', usecols=range(305,330))
analystmat13=multchoice13.loc[(rowindex)]           
analystmat13.fillna(0, inplace=True)
###################################################################
multchoice14 = pd.read_csv('../input/multipleChoiceResponses.csv', usecols=range(330,356))
analystmat14=multchoice14.loc[(rowindex)]           
analystmat14.fillna(0, inplace=True)
####################################################################
multchoice15 = pd.read_csv('../input/multipleChoiceResponses.csv', usecols=range(356,395))
analystmat15=multchoice15.loc[(rowindex)]           
analystmat15.fillna(0, inplace=True)

del multchoice1 
del multchoice2 
del multchoice3 
del multchoice4 
del multchoice5 
del multchoice6 
del multchoice7 
del multchoice8 
del multchoice9 
del multchoice10 
del multchoice11 
del multchoice12 
del multchoice13 
del multchoice14 
del multchoice15
numanal=len(analystmat4)
survlist1 = []
for i in range(numanal):
    param1 = column_arrange('Q11_Part_', analystmat1,i,7)
    param2 = column_arrange('Q13_Part_', analystmat2,i,15)
    param3 = column_arrange('Q14_Part_', analystmat3,i,11)
    param4 = column_arrange('Q15_Part_', analystmat3,i,7)
    param5 = column_arrange('Q16_Part_', analystmat4,i,18)
    param6 = column_arrange('Q19_Part_', analystmat5,i,19)
    param7 = column_arrange('Q21_Part_', analystmat6,i,13)
    param8 = column_arrange('Q27_Part_', analystmat7,i,20)
    param9 = column_arrange('Q28_Part_', analystmat8,i,43)
    param10 = column_arrange('Q29_Part_', analystmat9,i,28)
    param11 = column_arrange('Q30_Part_', analystmat10,i,25)
    param12 = column_arrange('Q31_Part_', analystmat11,i,12)
    param13 = column_arrange('Q33_Part_', analystmat11,i,11)
    param17 = column_arrange('Q36_Part_', analystmat12,i,13)
    param18 = column_arrange('Q38_Part_', analystmat13,i,22)
    param19 = column_arrange('Q39_Part_', analystmat14,i,2)
    param20 = column_arrange('Q41_Part_', analystmat14,i,3)
    param21 = column_arrange('Q42_Part_', analystmat14,i,5)
    param22 = column_arrange('Q44_Part_', analystmat14,i,6)
    param23 = column_arrange('Q45_Part_', analystmat14,i,6)
    param24 = column_arrange('Q47_Part_', analystmat15,i,16)
    param25 = column_arrange('Q49_Part_', analystmat15,i,12)
    param26 = column_arrange('Q50_Part_', analystmat15,i,8)
    survlist1.append({'role_work': param1, 'IDE_used': param2,'NB_used': param3, \
                    'Cloud_Serv': param4, 'Prog_Lang': param5, 'ML_Framework': param6,'Visual_Lib': param7, 'Cloud_prod': param8,\
                    'Cloud_ML': param9, 'RDBMS': param10, 'Big_Data_Prod': param11, 'data_types': param12, \
                    'public_datasets': param13,\
                    'online_learn': param17, \
                    'media_sources': param18, 'quality_learning': param19, 'important_topic': param20, \
                    'metrics': param21, 'algorithms_check': param22,\
                    'insights_interpret': param23, 'explain_interprete': param24, 'reproduce_work': param25, \
                    'barriers': param26})
multchoice = pd.DataFrame(survlist1)
#multchoice = multchoice.apply(lambda x: x.str.strip()).replace('', np.nan)
multchoice.algorithms_check = multchoice.algorithms_check.apply(lambda y: ['None'] if len(y)==0 else y)
multchoice.head(2)
del param1
del param2 
del param3 
del param4 
del param5
del param6 
del param7 
del param8 
del param9 
del param10 
del param11 
del param12 
del param13 
del param17 
del param18 
del param19 
del param20 
del param21 
del param22 
del param23 
del param24 
del param25 
del param26 
feature_multichoice = 'Big_Data_Prod Cloud_ML 	Cloud_Serv 	Cloud_prod 	IDE_used 	\
ML_Framework 	NB_used 	Prog_Lang 	RDBMS 	Visual_Lib 	algorithms_check 	\
barriers 	data_types 	explain_interprete 	important_topic 	\
insights_interpret 	media_sources 	metrics 	online_learn 	\
public_datasets 	quality_learning 	reproduce_work 	role_work '.split()
print(len(feature_multichoice))
def normalized_value(c,text, num):
    nv = c[text]/num
    return nv
def pie_plot_survey(x,y,title):
    sns.set(rc={"figure.figsize": (12, 8)})
    sns.set_style("white")
    cmap = plt.get_cmap('Spectral')
    colors = [cmap(i) for i in np.linspace(0, 1, 8)]
    #plt.pie(survIDE,labels=barriers, autopct='%.0f%%', shadow=True, colors=colors)
    patches, texts, autotexts = plt.pie(x,labels=y, autopct='%.0f%%', shadow=True, colors=colors)
    for i in range (len(y)):
        texts[i].set_fontsize(16)
        autotexts[i].set_fontsize(14)
    plt.suptitle(title, fontsize=20)
def bar_plot_surveydata(x,y,xlabel, ylabel,  width):
    sns.set(rc={"figure.figsize": (20, 6)})
    sns.set_style("white")
    sns.despine()
    plt.bar(x,y, width, color=sns.color_palette("Blues",3))
    sns.despine(top=True, right=True, left=False, bottom=False) 
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(direction='out', length=6, width=2, colors='k')
    plt.tick_params(axis='both', which='major', labelsize=14, rotation=10)
    plt.grid(True, which='major', axis='y', color="white", linewidth=1, zorder=1)
top_software1 = []
top_software2 = []
top_software3 = []
for j in range(23):
    AA = multchoice[feature_multichoice[j]].values
    result=[]
    for i in range(numanal):
        result += AA[i]
    role_work = np.unique(result)
    c1 = collections.Counter(result).most_common(3)
    count=0
    for i, values in enumerate(c1):
        count+=1
        if (count==1):
            top_software1.append(values[0])
        elif (count==2):
            top_software2.append(values[0])
        else:
            top_software3.append(values[0])
###################################################################################################
###create empty dataframe
#index_name = 'Cloud_ML 	Cloud_Serv 	Cloud_prod 	IDE_used 	\
#ML_Framework 	NB_used 	Prog_Lang 	RDBMS 	Visual_Lib 	algorithms_check 	\
#barriers 	data_types 	explain_interprete 	important_topic 	\
#insights_interpret 	media_sources 	metrics 	online_learn 	\
#public_datasets 	quality_learning 	reproduce_work 	role_work 	training_cata

index_name = 'Big_data_Product Cloud_ML	Cloud_Server	Cloud_product	 IDE 	\
ML_Framework	NB 	Programming_Language	RDBMS	Visual_Libaries Algorithms_check	Barriers	\
Data_Types Explain_interpretation_method	Topic_importance	Insights_interpret	Media_sources	\
Metrics	Online_learn	Public_datasets	quality_learning	Methods_Reproduce_work	Role_work'.split()
top_software = pd.DataFrame(columns=['top1','top2','top3'], index =index_name)
top_software.top1 = top_software1
top_software.top2 = top_software2
top_software.top3 = top_software3
top_software.style.set_properties(subset=['text'], **{'width': '600px'})
top_software.head(23)

timespent = analystmat12[['Q34_Part_1', 'Q34_Part_2', 'Q34_Part_3', 'Q34_Part_4', 'Q34_Part_5', \
                          'Q34_Part_6']] 
timespent.columns = ['Gathering_data', 'Cleaning_data', 'Visualizing_data', 'Model_build',\
                     'Model_production', 'Finding_insights']
sns.boxplot(data=timespent)
#sns.title('Time spend in projects')
timespent = analystmat12[['Q35_Part_1', 'Q35_Part_2', 'Q35_Part_3', 'Q35_Part_4', 'Q35_Part_5', \
                          'Q35_Part_6']] 
timespent.columns = ['Self-taught', 'Online courses', 'Work', 'University',\
                     'Kaggle competitions', 'Others']
sns.boxplot(data=timespent)
AA = multchoice.barriers.values
result=[]
for i in range(numanal):
    result += AA[i]
barriers = np.unique(result)
c1 = collections.Counter(result)
survIDE = []
for i in range(len(barriers)):
    survIDE.append(normalized_value(c1,barriers[i],numanal)*100)

#bar_plot_surveydata(barriers,survIDE,'Barriers to share work','Percentage population (%)')
pie_plot_survey(survIDE,barriers,'Barriers to share work')
AA = multchoice.barriers.values
resultrow=[]
for i in range(numanal):
    if 'Too time-consuming' in AA[i]:
        resultrow.append(i)
    else:
        continue
AA1 = (reduce(operator.concat,multchoice.loc[resultrow].reproduce_work))
AA2 = (reduce(operator.concat,multchoice.loc[resultrow].data_types))
AA3 = (reduce(operator.concat,multchoice.loc[resultrow].Cloud_Serv))
AA4 = (tempA.loc[resultrow].Q8)
barrier_reproduce_work=np.unique(AA1)
c1=collections.Counter(AA1)
survIDE = []
#print(barrier_reproduce_work)
for i in range(len(barrier_reproduce_work)):
    survIDE.append(normalized_value(c1,barrier_reproduce_work[i],len(resultrow))*100)
bar_plot_surveydata(['1','2','3','4','5','6','7','8','9','10','11'],\
                    survIDE,'Work Reproduction ','Percentage population (%)', 1/2.5)

data_types=np.unique(AA2)
c1=collections.Counter(AA2)
survIDE = []
for i in range(len(data_types)):
    survIDE.append(normalized_value(c1,data_types[i],len(resultrow))*100)
bar_plot_surveydata(data_types,survIDE,'Data type','Percentage population (%)',1/2.5)
barrier_Cloud_Serv=np.unique(AA3)
c1=collections.Counter(AA3)
survIDE = []
for i in range(len(barrier_Cloud_Serv)):
    survIDE.append(normalized_value(c1,barrier_Cloud_Serv[i],len(resultrow))*100)
bar_plot_surveydata(barrier_Cloud_Serv,survIDE,'Cloud Server','Percentage population (%)',1/2.5)