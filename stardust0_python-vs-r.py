import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import base64
import io
from scipy.misc import imread
import codecs
from IPython.display import HTML
from matplotlib_venn import venn2,venn2_circles

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
multi_response=pd.read_csv('../input/multipleChoiceResponses.csv',encoding='ISO-8859-1')
multi_response.shape
pd.set_option('display.max_columns',300)
multi_response.head(3)
conver_rates=pd.read_csv('../input/conversionRates.csv')
conver_rates.drop('Unnamed: 0',axis=1,inplace=True)
conver_rates.head()
multi_response['CompensationAmount']=multi_response['CompensationAmount'].str.replace(',','')
multi_response['CompensationAmount']=multi_response['CompensationAmount'].str.replace('-','')
response=multi_response.dropna(subset=['WorkToolsSelect'])
response=response.merge(conver_rates,left_on='CompensationCurrency',right_on='originCountry',how='left')
python_user=response[(response['WorkToolsSelect'].str.contains('Python'))&(~response['WorkToolsSelect'].str.contains('R'))]
R_user=response[(~response['WorkToolsSelect'].str.contains('Python'))&(response['WorkToolsSelect'].str.contains('R'))]
using_both=response[(response['WorkToolsSelect'].str.contains('Python'))&(response['WorkToolsSelect'].str.contains('R'))]
multi_response['LanguageRecommendationSelect'].value_counts()[:2].plot.bar(figsize = (15,10),width= .2)
plt.show()
index1=python_user['LanguageRecommendationSelect'].value_counts()[:5].index
value1=python_user['LanguageRecommendationSelect'].value_counts()[:5].values

index2=R_user['LanguageRecommendationSelect'].value_counts()[:5].index
value2=R_user['LanguageRecommendationSelect'].value_counts()[:5].values


layout= go.Layout(images= [dict(
        source= "https://images.plot.ly/language-icons/api-home/python-logo.png",
        xref= "paper",
        yref= "paper",
        x= .3,
        y= .39,
        sizex= .22,
        sizey= .22,
        xanchor= "right",
        yanchor= "bottom"
      ),
      dict(
        source= "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/R_logo.svg/1086px-R_logo.svg.png",
        xref= "paper",
        yref= "paper",
        x= .69,
        y= .6,
        sizex= .2,
        sizey= .2,
#         sizing= "stretch",
        opacity= 1,
        layer= "below"
      )],
     xaxis = dict( autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False),
    yaxis = dict( autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False), 
    title="What Python and R users recommend",
    annotations= [
            {
                "font": {
                    "size": 30
                },
                "showarrow": False,
                 "text": "",
                "x": 0.16,
                "y": 0.5
            },      
            {
                "font": {
                    "size": 30
                },
                "showarrow": False,
                "text": "",
                "x": 0.79,
                "y": 0.5}] )
fig=go.Figure(data= [
    {
      "values": value1,
      "labels": index1,
      "domain": {"x": [0, .48]},
      "name": "Recommends",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },     
    {
      "values": value2 ,
      "labels": index2,
      "text":"CO2",
      "textposition":"inside",
      "domain": {"x": [.54, 1]},
      "name": "Recommends",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],layout=layout)
plt.savefig('n.png')
py.iplot(fig)
fig,ax=plt.subplots(1,2,figsize=(19,9))
plt.title('gdg')
multi_response['JobSkillImportancePython'].value_counts().plot.pie(ax=ax[0],autopct='%1.1f%%',explode=[0.05,0,0],shadow=True)
ax[0].set_title('Python',fontsize = 25)
ax[0].set_ylabel('')
multi_response['JobSkillImportanceR'].value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',explode=[0,0.05,0],shadow=True)
ax[1].set_title('R',fontsize = 25)
ax[1].set_ylabel('')
fig.suptitle('Importance for Job',fontsize= 34,y=1.05,x =.51)
plt.show()

f,ax=plt.subplots(1,2,figsize=(18,8))
pd.Series([python_user.shape[0],R_user.shape[0],using_both.shape[0]],index=['Python','R','Both']).plot.bar(ax=ax[0],colors= '#BB3242',alpha= .7)
ax[0].set_title('Number of Users')

v= venn2(subsets = (python_user.shape[0],R_user.shape[0],using_both.shape[0]), set_labels = ('Python Users', 'R Users'))
plt.title('Venn Diagram for Users')

# Subset colors
v.get_patch_by_id('10').set_color('#4120A9')
v.get_patch_by_id('11').set_color('#3E004A')
v.get_patch_by_id('01').set_color('#1D5939')

# Subset alphas
v.get_patch_by_id('10').set_alpha(.8)
v.get_patch_by_id('01').set_alpha(.7)
v.get_patch_by_id('11').set_alpha(.7)

# Border styles
c = venn2_circles(subsets= (python_user.shape[0],R_user.shape[0],using_both.shape[0]), linestyle='solid')
c[0].set_ls('dashed')  # Line style
c[0].set_lw(2.0)       # Line width
plt.show()
py_salary=(pd.to_numeric(python_user['CompensationAmount'].dropna())*python_user['exchangeRate']).dropna()
py_salary=py_salary[py_salary<1000000]
R_salary=(pd.to_numeric(R_user['CompensationAmount'].dropna())*R_user['exchangeRate']).dropna()
R_salary=R_salary[R_salary<1000000]
both_salary=(pd.to_numeric(using_both['CompensationAmount'].dropna())*using_both['exchangeRate']).dropna()
both_salary=both_salary[both_salary<1000000]
total_salary=pd.DataFrame([py_salary,R_salary,both_salary])
total_salary=total_salary.transpose()
total_salary.columns=['Python','R','Both']
print('Median Salary of Python user:',total_salary['Python'].median())
print('Median Salary of R user:',total_salary['R'].median())
print('Median Salary of individual using both languages:',total_salary['Both'].median())
boxprops = dict(linestyle=':', linewidth=1)
medianprops = dict(linestyle='-.', linewidth=3)

#  bp = total_salary.boxplot(showfliers=True, showmeans=True,
#                  boxprops=boxprops, medianprops=medianprops)
sns.boxplot(data = total_salary,boxprops=boxprops, medianprops=medianprops)
plt.title('Compensation By Language',fontsize = 35)
fig=plt.gcf()
fig.set_size_inches(17,12)
plt.show()
p=python_user.copy()
r=R_user.copy()
p['WorkToolsSelect']='Python'
r['WorkToolsSelect']='R'
combined=pd.concat([p,r])
combined=combined.groupby(['CurrentJobTitleSelect','WorkToolsSelect'])['Age'].count().to_frame().reset_index()
combined.pivot('CurrentJobTitleSelect','WorkToolsSelect','Age').plot.barh(width=0.8,colors = ['#1D5939','#BB3242'],alpha = .8)
fig=plt.gcf()
fig.set_size_inches(10,15)
plt.title('Job Title vs Language Used',size=20)
plt.show()
combined=pd.concat([p,r])
combined=combined.groupby(['JobFunctionSelect','WorkToolsSelect'])['Age'].count().to_frame().reset_index()
combined.pivot('JobFunctionSelect','WorkToolsSelect','Age').plot.barh(width=0.8,colors = ['#1D5939','#BB3242'],alpha = .8)
fig=plt.gcf()
fig.set_size_inches(10,13)
plt.title('Job Description vs Language Used',fontsize = 35,x= -0.1,y= 1.05)
plt.show()
combined=pd.concat([p,r])
combined=combined.groupby(['Tenure','WorkToolsSelect'])['Age'].count().to_frame().reset_index()
combined.pivot('Tenure','WorkToolsSelect','Age').plot.barh(width=0.8,colors = ['#1D5939','#BB3242'],alpha = .8)
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.title('Experience vs Language Used',x = .3,y= 1.05)
plt.show()
combined=pd.concat([p,r])
combined=combined.groupby(['EmployerIndustry','WorkToolsSelect'])['Age'].count().to_frame().reset_index()
combined.pivot('EmployerIndustry','WorkToolsSelect','Age').plot.barh(width=0.8,colors = ['#1D5939','#BB3242'],alpha = .8)
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.title('Industry vs Language Used')
plt.show()