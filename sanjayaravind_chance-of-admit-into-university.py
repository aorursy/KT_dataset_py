# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
admission_data_v1=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
admission_data_v1.head()
admission_data_v1=admission_data_v1.drop(['Serial No.'],axis=1)
admission_data_v1
admission_data_v1.dtypes
f=plt.figure()
f.set_figheight(15)
f.set_figwidth(15)
color=['r','g','b','y','forestgreen','indigo','orangered','plum']
for i in range(0,len(admission_data_v1.columns)):
    plt.subplot(2,4,i+1)
    plt.title(admission_data_v1.columns[i])
    plt.hist(admission_data_v1[admission_data_v1.columns[i]],rwidth=0.9,color=color[i])
plt.tight_layout()
f=plt.figure()
f.set_figheight(15)
f.set_figwidth(15)
color=['r','g','b','y','forestgreen','indigo','orangered']
for i in range(0,len(admission_data_v1.columns)-1):
    plt.subplot(2,4,i+1)
    plt.title(''+admission_data_v1.columns[i]+' vs Chance of Admit')
    plt.xlabel(admission_data_v1.columns[i])
    plt.ylabel('Chance of Admit')
    plt.scatter(admission_data_v1[admission_data_v1.columns[i]],admission_data_v1[admission_data_v1.columns[len(admission_data_v1.columns)-1]],c=color[i])
plt.tight_layout()
groupby_data=gre_data.groupby('University Rating')
avg_gre_score=[]
avg_toefl_score=[]
avg_LOR=[]
avg_SOP=[]
avg_CGPA=[]
count_research_needed=[]
total_count=[]
for key,item in groupby_data:
    avg_gre_score.append(groupby_data.get_group(key)['GRE Score'].mean())
    avg_toefl_score.append(groupby_data.get_group(key)['TOEFL Score'].mean())
    avg_LOR.append(groupby_data.get_group(key)['LOR '].mean())
    avg_SOP.append(groupby_data.get_group(key)['SOP'].mean())
    avg_CGPA.append(groupby_data.get_group(key)['CGPA'].mean())
    count_research_needed.append(groupby_data.get_group(key)[groupby_data.get_group(key)['Research']==1]['Research'].count())
    total_count.append(groupby_data.get_group(key)['Research'].count())
avg_data={'University Rating':[1,2,3,4,5],'Avg GRE Score':avg_gre_score,'Avg TOEFL Score':avg_toefl_score,'Avg LOR':avg_LOR,'Avg SOP':avg_SOP,'Avg CGPA':avg_CGPA,'Count research':count_research_needed,'Total count':total_count}
avg_dataframe=pd.DataFrame(avg_data,columns=['University Rating','Avg GRE Score','Avg TOEFL Score','Avg LOR','Avg SOP','Avg CGPA','Count research','Total count'])
avg_dataframe
f=plt.figure()
f.set_figheight(15)
f.set_figwidth(10)
color=['r','g','b','y','forestgreen']
for i in range(1,6):
    plt.subplot(3,2,i)
    plt.title('University Rating vs '+avg_dataframe.columns[i])
    plt.xlabel('University Rating')
    plt.ylabel('Avg GRE Scores')
    if(avg_dataframe.columns[i]=='Avg LOR' or avg_dataframe.columns[i]=='Avg SOP' or avg_dataframe.columns[i]=='Avg CGPA'):
        plt.yticks(np.arange(math.floor(avg_dataframe[avg_dataframe.columns[i]].min()),math.ceil(avg_dataframe[avg_dataframe.columns[i]].max())))
    else:
        plt.yticks(np.arange(math.floor(avg_dataframe[avg_dataframe.columns[i]].min()),math.ceil(avg_dataframe[avg_dataframe.columns[i]].max()),3))
    plt.plot(avg_dataframe['University Rating'],avg_dataframe[avg_dataframe.columns[i]],c=color[i-1])
plt.tight_layout()
pie_colors=['r','g','b','y','forestgreen','indigo','orangered','aquamarine','beige','plum']
cats=['With Research Experience','Without Research Experience']
j=0
for i in range(0,5):
    count_cats=[count_research_needed[i],total_count[i]-count_research_needed[i]]
    fig, ax = plt.subplots()
    ax.set_title('Universities with Ratings '+str(i+1)+' vs Research Experience')
    ax.pie(count_cats,labels=cats,autopct='%1.1f%%',colors=[pie_colors[j],pie_colors[j+1]])
    ax.legend(cats,title='Research Exp',loc ="center left",bbox_to_anchor =(1, 0, 0.5, 1))
    j=j+2
plt.figure(figsize = (20,15))
sns.heatmap(admission_data_v1.corr(), annot = True, cmap="PuBuGn")
plt.show()
minmaxscaler=MinMaxScaler()
numerical_rows=admission_data_v1.columns[admission_data_v1.dtypes!='object']
admission_data_v1[numerical_rows]=minmaxscaler.fit_transform(admission_data_v1[numerical_rows])
admission_data_v1.head()
X=admission_data_v1.iloc[:, :-1]
y=admission_data_v1.iloc[:, -1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=80)
X_train.shape
linear_model=LinearRegression()
linear_model.fit(X_train,y_train)
y_pred = linear_model.predict(X_test)
MSE=mean_squared_error(y_test,y_pred)
RMSE=np.sqrt(MSE)
print('Root Mean squared error ',RMSE)
print('Mean squared error ',MSE)
model_score=linear_model.score(X_test,y_test)
print(model_score)
ax = sns.regplot(x=y_test, y=y_pred, color="g")
for i in range(0,len(X_test.columns)):
    print(X_test.columns[i],':',linear_model.coef_[i])
    
f=plt.figure()
f.set_figheight(8)
f.set_figwidth(10)
plt.bar(X_test.columns,linear_model.coef_)
plt.tight_layout()