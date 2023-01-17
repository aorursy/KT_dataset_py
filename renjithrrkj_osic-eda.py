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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

import pydicom
from pydicom.data import get_testdata_files
train=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
train
test.head()
smokers=train.loc[train.SmokingStatus=='Currently smokes']
train.info()
sns.pairplot(train)
sns.pairplot(train,hue='Sex')
sns.pairplot(train,hue='SmokingStatus')
sns.distplot(smokers['Age'])
sns.distplot(train['Age'])

df= train.groupby([train.Patient,train.Age,train.Sex, train.SmokingStatus])['Patient'].count()
df.index = df.index.set_names(['id','Age','Sex','SmokingStatus'])
df = df.reset_index()
df.rename(columns = {'Patient': 'freq'},inplace = True)
print(df.shape)
fig = px.bar(df, x='freq',y ='id',color='freq')
fig.update_layout(yaxis={'categoryorder':'total ascending'},title='No. of observations for each patient')
fig.update_yaxes(showticklabels=False)
fig.show()
plt.figure(figsize=[10,8])
plt.style.use('ggplot')
sns.countplot(x='Sex',data=train,hue='SmokingStatus')
train.corr()
sns.heatmap(train.corr(),annot=True)
fig = px.line(train.loc[train['Patient']=='ID00422637202311677017371'],x='Weeks',y='FVC')
fig.show()
fig = px.line(train.loc[train['Patient']=='ID00426637202313170790466'],x='Weeks',y='FVC')
fig.show()
p1=train.loc[train['Patient']=='ID00426637202313170790466']
p2=train.loc[train['Patient']=='ID00007637202177411956430']
p3=train.loc[train['Patient']=='ID00355637202295106567614']
pati=pd.concat([p1,p2,p3])
px.line(pati,x='Weeks',y='FVC',line_group='Patient',color='Patient')




PathDicom = '/kaggle/input/osic-pulmonary-fibrosis-progression/'
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))
print(lstFilesDCM[0])
RefDs = pydicom.dcmread(lstFilesDCM[4])
print(RefDs)
print('Patient id is {}'.format(RefDs.PatientID))
print('Sex..........{}'.format(RefDs.PatientSex))
print('Image Position {}'.format(RefDs.ImagePositionPatient))
print('Image Orientation {}'.format(RefDs.ImageOrientationPatient))
def MakeDF(lst):
    dictDf={}
    refd=pydicom.dcmread(lst[0])
    dictDf['Patient']=[refd.PatientID]
    dictDf['rows']=[refd.Rows]
    for i in range(1,len(lst)):
        refd=pydicom.dcmread(lst[i])
        dictDf['Patient'].append(refd.PatientID)
        dictDf['rows'].append(refd.Rows)
        #print(dictDf)
    return(dictDf)    
pd_dict=MakeDF(lstFilesDCM)
df=pd.DataFrame.from_dict(pd_dict)
df.head()
plt.figure(figsize=[10,10])
plt.imshow(RefDs.pixel_array, cmap=plt.cm.bone)
plt.show()
