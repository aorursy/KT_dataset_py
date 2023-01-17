%matplotlib inline  
import pandas as pd
datafile= 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
data = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.head()

data.describe()
data.boxplot('Age',by='Attrition')
AttributeName='EducationField'
gdata=data.groupby([AttributeName,'Attrition']).count().reset_index()
gdata
prev=''
r1=[]
yes=[]
no=[]
for index, row in gdata.iterrows():
    if prev != row[AttributeName]:
        r1.append(row[AttributeName])
        prev=row[AttributeName]
    if row['Attrition'] == 'Yes':
        yes.append(row['Age'])
    else:
        no.append(row['Age'])
total=[]
ypercent=[]
npercent=[]
for i in range(len(yes)):
    total.append(int(yes[i])+int(no[i]))
    ypercent.append(int(yes[i])*100/total[i])
    npercent.append(int(no[i])*100/total[i])
ndf={AttributeName:r1,'Yes':yes , 'No' :no}
kdf= {AttributeName:r1,'Yes':ypercent , 'No':npercent}
fdata=pd.DataFrame(ndf)
fdata
fdata.plot.bar(x=AttributeName,stacked=True)
ndata=pd.DataFrame(kdf)
ndata
ndata.plot.bar(x=AttributeName,stacked=True)