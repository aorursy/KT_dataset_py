import pandas as pd

pd.set_option('display.max_colwidth',5000)

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
a=pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

b=pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')

c=pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')

d=pd.read_csv('/kaggle/input/kaggle-survey-2019/survey_schema.csv')

#@title code



a4_dims = (21.7, 16.27)



fig, ax = plt.subplots(figsize=a4_dims)

z=a.groupby(['Q2'],as_index=False)['Q2'].agg({'count'}).reset_index()

z=z.iloc[:4,:]

total=sum(list(z.loc[:,'count']))



ax=sns.barplot(x='Q2',y='count', data=z)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format((height/total)*100),

            ha="center") 

plt.xlabel('genedr')

plt.ylabel('count')

plt.title('Distribution of gender in the survey')

ax.legend(ax.patches, list(z.Q2))



plt.show()

#@title code 

x=a.groupby(['Q2','Q4'],as_index=False)['Q2'].agg({'count'}).reset_index()

x.drop(x.index.max(),inplace=True,axis=0)

x1=x.groupby(['Q2'],as_index=False).agg({'count':'sum'})

x1.columns=['Q2','gp_wise_count']

x1

x2=x.merge(x1,on='Q2')

x2

x2['normalized_count']=x2.iloc[:,[2,3]].apply(lambda x:round((x[0]/x[1])*100,2),axis=1)

a4_dims = (21.7, 16.27)



fig, ax = plt.subplots(figsize=a4_dims)

ax=sns.barplot(x='Q2',y='normalized_count',hue='Q4',data=x2)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height+.5,

            '{:1.2f}%'.format(height),ha='center', va='center'            ) 

plt.xlabel('gender')

plt.title('Distribution of Qualification w.r.t Gender')

plt.ylabel('normalized count')

legend = ax.legend()

legend.texts[0].set_text("Bachleor's degree")# hack to remove the legend title

plt.show()
#@title code 

x=a.groupby(['Q2','Q1'],as_index=False)['Q2'].agg({'count'}).reset_index()

x.drop(x.index.max(),inplace=True,axis=0)

x1=x.groupby(['Q2'],as_index=False).agg({'count':'sum'})

x1.columns=['Q2','gp_wise_count']

x1

x2=x.merge(x1,on='Q2')

x2

x2['normalized_count']=x2.iloc[:,[2,3]].apply(lambda x:round((x[0]/x[1])*100,2),axis=1)

a4_dims = (25.7, 20.27)



fig, ax = plt.subplots(figsize=a4_dims)

ax=sns.barplot(x='Q2',y='normalized_count',hue='Q1',data=x2)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height+.1,

            '{:1.2f}%'.format(height),ha='center', va='center'            ) 

plt.xlabel('gender')

plt.title('Distribution of Qualification w.r.t Age')

plt.ylabel('normalized count')

legend = ax.legend()

legend.texts[0].set_text("18-21")# hack to remove the legend title

plt.show()
#@title code

import numpy as np

z=a.groupby(['Q3','Q2'],as_index=False)['Q3'].agg({'count'}).reset_index().pivot(columns='Q2', index='Q3').reset_index()

z.drop(z.index[20],inplace=True)

z.fillna(0,inplace=True)

z['total']=z.apply(lambda x:x[1]+x[2]+x[3]+x[4],axis=1)

#z=z.iloc[:,[0,1,2,3,4,6]]

def func(x):

  for i in range(0,5):

    x[i]=round((x[i]/x[5])*100,2)

  return x

z1=z.iloc[:,1:7].apply(lambda x:func(x),axis=1)

z2=z1.merge(z.iloc[:,0],left_on=z.index,right_on=z1.index)

z2.drop(z.loc[:,['key_0']],inplace=True,axis=1)

z=z2

l=['Algeria','Argentina','Australia','Austria','Bangladesh','Belarus','Belgium','Brazil','Canada','Chile','China','Colombia','Czech','Denmark','Egypt','France','Germany','Greece',

'HongKong','Hungary','India','Indonesia','Iran','Ireland','Israel','Italy','Japan','Kenya','Malaysia','Mexico','Morocco','Netherlands','NZ',

'Nigeria','Norway','Other','Pakistan','Peru','Philippine','Poland','Portugal','N_Korea','Romania','Russia','Saudi','Singapore','S_Africa','S_Korea','Spain','Sweden',

'Switzerland','Taiwan','Thailand','Tunisia','Turkey','Ukraine','Uk_NI','USA','Vietnam']



p={}

for i in range (0,len(l)):

   p[i]=l[i]

 





i=pd.DataFrame(p.values(),index=p.keys())

zz=z.merge(i,left_on=z.index,right_on=i.index)



a4_dims = (45.7, 35.27)



fig, ax = plt.subplots(figsize=a4_dims)

#z.plot(kind='bar', stacked=True)





configs = z.iloc[:,0]

N = len(configs)

ind = np.arange(N)





p1 = plt.bar(zz.iloc[:,-1], z.iloc[:,1], color='r')

p2 = plt.bar(zz.iloc[:,-1], z.iloc[:,2], bottom=z.iloc[:,1], color='b')

p3 = plt.bar(zz.iloc[:,-1], z.iloc[:,3], 

             bottom=np.array(z.iloc[:,1])+np.array(z.iloc[:,2]), color='g')

p4 = plt.bar(zz.iloc[:,-1], z.iloc[:,4],

             bottom=np.array(z.iloc[:,1])+np.array(z.iloc[:,2])+np.array(z.iloc[:,3]),

             color='c')

plt.title('Countrywise Stacked Barchat')

Glist=['Female',	'Male'	,'Prefer not to say'	,'Prefer to self-describe']

plt.legend(Glist, fontsize=12, ncol=4, framealpha=0, fancybox=True)

plt.show()
#@title code 

x=a.groupby(['Q2','Q6'],as_index=False)['Q2'].agg({'count'}).reset_index()

x.drop(x.index.max(),inplace=True,axis=0)

x1=x.groupby(['Q2'],as_index=False).agg({'count':'sum'})

x1.columns=['Q2','gp_wise_count']

x1

x2=x.merge(x1,on='Q2')

x2

x2['normalized_count']=x2.iloc[:,[2,3]].apply(lambda x:round((x[0]/x[1])*100,2),axis=1)

a4_dims = (21.7, 16.27)



zadc=['0-49 employees','50-249 employees', '250-999 employees',

        '1000-9,999 employees', '> 10,000 employees']



fig, ax = plt.subplots(figsize=a4_dims)

ax=sns.barplot(x='Q2',y='normalized_count',hue='Q6',data=x2,hue_order=zadc)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height+.2,

            '{:1.2f}%'.format(height),ha='center', va='center'            ) 

plt.xlabel('gender')

plt.title('Distribution of workplace w.r.t Workplace')

plt.ylabel('normalized count')

legend = ax.legend(bbox_to_anchor=(1.1, 1.05))

legend.texts[0].set_text("0-49")# hack to remove the legend title

plt.show()
#@title code 

x=a.groupby(['Q6','Q4'],as_index=False)['Q3'].agg({'count'}).reset_index()

x.drop(x.index.max(),inplace=True,axis=0)

x1=x.groupby(['Q6'],as_index=False).agg({'count':'sum'})

x1.columns=['Q6','gp_wise_count']

x1

x2=x.merge(x1,on='Q6')

x2

x2['normalized_count']=x2.iloc[:,[2,3]].apply(lambda x:round((x[0]/x[1])*100,2),axis=1)

a4_dims = (21.7, 16.27)



zadc=['0-49 employees','50-249 employees', '250-999 employees',

        '1000-9,999 employees', '> 10,000 employees']



fig, ax = plt.subplots(figsize=a4_dims)

ax=sns.barplot(x='Q6',y='normalized_count',hue='Q4',data=x2,order=zadc)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height+.2,

            '{:1.2f}%'.format(height),ha='center', va='center'            ) 

plt.xlabel('gender')

plt.title('Distribution of workplace w.r.t Qualification')

plt.ylabel('normalized count')

legend = ax.legend(bbox_to_anchor=(1.1, 1.05))

legend.texts[0].set_text("Bachelor's degree")# hack to remove the legend title

plt.show()
#@title code 

x=a.groupby(['Q2','Q10'],as_index=False)['Q2'].agg({'count'}).reset_index()

x.drop(x.index.max(),inplace=True,axis=0)

def less_levels(x):

  if(x =='$0-999' or x=='1,000-1,999'or x =='2,000-2,999' or x=='3,000-3,999' or x =='4,000-4,999'or x =='5,000-7,499' or x=='7,500-9,999'):

    x='0-9999'

    

  elif(x =='10,000-14,999' or x=='15,000-19,999' or x =='20,000-24,999' or x=='25,000-29,999'):

    x='10000-29999'

  elif(x =='30,000-39,999' or x=='40,000-49,999'):

    x='30000-49999'

  elif(x =='50,000-59,999' or x=='60,000-69,999'):

    x='50000-69999'

  elif(x =='70,000-79,999' or x=='80,000-89,999'or x=='90,000-99,999'):

    x='70000-99999'

  elif(x =='100,000-124,999' or x=='125,000-149,999'or x=='150,000-199,999'):

    x='100,000-299,999'

  elif(x =='200,000-249,999'or x=='250,000-299,999'):

    x='200,000-399,999'



  return(x)



x['Q10']=x.loc[:,'Q10'].apply(lambda x:less_levels(x))

#x=x.groupby(['Q2','Q10'],as_index=False)['Q2'].agg({'count':'sum'}).reset_index()







x1=x.groupby(['Q2'],as_index=False).agg({'count':'sum'})

x1.columns=['Q2','gp_wise_count']

x1

x2=x.merge(x1,on='Q2')

x2

x2['normalized_count']=x2.iloc[:,[2,3]].apply(lambda x:round((x[0]/x[1])*100,2),axis=1)

x2=x2.groupby(['Q2','Q10'],as_index=False).agg({'count':'sum','normalized_count':'sum'}).reset_index()



a4_dims = (21.7, 16.27)



zadc=['0-9999', '10000-29999','30000-49999', '50000-69999', '70000-99999',  '100,000-299,999', '200,000-399,999','300,000-500,000',

       '> $500,000']

fig, ax = plt.subplots(figsize=a4_dims)

ax=sns.barplot(x='Q2',y='normalized_count',hue='Q10',data=x2,hue_order=zadc)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height+.5,

            '{:1.2f}%'.format(height),ha='center', va='center'            ) 

plt.xlabel('gender')

plt.title('Distribution of Salary w.r.t Gender')

plt.ylabel('normalized count')

legend = ax.legend(bbox_to_anchor=(1.1, 1.05))

#legend.texts[0].set_text("0-9999")# hack to remove the legend title

plt.show()
#@title code 

x=a.groupby(['Q4','Q10'],as_index=False)['Q4'].agg({'count'}).reset_index()

x.drop(x.index.max(),inplace=True,axis=0)

def less_levels(x):

  if(x =='$0-999' or x=='1,000-1,999'or x =='2,000-2,999' or x=='3,000-3,999' or x =='4,000-4,999'or x =='5,000-7,499' or x=='7,500-9,999'):

    x='0-9999'

    

  elif(x =='10,000-14,999' or x=='15,000-19,999' or x =='20,000-24,999' or x=='25,000-29,999'):

    x='10000-29999'

  elif(x =='30,000-39,999' or x=='40,000-49,999'):

    x='30000-49999'

  elif(x =='50,000-59,999' or x=='60,000-69,999'):

    x='50000-69999'

  elif(x =='70,000-79,999' or x=='80,000-89,999'or x=='90,000-99,999'):

    x='70000-99999'

  elif(x =='100,000-124,999' or x=='125,000-149,999'or x=='150,000-199,999'):

    x='100,000-299,999'

  elif(x =='200,000-249,999'or x=='250,000-299,999'):

    x='200,000-399,999'



  return(x)



x['Q10']=x.loc[:,'Q10'].apply(lambda x:less_levels(x))

#x=x.groupby(['Q2','Q10'],as_index=False)['Q2'].agg({'count':'sum'}).reset_index()







x1=x.groupby(['Q4'],as_index=False).agg({'count':'sum'})

x1.columns=['Q4','gp_wise_count']

x1

x2=x.merge(x1,on='Q4')

x2

x2['normalized_count']=x2.iloc[:,[2,3]].apply(lambda x:round((x[0]/x[1])*100,2),axis=1)

x2=x2.groupby(['Q4','Q10'],as_index=False).agg({'count':'sum','normalized_count':'sum'}).reset_index()



a4_dims = (21.7, 16.27)



zadc=['0-9999', '10000-29999','30000-49999', '50000-69999', '70000-99999',  '100,000-299,999', '200,000-399,999','300,000-500,000',

       '> $500,000']

fig, ax = plt.subplots(figsize=a4_dims)

ax=sns.barplot(x='Q4',y='normalized_count',hue='Q10',data=x2)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height+.5,

            '{:1.2f}%'.format(height),ha='center', va='center'            ) 

plt.xlabel('gender')

plt.title('Distribution of Pay w.r.t Qualification')

plt.ylabel('normalized count')

legend = ax.legend(bbox_to_anchor=(1.1, 1.05))

#legend.texts[0].set_text("0-9999")# hack to remove the legend title

plt.show()