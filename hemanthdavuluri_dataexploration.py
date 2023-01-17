import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('/kaggle/input/insurance/insurance.csv')
print("")
print("Data Frame Shape is ", data.shape)
print("")
data.head()
data.isnull().sum()
data.info()
data['sex']=data['sex'].astype('category')
data['smoker']=data['smoker'].astype('category')
data.info()
data.describe()
col_list = ["shit","pistachio"]
col_list_palette = sns.xkcd_palette(col_list)
sns.set_palette(col_list_palette)

sns.set(rc={"figure.figsize": (10,6)},
            palette = sns.set_palette(col_list_palette),
            context="talk",
            style="ticks")

# How many smokers are in this study?
sns.catplot(x = 'smoker', data=data, kind='count',height=5,aspect=1.5, edgecolor="black")
plt.title('Count of Smokers in Study', fontsize = 20)
print(data['smoker'].value_counts())
# Count of Smokers by Sex
sns.catplot(x ='sex',hue='smoker', data=data, kind="count",height=6, aspect=1.8,legend_out=False, edgecolor="black")
plt.suptitle('Count of Smokers by Gender', fontsize = 20)
plt.xlabel('Gender')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
print(pd.crosstab(data['smoker'],data['sex']))
male_data = data[data['sex']=='male']
female_data = data[data['sex']=='female']
print('male data shape'+ str(male_data.shape))
print('Female data shape'+ str(female_data.shape))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
langs = ['Male','Femlae']
students = [676,672]
ax.pie(students, labels = langs,autopct='%1.2f%%',colors = ["#1f77b4", "#ff7f0e"])
plt.title('Pie chart ')
plt.show()
print("Median value of Male age is:"+str(male_data['age'].mean()))
print("Median value of Femalemale age is:"+str(female_data['age'].mean()))
male_smoking_young = data[(data['age']<38) & (data['sex']=='male') & (data['smoker']=='yes')]
male_nonsmoking_young = data[(data['age']<38) & (data['sex']=='male') & (data['smoker']=='no')]
male_smoking_older = data[(data['age']>38) & (data['sex']=='male') & (data['smoker']=='yes')]
male_nonsmoking_older = data[(data['age']>38) & (data['sex']=='male') & (data['smoker']=='no')]

female_smoking_young = data[(data['age']<38) & (data['sex']=='female') & (data['smoker']=='yes')]
female_nonsmoking_young = data[(data['age']<38) & (data['sex']=='female') & (data['smoker']=='no')]
female_smoking_older =  data[(data['age']>38) & (data['sex']=='female') & (data['smoker']=='yes')]
female_nonsmoking_older = data[(data['age']>38) & (data['sex']=='female') & (data['smoker']=='no')]
                               
print('Younger boys who smoke and their avg spending on medical is = ' + str(male_smoking_young['charges'].mean()) +' $')
print("")
print('Younger boys who dont smoke and their avg spending on medical is = ' + str(male_nonsmoking_young['charges'].mean()) +' $')
print("")
print('Older men who smoke and their avg spending on medical is = ' + str(male_smoking_older['charges'].mean()) +' $')
print("")
print('Older men who dont smoke and their avg spending on medical is = ' + str(male_nonsmoking_older['charges'].mean()) +' $')
print("")
print('Younger gilrs who smoke and their avg spending on medical is ='+ str(female_smoking_young['charges'].mean())+' $')
print("")
print('Younger gilrs who dont smoke and their avg spending on medical is ='+ str(female_nonsmoking_young['charges'].mean())+' $')
print("")
print('Older gilrs who smoke and their avg spending on medical is ='+ str(female_smoking_older['charges'].mean())+' $')
print("")
print('Older gilrs who dont smoke and their avg spending on medical is ='+ str(female_nonsmoking_older['charges'].mean())+' $')
print("")
values = [(male_smoking_young['charges'].mean()),(male_nonsmoking_young['charges'].mean()),(male_smoking_older['charges'].mean()),(male_nonsmoking_older['charges'].mean())]
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
names = ['YoungSmokers', 'Young-non-smoker', 'old-Smoker', 'old-Non-smokers']
ax.set_ylabel('Average charges')
ax.set_title('Average medical charges of Male')
ax.bar(names,values,width = 0.5,color=["#2ca02c"])
plt.show()
values = [(female_smoking_young['charges'].mean()),(female_nonsmoking_young['charges'].mean()),(female_smoking_older['charges'].mean()),(female_nonsmoking_older['charges'].mean())]
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
names = ['YoungSmokers', 'Young-non-smoker', 'old-Smoker', 'old-Non-smokers']
ax.set_ylabel('Average charges')
ax.set_title('Average medical charges of Female')
ax.bar(names,values,width = 0.5,color=["#1f77b4"])
plt.show()
print(data['region'].unique())
southwest = data[data['region']=='southwest']
southeast = data[data['region']=='southeast']
northeast = data[data['region']=='northeast']
northwest = data[data['region']=='northwest']
values = [(southwest['charges'].mean()),(southeast['charges'].mean()),(northeast['charges'].mean()),(northwest['charges'].mean())]
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
names = ['South-West','South-East','North-east','North-West']
ax.set_ylabel('Average charges')
ax.set_title('Average medical charges based on region')
ax.bar(names,values,width = 0.5,color=["#9467bd"])
plt.show()
values = [(southwest['charges'].max()),(southeast['charges'].max()),(northeast['charges'].max()),(northwest['charges'].max())]
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
names = ['South-West','South-East','North-east','North-West']
ax.set_ylabel('Average charges')
ax.set_title('Maximum medical charges based on region')
ax.bar(names,values,width = 0.5,color=["#9467bd"])
plt.show()
values = [(southwest['charges'].min()),(southeast['charges'].min()),(northeast['charges'].min()),(northwest['charges'].min())]
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
names = ['South-West','South-East','North-east','North-West']
ax.set_ylabel('Average charges')
ax.set_title('Minimum medical charges based on region')
ax.bar(names,values,width = 0.5,color=["#8c564b"])
plt.show()
#------------------------------------------------------#
#####   Medical Costs & Body Mass Index by Smoker  #####
#------------------------------------------------------#

a = sns.lmplot(x="bmi", y="charges", col="smoker",
           hue="smoker", data=data, height = 6)
plt.suptitle('Medical Costs & Body Mass Index by Smoker')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
a.set_axis_labels('Body Mass Index', 'Medical Costs in Dollars')
#------------------------------#
####  binning the bmi data  ####
#------------------------------#

mnb = round(min(data.bmi)-1)
mxb = round(max(data.bmi)+2)
bm = np.linspace(mnb, mxb, num=11).astype(int)
data['bmi_binned'] = pd.cut(data['bmi'], bins = bm); data.head(10)
# Group by the bin and calculate averages
avg_bmi  = data.groupby('bmi_binned').mean()

#------------------------------------------#
####  Body Mass Index & Medical Costs   ####
#------------------------------------------#

plt.bar(avg_bmi.index.astype(str), avg_bmi['charges'], color='#3f9b0b')
plt.xticks(rotation = 75); plt.xlabel('Body Mass Index'); plt.ylabel('Average Medical Cost in Dollars')
plt.suptitle('Body Mass and Associated Medical Cost');
#--------------------------------------------------#
####  Age Distribution of Study Participants   #####
#--------------------------------------------------#

print('#####  Age Statistics  #######\n')
print('      Minimum     ',int(data['age'].min()))
print('      Average     ',int(data['age'].mean()))
print('      Median      ', int(np.median(data['age'])))
print('      Max         ',int(data['age'].max()))
print('   Std Deviation  ', int(data['age'].std()))
####  Body Mass Distribution of Study Participants   #####
#--------------------------------------------------------#

# Distribution of Body Mass
print('#####  Body Mass Statistics  #######\n')
print('      Minimum     ',int(data['bmi'].min()))
print('      Average     ',int(data['bmi'].mean()))
print('      Median      ', int(np.median(data['bmi'])))
print('      Max         ',int(data['bmi'].max()))
print('   Std Deviation  ', int(data['bmi'].std()))
print('################################################')
print('############     Report Summary    #############')
print('################################################\n')
print('Average Age')
print('--------------')
print('Smoker      %0.0f' % data[data['smoker']=='yes']['age'].mean())
print('Non-Smoker  %0.0f' %data[data['smoker']=='no']['age'].mean(),'\n')
print('Average Body Mass')
print('--------------------')
print('Smoker      %0.0f' % data[data['smoker']=='yes']['bmi'].mean())
print('Non-Smoker  %0.0f' % data[data['smoker']=='no']['bmi'].mean(),'\n')
print('Average Medical Cost by Gender')
print('-----------------------------')
print('Female      ${:,.0f}'.format(data[data['sex']=='female']['charges'].mean()))
print('Male        ${:,.0f}'.format(data[data['sex']=='male']['charges'].mean()), '\n')
print('Average Medical Cost by Smoker')
print('--------------------------------')
print('Non-Smoker   ${:,.0f}'.format(data[data['smoker']=='no']['charges'].mean()))
print('Smoker      ${:,.0f}'.format(data[data['smoker']=='yes']['charges'].mean()))

temp = {"no" : 0, "yes" : 1}
data.smoker = [temp[item] for item in data.smoker]

print('##############################################')
print('###########    Correlations       ############')
print('##############################################')
print('\n')
print('Cost to Smoker   ', round(data['charges'].corr(data['smoker'])*100,1),'%')
print('Cost to Age      ', round(data['charges'].corr(data['age'])*100,1),'%')
print('Cost to Body Mass ', round(data['charges'].corr(data['bmi'])*100,1),'%')
