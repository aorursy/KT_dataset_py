def CDG(df,univariate,filterss=None,xislog=False,colorss='b',framess=False,labelss=None):

    if filterss is None:

        filterss=np.ones(len(df.index), dtype=bool)

    Act_Series = (df.loc[filterss,:].groupby(univariate)[univariate].count() \

        /len(df.loc[filterss,:].index)).cumsum()

    plt.plot(Act_Series,color=colorss,label=labelss)

    plt.xlabel(univariate)

    if xislog==True:

        plt.xscale('log')

        plt.xlabel('log'+univariate)

    plt.plot([df.loc[filterss,:][univariate].max()]*2,[0,1],linestyle='dashed',color=colorss)

    plt.plot([df.loc[filterss,:][univariate].min()]*2,[0,1],linestyle='dashed',color=colorss)

    if framess==True:

        plt.axis([0, df[univariate].max(),0,1])

    return Act_Series



def lownum(df,field,type1,type2,col1,col2):

    group1=df.loc[type1,:].groupby(field)[field].count()/len(df.loc[type1,:].index)

    group2=df.loc[type2,:].groupby(field)[field].count()/len(df.loc[type2,:].index)

    group3=df.loc[type1,:].groupby(field)[field].count()

    group4=df.loc[type2,:].groupby(field)[field].count()

    

    combined=pd.concat([group1,group3,group2,group4],axis=1)

    combined.columns=[col1+'%',col1+' Count',col2+'%',col2+' Count']

    combined['RelDiff']=100*(combined[col2+'%']-combined[col1+'%'])/combined[col1+'%']

    combined['AbsDiff']=combined[col2+'%']-combined[col1+'%']

    display(combined)
import matplotlib.pyplot as plt #Plotting 

import numpy as np #Math needs

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



plt.style.use('fivethirtyeight')

plt.figure(figsize=(12,6))

np.random.seed(1007)

Gauss=np.append(np.random.normal(loc=10,scale=1,size=1000),np.random.normal(loc=20,scale=1,size=1000))

data = pd.DataFrame(Gauss,columns=['Number'])

ax1 = plt.subplot2grid((2, 4), (0, 0),colspan=2)

ax1.boxplot(data.values, vert=False)

ax1.set_title('Boxplot')

ax2 = plt.subplot2grid((2, 4), (1, 0),colspan=2)

ax2.set_title('CDG')

CDG(data,univariate='Number',colorss='b')

#plt.xlim([data.Number.min(),data.Number.max()])

#plt.tight_layout()

np.random.shuffle(Gauss)



ax3 = plt.subplot2grid((2, 4), (0, 2), colspan=2,rowspan=2)

ax3.plot([np.median(Gauss[n:]) for n in range(Gauss.shape[0])],range(Gauss.shape[0]),label='Median')

ax3.plot([np.percentile(Gauss[n:],25) for n in range(Gauss.shape[0])],range(Gauss.shape[0]),label='25th Percentile')

ax3.plot([np.percentile(Gauss[n:],75) for n in range(Gauss.shape[0])],range(Gauss.shape[0]),label='75th Percentile')

#plt.yscale('log')

#plt.ylim([0,2000])

ax3.legend(loc='best',framealpha=0.5)

ax3.set_title('Median Jitter')

plt.tight_layout()

plt.show()

import numpy as np #Math needs

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #Plotting 

from scipy.stats import gaussian_kde #KDE

np.random.seed(1015)

plt.figure(figsize=(12,6))

typeA = pd.DataFrame({'Type':['A']*1000, 'Number':np.random.normal(size=1000)})

typeB = pd.DataFrame({'Type':['B']*5, 'Number':np.random.normal(size=5)})

typeALL = pd.concat([typeA,typeB],ignore_index=True)



#Free Parameters needed

xs = np.linspace(-4,4,100)

grid_lambda = [0.2,0.4,0.6]

colors = ['r','g','b']

iterator = zip(grid_lambda, colors)          

densityA = gaussian_kde(typeALL[typeALL['Type']=='A']['Number'].values) 

densityB = gaussian_kde(typeALL[typeALL['Type']=='B']['Number'].values) 

#Hunt and shoot a decent decay parameter. Automated Optimisation is another Computational Expense

for grid_lambda, colors in iterator:

    plt.subplot(2,2,3)

    densityA.covariance_factor = lambda : grid_lambda

    densityA._compute_covariance()

    plt.plot(xs,densityA(xs),color=colors,label=str(grid_lambda))

    plt.subplot(2,2,4)

    densityB.covariance_factor = lambda : grid_lambda 

    densityB._compute_covariance()

    plt.plot(xs,densityB(xs),color=colors, label=str(grid_lambda))



#No parameters needed

plt.subplot(2,2,1)

CDG(typeALL, univariate='Number',filterss=(typeALL['Type']=='A'),framess=True)

plt.subplot(2,2,2)

CDG(typeALL, univariate='Number',filterss=(typeALL['Type']=='B'),colorss='r',framess=True)



#Dress up

plt.subplot(2,2,1)

plt.xlim([-5,5])

plt.title('N=1,000')

plt.ylabel('CDG')

plt.subplot(2,2,2)

plt.xlim([-4,4])

plt.title('N=5')

plt.subplot(2,2,3)

plt.plot([typeALL[typeALL['Type']=='A']['Number'].min()]*2,[0,1],linestyle='dashed',color='k')

plt.plot([typeALL[typeALL['Type']=='A']['Number'].max()]*2,[0,1],linestyle='dashed',color='k')

plt.xlim([-5,5])

plt.title('N=1,000')

plt.ylabel('KDE')

plt.legend(loc='best',framealpha=0.25)

plt.subplot(2,2,4)

plt.plot([typeALL[typeALL['Type']=='B']['Number'].max()]*2,[0,1],linestyle='dashed',color='k')

plt.plot([typeALL[typeALL['Type']=='B']['Number'].min()]*2,[0,1],linestyle='dashed',color='k')

plt.xlim([-4,4])

plt.title('N=5')

plt.legend(loc='best',framealpha=0.25)

plt.tight_layout()

plt.show()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #Basic

from IPython.display import display, HTML



traindf = pd.read_csv('../input/train.csv')

testdf = pd.read_csv('../input/test.csv')

total=[traindf,testdf]

df = pd.concat(total)

print(df.info())

display(df.describe()) #Hunt out fields like Fare that have already processed null values

display(df.head())

plt.show()
plt.figure(figsize=(12,6))

CDG(df,'Age',(df['Age'].notnull()),framess=True)

plt.plot([16,16],[0,1],linestyle='dashed',color='k',linewidth=3,alpha=0.5)

plt.annotate('XX.5', xy=(25, 0.4), xytext=(40, 0.4),

            arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('Inflection point \nindicates rapid change \nin density',xy=(16,0.12),xytext=(30,0.1),

            arrowprops=dict(facecolor='black',shrink=0.05))

plt.annotate('Decreasing steepness \nindicates gradual \ndecay in density',xy=(50,0.9),xytext=(40,0.6),

            arrowprops=dict(facecolor='black',shrink=0.05))

plt.title('Fig 1.1 Age Cumulative Distribution')

plt.ylabel('Cumulation')

plt.show()
plt.figure(figsize=(12,6))

CDG(df,'Fare',(df['Fare'].notnull()),framess=True)

plt.ylabel('Cumulation')

plt.title('Fig 1.2 Fare Cumulative Distribution')

plt.show()
plt.figure(figsize=(12,6))

CDG(df,'Fare',(df['Fare'].notnull()),True,framess=True)

plt.ylabel('Cumulation')

plt.annotate('Density \nSpike', xy=(8, 0.2), xytext=(20,0.4),arrowprops=dict(facecolor='black', shrink=0.05)),

plt.title('Fig 1.3 Log Fare Cumulative Distribution')

plt.show()
plt.figure(figsize=(12,6))

CDG(df,'Fare',(df['Fare']>=6),xislog=True,framess=True)

plt.ylabel('Cumulation')

plt.title('Fig 1.4 Bounded Log Fare Cumulative Distribution')

plt.show()
lownum(df,'Pclass',df['Age'].notnull(),df['Age'].isnull(),'Age is recorded','Age is Null')

lownum(df,'Sex',df['Age'].notnull(),df['Age'].isnull(),'Age is recorded','Age is Null')

lownum(df,'Embarked',df['Age'].notnull(),df['Age'].isnull(),'Age is recorded','Age is Null')

lownum(df,'Survived',df['Age'].notnull(),df['Age'].isnull(),'Age is recorded','Age is Null')

plt.figure(figsize=(12,6))

CDG(df,'Fare',(df['Age'].isnull()) & (df['Fare']>=6),True,'r',labelss='No Age Recorded')

CDG(df,'Fare',(df['Age'].notnull()) & (df['Fare']>=6),True,'g',labelss='Age Recorded')

#Population Baseline

CDG(df,'Fare',(df['Fare']>=6),True,'k',True,'Population Baseline')

plt.annotate('Density \nSpike', xy=(8, 0.56), xytext=(10,0.7),arrowprops=dict(facecolor='black', shrink=0.05)),

plt.legend(loc='best')

plt.ylabel('Cumulation')

plt.title('Fig 1.5 Log Fare Cumulation for Age Null comparison')

plt.show()
plt.figure(figsize=(12,6))

CDG(df,'Fare',((df['Fare']>=6) & (df['Sex']=='male')),True,'b',labelss='Male')

CDG(df,'Fare',((df['Fare']>=6) & (df['Sex']=='female')),True,'g',True,'Female')

plt.ylabel('Cumulation')

plt.legend(loc='best')

plt.title('Fig 1.6 Log Fare Cumulation Gender comparison')

plt.show()
plt.figure(figsize=(12,6))

CDG(df,'Age',filterss=(df['Age'].notnull()) & (df['Sex']=='male'),colorss='b',labelss='Male')

CDG(df,'Age',filterss=(df['Age'].notnull()) & (df['Sex']=='female'),colorss='g',framess=True,labelss='Female')

plt.ylabel('Cumulation')

plt.legend(loc='best')

plt.title('Fig 1.7 Age Cumulation Gender Comparison')

plt.show()
plt.figure(figsize=(12,6))

CDG(df,'Fare',(df['Fare']>=6) & (df['Pclass']==1),True,'b',labelss='P Class 1')

CDG(df,'Fare',(df['Fare']>=6) & (df['Pclass']==2),True,'g',labelss='P Class 2')

CDG(df,'Fare',(df['Fare']>=6) & (df['Pclass']==3),True,'r',labelss='P Class 3',framess=True)

plt.annotate('Pclass 3 \njaggedness', xy=(45, 0.975), xytext=(30, 0.7),

            arrowprops=dict(facecolor='black', shrink=0.05)),

plt.annotate('Pclass3 \nDensity \nSpike', xy=(8, 0.3), xytext=(13, 0.55),

            arrowprops=dict(facecolor='black', shrink=0.05)),

plt.annotate('Pclass2 \nDensity \nSpike', xy=(13, 0.3), xytext=(15, 0.05),

            arrowprops=dict(facecolor='black', shrink=0.05)),

plt.title('Fig 1.8 Log Fare Cumulation Pclass Comparison')

plt.ylabel('Cumulation')

plt.legend(loc='best',framealpha=0.5)

plt.show()
plt.figure(figsize=(12,6))

CDG(df,'Age',(df['Age'].notnull()) & (df['Pclass']==1),colorss='b',labelss='P Class 1')

CDG(df,'Age',(df['Age'].notnull()) & (df['Pclass']==2),colorss='g',labelss='P Class 2')

CDG(df,'Age',(df['Age'].notnull()) & (df['Pclass']==3),colorss='r',labelss='P Class 3')

CDG(df,'Age',(df['Age'].notnull()),colorss='k',labelss='Population Baseline',framess=True)

plt.ylabel('Cumulative Percentage (as Decimal)')

plt.legend(loc='best',framealpha=0.5)

plt.title('Fig 1.9 R+E Age Cumulation Pclass Comparison')

plt.show()
plt.figure(figsize=(12,6))

Act_Series = CDG(df,'Age',df['Age'].notnull(),colorss='k',labelss='Recorded Age Baseline')

plt.fill_between(np.sort(df[df['Age'].notnull()].Age.unique()),Act_Series.values,np.ones(len(Act_Series)),facecolor='g',alpha=0.5)

plt.plot([27.5,27.5],[0,1],color='k',linestyle='dashed')

plt.plot([0,80],[0.5]*2,color='k',linestyle='dashed')

plt.legend(loc='lower right')

plt.text(2,0.8,'Likely region \nfor non \nrecorded ages')

plt.text(1,0.52, '50%')

plt.ylabel('Cumulation')

plt.title('Fig 1.10 CDF Region of Likelihood')

plt.show()
plt.figure(figsize=(12,6))

CDG(df,'Fare',(df['Fare']>=6) & (df['Embarked']=='C'),True,'b',labelss='Cherbourg')

CDG(df,'Fare',(df['Fare']>=6) & (df['Embarked']=='S'),True,colorss='g',labelss='Southampton')

CDG(df,'Fare',(df['Fare']>=6) & (df['Embarked']=='Q'),True,colorss='r',framess=True,labelss='Queenstown')

plt.ylabel('Cumulation')

plt.legend(loc='best',framealpha=0.5)

plt.title('Fig 1.11 Log Fare Cumulation Embark Comparison')

plt.show()



lownum(df,'Pclass',(df['Embarked']=='C'),(df['Embarked']=='Q'),'Cherbourg','Queenstown')
plt.figure(figsize=(12,6))

CDG(df,'Age',(df['Age'].notnull()) & (df['Embarked']=='C'),colorss='b',labelss='Cherborg')

CDG(df,'Age',(df['Age'].notnull()) & (df['Embarked']=='S'),colorss='g',labelss='Southampton')

CDG(df,'Age',(df['Age'].notnull()) & (df['Embarked']=='Q'),colorss='r',framess=True,labelss='Queenstown')

plt.ylabel('Cumulation')

plt.legend(loc='best',framealpha=0.5)

plt.title('Fig 1.12 Age Cumulation Embark Comparison')

plt.show()
plt.figure(figsize=(12,6))

for SibNum in range(6):

    plt.subplot(2,3,(SibNum+1))

    CDG(df,'Age',(df['Age'].notnull()) & (df['SibSp']==SibNum),framess=True)

    plt.ylabel('Cumulation')

    plt.title(str(SibNum)+' SibSp') 

    plt.xticks([25,50,75])

plt.tight_layout()

plt.subplot(2,3,1)

plt.annotate('Curve is\nmore\nGradual', xy=(17, 0.1), xytext=(40,0.2),arrowprops=dict(facecolor='black', shrink=0.05)),

plt.suptitle('Fig 1.13 Age \nParch comparison')

plt.subplots_adjust(top=0.83)

plt.show()
plt.figure(figsize=(12,6))

for SibNum in range(5):

    plt.subplot(2,3,(SibNum+1))

    CDG(df,'Fare',(df['Fare']>=6) & (df['SibSp']==SibNum),True,framess=True)

    plt.ylabel('Cumulation')

    plt.title(str(SibNum)+' SibSp')

plt.tight_layout()

plt.subplot(2,3,1)

plt.annotate('Density \nSpike', xy=(8, 0.2), xytext=(20,0.2),arrowprops=dict(facecolor='black', shrink=0.05)),

plt.suptitle('Fig 1.14 Log Fare \nParch comparison')

plt.subplots_adjust(top=0.83)

plt.show()
plt.figure(figsize=(12,6))

for ParchNum in range(6):

    plt.subplot(2,3,(ParchNum+1))

    CDG(df,'Age',(df['Age'].notnull()) & (df['Parch']==ParchNum),framess=True)

    plt.ylabel('Cumulation')

    plt.title(str(ParchNum)+' Parch')

    plt.xticks([25,50,75])

plt.tight_layout()

plt.subplot(2,3,1)

plt.annotate('Curve is\nmore\nGradual', xy=(17, 0.1), xytext=(40,0.2),arrowprops=dict(facecolor='black', shrink=0.05)),

plt.suptitle('Fig 1.15 Age \nParch comparison')

plt.subplots_adjust(top=0.83)

plt.show()
plt.figure(figsize=(12,6))

for ParchNum in range(6):

    plt.subplot(2,3,(ParchNum+1))

    CDG(df,'Fare',(df['Fare']>=6) & (df['Parch']==ParchNum),True,framess=True)

    plt.ylabel('Cumulation')

    plt.title(str(ParchNum)+' Parch')

plt.tight_layout()

plt.subplot(2,3,1)

plt.annotate('Density \nSpike', xy=(8, 0.2), xytext=(20,0.2),arrowprops=dict(facecolor='black', shrink=0.05)),

plt.suptitle('Fig 1.16 Log Fare \nParch comparison')

plt.subplots_adjust(top=0.83)

plt.show()
def categoriser(row):

    global cat

    if (row['SibSp'] == 0) & (row['Parch'] == 0):

        cat = 'Alone'

    elif (row['SibSp'] > 0) & (row['Parch'] == 0):

        cat = 'SibSp, No Parch'

    elif (row['SibSp'] == 0) & (row['Parch'] > 0):

        cat = 'Parch, No SibSp'  

    elif (row['SibSp'] > 0) & (row['Parch'] > 0):

        cat = 'Parch and SibSp'    

    return cat





df['Bin_SibSp'] = np.where(df['SibSp']==0,0,1)

df['Bin_Parch'] = np.where(df['Parch']==0,0,1)

df['FamCat']=df.apply(categoriser,axis=1)

plt.figure(figsize=(12,6))

ax1 = plt.subplot2grid((2, 8), (0, 0),colspan=4,rowspan=2)

CDG(df,'Age',((df['Age'].notnull()) & (df['Bin_SibSp']==0)),colorss='b',labelss='No SibSp')

CDG(df,'Age',((df['Age'].notnull()) & (df['Bin_SibSp']==1)),colorss='g',labelss='Has SibSp')

CDG(df,'Age',(df['Age'].notnull()),colorss='k',framess=True,labelss='Population Baseline')

#ax1.ylabel('Cumulative Percentage (as Decimal)')

ax1.set_title('Fig 1.17 R+E Age Cumulataion \nBinary SibSp Comparison')

ax1.plot([32,32],[0,1],linestyle='dashdot',linewidth=3,color='k',alpha=0.5,label='Recombination')

ax1.annotate('Approximate \nRecombination', xy=(32, 0.64), xytext=(50, 0.6),

            arrowprops=dict(facecolor='black', shrink=0.05)),

ax1.legend(loc='best',framealpha=0.5)



xs = np.linspace(0,80,1000)

densitybelowwith = gaussian_kde(df[(df['Bin_SibSp']==1) & (df['Age']<=32)]['Age'].values) 

densitybelowwithout = gaussian_kde(df[(df['Bin_SibSp']==0) & (df['Age']<=32)]['Age'].values) 

densityabovewith = gaussian_kde(df[(df['Bin_SibSp']==1) & (df['Age']>32)]['Age'].values) 

densityabovewithout = gaussian_kde(df[(df['Bin_SibSp']==0) & (df['Age']>32)]['Age'].values) 

densitywith = gaussian_kde(df[(df['Bin_SibSp']==1) & (df['Age'].notnull())]['Age'].values)

densitywithout = gaussian_kde(df[(df['Bin_SibSp']==0) & (df['Age'].notnull())]['Age'].values) 

densitybelowwith._compute_covariance()

densitybelowwithout._compute_covariance()

densityabovewith._compute_covariance()

densityabovewithout._compute_covariance()

densitywith._compute_covariance()

densitywithout._compute_covariance()

ax2 = plt.subplot2grid((2, 4), (0,2),colspan=1,rowspan=1)

ax2.plot(xs,densitybelowwith(xs),color='g')

ax2.plot(xs,densitybelowwithout(xs),color='b')

ax2.set_title('Below R+E 32 \n Partition')

ax2.set_xticks([25,50,75])

ax3 = plt.subplot2grid((2, 4), (0,3),colspan=1,rowspan=1)

ax3.plot(xs,densityabovewith(xs),color='g')

ax3.plot(xs,densityabovewithout(xs),color='b')

ax3.set_title('Above R+E 32 \n Partition')

ax3.set_xticks([25,50,75])

ax4 = plt.subplot2grid((2, 4), (1,2),colspan=2,rowspan=1)

ax4.plot(xs,densitywith(xs),color='g')

ax4.plot(xs,densitywithout(xs),color='b')

ax4.set_title('Binary SibSp \n No Partition')

plt.tight_layout()

plt.show()
plt.figure(figsize=(12,6))

CDG(df,'Fare',((df['Fare']>=6) & (df['Bin_SibSp']==0)),True,colorss='b',labelss='No SibSp')

CDG(df,'Fare',((df['Fare']>=6) & (df['Bin_SibSp']==1)),True,colorss='g', framess=True,labelss='Has SibSp')

plt.ylabel('Cumulation')

plt.annotate('Density \nSpike', xy=(8, 0.3), xytext=(15, 0.5),

            arrowprops=dict(facecolor='black', shrink=0.05)),

plt.legend(loc='best',framealpha=0.5)

plt.title('Fig 1.18 Log Fare Cumulation Binary SibSp comparison')

plt.show()
plt.figure(figsize=(12,6))

CDG(df,'Age',(df['Age'].notnull()) & (df['Bin_Parch']==0),colorss='b',labelss='No Parch')

CDG(df,'Age',(df['Age'].notnull()) & (df['Bin_Parch']==1),colorss='g',framess=True,labelss='Has Parch')

CDG(df,'Age',(df['Age'].notnull()),colorss='k',framess=True,labelss='Population Baseline')

plt.plot([35,35],[0,1],linestyle='dashdot',linewidth=3,color='k',alpha=0.5,label='Recombination')

plt.ylabel('Cumulation')

plt.legend(loc='best',framealpha=0.5)

plt.annotate('Approximate \nRecombination', xy=(35, 0.7), xytext=(50, 0.6),

            arrowprops=dict(facecolor='black', shrink=0.05)),

plt.title('Fig 1.19 R+E Age Cumulation Binary Parch Comparison')

plt.show()
plt.figure(figsize=(12,6))

CDG(df,'Fare',(df['Fare']>=6)& (df['Bin_Parch']==0),True,colorss='b',labelss='No Parch')

CDG(df,'Fare',(df['Fare']>=6)& (df['Bin_Parch']==1),True,colorss='g',framess=True, labelss='Has Parch')

plt.ylabel('Cumulation')

plt.annotate('Density \nSpike', xy=(8, 0.3), xytext=(15, 0.5),

            arrowprops=dict(facecolor='black', shrink=0.05)),

plt.title('Fig 1.20 Log Fare Cumulation Binary Parch Comparison')

plt.legend(loc='best',framealpha=0.5)

plt.show()
plt.figure(figsize=(12,6))

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='Alone'),colorss='k',labelss='Alone')

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='Parch, No SibSp'),colorss='b',labelss='Parch, No SibSp')

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='SibSp, No Parch'),colorss='g',labelss='SibSp, No Parch')

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='Parch and SibSp'),colorss='r',framess=True,labelss='SibSp and Parch')

plt.title('Fig 1.21 R+E Age Cumulation Family Comparison')

plt.ylabel('Cumulation')

plt.annotate('Slow \nRecombination', xy=(50, 0.85), xytext=(40, 0.6),

            arrowprops=dict(facecolor='black', shrink=0.05)),

plt.annotate('Monotonic \nDecrease in steepness\n=Monotonic Decrease \nin density ', xy=(20, 0.55), xytext=(13, 0.75),

            arrowprops=dict(facecolor='black', shrink=0.05)),

plt.legend(loc='best',framealpha=0.5)

plt.show()
plt.figure(figsize=(12,6))

CDG(df,'Fare',(df['Fare']>=6) & (df['FamCat']=='Alone'),True,colorss='k',labelss='Alone')

CDG(df,'Fare',(df['Fare']>=6) & (df['FamCat']=='Parch, No SibSp'),True,colorss='b',labelss='Parents, No Siblings')

CDG(df,'Fare',(df['Fare']>=6) & (df['FamCat']=='SibSp, No Parch'),True,colorss='g',labelss='Siblings, No Parents')

CDG(df,'Fare',(df['Fare']>=6) & (df['FamCat']=='Parch and SibSp'),True,colorss='r',framess=True, labelss= 'Parents and Siblings')

plt.ylabel('Cumulation')

plt.title('Fig 1.22 Log Fare Cumulation Family Comparison')

plt.annotate('Density \nSpike', xy=(8, 0.3), xytext=(15, 0.5),

            arrowprops=dict(facecolor='black', shrink=0.05)),

plt.legend(loc='best',framealpha=0.5)

plt.show()
plt.figure(figsize=(12,6))

plt.subplot(2,2,1)

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='Alone') & (df['Sex']=='male'), \

colorss='b',labelss='Male')

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='Alone') & (df['Sex']=='female'), \

colorss='g',framess=True)

plt.title('Alone')



plt.subplot(2,2,2)    

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='Parch, No SibSp') & (df['Sex']=='male'), \

colorss='b',labelss='Male')

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='Parch, No SibSp') & (df['Sex']=='female'), \

colorss='g',framess=True)

plt.title('Parents, No Siblings')



plt.subplot(2,2,3)

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='SibSp, No Parch') & (df['Sex']=='male'), \

colorss='b',labelss='Male')

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='SibSp, No Parch') & (df['Sex']=='female'), \

colorss='g',framess=True)

plt.title('Siblings, No Parents')



plt.subplot(2,2,4)

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='Parch and SibSp') & (df['Sex']=='male'), \

colorss='b',labelss='Male')

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='Parch and SibSp') & (df['Sex']=='female'), \

colorss='g',framess=True,labelss='Female')

plt.title('Parents and Siblings')

plt.legend(loc='best',framealpha=0.5)

plt.tight_layout()

plt.show()
plt.figure(figsize=(12,6))

plt.subplot(2,2,1)

CDG(df,'Fare',\

    (df['Fare']>=6) & (df['FamCat']=='Alone') & (df['Sex']=='male'),True,\

    'b')

CDG(df,'Fare',\

    (df['Fare']>=6) & (df['FamCat']=='Alone') & (df['Sex']=='female'),True, \

    'g',True)

plt.plot([100,100],[0,1],linestyle='dashed',color='k',linewidth=5,alpha=0.5)

plt.annotate('Female \nPassengers \nOnly',xy=(130,0.9),xytext=(120,0.2),

            arrowprops=dict(facecolor='black'))

plt.title('Alone')



plt.subplot(2,2,2)   

CDG(df,'Fare',\

    (df['Fare']>=6) & (df['FamCat']=='Parch, No SibSp') & (df['Sex']=='male'),True, \

    'b')

CDG(df,'Fare',\

    (df['Fare']>=6) & (df['FamCat']=='Parch, No SibSp') & (df['Sex']=='female'),True, \

    'g',True)

plt.title('Parents, No Siblings')



plt.subplot(2,2,3)

CDG(df,'Fare',\

    (df['Fare']>=6) & (df['FamCat']=='SibSp, No Parch') & (df['Sex']=='male'),True,\

    'b')

CDG(df,'Fare',\

    (df['Fare']>=6) & (df['FamCat']=='SibSp, No Parch') & (df['Sex']=='female'),True, \

    'g',True)

plt.title('Siblings, No Parents')



plt.subplot(2,2,4)

CDG(df,'Fare',\

    (df['Fare']>=6) & (df['FamCat']=='Parch and SibSp') & (df['Sex']=='male'),True, \

    'b',labelss='Male')

CDG(df,'Fare',\

    (df['Fare']>=6) & (df['FamCat']=='Parch and SibSp') & (df['Sex']=='female'),True, \

    'g',True,labelss='Female')

plt.title('Siblings and Parents')

plt.legend(loc='best',framealpha=0.5)

plt.tight_layout()

plt.show()
display(df[(df['FamCat']=='Alone') & \

           (df['Fare']>=100)][['Name','Sex','Fare']].sort_values(['Sex','Fare']))
plt.figure(figsize=(12,6))

CDG(df,'Fare',(df['Fare']>=6) & (df['Name'].str.contains('Miss')),colorss='k',labelss='Miss')

CDG(df,'Fare',(df['Fare']>=6) & (df['Name'].str.contains('Mrs')),colorss='b',labelss='Mrs')

CDG(df,'Fare',(df['Fare']>=6) & (df['Name'].str.contains('Master')),colorss='g',labelss='Master')

CDG(df,'Fare',(df['Fare']>=6) & (df['Name'].str.contains('Mr')),True,'r',True,'Mr')

plt.ylabel('Cumulation')

plt.legend(loc='best',framealpha=0.5)

plt.show()
plt.figure(figsize=(12,6))

CDG(df,'Age',(df['Age'].notnull()) & (df['Name'].str.contains('Miss')),colorss='k',labelss='Miss')

CDG(df,'Age',(df['Age'].notnull()) & (df['Name'].str.contains('Mrs')),colorss='b',labelss='Mrs')

CDG(df,'Age',(df['Age'].notnull()) & (df['Name'].str.contains('Master')),colorss='g',labelss='Master')

CDG(df,'Age',(df['Age'].notnull()) & (df['Name'].str.contains('Mr')),colorss='r',framess=True, labelss= 'Mr')

plt.ylabel('Cumulation')

plt.legend(loc='best',framealpha=0.5)

plt.show()
plt.figure(figsize=(12,6))

CDG(df,'Age',(df['Age'].notnull()) & (df['Survived']==1),colorss='g')

CDG(df,'Age',(df['Age'].notnull()) & (df['Survived']==0),colorss='r',framess=True)

plt.title('Fig 2.1 Age Cumulation Survival Comparison')

plt.annotate('R+E Age \nSurvival \nSkew',xy=(10,0.15),xytext=(10,0.4),

            arrowprops=dict(facecolor='black'))

plt.ylabel('Cumulation')

plt.show()
plt.figure(figsize=(12,6))

CDG(df,'Fare',(df['Fare']>=6) & (df['Survived']==1),True,'g')

CDG(df,'Fare',(df['Fare']>=6) & (df['Survived']==0),True,'r',framess=True)

plt.ylabel('Cumulation')

plt.show()
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)

CDG(df,'Age',(df['Age'].notnull()) & (df['Survived']==1) & (df['Sex']=='male'),colorss='g')

CDG(df,'Age',(df['Age'].notnull()) & (df['Survived']==0) & (df['Sex']=='male'),colorss='r',framess=True)

plt.title('Male')

plt.annotate('R+E Age \nSurvival \nSkew',xy=(10,0.22),xytext=(10,0.5),

            arrowprops=dict(facecolor='black'))

plt.subplot(1,2,2)

CDG(df,'Age',(df['Age'].notnull()) & (df['Survived']==1) & (df['Sex']=='female'),colorss='g',labelss='Survived')

CDG(df,'Age',(df['Age'].notnull()) & (df['Survived']==0) & (df['Sex']=='female'),colorss='r',framess=True,labelss='Perished')

plt.title('Female')

plt.legend(loc='best')

plt.tight_layout()

plt.show()
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)

CDG(df,'Fare',(df['Fare']>=6) & (df['Survived']==1) & (df['Sex']=='male'),True,'g')

CDG(df,'Fare',(df['Fare']>=6) & (df['Survived']==0) & (df['Sex']=='male'),True,'r',True)

plt.title('Male')

plt.subplot(1,2,2)

CDG(df,'Fare',(df['Fare']>=6) & (df['Survived']==1) & (df['Sex']=='female'),True,'g')

CDG(df,'Fare',(df['Fare']>=6) & (df['Survived']==0) & (df['Sex']=='female'),True,'r',True)

plt.title('Female')

plt.tight_layout()

plt.show()
plt.figure(figsize=(12,6))

plt.subplot(1,3,1)

CDG(df,'Age',((df['Age'].notnull()) & (df['Pclass']==1) & (df['Survived']==1)),colorss='g')

CDG(df,'Age',((df['Age'].notnull()) & (df['Pclass']==1) & (df['Survived']==0)),colorss='r',framess=True)

plt.title('P Class 1')



plt.subplot(1,3,2)

CDG(df,'Age',((df['Age'].notnull()) & (df['Pclass']==2) & (df['Survived']==1)),colorss='g')

CDG(df,'Age',((df['Age'].notnull()) & (df['Pclass']==2) & (df['Survived']==0)),colorss='r',framess=True)

plt.title('P Class 2')



plt.subplot(1,3,3)

CDG(df,'Age',((df['Age'].notnull()) & (df['Pclass']==3) & (df['Survived']==1)),colorss='g',labelss='Survived')

CDG(df,'Age',((df['Age'].notnull()) & (df['Pclass']==3) & (df['Survived']==0)),colorss='r',labelss='Perished',framess=True)

plt.title('P Class 3')

plt.legend(loc='best',framealpha=0.5)

plt.tight_layout()

plt.show()
plt.figure(figsize=(12,6))

plt.subplot(1,3,1)

CDG(df,'Fare',((df['Fare']>=6) & (df['Pclass']==1) & (df['Survived']==1)),True,'g')

CDG(df,'Fare',((df['Fare']>=6) & (df['Pclass']==1) & (df['Survived']==0)),True,'r',True)

plt.title('P Class 1')



plt.subplot(1,3,2)

CDG(df,'Fare',((df['Fare']>=6) & (df['Pclass']==2) & (df['Survived']==1)),True,'g')

CDG(df,'Fare',((df['Fare']>=6) & (df['Pclass']==2) & (df['Survived']==0)),True,'r',True)

plt.title('P Class 2')



plt.subplot(1,3,3)

CDG(df,'Fare',((df['Fare']>=6) & (df['Pclass']==3) & (df['Survived']==1)),True,'g',labelss='Survived')

CDG(df,'Fare',((df['Fare']>=6) & (df['Pclass']==3) & (df['Survived']==0)),True,'r',True,'Perished')

plt.title('P Class 3')

plt.legend(loc='best',framealpha=0.5)

plt.tight_layout()

plt.show()
plt.figure(figsize=(12,6))

plt.subplot(2,2,1)

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='Alone') & (df['Survived']==1), \

colorss='g')

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='Alone') & (df['Survived']==0), \

colorss='r',framess=True)

plt.title('Alone')



plt.subplot(2,2,2)    

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='Parch, No SibSp') & (df['Survived']==1), \

colorss='g')

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='Parch, No SibSp') & (df['Survived']==0), \

colorss='r',framess=True)

plt.title('Parch, No SibSp')



plt.subplot(2,2,3)

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='SibSp, No Parch') & (df['Survived']==1), \

colorss='g')

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='SibSp, No Parch') & (df['Survived']==0), \

colorss='r',framess=True)

plt.title('SibSp, No Parch')



plt.subplot(2,2,4)

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='Parch and SibSp') & (df['Survived']==1), \

colorss='g',labelss='Survived')

CDG(df,'Age',(df['Age'].notnull()) & (df['FamCat']=='Parch and SibSp') & (df['Survived']==0), \

colorss='r',framess=True,labelss='Perished')

plt.title('Parch and SibSp')

plt.legend(loc='best',framealpha=0.5)

plt.tight_layout()

plt.show()



plt.figure(figsize=(12,6))

plt.subplot(2,2,1)

CDG(df,'Fare',(df['Fare']>=6) & (df['FamCat']=='Alone') & (df['Survived']==1), \

colorss='g')

CDG(df,'Fare',(df['Fare']>=6) & (df['FamCat']=='Alone') & (df['Survived']==0),True, \

colorss='r',framess=True)

plt.title('Alone')



plt.subplot(2,2,2)    

CDG(df,'Fare',(df['Fare']>=6) & (df['FamCat']=='Parch, No SibSp') & (df['Survived']==1), \

colorss='g')

CDG(df,'Fare',(df['Fare']>=6) & (df['FamCat']=='Parch, No SibSp') & (df['Survived']==0),True, \

colorss='r',framess=True)

plt.title('Parents, No Siblings')



plt.subplot(2,2,3)

CDG(df,'Fare',(df['Fare']>=6) & (df['FamCat']=='SibSp, No Parch') & (df['Survived']==1), \

colorss='g')

CDG(df,'Fare',(df['Fare']>=6) & (df['FamCat']=='SibSp, No Parch') & (df['Survived']==0),True, \

colorss='r',framess=True)

plt.title('Siblings, No Parents')



plt.subplot(2,2,4)

CDG(df,'Fare',(df['Fare']>=6) & (df['FamCat']=='Parch and SibSp') & (df['Survived']==1), \

colorss='g',labelss='Survived')

CDG(df,'Fare',(df['Fare']>=6) & (df['FamCat']=='Parch and SibSp') & (df['Survived']==0),True, \

colorss='r',framess=True,labelss='Perished')

plt.title('Parents and Siblings')

plt.legend(loc='best',framealpha=0.5)

plt.tight_layout()

plt.show()
plt.figure(figsize=(12,6))

plt.subplot(1,3,1)

CDG(df,'Age',df['Age'].notnull() & (df['Embarked']=='C') & (df['Survived']==1),colorss='g')

CDG(df,'Age',df['Age'].notnull() & (df['Embarked']=='C') & (df['Survived']==0),colorss='r',framess=True)

plt.title('Cherbourg \nN=('+str(len(df[(df['Age'].notnull()) & (df['Embarked']=='C')].index))+')')

plt.subplot(1,3,2)

CDG(df,'Age',df['Age'].notnull() & (df['Embarked']=='S') & (df['Survived']==1),colorss='g')

CDG(df,'Age',df['Age'].notnull() & (df['Embarked']=='S') & (df['Survived']==0),colorss='r',framess=True)

plt.title('Southampton \nN=('+str(len(df[(df['Age'].notnull()) & (df['Embarked']=='S')].index))+')')

plt.subplot(1,3,3)

CDG(df,'Age',df['Age'].notnull() & (df['Embarked']=='Q') & (df['Survived']==1),colorss='g')

CDG(df,'Age',df['Age'].notnull() & (df['Embarked']=='Q') & (df['Survived']==0),colorss='r',framess=True)

plt.title('Queenstown \nN=('+str(len(df[(df['Age'].notnull()) & (df['Embarked']=='Q')].index))+')')

plt.tight_layout()

plt.show()
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)

CDG(df,'Fare',(df['Fare']>=6) & (df['Embarked']=='C') & (df['Survived']==1),colorss='g')

CDG(df,'Fare',(df['Fare']>=6) & (df['Embarked']=='C') & (df['Survived']==0),True,'r',True)

plt.title('Cherbourg N=('+str(len(df[(df['Fare']>=6) & (df['Embarked']=='C')].index))+')')

plt.subplot(1,2,2)

CDG(df,'Fare',(df['Fare']>=6) & (df['Embarked']=='S') & (df['Survived']==1),colorss='g')

CDG(df,'Fare',(df['Fare']>=6) & (df['Embarked']=='S') & (df['Survived']==0),True,'r',True)

plt.title('Southampton N=('+str(len(df[(df['Fare']>=6) & (df['Embarked']=='S')].index))+')')

plt.tight_layout()

plt.show()