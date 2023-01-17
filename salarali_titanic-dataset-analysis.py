import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

import math
titanic=pd.read_csv('../input/train.csv')

titanic.head(5)
titanic=titanic.fillna(method='ffill').fillna(method='bfill')

titanic.head(5)
titanic.info()
titanic.describe()
def split(passenger):

    age,sex=passenger

    if age <18:return 'child/teen'

    elif (age>=18 and age <50): return 'young'

    else: return 'old'

titanic['AgeGroup']=titanic[['Age','Sex']].apply(split,axis=1)



#titanic=pd.read_csv('train.csv')





children=titanic.loc[titanic['AgeGroup']=='child/teen']

young=titanic.loc[(titanic['AgeGroup']=='young')]

old=titanic.loc[titanic['AgeGroup']=='old']

titanic.head()
children.head(5)
young.head(5)
old.head(5)
male=titanic.loc[titanic['Sex']=='male']

female=titanic.loc[titanic['Sex']=='female']
male.head(5)
female.head(5)
plt.pie(titanic['Sex'].value_counts().values,labels=titanic['Sex'].value_counts().index,autopct='%1.1f%%')

plt.title('Male and Female %',color='red')
plt.pie(titanic['AgeGroup'].value_counts().values,labels=titanic['AgeGroup'].value_counts().index,autopct='%1.1f%%')

plt.title('% of different Age Groups:',color='red')
titanic['Age'].hist()
sb.catplot('Sex',kind='count',hue='AgeGroup',data=titanic)
sb.boxplot(x='Embarked',y='Age',data=titanic)
sb.swarmplot(x='Sex',y='Age',data=titanic)
plt.pie(titanic['Survived'].value_counts().values,labels=titanic['Survived'].value_counts().index,autopct='%1.1f%%')

plt.title('1:Survived, 0:Not survived :',color='red')
sb.catplot('Survived',kind='count',hue='AgeGroup',data=titanic)
titanic['AloneOrNot']=titanic['SibSp']+titanic['Parch']

titanic.loc[titanic['AloneOrNot']==0,'AloneOrNot']=0

titanic.loc[titanic['AloneOrNot']>0,'AloneOrNot']=1

titanic.head(5)
sb.swarmplot(x='AloneOrNot',y='Age',hue='Sex',data=titanic)
sb.catplot('Survived',kind='count',hue='Embarked',data=titanic)
sb.catplot('Survived',kind='count',hue='Sex',data=titanic)
sb.catplot('Survived',kind='count',hue='AloneOrNot',data=titanic)
#sb.catplot('Pclass',kind='count',data=titanic)

plt.pie(titanic['Pclass'].value_counts().values,labels=titanic['Pclass'].value_counts().index,autopct='%1.1f%%')

plt.title('Classes: 1,2 and 3:',color='red')
sb.catplot('Survived',kind='count',hue='AgeGroup',data=titanic)
sb.catplot('Survived',kind='count',hue='Pclass',data=titanic)
titanic.loc[:,'Fare'].hist()
# Elements Of Confusion Matrices:

def confusionmatrices(n,tp,tn,fp,fn):

    print('Total values of test data: ',n)

    print('True Positive',tp)

    print('False Positive',fp)

    print('False Negative',fn)

    print('True Negative',tn)

    print('So:')

    accuracy=(tp+tn)*100/(tp+tn+fp+fn)

    recall=(tp)*100/(tp+fn)

    precision=(tp)*100/(tp+fp)

    f1=(2*precision*recall)/(precision+recall)

    print('Accuracy = ',accuracy,'%')

    print('Recall = ',recall,'%')

    print('Precision = ',precision,'%')

    print('F1 Score = ',f1,'%')
#KNN Algorithm (No any biult-in modules):



#separation of training data and test data (cross validation):

trainval=titanic.loc[:830,'PassengerId':'AloneOrNot']

testval=titanic.loc[831:,'PassengerId':'AloneOrNot']



#values of k:

#k=int(math.sqrt(trainval.shape[0]))

k=7

# for confusion matrices:

tp=tn=fp=fn=0



# this processes for all values of test data:

for i in range(trainval.shape[0],trainval.shape[0]+testval.shape[0]):  

#for i in range(trainval.shape[0],trainval.shape[0]+testval.shape[0]):      

    

    # finding eucledian distance:

    euc_dist=np.sqrt(((testval.loc[i,'Fare']-trainval['Fare'])**2)+((testval.loc[i,'AloneOrNot']-trainval['AloneOrNot'])**2)+((testval.loc[i,'Age']-trainval['Age'])**2)+((testval.loc[i,'Pclass']-trainval['Pclass'])**2))

    trainval['Distance']=euc_dist

    

    # sorting the dataset according to the column of distance:

    newvals=trainval.sort_values('Distance')

    

    # actual survival status:

    print('Actual Survival status of test data (person#',i,'): ',testval.loc[i,'Survived'])

    

    # predicted survival status:

    if newvals['Survived'].head(k).value_counts().max()==newvals['Survived'].head(k).value_counts()[0]:

        predicted=newvals['Survived'].value_counts().index[0]

    elif newvals['Survived'].head(k).value_counts().max()==newvals['Survived'].head(k).value_counts()[1]:

        predicted=newvals['Survived'].value_counts().index[1]

    print('Predicted Survival status of test data (person#',i,'): ',predicted,'\n')

    

    # for the calculation of confusion matrices:

    if testval.loc[i,'Survived']==1 and predicted==1: tp+=1

    elif testval.loc[i,'Survived']==1 and predicted==0: fp+=1

    elif testval.loc[i,'Survived']==0 and predicted==1: fn+=1

    elif testval.loc[i,'Survived']==0 and predicted==0: tn+=1

#Checking KNN Accuracy:

confusionmatrices(testval.shape[0],tp,tn,fp,fn)
# Naive Bayes Approach (pclass, sex, alone or not)



titanic=titanic.loc[:,'PassengerId':'AloneOrNot']

#titanic



# Pclass:

x01=titanic.loc[(titanic['Survived']==1)]

x01=x01.loc[(x01['Pclass']==3),'Pclass']



x11=titanic.loc[(titanic['Survived']==1)]

x11=x11.loc[(x11['Pclass']==1),'Pclass']



x21=titanic.loc[(titanic['Survived']==1)]

x21=x21.loc[(x21['Pclass']==2),'Pclass']



x02=titanic.loc[(titanic['Survived']==0)]

x02=x02.loc[(x02['Pclass']==3),'Pclass']



x12=titanic.loc[(titanic['Survived']==0)]

x12=x12.loc[(x12['Pclass']==1),'Pclass']



x22=titanic.loc[(titanic['Survived']==0)]

x22=x22.loc[(x22['Pclass']==2),'Pclass']



#Sex

y01=titanic.loc[(titanic['Survived']==1)]

y01=y01.loc[(y01['Sex']=='male'),'Sex']



y11=titanic.loc[(titanic['Survived']==1)]

y11=y11.loc[(y11['Sex']=='female'),'Sex']



y02=titanic.loc[(titanic['Survived']==0)]

y02=y02.loc[(y02['Sex']=='male'),'Sex']



y12=titanic.loc[(titanic['Survived']==0)]

y12=y12.loc[(y12['Sex']=='female'),'Sex']



# Alone Or Not:

z01=titanic.loc[(titanic['Survived']==1)]

z01=z01.loc[(z01['AloneOrNot']==0),'AloneOrNot']



z11=titanic.loc[(titanic['Survived']==1)]

z11=z11.loc[(z11['AloneOrNot']==1),'AloneOrNot']



z02=titanic.loc[(titanic['Survived']==0)]

z02=z02.loc[(z02['AloneOrNot']==0),'AloneOrNot']



z12=titanic.loc[(titanic['Survived']==0)]

z12=z12.loc[(z12['AloneOrNot']==1),'AloneOrNot']



# Frequency Tables:

print('Frequency Tables:')

pclass_freq=pd.DataFrame(np.array([[titanic['Pclass'].value_counts().index[0],x01.value_counts(),x02.value_counts()],[titanic['Pclass'].value_counts().index[1],x11.value_counts(),x12.value_counts()],[titanic['Pclass'].value_counts().index[2],x21.value_counts(),x22.value_counts()]]),columns=['Pclass','Yes', 'No'])

print(pclass_freq)

print()



sex_freq=pd.DataFrame(np.array([[1,y01.value_counts(),y02.value_counts()],[0,y11.value_counts(),y12.value_counts()]]),columns=['Sex','Yes','No'])

print(sex_freq)

print()



alone_freq=pd.DataFrame(np.array([[titanic['AloneOrNot'].value_counts().index[0],z01.value_counts(),z02.value_counts()],[titanic['AloneOrNot'].value_counts().index[1],z11.value_counts(),z12.value_counts()]]),columns=['AloneOrNot','Yes', 'No'])

print(alone_freq)

print()



# Likelihood Tables:

print('Likelihood Tables:\n')



pclass_likelihood=pd.DataFrame(np.array([[titanic['Pclass'].value_counts().index[0],x01.value_counts()/titanic['Survived'].value_counts().values[1],x02.value_counts()/titanic['Survived'].value_counts().values[0]],[titanic['Pclass'].value_counts().index[1],x11.value_counts()/titanic['Survived'].value_counts().values[1],x12.value_counts()/titanic['Survived'].value_counts().values[0]],[titanic['Pclass'].value_counts().index[2],x21.value_counts()/titanic['Survived'].value_counts().values[1],x22.value_counts()/titanic['Survived'].value_counts().values[0]]]),columns=['Pclass','Yes', 'No'])

print(pclass_likelihood)

print()



sex_likelihood=pd.DataFrame(np.array([[1,y01.value_counts()/titanic['Survived'].value_counts().values[1],y02.value_counts()/titanic['Survived'].value_counts().values[0]],[0,y11.value_counts()/titanic['Survived'].value_counts().values[1],y12.value_counts()/titanic['Survived'].value_counts().values[0]]]),columns=['Sex','Yes','No'])

print(sex_likelihood)

print()



alone_likelihood=pd.DataFrame(np.array([[titanic['AloneOrNot'].value_counts().index[0],z01.value_counts()/titanic['Survived'].value_counts().values[1],z02.value_counts()/titanic['Survived'].value_counts().values[0]],[titanic['AloneOrNot'].value_counts().index[1],z11.value_counts()/titanic['Survived'].value_counts().values[1],z12.value_counts()/titanic['Survived'].value_counts().values[0]]]),columns=['AloneOrNot','Yes', 'No'])

print(alone_likelihood)

print()

# Naive Bayes Test on Test-Data:



test=testval



prob_no=titanic['Survived'].value_counts().values[0]/titanic['Survived'].value_counts().sum()

prob_yes=titanic['Survived'].value_counts().values[1]/titanic['Survived'].value_counts().sum()



tp=tn=fp=fn=0



for i in range(testval.index[0],testval.index[0]+testval.shape[0]):  

    likelihood_yes=likelihood_no=1



    if testval.loc[i,'Pclass']==3:

        likelihood_yes*=pclass_likelihood.loc[0,'Yes']

        likelihood_no*=pclass_likelihood.loc[0,'No']

    elif testval.loc[i,'Pclass']==1:

        likelihood_yes*=pclass_likelihood.loc[1,'Yes']

        likelihood_no*=pclass_likelihood.loc[1,'No']

    elif testval.loc[i,'Pclass']==2:

        likelihood_yes*=pclass_likelihood.loc[2,'Yes']

        likelihood_no*=pclass_likelihood.loc[2,'No']



    if testval.loc[i,'Sex']==1:

        likelihood_yes*=sex_likelihood.loc[0,'Yes']

        likelihood_no*=sex_likelihood.loc[0,'No']

    elif testval.loc[i,'Sex']==0:

        likelihood_yes*=sex_likelihood.loc[1,'Yes']

        likelihood_no*=sex_likelihood.loc[1,'No']



    if testval.loc[i,'AloneOrNot']==0:

        likelihood_yes*=alone_likelihood.loc[0,'Yes']

        likelihood_no*=alone_likelihood.loc[0,'No']

    elif testval.loc[i,'AloneOrNot']==1:

        likelihood_yes*=alone_likelihood.loc[1,'Yes']

        likelihood_no*=alone_likelihood.loc[1,'No']



    likelihood_yes*=prob_yes

    likelihood_no*=prob_no

    

    #print('Likelihood-YES: ',likelihood_yes)

    #print('Likelihood-NO: ',likelihood_no)

    actual=test.loc[i,'Survived']

    prob=likelihood_yes/(likelihood_yes+likelihood_no)

    predicted=(prob>0.5)

    #print('Probability of surival = ',prob)

    print('Actual Survival status = ',actual)

    print('Predicted Survival status = ',predicted)

    print()

    

    if (test.loc[i,'Survived']==1 and predicted==1): tp+=1

    elif actual==1 and predicted==0: fp+=1

    elif actual==0 and predicted==1: fn+=1

    elif actual==0 and predicted==0: tn+=1
# Naive-Bayes Accuracy:

confusionmatrices(testval.shape[0],tp,tn,fp,fn)