'''Data Dictionary



Variable	Definition	Key

survival	Survival	0 = No, 1 = Yes

pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd

sex	Sex	

Age	Age in years	

sibsp	# of siblings / spouses aboard the Titanic	

parch	# of parents / children aboard the Titanic	

ticket	Ticket number	

fare	Passenger fare	

cabin	Cabin number	

embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

Variable Notes



pclass: A proxy for socio-economic status (SES)

1st = Upper

2nd = Middle

3rd = Lower



age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5



sibsp: The dataset defines family relations in this way...

Sibling = brother, sister, stepbrother, stepsister

Spouse = husband, wife (mistresses and fiancés were ignored)



parch: The dataset defines family relations in this way...

Parent = mother, father

Child = daughter, son, stepdaughter, stepson

Some children travelled only with a nanny, therefore parch=0 for them.

'''
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import linear_model,preprocessing,tree,model_selection
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()

test.head()

train.shape
test.shape
train.count()

# data in columns Age,Cabin and Embarked is missing as we  can see the count is less than the actual count

# now find the min age and max age

train['Age'].min(),train['Age'].max()
# min age is 0.419 mean less than a year probably for babies and max is 80 years
#find the count of the passenger survived , 0 represents non-survivor and 1 represents survivor in Survived colun

train['Survived'].value_counts()
# count the surivior percentage

train['Survived'].value_counts()/train['PassengerId'].count()*100
# as we can see only 38.38 % passengers survived and 61.62 approx. didn't
# classify the passenger based on Sex/Gender

train['Sex'].value_counts()
train['Pclass'].value_counts()
#plot a bar graph for the passenger Survived 
%matplotlib inline

alpha_color = 0.5

train['Survived'].value_counts().plot(kind = 'bar')
# this bar represents the frequency distribution of passengers who Survived('1') or not('0') 
train['Sex'].value_counts().plot(kind = 'bar',color =['b','r'],alpha = alpha_color)
# this bar represents the frequency distribution of passengers who based ob=n gender(Male or female)
train['Pclass'].value_counts().plot(kind = 'bar',color = ['r','b','g'],alpha = alpha_color)
# As we can see from the graph that maximum no. of passengers belonged to class 3(cheap fare) than class 1(expensive) 

# then class 2(mid range fare)
train.corr() #find the correlation between the columns in the dataset it comapres only numerical columns

# we can see from the above table fare has the highest +ve value against the Survived column that means more 

# the Fare higher the value in survived column i.e people who paid high fare had higher chance of survival

# but that is not all , Pclass has -ve value against the Survived coulmn which means lesser the class of travel(1-High/priveleged class ,3-low/cheap and less previleged) 

# higher the chance of survival

# same with Age, age is also inversely proportional to the Survived column(-ve value means inverse corelation)

# Sex and Embarked is the missing column which may have an impact so we should verify those columns against the 

# survived 

train
# as we can see from above tables that there is no corelation between columns -Name,Sex,Ticket,Cabin,Embarked  to other columns
train.plot(kind = 'scatter',x = 'Survived', y ='Age')
#the above plot is not very definitive since the distribution is between only 2 values
train[train['Survived']==1]['Age'].value_counts().sort_index().plot(kind = 'bar')
#graph creatd above is too fussy, its not very clear since there are too many values on x axis

# so we divide the age groups in groups of 8 age classes 0-10,11-20,21-30 and so on and use pandas cut function

# to do so

bins = [0,10,20,30,40,50,60,70,80]

train['Agebin']=pd.cut(train['Age'],bins)
train[train['Survived']==1]['Agebin'].value_counts().sort_index().plot(kind = 'bar')
train[train['Survived']==0]['Agebin'].value_counts().sort_index().plot(kind = 'bar')
#frequency distribution of Passengers survived vs Passenger Class,

train[train['Survived']==1]['Pclass'].value_counts().sort_index().plot(kind = 'bar',color = ['r','b','g'])
#frequency distribution of passsengers in class 1 against survived

train[train['Pclass']==1]['Survived'].value_counts().plot(kind = 'bar',color = ['g','r'])

#we can see from graph majority passengers in class 1 survived 
train[train['Pclass']==2]['Survived'].value_counts().plot(kind = 'bar',color = ['r','g'])

#where as in class 2 there is not much of a difference, infact survival rate is low which is 
train[train['Pclass']==3]['Survived'].value_counts().plot(kind = 'bar',color = ['r','g'])
# in class 3 only few passengers survived

#plot survived against the Sex to find is there any relation between them

train[train['Survived']==0]['Sex'].value_counts().plot(kind = 'bar',color = ['b','r'],alpha = alpha_color)
#above plot shows that more male passengers died when compared to female
train[train['Survived']==1]['Sex'].value_counts().plot(kind = 'bar',color = ['b','r'],alpha = alpha_color)
#above plot shows that more female passengers survived when compared to male

#in both above bar plots we see that the difference is quite visible its more than 50% 

#so sex is a strong factor in surviving, probably people in charge must have decided to 

#evacuate female passengers first and then men

#after seeing this genlemen behaviour we should also observe this for Age group <15
train[train['Age']<15]['Survived'].value_counts().sort_index().plot(kind = 'bar',color = ['r','g'])

train[train['Age']<15].shape
#majority of juveniles survived but the difference is not so convincing
train[train['Age']<15]['Sex'].value_counts().sort_index().plot(kind = 'bar',color = ['r','b'],alpha = alpha_color)
train[train['Age']<15]['Pclass'].value_counts().sort_index().plot(kind = 'bar',color = ['r','b','g'],alpha = alpha_color)
# so above plot shows that there were very few juveniles in class 1 the higher class and most of them were

# in class 3 that is why very few of them survived
train[train['Sex']=='female']['Pclass'].value_counts().sort_index().plot(kind = 'bar')
train[train['Survived']==1]['Embarked'].value_counts().sort_index().plot(kind = 'bar',color = ['b','r','g'],alpha = alpha_color)
# the above graph shows that the majority of survivors  happened to board the ship from Southampton port

# this could be coincidence but we should check how many of them were female and how many were in Pclass 1
train[train['Embarked']=='S']['Pclass'].value_counts().sort_index().plot(kind = 'bar',color = ['b','r','g'],alpha = alpha_color)
train[train['Embarked']=='S']['Sex'].value_counts().sort_index().plot(kind = 'bar',color = ['b','r','g'],alpha = alpha_color)
#train_temp =train[train['Embarked']=='S'][train['Survived']==1]
#train_female_embark_S_survivor=train_temp[train_temp['Sex']=='female']
#train_female_embark_S_survivor[train_female_embark_S_survivor['Pclass']==1].shape
# So from above observations 'Sex','Pclass',Embarked' are important factors in the dataset to predict the surival

# if a passenger is a female she has high chance of survival and if same female is in class 1 she has

# even higher chance of survival and if she boarded from Southmapton than even more.
train[train['Survived']==1]['SibSp'].value_counts().sort_index().plot(kind ='bar')

#Survivors with 0 siblings or spouses onboard were in majority
train[train['Survived']==1]['Parch'].value_counts().sort_index().plot(kind ='bar')
train[train['Sex']=='female']['SibSp'].value_counts().sort_index().plot(kind ='bar')
train[train['Survived']==1]['Parch'].value_counts().sort_index().plot(kind ='bar')
%matplotlib inline

train = pd.read_csv('../input/train.csv')

#test = pd.read_csv('../input/test.csv')

train = train.drop(['Ticket','Cabin'],axis=1)

train = train.dropna()

fig = plt.figure(figsize = (20,15))

alpha = alphascatterplot = 0.2

alpha_bar = 0.5



ax1 =plt.subplot2grid((3,3),(0,0))

train.Survived.value_counts().sort_index().plot(kind = "bar",alpha = alpha_bar)



ax1.set_xlim(-1,2)



plt.title("Distirbution of passsengers survived")

plt.grid(b = True , which = "major")



ax2 =plt.subplot2grid((3,3),(0,1))

plt.scatter(train.Survived, train.Age, alpha = alphascatterplot)

plt.xlabel("Survived")

plt.ylabel("Age")

plt.title("Survival by Age, Surivived = 1")

plt.grid(b = True , which = "major")



ax3 =plt.subplot2grid((3,3),(0,2))

train.Pclass.value_counts().sort_index().plot(kind = "bar",alpha = alpha_bar) 

plt.xlabel("Passenger Class")

ax3.set_xlim(-1,len(train.Pclass.value_counts()))

plt.title("Distirbution of passsengers class")

plt.grid(b = True , which = "major")



ax4 =plt.subplot2grid((3,3),(1,0))

train.Sex.value_counts().sort_index().plot(kind = "bar",alpha = alpha_bar) 

plt.ylabel("Sex")

ax4.set_xlim(-1,len(train.Sex.value_counts()))

plt.grid(b = True , which = "major")

plt.title("Sex Distribution")



ax5 =plt.subplot2grid((3,3),(1,1))

train.Embarked.value_counts().sort_index().plot(kind = "bar",alpha = alpha_bar) 

plt.ylabel("Boarding station")

ax5.set_xlim(-1,len(train.Embarked.value_counts()))

plt.grid(b = True , which = "major")

plt.title("Passengers per boarding station")











ax6 =plt.subplot2grid((3,3),(2,0), colspan = 2)

train.Age[train.Pclass == 1].plot(kind = "kde")

train.Age[train.Pclass == 2].plot(kind = "kde")

train.Age[train.Pclass == 3].plot(kind = "kde")

plt.xlabel("Age")

plt.title("Age Density in each Passenger class")

plt.legend(('1st Class','2nd Class','3rd Class'),loc = 'best')

plt.title("Age Density in each Passenger class")

plt.grid(b = True , which = "major")
# create a new column Hyp and set it as 0, this column will store the predicted values

# since one of the major factors in predicting survival is Sex == female, so we set Hyp as 1 wherever we 

# found sex = female and then compare it with survived value.

# the output shows the approx. 78% time the predictions were accurate 



train = pd.read_csv('../input/train.csv')

train['Hyp'] = 0

train.loc[train.Sex == 'female' ,'Hyp']=1



train['Result'] = 0

train.loc[train.Survived== train['Hyp'],'Result']=1



train.Result.value_counts(normalize = True)

def clean_data(data):

    data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())

    data['Age'] = data['Age'].fillna(data['Age'].dropna().median())

    

    data.loc[data['Sex'] == 'male','Sex'] = 0

    data.loc[data['Sex'] == 'female','Sex'] = 1

    data['Embarked'] = data['Embarked'].fillna('S')

    data.loc[data['Embarked'] == 'S','Embarked'] = 0

    data.loc[data['Embarked'] == 'C','Embarked'] = 1

    data.loc[data['Embarked'] == 'Q','Embarked'] = 2

    return data

    

train = pd.read_csv("../input/train.csv")

#train['Fare'] = train['Fare'].fillna(train['Fare'].dropna().median())

#train.loc[train['Sex'] == 'male','Sex'] = 0

#train.loc[train['Sex'] == 'female','Sex'] = 1

train=clean_data(train)

test = pd.read_csv("../input/test.csv")

test = clean_data(test)



feature_names = ['Pclass','Sex','Embarked','Parch','SibSp']

features = train[feature_names].values

target = train['Survived'].values

classifier = linear_model.LogisticRegression()

c = classifier.fit(features,target)

print (c.score(features,target))
x_test=test[feature_names].values

y_pred=c.predict(x_test)

y_pred.shape
submission =pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred

    })



submission.to_csv('../output/submission.csv', index=False)
poly = preprocessing.PolynomialFeatures(degree = 2)

poly_features = poly.fit_transform(features)



c2 = classifier.fit(poly_features,target)

print(c2.score(poly_features,target))
classifier_tree = tree.DecisionTreeClassifier()

c3 = classifier_tree.fit(features,target)

print (c3.score(features,target))
scores = model_selection.cross_val_score(classifier_tree,features,target,scoring='accuracy',cv=50)

print(scores)
classifier_tree = tree.DecisionTreeClassifier(

        random_state = 1,

        max_depth = 7,

        min_samples_split = 2

)

c4 = classifier_tree.fit(features,target)

print (c4.score(features,target))

scores = model_selection.cross_val_score(classifier_tree,features,target,scoring='accuracy',cv=50)

print(scores)

print(scores.mean())